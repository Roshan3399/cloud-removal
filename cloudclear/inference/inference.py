import os
import numpy as np
import torch
import cv2
from pathlib import Path

from cloudclear.models.generator import CloudRemovalGenerator


def normalize_input(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.max() > 1.0:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def pad_to_multiple(img, multiple=16):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0, 0, 0)
    padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    return padded, (0, 0, pad_h, pad_w)


def remove_padding(img, pad):
    pad_t, pad_b, pad_l, pad_r = pad
    h, w = img.shape[:2]
    return img[pad_t:h - pad_b, pad_l:w - pad_r]


class CloudRemovalInference:
    def __init__(self, checkpoint_path, config, device='cpu'):
        self.config = config
        self.device = device

        self.generator = CloudRemovalGenerator(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            features=tuple(config.model.features),
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get('generator', ckpt)
        if 'generator' in ckpt:
            state_dict = ckpt['generator']
        self.generator.load_state_dict(state_dict)
        print(f"[Inference] Loaded generator weights from {checkpoint_path}")

        self.generator.eval()
        self.patch_size = self.config.data.img_size
        self.overlap = 32

    @torch.no_grad()
    def process_image(self, image_path, output_path=None):
        import rasterio

        with rasterio.open(image_path) as src:
            profile = src.profile
            img = np.stack([src.read(i + 1) for i in range(min(3, src.count))], axis=-1)
            img = normalize_input(img)

        h, w = img.shape[:2]

        if h > self.patch_size or w > self.patch_size:
            result = self._tiled_inference(img)
        else:
            img_padded, pad = pad_to_multiple(img, 16)
            tensor = torch.from_numpy(img_padded.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
            output, _, _, _ = self.generator(tensor)
            output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            output = remove_padding(output, pad)
            result = output

        result = np.clip(result, 0, 1).astype(np.float32)

        if output_path:
            if output_path.endswith(('.tif', '.tiff')):
                profile.update(count=3, dtype='float32')
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for i in range(3):
                        dst.write(result[:, :, i].astype(np.float32), i + 1)
                print(f"Saved GeoTIFF: {output_path}")
            else:
                out_img = (np.clip(result, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(output_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                print(f"Saved PNG: {output_path}")

        return result

    @torch.no_grad()
    def _tiled_inference(self, img):
        h, w = img.shape[:2]
        result = np.zeros((h, w, 3), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        stride = self.patch_size - self.overlap

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1, y2 = y, min(y + self.patch_size, h)
                x1, x2 = x, min(x + self.patch_size, w)

                patch = img[y1:y2, x1:x2]
                ph, pw = patch.shape[:2]

                if ph < 16 or pw < 16:
                    continue

                patch_pad, pad = pad_to_multiple(patch, 16)
                tensor = torch.from_numpy(patch_pad.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                out, _, _, _ = self.generator(tensor)
                out = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                out = remove_padding(out, pad)
                out = out[:ph, :pw]

                wy = 0.5 * (1 - np.cos(np.pi * np.arange(ph) / max(ph - 1, 1))) if ph > 1 else np.ones(ph)
                wx = 0.5 * (1 - np.cos(np.pi * np.arange(pw) / max(pw - 1, 1))) if pw > 1 else np.ones(pw)
                w2d = np.outer(wy, wx)

                result[y1:y2, x1:x2] += out * w2d[..., None]
                weight_map[y1:y2, x1:x2] += w2d

        weight_map = np.clip(weight_map, 1e-8, None)
        return np.clip(result / weight_map[..., None], 0, 1)

    def process_directory(self, input_dir, output_dir, ext='*.tif'):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = list(input_dir.glob(ext))
        if not paths:
            paths = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))

        print(f"[Batch] Processing {len(paths)} images...")
        for p in paths:
            out_path = output_dir / f"{p.stem}_clear{p.suffix}"
            self.process_image(str(p), str(out_path))
            print(f"  {p.name} -> {out_path.name}")
