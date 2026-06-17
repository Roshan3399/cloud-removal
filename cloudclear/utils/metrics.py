import numpy as np
from skimage.metrics import structural_similarity as ssim_ski
from skimage.metrics import peak_signal_noise_ratio as psnr_ski


class MetricsCalculator:
    @staticmethod
    def psnr(pred, target, data_range=1.0):
        return psnr_ski(target, pred, data_range=data_range)

    @staticmethod
    def ssim(pred, target, data_range=1.0, multichannel=True):
        return ssim_ski(target, pred, data_range=data_range, channel_axis=-1 if multichannel else None)

    @staticmethod
    def sam(pred, target):
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        dot = np.sum(pred_flat * target_flat, axis=1)
        norm = np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(target_flat, axis=1) + 1e-10
        angle = np.arccos(np.clip(dot / norm, -1, 1))
        return np.mean(angle) * (180 / np.pi)

    def __call__(self, pred, target, cloud_mask=None):
        result = {
            'psnr': self.psnr(pred, target),
            'ssim': self.ssim(pred, target),
            'sam': self.sam(pred, target),
        }
        return result
