import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from cloudclear.models.generator import CloudRemovalGenerator
from cloudclear.models.discriminator import MultiScaleDiscriminator
from cloudclear.models.losses import CloudRemovalLoss, DiscriminatorLoss
from cloudclear.utils.metrics import MetricsCalculator


class CloudRemovalTrainer:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.current_epoch = 0
        self.stop_training = False

        features = tuple(config.model.features)

        self.generator = CloudRemovalGenerator(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            features=features,
        ).to(device)

        self.discriminator = MultiScaleDiscriminator(
            in_channels=config.model.in_channels,
            use_sn=True,
        ).to(device)

        self.opt_G = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.training.lr,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )
        self.opt_D = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.training.lr * 2,
            betas=(0.5, 0.999),
            weight_decay=1e-4,
        )

        self.criterion_g = CloudRemovalLoss(device)
        self.criterion_d = DiscriminatorLoss(r1_reg=10.0)

        self.metrics_calc = MetricsCalculator()

        self.history = {
            'g_loss': [], 'd_loss': [], 'psnr': [], 'ssim': [], 'sam': [],
        }

        g = sum(p.numel() for p in self.generator.parameters())
        d = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator: {g:,} params | Discriminator: {d:,} params | Device: {device}")

    def train_epoch(self, dataloader):
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            cloudy = batch['cloudy'].to(self.device)
            clear = batch['clear'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Discriminator
            self.opt_D.zero_grad()

            with torch.no_grad():
                fake_clear, _, _, _ = self.generator(cloudy)

            real_preds = self.discriminator(cloudy, clear, return_features=False)
            fake_preds = self.discriminator(cloudy, fake_clear.detach(), return_features=False)

            d_loss, d_log = self.criterion_d(real_preds, fake_preds, real_input=clear)

            if torch.isfinite(d_loss):
                d_loss.backward()
                for p in self.discriminator.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.data = torch.nan_to_num(p.grad.data)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
                self.opt_D.step()

            # Generator
            self.opt_G.zero_grad()

            fake_clear, cloud_attn, cloud_density, edge_out = self.generator(cloudy)
            disc_outputs, disc_feats = self.discriminator(cloudy, fake_clear, return_features=True)
            real_preds_fm, real_feats = self.discriminator(cloudy, clear, return_features=True)

            g_loss, g_log = self.criterion_g(
                fake_clear, clear, cloudy_input=cloudy,
                disc_outputs=disc_outputs,
                real_feats=real_feats,
                fake_feats=disc_feats,
                cloud_mask=mask,
                edge_pred=edge_out, edge_target=clear,
            )

            if torch.isfinite(g_loss):
                g_loss.backward()
                for p in self.generator.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.data = torch.nan_to_num(p.grad.data)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 10.0)
                self.opt_G.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss

            pbar.set_postfix({
                'G': f"{g_loss.item():.3f}",
                'D': f"{d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss:.3f}",
                'L1': f"{g_log.get('char', 0):.3f}",
                'SAM': f"{g_log.get('sam', 0):.4f}",
            })

        return epoch_g_loss / max(1, len(dataloader)), epoch_d_loss / max(1, len(dataloader))

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.generator.eval()
        metrics = {'psnr': [], 'ssim': [], 'sam': []}

        for batch in dataloader:
            cloudy = batch['cloudy'].to(self.device)
            clear = batch['clear'].to(self.device)

            fake_clear, _, _, _ = self.generator(cloudy)

            for i in range(cloudy.shape[0]):
                pred = fake_clear[i].cpu().numpy().transpose(1, 2, 0)
                target = clear[i].cpu().numpy().transpose(1, 2, 0)
                m = self.metrics_calc(pred, target)
                metrics['psnr'].append(m['psnr'])
                metrics['ssim'].append(m['ssim'])
                metrics['sam'].append(m['sam'])

        return {k: np.mean(v) if v else 0 for k, v in metrics.items()}

    def train(self, train_loader, val_loader=None, epochs=None):
        epochs = epochs or self.config.training.epochs
        os.makedirs(self.config.paths.output_dir, exist_ok=True)
        os.makedirs(self.config.paths.checkpoint_dir, exist_ok=True)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch + 1

            avg_g, avg_d = self.train_epoch(train_loader)

            self.history['g_loss'].append(avg_g)
            self.history['d_loss'].append(avg_d)

            status = f"Epoch {self.current_epoch}: G={avg_g:.3f} D={avg_d:.3f}"
            print(status)

            if val_loader is not None and epoch % 5 == 0:
                val_metrics = self.evaluate(val_loader)
                for k, v in val_metrics.items():
                    self.history.setdefault(k, []).append(v)
                print(f"  Val: PSNR={val_metrics['psnr']:.2f} SSIM={val_metrics['ssim']:.4f} "
                      f"SAM={val_metrics['sam']:.2f}")

            if epoch % self.config.training.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.paths.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
                )
                torch.save({
                    'epoch': self.current_epoch,
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'opt_G': self.opt_G.state_dict(),
                    'opt_D': self.opt_D.state_dict(),
                    'history': self.history,
                }, checkpoint_path)

        final_path = os.path.join(self.config.paths.checkpoint_dir, "final_model.pth")
        torch.save({
            'epoch': self.current_epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'history': self.history,
        }, final_path)
        print(f"Final model saved: {final_path}")

        return self.history
