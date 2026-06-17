import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))


class MS_SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def _gaussian_window(self, channels, device):
        sigma = 1.5
        gauss = torch.arange(self.window_size, dtype=torch.float32, device=device)
        gauss = torch.exp(-((gauss - self.window_size // 2) ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel = kernel.expand(channels, 1, self.window_size, self.window_size).contiguous()
        return kernel

    def _ssim_per_scale(self, pred, target, kernel, k1=0.01, k2=0.03):
        c1 = k1 ** 2
        c2 = k2 ** 2
        mu_pred = F.conv2d(pred, kernel, groups=pred.shape[1], padding=self.window_size // 2)
        mu_target = F.conv2d(target, kernel, groups=target.shape[1], padding=self.window_size // 2)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        sigma_pred = F.conv2d(pred ** 2, kernel, groups=pred.shape[1], padding=self.window_size // 2) - mu_pred_sq
        sigma_target = F.conv2d(target ** 2, kernel, groups=target.shape[1], padding=self.window_size // 2) - mu_target_sq
        sigma_joint = F.conv2d(pred * target, kernel, groups=pred.shape[1], padding=self.window_size // 2) - mu_pred_target
        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_joint + c2)) / \
                   ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2))
        return 1 - ssim_map.mean()

    def forward(self, pred, target):
        device = pred.device
        channels = pred.shape[1]
        kernel = self._gaussian_window(channels, device)
        loss = 0.0
        for scale in range(4):
            loss += self._ssim_per_scale(pred, target, kernel)
            if scale < 3:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
        return loss / 4


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        pred_grad_x = F.conv2d(pred.view(B * C, 1, H, W), self.sobel_x, padding=1).view(B, C, H, W)
        pred_grad_y = F.conv2d(pred.view(B * C, 1, H, W), self.sobel_y, padding=1).view(B, C, H, W)
        target_grad_x = F.conv2d(target.view(B * C, 1, H, W), self.sobel_x, padding=1).view(B, C, H, W)
        target_grad_y = F.conv2d(target.view(B * C, 1, H, W), self.sobel_y, padding=1).view(B, C, H, W)
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)


class SpectralAngleLoss(nn.Module):
    def forward(self, pred, target):
        pred_norm = F.normalize(pred.view(pred.shape[0], pred.shape[1], -1), dim=1)
        target_norm = F.normalize(target.view(target.shape[0], target.shape[1], -1), dim=1)
        dot = (pred_norm * target_norm).sum(dim=1).clamp(-1, 1)
        angle = torch.acos(dot)
        return angle.mean()


class CloudRemovalLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.char = CharbonnierLoss()
        self.ms_ssim = MS_SSIMLoss()
        self.grad = GradientLoss()
        self.sam = SpectralAngleLoss()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, cloudy_input=None,
                disc_outputs=None, real_feats=None, fake_feats=None,
                cloud_mask=None, edge_pred=None, edge_target=None):
        log = {}
        w = {'char': 50, 'ssim': 20, 'grad': 10, 'sam': 15,
             'adv': 1, 'clear': 100, 'edge': 5}

        loss = torch.tensor(0.0, device=pred.device)

        if w.get('char', 0):
            loss_char = self.char(pred, target)
            loss = loss + w['char'] * loss_char
            log['char'] = loss_char.item()

        if w.get('ssim', 0):
            loss_ssim = self.ms_ssim(pred, target)
            loss = loss + w['ssim'] * loss_ssim
            log['ssim'] = loss_ssim.item()

        if w.get('grad', 0):
            loss_grad = self.grad(pred, target)
            loss = loss + w['grad'] * loss_grad
            log['grad'] = loss_grad.item()

        if w.get('sam', 0):
            loss_sam = self.sam(pred, target)
            loss = loss + w['sam'] * loss_sam
            log['sam'] = loss_sam.item()

        if real_feats is not None and fake_feats is not None:
            loss_feat = 0.0
            n_feat = 0
            for real_f, fake_f in zip(real_feats, fake_feats):
                loss_feat += self.l1(real_f, fake_f)
                n_feat += 1
            if n_feat > 0:
                loss_feat = loss_feat / n_feat
                loss = loss + 5 * loss_feat
                log['feat'] = loss_feat.item() if isinstance(loss_feat, torch.Tensor) else loss_feat

        if disc_outputs is not None and w.get('adv', 0):
            loss_adv = 0.0
            n_adv = 0
            for d_out in disc_outputs:
                if isinstance(d_out, torch.Tensor):
                    loss_adv += self.mse(d_out, torch.ones_like(d_out))
                    n_adv += 1
            if n_adv > 0:
                loss_adv = loss_adv / n_adv
                loss = loss + w['adv'] * loss_adv
                log['adv'] = loss_adv.item() if isinstance(loss_adv, torch.Tensor) else loss_adv

        if cloudy_input is not None and cloud_mask is not None and w.get('clear', 0):
            clear_mask = 1 - cloud_mask
            if clear_mask.sum() > 0:
                loss_clear = self.l1(pred * clear_mask, cloudy_input * clear_mask)
                loss = loss + w['clear'] * loss_clear
                log['clear'] = loss_clear.item() if isinstance(loss_clear, torch.Tensor) else loss_clear

        if edge_pred is not None and edge_target is not None and w.get('edge', 0):
            loss_edge = self.l1(edge_pred, edge_target)
            loss = loss + w['edge'] * loss_edge
            log['edge'] = loss_edge.item() if isinstance(loss_edge, torch.Tensor) else loss_edge

        return loss, log


class DiscriminatorLoss(nn.Module):
    def __init__(self, r1_reg=10.0):
        super().__init__()
        self.r1_reg = r1_reg

    def forward(self, real_preds, fake_preds, real_input=None):
        loss = 0.0
        r1_loss = 0.0
        n = 0

        for real_pred, fake_pred in zip(real_preds, fake_preds):
            if isinstance(real_pred, torch.Tensor):
                loss += (F.relu(1 - (real_pred - fake_pred.mean())).mean() +
                         F.relu(1 + (fake_pred - real_pred.mean())).mean()) / 2
                n += 1

        if real_input is not None and self.r1_reg > 0:
            real_input.requires_grad_(True)
            try:
                grad_real = torch.autograd.grad(
                    outputs=sum(r.sum() for r in real_preds if isinstance(r, torch.Tensor)),
                    inputs=real_input,
                    create_graph=True, retain_graph=True,
                )[0]
                if grad_real is not None:
                    r1_loss = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                    r1_loss = r1_loss * (self.r1_reg / 2)
            except Exception:
                pass

        if n > 0:
            loss = loss / n

        return loss + r1_loss, {'d_hinge': loss.item() if isinstance(loss, torch.Tensor) else 0.0,
                                'r1': r1_loss.item() if isinstance(r1_loss, torch.Tensor) else r1_loss}
