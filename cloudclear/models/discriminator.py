import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SpectralNormConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, pad=0, use_sn=True):
        super().__init__()
        conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=pad, bias=False)
        self.conv = spectral_norm(conv) if use_sn else conv

    def forward(self, x):
        return self.conv(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2, use_sn=True):
        super().__init__()
        self.conv = SpectralNormConv2d(in_c, out_c, 3, stride=stride, pad=1, use_sn=use_sn)
        self.bn = nn.BatchNorm2d(out_c) if not use_sn else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class AttentionDiscriminatorBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2, use_sn=True):
        super().__init__()
        self.conv = SpectralNormConv2d(in_c, out_c, 3, stride=stride, pad=1, use_sn=use_sn)
        self.attn = nn.Sequential(
            nn.Conv2d(out_c, out_c // 8, 1),
            nn.ReLU(True),
            nn.Conv2d(out_c // 8, 1, 1),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(out_c) if not use_sn else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        attn = self.attn(x)
        x = x * attn
        x = self.bn(x)
        return self.activation(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=3, n_layers=4, ndf=64, use_sn=True):
        super().__init__()
        layers = []
        layers.append(nn.Sequential(
            SpectralNormConv2d(in_channels * 2, ndf, 3, stride=2, pad=1, use_sn=use_sn),
            nn.LeakyReLU(0.2),
        ))
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(AttentionDiscriminatorBlock(
                ndf * nf_mult_prev, ndf * nf_mult, stride=2, use_sn=use_sn
            ))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(AttentionDiscriminatorBlock(
            ndf * nf_mult_prev, ndf * nf_mult, stride=1, use_sn=use_sn
        ))
        self.model = nn.Sequential(*layers)
        self.final = SpectralNormConv2d(ndf * nf_mult, 1, 3, pad=1, use_sn=use_sn)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, img_A, img_B, return_features=False):
        x = torch.cat([img_A, img_B], dim=1)
        feats = []
        for layer in self.model:
            x = layer(x)
            feats.append(x)
        if x.shape[2] < 3 or x.shape[3] < 3:
            x = self.adaptive_pool(x)
        out = self.final(x)
        if return_features:
            return out, feats
        return out


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3, use_sn=True):
        super().__init__()
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(in_channels, n_layers=5, ndf=64, use_sn=use_sn),
            NLayerDiscriminator(in_channels, n_layers=4, ndf=64, use_sn=use_sn),
            NLayerDiscriminator(in_channels, n_layers=3, ndf=64, use_sn=use_sn),
        ])

    def forward(self, cloudy, target, return_features=False):
        scales = [1.0, 0.5, 0.25]
        outputs = []
        all_feats = []

        for disc, scale in zip(self.discriminators, scales):
            if scale < 1.0:
                h, w = cloudy.shape[2:]
                nh, nw = int(h * scale), int(w * scale)
                cloudy_scaled = F.interpolate(cloudy, (nh, nw), mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, (nh, nw), mode='bilinear', align_corners=False)
            else:
                cloudy_scaled = cloudy
                target_scaled = target

            if return_features:
                d_out, d_feats = disc(cloudy_scaled, target_scaled, return_features=True)
                outputs.append(d_out)
                all_feats.extend(d_feats)
            else:
                outputs.append(disc(cloudy_scaled, target_scaled))

        if return_features:
            return outputs, all_feats
        return outputs

    def get_feature_maps(self, cloudy, target):
        _, feats = self.forward(cloudy, target, return_features=True)
        return feats
