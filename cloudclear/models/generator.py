import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class CloudAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cloud_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.cloud_detector(x)
        return x * attn, attn


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=g1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        if psi.shape[2:] != x.shape[2:]:
            psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x * psi


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 0 * growth, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + 1 * growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, channels, growth=32, num_dense=3):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualDenseBlock(channels, growth) for _ in range(num_dense)
        ])

    def forward(self, x):
        return self.blocks(x) * 0.2 + x


class CoordConv(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, pad=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c + 2, out_c, kernel, padding=pad)

    def forward(self, x):
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        yy = yy.float() / (H - 1) * 2 - 1
        xx = xx.float() / (W - 1) * 2 - 1
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return self.conv(torch.cat([x, coords], dim=1))


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = CoordConv(in_c, out_c, 3, 1)
        self.rrdb = RRDB(out_c, growth=min(out_c // 4, 32))
        self.attn = ChannelAttention(out_c)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.lrelu(x)
        x = self.rrdb(x)
        x = self.attn(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, skip_c, out_c):
        super().__init__()
        fusion_c = skip_c + out_c
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
            RRDB(out_c, growth=min(out_c // 4, 32)),
        )

    def forward(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([x, skip], dim=1))


class CloudDensityEstimator(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class SpectralConsistencyModule(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.transform = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, pred, input_img):
        correction = self.transform(torch.cat([pred, input_img], dim=1))
        return pred + correction


class EdgePreservationBranch(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, 3, 3, padding=1)

    def forward(self, features):
        return self.conv(features)


class CloudRemovalGenerator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=(64, 128, 256, 512),
    ):
        super().__init__()

        self.head = CoordConv(in_channels, features[0], 3, 1)

        self.encoder = nn.ModuleList()
        self.encoder_attn = nn.ModuleList()
        self.down = nn.ModuleList()
        prev_c = features[0]
        for feat in features:
            self.encoder.append(EncoderBlock(prev_c, feat))
            self.encoder_attn.append(ChannelAttention(feat))
            self.down.append(nn.Conv2d(feat, feat, 3, stride=2, padding=1))
            prev_c = feat

        bottleneck_c = features[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_c, bottleneck_c * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bottleneck_c * 2, bottleneck_c, 1),
        )

        self.bottleneck_rrdb = RRDB(bottleneck_c, growth=min(bottleneck_c // 8, 64), num_dense=3)

        self.decoder = nn.ModuleList()
        self.decoder_fusion = nn.ModuleList()
        prev_c = bottleneck_c
        for feat in reversed(features):
            self.decoder.append(nn.Sequential(
                nn.Conv2d(prev_c, feat, 3, padding=1),
                nn.BatchNorm2d(feat),
                nn.LeakyReLU(0.2),
                RRDB(feat, growth=min(feat // 4, 32)),
            ))
            self.decoder_fusion.append(nn.Sequential(
                nn.Conv2d(feat * 2, feat, 3, padding=1),
                nn.BatchNorm2d(feat),
                nn.LeakyReLU(0.2),
            ))
            prev_c = feat

        self.cloud_density = CloudDensityEstimator(features[0])
        self.cloud_attention = CloudAttention(features[0])

        self.out_conv = nn.Sequential(
            RRDB(features[0], growth=32, num_dense=2),
            nn.Conv2d(features[0], features[0] // 2, 3, padding=1),
            nn.BatchNorm2d(features[0] // 2),
            nn.ReLU(),
            nn.Conv2d(features[0] // 2, out_channels, 1),
        )

        self.post_denoiser = nn.Sequential(
            nn.Conv2d(out_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            RRDB(64, growth=32, num_dense=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.post_denoiser[-1].weight)
        nn.init.zeros_(self.post_denoiser[-1].bias)

        self.edge_branch = EdgePreservationBranch(features[0])
        self.spectral_refine = SpectralConsistencyModule(out_channels)

        self.final_refine = nn.Sequential(
            nn.Conv2d(out_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1),
        )
        nn.init.zeros_(self.final_refine[-1].weight)
        nn.init.zeros_(self.final_refine[-1].bias)

    def forward(self, x):
        input_img = x
        skips = []

        x = self.head(x)

        for enc, attn, down in zip(self.encoder, self.encoder_attn, self.down):
            x = enc(x)
            x = attn(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)
        x = self.bottleneck_rrdb(x)

        skips = skips[::-1]
        decoded_features = None
        for idx, (skip, dec, fusion) in enumerate(zip(skips, self.decoder, self.decoder_fusion)):
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = dec(x)
            x = fusion(torch.cat([x, skip], dim=1))
            decoded_features = x

        cloud_density = self.cloud_density(decoded_features)
        attended, cloud_attention = self.cloud_attention(decoded_features)

        residual = self.out_conv(attended)
        output = input_img + residual

        edge_out = self.edge_branch(attended)
        output = self.spectral_refine(output, input_img)
        output = output + self.post_denoiser(output)
        output = output + self.final_refine(output)

        output = torch.clamp(output, -1, 1)
        return output, cloud_attention, cloud_density, edge_out
