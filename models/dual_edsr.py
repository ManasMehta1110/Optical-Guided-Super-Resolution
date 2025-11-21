import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * 0.1


class DualEDSR(nn.Module):
    def __init__(self, n_resblocks=8, n_feats=64, upscale=3):
        super().__init__()
        self.upscale = upscale
        self.convT = nn.Conv2d(1, n_feats, 3, padding=1)
        self.convO = nn.Conv2d(3, n_feats, 3, padding=1)
        self.resBlocksT = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.resBlocksO = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.convFuse = nn.Conv2d(2 * n_feats, n_feats, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.convOut = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, xT, xO):
        fT = F.relu(self.convT(xT))
        fO = F.relu(self.convO(xO))
        fT = self.resBlocksT(fT)
        fO = self.resBlocksO(fO)
        fT_up = F.interpolate(fT, size=(fO.shape[2], fO.shape[3]),
                              mode='bilinear', align_corners=False)
        f = torch.cat([fT_up, fO], dim=1)
        f = F.relu(self.convFuse(f))
        f = self.refine(f)
        out = self.convOut(f)
        return out


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.mlp(x)
        return x * w


class EdgeExtractor(nn.Module):
    """Fixed Sobel edge magnitude."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)
        kernel = torch.stack([sobel_x, sobel_y])  # [2, 3, 3]
        self.register_buffer("weight", kernel.unsqueeze(1))  # [2,1,3,3]

    def forward(self, x):
        # x: [B, C, H, W] optical guidance. Convert to luminance then gradient.
        if x.shape[1] > 1:
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        else:
            gray = x
        grad = F.conv2d(gray, self.weight, padding=1)
        mag = torch.sqrt((grad ** 2).sum(dim=1, keepdim=True) + 1e-6)
        return mag


class DualEDSRGated(nn.Module):
    """
    Higher-capacity dual-stream with edge guidance and gated fusion.
    Fusion is conditioned on both streams and optical edges to reduce texture leakage.
    """
    def __init__(self, n_resblocks=16, n_feats=96):
        super().__init__()
        self.edge = EdgeExtractor()
        self.convT = nn.Conv2d(1, n_feats, 3, padding=1)
        self.convO = nn.Conv2d(3, n_feats, 3, padding=1)
        self.resBlocksT = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.resBlocksO = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.gate = ChannelGate(n_feats)
        self.fuse = nn.Conv2d(2 * n_feats + 1, n_feats, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, xT, xO):
        edge = self.edge(xO)
        fT = F.relu(self.convT(xT))
        fO = F.relu(self.convO(xO))
        fT = self.resBlocksT(fT)
        fO = self.resBlocksO(fO)
        fT_up = F.interpolate(fT, size=(fO.shape[2], fO.shape[3]),
                              mode='bilinear', align_corners=False)
        fT_up = self.gate(fT_up)
        fused = torch.cat([fT_up, fO, edge], dim=1)
        fused = F.relu(self.fuse(fused))
        fused = self.refine(fused)
        out = self.out(fused)
        return out


def edge_aware_l1(pred, target, optical, alpha=1.0, eps=1e-6):
    """
    Edge-aware L1 loss that upweights errors near optical edges.
    pred, target: [B,1,H,W]; optical: [B,3,H,W] or [B,1,H,W]
    """
    if optical.shape[1] > 1:
        gray = 0.2989 * optical[:, 0:1] + 0.5870 * optical[:, 1:2] + 0.1140 * optical[:, 2:3]
    else:
        gray = optical
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)
    grad = torch.cat([
        F.conv2d(gray, sobel_x, padding=1),
        F.conv2d(gray, sobel_y, padding=1)
    ], dim=1)
    edge_mag = torch.sqrt((grad ** 2).sum(dim=1, keepdim=True) + eps)
    edge_w = (edge_mag - edge_mag.min()) / (edge_mag.max() - edge_mag.min() + eps)
    weight = 1.0 + alpha * edge_w
    return (weight * torch.abs(pred - target)).mean()
