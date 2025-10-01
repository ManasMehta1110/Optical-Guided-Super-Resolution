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
