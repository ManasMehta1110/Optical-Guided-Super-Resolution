import os
import math
import logging
from io import BytesIO

import numpy as np
import rasterio
from rasterio.io import MemoryFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------
# CONFIG
# ---------------------
UPSCALE = 2
MODELS_DIR = "models"
BEST_PATH = os.path.join(MODELS_DIR, "ssl4eo_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ssl4eo_app")

BAND_IDX = {
    "B2": 2,    # Blue
    "B3": 3,    # Green
    "B4": 4,    # Red
    "B10": 10,  # Thermal 1
    "B11": 11   # Thermal 2
}

# ---------------------
# UTILS
# ---------------------
def norm_np(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=np.float32)
    if np.isnan(a).any() or np.isinf(a).any():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if mx - mn < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)

def to_uint8(a: np.ndarray):
    a = np.nan_to_num(a, nan=0.0)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a

# ---------------------
# MODEL (same as training)
# ---------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avgpool(x)
        y = self.fc(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, max(8, in_channels//2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_channels//2), 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.conv(x)
        return x * att

class RCAB(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad)
        )
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.res_scale = 0.1

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res * self.res_scale

class ResidualGroup(nn.Module):
    def __init__(self, channels, n_rcab=4):
        super().__init__()
        layers = [RCAB(channels) for _ in range(n_rcab)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x) + x

class LearnedUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale=UPSCALE):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(in_channels, out_channels * (scale*scale), kernel_size=3, padding=1)
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size=None):
        x = self.proj(x)
        x = F.pixel_shuffle(x, self.scale)
        x = self.post(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class DualEDSRPlus(nn.Module):
    def __init__(self, n_resgroups=4, n_rcab=4, n_feats=64, upscale=UPSCALE):
        super().__init__()
        self.upscale = upscale
        self.n_feats = n_feats

        self.convT_in = nn.Conv2d(1, n_feats, 3, padding=1)
        self.convO_in = nn.Conv2d(3, n_feats, 3, padding=1)

        self.t_groups = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])
        self.o_groups = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])

        self.t_upsampler = LearnedUpsampler(n_feats, n_feats, scale=upscale)

        self.convFuse = nn.Conv2d(2 * n_feats, n_feats, kernel_size=1)
        self.fuse_ca  = ChannelAttention(n_feats)
        self.fuse_sa  = SpatialAttention(n_feats)

        self.refine = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.convOut = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xT, xO):
        fT = F.relu(self.convT_in(xT))
        fO = F.relu(self.convO_in(xO))

        fT = self.t_groups(fT)
        fO = self.o_groups(fO)

        fT_up_raw = self.t_upsampler(fT)
        target_hw = (fO.shape[2], fO.shape[3])
        fT_up = F.interpolate(fT_up_raw, size=target_hw, mode="bilinear", align_corners=False)

        f = torch.cat([fT_up, fO], dim=1)
        f = F.relu(self.convFuse(f))
        f = self.fuse_ca(f)
        f = self.fuse_sa(f)
        f = self.refine(f)
        out = self.convOut(f)
        return out

@st.cache_resource
def load_model():
    model = DualEDSRPlus().to(DEVICE)
    if not os.path.exists(BEST_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {BEST_PATH}")
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

model = load_model()

# ---------------------
# IO from uploaded files
# ---------------------
def read_optical_from_upload(uploaded_file):
    # Expect either 3 bands (RGB) or more; we take first 3
    data = uploaded_file.read()
    with MemoryFile(data) as mem:
        with mem.open() as src:
            if src.count >= 3:
                r = src.read(1).astype(np.float32)
                g = src.read(2).astype(np.float32)
                b = src.read(3).astype(np.float32)
                rgb = np.stack([r, g, b], axis=0)  # (3,H,W)
            elif src.count == 1:
                band = src.read(1).astype(np.float32)
                rgb = np.stack([band, band, band], axis=0)
            else:
                raise ValueError("Optical image must have at least 1 band.")
    return rgb  # (3,H,W)

def read_thermal_from_upload(uploaded_file):
    data = uploaded_file.read()
    with MemoryFile(data) as mem:
        with mem.open() as src:
            if src.count < 1:
                raise ValueError("Thermal image must have at least 1 band.")
            thr = src.read(1).astype(np.float32)  # (H,W)
    return thr

# ---------------------
# Inference
# ---------------------
def run_inference(opt_rgb_raw: np.ndarray, thr_raw: np.ndarray):
    """
    opt_rgb_raw: (3,H,W) raw optical
    thr_raw: (H,W) raw thermal (we treat as HR, then synthesize LR like training)
    """
    # align shapes
    H_o, W_o = opt_rgb_raw.shape[1:]
    H_t, W_t = thr_raw.shape
    H = min(H_o, H_t)
    W = min(W_o, W_t)
    opt_rgb_raw = opt_rgb_raw[:, :H, :W]
    thr_raw = thr_raw[:H, :W]

    # make divisible by UPSCALE
    H_hr = H - (H % UPSCALE)
    W_hr = W - (W % UPSCALE)
    opt_rgb_raw = opt_rgb_raw[:, :H_hr, :W_hr]
    thr_raw = thr_raw[:H_hr, :W_hr]

    # normalize like training
    rgb_n = np.stack([norm_np(opt_rgb_raw[c]) for c in range(3)], axis=0)  # (3, H_hr, W_hr)
    thr_hr_n = norm_np(thr_raw)                                           # (H_hr, W_hr)

    # synthesize LR from HR thermal (exact same as dataset)
    H_lr, W_lr = H_hr // UPSCALE, W_hr // UPSCALE
    lr_full = F.interpolate(
        torch.from_numpy(thr_hr_n).unsqueeze(0).unsqueeze(0).float(),
        size=(H_lr, W_lr),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()  # (H_lr, W_lr)

    # tensors for model
    xO = torch.from_numpy(rgb_n).unsqueeze(0).to(DEVICE)       # (1,3,H_hr,W_hr)
    xT = torch.from_numpy(lr_full).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H_lr,W_lr)

    with torch.no_grad():
        sr = model(xT, xO)  # (1,1,H_hr,W_hr) ideally
        if sr.shape[2:] != (H_hr, W_hr):
            sr = F.interpolate(sr, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

    sr_np = sr.squeeze().cpu().numpy()       # (H_hr,W_hr)
    rgb_vis = np.transpose(to_uint8(rgb_n), (1, 2, 0))  # (H,W,3)
    thr_vis = to_uint8(thr_hr_n)
    sr_vis = to_uint8(sr_np)

    return rgb_vis, thr_vis, sr_vis

# ---------------------
# STREAMLIT UI
# ---------------------
st.title("🌡️ SSL4EO Optical-Guided Thermal Super-Resolution")
st.write("Upload an **optical GeoTIFF** (RGB) and a **thermal GeoTIFF**. The app will use your SSL4EO DualEDSRPlus model to super-resolve the thermal image.")

opt_file = st.file_uploader("Optical GeoTIFF (RGB or 1-band)", type=["tif", "tiff"])
thr_file = st.file_uploader("Thermal GeoTIFF (1-band)", type=["tif", "tiff"])

if opt_file is not None and thr_file is not None:
    if st.button("Run Super-Resolution"):
        try:
            st.info("Reading inputs...")
            # IMPORTANT: need fresh buffers because we read twice if user re-runs
            opt_rgb_raw = read_optical_from_upload(opt_file)
            thr_raw     = read_thermal_from_upload(thr_file)

            st.info("Running model (this may take a moment)...")
            rgb_vis, thr_vis, sr_vis = run_inference(opt_rgb_raw, thr_raw)

            st.subheader("Results")

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(rgb_vis)
            axs[0].set_title("Optical")
            axs[0].axis("off")

            axs[1].imshow(thr_vis, cmap="inferno")
            axs[1].set_title("Thermal (input / HR ref)")
            axs[1].axis("off")

            axs[2].imshow(sr_vis, cmap="inferno")
            axs[2].set_title("Super-Resolved Thermal (model)")
            axs[2].axis("off")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during inference: {e}")
else:
    st.info("Upload both optical and thermal images to start.")
