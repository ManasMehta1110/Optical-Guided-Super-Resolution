"""
InfraNova — Optical-Guided Thermal Super-Resolution
Dual-Stream EDSR with Channel + Spatial Attention
"""

import os
import logging
from io import BytesIO

import numpy as np
from rasterio.io import MemoryFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="InfraNova · Thermal SR",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — bold academic / research style
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Playfair+Display:wght@700;900&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: #0d0d0d;
    color: #e8e4dc;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1400px; }

.masthead {
    border-top: 4px solid #e8e4dc;
    border-bottom: 1px solid #333;
    padding: 2.5rem 0 2rem 0;
    margin-bottom: 2.5rem;
}
.masthead-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #777;
    margin-bottom: 0.6rem;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    line-height: 1.05;
    color: #f5f0e8;
    letter-spacing: -0.02em;
}
.masthead-title span { color: #ff6b35; }
.masthead-subtitle {
    font-family: 'Source Serif 4', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: #999;
    margin-top: 0.8rem;
    max-width: 700px;
    line-height: 1.65;
}
.masthead-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    margin-top: 1.4rem;
    letter-spacing: 0.05em;
}
.masthead-meta span { color: #ff6b35; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #ff6b35;
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f5f0e8;
    border-bottom: 1px solid #222;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

.abstract-card {
    background: #111;
    border-left: 3px solid #ff6b35;
    padding: 1.4rem 1.8rem;
    margin-bottom: 2rem;
    border-radius: 0 4px 4px 0;
}
.abstract-card p {
    font-size: 0.93rem;
    line-height: 1.78;
    color: #bbb;
    margin: 0;
}

.metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #111;
    border: 1px solid #222;
    padding: 1.2rem 1.4rem;
    border-radius: 4px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: #ff6b35;
}
.metric-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #f5f0e8;
    line-height: 1;
}
.metric-card .lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    margin-top: 0.4rem;
}
.metric-card .desc { font-size: 0.76rem; color: #666; margin-top: 0.3rem; font-style: italic; }

.upload-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #777;
    margin-bottom: 0.5rem;
}

.stButton > button {
    background: #ff6b35 !important;
    color: #0d0d0d !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: #ff8c5a !important; }
.stButton > button:disabled { background: #2a2a2a !important; color: #555 !important; }

.arch-box {
    background: #0a0a0a;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 1.4rem 1.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: #888;
    line-height: 1.95;
    white-space: pre;
    overflow-x: auto;
}
.arch-box .hl { color: #ff6b35; font-weight: 600; }

section[data-testid="stSidebar"] { background: #080808 !important; border-right: 1px solid #1a1a1a; }
[data-testid="stFileUploader"] { background: #0a0a0a !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
UPSCALE    = 2
DATA_PROCESSED = "data_processed"
MODELS_DIR = "models"
BEST_PATH  = os.path.join(DATA_PROCESSED, "best_model.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infranova")

# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────
def norm_np(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.array(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
    return np.zeros_like(a) if mx - mn < 1e-6 else ((a - mn) / (mx - mn)).astype(np.float32)

def to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a, nan=0.0) - np.nan_to_num(a, nan=0.0).min()
    if a.max() > 0:
        a /= a.max()
    return (a * 255.0).clip(0, 255).astype(np.uint8)

def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    g = gt.astype(np.float32) / 255.0
    p = pred.astype(np.float32) / 255.0
    return (
        compare_psnr(g, p, data_range=1.0),
        compare_ssim(g, p, data_range=1.0),
        float(np.sqrt(np.mean((g - p) ** 2))),
    )

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.fc(self.avgpool(x))

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = max(8, in_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 3, padding=1), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.conv(x)

class RCAB(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16):
        super().__init__()
        p = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=p), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=p),
        )
        self.ca = ChannelAttention(channels, reduction)
        self.res_scale = 0.1
    def forward(self, x): return x + self.ca(self.body(x)) * self.res_scale

class ResidualGroup(nn.Module):
    def __init__(self, channels, n_rcab=4):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(channels) for _ in range(n_rcab)])
    def forward(self, x): return self.body(x) + x

class LearnedUpsampler(nn.Module):
    def __init__(self, in_ch, out_ch, scale=UPSCALE):
        super().__init__()
        self.scale = scale
        self.proj  = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        self.post  = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x, target_size=None):
        x = self.post(F.pixel_shuffle(self.proj(x), self.scale))
        if target_size: x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x

class DualEDSRPlus(nn.Module):
    """Dual-stream EDSR with Channel + Spatial Attention for optical-guided thermal SR."""
    def __init__(self, n_resgroups=4, n_rcab=4, n_feats=64, upscale=UPSCALE):
        super().__init__()
        self.convT_in   = nn.Conv2d(1, n_feats, 3, padding=1)
        self.convO_in   = nn.Conv2d(3, n_feats, 3, padding=1)
        self.t_groups   = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])
        self.o_groups   = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])
        self.t_upsampler = LearnedUpsampler(n_feats, n_feats, scale=upscale)
        self.convFuse   = nn.Conv2d(2 * n_feats, n_feats, 1)
        self.fuse_ca    = ChannelAttention(n_feats)
        self.fuse_sa    = SpatialAttention(n_feats)
        self.refine     = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.convOut = nn.Conv2d(n_feats, 1, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, xT, xO):
        fT = self.t_groups(F.relu(self.convT_in(xT)))
        fO = self.o_groups(F.relu(self.convO_in(xO)))
        fT_up = F.interpolate(self.t_upsampler(fT), size=fO.shape[2:], mode="bilinear", align_corners=False)
        f = self.fuse_sa(self.fuse_ca(F.relu(self.convFuse(torch.cat([fT_up, fO], dim=1)))))
        return self.convOut(self.refine(f))

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = DualEDSRPlus().to(DEVICE)
    if not os.path.exists(BEST_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {BEST_PATH}")
    ckpt  = torch.load(BEST_PATH, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

# ─────────────────────────────────────────────
#  I/O HELPERS
# ─────────────────────────────────────────────
def read_optical(f) -> np.ndarray:
    with MemoryFile(f.read()) as mem:
        with mem.open() as src:
            if src.count >= 3:
                return np.stack([src.read(i + 1).astype(np.float32) for i in range(3)])
            b = src.read(1).astype(np.float32)
            return np.stack([b, b, b])

def read_thermal(f) -> np.ndarray:
    with MemoryFile(f.read()) as mem:
        with mem.open() as src:
            return src.read(1).astype(np.float32)

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def run_inference(opt_rgb, thr_raw):
    H = min(opt_rgb.shape[1], thr_raw.shape[0]) - (min(opt_rgb.shape[1], thr_raw.shape[0]) % UPSCALE)
    W = min(opt_rgb.shape[2], thr_raw.shape[1]) - (min(opt_rgb.shape[2], thr_raw.shape[1]) % UPSCALE)
    opt_rgb, thr_raw = opt_rgb[:, :H, :W], thr_raw[:H, :W]

    rgb_n  = np.stack([norm_np(opt_rgb[c]) for c in range(3)])
    thr_hr = norm_np(thr_raw)

    lr = F.interpolate(
        torch.from_numpy(thr_hr).unsqueeze(0).unsqueeze(0).float(),
        size=(H // UPSCALE, W // UPSCALE), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    xO = torch.from_numpy(rgb_n).unsqueeze(0).to(DEVICE)
    xT = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sr = model(xT, xO)
    if sr.shape[2:] != (H, W):
        sr = F.interpolate(sr, size=(H, W), mode="bilinear", align_corners=False)

    bl = F.interpolate(
        torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    return (
        np.transpose(to_uint8(rgb_n), (1, 2, 0)),
        to_uint8(thr_hr),
        to_uint8(sr.squeeze().cpu().numpy()),
        to_uint8(bl),
    )

# ─────────────────────────────────────────────
#  COLORMAP
# ─────────────────────────────────────────────
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    "infranova",
    ["#0d0205", "#3d0010", "#8b1a1a", "#d44000", "#ff8c00", "#ffd060", "#ffffff"],
)

# ─────────────────────────────────────────────
#  RESULT FIGURE
# ─────────────────────────────────────────────
def render_figure(rgb_vis, thr_vis, bl_vis, sr_vis):
    fig = plt.figure(figsize=(18, 5.5), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)
    panels = [
        (rgb_vis, "Optical Guide",          None,         False),
        (thr_vis, "Thermal Input (30 m)",   THERMAL_CMAP, True),
        (bl_vis,  "Bilinear Baseline",      THERMAL_CMAP, True),
        (sr_vis,  "InfraNova SR (10 m)",    THERMAL_CMAP, True),
    ]
    for i, (img, title, cmap, use_cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#0d0d0d")
        ax.imshow(img, cmap=cmap if use_cmap else None, vmin=0 if use_cmap else None, vmax=255 if use_cmap else None)
        ax.set_title(title, color="#ff6b35" if i == 3 else "#c0bbb2",
                     fontsize=8.5, fontfamily="monospace",
                     fontweight="bold" if i == 3 else "normal", pad=8)
        ax.axis("off")
        if i == 3:
            for sp in ax.spines.values():
                sp.set_edgecolor("#ff6b35"); sp.set_linewidth(1.5); sp.set_visible(True)
    fig.text(0.5, 0.01,
             "Fig. 1  —  Qualitative comparison: optical guide · thermal input · bilinear baseline · InfraNova SR output.",
             ha="center", color="#444", fontsize=7.5, fontstyle="italic", fontfamily="monospace")
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;letter-spacing:0.3em;
                text-transform:uppercase;color:#ff6b35;margin-bottom:0.3rem;">InfraNova</div>
    <div style="font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700;
                color:#f5f0e8;margin-bottom:1.5rem;line-height:1.25;">Model<br>Configuration</div>
    <hr style="border-color:#1a1a1a;margin-bottom:1.2rem;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;
                color:#555;text-transform:uppercase;margin-bottom:0.8rem;">Architecture</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Scale", "×2"); c1.metric("Res. Groups", "4")
    c2.metric("Features", "64"); c2.metric("RCAB / Grp", "4")

    st.markdown("""
    <hr style="border-color:#1a1a1a;margin:1.2rem 0;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;
                color:#555;text-transform:uppercase;margin-bottom:0.8rem;">Published Benchmark</div>
    <div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:4px;padding:1rem;">
    <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;">
    <tr style="border-bottom:1px solid #1e1e1e;">
      <td style="padding:0.4rem 0;color:#555;">PSNR</td>
      <td style="padding:0.4rem 0;color:#ff6b35;text-align:right;font-weight:600;">42.4 dB</td>
    </tr>
    <tr style="border-bottom:1px solid #1e1e1e;">
      <td style="padding:0.4rem 0;color:#555;">SSIM</td>
      <td style="padding:0.4rem 0;color:#ff6b35;text-align:right;font-weight:600;">0.9269</td>
    </tr>
    <tr>
      <td style="padding:0.4rem 0;color:#555;">RMSE</td>
      <td style="padding:0.4rem 0;color:#ff6b35;text-align:right;font-weight:600;">0.0159</td>
    </tr>
    </table></div>
    <hr style="border-color:#1a1a1a;margin:1.2rem 0;">
    """, unsafe_allow_html=True)

    dev_col = "#4caf50" if torch.cuda.is_available() else "#666"
    dev_lbl = "CUDA" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#444;margin-bottom:2rem;">
        DEVICE &nbsp;<span style="color:{dev_col};font-weight:600;">● {dev_lbl}</span>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#333;line-height:1.9;">
        Landsat-8 TIRS / OLI &nbsp;·&nbsp; PyTorch<br>
        Streamlit &nbsp;·&nbsp; MIT License<br>
        github.com/ManasMehta1110
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MASTHEAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div class="masthead-label">Research Demo &nbsp;·&nbsp; Deep Learning &nbsp;·&nbsp; Remote Sensing</div>
  <div class="masthead-title">Infra<span>Nova</span></div>
  <div class="masthead-subtitle">
    Dual-Stream EDSR for Optical-Guided Thermal Super-Resolution —
    reconstructing 10 m thermal imagery from 30 m Landsat-8 TIRS inputs
    through optical–thermal feature fusion with channel and spatial attention.
  </div>
  <div class="masthead-meta">
    Model: DualEDSRPlus &nbsp;·&nbsp; Scale: ×2 &nbsp;·&nbsp;
    Dataset: Landsat-8 OLI / TIRS &nbsp;·&nbsp; Loss: L1 + SSIM &nbsp;·&nbsp;
    <span>PSNR 42.4 dB</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ABSTRACT
# ─────────────────────────────────────────────
st.markdown("""
<div class="abstract-card">
  <p><strong style="color:#e8e4dc;">Abstract.</strong>
  InfraNova addresses the spatial resolution gap inherent in spaceborne thermal sensors by
  leveraging the rich structural information present in co-registered optical imagery.
  A dual-stream convolutional architecture independently encodes thermal radiometry and
  optical spatial features, fusing them through a ConvFuse module augmented with Channel
  and Spatial Attention. The fused representation is refined by a deep EDSR decoder to
  produce thermally consistent, spatially sharp 10 m outputs from 30 m TIRS inputs.
  The model achieves <strong style="color:#ff6b35;">42.4 dB PSNR</strong> and
  <strong style="color:#ff6b35;">0.9269 SSIM</strong> on held-out Landsat-8 tiles,
  substantially outperforming bilinear interpolation across all metrics.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ARCHITECTURE
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">§ 1</div><div class="section-title">Architecture</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="arch-box"><span class="hl">Thermal (30 m)</span>  ──►  Thermal Encoder  ──►  ResGroups ×4  ──►  PixelShuffle ↑2  ──┐
                                                                                           │
                                                                     ConvFuse + CA + SA  ──►  EDSR Refine  ──►  <span class="hl">SR Output (10 m)</span>
                                                                                           │
<span class="hl">Optical (10 m)</span>  ──►  Optical Encoder  ──►  ResGroups ×4  ─────────────────────────┘

  ResGroup  =  [ RCAB × 4 ] + residual skip
  RCAB      =  Conv → ReLU → Conv → ChannelAttention    (res_scale 0.1)
  CA        =  AdaptiveAvgPool → FC → Sigmoid
  SA        =  Conv3×3 → ReLU → Conv3×3 → Sigmoid
  Loss      =  λ·L1  +  (1−λ)·SSIM    λ = 0.84
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INFERENCE SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">§ 2</div><div class="section-title">Run Inference</div>',
            unsafe_allow_html=True)

col_o, col_t = st.columns(2, gap="large")
with col_o:
    st.markdown('<div class="upload-label">Optical GeoTIFF (RGB / 1-band)</div>', unsafe_allow_html=True)
    opt_file = st.file_uploader("Optical", type=["tif", "tiff"], label_visibility="collapsed")
    st.caption("Landsat-8 OLI B2–B4 recommended. First 3 bands used. Must be co-registered with thermal input.")
with col_t:
    st.markdown('<div class="upload-label">Thermal GeoTIFF (single-band)</div>', unsafe_allow_html=True)
    thr_file = st.file_uploader("Thermal", type=["tif", "tiff"], label_visibility="collapsed")
    st.caption("Landsat-8 TIRS Band 10 or 11. Sample files available in the repository.")

st.markdown("<br>", unsafe_allow_html=True)

# Status line
if opt_file and thr_file:
    status_html = '<span style="color:#4caf50;">● Both inputs loaded — ready.</span>'
elif opt_file or thr_file:
    missing = "thermal" if opt_file else "optical"
    status_html = f'<span style="color:#ff6b35;">◌ Waiting for {missing} image…</span>'
else:
    status_html = '<span style="color:#444;">◌ Upload both files to begin. Sample files: <code>sample_12_optical.tif</code>, <code>sample_12_thermal.tif</code></span>'

st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;margin-bottom:1rem;">{status_html}</div>',
            unsafe_allow_html=True)

btn_col, _ = st.columns([1, 2])
with btn_col:
    run_btn = st.button("▶  Run Super-Resolution", disabled=not (opt_file and thr_file))

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if run_btn and opt_file and thr_file:
    try:
        with st.spinner("Loading model weights…"):
            model = load_model()
        with st.spinner("Preprocessing inputs…"):
            opt_rgb = read_optical(opt_file)
            thr_raw = read_thermal(thr_file)
        with st.spinner("Running DualEDSRPlus forward pass…"):
            rgb_vis, thr_vis, sr_vis, bl_vis = run_inference(opt_rgb, thr_raw)

        psnr_bl, ssim_bl, rmse_bl = compute_metrics(thr_vis, bl_vis)
        psnr_sr, ssim_sr, rmse_sr = compute_metrics(thr_vis, sr_vis)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">§ 3</div><div class="section-title">Results</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="val">{psnr_sr:.1f}</div>
            <div class="lbl">PSNR (dB)</div>
            <div class="desc">bilinear: {psnr_bl:.1f} &nbsp;Δ {psnr_sr - psnr_bl:+.1f} dB</div>
          </div>
          <div class="metric-card">
            <div class="val">{ssim_sr:.4f}</div>
            <div class="lbl">SSIM</div>
            <div class="desc">bilinear: {ssim_bl:.4f} &nbsp;Δ {ssim_sr - ssim_bl:+.4f}</div>
          </div>
          <div class="metric-card">
            <div class="val">{rmse_sr:.4f}</div>
            <div class="lbl">RMSE</div>
            <div class="desc">bilinear: {rmse_bl:.4f} &nbsp;Δ {rmse_sr - rmse_bl:+.4f}</div>
          </div>
          <div class="metric-card">
            <div class="val">{sr_vis.shape[1]}×{sr_vis.shape[0]}</div>
            <div class="lbl">Output Resolution</div>
            <div class="desc">×{UPSCALE} from {sr_vis.shape[1]//UPSCALE}×{sr_vis.shape[0]//UPSCALE} LR</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        fig = render_figure(rgb_vis, thr_vis, bl_vis, sr_vis)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        buf = BytesIO()
        plt.imsave(buf, sr_vis, cmap=THERMAL_CMAP, format="png")
        st.download_button(
            "⬇  Download SR Output (PNG)",
            data=buf.getvalue(),
            file_name="infranova_sr_output.png",
            mime="image/png",
        )

    except FileNotFoundError as e:
        st.error(f"**Checkpoint not found.** {e}")
    except Exception as e:
        st.error(f"**Inference failed:** {e}")
        logger.exception("Inference error")
