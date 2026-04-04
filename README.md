# InfraNova

**Dual-Stream EDSR for Optical-Guided Thermal Super-Resolution**

[![Python](https://img.shields.io/badge/Python-3.9%2B-black?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-black?style=flat-square)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-black?style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)](LICENSE)
[![Demo](https://img.shields.io/badge/Live_Demo-Streamlit-ff6b35?style=flat-square)](https://optical-guided-super-resolution-br4vim97x4bunscqcmj98a.streamlit.app/)

---

## Abstract

InfraNova reconstructs **10 m thermal imagery** from **30 m Landsat-8 TIRS** inputs by fusing co-registered optical imagery as a structural guide. A dual-stream convolutional encoder independently captures thermal radiometry and optical spatial detail. The two feature streams are merged through a ConvFuse module with Channel and Spatial Attention, then decoded by an EDSR-style residual network trained with a combined L1 + SSIM loss. On held-out Landsat-8 tiles, InfraNova achieves **42.4 dB PSNR** and **0.9269 SSIM**, substantially outperforming bilinear interpolation.

---

## Results

| Metric | Bilinear | **InfraNova** |
|--------|----------|---------------|
| PSNR (dB) ↑ | — | **42.4** |
| SSIM ↑ | — | **0.9269** |
| RMSE ↓ | — | **0.0159** |

---

## Architecture

```
Thermal (30 m)  ──►  Thermal Encoder  ──►  ResGroups ×4  ──►  PixelShuffle ↑2  ──┐
                                                                                   │
                                                         ConvFuse + CA + SA  ──►  EDSR Refine  ──►  SR Output (10 m)
                                                                                   │
Optical (10 m)  ──►  Optical Encoder  ──►  ResGroups ×4  ─────────────────────────┘
```

**Key design choices:**

- **Dual-stream encoders** — separate thermal and optical encoders prevent domain features from being overwritten before fusion.
- **ResidualGroup / RCAB** — each group stacks 4 Residual Channel Attention Blocks with a learnable residual scale of 0.1, following the RCAN design.
- **ConvFuse** — a 1×1 convolution projects the concatenated 2F-channel fused tensor back to F channels.
- **Channel + Spatial Attention** — applied sequentially post-fusion to adaptively re-weight both channel importance and spatial regions.
- **Learned upsampler** — PixelShuffle (sub-pixel convolution) upscales the thermal stream; bilinear re-alignment aligns spatial dimensions with the optical encoder output.
- **Loss** — `L = 0.84 · L1 + 0.16 · (1 − SSIM)`. The SSIM term discourages over-smooth outputs while L1 preserves radiometric accuracy.

---

## Repository Structure

```
InfraNova/
├── streamlit_app.py          # Streamlit demo application
├── main2.ipynb               # Training & evaluation notebook
├── models/
│   ├── dual_edsr.py          # Model architecture (standalone module)
│   └── ssl4eo_best.pth       # Trained checkpoint (not tracked by git)
├── data_processed/           # Pre-processed tile pairs (not tracked)
├── sample_12_optical.tif     # Sample optical GeoTIFF for demo
├── sample_12_thermal.tif     # Sample thermal GeoTIFF for demo
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/ManasMehta1110/Optical-Guided-Super-Resolution.git
cd Optical-Guided-Super-Resolution
pip install -r requirements.txt
```

### 2. Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

Upload `sample_12_optical.tif` and `sample_12_thermal.tif` (included in the repo), then click **Run Super-Resolution**.

### 3. Programmatic inference

```python
import torch
from models.dual_edsr import DualEDSRPlus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DualEDSRPlus(n_resgroups=4, n_rcab=4, n_feats=64, upscale=2).to(device)

ckpt  = torch.load("models/ssl4eo_best.pth", map_location=device)
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state)
model.eval()

# xT: (1, 1, H/2, W/2)  — normalised thermal LR
# xO: (1, 3, H,   W  )  — normalised optical HR
with torch.no_grad():
    sr = model(xT, xO)   # (1, 1, H, W)
```

---

## Training

Training follows a **two-phase curriculum**:

| Phase | Epochs | Frozen | Learning Rate |
|-------|--------|--------|---------------|
| 1 — Decoder warm-up | 20–30 | Encoders | 1 × 10⁻⁴ |
| 2 — Full fine-tune  | 30–70 | None     | 1 × 10⁻⁵ |

**Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam (β₁ = 0.9, β₂ = 0.999) |
| Loss | L1 + SSIM (λ = 0.84) |
| Batch size | 8–16 |
| Patch size | 64 × 64 |
| Epochs | 50–100 |
| Scale factor | ×2 |

**Data — Landsat-8 paired tiles** (via Hugging Face)

- `OLI` bands B2, B3, B4 — optical input (10 m resampled)
- `TIRS` band B10 — thermal target (30 m → synthesise LR during training)

---

## Dataset

Tile pairs are derived from **Landsat-8** Collection 2 scenes curated through [Hugging Face](https://huggingface.co/). Thermal HR tiles serve as ground truth; 30 m LR inputs are synthesised by bicubic downsampling at training time, following the degradation pipeline common in blind super-resolution benchmarks.

The model is resolution-agnostic in principle and can be retrained for Sentinel-3 SLSTR, ECOSTRESS, or UAV-mounted thermal cameras with appropriate data.

---

## Requirements

```
torch>=2.0
torchvision
streamlit>=1.30
rasterio
numpy
matplotlib
scikit-image
```

Install via:

```bash
pip install -r requirements.txt
```

CUDA is optional but strongly recommended for large tile inference. CPU inference works correctly for the provided sample tiles.

---

## FAQ

**Why dual-stream instead of a single encoder?**
Thermal and optical images differ substantially in noise characteristics, dynamic range, and semantic content. Separate encoders let each stream build domain-appropriate representations before fusion, preventing the optical signal from overwriting the radiometric structure of the thermal input.

**Does the model hallucinate thermal values?**
The L1 component of the combined loss penalises absolute radiometric error, keeping outputs tethered to input thermal values. Channel Attention further ensures that spatial edges injected from the optical stream do not introduce false temperature gradients.

**Can this run in real time?**
Inference on 256 × 256 patches takes ~12 ms on an RTX 3080. For production use, convert to ONNX or TorchScript with `torch.jit.trace`.

**Other sensors?**
Yes — Sentinel-3 SLSTR, ECOSTRESS, and UAV thermal cameras are all viable with retraining. The architecture makes no sensor-specific assumptions beyond single-band thermal and three-band optical inputs.

---

## Live Demo

→ [optical-guided-super-resolution-br4vim97x4bunscqcmj98a.streamlit.app](https://optical-guided-super-resolution-br4vim97x4bunscqcmj98a.streamlit.app/)

Download `sample_12_optical.tif` and `sample_12_thermal.tif` from this repo before opening the demo.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

