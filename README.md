# InfraNova: Dual-Stream EDSR for Optical-Guided Thermal Super-Resolution

InfraNova is a complete deep-learning pipeline for reconstructing **high-resolution thermal imagery** by leveraging **optical guidance**.  
This repository provides the implementation, training flow, inference tools, and a detailed conceptual walkthrough for understanding multi-modal thermal super-resolution using PyTorch.

---

## Contents
- [Objective](#objective)
- [Concepts](#concepts)
- [Overview](#overview)
- [Architecture](#architecture)
- [Implementation](#implementation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Datasets](#datasets)
- [Examples](#examples)
- [FAQs](#faqs)

---

## Objective
**To generate a 10m-resolution thermal image from a 30m thermal input**  
using high-resolution optical imagery as a structural guide.

Traditional thermal sensors (e.g., Landsat-8 TIRS) capture reliable radiometric information but at low spatial resolution. Optical sensors, however, provide rich spatial detail.  
InfraNova fuses these two modalities to create **thermally consistent, spatially sharp** outputs.

---

## Concepts
### Thermal Super-Resolution
Upsampling coarse thermal data to finer resolutions while maintaining radiometric accuracy.

### Optical–Thermal Fusion
Optical images contain spatial edges and textures needed to upscale thermal signals.

### Dual-Stream CNNs
Separate encoders preserve domain-specific features before fusion.

### EDSR Architecture
High-performance super-resolution model using deep residual blocks.

### Channel Attention
Learns adaptive importance weighting across channels.

### Radiometric Preservation
Ensures temperature gradients are not overwritten by visual artifacts.

---

## Overview
InfraNova workflow:
1. Extract thermal & optical features.
2. Fuse using ConvFuse + Channel Attention.
3. Reconstruct through EDSR.
4. Output 10m thermal super-resolved imagery.

---

## Architecture

```
Thermal (30m) ──► Thermal Encoder ─┐
                                   │──► ConvFuse + Channel Attention ─► EDSR Decoder ─► Thermal SR (10m)
Optical (10m) ───► Optical Encoder ┘
```

### Optical & Thermal Encoders
Learn spatial vs radiometric features independently.

### Fusion Module
ConvFuse + Channel Attention ensures balanced multi-modal integration.

### EDSR Decoder
Residual learning for high-quality reconstruction.

### Upscaling
PixelShuffle or learned transposed convolution.

---

## Implementation

```
InfraNova/
├── app.py
├── models/
│   └── dual_edsr.py
├── data_raw/
├── data_processed/
├── main2.ipynb
├── requirements.txt
└── README.md
```

### Key Modules
- `dual_edsr.py`: model architecture  
- `app.py`: Streamlit interface  
- `main2.ipynb`: training and testing  

---

## Training

Two-phase training:

### Phase 1 — Decoder-only
- Encoders frozen  
- Stabilizes reconstruction  

### Phase 2 — Full Fine-Tuning
- Small LR  
- Aligns modalities  

### Hyperparameters
| Param | Value |
|-------|--------|
| LR | 1e-4 → 1e-5 |
| Optimizer | Adam |
| Loss | L1 + SSIM (+ optional edge loss) |
| Epochs | 50–100 |
| Batch Size | 8–16 |

---

## Evaluation

| Metric | Score |
|--------|--------|
| PSNR | 42.4 dB |
| SSIM | 0.9269 |
| RMSE | 0.0159 |

---

## Inference

### Streamlit App
```
streamlit run app.py
```

### Programmatic Usage
```python
from models.dual_edsr import InfraNovaModel
model = InfraNovaModel().load_from_checkpoint("model.pth")
sr = model(thermal, optical)
```

---
## Datasets
- Landsat-8 TIRS (thermal)
- Landsat-8 OLI (optical)
- Hugging Face curated tiles
---

## Examples
| Optical | Thermal | SR Output |
|---------|----------|-----------|
| image | image | image |

---

## FAQs

### Why dual-stream?
To preserve domain-specific features before fusion.

### Does the model hallucinate?
Loss functions ensure radiometric stability.

### Can this work on Sentinel / ECOSTRESS / UAV data?
Yes, with retraining.

### Is real-time inference possible?
Yes after conversion to ONNX or TorchScript.

