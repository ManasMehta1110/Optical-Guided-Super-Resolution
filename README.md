# InfraNova: Dual-Stream EDSR for Optical-Guided Thermal Super-Resolution

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SIH-2024%20Finalist-gold?style=for-the-badge"/>
</p>

<p align="center">
  <b>🚀 <a href="https://optical-guided-super-resolution-br4vim97x4bunscqcmj98a.streamlit.app/">Live Demo</a></b> — Upload a thermal + optical .tif pair and see super-resolution in action.
  <br/>
  <i>Download <a href="sample_12_optical.tif">sample_12_optical.tif</a> and <a href="sample_12_thermal.tif">sample_12_thermal.tif</a> before visiting the demo.</i>
</p>

---

## What is InfraNova?

InfraNova is a deep-learning pipeline that reconstructs **high-resolution (10m) thermal imagery** from coarse **30m thermal inputs**, guided by a co-registered **10m optical image**.

Traditional thermal sensors like Landsat-8 TIRS capture reliable radiometric data but at low spatial resolution (30m). Optical sensors (Landsat-8 OLI) capture rich spatial detail at 10m. InfraNova fuses both modalities through a custom Dual-Stream EDSR architecture to produce outputs that are both **thermally accurate** and **spatially sharp**.

> 🏆 Selected as a finalist at **Smart India Hackathon (SIH) 2024** in the Remote Sensing & Geospatial domain.

---

## Results

| Metric | Score |
|--------|-------|
| PSNR   | **42.4 dB** |
| SSIM   | **0.9269** |
| RMSE   | **0.0159** |

---

## Architecture

```
Thermal Input (30m) ──► Thermal Encoder ──┐
                                           ├──► ConvFuse + Channel Attention ──► EDSR Decoder ──► Thermal SR (10m)
Optical Input (10m) ──► Optical Encoder ──┘
```

### Key Components

**Dual-Stream Encoders**
Separate CNN encoders process thermal and optical inputs independently. This preserves domain-specific features — spatial edges from optical, radiometric gradients from thermal — before any fusion occurs.

**ConvFuse + Channel Attention**
A convolutional fusion module merges the two encoded feature maps. Channel Attention then learns to adaptively re-weight channels, letting the model decide how much optical spatial detail vs. thermal radiometric information to emphasise at each layer.

**EDSR Decoder**
Based on the Enhanced Deep Super-Resolution architecture (Lim et al., CVPRW 2017). Batch Normalization is removed to improve accuracy and reduce memory usage by ~40%. Residual blocks enable deep feature learning without degradation.

**PixelShuffle Upscaling**
Sub-pixel convolution upsamples the fused feature map from 30m to 10m resolution (3× scale factor).

---

## Repository Structure

```
Optical-Guided-Super-Resolution/
├── models/
│   └── dual_edsr.py        # Full model architecture (encoders, fusion, EDSR decoder)
├── data_processed/         # Preprocessed .tif patches for training
├── main2.ipynb             # Training, evaluation, and inference notebook
├── streamlit_app.py        # Streamlit web demo
├── requirements.txt        # Python dependencies
├── sample_12_optical.tif   # Sample optical image for demo
├── sample_12_thermal.tif   # Sample thermal image for demo
├── CONTRIBUTING.md
└── README.md
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/ManasMehta1110/Optical-Guided-Super-Resolution.git
cd Optical-Guided-Super-Resolution
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include: `torch`, `torchvision`, `numpy`, `scikit-image`, `matplotlib`, `rasterio`, `streamlit`, `Pillow`, `tqdm`, `earthaccess`

### 3. Run the Streamlit Demo Locally

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser. Upload the provided `sample_12_optical.tif` and `sample_12_thermal.tif` files to test inference.

### 4. Run Training (Notebook)

Open `main2.ipynb` in Jupyter or Google Colab. The notebook covers:
- Data loading and preprocessing from Landsat-8 tiles
- Two-phase training (decoder-only → full fine-tuning)
- Evaluation with PSNR, SSIM, and RMSE
- Inference and visualisation

---

## Programmatic Usage

```python
from models.dual_edsr import InfraNovaModel
import torch

model = InfraNovaModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# thermal: (B, 1, H, W) tensor at 30m resolution
# optical: (B, 3, H/3, H/3) tensor at 10m resolution  ← note: 3x smaller spatial dims
with torch.no_grad():
    sr_output = model(thermal, optical)  # returns (B, 1, H*3, W*3) at 10m
```

---

## Training Details

Training uses a two-phase strategy for stable convergence:

**Phase 1 — Decoder Only (Warm-up)**
- Thermal and Optical encoders are frozen
- Only the EDSR decoder weights are updated
- Stabilises reconstruction before full joint training

**Phase 2 — Full Fine-Tuning**
- All weights unfrozen
- Small learning rate to prevent encoder representations from collapsing

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 → 1e-5 |
| Loss Function | L1 + SSIM (+ optional edge loss) |
| Epochs | 50–100 |
| Batch Size | 8–16 |

---

## Dataset

| Source | Description |
|--------|-------------|
| [Landsat-8 TIRS](https://www.usgs.gov/landsat-missions/landsat-8) | Thermal Infrared Sensor — 30m resolution, Band 10 |
| [Landsat-8 OLI](https://www.usgs.gov/landsat-missions/landsat-8) | Optical Land Imager — 10m resolution, Bands 2–7 |
| HuggingFace curated tiles | Pre-cropped and co-registered patch pairs |

Data is downloaded and preprocessed using the [`earthaccess`](https://earthaccess.readthedocs.io/) library. See `main2.ipynb` for the full data pipeline.

---

## FAQs

**Why dual-stream instead of a single encoder?**
A single encoder would force the model to mix radiometric and spatial features from the start, degrading both. Separate encoders preserve domain-specific representations until the fusion module combines them deliberately.

**Does the model hallucinate spatial features onto the thermal output?**
The L1 + SSIM loss combination specifically penalises radiometric drift. The model is trained to enhance spatial sharpness only where the optical image provides consistent structural cues.

**Can this work on other sensors (Sentinel, ECOSTRESS, UAV)?**
Yes, with retraining on co-registered pairs from those sensors. The architecture is sensor-agnostic.

**Is real-time inference possible?**
Not on CPU at full resolution. After ONNX or TorchScript export and GPU deployment, near-real-time inference is feasible for patch-based processing.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch) — original EDSR implementation
- [earthaccess](https://earthaccess.readthedocs.io/) — NASA Earthdata API access
- Smart India Hackathon 2024 — problem statement and domain guidance
