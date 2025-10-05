<h1 align="center">🌡️ InfraNova: Dual-Stream EDSR for Thermal Super-Resolution</h1>

<p align="center">
  <i>Thermally Accurate. Spatially Sharp.</i><br>
  <b>Supercharging Thermal Vision with Deep Learning.</b>
</p>

---

<p align="center">
  <a href="https://optical-guided-super-resolution-br4vim97x4bunscqcmj98a.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" height="45"/>
  </a>
</p>

---

## 🚀 Project Overview

**InfraNova** is a deep learning pipeline designed to fuse **Optical** and **Thermal** imagery for generating **high-resolution thermal maps**.  
Using a **Dual-Stream EDSR architecture**, it preserves **spatial sharpness** from optical data while maintaining **temperature accuracy** from thermal data.

> 🎯 **Goal:** Convert coarse (30m) thermal imagery into fine-grained (10m) super-resolved thermal maps aligned with optical data.

---

## 🧩 Model Architecture

The model follows a **Dual-Stream EDSR** workflow:

1. **Feature Extraction**  
   - Separate convolutional encoders extract low-level features from both **Optical** and **Thermal** streams.

2. **Deep Residual Learning**  
   - Multiple **Residual Blocks (Conv → ReLU → Conv + Skip)** capture spatial and contextual details.

3. **Feature Alignment & Fusion**  
   - **ConvFuse** layer merges both modalities.
   - **Channel Attention** ensures adaptive weighting between thermal and optical features.

4. **Reconstruction & Upscaling**  
   - **EDSR-based decoder** reconstructs a high-resolution thermal image, preserving fine structures.

5. **Output**  
   - Produces a **10m resolution** super-resolved thermal map.

<p align="center">
  <img src="assets/dual_stream_pipeline.png" width="700">
  <br><em>Figure: Dual-Stream Optical-Thermal EDSR pipeline.</em>
</p>

---

## 🧠 Key Components

| Component | Description |
|------------|--------------|
| **Optical Encoder** | Extracts texture and spatial edges |
| **Thermal Encoder** | Encodes accurate temperature gradients |
| **Feature Alignment** | Aligns feature channels using ConvFuse and Channel Attention |
| **EDSR Decoder** | Performs residual learning and reconstruction |
| **Upscaling** | Increases spatial resolution to 10m |

---

## 📊 Performance Metrics

| Metric | Value |
|:-------|:------|
| **PSNR** | 42.4 dB |
| **SSIM** | 0.9269 |
| **RMSE** | 0.0159 |

---

## 🧰 Tech Stack

| Domain | Libraries / Tools |
|:--------|:------------------|
| **Deep Learning** | PyTorch, TorchVision |
| **Computer Vision** | Pillow, OpenCV |
| **Data Handling** | Rasterio, NumPy |
| **Web App** | Streamlit |
| **Datasets** | NASA Landsat-8, Hugging Face |

---

## 💻 Getting Started

### Clone the Repository
```bash
git clone https://github.com/YourUsername/InfraNova.git
cd InfraNova
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```

Or open the hosted demo:

<p align="center">
  <a href="https://optical-guided-super-resolution-br4vim97x4bunscqcmj98a.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Launch Demo" height="45"/>
  </a>
</p>

---

## 🗂️ Repository Structure

```
InfraNova/
├── app.py                     # Streamlit demo app
├── models/
│   ├── dual_edsr.py           # Dual-stream EDSR model
│   ├── resblock.py            # Residual block implementation
├── data_raw/                  # Raw .tif input files (Optical, Thermal)
├── data_processed/            # Super-resolved results / trained model
├── utils/                     # Helper scripts for visualization & metrics
├── requirements.txt
└── README.md
```

---

## 🧪 Usage Instructions

1. Upload your **Optical** and **Thermal** `.tif` files in the Streamlit interface.
2. The model will:
   - Extract features from both inputs  
   - Align and fuse using attention  
   - Perform EDSR-based upscaling  
3. Outputs:
   - Super-Resolved Thermal Image  
   - Comparison metrics (PSNR, SSIM, RMSE)  

---

## 🖼️ Example Results

| Optical Input | Thermal Input | Super-Resolved Output |
|:--------------:|:--------------:|:----------------------:|
| ![optical](assets/optical.png) | ![thermal](assets/thermal.png) | ![sr](assets/super_resolved.png) |

---

## 📁 Dataset Sources

- **NASA Landsat-8** – Thermal Infrared Sensor (TIRS)  
- **Hugging Face Datasets** – Curated paired optical and thermal tiles  

---

## 🧮 Physics-Informed Loss (Optional Extension)

The model can integrate a **physics-informed constraint** ensuring temperature conservation:

\[
\mathcal{L}_{total} = \mathcal{L}_{pixel} + \lambda_1 \mathcal{L}_{perceptual} + \lambda_2 \mathcal{L}_{physics}
\]

Where:
- \( \mathcal{L}_{physics} \) enforces radiometric consistency.  
- \( \lambda_1, \lambda_2 \) are tunable coefficients.

---

## 🧑‍💻 Contributors

- **Manas Mehta** – Model Development & Streamlit Integration  
- **[Add teammates here]** – Data Handling, Visualization, and Documentation  

---

## 🏁 Acknowledgements

This project is part of **Smart India Hackathon 2025**.  
Grateful to open-source communities and satellite data providers (NASA, ESA) for making this research possible.

---

<p align="center">
  <i>“Supercharging Thermal Vision with Deep Learning.”</i><br>
  <b>InfraNova © 2025</b>
</p>
