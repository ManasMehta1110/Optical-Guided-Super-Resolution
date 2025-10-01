# Optical-Guided Thermal Super-Resolution

This repository contains an implementation of an Optical-Guided Super-Resolution (OGSR) framework that enhances thermal satellite images (30m resolution) to higher resolution (10m) using optical bands as guidance. The project uses deep learning techniques inspired by Enhanced Deep Super-Resolution (EDSR) with dual-stream inputs.

## Features
- Downloads HLS (Harmonized Landsat Sentinel-2) data using NASA Earthdata via `earthaccess`
- Extracts relevant optical and thermal bands
- Resamples thermal bands to optical grid
- Trains a Dual-EDSR based model for super-resolution
- Evaluates using PSNR, SSIM, and RMSE
- Provides inference pipeline and Streamlit demo for real-world testing

## Tech Stack
- **Python 3.9+**
- **PyTorch** for model training and evaluation
- **rasterio** for geospatial raster processing
- **earthaccess** for NASA Earthdata download
- **scikit-image** for evaluation metrics (SSIM)
- **Streamlit** for deployment and interactive demo

## Training
To train the model:
```bash
python main.py
```
Training consists of:
1. Downloading L30 and S30 granules from NASA Earthdata
2. Extracting and aligning bands
3. Resampling 30m thermal to 10m optical resolution
4. Training DualEDSR with supervised learning

## Evaluation
Validation metrics include:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**
- **RMSE (Root Mean Square Error)**

These values are logged during training.

## Deployment
You can run a Streamlit app to test super-resolution on new images:
```bash
streamlit run app.py
```

## Repository Structure
```
├── main.py              # Main training pipeline
├── app.py               # Streamlit inference app
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Results
Example validation scores after 2 epochs:
- PSNR ≈ 40.68
- SSIM ≈ 0.927
- RMSE ≈ 0.0164

## Acknowledgements
- NASA Earthdata for open-access satellite imagery
- PyTorch community for model building blocks
- Original EDSR authors for architecture inspiration
