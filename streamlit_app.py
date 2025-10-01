import os
import sys
import math
import logging
from glob import glob
from tqdm import tqdm
import earthaccess
import rasterio
from rasterio.enums import Resampling
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from models.dual_edsr import DualEDSR  # Assuming this is your model file

MODEL_PATH = Path("data_processed/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = DualEDSR().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

def load_band(path, rescale=True, is_optical=False):
    with rasterio.open(path) as src:
        if is_optical:
            # Read 3 bands (assuming band 1=R, 2=G, 3=B)
            band = src.read([1, 2, 3]).astype(np.float32)  # Shape: [3, H, W]
            if rescale:
                # Normalize per channel
                for c in range(3):
                    mn, mx = band[c].min(), band[c].max()
                    if mx - mn > 1e-8:
                        band[c] = (band[c] - mn) / (mx - mn)
        else:
            band = src.read(1).astype(np.float32)  # Single band: [H, W]
            if rescale:
                mn, mx = band.min(), band.max()
                if mx - mn > 1e-8:
                    band = (band - mn) / (mx - mn)
        return band

def run_inference(optical_path, thermal_path):
    opt = load_band(optical_path, is_optical=True)  # [3, H, W] -> will be permuted below
    thr = load_band(thermal_path)  # [H, W]

    # For optical: Permute to [C, H, W] and add batch dim -> [1, 3, H, W]
    xO = torch.from_numpy(opt).permute(1, 2, 0).unsqueeze(0).to(DEVICE)  # Wait, no: opt is [3,H,W], so permute(0,1,2) is already [3,H,W], then unsqueeze(0) to [1,3,H,W]
    # Correction: Since read([1,2,3]) gives [3,H,W], just:
    xO = torch.from_numpy(opt).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    # For thermal: [H, W] -> [1, 1, H, W]
    xT = torch.from_numpy(thr).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sr = model(xT, xO)

    sr = sr.squeeze().cpu().numpy()

    # For plotting: opt is [3, H, W], transpose to [H, W, 3] for imshow
    opt_plot = np.transpose(opt, (1, 2, 0))
    return opt_plot, thr, sr

st.title("🌍 Optical-Guided Thermal Super-Resolution")
st.write("Upload optical and thermal images, and the model will super-resolve thermal resolution to 10m.")

opt_file = st.file_uploader("Upload Optical Image (GeoTIFF)", type=["tif", "tiff"])
thr_file = st.file_uploader("Upload Thermal Image (GeoTIFF)", type=["tif", "tiff"])

if opt_file and thr_file:
    opt_path = Path("temp_opt.tif")
    thr_path = Path("temp_thr.tif")

    with open(opt_path, "wb") as f:
        f.write(opt_file.read())
    with open(thr_path, "wb") as f:
        f.write(thr_file.read())

    st.info("Running inference on uploaded images...")
    opt, thr, sr = run_inference(opt_path, thr_path)

    st.subheader("Results")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(opt)  # No cmap for RGB
    axs[0].set_title("Optical (10m)")
    axs[0].axis("off")

    axs[1].imshow(thr, cmap="inferno")
    axs[1].set_title("Thermal Input")
    axs[1].axis("off")

    axs[2].imshow(sr, cmap="inferno")
    axs[2].set_title("Super-Resolved Thermal")
    axs[2].axis("off")

    st.pyplot(fig)
