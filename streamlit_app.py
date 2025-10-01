import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from pathlib import Path

from models.dual_edsr import DualEDSR

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


def load_band(path, rescale=True):
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
        if rescale:
            band = (band - band.min()) / (band.max() - band.min() + 1e-8)
        return band


def run_inference(optical_path, thermal_path):
    opt = load_band(optical_path)   
    thr = load_band(thermal_path)   

    
    xO = torch.from_numpy(opt).unsqueeze(0).unsqueeze(0).to(DEVICE)  
    xT = torch.from_numpy(thr).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sr = model(xT, xO)

    sr = sr.squeeze().cpu().numpy()
    return opt, thr, sr



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
    axs[0].imshow(opt, cmap="gray")
    axs[0].set_title("Optical (10m)")
    axs[0].axis("off")

    axs[1].imshow(thr, cmap="inferno")
    axs[1].set_title("Thermal Input")
    axs[1].axis("off")

    axs[2].imshow(sr, cmap="inferno")
    axs[2].set_title("Super-Resolved Thermal")
    axs[2].axis("off")

    st.pyplot(fig)

