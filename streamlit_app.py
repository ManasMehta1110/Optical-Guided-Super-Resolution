from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import streamlit as st
import torch

from models.dual_edsr import DualEDSR

MODEL_PATH = Path("data_processed/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model weights not found at `{MODEL_PATH}`. Please place the trained `best_model.pth` there.")
        st.stop()
    model = DualEDSR().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + EPS) if mx - mn > EPS else np.zeros_like(arr)


def load_raster(path, is_optical=False):
    with rasterio.open(path) as src:
        profile = src.profile
        if is_optical:
            data = src.read([1, 2, 3]).astype(np.float32)
            for c in range(3):
                data[c] = _normalize(data[c])
        else:
            data = src.read(1).astype(np.float32)
            data = _normalize(data)
    return data, profile


def align_thermal_to_optical(thermal, t_profile, o_profile):
    needs_reproject = (t_profile.get("crs") != o_profile.get("crs")) or (
        t_profile.get("transform") != o_profile.get("transform")
    ) or (thermal.shape[-2:] != (o_profile["height"], o_profile["width"]))

    if not needs_reproject:
        return thermal, t_profile, False

    dst = np.zeros((o_profile["height"], o_profile["width"]), dtype=np.float32)
    reproject(
        source=thermal,
        destination=dst,
        src_transform=t_profile["transform"],
        src_crs=t_profile["crs"],
        dst_transform=o_profile["transform"],
        dst_crs=o_profile["crs"],
        resampling=Resampling.bilinear,
    )
    new_profile = o_profile.copy()
    new_profile.update(count=1, dtype="float32")
    return dst, new_profile, True


def run_inference(optical_path, thermal_path):
    opt, opt_profile = load_raster(optical_path, is_optical=True)
    thr, thr_profile = load_raster(thermal_path, is_optical=False)

    thr_aligned, thr_profile_aligned, was_resampled = align_thermal_to_optical(
        thr, thr_profile, opt_profile
    )

    xO = torch.from_numpy(opt).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    xT = torch.from_numpy(thr_aligned).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]

    with torch.no_grad():
        sr = load_model()(xT, xO)

    sr = sr.squeeze().cpu().numpy()
    opt_plot = np.transpose(opt, (1, 2, 0))  # [H, W, 3]
    return opt_plot, thr_aligned, sr, opt_profile, was_resampled


def save_geotiff(sr, profile):
    tmp = NamedTemporaryFile(delete=False, suffix=".tif")
    profile_out = profile.copy()
    profile_out.update(count=1, dtype="float32")
    with rasterio.open(tmp.name, "w", **profile_out) as dst:
        dst.write(sr.astype(np.float32), 1)
    return tmp.name


def to_png_bytes(arr, cmap="inferno"):
    arr_norm = _normalize(arr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr_norm, cmap=cmap)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf


st.title("🌍 Optical-Guided Thermal Super-Resolution")
st.write("Upload optical and thermal GeoTIFFs. The app checks CRS/transform alignment, resamples thermal if needed, and returns SR thermal plus downloads.")

opt_file = st.file_uploader("Upload Optical Image (GeoTIFF)", type=["tif", "tiff"])
thr_file = st.file_uploader("Upload Thermal Image (GeoTIFF)", type=["tif", "tiff"])

if opt_file and thr_file:
    opt_path = Path("temp_opt.tif")
    thr_path = Path("temp_thr.tif")

    with open(opt_path, "wb") as f:
        f.write(opt_file.read())
    with open(thr_path, "wb") as f:
        f.write(thr_file.read())

    try:
        with st.spinner("Running inference..."):
            opt, thr, sr, profile, was_resampled = run_inference(opt_path, thr_path)
    except Exception as e:
        st.error("Inference failed. Please ensure the files are GeoTIFFs with valid CRS/transform.")
        st.exception(e)
        st.stop()

    if was_resampled:
        st.info("Thermal image was reprojected/resampled to optical CRS/grid for alignment.")

    st.subheader("Results")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(opt)
    axs[0].set_title("Optical (guidance)")
    axs[0].axis("off")

    axs[1].imshow(thr, cmap="inferno")
    axs[1].set_title("Thermal (aligned)")
    axs[1].axis("off")

    axs[2].imshow(sr, cmap="inferno")
    axs[2].set_title("Super-Resolved Thermal")
    axs[2].axis("off")

    st.pyplot(fig)

    geotiff_path = save_geotiff(sr, profile)
    png_buf = to_png_bytes(sr)

    st.download_button(
        label="Download SR Thermal (GeoTIFF)",
        data=open(geotiff_path, "rb").read(),
        file_name="sr_thermal.tif",
        mime="image/tiff",
    )
    st.download_button(
        label="Download SR Thermal (PNG preview)",
        data=png_buf,
        file_name="sr_thermal.png",
        mime="image/png",
    )
