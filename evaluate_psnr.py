"""
Compute PSNR between a super-resolved thermal GeoTIFF and ground-truth thermal GeoTIFF.

Usage:
  python evaluate_psnr.py --sr path/to/sr.tif --gt path/to/gt.tif [--data_range 1.0]

Notes:
- The GT is reprojected/resampled to the SR grid if needed (CRS/transform/shape).
- If you trained on normalized [0,1], set --data_range 1.0. For physical units (Kelvin),
  either omit --data_range to auto-use GT min–max, or pass a fixed range.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.metrics import peak_signal_noise_ratio as psnr


def load_raster(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        profile = ds.profile
        nodata = ds.nodata
    return arr, profile, nodata


def align_to_sr(gt, gt_profile, sr_profile):
    same_grid = (
        gt_profile.get("crs") == sr_profile.get("crs")
        and gt_profile.get("transform") == sr_profile.get("transform")
        and gt_profile.get("height") == sr_profile.get("height")
        and gt_profile.get("width") == sr_profile.get("width")
    )
    if same_grid:
        return gt

    dst = np.zeros((sr_profile["height"], sr_profile["width"]), dtype=np.float32)
    reproject(
        source=gt,
        destination=dst,
        src_transform=gt_profile["transform"],
        src_crs=gt_profile["crs"],
        dst_transform=sr_profile["transform"],
        dst_crs=sr_profile["crs"],
        resampling=Resampling.bilinear,
    )
    return dst


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR between SR and GT GeoTIFFs.")
    parser.add_argument("--sr", required=True, help="Path to SR GeoTIFF")
    parser.add_argument("--gt", required=True, help="Path to GT GeoTIFF")
    parser.add_argument(
        "--data_range",
        type=float,
        default=None,
        help="Value range for PSNR. Set to 1.0 if using normalized [0,1]. "
             "If omitted, uses GT min–max.",
    )
    args = parser.parse_args()

    sr_path = Path(args.sr)
    gt_path = Path(args.gt)
    if not sr_path.exists() or not gt_path.exists():
        sys.stderr.write("SR or GT path does not exist.\n")
        sys.exit(1)

    sr, sr_prof, sr_nodata = load_raster(sr_path)
    gt, gt_prof, gt_nodata = load_raster(gt_path)

    gt_aligned = align_to_sr(gt, gt_prof, sr_prof)

    mask = np.isfinite(gt_aligned) & np.isfinite(sr)
    if gt_nodata is not None:
        mask &= gt_aligned != gt_nodata
    if sr_nodata is not None:
        mask &= sr != sr_nodata

    if not np.any(mask):
        sys.stderr.write("No valid pixels after masking; cannot compute PSNR.\n")
        sys.exit(1)

    gt_m = gt_aligned[mask]
    sr_m = sr[mask]

    dr = args.data_range
    if dr is None:
        dr = float(gt_m.max() - gt_m.min() + 1e-8)

    score = psnr(gt_m, sr_m, data_range=dr)
    print(f"PSNR: {score:.4f} dB  (data_range={dr:.5g}, pixels={mask.sum()})")


if __name__ == "__main__":
    main()
