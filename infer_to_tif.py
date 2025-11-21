"""
Run inference on a single sample from a manifest and write SR GeoTIFF.

Usage example:
  python infer_to_tif.py --ckpt dualedsr_band10_scale4.pth \
    --manifest manifest_val.json --idx 0 --band 10 --scale 4 \
    --out sr_band10_idx0.tif

Options:
  --scene-id LC08_...   # pick by scene_id instead of idx
  --denorm 65535        # multiply outputs before writing (if you want DN-like values)
  --device cpu|cuda|mps
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from dataset_ssl4eo import SSL4EOThermalSR
from models.dual_edsr import DualEDSRGated


def pick_index(manifest, idx, scene_id):
    if scene_id is None:
        return idx
    for i, entry in enumerate(manifest):
        if entry.get("scene_id") == scene_id:
            return i
    raise ValueError(f"scene_id {scene_id} not found in manifest")


def main():
    ap = argparse.ArgumentParser(description="Run SR inference on one sample and save GeoTIFF.")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    ap.add_argument("--manifest", default="manifest_val.json", help="Manifest JSON")
    ap.add_argument("--idx", type=int, default=0, help="Index in manifest to process")
    ap.add_argument("--scene-id", default=None, help="Optional scene_id to select instead of idx")
    ap.add_argument("--band", type=int, default=10, choices=[10, 11], help="Thermal band (10 or 11)")
    ap.add_argument("--scale", type=int, default=3, help="Downsample factor used in training (e.g., 3 for 30m->10m)")
    ap.add_argument("--out", default="sr.tif", help="Output GeoTIFF path")
    ap.add_argument("--denorm", type=float, default=None, help="Multiply SR by this before writing (e.g., 65535)")
    # prefer CUDA, then MPS (macOS 14+), else CPU
    if torch.cuda.is_available():
        default_dev = "cuda"
    elif torch.backends.mps.is_available():
        default_dev = "mps"
    else:
        default_dev = "cpu"
    ap.add_argument("--device", default=default_dev, help="cuda | mps | cpu")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    sel_idx = pick_index(manifest, args.idx, args.scene_id)
    meta = manifest[sel_idx]
    print(f"Selected idx {sel_idx} scene_id {meta['scene_id']}")

    device = torch.device(args.device)

    # load sample
    ds = SSL4EOThermalSR(args.manifest, thermal_band=args.band, scale=args.scale)
    sample = ds[sel_idx]
    lr_t = sample["lr"].unsqueeze(0).to(device)
    rgb = sample["rgb"].unsqueeze(0).to(device)

    # load model
    model = DualEDSRGated().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        sr = model(lr_t, rgb)
    sr_np = sr.squeeze(0).squeeze(0).cpu().numpy()
    if args.denorm:
        sr_np = sr_np * args.denorm

    # read profile from source tif to preserve georef
    tif_path = Path(meta["tif"])
    with rasterio.open(tif_path) as src:
        profile = src.profile

    profile.update(count=1, dtype="float32", nodata=None)
    out_path = Path(args.out)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(sr_np.astype(np.float32), 1)

    print(f"Wrote SR GeoTIFF to {out_path}")


if __name__ == "__main__":
    main()
