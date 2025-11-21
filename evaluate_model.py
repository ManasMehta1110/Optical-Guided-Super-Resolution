"""
Compute PSNR of a saved checkpoint on a manifest split (no GeoTIFF writing).

Usage:
  python evaluate_model.py --ckpt dualedsr_band10_scale4.pth --manifest manifest_val.json --band 10 --scale 4

Options:
  --device cpu|cuda|mps
  --batch-size 4
  --limit N          # evaluate only first N samples (for a quick check)
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset_ssl4eo import SSL4EOThermalSR
from models.dual_edsr import DualEDSRGated


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse.item() == 0:
        return float("inf")
    return 10 * math.log10(data_range * data_range / mse.item())


def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target, reduction="mean")).item()


def main():
    ap = argparse.ArgumentParser(description="Evaluate checkpoint PSNR on a manifest split.")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pth file")
    ap.add_argument("--manifest", default="manifest_val.json", help="Manifest JSON to evaluate")
    ap.add_argument("--band", type=int, default=10, choices=[10, 11], help="Thermal band (10 or 11)")
    ap.add_argument("--scale", type=int, default=3, help="Downsample factor used in training (e.g., 3 for 30m->10m)")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda | mps | cpu")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 avoids macOS shm issues)")
    ap.add_argument("--summary-csv", default=None, help="Optional CSV path to save aggregate metrics")
    ap.add_argument("--log-prefix", default="infranova_ssl4eo", help="Prefix for INFO log-style prints")
    args = ap.parse_args()

    device = torch.device(args.device)

    ds_full = SSL4EOThermalSR(args.manifest, thermal_band=args.band, scale=args.scale)
    if args.limit is not None:
        indices = list(range(min(args.limit, len(ds_full))))
        ds = Subset(ds_full, indices)
    else:
        ds = ds_full

    pin = args.device.startswith("cuda")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = DualEDSRGated().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    psnrs = []
    ssims = []
    rmses = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="eval"):
            lr_t = batch["lr"].to(device)
            rgb = batch["rgb"].to(device)
            hr_t = batch["hr"].to(device)
            sr = model(lr_t, rgb)
            sr_c = sr.clamp(0, 1)
            hr_c = hr_t.clamp(0, 1)
            psnrs.append(psnr(sr_c, hr_c))
            rmses.append(rmse(sr_c, hr_c))
            # compute SSIM per-batch averaged over samples
            sr_np = sr_c.cpu().numpy()
            hr_np = hr_c.cpu().numpy()
            batch_ssim = []
            for i in range(sr_np.shape[0]):
                batch_ssim.append(
                    ssim(
                        hr_np[i, 0],
                        sr_np[i, 0],
                        data_range=1.0,
                    )
                )
            ssims.extend(batch_ssim)

    mean_psnr = sum(psnrs) / len(psnrs)
    mean_rmse = sum(rmses) / len(rmses)
    mean_ssim = sum(ssims) / len(ssims)
    prefix = args.log_prefix
    print(f"INFO:{prefix}: EVAL SUMMARY: PSNR={mean_psnr:.3f} dB, SSIM={mean_ssim:.4f}, RMSE={mean_rmse:.6f}")

    if args.summary_csv:
        out_path = Path(args.summary_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["psnr_mean_db", f"{mean_psnr:.6f}"])
            writer.writerow(["ssim_mean", f"{mean_ssim:.6f}"])
            writer.writerow(["rmse_mean", f"{mean_rmse:.6f}"])
        print(f"INFO:{prefix}: Saved summary -> {out_path}")


if __name__ == "__main__":
    main()
