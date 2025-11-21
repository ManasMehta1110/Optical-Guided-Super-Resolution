"""
Simple training loop for dual-stream thermal SR on SSL4EO (Landsat-8).

Assumes you have:
- manifest_train.json and manifest_val.json created from make_manifest.py output.
- A virtualenv activated: `source .venv/bin/activate`

Usage (defaults: band 10, 4x):
  python train.py --train manifest_train.json --val manifest_val.json --band 10 --scale 4
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_ssl4eo import SSL4EOThermalSR
from models.dual_edsr import DualEDSRGated


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return float("inf")
    return 10 * math.log10(data_range * data_range / mse.item())


def get_loaders(train_path, val_path, band, scale, batch_size, num_workers, pin_memory, patch_hr, center_crop):
    patch_hr_use = patch_hr if patch_hr and patch_hr > 0 else None
    train_ds = SSL4EOThermalSR(
        train_path,
        thermal_band=band,
        scale=scale,
        patch_hr=patch_hr_use,
        center_crop=center_crop and patch_hr_use is not None,
    )
    val_ds = SSL4EOThermalSR(
        val_path,
        thermal_band=band,
        scale=scale,
        patch_hr=patch_hr_use,
        center_crop=True if patch_hr_use is not None else False,
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, val_dl


def main():
    ap = argparse.ArgumentParser(description="Train DualEDSRGated on SSL4EO thermal SR.")
    ap.add_argument("--train", default="manifest_train.json", help="Training manifest JSON")
    ap.add_argument("--val", default="manifest_val.json", help="Validation manifest JSON")
    ap.add_argument("--band", type=int, default=10, choices=[10, 11], help="Thermal band to train on (10 or 11)")
    ap.add_argument("--scale", type=int, default=3, help="Downsample factor to create LR input (e.g., 3 for 30m->10m)")
    ap.add_argument("--patch-hr", type=int, default=126, help="HR patch size (must be divisible by scale). Set 0 to use full frame.")
    ap.add_argument("--center-crop", action="store_true", help="Use center crop instead of random for patches.")
    ap.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    # prefer CUDA, then MPS (macOS 14+), else CPU
    if torch.cuda.is_available():
        default_dev = "cuda"
    elif torch.backends.mps.is_available():
        default_dev = "mps"
    else:
        default_dev = "cpu"
    ap.add_argument("--device", default=default_dev, help="cuda, mps (macOS 14+), or cpu")
    ap.add_argument("--val-batches", type=int, default=5, help="How many val batches to evaluate each epoch (None for full)")
    ap.add_argument("--save", default="dualedsr_band{band}_scale{scale}.pth", help="Checkpoint path (format with band/scale)")
    ap.add_argument("--split-from", default=None, help="If train/val files are missing, split this manifest into 90/10.")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction when using --split-from.")
    args = ap.parse_args()

    save_path = Path(args.save.format(band=args.band, scale=args.scale))
    device = torch.device(args.device)

    # Create splits on the fly if requested
    if args.split_from:
        base = Path(args.split_from)
        train_path = Path(args.train)
        val_path = Path(args.val)
        if not train_path.exists() or not val_path.exists():
            data = json.loads(base.read_text())
            cutoff = int(len(data) * (1 - args.val_frac))
            train_data = data[:cutoff]
            val_data = data[cutoff:]
            train_path.write_text(json.dumps(train_data, indent=2))
            val_path.write_text(json.dumps(val_data, indent=2))
            print(f"Created splits: {len(train_data)} train, {len(val_data)} val")

    # pin_memory is not supported on MPS; disable there to avoid warnings
    use_pin = args.device.startswith("cuda")

    train_dl, val_dl = get_loaders(
        train_path=args.train,
        val_path=args.val,
        band=args.band,
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        patch_hr=args.patch_hr,
        center_crop=args.center_crop,
    )

    model = DualEDSRGated().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tbar = tqdm(train_dl, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(tbar, 1):
            lr_t = batch["lr"].to(device)
            rgb = batch["rgb"].to(device)
            hr_t = batch["hr"].to(device)

            sr = model(lr_t, rgb)
            loss = F.l1_loss(sr, hr_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 10 == 0:
                tbar.set_postfix(loss=f"{loss.item():.4f}")

        # validation
        model.eval()
        val_losses = []
        val_psnr = []
        with torch.no_grad():
            for vstep, batch in enumerate(val_dl, 1):
                lr_t = batch["lr"].to(device)
                rgb = batch["rgb"].to(device)
                hr_t = batch["hr"].to(device)

                sr = model(lr_t, rgb)
                val_losses.append(F.l1_loss(sr, hr_t).item())
                val_psnr.append(psnr(sr.clamp(0, 1), hr_t.clamp(0, 1)))

                if args.val_batches and vstep >= args.val_batches:
                    break
        mean_val_loss = sum(val_losses) / len(val_losses)
        mean_val_psnr = sum(val_psnr) / len(val_psnr)
        print(f"[epoch {epoch}] val_loss {mean_val_loss:.4f} val_psnr {mean_val_psnr:.2f} dB")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
