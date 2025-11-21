"""
Minimal PyTorch Dataset for SSL4EO Landsat-8 tiles listed in manifest.json.

Each manifest entry should have:
{
  "scene_id": "...",
  "tile_id": "...",
  "tif": "data_raw/0000000/LC08_.../all_bands.tif",
  "thermal_bands": [10, 11],
  "rgb_bands": [2, 3, 4]
}

Returns (lr_thermal, rgb, hr_thermal) tensors for training.
"""

import json
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SSL4EOThermalSR(Dataset):
    def __init__(
        self,
        manifest_path: str,
        thermal_band: int = 10,
        scale: int = 3,
        normalize_divisor: Optional[float] = 65535.0,
        patch_hr: Optional[int] = None,
        center_crop: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        manifest_path: path to manifest.json produced by make_manifest.py
        thermal_band: 10 or 11 (1-based indexing into all_bands.tif)
        scale: downsample factor to create LR input (2–4 typical; 3 for 30m->10m)
        normalize_divisor: divide DN by this value; set None to skip normalization
        patch_hr: optional HR patch size (pixels on thermal/optical grid). Must be divisible by scale.
        center_crop: if True, use center crop; otherwise random crop each call.
        transform: optional callable applied to dict sample before return
        """
        self.items = json.loads(Path(manifest_path).read_text())
        self.thermal_band = thermal_band
        self.scale = scale
        self.normalize_divisor = normalize_divisor
        self.transform = transform
        self.patch_hr = patch_hr
        self.center_crop = center_crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        tif_path = meta["tif"]

        with rasterio.open(tif_path) as ds:
            thr = ds.read(self.thermal_band).astype(np.float32)  # HR thermal
            rgb = ds.read([2, 3, 4]).astype(np.float32)           # RGB guidance

        if self.normalize_divisor:
            thr /= self.normalize_divisor
            rgb /= self.normalize_divisor

        if self.patch_hr is not None:
            if self.patch_hr % self.scale != 0:
                raise ValueError(f"patch_hr {self.patch_hr} must be divisible by scale {self.scale}")
            h, w = thr.shape
            ph = self.patch_hr
            pw = self.patch_hr
            if h < ph or w < pw:
                raise ValueError(f"Patch {ph}x{pw} bigger than image {h}x{w}")
            if self.center_crop:
                top = (h - ph) // 2
                left = (w - pw) // 2
            else:
                top = random.randint(0, h - ph)
                left = random.randint(0, w - pw)
            thr = thr[top : top + ph, left : left + pw]
            rgb = rgb[:, top : top + ph, left : left + pw]

        # create LR by strided subsample; swap to rasterio resampling if preferred
        thr_lr = thr[:: self.scale, :: self.scale]

        sample = {
            "lr": torch.from_numpy(thr_lr).unsqueeze(0),  # [1,h,w]
            "rgb": torch.from_numpy(rgb),                 # [3,H,W]
            "hr": torch.from_numpy(thr).unsqueeze(0),     # [1,H,W]
            "meta": meta,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
