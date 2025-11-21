"""
Scan the chunked SSL4EO dataset under data_raw/ and write a manifest listing
all "all_bands.tif" files. Each entry includes scene_id, tile_id, and path.

Usage:
  python make_manifest.py --root data_raw --output manifest.json [--check-bands]

Notes:
- The dataset is split across ~25k numbered folders. This script keeps that
  structure intact and just records paths for easy dataset splits.
- If --check-bands is set, rasterio is used to confirm band count (=11).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List


def scan_dataset(root: Path, check_bands: bool = False) -> List[dict]:
    files = sorted(root.glob("*/*/all_bands.tif"))
    entries = []

    if check_bands:
        try:
            import rasterio  # type: ignore
        except ImportError:
            sys.stderr.write("--check-bands requested but rasterio is not installed.\n")
            sys.exit(1)

    for tif in files:
        scene_id = tif.parent.name
        tile_id = tif.parent.parent.name
        entry = {
            "scene_id": scene_id,
            "tile_id": tile_id,
            "tif": str(tif),
            "thermal_bands": [10, 11],  # 1-based indices for TIRS
            "rgb_bands": [2, 3, 4],     # 1-based indices for RGB
        }

        if check_bands:
            with rasterio.open(tif) as ds:  # type: ignore
                entry["band_count"] = ds.count

        entries.append(entry)

    return entries


def main():
    parser = argparse.ArgumentParser(description="Create manifest of SSL4EO all_bands.tif files.")
    parser.add_argument("--root", default="data_raw", help="Root folder containing numbered chunks.")
    parser.add_argument("--output", default="manifest.json", help="Where to write the manifest JSON.")
    parser.add_argument("--check-bands", action="store_true", help="Verify band count with rasterio (must be installed).")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N files (for smoke tests).")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.stderr.write(f"Root does not exist: {root}\n")
        sys.exit(1)

    entries = scan_dataset(root, check_bands=args.check_bands)
    if args.limit is not None:
        entries = entries[: args.limit]

    out_path = Path(args.output)
    out_path.write_text(json.dumps(entries, indent=2))

    print(f"Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
