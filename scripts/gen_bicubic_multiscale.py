#!/usr/bin/env python3
"""
Issue #92: Generate 2x and 8x bicubic SR images for BraTS and CXR.

For each scale factor (2, 8):
  1. Downsample HR image: 256×256 → 256/scale
  2. Upsample back to 256×256 via bicubic interpolation
  3. Save to outputs/experiments/{dataset}_bicubic_{scale}x/sr_images/

The existing 4x bicubic uses LR images (256→64→256); this matches that approach
for consistency but at different scale factors.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_paths import data_root, outputs_root

SCALES = [2, 8]
HR_SIZE = 256


def gen_bicubic(hr_dir: Path, out_base: Path, dataset: str) -> None:
    hr_files = sorted(hr_dir.glob("*.png"))
    if not hr_files:
        print(f"  [SKIP] No PNG files in {hr_dir}")
        return

    print(f"  {dataset}: {len(hr_files)} HR images from {hr_dir}")

    for scale in SCALES:
        lr_size = HR_SIZE // scale
        out_dir = out_base / f"{dataset}_bicubic_{scale}x" / "sr_images"
        out_dir.mkdir(parents=True, exist_ok=True)

        for hr_path in hr_files:
            img = Image.open(hr_path).convert("L")
            # Downsample then upsample (simulates SR degradation at this scale)
            lr = img.resize((lr_size, lr_size), Image.BICUBIC)
            sr = lr.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
            sr.save(out_dir / hr_path.name)

        print(f"    scale={scale}x: {len(hr_files)} images → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-base", default=str(data_root()))
    parser.add_argument("--out-base",  default=str(outputs_root() / "experiments"))
    args = parser.parse_args()

    data_base = Path(args.data_base)
    out_base  = Path(args.out_base)

    datasets = {
        "brats": data_base / "brats2023-sr/valid/hr",
        "cxr":   data_base / "mimic-cxr-sr/valid/hr",
    }

    for dataset, hr_dir in datasets.items():
        if not hr_dir.exists():
            print(f"  [SKIP] HR dir missing: {hr_dir}")
            continue
        gen_bicubic(hr_dir, out_base, dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
