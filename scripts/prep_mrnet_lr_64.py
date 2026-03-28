#!/usr/bin/env python3
"""
Re-prepare MRNet LR images at 64x64 (true 4x downsampling from 256x256 HR).

The current MRNet LR images are stored at 256x256 (same as HR), making the
SR task a refinement rather than true super-resolution. This script creates
proper 64x64 LR images by downsampling HR with bicubic interpolation.

Usage:
    python scripts/prep_mrnet_lr_64.py \
        --hr-dir /path/to/MRNet-v1.0-middle/{split}/hr \
        --output-dir /path/to/MRNet-v1.0-middle/{split}/lr_64

See issue #20.
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Downsample MRNet HR to 64x64 LR")
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hr_files = sorted(args.hr_dir.glob("*.png"))
    print(f"Downsampling {len(hr_files)} images from {args.hr_dir} to {args.size}x{args.size}")

    for f in tqdm(hr_files, desc="Downsampling"):
        img = Image.open(f).convert("L")
        lr = img.resize((args.size, args.size), Image.BICUBIC)
        lr.save(args.output_dir / f.name)

    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
