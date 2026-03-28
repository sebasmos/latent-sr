#!/usr/bin/env python3
"""
Create low-resolution (LR) images from high-resolution (HR) images for super-resolution training.

This script downsamples HR images to create LR versions, which are then paired for
diffusion-based super-resolution training.
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def create_lr_images(
    hr_dir: Path,
    lr_dir: Path,
    scale_factor: int = 4,
    interpolation: str = "bicubic"
):
    """
    Create LR images by downsampling HR images.

    Args:
        hr_dir: Directory containing HR images
        lr_dir: Directory to save LR images
        scale_factor: Downsampling factor (default: 4 for 4×)
        interpolation: Interpolation method (bicubic, bilinear, nearest)
    """
    # Create output directory
    lr_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(hr_dir.glob("*.png")) + sorted(hr_dir.glob("*.jpg"))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {hr_dir}")

    print(f"Found {len(image_files)} HR images")
    print(f"Scale factor: {scale_factor}× (e.g., 256×256 → {256//scale_factor}×{256//scale_factor})")
    print(f"Interpolation: {interpolation}")

    # Map interpolation string to PIL constant
    interp_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }

    if interpolation.lower() not in interp_map:
        raise ValueError(f"Invalid interpolation: {interpolation}. Choose from: {list(interp_map.keys())}")

    resample = interp_map[interpolation.lower()]

    # Process each image
    for img_path in tqdm(image_files, desc="Creating LR images"):
        # Load HR image
        hr_img = Image.open(img_path)

        # Get dimensions
        width, height = hr_img.size
        new_width = width // scale_factor
        new_height = height // scale_factor

        # Downsample to create LR
        lr_img = hr_img.resize((new_width, new_height), resample=resample)

        # Upsample back to original size for network input
        lr_img_upsampled = lr_img.resize((width, height), resample=resample)

        # Save LR image (upsampled to match HR dimensions)
        lr_path = lr_dir / img_path.name
        lr_img_upsampled.save(lr_path)

    print(f"\n✅ Created {len(image_files)} LR images in {lr_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create LR images from HR images for super-resolution training"
    )
    parser.add_argument(
        "--hr-root",
        type=Path,
        required=True,
        help="Root directory containing splits with hr/ subdirectories"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Which splits to process (default: train)"
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=4,
        help="Downsampling factor (default: 4 for 4× SR)"
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Interpolation method (default: bicubic)"
    )

    args = parser.parse_args()

    print("="*80)
    print("LR Image Generator for Super-Resolution")
    print("="*80)
    print(f"HR root: {args.hr_root}")
    print(f"Splits: {args.splits}")
    print(f"Scale factor: {args.scale_factor}×")
    print(f"Interpolation: {args.interpolation}")
    print("="*80)

    for split in args.splits:
        hr_dir = args.hr_root / split / "hr"
        lr_dir = args.hr_root / split / "lr"

        if not hr_dir.exists():
            print(f"\n⚠️  Warning: {hr_dir} not found, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {split} split")
        print(f"{'='*80}")

        create_lr_images(
            hr_dir=hr_dir,
            lr_dir=lr_dir,
            scale_factor=args.scale_factor,
            interpolation=args.interpolation
        )

    print(f"\n{'='*80}")
    print("✅ LR image creation complete!")
    print(f"\n📋 Next steps:")
    print(f"1. Verify LR images were created:")
    for split in args.splits:
        lr_dir = args.hr_root / split / "lr"
        if lr_dir.exists():
            print(f"   ls {lr_dir}")
    print(f"\n2. Extract MedVAE latents:")
    print(f"   python medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \\")
    print(f"     --data-root {args.hr_root} \\")
    print(f"     --latent-root embeddings/knee_phase1_latents \\")
    print(f"     --model-name medvae_4_3_2d \\")
    print(f"     --modality mri")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
