#!/usr/bin/env python3
"""
Prepare BraTS 2023 brain MRI for SR pipeline.

Extracts 2D axial slices from 3D NIfTI volumes (T2w modality),
creates HR (256x256) / LR (64x64) pairs, and saves segmentation
masks for downstream evaluation.

Each subject contributes N slices from the middle portion of the volume
(skipping empty border slices).
"""

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_nifti_volume(path):
    """Load a NIfTI volume and return as float32 numpy array."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data


def normalize_slice(arr):
    """Normalize a 2D slice to [0, 255] uint8."""
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def extract_slices(volume, seg_volume, slice_range, n_slices, seed):
    """Extract axial slices from middle portion of volume."""
    depth = volume.shape[2]  # BraTS is (H, W, D)
    start = int(depth * slice_range[0])
    end = int(depth * slice_range[1])

    # Filter out slices with very little brain content
    valid_indices = []
    for z in range(start, end):
        sl = volume[:, :, z]
        if sl.max() > sl.mean() + sl.std():  # has some signal
            valid_indices.append(z)

    if not valid_indices:
        return [], []

    rng = random.Random(seed)
    n = min(n_slices, len(valid_indices))
    selected = rng.sample(valid_indices, n)

    slices = []
    masks = []
    for z in sorted(selected):
        slices.append(volume[:, :, z])
        masks.append((seg_volume[:, :, z] > 0).astype(np.uint8) * 255)  # binary tumor mask

    return slices, masks


def process_subjects(subject_dirs, output_dir, split_name, mri_seq, slice_range, n_slices, seed):
    """Process all subjects in a split."""
    hr_dir = output_dir / split_name / "hr"
    lr_dir = output_dir / split_name / "lr"
    seg_dir = output_dir / split_name / "seg_masks"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    for subj_dir in tqdm(subject_dirs, desc=f"{split_name}"):
        subj_name = subj_dir.name

        # Find the MRI sequence and segmentation files
        # BraTS structure can be either:
        #   {subj}/{subj}-t2w.nii.gz  (flat)
        #   {subj}/{subj}-t2w.nii/    (directory containing the actual .nii inside)
        mri_file = None
        for candidate in [
            subj_dir / f"{subj_name}-{mri_seq}.nii.gz",
            subj_dir / f"{subj_name}-{mri_seq}.nii",
        ]:
            if candidate.is_file():
                mri_file = candidate
                break
            elif candidate.is_dir():
                # Nested: directory contains the actual .nii file
                nii_files = list(candidate.glob("*.nii")) + list(candidate.glob("*.nii.gz"))
                if nii_files:
                    mri_file = nii_files[0]
                    break

        seg_file = None
        for candidate in [
            subj_dir / f"{subj_name}-seg.nii.gz",
            subj_dir / f"{subj_name}-seg.nii",
        ]:
            if candidate.is_file():
                seg_file = candidate
                break
            elif candidate.is_dir():
                nii_files = list(candidate.glob("*.nii")) + list(candidate.glob("*.nii.gz"))
                if nii_files:
                    seg_file = nii_files[0]
                    break

        if mri_file is None:
            continue

        volume = load_nifti_volume(mri_file)
        seg_volume = load_nifti_volume(seg_file) if seg_file is not None else np.zeros_like(volume)

        slices, masks = extract_slices(volume, seg_volume, slice_range, n_slices, seed)

        for i, (sl, mask) in enumerate(zip(slices, masks)):
            sl_norm = normalize_slice(sl)
            img = Image.fromarray(sl_norm, mode="L")

            # Center crop to square
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))

            hr = img.resize((256, 256), Image.LANCZOS)
            lr = img.resize((64, 64), Image.LANCZOS)

            fname = f"brats_{subj_name}_z{i:02d}.png"
            hr.save(hr_dir / fname)
            lr.save(lr_dir / fname)

            # Save segmentation mask (same crop + resize)
            mask_img = Image.fromarray(mask, mode="L")
            mask_img = mask_img.crop((left, top, left + side, top + side))
            mask_hr = mask_img.resize((256, 256), Image.NEAREST)
            mask_hr.save(seg_dir / fname)

            total_slices += 1

    print(f"  {split_name}: {total_slices} slices from {len(subject_dirs)} subjects")
    return total_slices


def main():
    parser = argparse.ArgumentParser(description="Prepare BraTS 2023 for SR")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="BraTS training data root (contains BraTS-GLI-* dirs)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mri-sequence", type=str, default="t2w",
                        choices=["t1c", "t1n", "t2f", "t2w"])
    parser.add_argument("--slice-range", type=float, nargs=2, default=[0.25, 0.75],
                        help="Fraction of volume depth to use (default: middle 50%%)")
    parser.add_argument("--slices-per-volume", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=35)
    parser.add_argument("--n-test", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Collect all subject directories
    all_subjects = sorted([d for d in args.data_root.iterdir() if d.is_dir() and d.name.startswith("BraTS")])
    print(f"Found {len(all_subjects)} BraTS subjects")

    random.shuffle(all_subjects)

    # Split subjects
    train_subjs = all_subjects[:args.n_train]
    val_subjs = all_subjects[args.n_train:args.n_train + args.n_val]
    test_subjs = all_subjects[args.n_train + args.n_val:args.n_train + args.n_val + args.n_test]

    print(f"Split: {len(train_subjs)} train, {len(val_subjs)} val, {len(test_subjs)} test")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, subjects in [("train", train_subjs), ("valid", val_subjs), ("test", test_subjs)]:
        process_subjects(
            subjects, args.output_dir, split_name,
            mri_seq=args.mri_sequence,
            slice_range=args.slice_range,
            n_slices=args.slices_per_volume,
            seed=args.seed,
        )

    summary = {
        "dataset": "BraTS2023-GLI",
        "mri_sequence": args.mri_sequence,
        "slice_range": args.slice_range,
        "slices_per_volume": args.slices_per_volume,
        "n_subjects": {"train": len(train_subjs), "val": len(val_subjs), "test": len(test_subjs)},
        "hr_size": 256,
        "lr_size": 64,
        "seed": args.seed,
    }
    with open(args.output_dir / "dataset_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Dataset ready at {args.output_dir}")


if __name__ == "__main__":
    main()
