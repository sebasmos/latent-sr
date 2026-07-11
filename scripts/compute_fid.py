#!/usr/bin/env python3
"""Compute FID (Frechet Inception Distance) between SR and HR images.

FID measures distributional similarity between generated (SR) and real (HR)
images using Inception-v3 features. Lower FID = more similar distributions.

NOTE: FID is most reliable with >= 2048 images. MRNet has only 120 test
images, so those scores should be interpreted with caution.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_paths import repo_root, data_root

BASE = repo_root()

HR_DIRS = {
    "mrnet": data_root() / "mrnetkneemris/MRNet-v1.0-middle/valid/hr",
    "brats": data_root() / "brats2023-sr/test/hr",
    "cxr":   data_root() / "mimic-cxr-sr/test/hr",
}

SR_EXPERIMENTS = {
    "mrnet": [
        ("MedVAE-S1",   BASE / "outputs/experiments/mrnet_medvae_s1/sr_images"),
        ("SD-VAE",       BASE / "outputs/experiments/mrnet_sdvae/sr_images"),
        ("Bicubic",      BASE / "outputs/experiments/mrnet_bicubic/sr_images"),
        ("ESRGAN",       BASE / "outputs/experiments/mrnet_realesrgan/sr_images"),
        ("MedVAE-AE",   BASE / "outputs/experiments/mrnet_medvae_ae/sr_images"),
        ("SwinIR",       BASE / "outputs/experiments/mrnet_swinir/sr_images"),
    ],
    "brats": [
        ("MedVAE-S1",   BASE / "outputs/experiments/brats_medvae_s1/sr_images"),
        ("SD-VAE",       BASE / "outputs/experiments/brats_sdvae/sr_images"),
        ("Bicubic",      BASE / "outputs/experiments/brats_bicubic/sr_images"),
        ("ESRGAN",       BASE / "outputs/experiments/brats_realesrgan/sr_images"),
        ("MedVAE-AE",   BASE / "outputs/experiments/brats_medvae_ae/sr_images"),
        ("SwinIR",       BASE / "outputs/experiments/brats_swinir/sr_images"),
    ],
    "cxr": [
        ("MedVAE-S1",   BASE / "outputs/experiments/cxr_medvae_s1/sr_images"),
        ("SD-VAE",       BASE / "outputs/experiments/cxr_sdvae/sr_images"),
        ("Bicubic",      BASE / "outputs/experiments/cxr_bicubic/sr_images"),
        ("ESRGAN",       BASE / "outputs/experiments/cxr_realesrgan/sr_images"),
        ("MedVAE-AE",   BASE / "outputs/experiments/cxr_medvae_ae/sr_images"),
        ("SwinIR",       BASE / "outputs/experiments/cxr_swinir/sr_images"),
    ],
}

BATCH_SIZE = 64
NUM_WORKERS = 4
FID_FEATURE_DIM = 2048  # Inception pool3 features


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageFolderFlat(Dataset):
    """Load all PNGs from a flat directory, return uint8 [3, H, W] tensors."""

    def __init__(self, root: Path, file_list: Optional[list] = None):
        self.root = root
        if file_list is not None:
            self.files = sorted([root / f for f in file_list])
        else:
            self.files = sorted(root.glob("*.png"))
        if len(self.files) == 0:
            raise RuntimeError(f"No PNG files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")  # ensure grayscale
        t = torch.from_numpy(
            __import__("numpy").array(img, dtype="uint8")
        )  # [H, W]
        # Inception expects 3-channel uint8 images
        t = t.unsqueeze(0).expand(3, -1, -1).contiguous()  # [3, H, W]
        return t


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------
def compute_fid(hr_dir: Path, sr_dir: Path, device: torch.device) -> float:
    """Compute FID between HR and SR image directories.

    Only images present in both directories (matched by filename) are used.
    """
    # Find common filenames
    hr_files = {p.name for p in hr_dir.glob("*.png")}
    sr_files = {p.name for p in sr_dir.glob("*.png")}
    common = sorted(hr_files & sr_files)

    if len(common) == 0:
        print(f"  WARNING: No common filenames between {hr_dir} and {sr_dir}")
        return float("nan")

    print(f"  Using {len(common)} matched image pairs")

    # Create datasets with matched files only
    hr_dataset = ImageFolderFlat(hr_dir, file_list=common)
    sr_dataset = ImageFolderFlat(sr_dir, file_list=common)

    hr_loader = DataLoader(
        hr_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    sr_loader = DataLoader(
        sr_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    fid = FrechetInceptionDistance(feature=FID_FEATURE_DIM, normalize=False)
    fid = fid.to(device)

    # Update with real (HR) images
    for batch in hr_loader:
        batch = batch.to(device)
        fid.update(batch, real=True)

    # Update with fake (SR) images
    for batch in sr_loader:
        batch = batch.to(device)
        fid.update(batch, real=False)

    score = fid.compute().item()

    # Free memory
    del fid
    torch.cuda.empty_cache()

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}

    for dataset_name, hr_dir in HR_DIRS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"HR dir:  {hr_dir}")
        n_hr = len(list(hr_dir.glob("*.png")))
        print(f"HR images: {n_hr}")
        if n_hr < 2048:
            print(f"  WARNING: Only {n_hr} images — FID may be unreliable "
                  f"(recommend >= 2048)")
        print(f"{'='*60}")

        results[dataset_name] = {}

        experiments = SR_EXPERIMENTS.get(dataset_name, [])
        for method_name, sr_dir in experiments:
            print(f"\n  Method: {method_name}")
            print(f"  SR dir: {sr_dir}")

            if not sr_dir.exists():
                print(f"  SKIPPED — directory does not exist")
                results[dataset_name][method_name] = None
                continue

            n_sr = len(list(sr_dir.glob("*.png")))
            print(f"  SR images: {n_sr}")
            if n_sr == 0:
                print(f"  SKIPPED — no PNG files")
                results[dataset_name][method_name] = None
                continue

            try:
                score = compute_fid(hr_dir, sr_dir, device)
                results[dataset_name][method_name] = round(score, 4)
                print(f"  FID = {score:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                results[dataset_name][method_name] = None

    # Save results
    out_dir = BASE / "outputs/statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fid_scores.json"

    # Add metadata
    output = {
        "metric": "FID (Frechet Inception Distance)",
        "description": "Lower is better. Measures distributional similarity "
                       "between SR and HR images using Inception-v3 features.",
        "notes": {
            "mrnet": f"Only {len(list(HR_DIRS['mrnet'].glob('*.png')))} images — "
                     "FID unreliable (recommend >= 2048)",
            "grayscale": "Images expanded from 1-ch to 3-ch (repeat) for Inception",
            "feature_dim": FID_FEATURE_DIM,
        },
        "scores": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("FID SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<10} {'Method':<15} {'FID':>10}")
    print("-" * 40)
    for ds in results:
        for method, score in results[ds].items():
            s = f"{score:.2f}" if score is not None else "N/A"
            print(f"{ds:<10} {method:<15} {s:>10}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
