#!/usr/bin/env python3
"""
Compute bicubic-interpolation baseline PSNR and MS-SSIM for each dataset.

For each dataset:
  1. Load every LR image.
  2. Resize to 256x256 with PIL bicubic interpolation (identity for MRNet
     whose LR is already 256x256).
  3. Compute PSNR and MS-SSIM against the HR ground truth.

Also computes per-image PSNR/MS-SSIM for downstream statistical testing, and
runs paired Wilcoxon signed-rank tests comparing MedVAE-S1 vs SD-VAE SR
per-image PSNR (computed from saved SR PNG images).

Results are saved to:
  outputs/bicubic_baseline/results.json
  outputs/statistical_tests/pvalues.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from tqdm import tqdm

from repro_paths import dataset_hr_dir, dataset_lr_dir, outputs_root, repo_root

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = {
    "MRNet": {
        "hr": dataset_hr_dir("mrnet", "valid"),
        "lr": dataset_lr_dir("mrnet", "valid"),
        "lr_size": 256,  # already 256x256 -> bicubic is identity
    },
    "BraTS": {
        "hr": dataset_hr_dir("brats", "test"),
        "lr": dataset_lr_dir("brats", "test"),
        "lr_size": 64,
    },
    "CXR": {
        "hr": dataset_hr_dir("cxr", "test"),
        "lr": dataset_lr_dir("cxr", "test"),
        "lr_size": 64,
    },
}

EXPERIMENTS = {
    "MRNet": {
        "medvae_s1": "outputs/experiments/mrnet_medvae_s1/sr_images",
        "sdvae": "outputs/experiments/mrnet_sdvae/sr_images",
    },
    "BraTS": {
        "medvae_s1": "outputs/experiments/brats_medvae_s1/sr_images",
        "sdvae": "outputs/experiments/brats_sdvae/sr_images",
    },
}

ROOT = Path(__file__).resolve().parents[1]
ROOT = repo_root()
TARGET_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_gray_tensor(path: str | Path) -> torch.Tensor:
    """Load a grayscale PNG as a [1,1,H,W] float32 tensor in [0,1]."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def bicubic_upscale(path: str | Path, target: int = 256) -> torch.Tensor:
    """Load a grayscale PNG, bicubic-resize to target, return [1,1,H,W]."""
    img = Image.open(path).convert("L")
    if img.size != (target, target):
        img = img.resize((target, target), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Task 1 — Bicubic baseline PSNR / MS-SSIM
# ---------------------------------------------------------------------------

def compute_bicubic_baseline() -> dict:
    """Return {dataset: {psnr_mean, psnr_std, msssim_mean, msssim_std, per_image}}."""
    results = {}

    for name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Bicubic baseline  —  {name}")
        print(f"{'='*60}")

        hr_dir = cfg["hr"]
        lr_dir = cfg["lr"]
        fnames = sorted(os.listdir(hr_dir))
        assert len(fnames) == len(os.listdir(lr_dir)), (
            f"HR/LR file count mismatch for {name}"
        )

        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(
            DEVICE
        )

        per_image = []
        psnr_vals = []
        msssim_vals = []

        for fname in tqdm(fnames, desc=name):
            hr_t = load_gray_tensor(os.path.join(hr_dir, fname)).to(DEVICE)
            bic_t = bicubic_upscale(os.path.join(lr_dir, fname), TARGET_SIZE).to(
                DEVICE
            )

            p = psnr_fn(bic_t, hr_t).item()
            m = msssim_fn(bic_t, hr_t).item()

            psnr_vals.append(p)
            msssim_vals.append(m)
            per_image.append({"filename": fname, "psnr": p, "msssim": m})

        psnr_arr = np.array(psnr_vals)
        msssim_arr = np.array(msssim_vals)

        results[name] = {
            "n_samples": len(fnames),
            "psnr_mean": float(psnr_arr.mean()),
            "psnr_std": float(psnr_arr.std()),
            "msssim_mean": float(msssim_arr.mean()),
            "msssim_std": float(msssim_arr.std()),
            "per_image": per_image,
        }

        print(f"  PSNR  = {psnr_arr.mean():.4f} +/- {psnr_arr.std():.4f}")
        print(f"  MS-SSIM = {msssim_arr.mean():.6f} +/- {msssim_arr.std():.6f}")

    return results


# ---------------------------------------------------------------------------
# Task 2 — Paired Wilcoxon signed-rank test on per-image PSNR
# ---------------------------------------------------------------------------

def compute_per_image_psnr(sr_dir: str, hr_dir: str) -> dict[str, float]:
    """Compute per-image PSNR for SR images against HR, return {fname: psnr}."""
    sr_dir = Path(sr_dir)
    hr_dir = Path(hr_dir)
    fnames = sorted(os.listdir(sr_dir))
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    result = {}
    for fname in tqdm(fnames, desc=f"PSNR from {sr_dir.parent.name}"):
        sr_t = load_gray_tensor(sr_dir / fname).to(DEVICE)
        hr_t = load_gray_tensor(hr_dir / fname).to(DEVICE)
        result[fname] = psnr_fn(sr_t, hr_t).item()
    return result


def compute_pvalues() -> dict:
    """Run paired Wilcoxon signed-rank tests between MedVAE-S1 and SD-VAE."""
    results = {}

    for dset, exps in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"  Wilcoxon test  —  {dset}")
        print(f"{'='*60}")

        medvae_sr_dir = ROOT / exps["medvae_s1"]
        sdvae_sr_dir = ROOT / exps["sdvae"]

        if not medvae_sr_dir.exists() or not sdvae_sr_dir.exists():
            print(f"  SKIP — missing SR images for {dset}")
            continue

        # Determine corresponding HR directory
        hr_dir = DATASETS[dset]["hr"]

        psnr_medvae = compute_per_image_psnr(str(medvae_sr_dir), hr_dir)
        psnr_sdvae = compute_per_image_psnr(str(sdvae_sr_dir), hr_dir)

        # Align by filename (intersection)
        common = sorted(set(psnr_medvae) & set(psnr_sdvae))
        assert len(common) > 0, f"No common filenames for {dset}"

        a = np.array([psnr_medvae[f] for f in common])
        b = np.array([psnr_sdvae[f] for f in common])

        stat, pval = stats.wilcoxon(a, b, alternative="two-sided")

        results[dset] = {
            "n_paired": len(common),
            "medvae_s1_psnr_mean": float(a.mean()),
            "medvae_s1_psnr_std": float(a.std()),
            "sdvae_psnr_mean": float(b.mean()),
            "sdvae_psnr_std": float(b.std()),
            "wilcoxon_statistic": float(stat),
            "p_value": float(pval),
            "significant_at_005": bool(pval < 0.05),
        }

        print(f"  MedVAE-S1 PSNR = {a.mean():.4f} +/- {a.std():.4f}")
        print(f"  SD-VAE    PSNR = {b.mean():.4f} +/- {b.std():.4f}")
        print(f"  Wilcoxon stat  = {stat:.2f}")
        print(f"  p-value        = {pval:.2e}")
        print(f"  Significant at 0.05: {pval < 0.05}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Task 1 — Bicubic baseline
    bicubic_results = compute_bicubic_baseline()

    # Save (strip per_image for the summary file; keep a full copy)
    out_dir = ROOT / "outputs" / "bicubic_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary (without per-image for readability)
    summary = {}
    for dset, vals in bicubic_results.items():
        summary[dset] = {k: v for k, v in vals.items() if k != "per_image"}

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBicubic baseline results saved to {results_path}")

    # Full results including per-image
    full_path = out_dir / "results_full.json"
    with open(full_path, "w") as f:
        json.dump(bicubic_results, f, indent=2)
    print(f"Full per-image results saved to {full_path}")

    # Task 2 — Wilcoxon p-values
    pvalue_results = compute_pvalues()

    pval_dir = ROOT / "outputs" / "statistical_tests"
    pval_dir.mkdir(parents=True, exist_ok=True)
    pval_path = pval_dir / "pvalues.json"
    with open(pval_path, "w") as f:
        json.dump(pvalue_results, f, indent=2)
    print(f"\nWilcoxon p-value results saved to {pval_path}")


if __name__ == "__main__":
    main()
