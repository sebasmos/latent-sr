#!/usr/bin/env python3
"""
Compute bicubic SR baselines at multiple scale factors (2x, 4x, 8x).

For each scale factor and each dataset (BraTS, CXR):
  - Downsample HR (256x256) to LR (256/scale x 256/scale) using PIL BICUBIC
  - Upsample LR back to 256x256 using PIL BICUBIC
  - For 4x BraTS/CXR: use existing sr_images if available
  - Compute PSNR and SSIM vs HR

Results saved to:
  outputs/experiments/multiscale_analysis/results.json

Figure saved to:
  outputs/figures/multiscale_psnr_curve.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from tqdm import tqdm

from repro_paths import dataset_hr_dir, outputs_root, repo_root

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = repo_root()

DATASETS = {
    "BraTS": {
        "hr": dataset_hr_dir("brats", "test"),
        "existing_4x": str(outputs_root() / "experiments/brats_bicubic/sr_images"),
    },
    "CXR": {
        "hr": dataset_hr_dir("cxr", "test"),
        "existing_4x": str(outputs_root() / "experiments/cxr_bicubic/sr_images"),
    },
}

SCALE_FACTORS = [2, 4, 8]
TARGET_SIZE = 256

# Reference diffusion SR PSNR values for annotation
DIFFUSION_REF = {
    "BraTS": 27.1,
    "CXR": 28.9,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_gray_array(path: str | Path) -> np.ndarray:
    """Load a grayscale PNG as a float32 numpy array in [0, 1], shape (H, W)."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def bicubic_downsample_upsample(hr_arr: np.ndarray, scale: int, target: int = 256) -> np.ndarray:
    """
    Downsample hr_arr to (target/scale x target/scale) then upsample back to
    (target x target) using PIL BICUBIC.  Returns float32 in [0,1].
    """
    lr_size = target // scale
    hr_img = Image.fromarray((hr_arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)
    sr_img = lr_img.resize((target, target), Image.BICUBIC)
    return np.array(sr_img, dtype=np.float32) / 255.0


def compute_psnr_ssim(pred: np.ndarray, target: np.ndarray):
    """Compute PSNR and SSIM between two float32 arrays in [0,1]."""
    psnr_val = ski_psnr(target, pred, data_range=1.0)
    ssim_val = ski_ssim(target, pred, data_range=1.0)
    return float(psnr_val), float(ssim_val)


def load_existing_sr(sr_path: str | Path) -> np.ndarray | None:
    """Load an existing SR image as float32 array. Returns None if not found."""
    p = Path(sr_path)
    if p.exists():
        return load_gray_array(p)
    return None


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run_multiscale_analysis() -> dict:
    """
    Run bicubic SR evaluation at scales [2, 4, 8] for BraTS and CXR.
    Returns nested dict: results[dataset][scale] = {psnr_mean, psnr_std, ssim_mean, ssim_std, n}
    """
    results = {}

    for dset_name, cfg in DATASETS.items():
        hr_dir = Path(cfg["hr"])
        existing_4x_dir = Path(cfg["existing_4x"])

        hr_files = sorted(hr_dir.iterdir())
        print(f"\n{'='*60}")
        print(f"  Dataset: {dset_name}  ({len(hr_files)} images)")
        print(f"{'='*60}")

        results[dset_name] = {}

        for scale in SCALE_FACTORS:
            print(f"\n  Scale {scale}x ...")

            # For 4x, try existing sr_images first
            use_existing = (scale == 4) and existing_4x_dir.exists()
            if use_existing:
                existing_files = sorted(existing_4x_dir.iterdir())
                n_existing = len(existing_files)
                print(f"    Using existing 4x SR images: {existing_4x_dir} ({n_existing} files)")

            psnr_vals = []
            ssim_vals = []

            for hr_file in tqdm(hr_files, desc=f"{dset_name} {scale}x"):
                hr_arr = load_gray_array(hr_file)

                if use_existing:
                    # Try to find the matching SR file by filename
                    sr_candidate = existing_4x_dir / hr_file.name
                    sr_arr = load_existing_sr(sr_candidate)
                    if sr_arr is None:
                        # Fall back to computing bicubic
                        sr_arr = bicubic_downsample_upsample(hr_arr, scale, TARGET_SIZE)
                else:
                    sr_arr = bicubic_downsample_upsample(hr_arr, scale, TARGET_SIZE)

                psnr_val, ssim_val = compute_psnr_ssim(sr_arr, hr_arr)
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)

            psnr_arr = np.array(psnr_vals)
            ssim_arr = np.array(ssim_vals)

            results[dset_name][scale] = {
                "psnr_mean": float(psnr_arr.mean()),
                "psnr_std": float(psnr_arr.std()),
                "ssim_mean": float(ssim_arr.mean()),
                "ssim_std": float(ssim_arr.std()),
                "n": len(psnr_vals),
                "used_existing_4x": use_existing,
            }

            print(f"    PSNR  = {psnr_arr.mean():.4f} ± {psnr_arr.std():.4f}")
            print(f"    SSIM  = {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: dict):
    header = f"{'Dataset':<10} | {'Scale':>5} | {'PSNR (mean±std)':>18} | {'SSIM (mean±std)':>18}"
    print("\n" + "="*len(header))
    print(header)
    print("-"*len(header))
    for dset in ["BraTS", "CXR"]:
        for scale in SCALE_FACTORS:
            r = results[dset][scale]
            psnr_str = f"{r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}"
            ssim_str = f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}"
            print(f"{dset:<10} | {str(scale)+'x':>5} | {psnr_str:>18} | {ssim_str:>18}")
    print("="*len(header))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_psnr_curve(results: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"BraTS": "royalblue", "CXR": "darkorange"}
    markers = {"BraTS": "o", "CXR": "s"}

    for dset, color in colors.items():
        scales = SCALE_FACTORS
        psnr_means = [results[dset][s]["psnr_mean"] for s in scales]
        psnr_stds = [results[dset][s]["psnr_std"] for s in scales]

        ax.errorbar(
            scales, psnr_means, yerr=psnr_stds,
            label=f"{dset} bicubic",
            color=color,
            marker=markers[dset],
            linewidth=2,
            markersize=7,
            capsize=4,
        )

    # Reference diffusion SR lines at 4x
    ref_colors = {"BraTS": "royalblue", "CXR": "darkorange"}
    for dset, ref_psnr in DIFFUSION_REF.items():
        ax.axhline(
            y=ref_psnr,
            color=ref_colors[dset],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    # Single annotation for diffusion reference
    ax.annotate(
        "Diffusion SR (4×): MedVAE 27.1 dB / BraTS, 28.9 dB / CXR",
        xy=(4, 27.1),
        xytext=(5.5, 26.2),
        fontsize=8.5,
        color="dimgray",
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.0),
    )

    ax.set_xlabel("Scale Factor", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Bicubic SR Baseline: PSNR vs Scale Factor", fontsize=13)
    ax.set_xticks(SCALE_FACTORS)
    ax.set_xticklabels([f"{s}×" for s in SCALE_FACTORS])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = run_multiscale_analysis()

    print_summary_table(results)

    # Save results JSON
    out_dir = ROOT / "outputs" / "experiments" / "multiscale_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Create figure
    fig_path = ROOT / "outputs" / "figures" / "multiscale_psnr_curve.png"
    create_psnr_curve(results, fig_path)


if __name__ == "__main__":
    main()
