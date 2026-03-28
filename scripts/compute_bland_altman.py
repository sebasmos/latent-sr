#!/usr/bin/env python3
"""
Bland-Altman analysis: SR vs HR intensity agreement.

For each image pair (SR, HR):
  D = mean(SR) - mean(HR)   [per-image mean intensity, normalized to [0,1]]
  M = (mean(SR) + mean(HR)) / 2

Plots D vs M with mean bias and 95% LoA (mean ± 1.96 * std(D)).

Methods compared per dataset:
  - MedVAE SR  (blue)
  - SD-VAE SR  (red)
  - Bicubic    (green)   [MRNet only]

Usage:
  python compute_bland_altman.py [--dataset {mrnet,brats,cxr,all}]

Outputs:
  outputs/figures/bland_altman.png       (MRNet only, 3-panel)
  outputs/figures/bland_altman_all.png   (all 3 datasets, 3-panel)
"""

import argparse
import os
import pathlib
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BASE_SR = ROOT / "outputs/experiments"
BASE_HR = Path("/orcd/pool/006/lceli_shared/DATASET")

DATASET_CONFIGS = {
    "mrnet": {
        "title": "MRNet",
        "hr_dir": BASE_HR / "mrnetkneemris/MRNet-v1.0-middle/valid/hr",
        "methods": {
            "MedVAE SR": {
                "sr_dir": BASE_SR / "mrnet_medvae_s1/sr_images",
                "color": "#1f77b4",
            },
            "SD-VAE SR": {
                "sr_dir": BASE_SR / "mrnet_sdvae/sr_images",
                "color": "#d62728",
            },
            "Bicubic": {
                "sr_dir": BASE_SR / "mrnet_bicubic/sr_images",
                "color": "#2ca02c",
            },
        },
    },
    "brats": {
        "title": "BraTS",
        "hr_dir": BASE_HR / "brats2023-sr/valid/hr",
        "methods": {
            "MedVAE SR": {
                "sr_dir": BASE_SR / "brats_medvae_s1_valid/sr_images",
                "color": "#1f77b4",
            },
            "SD-VAE SR": {
                "sr_dir": BASE_SR / "brats_sdvae_valid/sr_images",
                "color": "#d62728",
            },
        },
    },
    "cxr": {
        "title": "CXR",
        "hr_dir": BASE_HR / "mimic-cxr-sr/valid/hr",
        "methods": {
            "MedVAE SR": {
                "sr_dir": BASE_SR / "cxr_medvae_s1/sr_images",
                "color": "#1f77b4",
            },
            "SD-VAE SR": {
                "sr_dir": BASE_SR / "cxr_sdvae/sr_images",
                "color": "#d62728",
            },
        },
    },
}

OUT_DIR = ROOT / "outputs/figures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_mean(path: Path) -> float:
    """Load a PNG image, normalize to [0,1], and return the mean pixel value."""
    img = Image.open(path).convert("L")          # grayscale
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return float(arr.mean())


def match_pairs(sr_dir: Path, hr_dir: Path):
    """
    Match SR and HR images by filename stem.
    Returns sorted list of (sr_path, hr_path) tuples.
    """
    sr_dir = Path(sr_dir)
    hr_dir = Path(hr_dir)

    sr_files = {p.stem: p for p in sr_dir.glob("*.png")}
    hr_files = {p.stem: p for p in hr_dir.glob("*.png")}

    common = sorted(set(sr_files.keys()) & set(hr_files.keys()))
    if not common:
        raise RuntimeError(
            f"No matching stems found between\n  {sr_dir}\n  {hr_dir}\n"
            f"SR stems sample: {list(sr_files.keys())[:3]}\n"
            f"HR stems sample: {list(hr_files.keys())[:3]}"
        )
    pairs = [(sr_files[s], hr_files[s]) for s in common]
    return pairs


def compute_bland_altman_stats(pairs):
    """
    Returns arrays D and M, and summary stats dict.
    """
    D_list = []
    M_list = []
    for sr_path, hr_path in pairs:
        sr_mean = load_image_mean(sr_path)
        hr_mean = load_image_mean(hr_path)
        D_list.append(sr_mean - hr_mean)
        M_list.append((sr_mean + hr_mean) / 2.0)

    D = np.array(D_list)
    M = np.array(M_list)

    bias = float(D.mean())
    std_d = float(D.std(ddof=1))
    loa_lower = bias - 1.96 * std_d
    loa_upper = bias + 1.96 * std_d

    return D, M, {
        "n": len(D),
        "bias": bias,
        "std": std_d,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_subplot(ax, method_data_list, dataset_title):
    """
    Draw a Bland-Altman subplot with multiple methods overlaid.

    method_data_list: list of (method_name, D, M, stats, color)
    """
    # Collect all D/M values for axis limits
    all_M = np.concatenate([item[2] for item in method_data_list])
    all_D = np.concatenate([item[1] for item in method_data_list])

    # Reference zero line
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)

    for method_name, D, M, stats, color in method_data_list:
        bias      = stats["bias"]
        loa_upper = stats["loa_upper"]
        loa_lower = stats["loa_lower"]
        n         = stats["n"]

        ax.scatter(M, D, s=10, alpha=0.45, color=color, edgecolors="none",
                   rasterized=True, zorder=2, label=f"{method_name} (n={n})")

        # Bias and LoA lines
        ax.axhline(bias, color=color, linewidth=1.8, linestyle="-", zorder=3,
                   label=f"{method_name} bias={bias:+.4f}")
        ax.axhline(loa_upper, color=color, linewidth=1.1, linestyle="--", zorder=3,
                   label=f"{method_name} +1.96SD={loa_upper:+.4f}")
        ax.axhline(loa_lower, color=color, linewidth=1.1, linestyle="--", zorder=3,
                   label=f"{method_name} −1.96SD={loa_lower:+.4f}")

        # Subtle shading for LoA band
        ax.fill_between(
            [all_M.min() - 0.01, all_M.max() + 0.01],
            loa_lower, loa_upper,
            alpha=0.06, color=color, zorder=0
        )

    ax.set_title(dataset_title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Mean intensity  (SR + HR) / 2", fontsize=9)
    ax.set_ylabel("Difference  SR − HR", fontsize=9)
    ax.tick_params(labelsize=8)

    # Compact legend: use two columns when there are many entries
    n_legend = sum(3 + 1 for _ in method_data_list)  # bias + 2 LoA + scatter per method
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85,
              ncol=1 if len(method_data_list) <= 2 else 2)

    y_pad = 0.05
    ax.set_ylim(all_D.min() - y_pad, all_D.max() + y_pad)


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(dataset_key: str):
    """Process one dataset, return (method_data_list, summary_rows)."""
    cfg = DATASET_CONFIGS[dataset_key]
    hr_dir = cfg["hr_dir"]
    method_data_list = []
    summary_rows = []

    for method_name, minfo in cfg["methods"].items():
        sr_dir = minfo["sr_dir"]
        color  = minfo["color"]
        label  = f"{cfg['title']} {method_name}"
        print(f"\n  [{label}]")
        try:
            pairs = match_pairs(sr_dir, hr_dir)
        except RuntimeError as e:
            print(f"  WARNING: {e}\n  Skipping {label}.")
            continue
        print(f"    Matched {len(pairs)} image pairs")
        D, M, stats = compute_bland_altman_stats(pairs)
        print(f"    N={stats['n']}  Bias={stats['bias']:+.4f}  "
              f"Std={stats['std']:.4f}  "
              f"LoA=[{stats['loa_lower']:+.4f}, {stats['loa_upper']:+.4f}]")
        method_data_list.append((method_name, D, M, stats, color))
        summary_rows.append((label, stats))

    return method_data_list, summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bland-Altman analysis for SR vs HR intensity agreement."
    )
    parser.add_argument(
        "--dataset",
        choices=["mrnet", "brats", "cxr", "all"],
        default="all",
        help="Dataset to analyse (default: all — produces 3-panel figure).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        datasets = ["mrnet", "brats", "cxr"]
        out_fig  = OUT_DIR / "bland_altman_all.png"
        suptitle = (
            "Bland–Altman Analysis: SR vs HR Mean Intensity Agreement\n"
            "MRNet / BraTS / CXR validation sets"
        )
    elif args.dataset == "mrnet":
        datasets = ["mrnet"]
        out_fig  = OUT_DIR / "bland_altman.png"
        suptitle = (
            "Bland–Altman Analysis: SR vs HR Mean Intensity Agreement\n"
            "(MRNet validation set)"
        )
    else:
        datasets = [args.dataset]
        out_fig  = OUT_DIR / f"bland_altman_{args.dataset}.png"
        suptitle = (
            f"Bland–Altman Analysis: SR vs HR Mean Intensity Agreement\n"
            f"({DATASET_CONFIGS[args.dataset]['title']} validation set)"
        )

    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    all_summary = []
    for ax, dataset_key in zip(axes, datasets):
        cfg = DATASET_CONFIGS[dataset_key]
        print(f"\n=== Dataset: {cfg['title']} ===")
        method_data_list, summary_rows = process_dataset(dataset_key)
        if not method_data_list:
            print(f"  No data available for {cfg['title']}, skipping panel.")
            ax.set_visible(False)
            continue
        make_subplot(ax, method_data_list, cfg["title"])
        all_summary.extend(summary_rows)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {out_fig}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Method':<28} {'N':>5} {'Mean Bias':>12} {'LoA Lower':>12} "
          f"{'LoA Upper':>12} {'Std(D)':>10}")
    print("-" * 80)
    for method_name, stats in all_summary:
        print(
            f"{method_name:<28} {stats['n']:>5} "
            f"{stats['bias']:>+12.5f} "
            f"{stats['loa_lower']:>+12.5f} "
            f"{stats['loa_upper']:>+12.5f} "
            f"{stats['std']:>10.5f}"
        )
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("\nInterpretation:")
    for method_name, stats in all_summary:
        b = stats["bias"]
        loa_w = stats["loa_upper"] - stats["loa_lower"]
        bias_flag = "near-zero" if abs(b) < 0.01 else ("small" if abs(b) < 0.05 else "LARGE")
        print(f"  {method_name}: bias={b:+.4f} ({bias_flag}), LoA width={loa_w:.4f}")


if __name__ == "__main__":
    main()
