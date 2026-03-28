#!/usr/bin/env python3
"""
eval_sr_hr_diffmaps.py
Compute |SR - HR| and signed (SR - HR) difference maps.

Compares MedVAE SR, SD-VAE SR, and AE — all vs HR ground truth.
Key difference from eval_diffmaps.py: comparisons are made against HR,
not against AE. AE is included as a noise-floor baseline.

Datasets: mrnet (120 val), brats (462-700 val), cxr (1000 val)

Usage:
  python scripts/eval_sr_hr_diffmaps.py --dataset mrnet
  python scripts/eval_sr_hr_diffmaps.py --dataset brats
  python scripts/eval_sr_hr_diffmaps.py --dataset cxr

Inputs:
  MedVAE SR : outputs/experiments/{dataset}_medvae_s1/sr_images/
  SD-VAE SR : outputs/experiments/{dataset}_sdvae/sr_images/
  AE images : outputs/experiments/{dataset}_medvae_ae/sr_images/
  HR images : /orcd/pool/006/lceli_shared/DATASET/{dataset_path}/valid/hr/

Outputs (all under outputs/experiments/sr_hr_diffmaps_{dataset}/):
  heatmaps/       — per-method mean diff heatmap PNGs
  comparisons/    — 6-panel comparison figures for first 10 images
  results.json    — per-image and aggregate stats

No GPU required. Dependencies: numpy, PIL, matplotlib, json, argparse, pathlib.
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SR-HR absolute/signed difference maps")
parser.add_argument("--dataset", choices=["mrnet", "brats", "cxr"], default="mrnet",
                    help="Dataset to evaluate")
args = parser.parse_args()
DATASET = args.dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_BASE = "/orcd/pool/006/lceli_shared"

HR_DIRS = {
    "mrnet": pathlib.Path(f"{DATA_BASE}/DATASET/mrnetkneemris/MRNet-v1.0-middle/valid/hr"),
    "brats": pathlib.Path(f"{DATA_BASE}/DATASET/brats2023-sr/valid/hr"),
    "cxr":   pathlib.Path(f"{DATA_BASE}/DATASET/mimic-cxr-sr/valid/hr"),
}

# BraTS SR images are on test split; use _valid dirs for validation-split HR comparison
_suffix = "_valid" if DATASET == "brats" else ""
MEDVAE_SR_DIR = ROOT / f"outputs/experiments/{DATASET}_medvae_s1{_suffix}/sr_images"
SDVAE_SR_DIR  = ROOT / f"outputs/experiments/{DATASET}_sdvae{_suffix}/sr_images"
AE_DIR        = ROOT / f"outputs/experiments/{DATASET}_medvae_ae{_suffix}/sr_images"
HR_DIR        = HR_DIRS[DATASET]

OUT_DIR        = ROOT / f"outputs/experiments/sr_hr_diffmaps_{DATASET}"
HEATMAP_DIR    = OUT_DIR / "heatmaps"
COMPARISON_DIR = OUT_DIR / "comparisons"
JSON_PATH      = OUT_DIR / "results.json"

HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_LIMIT = 10  # save 6-panel figures for first N images

print(f"Dataset       : {DATASET}")
print(f"MedVAE SR dir : {MEDVAE_SR_DIR}")
print(f"SD-VAE SR dir : {SDVAE_SR_DIR}")
print(f"AE dir        : {AE_DIR}")
print(f"HR dir        : {HR_DIR}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_gray(path: pathlib.Path) -> np.ndarray:
    """Load a grayscale PNG as float32 array in [0, 1], shape (H, W)."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Build matched 4-tuples: (SR_medvae, SR_sdvae, AE, HR)
# Match by filename stem (all must share the same base name).
# ---------------------------------------------------------------------------
def build_tuples() -> list[tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]]:
    medvae_by_stem = {p.stem: p for p in sorted(MEDVAE_SR_DIR.glob("*.png"))}
    sdvae_by_stem  = {p.stem: p for p in sorted(SDVAE_SR_DIR.glob("*.png"))}
    ae_by_stem     = {p.stem: p for p in sorted(AE_DIR.glob("*.png"))}
    hr_by_stem     = {p.stem: p for p in sorted(HR_DIR.glob("*.png"))}

    common = sorted(set(medvae_by_stem) & set(sdvae_by_stem) & set(ae_by_stem) & set(hr_by_stem))
    if not common:
        raise RuntimeError(
            f"No matching filenames across all four directories.\n"
            f"  MedVAE SR: {MEDVAE_SR_DIR} ({len(medvae_by_stem)} files)\n"
            f"  SD-VAE SR: {SDVAE_SR_DIR} ({len(sdvae_by_stem)} files)\n"
            f"  AE:        {AE_DIR} ({len(ae_by_stem)} files)\n"
            f"  HR:        {HR_DIR} ({len(hr_by_stem)} files)"
        )
    return [(medvae_by_stem[s], sdvae_by_stem[s], ae_by_stem[s], hr_by_stem[s])
            for s in common]


tuples = build_tuples()
print(f"Matched 4-tuples: {len(tuples)}")

# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
all_abs_diff_medvae: list[np.ndarray] = []
all_abs_diff_sdvae:  list[np.ndarray] = []
all_abs_diff_ae:     list[np.ndarray] = []
per_image_stats: list[dict] = []

for idx, (medvae_path, sdvae_path, ae_path, hr_path) in enumerate(tuples):
    # Skip if any file missing
    for p in (medvae_path, sdvae_path, ae_path, hr_path):
        if not p.exists():
            print(f"  SKIP {idx}: missing {p}")
            break
    else:
        sr_m = load_gray(medvae_path)
        sr_s = load_gray(sdvae_path)
        ae   = load_gray(ae_path)
        hr   = load_gray(hr_path)

        # Sanity check shapes
        if not (sr_m.shape == sr_s.shape == ae.shape == hr.shape):
            print(f"  SKIP {idx}: shape mismatch — SR_m={sr_m.shape}, SR_s={sr_s.shape}, "
                  f"AE={ae.shape}, HR={hr.shape}")
            continue

        # Compute difference maps
        abs_diff_m  = np.abs(sr_m - hr)
        abs_diff_s  = np.abs(sr_s - hr)
        abs_diff_ae = np.abs(ae - hr)
        signed_m    = sr_m - hr

        all_abs_diff_medvae.append(abs_diff_m)
        all_abs_diff_sdvae.append(abs_diff_s)
        all_abs_diff_ae.append(abs_diff_ae)

        # Per-image stats
        per_image_stats.append({
            "index": idx,
            "filename": hr_path.name,
            "medvae_sr": {
                "abs_diff_mean": float(abs_diff_m.mean()),
                "abs_diff_std":  float(abs_diff_m.std()),
                "abs_diff_max":  float(abs_diff_m.max()),
            },
            "sdvae_sr": {
                "abs_diff_mean": float(abs_diff_s.mean()),
                "abs_diff_std":  float(abs_diff_s.std()),
                "abs_diff_max":  float(abs_diff_s.max()),
            },
            "ae": {
                "abs_diff_mean": float(abs_diff_ae.mean()),
                "abs_diff_std":  float(abs_diff_ae.std()),
                "abs_diff_max":  float(abs_diff_ae.max()),
            },
        })

        # --- 6-panel comparison for first COMPARISON_LIMIT images ---
        if idx < COMPARISON_LIMIT:
            # LR proxy: we use AE as stand-in for LR appearance in pixel space
            # since LR is not directly stored here; panels: AE | SR_medvae | SR_sdvae | AE | HR | diff_medvae
            fig, axes = plt.subplots(1, 6, figsize=(24, 4), dpi=100)
            panels = [
                (ae,          "AE (LR approx)", "gray", 1.0),
                (sr_m,        "SR MedVAE",      "gray", 1.0),
                (sr_s,        "SR SD-VAE",       "gray", 1.0),
                (hr,          "HR (ground truth)","gray", 1.0),
                (abs_diff_m,  "|SR_MedVAE - HR|", "hot", max(abs_diff_m.max(), 1e-6)),
                (abs_diff_s,  "|SR_SDVAE - HR|",  "hot", max(abs_diff_s.max(), 1e-6)),
            ]
            for ax, (arr, title, cmap, vmax) in zip(axes, panels):
                im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(f"{hr_path.stem}  (idx={idx})", fontsize=9)
            fig.tight_layout()
            fig.savefig(COMPARISON_DIR / f"{hr_path.stem}_comparison.png", bbox_inches="tight")
            plt.close(fig)

        if (idx + 1) % 50 == 0:
            print(f"  processed {idx + 1}/{len(tuples)} ...")
        continue

n_processed = len(all_abs_diff_medvae)
print(f"\nProcessed {n_processed} image tuples.")

if n_processed == 0:
    raise RuntimeError("No images were processed. Check input directories.")

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
def flat_stats(diffs: list[np.ndarray]) -> dict:
    all_flat = np.concatenate([d.ravel() for d in diffs])
    return {
        "mean": float(all_flat.mean()),
        "std":  float(all_flat.std()),
        "max":  float(all_flat.max()),
    }

agg_stats = {
    "medvae_sr": flat_stats(all_abs_diff_medvae),
    "sdvae_sr":  flat_stats(all_abs_diff_sdvae),
    "ae":        flat_stats(all_abs_diff_ae),
}

print("\n--- Aggregate |SR - HR| stats ---")
for method, s in agg_stats.items():
    print(f"  {method:<12}: mean={s['mean']:.6f}  std={s['std']:.6f}  max={s['max']:.6f}")

# ---------------------------------------------------------------------------
# Mean difference maps across all images
# ---------------------------------------------------------------------------
mean_diff_medvae = np.mean(np.stack(all_abs_diff_medvae, axis=0), axis=0)
mean_diff_sdvae  = np.mean(np.stack(all_abs_diff_sdvae,  axis=0), axis=0)
mean_diff_ae     = np.mean(np.stack(all_abs_diff_ae,     axis=0), axis=0)

# Save individual heatmaps
for method_name, mean_map in [
    ("medvae_sr", mean_diff_medvae),
    ("sdvae_sr",  mean_diff_sdvae),
    ("ae",        mean_diff_ae),
]:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    im = ax.imshow(mean_map, cmap="hot", vmin=0, vmax=mean_map.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Mean |{method_name} - HR|  ({DATASET.upper()}, N={n_processed})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / f"mean_diffmap_{method_name}.png", bbox_inches="tight")
    plt.close(fig)

print(f"Per-method heatmaps saved to {HEATMAP_DIR}")

# --- Summary figure: 3-panel mean absolute difference maps side by side ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
vmax_global = max(mean_diff_medvae.max(), mean_diff_sdvae.max(), mean_diff_ae.max())

method_panels = [
    (mean_diff_medvae, "Mean |MedVAE SR - HR|"),
    (mean_diff_sdvae,  "Mean |SD-VAE SR - HR|"),
    (mean_diff_ae,     "Mean |AE - HR| (noise floor)"),
]
for ax, (mean_map, title) in zip(axes, method_panels):
    im = ax.imshow(mean_map, cmap="hot", vmin=0, vmax=vmax_global)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f"Mean Absolute Difference vs HR — {DATASET.upper()}  (N={n_processed})", fontsize=12)
fig.tight_layout()
summary_fig_path = OUT_DIR / "summary_mean_diffmaps.png"
fig.savefig(str(summary_fig_path), bbox_inches="tight")
plt.close(fig)
print(f"Summary figure saved to {summary_fig_path}")

# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
results = {
    "dataset": DATASET,
    "n_processed": n_processed,
    "aggregate_stats": agg_stats,
    "per_image": per_image_stats,
}

with open(JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {JSON_PATH}")
print("\nDone.")
