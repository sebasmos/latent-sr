#!/usr/bin/env python3
"""
eval_hallucination_quantification.py
Quantify hallucinated content (diffusion adds fake detail) and lost content
(diffusion removes real detail) relative to HR ground truth.

Method:
  - Compute per-image noise floor from |AE - HR|
  - threshold = noise_mean + 2 * noise_std
  - Signed diff D = SR_medvae - HR
  - Hallucination mask: D > threshold  (diffusion added content not in HR)
  - Loss mask:          D < -threshold (diffusion removed content present in HR)
  - Same analysis repeated for SD-VAE SR

Datasets: mrnet (120 val), brats (462-700 val), cxr (1000 val)

Usage:
  python scripts/eval_hallucination_quantification.py --dataset mrnet
  python scripts/eval_hallucination_quantification.py --dataset brats
  python scripts/eval_hallucination_quantification.py --dataset cxr

Inputs:
  MedVAE SR : outputs/experiments/{dataset}_medvae_s1/sr_images/
  SD-VAE SR : outputs/experiments/{dataset}_sdvae/sr_images/
  AE images : outputs/experiments/{dataset}_medvae_ae/sr_images/
  HR images : /orcd/pool/006/lceli_shared/DATASET/{dataset_path}/valid/hr/

Outputs:
  outputs/experiments/hallucination_{dataset}/results.json
  outputs/experiments/hallucination_{dataset}/comparisons/  (first 10 images)
  outputs/experiments/hallucination_{dataset}/mean_hallu_masks.png
  outputs/experiments/hallucination_{dataset}/summary_bar_chart.png

No GPU required. Dependencies: numpy, matplotlib, PIL, json, argparse, pathlib.
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
parser = argparse.ArgumentParser(description="Hallucination quantification: SR vs HR")
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

OUT_DIR        = ROOT / f"outputs/experiments/hallucination_{DATASET}"
COMPARISON_DIR = OUT_DIR / "comparisons"
JSON_PATH      = OUT_DIR / "results.json"

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_LIMIT = 10

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


def compute_hallucination_metrics(
    sr: np.ndarray,
    hr: np.ndarray,
    threshold: float,
) -> dict:
    """
    Given signed diff D = SR - HR and a threshold derived from the AE noise floor,
    compute hallucination and loss masks and their statistics.

    Returns dict with:
      hallu_rate       : fraction of pixels where D > threshold
      loss_rate        : fraction of pixels where D < -threshold
      hallu_magnitude  : mean D value in hallucination region (or 0 if empty)
      loss_magnitude   : mean |D| in loss region (or 0 if empty)
      threshold        : the threshold used
    """
    D = sr - hr
    hallu_mask = D > threshold
    loss_mask  = D < -threshold

    hallu_rate = float(hallu_mask.mean())
    loss_rate  = float(loss_mask.mean())

    hallu_mag = float(D[hallu_mask].mean()) if hallu_mask.any() else 0.0
    loss_mag  = float(np.abs(D[loss_mask]).mean()) if loss_mask.any() else 0.0

    return {
        "hallu_rate":      hallu_rate,
        "loss_rate":       loss_rate,
        "hallu_magnitude": hallu_mag,
        "loss_magnitude":  loss_mag,
        "threshold":       float(threshold),
        "hallu_mask":      hallu_mask,
        "loss_mask":       loss_mask,
        "signed_diff":     D,
    }


# ---------------------------------------------------------------------------
# Build matched 4-tuples: (SR_medvae, SR_sdvae, AE, HR) by filename stem
# ---------------------------------------------------------------------------
def build_tuples():
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
# Per-image analysis
# ---------------------------------------------------------------------------
per_image_stats: list[dict] = []
hallu_masks_medvae: list[np.ndarray] = []
hallu_masks_sdvae:  list[np.ndarray] = []
loss_masks_medvae:  list[np.ndarray] = []
loss_masks_sdvae:   list[np.ndarray] = []

for idx, (medvae_path, sdvae_path, ae_path, hr_path) in enumerate(tuples):
    # Skip if any file missing
    missing = False
    for p in (medvae_path, sdvae_path, ae_path, hr_path):
        if not p.exists():
            print(f"  SKIP {idx}: missing {p}")
            missing = True
            break
    if missing:
        continue

    sr_m = load_gray(medvae_path)
    sr_s = load_gray(sdvae_path)
    ae   = load_gray(ae_path)
    hr   = load_gray(hr_path)

    if not (sr_m.shape == sr_s.shape == ae.shape == hr.shape):
        print(f"  SKIP {idx}: shape mismatch")
        continue

    # Compute per-image noise floor from |AE - HR|
    ae_diff = np.abs(ae - hr)
    noise_mean = float(ae_diff.mean())
    noise_std  = float(ae_diff.std())
    threshold  = noise_mean + 2.0 * noise_std

    # Hallucination/loss metrics for MedVAE SR
    metrics_m = compute_hallucination_metrics(sr_m, hr, threshold)
    # Hallucination/loss metrics for SD-VAE SR
    metrics_s = compute_hallucination_metrics(sr_s, hr, threshold)

    hallu_masks_medvae.append(metrics_m["hallu_mask"].astype(np.float32))
    hallu_masks_sdvae.append(metrics_s["hallu_mask"].astype(np.float32))
    loss_masks_medvae.append(metrics_m["loss_mask"].astype(np.float32))
    loss_masks_sdvae.append(metrics_s["loss_mask"].astype(np.float32))

    per_image_stats.append({
        "index":    idx,
        "filename": hr_path.name,
        "noise_floor": {
            "ae_noise_mean": noise_mean,
            "ae_noise_std":  noise_std,
            "threshold":     threshold,
        },
        "medvae_sr": {
            "hallu_rate":      metrics_m["hallu_rate"],
            "loss_rate":       metrics_m["loss_rate"],
            "hallu_magnitude": metrics_m["hallu_magnitude"],
            "loss_magnitude":  metrics_m["loss_magnitude"],
        },
        "sdvae_sr": {
            "hallu_rate":      metrics_s["hallu_rate"],
            "loss_rate":       metrics_s["loss_rate"],
            "hallu_magnitude": metrics_s["hallu_magnitude"],
            "loss_magnitude":  metrics_s["loss_magnitude"],
        },
    })

    # --- 5-panel comparison for first COMPARISON_LIMIT images ---
    if idx < COMPARISON_LIMIT:
        abs_D_m = np.abs(metrics_m["signed_diff"])

        fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=100)
        panels = [
            (hr,                               "HR (ground truth)",      "gray", 1.0),
            (sr_m,                             "SR MedVAE",              "gray", 1.0),
            (abs_D_m,                          "|SR_MedVAE - HR|",       "hot",  max(abs_D_m.max(), 1e-6)),
            (metrics_m["hallu_mask"].astype(float), "Hallucination mask", "Reds", 1.0),
            (metrics_m["loss_mask"].astype(float),  "Loss mask",          "Blues", 1.0),
        ]
        for ax, (arr, title, cmap, vmax) in zip(axes, panels):
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thr_str = f"{threshold:.4f}"
        fig.suptitle(
            f"{hr_path.stem}  |  threshold={thr_str}  |  "
            f"hallu={metrics_m['hallu_rate']:.4f}  loss={metrics_m['loss_rate']:.4f}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(COMPARISON_DIR / f"{hr_path.stem}_hallucination.png", bbox_inches="tight")
        plt.close(fig)

    if (idx + 1) % 50 == 0:
        print(f"  processed {idx + 1}/{len(tuples)} ...")

n_processed = len(hallu_masks_medvae)
print(f"\nProcessed {n_processed} image tuples.")

if n_processed == 0:
    raise RuntimeError("No images were processed. Check input directories.")

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
def agg_metric(stats: list[dict], method: str, metric: str) -> tuple[float, float]:
    vals = np.array([s[method][metric] for s in per_image_stats])
    return float(vals.mean()), float(vals.std())


agg = {}
for method in ("medvae_sr", "sdvae_sr"):
    agg[method] = {
        metric: {"mean": agg_metric(per_image_stats, method, metric)[0],
                 "std":  agg_metric(per_image_stats, method, metric)[1]}
        for metric in ("hallu_rate", "loss_rate", "hallu_magnitude", "loss_magnitude")
    }

print("\n--- Aggregate Hallucination / Loss Stats ---")
for method, metrics in agg.items():
    print(f"  {method}:")
    for metric, vals in metrics.items():
        print(f"    {metric:<20}: {vals['mean']:.6f} ± {vals['std']:.6f}")

# ---------------------------------------------------------------------------
# Mean hallucination masks spatially averaged across all images
# ---------------------------------------------------------------------------
mean_hallu_medvae = np.mean(np.stack(hallu_masks_medvae, axis=0), axis=0)
mean_hallu_sdvae  = np.mean(np.stack(hallu_masks_sdvae,  axis=0), axis=0)
mean_loss_medvae  = np.mean(np.stack(loss_masks_medvae,  axis=0), axis=0)
mean_loss_sdvae   = np.mean(np.stack(loss_masks_sdvae,   axis=0), axis=0)

# Save mean hallucination mask figure (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=120)
panels = [
    (mean_hallu_medvae, "Mean Hallucination Mask — MedVAE SR",  axes[0, 0]),
    (mean_hallu_sdvae,  "Mean Hallucination Mask — SD-VAE SR",  axes[0, 1]),
    (mean_loss_medvae,  "Mean Loss Mask — MedVAE SR",           axes[1, 0]),
    (mean_loss_sdvae,   "Mean Loss Mask — SD-VAE SR",           axes[1, 1]),
]
for mask, title, ax in panels:
    im = ax.imshow(mask, cmap="hot", vmin=0, vmax=max(mask.max(), 1e-6))
    ax.set_title(f"{title}\n({DATASET.upper()}, N={n_processed})", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Spatial Distribution of Hallucinated / Lost Content", fontsize=13)
fig.tight_layout()
masks_fig_path = OUT_DIR / "mean_hallu_masks.png"
fig.savefig(str(masks_fig_path), bbox_inches="tight")
plt.close(fig)
print(f"Mean hallucination masks saved to {masks_fig_path}")

# ---------------------------------------------------------------------------
# Summary bar chart: hallu_rate and loss_rate for both methods
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

methods = ["MedVAE SR", "SD-VAE SR"]
method_keys = ["medvae_sr", "sdvae_sr"]
colors_hallu = ["steelblue", "tomato"]
colors_loss  = ["cornflowerblue", "salmon"]

x = np.arange(len(methods))
bar_width = 0.35

hallu_means = [agg[k]["hallu_rate"]["mean"] for k in method_keys]
hallu_stds  = [agg[k]["hallu_rate"]["std"]  for k in method_keys]
loss_means  = [agg[k]["loss_rate"]["mean"]  for k in method_keys]
loss_stds   = [agg[k]["loss_rate"]["std"]   for k in method_keys]

bars_h = ax.bar(x - bar_width / 2, hallu_means, width=bar_width, yerr=hallu_stds,
                label="Hallucination rate", color=colors_hallu, capsize=5, alpha=0.85)
bars_l = ax.bar(x + bar_width / 2, loss_means,  width=bar_width, yerr=loss_stds,
                label="Loss rate",           color=colors_loss, capsize=5, alpha=0.85)

# Use a single legend for both hatching patterns
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="steelblue", label="Hallucination rate"),
    Patch(facecolor="cornflowerblue", label="Loss rate"),
]
ax.legend(handles=legend_elements, fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel("Mean Rate (fraction of pixels)", fontsize=11)
ax.set_title(
    f"Hallucination vs Loss Rate — {DATASET.upper()}  (N={n_processed})\n"
    f"threshold = AE_noise_mean + 2 × AE_noise_std",
    fontsize=11,
)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
bar_fig_path = OUT_DIR / "summary_bar_chart.png"
fig.savefig(str(bar_fig_path), bbox_inches="tight")
plt.close(fig)
print(f"Summary bar chart saved to {bar_fig_path}")

# ---------------------------------------------------------------------------
# Save JSON results
# ---------------------------------------------------------------------------
results = {
    "dataset": DATASET,
    "n_processed": n_processed,
    "method": "noise_floor = AE_noise_mean + 2 * AE_noise_std",
    "aggregate": agg,
    "per_image": per_image_stats,
}

with open(JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {JSON_PATH}")
print("\nDone.")
