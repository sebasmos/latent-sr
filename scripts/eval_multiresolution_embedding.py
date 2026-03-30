#!/usr/bin/env python3
"""
eval_multiresolution_embedding.py
Compare SR latents vs HR latents at multiple spatial pooling scales.

Shows WHERE in frequency space MedVAE preserves information by computing
MSE, PSNR, and cosine similarity at progressively coarser spatial poolings
of the (3, 64, 64) latent tensors.

Spatial scales evaluated: 64, 32, 16, 8, 4, 2, 1

Datasets: mrnet (120 val), brats (462-700 val), cxr (1000 val)

Usage:
  python scripts/eval_multiresolution_embedding.py --dataset mrnet
  python scripts/eval_multiresolution_embedding.py --dataset brats
  python scripts/eval_multiresolution_embedding.py --dataset cxr

Inputs:
  SR latents : outputs/experiments/{dataset}_decoder_finetune/sr_latents/sr_*.npy
               shape: (3, 64, 64) float32
  HR latents : ${LATENT_SR_EMBEDDINGS_ROOT}/medvae_{dataset}_s1/
               valid_latent/hr_*.npy
               shape: (3, 64, 64) float32
               NOTE: HR files are named hr_0.npy, hr_1.npy, ... (non-zero-padded integers)
                     SR files are named sr_00000.npy, sr_00001.npy, ... (zero-padded 5-digit)
               Both lists are sorted and matched by index position.

Outputs:
  outputs/experiments/multiresolution_embedding_{dataset}/results.json
  outputs/figures/multiresolution_embedding_{dataset}.png

No GPU required. Dependencies: numpy, matplotlib, json, argparse, pathlib, skimage.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from math import log10

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from repro_paths import embeddings_root, outputs_root, repo_root

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-resolution latent embedding comparison")
parser.add_argument("--dataset", choices=["mrnet", "brats", "cxr"], default="mrnet",
                    help="Dataset to evaluate")
parser.add_argument("--sr-latent-dir", type=pathlib.Path, default=None,
                    help="Override SR latent directory (default: outputs/experiments/{dataset}_decoder_finetune/sr_latents)")
args = parser.parse_args()
DATASET = args.dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = repo_root()

if args.sr_latent_dir is not None:
    # Use the explicitly provided SR latent directory (may be absolute or relative to cwd)
    SR_LATENT_DIR = args.sr_latent_dir if args.sr_latent_dir.is_absolute() else ROOT / args.sr_latent_dir
else:
    # Default: training-generated SR latents from decoder fine-tune pipeline
    _default_sr = ROOT / f"outputs/experiments/cxr_sr_latents_valid"
    _fallback_sr = ROOT / f"outputs/experiments/{DATASET}_decoder_finetune/sr_latents"
    if DATASET == "cxr" and _default_sr.exists():
        # Prefer the correct validation-split re-encoded latents for CXR (issue #97)
        SR_LATENT_DIR = _default_sr
        print(f"NOTE: Using validation-split SR latents for CXR (issue #97 fix): {SR_LATENT_DIR}")
    else:
        SR_LATENT_DIR = _fallback_sr

HR_LATENT_DIR = embeddings_root() / f"medvae_{DATASET}_s1/valid_latent"

OUT_DIR   = outputs_root() / f"experiments/multiresolution_embedding_{DATASET}"
FIG_DIR   = outputs_root() / "figures"
JSON_PATH = OUT_DIR / "results.json"
FIG_PATH  = FIG_DIR / f"multiresolution_embedding_{DATASET}.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Spatial scales to evaluate (native latent is 64x64)
SCALES = [64, 32, 16, 8, 4, 2, 1]

print(f"Dataset        : {DATASET}")
print(f"SR latent dir  : {SR_LATENT_DIR}")
print(f"HR latent dir  : {HR_LATENT_DIR}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def average_pool_latent(lat: np.ndarray, target_size: int) -> np.ndarray:
    """
    Average pool a (C, H, W) latent to (C, target_size, target_size).
    Uses stride-based block averaging (no skimage required for power-of-2 scales).

    Args:
        lat: numpy array of shape (C, H, W)
        target_size: desired spatial size (must divide H and W evenly)
    Returns:
        pooled array of shape (C, target_size, target_size)
    """
    C, H, W = lat.shape
    assert H % target_size == 0 and W % target_size == 0, (
        f"target_size={target_size} must divide H={H} and W={W}"
    )
    if target_size == H:
        return lat.copy()
    block_h = H // target_size
    block_w = W // target_size
    # Reshape to (C, target_size, block_h, target_size, block_w) then mean
    pooled = lat.reshape(C, target_size, block_h, target_size, block_w).mean(axis=(2, 4))
    return pooled


def compute_psnr(sr_arr: np.ndarray, hr_arr: np.ndarray) -> float:
    """
    Compute PSNR between two arrays. max_val is the max absolute value in hr_arr.
    Returns 100.0 if MSE < 1e-12.
    """
    mse = float(np.mean((sr_arr - hr_arr) ** 2))
    if mse < 1e-12:
        return 100.0
    max_val = float(np.abs(hr_arr).max())
    if max_val < 1e-12:
        return 0.0
    return float(10.0 * log10(max_val ** 2 / mse))


def compute_cosine_similarity(sr_arr: np.ndarray, hr_arr: np.ndarray) -> float:
    """
    Compute cosine similarity between flattened sr_arr and hr_arr.
    Returns 0.0 if either norm is near zero.
    """
    sr_flat = sr_arr.ravel().astype(np.float64)
    hr_flat = hr_arr.ravel().astype(np.float64)
    norm_sr = np.linalg.norm(sr_flat)
    norm_hr = np.linalg.norm(hr_flat)
    if norm_sr < 1e-12 or norm_hr < 1e-12:
        return 0.0
    return float(np.dot(sr_flat, hr_flat) / (norm_sr * norm_hr))


# ---------------------------------------------------------------------------
# Build matched SR / HR latent pairs by sort-order index position
# HR files: hr_0.npy, hr_1.npy, ... (sorted by integer index in filename)
# SR files: sr_00000.npy, sr_00001.npy, ... (sorted lexicographically)
# ---------------------------------------------------------------------------

def sort_by_integer_suffix(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """
    Sort paths by the integer embedded in the stem (after the last underscore).
    E.g., hr_0.npy < hr_1.npy < hr_10.npy (not lexicographic).
    """
    def extract_int(p: pathlib.Path) -> int:
        stem = p.stem  # e.g., "hr_0" or "sr_00000"
        suffix = stem.rsplit("_", 1)[-1]
        return int(suffix)

    return sorted(paths, key=extract_int)


sr_files_raw = list(SR_LATENT_DIR.glob("sr_*.npy"))
hr_files_raw = list(HR_LATENT_DIR.glob("hr_*.npy"))

if not sr_files_raw:
    raise RuntimeError(f"No SR latent files found in {SR_LATENT_DIR}")
if not hr_files_raw:
    raise RuntimeError(f"No HR latent files found in {HR_LATENT_DIR}")

sr_files = sort_by_integer_suffix(sr_files_raw)
hr_files = sort_by_integer_suffix(hr_files_raw)

n_pairs = min(len(sr_files), len(hr_files))
print(f"SR latents: {len(sr_files)}")
print(f"HR latents: {len(hr_files)}")
print(f"Using {n_pairs} matched pairs (by sort-order index)")

sr_files = sr_files[:n_pairs]
hr_files = hr_files[:n_pairs]

# ---------------------------------------------------------------------------
# Per-pair, per-scale analysis
# ---------------------------------------------------------------------------
# Accumulators: scale -> list of (mse, cosine_sim, psnr)
scale_metrics: dict[int, dict[str, list[float]]] = {
    s: {"mse": [], "cosine_sim": [], "psnr": []}
    for s in SCALES
}

for idx, (sr_path, hr_path) in enumerate(zip(sr_files, hr_files)):
    if not sr_path.exists() or not hr_path.exists():
        print(f"  SKIP {idx}: missing {sr_path.name} or {hr_path.name}")
        continue

    sr_lat = np.load(str(sr_path)).astype(np.float32)
    hr_lat = np.load(str(hr_path)).astype(np.float32)

    # Handle different latent shapes — ensure (C, H, W)
    if sr_lat.ndim == 4:
        sr_lat = sr_lat.squeeze(0)  # (1, C, H, W) -> (C, H, W)
    if hr_lat.ndim == 4:
        hr_lat = hr_lat.squeeze(0)
    if sr_lat.ndim == 2:
        sr_lat = sr_lat[np.newaxis, ...]  # (H, W) -> (1, H, W)
    if hr_lat.ndim == 2:
        hr_lat = hr_lat[np.newaxis, ...]

    if sr_lat.shape != hr_lat.shape:
        print(f"  SKIP {idx}: shape mismatch — SR={sr_lat.shape}, HR={hr_lat.shape}")
        continue

    C, H, W = sr_lat.shape

    for scale in SCALES:
        if H % scale != 0 or W % scale != 0:
            # Skip this scale if it doesn't divide evenly
            continue

        sr_pool = average_pool_latent(sr_lat, scale)
        hr_pool = average_pool_latent(hr_lat, scale)

        mse      = float(np.mean((sr_pool - hr_pool) ** 2))
        cosine   = compute_cosine_similarity(sr_pool, hr_pool)
        psnr_val = compute_psnr(sr_pool, hr_pool)

        scale_metrics[scale]["mse"].append(mse)
        scale_metrics[scale]["cosine_sim"].append(cosine)
        scale_metrics[scale]["psnr"].append(psnr_val)

    if (idx + 1) % 20 == 0:
        print(f"  processed {idx + 1}/{n_pairs} pairs ...")

n_processed = len(scale_metrics[SCALES[0]]["mse"])
print(f"\nProcessed {n_processed} pairs.")

if n_processed == 0:
    raise RuntimeError("No pairs were processed. Check input directories and latent shapes.")

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
agg_results: dict[int, dict] = {}

print(f"\n--- Multi-resolution Latent Comparison (mean ± std) ---")
print(f"{'Scale':>6}  {'PSNR (dB)':>18}  {'Cosine Sim':>18}  {'MSE':>18}")
print("-" * 65)

for scale in SCALES:
    psnr_arr   = np.array(scale_metrics[scale]["psnr"])
    cosine_arr = np.array(scale_metrics[scale]["cosine_sim"])
    mse_arr    = np.array(scale_metrics[scale]["mse"])

    n = len(psnr_arr)
    agg_results[scale] = {
        "n": n,
        "psnr":       {"mean": float(psnr_arr.mean()),   "std": float(psnr_arr.std())},
        "cosine_sim": {"mean": float(cosine_arr.mean()), "std": float(cosine_arr.std())},
        "mse":        {"mean": float(mse_arr.mean()),    "std": float(mse_arr.std())},
    }

    print(
        f"{str(scale)+'x'+str(scale):>6}  "
        f"{psnr_arr.mean():8.3f} ± {psnr_arr.std():.3f}    "
        f"{cosine_arr.mean():.5f} ± {cosine_arr.std():.5f}    "
        f"{mse_arr.mean():.6f} ± {mse_arr.std():.6f}"
    )

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "dataset": DATASET,
    "n_processed": n_processed,
    "sr_latent_dir": str(SR_LATENT_DIR),
    "hr_latent_dir": str(HR_LATENT_DIR),
    "spatial_scales": SCALES,
    "per_scale": {str(s): agg_results[s] for s in SCALES},
}

with open(JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {JSON_PATH}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
scales_x = np.array(SCALES, dtype=float)

psnr_means   = np.array([agg_results[s]["psnr"]["mean"]       for s in SCALES])
psnr_stds    = np.array([agg_results[s]["psnr"]["std"]        for s in SCALES])
cosine_means = np.array([agg_results[s]["cosine_sim"]["mean"] for s in SCALES])
cosine_stds  = np.array([agg_results[s]["cosine_sim"]["std"]  for s in SCALES])

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

# Panel (a): PSNR vs spatial scale
ax = axes[0]
ax.plot(scales_x, psnr_means, marker="o", color="steelblue", linewidth=2, markersize=7,
        label="MedVAE SR vs HR")
ax.fill_between(scales_x, psnr_means - psnr_stds, psnr_means + psnr_stds,
                color="steelblue", alpha=0.2, label="±1 std")
ax.set_xscale("log", base=2)
ax.set_xticks(SCALES)
ax.set_xticklabels([f"{s}×{s}" for s in SCALES], rotation=30, ha="right", fontsize=9)
ax.set_xlabel("Spatial Scale (after average pooling)", fontsize=11)
ax.set_ylabel("PSNR (dB)", fontsize=11)
ax.set_title(f"(a) Latent PSNR vs Spatial Scale\n{DATASET.upper()}  (N={n_processed})", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.invert_xaxis()  # fine → coarse from left to right

# Panel (b): Cosine similarity vs spatial scale
ax2 = axes[1]
ax2.plot(scales_x, cosine_means, marker="s", color="darkorange", linewidth=2, markersize=7,
         label="MedVAE SR vs HR")
ax2.fill_between(scales_x, cosine_means - cosine_stds, cosine_means + cosine_stds,
                 color="darkorange", alpha=0.2, label="±1 std")
ax2.set_xscale("log", base=2)
ax2.set_xticks(SCALES)
ax2.set_xticklabels([f"{s}×{s}" for s in SCALES], rotation=30, ha="right", fontsize=9)
ax2.set_xlabel("Spatial Scale (after average pooling)", fontsize=11)
ax2.set_ylabel("Cosine Similarity", fontsize=11)
ax2.set_title(f"(b) Latent Cosine Similarity vs Spatial Scale\n{DATASET.upper()}  (N={n_processed})",
              fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.invert_xaxis()  # fine → coarse from left to right

plt.tight_layout()
fig.savefig(str(FIG_PATH), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to {FIG_PATH}")
print("\nDone.")
