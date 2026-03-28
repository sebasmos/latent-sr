#!/usr/bin/env python3
"""
eval_multit_embedding.py
Compare SR latents vs HR latents across multiple diffusion timesteps
(T=50, 100, 250, 1000) and all three medical imaging datasets.

For each dataset x timestep combination:
  - Load SR latents from:
      /orcd/pool/006/lceli_shared/mri-uganda/weights/decoder_multit_medvae_{dataset}_s1/sr_latents/{t_key}/sr_*.npy
  - Load HR latents from:
      /orcd/pool/006/lceli_shared/mri-uganda/embeddings/medvae_{dataset}_s1/valid_latent/hr_*.npy
  - Match by sort-order index (same convention as eval_multiresolution_embedding.py)
  - Compute cosine similarity at full resolution (64x64) and global mean (1x1)

Outputs:
  outputs/experiments/multit_embedding_{dataset}/results.json
      {t_value: {cosine_64: ..., cosine_1: ...}, ...}
  outputs/figures/multit_embedding.png
      x-axis = T value (50, 100, 250, 1000), y-axis = cosine sim at 64x64, one line per dataset

Usage:
  python scripts/eval_multit_embedding.py

No GPU required. Dependencies: numpy, matplotlib, json, pathlib.
"""

from __future__ import annotations

import json
import pathlib
from math import log10

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS_BASE = pathlib.Path("/orcd/pool/006/lceli_shared/mri-uganda/weights")
LATENT_BASE  = pathlib.Path("/orcd/pool/006/lceli_shared/mri-uganda/embeddings")

DATASETS = ["mrnet", "brats", "cxr"]
TIMESTEPS = [50, 100, 250, 1000]  # T values
T_KEYS = {50: "t50", 100: "t100", 250: "t250", 1000: "t1000"}

FIG_DIR = ROOT / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = FIG_DIR / "multit_embedding.png"

# ---------------------------------------------------------------------------
# Helper functions (same as eval_multiresolution_embedding.py)
# ---------------------------------------------------------------------------

def sort_by_integer_suffix(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """
    Sort paths by the integer embedded in the stem (after the last underscore).
    E.g., hr_0.npy < hr_1.npy < hr_10.npy  (not lexicographic).
    """
    def extract_int(p: pathlib.Path) -> int:
        stem = p.stem  # e.g., "hr_0" or "sr_00000"
        suffix = stem.rsplit("_", 1)[-1]
        return int(suffix)
    return sorted(paths, key=extract_int)


def average_pool_latent(lat: np.ndarray, target_size: int) -> np.ndarray:
    """
    Average pool a (C, H, W) latent to (C, target_size, target_size).
    Uses stride-based block averaging.
    """
    C, H, W = lat.shape
    if target_size == H:
        return lat.copy()
    assert H % target_size == 0 and W % target_size == 0, (
        f"target_size={target_size} must divide H={H} and W={W}"
    )
    block_h = H // target_size
    block_w = W // target_size
    pooled = lat.reshape(C, target_size, block_h, target_size, block_w).mean(axis=(2, 4))
    return pooled


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between flattened arrays. Returns 0.0 if either norm is near zero."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

# all_results[dataset][t_value] = {cosine_64: float, cosine_1: float, n: int}
all_results: dict[str, dict[int, dict]] = {ds: {} for ds in DATASETS}

for dataset in DATASETS:
    hr_latent_dir = LATENT_BASE / f"medvae_{dataset}_s1/valid_latent"
    out_dir = ROOT / f"outputs/experiments/multit_embedding_{dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}")
    print(f"HR latent dir: {hr_latent_dir}")

    # Skip already-completed datasets (results.json exists and has all timesteps)
    if json_path.exists():
        try:
            with open(json_path) as _f:
                _cached = json.load(_f)
            _cached_results = _cached.get("results", {})
            if all(str(t) in _cached_results for t in TIMESTEPS):
                print(f"  Already complete — loading cached results and skipping.")
                for t_val in TIMESTEPS:
                    all_results[dataset][t_val] = _cached_results[str(t_val)]
                continue
        except Exception:
            pass  # Re-run if JSON is corrupt

    if not hr_latent_dir.exists():
        print(f"  WARNING: HR latent dir not found — skipping dataset {dataset}")
        continue

    hr_files_raw = list(hr_latent_dir.glob("hr_*.npy"))
    if not hr_files_raw:
        print(f"  WARNING: No HR latents found in {hr_latent_dir} — skipping")
        continue

    hr_files_sorted = sort_by_integer_suffix(hr_files_raw)
    print(f"HR latents found: {len(hr_files_sorted)}")

    dataset_json: dict[str, dict] = {}

    for t_val in TIMESTEPS:
        t_key = T_KEYS[t_val]
        sr_latent_dir = WEIGHTS_BASE / f"decoder_multit_medvae_{dataset}_s1/sr_latents/{t_key}"

        print(f"\n  T={t_val} ({t_key}): {sr_latent_dir}")

        if not sr_latent_dir.exists():
            print(f"    SKIP: directory not found")
            continue

        # Optimisation: instead of globbing all SR files (slow over NFS for large dirs),
        # construct paths directly for only the first n_hr files using the zero-padded
        # naming convention sr_00000.npy, sr_00001.npy, ...
        # This avoids a costly directory listing when there are many more SR files than HR.
        n_hr = len(hr_files_sorted)
        sr_files_sorted = [sr_latent_dir / f"sr_{i:05d}.npy" for i in range(n_hr)]
        # Verify at least the first file actually exists
        if not sr_files_sorted[0].exists():
            # Fall back to glob if naming convention differs
            sr_files_raw = list(sr_latent_dir.glob("sr_*.npy"))
            if not sr_files_raw:
                print(f"    SKIP: no SR latent files found")
                continue
            sr_files_sorted = sort_by_integer_suffix(sr_files_raw)

        n_pairs = min(len(sr_files_sorted), n_hr)
        print(f"    SR latents: direct-path construction for {n_pairs} pairs (n_hr={n_hr})")

        sr_files = sr_files_sorted[:n_pairs]
        hr_files = hr_files_sorted[:n_pairs]

        cosine_64_list: list[float] = []
        cosine_1_list:  list[float] = []
        skipped = 0

        for idx, (sr_path, hr_path) in enumerate(zip(sr_files, hr_files)):
            if not sr_path.exists() or not hr_path.exists():
                skipped += 1
                continue

            sr_lat = np.load(str(sr_path)).astype(np.float32)
            hr_lat = np.load(str(hr_path)).astype(np.float32)

            # Normalise shape to (C, H, W)
            if sr_lat.ndim == 4:
                sr_lat = sr_lat.squeeze(0)
            if hr_lat.ndim == 4:
                hr_lat = hr_lat.squeeze(0)
            if sr_lat.ndim == 2:
                sr_lat = sr_lat[np.newaxis, ...]
            if hr_lat.ndim == 2:
                hr_lat = hr_lat[np.newaxis, ...]

            if sr_lat.shape != hr_lat.shape:
                print(f"    SKIP pair {idx}: shape mismatch SR={sr_lat.shape} HR={hr_lat.shape}")
                skipped += 1
                continue

            # Full resolution (64x64)
            cosine_64_list.append(compute_cosine_similarity(sr_lat, hr_lat))

            # Global mean (1x1)
            sr_1 = average_pool_latent(sr_lat, 1)
            hr_1 = average_pool_latent(hr_lat, 1)
            cosine_1_list.append(compute_cosine_similarity(sr_1, hr_1))

            if (idx + 1) % 100 == 0:
                print(f"    processed {idx + 1}/{n_pairs} pairs ...")

        n_processed = len(cosine_64_list)
        if skipped:
            print(f"    skipped {skipped} pairs (missing / shape mismatch)")

        if n_processed == 0:
            print(f"    WARNING: no pairs processed for T={t_val}")
            continue

        cos64_arr = np.array(cosine_64_list)
        cos1_arr  = np.array(cosine_1_list)

        result_entry = {
            "n": n_processed,
            "cosine_64": {
                "mean": float(cos64_arr.mean()),
                "std":  float(cos64_arr.std()),
            },
            "cosine_1": {
                "mean": float(cos1_arr.mean()),
                "std":  float(cos1_arr.std()),
            },
        }

        print(
            f"    Results: cosine_64 = {cos64_arr.mean():.5f} ± {cos64_arr.std():.5f}, "
            f"cosine_1 = {cos1_arr.mean():.5f} ± {cos1_arr.std():.5f}  (n={n_processed})"
        )

        all_results[dataset][t_val] = result_entry
        dataset_json[str(t_val)] = result_entry

    # Save per-dataset JSON
    full_json = {
        "dataset": dataset,
        "hr_latent_dir": str(hr_latent_dir),
        "sr_latent_base": str(WEIGHTS_BASE / f"decoder_multit_medvae_{dataset}_s1/sr_latents"),
        "timesteps_evaluated": TIMESTEPS,
        "results": dataset_json,
    }
    with open(json_path, "w") as f:
        json.dump(full_json, f, indent=2)
    print(f"\n  Results saved to {json_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY: cosine similarity at 64x64 across datasets and T values")
print(f"{'Dataset':<10}", end="")
for t in TIMESTEPS:
    print(f"  T={t:>4}", end="")
print()
print("-" * (10 + 9 * len(TIMESTEPS)))

for dataset in DATASETS:
    print(f"{dataset:<10}", end="")
    for t in TIMESTEPS:
        if t in all_results[dataset]:
            mean_val = all_results[dataset][t]["cosine_64"]["mean"]
            print(f"  {mean_val:.4f}", end="")
        else:
            print(f"  {'N/A':>6}", end="")
    print()

# ---------------------------------------------------------------------------
# Figure: x-axis = T value, y-axis = cosine_64, one line per dataset
# ---------------------------------------------------------------------------
DATASET_COLORS = {"mrnet": "steelblue", "brats": "darkorange", "cxr": "forestgreen"}
DATASET_MARKERS = {"mrnet": "o", "brats": "s", "cxr": "^"}

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

for dataset in DATASETS:
    t_avail = sorted(k for k in all_results[dataset])
    if not t_avail:
        continue
    t_vals    = np.array(t_avail)
    cos64_means = np.array([all_results[dataset][t]["cosine_64"]["mean"] for t in t_avail])
    cos64_stds  = np.array([all_results[dataset][t]["cosine_64"]["std"]  for t in t_avail])
    n_vals      = [all_results[dataset][t]["n"] for t in t_avail]

    label = f"{dataset.upper()} (n={n_vals[0]})" if n_vals else dataset.upper()
    color  = DATASET_COLORS.get(dataset, "black")
    marker = DATASET_MARKERS.get(dataset, "o")

    ax.plot(t_vals, cos64_means,
            marker=marker, color=color, linewidth=2, markersize=8, label=label)
    ax.fill_between(t_vals, cos64_means - cos64_stds, cos64_means + cos64_stds,
                    color=color, alpha=0.15)

ax.set_xscale("log")
ax.set_xticks(TIMESTEPS)
ax.set_xticklabels([str(t) for t in TIMESTEPS], fontsize=11)
ax.set_xlabel("Diffusion inference steps (T)", fontsize=12)
ax.set_ylabel("Cosine Similarity at 64×64", fontsize=12)
ax.set_title("Latent Cosine Similarity vs Inference Steps\n(MedVAE SR vs HR, full 64×64 resolution)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(str(FIG_PATH), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIG_PATH}")
print("\nDone.")
