#!/usr/bin/env python3
"""
eval_frequency_analysis.py
Multi-resolution frequency analysis comparing SR vs HR images.

Shows WHERE the quality difference comes from via:
  - 3-level Haar wavelet decomposition (subband PSNR)
  - Radial FFT power spectrum comparison

Datasets: mrnet (120 val), brats (462-700 val), cxr (1000 val)

Usage:
  python scripts/eval_frequency_analysis.py --dataset mrnet
  python scripts/eval_frequency_analysis.py --dataset brats
  python scripts/eval_frequency_analysis.py --dataset cxr

Inputs:
  MedVAE SR : outputs/experiments/{dataset}_medvae_s1/sr_images/*.png
  SD-VAE SR : outputs/experiments/{dataset}_sdvae/sr_images/*.png
  HR images : ${LATENT_SR_DATA_ROOT}/{dataset_path}/valid/hr/*.png

Outputs:
  outputs/experiments/frequency_analysis_{dataset}/results.json
  outputs/figures/frequency_analysis_{dataset}.png

No GPU required. Dependencies: pywt, numpy, matplotlib, PIL, json, argparse, pathlib.
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
import pywt
from PIL import Image

from repro_paths import dataset_hr_dir, outputs_root, repo_root

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Multi-resolution frequency analysis: SR vs HR")
parser.add_argument("--dataset", choices=["mrnet", "brats", "cxr"], default="mrnet",
                    help="Dataset to evaluate")
args = parser.parse_args()
DATASET = args.dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = repo_root()

HR_DIRS = {
    "mrnet": dataset_hr_dir("mrnet", "valid"),
    "brats": dataset_hr_dir("brats", "valid"),
    "cxr": dataset_hr_dir("cxr", "valid"),
}

# BraTS SR images on test split; use _valid dirs for validation-split HR comparison
_suffix = "_valid" if DATASET == "brats" else ""
MEDVAE_SR_DIR = ROOT / f"outputs/experiments/{DATASET}_medvae_s1{_suffix}/sr_images"
SDVAE_SR_DIR  = ROOT / f"outputs/experiments/{DATASET}_sdvae{_suffix}/sr_images"
HR_DIR        = HR_DIRS[DATASET]

OUT_DIR    = outputs_root() / f"experiments/frequency_analysis_{DATASET}"
FIG_DIR    = outputs_root() / "figures"
JSON_PATH  = OUT_DIR / "results.json"
FIG_PATH   = FIG_DIR / f"frequency_analysis_{DATASET}.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

WAVELET = "haar"
LEVEL   = 3
N_RADIAL_BINS = 10

# ---------------------------------------------------------------------------
# Subband labels (coarse to fine)
# ---------------------------------------------------------------------------
SUBBAND_LABELS = [
    "LL3",
    "LH3", "HL3", "HH3",
    "LH2", "HL2", "HH2",
    "LH1", "HL1", "HH1",
]

print(f"Dataset        : {DATASET}")
print(f"MedVAE SR dir  : {MEDVAE_SR_DIR}")
print(f"SD-VAE SR dir  : {SDVAE_SR_DIR}")
print(f"HR dir         : {HR_DIR}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_gray(path: pathlib.Path) -> np.ndarray:
    """Load a grayscale PNG as float32 numpy array in [0, 1], shape (H, W)."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def subband_psnr(sr_coeff: np.ndarray, hr_coeff: np.ndarray) -> float:
    """
    Compute PSNR for a single wavelet subband.
    max_val is the max absolute pixel value in the HR subband.
    """
    mse = float(np.mean((sr_coeff - hr_coeff) ** 2))
    if mse < 1e-12:
        return 100.0
    max_val = float(np.abs(hr_coeff).max())
    if max_val < 1e-12:
        return 0.0
    return float(10.0 * log10(max_val ** 2 / mse))


def wavelet_subband_psnr(sr: np.ndarray, hr: np.ndarray) -> dict[str, float]:
    """
    3-level Haar wavelet decomposition on SR and HR images.
    Returns dict mapping subband label -> PSNR (dB).
    """
    sr_coeffs = pywt.wavedec2(sr, wavelet=WAVELET, level=LEVEL)
    hr_coeffs = pywt.wavedec2(hr, wavelet=WAVELET, level=LEVEL)

    results: dict[str, float] = {}

    # Level 3 approximation (LL3)
    results["LL3"] = subband_psnr(sr_coeffs[0], hr_coeffs[0])

    # Detail subbands: wavedec2 returns [cA, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    # Index 1 = level 3, index 2 = level 2, index 3 = level 1
    detail_names = [
        ("LH3", "HL3", "HH3"),  # level index 1 (coarsest detail)
        ("LH2", "HL2", "HH2"),  # level index 2
        ("LH1", "HL1", "HH1"),  # level index 3 (finest)
    ]
    for lvl_idx, (lh_name, hl_name, hh_name) in enumerate(detail_names, start=1):
        sr_lh, sr_hl, sr_hh = sr_coeffs[lvl_idx]
        hr_lh, hr_hl, hr_hh = hr_coeffs[lvl_idx]
        results[lh_name] = subband_psnr(sr_lh, hr_lh)
        results[hl_name] = subband_psnr(sr_hl, hr_hl)
        results[hh_name] = subband_psnr(sr_hh, hr_hh)

    return results


def radial_power_spectrum(img: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Compute mean log10 power in radial bins of the 2D FFT power spectrum.

    Returns array of shape (n_bins,) with mean log10 power per bin.
    Bins are equally spaced from DC (0) to Nyquist (0.5 * sqrt(2)).
    """
    H, W = img.shape
    fft2 = np.fft.fft2(img)
    fft2_shifted = np.fft.fftshift(fft2)
    power = np.abs(fft2_shifted) ** 2

    # Build radial frequency map
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[0:H, 0:W]
    # Normalize frequencies to [0, 1] where 1 = Nyquist along each axis
    fy = (yy - cy) / H
    fx = (xx - cx) / W
    freq_r = np.sqrt(fx ** 2 + fy ** 2)  # range [0, ~0.707]

    max_freq = freq_r.max()
    bin_edges = np.linspace(0.0, max_freq, n_bins + 1)

    radial_means = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (freq_r >= bin_edges[i]) & (freq_r < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_means[i] = float(np.mean(np.log10(power[mask] + 1e-12)))
        else:
            radial_means[i] = -12.0

    return radial_means


# ---------------------------------------------------------------------------
# Build matched image triplets (by filename intersection)
# ---------------------------------------------------------------------------
def build_triplets() -> list[tuple[pathlib.Path, pathlib.Path, pathlib.Path]]:
    """Match SR_medvae, SR_sdvae, HR by filename (stem intersection)."""
    medvae_by_stem = {p.stem: p for p in sorted(MEDVAE_SR_DIR.glob("*.png"))}
    sdvae_by_stem  = {p.stem: p for p in sorted(SDVAE_SR_DIR.glob("*.png"))}
    hr_by_stem     = {p.stem: p for p in sorted(HR_DIR.glob("*.png"))}

    common = sorted(set(medvae_by_stem) & set(sdvae_by_stem) & set(hr_by_stem))
    if not common:
        raise RuntimeError(
            f"No matching filenames across all three directories.\n"
            f"  MedVAE SR: {MEDVAE_SR_DIR} ({len(medvae_by_stem)} files)\n"
            f"  SD-VAE SR: {SDVAE_SR_DIR} ({len(sdvae_by_stem)} files)\n"
            f"  HR:        {HR_DIR} ({len(hr_by_stem)} files)"
        )
    return [(medvae_by_stem[s], sdvae_by_stem[s], hr_by_stem[s]) for s in common]


triplets = build_triplets()
print(f"Matched triplets: {len(triplets)}")

# ---------------------------------------------------------------------------
# Per-image analysis
# ---------------------------------------------------------------------------
# Subband PSNR accumulators
subband_psnr_medvae: dict[str, list[float]] = {k: [] for k in SUBBAND_LABELS}
subband_psnr_sdvae:  dict[str, list[float]] = {k: [] for k in SUBBAND_LABELS}

# Radial power spectrum accumulators
radial_medvae_list: list[np.ndarray] = []
radial_sdvae_list:  list[np.ndarray] = []
radial_hr_list:     list[np.ndarray] = []

for idx, (medvae_path, sdvae_path, hr_path) in enumerate(triplets):
    # Skip if any file is missing
    if not medvae_path.exists() or not sdvae_path.exists() or not hr_path.exists():
        print(f"  SKIP {idx}: missing file(s)")
        continue

    sr_m = load_gray(medvae_path)
    sr_s = load_gray(sdvae_path)
    hr   = load_gray(hr_path)

    # Wavelet PSNR
    wb_m = wavelet_subband_psnr(sr_m, hr)
    wb_s = wavelet_subband_psnr(sr_s, hr)
    for band in SUBBAND_LABELS:
        subband_psnr_medvae[band].append(wb_m[band])
        subband_psnr_sdvae[band].append(wb_s[band])

    # Radial power spectrum
    radial_medvae_list.append(radial_power_spectrum(sr_m, N_RADIAL_BINS))
    radial_sdvae_list.append(radial_power_spectrum(sr_s, N_RADIAL_BINS))
    radial_hr_list.append(radial_power_spectrum(hr, N_RADIAL_BINS))

    if (idx + 1) % 50 == 0:
        print(f"  processed {idx + 1}/{len(triplets)} ...")

n_processed = len(radial_hr_list)
print(f"Processed {n_processed} triplets.")

if n_processed == 0:
    raise RuntimeError("No images were processed. Check input directories.")

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
def aggregate(vals: list[float]) -> tuple[float, float]:
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


subband_stats_medvae = {band: aggregate(subband_psnr_medvae[band]) for band in SUBBAND_LABELS}
subband_stats_sdvae  = {band: aggregate(subband_psnr_sdvae[band])  for band in SUBBAND_LABELS}

radial_medvae_arr = np.stack(radial_medvae_list, axis=0)  # (N, n_bins)
radial_sdvae_arr  = np.stack(radial_sdvae_list,  axis=0)
radial_hr_arr     = np.stack(radial_hr_list,     axis=0)

radial_medvae_mean = radial_medvae_arr.mean(axis=0)
radial_medvae_std  = radial_medvae_arr.std(axis=0)
radial_sdvae_mean  = radial_sdvae_arr.mean(axis=0)
radial_sdvae_std   = radial_sdvae_arr.std(axis=0)
radial_hr_mean     = radial_hr_arr.mean(axis=0)
radial_hr_std      = radial_hr_arr.std(axis=0)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print(f"\n--- Wavelet Subband PSNR (mean ± std) ---")
print(f"{'Band':<6}  {'MedVAE SR':>18}  {'SD-VAE SR':>18}")
for band in SUBBAND_LABELS:
    m_mean, m_std = subband_stats_medvae[band]
    s_mean, s_std = subband_stats_sdvae[band]
    print(f"{band:<6}  {m_mean:7.3f} ± {m_std:6.3f}   {s_mean:7.3f} ± {s_std:6.3f}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "dataset": DATASET,
    "n_processed": n_processed,
    "wavelet": WAVELET,
    "wavelet_level": LEVEL,
    "n_radial_bins": N_RADIAL_BINS,
    "subband_psnr": {
        "medvae": {band: {"mean": subband_stats_medvae[band][0], "std": subband_stats_medvae[band][1]}
                   for band in SUBBAND_LABELS},
        "sdvae":  {band: {"mean": subband_stats_sdvae[band][0],  "std": subband_stats_sdvae[band][1]}
                   for band in SUBBAND_LABELS},
    },
    "radial_power_spectrum": {
        "medvae": {"mean": radial_medvae_mean.tolist(), "std": radial_medvae_std.tolist()},
        "sdvae":  {"mean": radial_sdvae_mean.tolist(),  "std": radial_sdvae_std.tolist()},
        "hr":     {"mean": radial_hr_mean.tolist(),     "std": radial_hr_std.tolist()},
    },
}

with open(JSON_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {JSON_PATH}")

# ---------------------------------------------------------------------------
# Plotting — 1x2 horizontal layout (subband PSNR | power spectrum)
# ---------------------------------------------------------------------------
CM = 1 / 2.54  # inches per cm
fig, (ax_bar, ax_power) = plt.subplots(1, 2, figsize=(16 * CM * 2, 7 * CM * 2), dpi=150)

# --- LEFT panel: Wavelet subband PSNR bar chart ---
x = np.arange(len(SUBBAND_LABELS))
bar_width = 0.38

medvae_means = np.array([subband_stats_medvae[b][0] for b in SUBBAND_LABELS])
medvae_stds  = np.array([subband_stats_medvae[b][1] for b in SUBBAND_LABELS])
sdvae_means  = np.array([subband_stats_sdvae[b][0]  for b in SUBBAND_LABELS])
sdvae_stds   = np.array([subband_stats_sdvae[b][1]  for b in SUBBAND_LABELS])

ax_bar.bar(x - bar_width / 2, medvae_means, width=bar_width, yerr=medvae_stds,
           label="MedVAE SR", color="steelblue", capsize=4, alpha=0.85)
ax_bar.bar(x + bar_width / 2, sdvae_means,  width=bar_width, yerr=sdvae_stds,
           label="SD-VAE SR", color="gray", capsize=4, alpha=0.75)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(SUBBAND_LABELS, fontsize=8, rotation=45, ha="right")
ax_bar.set_xlabel("Wavelet Subband  (coarse -> fine)", fontsize=10)
ax_bar.set_ylabel("PSNR (dB)", fontsize=10)
ax_bar.set_title(f"Subband PSNR -- {DATASET.upper()} (N={n_processed})", fontsize=10)
ax_bar.legend(fontsize=9)
ax_bar.grid(axis="y", alpha=0.3)

# Add vertical separators between wavelet levels
ylim_bar = ax_bar.get_ylim()
ax_bar.axvline(x=3.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax_bar.text(3.6, ylim_bar[0] + (ylim_bar[1] - ylim_bar[0]) * 0.02,
            "level 2", fontsize=7, color="gray")
ax_bar.axvline(x=6.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax_bar.text(6.6, ylim_bar[0] + (ylim_bar[1] - ylim_bar[0]) * 0.02,
            "level 1", fontsize=7, color="gray")

# --- RIGHT panel: Radial power spectrum ---
freq_axis = np.linspace(0.0, 1.0, N_RADIAL_BINS)  # normalized 0-1

ax_power.plot(freq_axis, radial_hr_mean, color="black", linestyle="--", linewidth=2,
              label="HR (ground truth)")
ax_power.fill_between(freq_axis,
                      radial_hr_mean - radial_hr_std,
                      radial_hr_mean + radial_hr_std,
                      color="black", alpha=0.10)

ax_power.plot(freq_axis, radial_medvae_mean, color="steelblue", linewidth=2,
              label="MedVAE SR")
ax_power.fill_between(freq_axis,
                      radial_medvae_mean - radial_medvae_std,
                      radial_medvae_mean + radial_medvae_std,
                      color="steelblue", alpha=0.15)

ax_power.plot(freq_axis, radial_sdvae_mean, color="gray", linewidth=2,
              label="SD-VAE SR")
ax_power.fill_between(freq_axis,
                      radial_sdvae_mean - radial_sdvae_std,
                      radial_sdvae_mean + radial_sdvae_std,
                      color="gray", alpha=0.15)

ax_power.set_xlabel("Spatial Frequency (cycles/pixel)", fontsize=10)
ax_power.set_ylabel("Mean Log Power (dB)", fontsize=10)
ax_power.set_title(f"Radial FFT Power Spectrum -- {DATASET.upper()}", fontsize=10)
ax_power.legend(fontsize=9)
ax_power.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(str(FIG_PATH), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to {FIG_PATH}")
print("\nDone.")
