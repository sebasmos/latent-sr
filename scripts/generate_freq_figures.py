#!/usr/bin/env python3
"""
generate_freq_figures.py
Create per-dataset frequency analysis figures with horizontal (1 row x 2 col) layout.

LEFT panel:  grouped bar chart of per-subband PSNR (MedVAE SR vs SD-VAE SR)
RIGHT panel: mean log radial power spectrum

Reads pre-computed results from:
  outputs/experiments/frequency_analysis_mrnet/results.json
  outputs/experiments/frequency_analysis_brats/results.json
  outputs/experiments/frequency_analysis_cxr/results.json

Saves to:
  outputs/figures/freq_mrnet_horizontal.png
  outputs/figures/freq_brats_horizontal.png
  outputs/figures/freq_cxr_horizontal.png

Then copies to figures-paper-1/:
  figures-paper-1/fig3a_mrnet_freq.png
  figures-paper-1/fig3b_brats_freq.png
  figures-paper-1/fig3c_cxr_freq.png

Usage:
  python scripts/generate_freq_figures.py
"""

from __future__ import annotations

import json
import pathlib
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures-paper-1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "key": "mrnet",
        "json": ROOT / "outputs/experiments/frequency_analysis_mrnet/results.json",
        "out_png": OUT_DIR / "freq_mrnet_horizontal.png",
        "paper_png": FIG_DIR / "fig3a_mrnet_freq.png",
        "title": "MRNet",
    },
    {
        "key": "brats",
        "json": ROOT / "outputs/experiments/frequency_analysis_brats/results.json",
        "out_png": OUT_DIR / "freq_brats_horizontal.png",
        "paper_png": FIG_DIR / "fig3b_brats_freq.png",
        "title": "BraTS",
    },
    {
        "key": "cxr",
        "json": ROOT / "outputs/experiments/frequency_analysis_cxr/results.json",
        "out_png": OUT_DIR / "freq_cxr_horizontal.png",
        "paper_png": FIG_DIR / "fig3c_cxr_freq.png",
        "title": "CXR",
    },
]

# ---------------------------------------------------------------------------
# Style constants (Nature-style)
# ---------------------------------------------------------------------------
COLOR_MEDVAE  = "#4C9BE8"   # blue
COLOR_SDVAE   = "#AAAAAA"   # gray
COLOR_HR      = "#000000"   # black
COLOR_BICUBIC = "#E87D2B"   # dashed orange

BASE_FONTSIZE  = 8
LABEL_FONTSIZE = 9

# Figure size: 16 cm wide x 7 cm tall (Nature single-column-ish)
FIG_W_IN = 16 / 2.54
FIG_H_IN = 7 / 2.54


def set_nature_style() -> None:
    """Apply Nature-style rc params."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "axes.titlesize": LABEL_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def plot_subband_psnr(ax: plt.Axes, subband_psnr: dict, title: str) -> None:
    """Grouped bar chart: per-subband PSNR for MedVAE SR vs SD-VAE SR."""
    medvae_data = subband_psnr.get("medvae", {})
    sdvae_data  = subband_psnr.get("sdvae", {})

    # Use medvae's subband order as canonical
    subbands = list(medvae_data.keys())

    medvae_means = [medvae_data[s]["mean"] for s in subbands]
    medvae_stds  = [medvae_data[s]["std"]  for s in subbands]
    sdvae_means  = [sdvae_data[s]["mean"]  for s in subbands if s in sdvae_data]
    sdvae_stds   = [sdvae_data[s]["std"]   for s in subbands if s in sdvae_data]

    x = np.arange(len(subbands))
    bar_w = 0.35

    ax.bar(
        x - bar_w / 2, medvae_means, bar_w,
        yerr=medvae_stds, capsize=2,
        color=COLOR_MEDVAE, label="MedVAE SR",
        error_kw={"elinewidth": 0.8, "ecolor": "black"},
        zorder=3,
    )
    ax.bar(
        x + bar_w / 2, sdvae_means, bar_w,
        yerr=sdvae_stds, capsize=2,
        color=COLOR_SDVAE, label="SD-VAE SR",
        error_kw={"elinewidth": 0.8, "ecolor": "black"},
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subbands, rotation=45, ha="right", fontsize=BASE_FONTSIZE - 1)
    ax.set_xlabel("Wavelet subband", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("PSNR (dB)", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"{title} — per-subband PSNR", fontsize=LABEL_FONTSIZE, pad=4)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def plot_radial_power(ax: plt.Axes, rps: dict, n_bins: int, title: str) -> None:
    """Mean log radial power spectrum for HR, MedVAE SR, SD-VAE SR, Bicubic (if present)."""
    freqs = np.arange(1, n_bins + 1)  # 1-indexed spatial frequency bins

    def _extract_mean(entry):
        if isinstance(entry, dict) and "mean" in entry:
            return np.array(entry["mean"])
        if isinstance(entry, list):
            return np.array(entry)
        return None

    # HR
    if "hr" in rps:
        hr_mean = _extract_mean(rps["hr"])
        if hr_mean is not None:
            ax.plot(freqs, hr_mean, color=COLOR_HR, lw=1.2, label="HR", zorder=4)

    # MedVAE SR
    if "medvae" in rps:
        medvae_mean = _extract_mean(rps["medvae"])
        if medvae_mean is not None:
            ax.plot(freqs, medvae_mean, color=COLOR_MEDVAE, lw=1.2, label="MedVAE SR", zorder=3)

    # SD-VAE SR
    if "sdvae" in rps:
        sdvae_mean = _extract_mean(rps["sdvae"])
        if sdvae_mean is not None:
            ax.plot(freqs, sdvae_mean, color=COLOR_SDVAE, lw=1.2, label="SD-VAE SR", zorder=3)

    # Bicubic (optional)
    if "bicubic" in rps:
        bic_mean = _extract_mean(rps["bicubic"])
        if bic_mean is not None:
            ax.plot(
                freqs, bic_mean,
                color=COLOR_BICUBIC, lw=1.0, linestyle="--",
                label="Bicubic", zorder=2,
            )

    ax.set_xlabel("Spatial frequency bin", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Mean log power (dB)", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"{title} — radial power spectrum", fontsize=LABEL_FONTSIZE, pad=4)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(linewidth=0.4, alpha=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def make_freq_figure(cfg: dict) -> None:
    """Create and save the horizontal 1x2 figure for one dataset."""
    json_path = cfg["json"]
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    subband_psnr = data.get("subband_psnr", {})
    rps          = data.get("radial_power_spectrum", {})
    n_bins       = data.get("n_radial_bins", len(next(iter(rps.values()))["mean"]) if rps else 10)
    title        = cfg["title"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_H_IN))

    plot_subband_psnr(axes[0], subband_psnr, title)
    plot_radial_power(axes[1], rps, n_bins, title)

    # Panel labels
    for ax, label in zip(axes, ["a", "b"]):
        ax.text(
            -0.12, 1.05, label, transform=ax.transAxes,
            fontsize=LABEL_FONTSIZE + 1, fontweight="bold", va="top",
        )

    fig.tight_layout(pad=0.5)
    fig.savefig(cfg["out_png"])
    plt.close(fig)
    print(f"  Saved: {cfg['out_png']}")

    # Copy to figures-paper-1/
    shutil.copy2(cfg["out_png"], cfg["paper_png"])
    print(f"  Copied: {cfg['paper_png']}")


def main() -> None:
    set_nature_style()
    print("Generating horizontal frequency analysis figures...")
    for cfg in DATASETS:
        print(f"\n[{cfg['title']}]")
        make_freq_figure(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
