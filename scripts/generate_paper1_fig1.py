#!/usr/bin/env python3
"""
Generate Fig. 1 — grouped bar chart, grouped by METHOD (x-axis) coloured by dataset.

Layout:
  Left panel  (a): PSNR (dB) — 6 method groups × 3 dataset bars each
  Right panel (b): LPIPS     — same

MedVAE SR bars are bold/highlighted. Improvement annotation (+X.XX dB) shown
above each dataset group connecting SD-VAE SR → MedVAE SR.

Output: outputs/figures/paper1_fig1.png / .pdf
Nature style: 18 cm wide, 8–9 pt fonts, 300 DPI.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
REPO    = Path(__file__).resolve().parents[1]
EXP     = REPO / "outputs" / "experiments"
STATS   = REPO / "outputs" / "statistical_tests" / "effect_sizes.json"
OUT_DIR = REPO / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return d.get("diffusion_sr", d)


def build_data():
    exp_map = {
        "MRNet": {
            "bicubic":   "mrnet_bicubic",
            "esrgan":    "mrnet_realesrgan",
            "swinir":    "mrnet_swinir",
            "sdvae_sr":  "mrnet_sdvae",
            "medvae_sr": "mrnet_medvae_s1",
            "medvae_ae": "mrnet_medvae_ae",
        },
        "BraTS": {
            "bicubic":   "brats_bicubic",
            "esrgan":    "brats_realesrgan",
            "swinir":    "brats_swinir",
            "sdvae_sr":  "brats_sdvae_valid",
            "medvae_sr": "brats_medvae_s1_valid",
            "medvae_ae": "brats_medvae_ae_valid",
        },
        "CXR": {
            "bicubic":   "cxr_bicubic",
            "esrgan":    "cxr_realesrgan",
            "swinir":    "cxr_swinir",
            "sdvae_sr":  "cxr_sdvae",
            "medvae_sr": "cxr_medvae_s1",
            "medvae_ae": "cxr_medvae_ae",
        },
    }
    mrnet_lpips = EXP / "mrnet_medvae_lpips_eval" / "diffusion_eval_results.json"

    data = {}
    for ds, methods in exp_map.items():
        data[ds] = {}
        for mkey, exp_name in methods.items():
            path = EXP / exp_name / "diffusion_eval_results.json"
            data[ds][mkey] = load_metrics(path) if path.exists() else {}

    if mrnet_lpips.exists():
        m = load_metrics(mrnet_lpips)
        data["MRNet"]["medvae_sr"]["lpips_mean"] = m.get("lpips_mean")
        data["MRNet"]["medvae_sr"]["lpips_std"]  = m.get("lpips_std")

    return data


def load_cis():
    with open(STATS) as f:
        d = json.load(f)
    return {
        comp["comparison"].split(":")[0]: {
            "diff":   comp["mean_difference"],
            "ci_lo":  comp["difference_ci_lower"],
            "ci_hi":  comp["difference_ci_upper"],
        }
        for comp in d["comparisons"]
    }


# ---------------------------------------------------------------------------
# Visual design
# ---------------------------------------------------------------------------

METHODS = [
    # (key,        x-axis label,          is_hero)
    ("bicubic",   "Bicubic",              False),
    ("esrgan",    "ESRGAN",               False),
    ("swinir",    "SwinIR",               False),
    ("sdvae_sr",  "SD-VAE SR",            False),
    ("medvae_sr", "MedVAE SR\n(ours)",    True),
    ("medvae_ae", "MedVAE AE\n(ceiling)", False),
]

DATASETS = ["MRNet", "BraTS", "CXR"]

# One colour per dataset, consistent across both panels
DS_COLORS = {
    "MRNet": "#4C9BE8",   # blue
    "BraTS": "#E87D4C",   # orange
    "CXR":   "#56B08B",   # green
}

METRICS = [
    ("psnr",  "PSNR (dB)", True),    # higher is better
    ("lpips", "LPIPS ↓",   False),   # lower is better
]


def make_figure(data, cis):
    CM = 1 / 2.54
    fig_w = 18 * CM
    fig_h = 10 * CM

    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       8,
        "axes.titlesize":  9,
        "axes.labelsize":  8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "axes.linewidth":  0.6,
        "lines.linewidth": 0.9,
        "figure.dpi":      300,
    })

    n_m = len(METHODS)
    n_ds = len(DATASETS)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.82, bottom=0.32, wspace=0.38)

    # Bar geometry: methods on x-axis, datasets as grouped bars within each method
    group_w   = 1.0
    bar_w     = group_w / (n_ds + 1.0)
    group_gap = 0.55
    group_centers = np.arange(n_m) * (group_w + group_gap)
    bar_offsets   = np.linspace(
        -(n_ds - 1) / 2 * bar_w,
         (n_ds - 1) / 2 * bar_w,
        n_ds,
    )

    panel_labels = ["a", "b"]

    for ax_idx, (metric_key, metric_label, higher_better) in enumerate(METRICS):
        ax = axes[ax_idx]
        mean_key = f"{metric_key}_mean"
        std_key  = f"{metric_key}_std"

        all_vals = []

        for m_idx, (mkey, mlabel, is_hero) in enumerate(METHODS):
            gc = group_centers[m_idx]
            for ds_idx, ds in enumerate(DATASETS):
                x    = gc + bar_offsets[ds_idx]
                m    = data[ds].get(mkey, {})
                mean = m.get(mean_key)
                std  = m.get(std_key)

                if mean is None:
                    continue

                all_vals.append(mean)
                color = DS_COLORS[ds]

                # Hero (MedVAE SR) gets a thicker/darker border
                ec = "#0A2D7D" if is_hero else "#888888"
                lw = 1.6 if is_hero else 0.5

                ax.bar(
                    x, mean,
                    width=bar_w * 0.88,
                    color=color,
                    edgecolor=ec,
                    linewidth=lw,
                    zorder=3,
                )
                if std is not None:
                    ax.errorbar(
                        x, mean, yerr=std,
                        fmt="none",
                        ecolor="#444444",
                        elinewidth=0.7,
                        capsize=2,
                        capthick=0.7,
                        zorder=4,
                    )

        # Improvement annotation between SD-VAE SR and MedVAE SR groups
        sdvae_gc  = group_centers[[i for i, (k,*_) in enumerate(METHODS) if k == "sdvae_sr"][0]]
        medvae_gc = group_centers[[i for i, (k,*_) in enumerate(METHODS) if k == "medvae_sr"][0]]

        for ds_idx, ds in enumerate(DATASETS):
            sd_mean  = data[ds].get("sdvae_sr",  {}).get(mean_key)
            med_mean = data[ds].get("medvae_sr", {}).get(mean_key)
            sd_std   = data[ds].get("sdvae_sr",  {}).get(std_key) or 0
            med_std  = data[ds].get("medvae_sr", {}).get(std_key) or 0

            if sd_mean is None or med_mean is None:
                continue

            x_sd  = sdvae_gc  + bar_offsets[ds_idx]
            x_med = medvae_gc + bar_offsets[ds_idx]

            y_top = max(sd_mean, med_mean)
            y_err = max(sd_std, med_std)

            if metric_key == "psnr":
                ci     = cis.get(ds, {})
                diff   = ci.get("diff", med_mean - sd_mean)
                ci_lo  = ci.get("ci_lo", diff)
                ci_hi  = ci.get("ci_hi", diff)
                by     = y_top + y_err + 0.5
                bh     = 0.08
            else:
                diff   = sd_mean - med_mean      # positive = MedVAE better
                by     = y_top + y_err + 0.012
                bh     = 0.003

            color = DS_COLORS[ds]
            ax.plot(
                [x_sd, x_sd, x_med, x_med],
                [by, by + bh, by + bh, by],
                color=color, linewidth=1.2, zorder=6,
            )
            if metric_key == "psnr":
                label = f"+{diff:.2f} dB"
                y_offset = by + bh + 0.15
            else:
                label = f"\u2212{diff:.4f}" if diff >= 0 else f"+{abs(diff):.4f}"
                y_offset = by + bh + 0.008

            ax.text(
                (x_sd + x_med) / 2,
                y_offset,
                label,
                ha="center", va="bottom",
                fontsize=7, color=color,
                fontweight="bold",
            )

        # Axes formatting
        ax.set_xticks(group_centers)
        ax.set_xticklabels([m[1] for m in METHODS], fontsize=8, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_ylabel(metric_label, fontsize=9, labelpad=4)
        ax.tick_params(axis="both", which="major", length=3, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.yaxis.grid(True, linewidth=0.3, color="#E0E0E0", zorder=0)
        ax.set_axisbelow(True)

        # Shade MedVAE SR group background
        medvae_gc = group_centers[[i for i, (k,*_) in enumerate(METHODS) if k == "medvae_sr"][0]]
        ax.axvspan(
            medvae_gc - group_w * 0.52, medvae_gc + group_w * 0.52,
            color="#EBF2FC", alpha=0.55, zorder=1, linewidth=0,
        )

        if all_vals:
            vmin = min(all_vals)
            vmax = max(all_vals)
            rng  = vmax - vmin
            pad  = 0.55 * rng if metric_key == "psnr" else 0.28 * rng
            ax.set_ylim(max(0, vmin - 0.05 * rng), vmax + pad)

        # Panel letter
        ax.text(-0.13, 1.04, panel_labels[ax_idx],
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")
        ax.set_title(metric_label, fontsize=9, pad=5)

    # Dataset legend (colours) + AE ceiling note
    ds_patches = [
        mpatches.Patch(facecolor=DS_COLORS[ds], edgecolor="#888", linewidth=0.5, label=lbl)
        for ds, lbl in [("MRNet", "MRNet (Knee MRI)"),
                        ("BraTS", "BraTS (Brain MRI)"),
                        ("CXR",   "CXR (Chest X-ray)")]
    ]
    ae_patch = mpatches.Patch(
        facecolor="none", edgecolor="#888", linewidth=0.8,
        label="Bars: each colour = one dataset", linestyle="--",
    )
    fig.legend(
        handles=ds_patches,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=8.5,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.2,
    )

    return fig


def main():
    data = build_data()
    cis  = load_cis()
    fig  = make_figure(data, cis)

    out_png = OUT_DIR / "paper1_fig1.png"
    out_pdf = OUT_DIR / "paper1_fig1.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
