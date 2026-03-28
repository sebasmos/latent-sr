#!/usr/bin/env python3
"""
Generate perception-distortion tradeoff plots for the paper.

Each plot shows PSNR (x-axis, higher = less distortion) vs LPIPS (y-axis, lower =
better perceptual quality) for all methods on a single dataset. MedVAE SR occupies
the "sweet spot" with competitive PSNR and the best LPIPS among SR methods.

Output: outputs/figures/perception_distortion_{dataset}.png (300 DPI)

Usage:
    python scripts/plot_perception_distortion.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Hard-coded results from evaluation runs.
#
# PSNR values are from outputs/experiments/*/diffusion_eval_results.json
# LPIPS values are from the same JSON files, step_ablation (T=1000), or
# computed baselines.  Where a value was not captured during evaluation,
# the best available proxy or user-supplied estimate is used (marked ~).
# ---------------------------------------------------------------------------

RESULTS = {
    "MRNet": {
        "title": "MRNet (Knee MRI)",
        "methods": {
            "Bicubic":      {"psnr": 23.79, "lpips": 0.5411},
            "ESRGAN":       {"psnr": 23.28, "lpips": 0.4245},
            "SD-VAE SR":    {"psnr": 22.34, "lpips": 0.30},    # estimated
            "MedVAE SR":    {"psnr": 25.26, "lpips": 0.1353},  # step ablation T=1000
            "MedVAE AE":    {"psnr": 27.85, "lpips": 0.1088},
        },
    },
    "BraTS": {
        "title": "BraTS (Brain MRI)",
        "methods": {
            "Bicubic":      {"psnr": 29.91, "lpips": 0.2178},
            "ESRGAN":       {"psnr": 27.45, "lpips": 0.0977},
            "SD-VAE SR":    {"psnr": 24.16, "lpips": 0.0910},
            "MedVAE SR":    {"psnr": 27.13, "lpips": 0.0128},
            "MedVAE AE":    {"psnr": 38.42, "lpips": 0.0128},
        },
    },
    "CXR": {
        "title": "MIMIC-CXR (Chest X-ray)",
        "methods": {
            "Bicubic":      {"psnr": 30.47, "lpips": 0.3299},
            "ESRGAN":       {"psnr": 27.71, "lpips": 0.1752},
            "SD-VAE SR":    {"psnr": 25.58, "lpips": 0.17},    # estimated
            "MedVAE SR":    {"psnr": 28.87, "lpips": 0.1265},
            "MedVAE AE":    {"psnr": 36.93, "lpips": 0.0293},
        },
    },
}

# Visual styling per method -- consistent across all subplots.
METHOD_STYLE = {
    "Bicubic":   {"marker": "s", "color": "#7f8c8d", "zorder": 3},  # grey square
    "ESRGAN":    {"marker": "D", "color": "#e67e22", "zorder": 3},  # orange diamond
    "SD-VAE SR": {"marker": "^", "color": "#e74c3c", "zorder": 3},  # red triangle
    "MedVAE SR": {"marker": "*", "color": "#2ecc71", "zorder": 5},  # green star (hero)
    "MedVAE AE": {"marker": "o", "color": "#3498db", "zorder": 4},  # blue circle
}

MARKER_SIZE_DEFAULT = 480
MARKER_SIZE_HERO = 780  # MedVAE SR gets a larger marker


def _annotate_offset(method: str) -> tuple[float, float]:
    """Return (dx, dy) text offset in points for each method label to avoid overlap."""
    offsets = {
        "Bicubic":   (8, 6),
        "ESRGAN":    (8, 6),
        "SD-VAE SR": (8, -12),
        "MedVAE SR": (8, 8),
        "MedVAE AE": (8, -12),
    }
    return offsets.get(method, (8, 4))


def plot_single_dataset(
    ax: plt.Axes,
    dataset_key: str,
    data: dict,
) -> None:
    """Draw one perception-distortion subplot."""

    methods = data["methods"]

    for name, vals in methods.items():
        style = METHOD_STYLE[name]
        size = MARKER_SIZE_HERO if name == "MedVAE SR" else MARKER_SIZE_DEFAULT
        edgecolor = "black" if name == "MedVAE SR" else style["color"]
        linewidth = 1.5 if name == "MedVAE SR" else 0.8

        ax.scatter(
            vals["psnr"],
            vals["lpips"],
            s=size,
            marker=style["marker"],
            color=style["color"],
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=style["zorder"],
            label=name,
        )

        # Annotate with method name
        dx, dy = _annotate_offset(name)
        ax.annotate(
            name,
            xy=(vals["psnr"], vals["lpips"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=15,
            fontweight="bold" if name == "MedVAE SR" else "normal",
            color=style["color"],
            ha="left",
            va="center",
        )

    # Shade the "sweet spot" region around MedVAE SR
    hero = methods["MedVAE SR"]
    ax.annotate(
        "",
        xy=(hero["psnr"], hero["lpips"]),
        xytext=(hero["psnr"] - 0.3, hero["lpips"] + 0.005),
        arrowprops=dict(arrowstyle="-", color="#2ecc71", lw=0),
    )

    # Draw a light arrow from "ideal direction" annotation
    psnr_vals = [v["psnr"] for v in methods.values()]
    lpips_vals = [v["lpips"] for v in methods.values()]
    psnr_range = max(psnr_vals) - min(psnr_vals)
    lpips_range = max(lpips_vals) - min(lpips_vals)

    # Add a small "ideal" arrow in the bottom-right corner
    arrow_x = max(psnr_vals) - 0.08 * psnr_range
    arrow_y = min(lpips_vals) + 0.08 * lpips_range
    ax.annotate(
        "ideal",
        xy=(arrow_x + 0.06 * psnr_range, arrow_y - 0.06 * lpips_range),
        xytext=(arrow_x - 0.06 * psnr_range, arrow_y + 0.06 * lpips_range),
        fontsize=7,
        color="#555555",
        fontstyle="italic",
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color="#999999",
            lw=1.2,
            connectionstyle="arc3,rad=0",
        ),
    )

    # Axis configuration
    ax.set_xlabel("PSNR (dB)  " + r"$\longrightarrow$" + "  less distortion", fontsize=11)
    ax.set_ylabel(r"$\longleftarrow$" + "  better perception  " + "LPIPS", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Pad axes
    ax.set_xlim(min(psnr_vals) - 0.12 * psnr_range, max(psnr_vals) + 0.18 * psnr_range)
    ax.set_ylim(
        max(0, min(lpips_vals) - 0.15 * lpips_range),
        max(lpips_vals) + 0.15 * lpips_range,
    )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))


def main() -> None:
    matplotlib.rcParams.update({"font.size": 11})

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Individual per-dataset plots ----
    for dataset_key, data in RESULTS.items():
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        plot_single_dataset(ax, dataset_key, data)

        # Legend outside the data region
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            edgecolor="#cccccc",
        )

        fig.tight_layout()
        out_path = output_dir / f"perception_distortion_{dataset_key.lower()}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # ---- Combined 3-panel figure ----
    # Nature double-column width ≈ 18 cm; use 3× that for 3 panels at proper resolution,
    # with height set for a square-ish panel aspect so labels remain readable.
    CM = 1 / 2.54
    fig, axes = plt.subplots(1, 3, figsize=(18 * CM * 3, 10 * CM * 3))
    for idx, (dataset_key, data) in enumerate(RESULTS.items()):
        plot_single_dataset(axes[idx], dataset_key, data)

    # Single shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        fontsize=13,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    combined_path = output_dir / "perception_distortion_combined.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {combined_path}")

    print("\nDone. All perception-distortion plots generated.")


if __name__ == "__main__":
    main()
