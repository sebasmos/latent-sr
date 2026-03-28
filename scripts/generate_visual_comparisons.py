#!/usr/bin/env python3
"""
Generate visual comparison figure panels for the SR paper (issue #13).

For each dataset (MRNet, BraTS), creates a figure with 3 representative
sample rows. Each row shows: LR (bicubic-upsampled) | SR (MedVAE) |
SR (SD-VAE) | HR, with a zoomed ROI inset overlaid on each panel.

Columns whose SR directory does not exist or is empty are skipped.

Output: outputs/figures/visual_comparison_{dataset}.png at 300 DPI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "mrnet": {
        "hr_dir": Path("/orcd/pool/006/lceli_shared/DATASET/mrnetkneemris/MRNet-v1.0-middle/valid/hr/"),
        "lr_dir": Path("/orcd/pool/006/lceli_shared/DATASET/mrnetkneemris/MRNet-v1.0-middle/valid/lr/"),
        "sr_medvae_dir": PROJECT_ROOT / "outputs/experiments/mrnet_medvae_s1/sr_images/",
        "sr_sdvae_dir": PROJECT_ROOT / "outputs/experiments/mrnet_sdvae/sr_images/",
    },
    "brats": {
        "hr_dir": Path("/orcd/pool/006/lceli_shared/DATASET/brats2023-sr/test/hr/"),
        "lr_dir": Path("/orcd/pool/006/lceli_shared/DATASET/brats2023-sr/test/lr/"),
        "sr_medvae_dir": PROJECT_ROOT / "outputs/experiments/brats_medvae_s1/sr_images/",
        "sr_sdvae_dir": PROJECT_ROOT / "outputs/experiments/brats_sdvae/sr_images/",
    },
}

OUTPUT_DIR = PROJECT_ROOT / "outputs/figures"
TARGET_SIZE = (256, 256)
ROI_SIZE = 64         # side length of the ROI crop (in HR-space pixels)
ROI_MAG = 3           # magnification factor for the inset
NUM_SAMPLES = 3       # representative samples per dataset
DPI = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dir_is_usable(d: Path) -> bool:
    """Return True if *d* exists and contains at least one .png file."""
    if not d.is_dir():
        return False
    return any(d.glob("*.png"))


def find_common_samples(dirs: list[Path], n: int, prefer_mid_slices: bool = False) -> list[str]:
    """Return up to *n* filenames present in every directory in *dirs*.

    When *prefer_mid_slices* is True (useful for BraTS volumetric data),
    favour slices near the middle of each subject's z-range where there
    is more brain tissue visible.  Otherwise picks evenly-spaced samples.
    """
    sets = [set(p.name for p in d.glob("*.png")) for d in dirs]
    common = sorted(set.intersection(*sets))
    if len(common) == 0:
        return []
    if len(common) <= n:
        return common

    if prefer_mid_slices:
        # Group by subject, pick a mid-z slice from spread-out subjects
        from collections import defaultdict
        subj_slices: dict[str, list[str]] = defaultdict(list)
        for fname in common:
            # expected pattern: brats_<SUBJECT>_z<NN>.png
            stem = Path(fname).stem
            parts = stem.rsplit("_z", 1)
            if len(parts) == 2:
                subj_slices[parts[0]].append(fname)
            else:
                subj_slices[stem].append(fname)
        subjects = sorted(subj_slices.keys())
        # Pick n subjects spread across the list
        subj_indices = np.linspace(0, len(subjects) - 1, min(n, len(subjects)), dtype=int)
        picked = []
        for si in subj_indices:
            slices = sorted(subj_slices[subjects[si]])
            # Pick a slice from the middle of the volume (more tissue)
            mid = len(slices) // 2
            picked.append(slices[mid])
        return picked[:n]
    else:
        # Pick evenly-spaced samples (start, middle, end area)
        indices = np.linspace(0, len(common) - 1, n, dtype=int)
        return [common[i] for i in indices]


def load_gray(path: Path, resize: tuple[int, int] | None = None) -> np.ndarray:
    """Load a grayscale image as a uint8 numpy array, optionally resizing."""
    img = Image.open(path).convert("L")
    if resize is not None and img.size != resize:
        img = img.resize(resize, Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def auto_roi_center(image: np.ndarray, roi_size: int) -> tuple[int, int]:
    """Return (x, y) of a center-crop ROI box.

    Places the ROI at the image center, which is typically the most
    diagnostically relevant region for both brain and knee MRI.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    x = max(0, min(cx - roi_size // 2, w - roi_size))
    y = max(0, min(cy - roi_size // 2, h - roi_size))
    return x, y


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_figure(dataset_name: str, cfg: dict) -> Path | None:
    """Build and save the comparison figure for one dataset.

    Returns the output path on success, or None if not enough data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    hr_dir = cfg["hr_dir"]
    lr_dir = cfg["lr_dir"]
    sr_medvae_dir = cfg["sr_medvae_dir"]
    sr_sdvae_dir = cfg["sr_sdvae_dir"]

    # ------------------------------------------------------------------
    # Determine which columns are available
    # ------------------------------------------------------------------
    # Columns: LR (always), Bicubic (= LR upscaled, always), SR MedVAE, SR SD-VAE, HR (always)
    columns: list[tuple[str, str, Path | None]] = []
    columns.append(("LR", "lr", lr_dir))

    has_medvae = dir_is_usable(sr_medvae_dir)
    has_sdvae = dir_is_usable(sr_sdvae_dir)

    if has_medvae:
        columns.append(("SR (MedVAE)", "sr_medvae", sr_medvae_dir))
    if has_sdvae:
        columns.append(("SR (SD-VAE)", "sr_sdvae", sr_sdvae_dir))

    columns.append(("HR", "hr", hr_dir))

    # ------------------------------------------------------------------
    # Find common samples across all available directories
    # ------------------------------------------------------------------
    all_dirs = [d for (_, _, d) in columns if d is not None]
    prefer_mid = dataset_name.lower() == "brats"
    samples = find_common_samples(all_dirs, NUM_SAMPLES, prefer_mid_slices=prefer_mid)
    if not samples:
        print(f"[{dataset_name}] No common samples found -- skipping.")
        return None

    print(f"[{dataset_name}] Using {len(samples)} samples: {samples}")
    print(f"[{dataset_name}] Columns: {[c[0] for c in columns]}")

    n_rows = len(samples)
    n_cols = len(columns)

    # Each panel: image on top, zoomed inset strip below
    # We use a 2-row-per-sample layout: top row = full image, bottom row = zoomed ROI
    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(n_cols * 2.8, n_rows * 3.6),
        gridspec_kw={"height_ratios": [3, 1] * n_rows, "hspace": 0.08, "wspace": 0.04},
    )
    # Ensure axes is 2-D even when n_cols == 1
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for row_idx, fname in enumerate(samples):
        # Determine ROI from the HR image
        hr_img = load_gray(hr_dir / fname, resize=TARGET_SIZE)
        roi_x, roi_y = auto_roi_center(hr_img, ROI_SIZE)

        for col_idx, (col_title, col_key, col_dir) in enumerate(columns):
            img_path = col_dir / fname
            if not img_path.exists():
                print(f"  WARNING: missing {img_path}, leaving blank")
                axes[row_idx * 2, col_idx].axis("off")
                axes[row_idx * 2 + 1, col_idx].axis("off")
                continue

            img = load_gray(img_path, resize=TARGET_SIZE)

            # ---------- Main panel ----------
            ax_main = axes[row_idx * 2, col_idx]
            ax_main.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="none")
            # Draw ROI rectangle on the main panel
            rect = Rectangle(
                (roi_x, roi_y), ROI_SIZE, ROI_SIZE,
                linewidth=1.5, edgecolor="#e63946", facecolor="none",
            )
            ax_main.add_patch(rect)
            ax_main.set_xticks([])
            ax_main.set_yticks([])

            # Column title only on first row
            if row_idx == 0:
                ax_main.set_title(col_title, fontsize=11, fontweight="bold", pad=6)

            # Row label on the leftmost column
            if col_idx == 0:
                stem = Path(fname).stem
                # Shorten long BraTS names
                if len(stem) > 25:
                    stem = stem[:12] + "..." + stem[-10:]
                ax_main.set_ylabel(stem, fontsize=7, rotation=90, labelpad=8)

            # ---------- Zoomed inset panel ----------
            ax_zoom = axes[row_idx * 2 + 1, col_idx]
            crop = img[roi_y : roi_y + ROI_SIZE, roi_x : roi_x + ROI_SIZE]
            # Magnify via nearest-neighbor for a crisp zoom
            crop_mag = np.kron(crop, np.ones((ROI_MAG, ROI_MAG), dtype=np.uint8))
            ax_zoom.imshow(crop_mag, cmap="gray", vmin=0, vmax=255, interpolation="none")
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])
            for spine in ax_zoom.spines.values():
                spine.set_edgecolor("#e63946")
                spine.set_linewidth(1.5)

    # Suptitle
    arrow = " \u2192 "
    col_names = [c[0] for c in columns]
    fig.suptitle(
        f"{dataset_name.upper()}: {arrow.join(col_names)}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.02)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"visual_comparison_{dataset_name}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[{dataset_name}] Saved figure to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = {}
    for name, cfg in DATASETS.items():
        out = render_figure(name, cfg)
        if out is not None:
            results[name] = str(out)

    print("\n=== Summary ===")
    if not results:
        print("No figures generated (missing data).")
        sys.exit(1)
    for name, path in results.items():
        print(f"  {name}: {path}")
    print("Done.")


if __name__ == "__main__":
    main()
