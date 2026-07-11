"""
ROI-based PSNR/SSIM analysis for BraTS super-resolution outputs.

Computes PSNR and SSIM separately for:
  - Tumor region (seg_mask > 128)
  - Background region (seg_mask == 0)
  - Whole image

Methods evaluated: MedVAE SR, SD-VAE SR, Bicubic
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_paths import data_root, outputs_root

# ── Paths ─────────────────────────────────────────────────────────────────────
HR_DIR      = data_root() / "brats2023-sr/test/hr"
SEG_DIR     = data_root() / "brats2023-sr/test/seg_masks"
OUT_DIR     = outputs_root() / "experiments/roi_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_EXP = outputs_root() / "experiments"
METHODS = {
    "MedVAE-SR":  _EXP / "brats_medvae_s1/sr_images",
    "SD-VAE-SR":  _EXP / "brats_sdvae/sr_images",
    "Bicubic":    _EXP / "brats_bicubic/sr_images",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gray_float(path: Path) -> np.ndarray:
    """Load image as float32 in [0, 1]."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def psnr_masked(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """PSNR computed only over pixels where mask == True."""
    pixels_gt   = gt[mask]
    pixels_pred = pred[mask]
    mse = np.mean((pixels_gt - pixels_pred) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim_masked(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """
    SSIM estimated over the masked region by zeroing out non-mask pixels
    and computing SSIM on the cropped bounding box of the mask.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Add a small border so SSIM window has enough context
    border = 7
    rmin = max(0, rmin - border)
    rmax = min(gt.shape[0] - 1, rmax + border)
    cmin = max(0, cmin - border)
    cmax = min(gt.shape[1] - 1, cmax + border)

    gt_crop   = gt[rmin:rmax+1, cmin:cmax+1]
    pred_crop = pred[rmin:rmax+1, cmin:cmax+1]
    mask_crop = mask[rmin:rmax+1, cmin:cmax+1]

    # Use skimage SSIM with full=True to get the SSIM map, then average over mask
    _, ssim_map = ssim_fn(gt_crop, pred_crop, data_range=1.0, full=True)
    return float(np.mean(ssim_map[mask_crop]))


def compute_metrics(gt: np.ndarray, pred: np.ndarray, tumor_mask: np.ndarray):
    """Return dict of (psnr/ssim) x (tumor/background/whole)."""
    bg_mask = ~tumor_mask
    has_tumor = tumor_mask.any()

    # Whole image
    mse_whole = np.mean((gt - pred) ** 2)
    psnr_whole = float(10.0 * np.log10(1.0 / mse_whole)) if mse_whole > 0 else float("inf")
    _, ssim_map_whole = ssim_fn(gt, pred, data_range=1.0, full=True)
    ssim_whole = float(np.mean(ssim_map_whole))

    result = {
        "psnr_whole": psnr_whole,
        "ssim_whole": ssim_whole,
        "has_tumor": bool(has_tumor),
    }

    if has_tumor:
        result["psnr_tumor"]      = psnr_masked(gt, pred, tumor_mask)
        result["ssim_tumor"]      = ssim_masked(gt, pred, tumor_mask)

    if bg_mask.any():
        result["psnr_background"] = psnr_masked(gt, pred, bg_mask)
        # Background SSIM: use whole SSIM map averaged over bg pixels
        _, ssim_map_w = ssim_fn(gt, pred, data_range=1.0, full=True)
        result["ssim_background"] = float(np.mean(ssim_map_w[bg_mask]))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def process_method(name: str, sr_dir: Path):
    sr_files = sorted(sr_dir.glob("*.png"))
    print(f"\n[{name}] Found {len(sr_files)} SR images in {sr_dir}")

    records = []
    skipped = 0

    for sr_path in sr_files:
        stem = sr_path.stem  # e.g. brats_BraTS-GLI-00012-000_z00
        hr_path  = HR_DIR  / f"{stem}.png"
        seg_path = SEG_DIR / f"{stem}.png"

        if not hr_path.exists():
            skipped += 1
            continue
        if not seg_path.exists():
            skipped += 1
            continue

        gt   = load_gray_float(hr_path)
        pred = load_gray_float(sr_path)
        seg  = np.array(Image.open(seg_path).convert("L"))
        tumor_mask = seg > 128  # binary: True = tumor

        metrics = compute_metrics(gt, pred, tumor_mask)
        metrics["file"] = stem
        records.append(metrics)

    print(f"  Processed: {len(records)}  |  Skipped (missing): {skipped}")

    # Aggregate
    tumor_records = [r for r in records if r.get("has_tumor")]
    print(f"  Images with non-empty tumor mask: {len(tumor_records)}")

    def mean_of(key):
        vals = [r[key] for r in records if key in r and not np.isinf(r[key])]
        return float(np.mean(vals)) if vals else None

    def mean_of_tumor(key):
        vals = [r[key] for r in tumor_records if key in r and not np.isinf(r[key])]
        return float(np.mean(vals)) if vals else None

    agg = {
        "n_total":       len(records),
        "n_tumor":       len(tumor_records),
        "psnr_whole":    mean_of("psnr_whole"),
        "ssim_whole":    mean_of("ssim_whole"),
        "psnr_tumor":    mean_of_tumor("psnr_tumor"),
        "ssim_tumor":    mean_of_tumor("ssim_tumor"),
        "psnr_background": mean_of("psnr_background"),
        "ssim_background": mean_of("ssim_background"),
    }
    return agg, records


def main():
    all_results = {}
    all_per_image = {}

    for name, sr_dir in METHODS.items():
        if not sr_dir.exists():
            print(f"[{name}] SR directory not found: {sr_dir} — skipping")
            continue
        agg, records = process_method(name, sr_dir)
        all_results[name] = agg
        all_per_image[name] = records

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Method':<14} | {'N-tumor':>7} | {'PSNR-Tumor':>10} | {'PSNR-BG':>9} | {'PSNR-Whole':>10} | {'SSIM-Tumor':>10} | {'SSIM-BG':>9} | {'SSIM-Whole':>10}")
    print("-" * 90)
    for name, agg in all_results.items():
        def fmt(v):
            return f"{v:.3f}" if v is not None else "  N/A  "
        print(
            f"{name:<14} | {agg['n_tumor']:>7} | "
            f"{fmt(agg['psnr_tumor']):>10} | {fmt(agg['psnr_background']):>9} | {fmt(agg['psnr_whole']):>10} | "
            f"{fmt(agg['ssim_tumor']):>10} | {fmt(agg['ssim_background']):>9} | {fmt(agg['ssim_whole']):>10}"
        )
    print("=" * 90)

    # Save JSON
    output = {
        "aggregated": all_results,
        "per_image":  {k: v[:10] for k, v in all_per_image.items()},  # first 10 per method as sample
    }
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
