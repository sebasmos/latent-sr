#!/usr/bin/env python3
"""Compute paired Wilcoxon signed-rank p-values: MedVAE S1 vs SD-VAE per-image PSNR."""

import json
from pathlib import Path
from scipy import stats
import numpy as np

results = {}

datasets = {
    "mrnet": {
        "medvae": [
            "outputs/step_ablation_s1/T1000/diffusion_eval_results.json",  # has per-image
            "outputs/experiments/mrnet_medvae_s1/diffusion_eval_results.json",
        ],
        "sdvae": [
            "outputs/experiments/mrnet_sdvae_v2/diffusion_eval_results.json",  # re-eval with per-image
            "outputs/experiments/mrnet_sdvae/diffusion_eval_results.json",
        ],
    },
    "brats": {
        "medvae": [
            "outputs/experiments/brats_medvae_s1_v2/diffusion_eval_results.json",  # re-eval with per-image
            "outputs/experiments/brats_medvae_s1/diffusion_eval_results.json",
        ],
        "sdvae": [
            "outputs/experiments/brats_sdvae_v2/diffusion_eval_results.json",  # re-eval with per-image
            "outputs/experiments/brats_sdvae/diffusion_eval_results.json",
        ],
    },
    "cxr": {
        "medvae": [
            "outputs/experiments/cxr_medvae_s1/diffusion_eval_results.json",  # has per-image (Sahil's eval)
        ],
        "sdvae": [
            "outputs/experiments/cxr_sdvae_v2/diffusion_eval_results.json",  # re-eval with per-image
            "outputs/experiments/cxr_sdvae/diffusion_eval_results.json",
        ],
    },
}

for name, paths in datasets.items():
    # Find first existing file with per-image metrics
    med_path = None
    for p in paths["medvae"]:
        if Path(p).exists():
            d = json.load(open(p))
            if "per_image_metrics" in d:
                med_path = Path(p)
                break
    sd_path = None
    for p in paths["sdvae"]:
        if Path(p).exists():
            d = json.load(open(p))
            if "per_image_metrics" in d:
                sd_path = Path(p)
                break

    if not med_path or not sd_path:
        print(f"{name}: skipping (missing per-image metrics)")
        print(f"  medvae: {med_path}")
        print(f"  sdvae: {sd_path}")
        continue

    with open(med_path) as f:
        med = json.load(f)
    with open(sd_path) as f:
        sd = json.load(f)

    # Try per-image metrics first (Sahil's format), fall back to top-level
    if "per_image_metrics" in med and "per_image_metrics" in sd:
        med_psnr = [m["psnr"] for m in med["per_image_metrics"]]
        sd_psnr = [m["psnr"] for m in sd["per_image_metrics"]]

        # Match by count (both should have same samples)
        n = min(len(med_psnr), len(sd_psnr))
        med_psnr = med_psnr[:n]
        sd_psnr = sd_psnr[:n]
    else:
        print(f"{name}: no per-image metrics, skipping p-value")
        continue

    stat, pval = stats.wilcoxon(med_psnr, sd_psnr, alternative="greater")

    results[name] = {
        "n_samples": n,
        "medvae_psnr_mean": float(np.mean(med_psnr)),
        "sdvae_psnr_mean": float(np.mean(sd_psnr)),
        "delta_psnr": float(np.mean(med_psnr) - np.mean(sd_psnr)),
        "wilcoxon_statistic": float(stat),
        "p_value": float(pval),
        "significant_p005": pval < 0.005,
        "significant_p001": pval < 0.001,
    }

    print(f"{name}: MedVAE {np.mean(med_psnr):.2f} vs SD-VAE {np.mean(sd_psnr):.2f} dB, "
          f"Δ={np.mean(med_psnr)-np.mean(sd_psnr):+.2f}, p={pval:.2e} (n={n})")

out = Path("outputs/statistical_tests")
out.mkdir(parents=True, exist_ok=True)
with open(out / "pvalues.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out / 'pvalues.json'}")
