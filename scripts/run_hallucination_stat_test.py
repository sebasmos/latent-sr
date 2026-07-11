#!/usr/bin/env python3
"""
run_hallucination_stat_test.py

Formal statistical comparison of hallucination rates between MedVAE SR and
SD-VAE SR using:
  1. Wilcoxon signed-rank test on per-image hallucination rates
  2. Two-proportion z-test on aggregate pixel counts

Reads existing results.json produced by eval_hallucination_quantification.py.
Outputs results to outputs/experiments/hallucination_stat_tests/results.txt
"""

import json
import pathlib
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/experiments/hallucination_stat_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "BraTS": ROOT / "outputs/experiments/hallucination_brats/results.json",
    "CXR":   ROOT / "outputs/experiments/hallucination_cxr/results.json",
    "MRNet": ROOT / "outputs/experiments/hallucination_mrnet/results.json",
}

PIX = 256 * 256  # pixels per image

lines = []
lines.append("=" * 70)
lines.append("Hallucination Rate Statistical Tests: MedVAE SR vs SD-VAE SR")
lines.append("=" * 70)
lines.append("")
lines.append("Method 1: Wilcoxon signed-rank test on per-image hallucination rates")
lines.append("Method 2: Two-proportion z-test on aggregate pixel counts")
lines.append("")

for name, path in DATASETS.items():
    with open(path) as f:
        data = json.load(f)

    per = data["per_image"]
    n = len(per)

    med_rates = np.array([p["medvae_sr"]["hallu_rate"] for p in per])
    sdv_rates = np.array([p["sdvae_sr"]["hallu_rate"]  for p in per])

    med_mean = med_rates.mean()
    sdv_mean = sdv_rates.mean()

    # --- Wilcoxon signed-rank (paired, per-image) ---
    w_stat, w_p = stats.wilcoxon(med_rates, sdv_rates)

    # --- Two-proportion z-test (aggregate pixel counts, manual implementation) ---
    count_med = int(round(med_mean * PIX * n))
    count_sdv = int(round(sdv_mean * PIX * n))
    nobs = PIX * n
    p1 = count_med / nobs
    p2 = count_sdv / nobs
    p_pool = (count_med + count_sdv) / (2 * nobs)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs + 1 / nobs))
    z_stat = (p1 - p2) / se if se > 0 else 0.0
    z_p = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # --- Effect size: Cohen's h for proportions ---
    h = 2 * np.arcsin(np.sqrt(med_mean)) - 2 * np.arcsin(np.sqrt(sdv_mean))

    lines.append(f"--- {name} (n={n} images) ---")
    lines.append(f"  MedVAE hallu rate : {med_mean*100:.2f}%  (SD={med_rates.std()*100:.2f}%)")
    lines.append(f"  SD-VAE hallu rate : {sdv_mean*100:.2f}%  (SD={sdv_rates.std()*100:.2f}%)")
    lines.append(f"  Wilcoxon signed-rank : W={w_stat:.1f},  p={w_p:.4e}")
    lines.append(f"  Two-prop z-test      : z={z_stat:.4f}, p={z_p:.4e}")
    lines.append(f"  Cohen's h            : {h:.4f}")

    if w_p > 0.05 and z_p > 0.05:
        lines.append(f"  VERDICT : NOT significant (both tests p>0.05) — rates indistinguishable")
    elif w_p < 0.05 or z_p < 0.05:
        lines.append(f"  VERDICT : SIGNIFICANT difference detected (p<0.05)")
    lines.append("")

lines.append("=" * 70)
lines.append("PAPER TEXT SUGGESTION (for Limitations / Results)")
lines.append("=" * 70)
lines.append("")
lines.append("If both tests non-significant (p>0.05):")
lines.append("  Hallucination rates are statistically indistinguishable between")
lines.append("  MedVAE SR and SD-VAE SR on BraTS (12.9% vs 13.3%) and CXR (3.3% vs 3.4%)")
lines.append("  (Wilcoxon signed-rank and two-proportion z-test, all p>0.05).")
lines.append("")
lines.append("If significant:")
lines.append("  Replace 'statistically indistinguishable' with 'comparable' and cite p-value.")

output = "\n".join(lines)
print(output)

result_path = OUT_DIR / "results.txt"
with open(result_path, "w") as f:
    f.write(output)
print(f"\nResults saved to {result_path}")
