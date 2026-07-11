#!/usr/bin/env python3
"""
Recompute the AE-ceiling -> SR-PSNR headline regression under a SINGLE, consistent
metric: the TRUE-HR metric, for the expanded set of 5 VAE geometries x 3 datasets = 15 points.

Motivation (TASK R4):
  The lead correlation was previously reported as n=15, r=0.73 computed under the
  SELF-REFERENTIAL SR metric [PSNR(SR, decode(hr_lat))], while the figure showed
  n=6, r=0.87 under the TRUE-HR metric [PSNR(SR, true_HR_png)]. The primary number
  and the figure therefore used different metrics. This script recomputes the n=15
  regression entirely under the TRUE-HR metric so both use the same metric.

Data source (all under outputs/experiments/):
  For each (dataset, geometry) the file
      truehr_<dataset>_<geometry>/diffusion_eval_truehr_results.json
  produced by eval_diffusion_sr_truehr.py contains, computed on the SAME samples
  against the SAME true-HR PNG ground truth:
    - diffusion_sr_truehr.psnr_mean               -> true-HR SR PSNR  (y)
    - ae_ceiling_inline_forcomparison.psnr_mean   -> true-HR AE-ceiling PSNR (x)
      = PSNR(decode(hr_lat), true_HR_png), i.e. the VAE reconstruction ceiling
      measured against the real image (the AE-ceiling true-HR value).

  5 geometries: klf4, medvae_4_1, medvae_4_3, medvae_8_4, sdvae
  3 datasets:   brats, cxr, mrnet

No numbers are hard-coded; everything is read from the JSONs at runtime. Missing
points are reported explicitly and n is labeled with whatever is available.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_paths import outputs_root

EXP_DIR = outputs_root() / "experiments"

DATASETS = ["mrnet", "brats", "cxr"]
GEOMETRIES = ["sdvae", "medvae_4_1", "medvae_4_3", "medvae_8_4", "klf4"]

# For reference: prior reported comparators (self-referential n=15, and true-HR n=6).
PRIOR_SELFREF_N15_R = 0.73
PRIOR_TRUEHR_N6_R = 0.87


def load_point(dataset: str, geometry: str):
    """Return (ae_ceiling_truehr, sr_truehr, n_samples, sanity_passed) or None if missing."""
    jpath = EXP_DIR / f"truehr_{dataset}_{geometry}" / "diffusion_eval_truehr_results.json"
    if not jpath.exists():
        return None
    with open(jpath) as fh:
        r = json.load(fh)
    try:
        sr = float(r["diffusion_sr_truehr"]["psnr_mean"])
        ae = float(r["ae_ceiling_inline_forcomparison"]["psnr_mean"])
    except (KeyError, TypeError):
        return None
    if sr is None or ae is None:
        return None
    n = int(r.get("n_samples", -1))
    passed = bool(r.get("sanity_check", {}).get("passed", False))
    return ae, sr, n, passed, str(jpath)


def fisher_z_ci(r: float, n: int, alpha: float = 0.05):
    """Two-sided (1-alpha) CI on Pearson r via the Fisher z transform."""
    if n < 4:
        return None, None
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = z - zcrit * se, z + zcrit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def main() -> None:
    rows = []
    missing = []
    not_passed = []
    for ds in DATASETS:
        for geo in GEOMETRIES:
            pt = load_point(ds, geo)
            if pt is None:
                missing.append((ds, geo))
                continue
            ae, sr, n, passed, jpath = pt
            if not passed:
                not_passed.append((ds, geo))
            rows.append((ds, geo, ae, sr, n, passed))

    print("=" * 78)
    print("TRUE-HR AE-ceiling -> SR-PSNR regression (expanded 5 geometries x 3 datasets)")
    print("=" * 78)
    print(f"{'dataset':<7} {'geometry':<12} {'AE_ceiling(HR)':>14} {'SR(HR)':>9} {'n_samp':>7} {'sanity':>7}")
    for ds, geo, ae, sr, n, passed in rows:
        print(f"{ds:<7} {geo:<12} {ae:>14.3f} {sr:>9.3f} {n:>7d} {str(passed):>7}")

    if missing:
        print("\nMISSING points (no JSON / no metric):")
        for ds, geo in missing:
            print(f"  - {ds} / {geo}")
    else:
        print("\nMISSING points: none — all 15 present.")

    if not_passed:
        print("\nWARNING: sanity check NOT passed for:")
        for ds, geo in not_passed:
            print(f"  - {ds} / {geo}")

    ae_arr = np.array([r[2] for r in rows])
    sr_arr = np.array([r[3] for r in rows])
    n_points = len(rows)

    print("\n" + "-" * 78)
    print(f"n (points used) = {n_points}")

    if n_points < 3:
        print("Too few points to regress.")
        return

    r_val, p_val = pearsonr(ae_arr, sr_arr)
    r2_val = r_val ** 2
    ci_lo, ci_hi = fisher_z_ci(r_val, n_points)

    # OLS slope/intercept for completeness
    slope, intercept = np.polyfit(ae_arr, sr_arr, 1)

    print(f"Pearson r      = {r_val:.4f}")
    print(f"R^2            = {r2_val:.4f}")
    print(f"two-sided p    = {p_val:.4g}")
    if ci_lo is not None:
        print(f"Fisher-z 95% CI on r = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"OLS: SR = {slope:.4f} * AE_ceiling + {intercept:.4f}")

    print("\n" + "-" * 78)
    print("Comparison to prior reported values")
    print("-" * 78)
    print(f"  self-referential n=15 : r = {PRIOR_SELFREF_N15_R:.2f}")
    print(f"  true-HR          n=6  : r = {PRIOR_TRUEHR_N6_R:.2f}")
    print(f"  true-HR          n={n_points} : r = {r_val:.2f}  (this recompute, metric-consistent)")

    print("\nFor LaTeX:  n = %d,  r = %.2f,  R^2 = %.2f,  p = %s,  95%% CI = [%.2f, %.2f]"
          % (n_points, r_val, r2_val,
             (f"{p_val:.3f}" if p_val >= 1e-3 else f"{p_val:.2e}"),
             ci_lo, ci_hi))


if __name__ == "__main__":
    main()
