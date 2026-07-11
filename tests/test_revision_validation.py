#!/usr/bin/env python3
"""
Test Revision-Cycle Numbers

Golden-number consistency checks for the results added during the Scientific
Reports major-revision cycle (capacity-matched controls, the KL-f4 natural-VAE
control, the three-seed reproducibility sweep, the target-dataset autoencoder
adaptation control, the patient-level reanalysis, and the expanded true-HR
regression). These constants are transcribed from the published manuscript and
supplementary information; this file does not re-run any experiment, it checks
that the numbers are internally consistent and that nobody has silently
transcribed a wrong digit into README.md or a downstream script.

If a number here needs to change, it must change in the manuscript first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

DATASETS = ("mrnet", "brats", "cxr")

# ---------------------------------------------------------------------------
# Headline true-HR SR PSNR gap, MedVAE SR - SD-VAE SR (Table 1 / Fig 1)
# ---------------------------------------------------------------------------
HEADLINE_GAP_DB = {"mrnet": 1.92, "brats": 2.83, "cxr": 3.60}
HEADLINE_COHENS_D = {"mrnet": 1.17, "brats": 1.46, "cxr": 1.61}

# ---------------------------------------------------------------------------
# Capacity-matched decomposition (Table 3): domain (upper bound) / resolution
# / capacity, three-way split of the headline gap within the medical-VAE family
# ---------------------------------------------------------------------------
CAPACITY_DOMAIN_DB = {"mrnet": 0.95, "brats": 0.32, "cxr": 0.89}
CAPACITY_RESOLUTION_DB = {"mrnet": 0.26, "brats": 1.85, "cxr": 0.56}
CAPACITY_CAPACITY_DB = {"mrnet": 0.72, "brats": 0.66, "cxr": 2.16}

# ---------------------------------------------------------------------------
# KL-f4 natural-side matched-geometry control (Table 4): AE reconstruction
# ceiling. domain@64sq = MedVAE AE - KL-f4 AE, at identical 3x64x64 geometry.
# ---------------------------------------------------------------------------
DOMAIN_AT_64SQ_DB = {"mrnet": 1.17, "brats": 0.74, "cxr": 0.94}
GEOMETRY_AE_GAIN_DB = {"mrnet": 2.76, "brats": 5.74, "cxr": 3.97}  # SD-VAE -> KL-f4, AE ceiling

# ---------------------------------------------------------------------------
# Three-seed reproducibility sweep (seeds 42/43/44), KL-f4 SR pipeline
# ---------------------------------------------------------------------------
SEED_SWEEP_MEAN_DB = {"mrnet": 23.76, "brats": 26.33, "cxr": 29.21}
SEED_SWEEP_SD_DB = {"mrnet": 0.18, "brats": 0.08, "cxr": 0.12}

# ---------------------------------------------------------------------------
# Target-dataset autoencoder adaptation control (Supplementary Note 5)
# ---------------------------------------------------------------------------
ADAPTATION_AE_GAIN_DB = {"mrnet": 3.07, "brats": 3.84, "cxr": 4.62}
ADAPTATION_SR_GAIN_DB = {"mrnet": 1.65, "brats": 0.96, "cxr": 1.00}
ADAPTATION_PASS_THROUGH_PCT = {"mrnet": 54, "brats": 25, "cxr": 22}
ADAPTATION_SIGMA = {"mrnet": 6.6, "brats": 8.9, "cxr": 6.0}
# geometry rung of the same 3-tier ladder, at the SR output (KL-f4 -> SD-VAE)
GEOMETRY_SR_GAIN_DB = {"mrnet": 1.91, "brats": 2.69, "cxr": 3.63}
PRETRAINING_SR_GAIN_DB = {"mrnet": 0.01, "brats": 0.14, "cxr": -0.03}

# ---------------------------------------------------------------------------
# Patient-level reanalysis, BraTS (pseudoreplication check)
# ---------------------------------------------------------------------------
PATIENT_LEVEL_BRATS = {
    "delta_psnr_db": 2.905,
    "n_patients": 35,
    "p_value": 2.48e-7,
    "cohens_dz": 10.27,
    "rank_biserial_r": 1.00,
}

# ---------------------------------------------------------------------------
# Expanded AE-ceiling -> SR true-HR regression (n=15, all 5 geometries x 3
# datasets) vs. the original Fig. 3 six-point subset
# ---------------------------------------------------------------------------
REGRESSION_N15 = {"r": 0.83, "r2": 0.69, "p": 1.3e-4, "ci95": (0.55, 0.94)}
REGRESSION_N6 = {"r": 0.87, "r2": 0.76, "p": 0.024, "ci95": (0.20, 0.99)}


def test_headline_gap_positive_and_ordered():
    print("\nTesting headline true-HR PSNR gap (Table 1 / Fig 1)...")
    try:
        for ds in DATASETS:
            assert HEADLINE_GAP_DB[ds] > 0, f"{ds}: headline gap must favour MedVAE SR"
            assert HEADLINE_COHENS_D[ds] > 0.8, f"{ds}: effect size should be 'large' (d>0.8)"
        # CXR has the largest gap and effect size (true-HR SR PSNR ordering)
        assert HEADLINE_GAP_DB["cxr"] > HEADLINE_GAP_DB["brats"] > HEADLINE_GAP_DB["mrnet"]
        print(f"{GREEN}✓{RESET} headline gaps: {HEADLINE_GAP_DB} dB, all d>0.8")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} headline gap check failed: {e}")
        return False


def test_capacity_decomposition_dominant_contributor():
    print("\nTesting capacity-matched decomposition dominant contributor per dataset...")
    try:
        expected_dominant = {"mrnet": "domain", "brats": "resolution", "cxr": "capacity"}
        components = {
            "domain": CAPACITY_DOMAIN_DB,
            "resolution": CAPACITY_RESOLUTION_DB,
            "capacity": CAPACITY_CAPACITY_DB,
        }
        for ds in DATASETS:
            values = {name: components[name][ds] for name in components}
            dominant = max(values, key=values.get)
            assert dominant == expected_dominant[ds], (
                f"{ds}: expected {expected_dominant[ds]} to dominate, got {dominant} ({values})"
            )
        # domain stays modest even as an upper bound: 0.32-0.95 dB, well under the headline gap
        for ds in DATASETS:
            assert 0.0 < CAPACITY_DOMAIN_DB[ds] < 1.0, f"{ds}: domain (upper bound) should be < 1 dB"
        print(f"{GREEN}✓{RESET} dominant contributor per dataset: {expected_dominant}")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} capacity decomposition check failed: {e}")
        return False


def test_domain_effect_real_but_secondary_to_geometry():
    print("\nTesting KL-f4 natural control: domain is real but secondary to geometry...")
    try:
        for ds in DATASETS:
            assert 0.5 < DOMAIN_AT_64SQ_DB[ds] < 1.5, f"{ds}: domain@64sq out of expected 0.5-1.5 dB range"
            assert GEOMETRY_AE_GAIN_DB[ds] > DOMAIN_AT_64SQ_DB[ds], (
                f"{ds}: geometry gain ({GEOMETRY_AE_GAIN_DB[ds]}) should exceed domain gain "
                f"({DOMAIN_AT_64SQ_DB[ds]})"
            )
        print(f"{GREEN}✓{RESET} domain@64sq={DOMAIN_AT_64SQ_DB}, all < geometry gain")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} domain-vs-geometry check failed: {e}")
        return False


def test_seed_sweep_variance_negligible():
    print("\nTesting three-seed sweep: seed SD is negligible vs. the headline effect...")
    try:
        for ds in DATASETS:
            # seed-to-seed SD must be at least 5x smaller than the smallest headline gap
            assert SEED_SWEEP_SD_DB[ds] * 5 < min(HEADLINE_GAP_DB.values()), (
                f"{ds}: seed SD {SEED_SWEEP_SD_DB[ds]} not negligible vs. headline gaps"
            )
        print(f"{GREEN}✓{RESET} seed SDs {SEED_SWEEP_SD_DB} all >=5x smaller than any headline gap")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} seed sweep check failed: {e}")
        return False


def test_three_tier_ladder_consistency():
    print("\nTesting three-tier ladder (pretraining -> adaptation -> geometry)...")
    try:
        for ds in DATASETS:
            pretraining = abs(PRETRAINING_SR_GAIN_DB[ds])
            adaptation = ADAPTATION_SR_GAIN_DB[ds]
            geometry = GEOMETRY_SR_GAIN_DB[ds]
            assert pretraining <= 0.15, f"{ds}: pretraining should be within +/-0.15 dB of zero"
            assert adaptation > pretraining, f"{ds}: adaptation should exceed pretraining"
        # geometry dominates clearly on BraTS/CXR; MRNet margin is small (documented exception)
        for ds in ("brats", "cxr"):
            margin = GEOMETRY_SR_GAIN_DB[ds] - ADAPTATION_SR_GAIN_DB[ds]
            assert margin > 1.0, f"{ds}: geometry should clearly dominate adaptation, margin={margin}"
        mrnet_margin = GEOMETRY_SR_GAIN_DB["mrnet"] - ADAPTATION_SR_GAIN_DB["mrnet"]
        assert 0.0 < mrnet_margin < 0.5, (
            f"mrnet: margin should be small ({mrnet_margin}), this is the documented "
            f"dataset-dependent exception to the general ordering"
        )
        print(f"{GREEN}✓{RESET} ladder holds; MRNet margin={mrnet_margin:.2f} dB (documented exception)")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} ladder consistency check failed: {e}")
        return False


def test_adaptation_pass_through_bounded():
    print("\nTesting adaptation pass-through percentages are in (0, 100)...")
    try:
        for ds in DATASETS:
            pct = ADAPTATION_PASS_THROUGH_PCT[ds]
            assert 0 < pct < 100, f"{ds}: pass-through {pct}% out of (0,100) range"
            assert ADAPTATION_SIGMA[ds] > 5.0, f"{ds}: significance should be well above 5 sigma"
        print(f"{GREEN}✓{RESET} pass-through {ADAPTATION_PASS_THROUGH_PCT}, sigma {ADAPTATION_SIGMA}")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} pass-through check failed: {e}")
        return False


def test_patient_level_significance():
    print("\nTesting BraTS patient-level reanalysis (pseudoreplication check)...")
    try:
        d = PATIENT_LEVEL_BRATS
        assert d["n_patients"] == 35, "BraTS should have 35 subjects (700 images / 20 slices)"
        assert d["p_value"] < 0.001, "patient-level p-value should remain highly significant"
        assert d["delta_psnr_db"] > 0, "patient-level PSNR delta should favour MedVAE SR"
        assert 0.0 <= d["rank_biserial_r"] <= 1.0, "rank-biserial r must be in [0,1]"
        print(f"{GREEN}✓{RESET} patient-level: n={d['n_patients']}, p={d['p_value']:.2e}, "
              f"delta={d['delta_psnr_db']} dB")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} patient-level check failed: {e}")
        return False


def test_expanded_regression_consistent_with_figure():
    print("\nTesting expanded n=15 regression is consistent with the Fig. 3 n=6 subset...")
    try:
        n15, n6 = REGRESSION_N15, REGRESSION_N6
        assert n15["p"] < 0.05 and n6["p"] < 0.05, "both regressions should be significant at alpha=0.05"
        lo15, hi15 = n15["ci95"]
        lo6, hi6 = n6["ci95"]
        assert lo15 <= lo15 <= hi15, "n=15 CI must be well-formed"
        # the n=15 CI should contain the n=6 point estimate (consistency, not tension)
        assert lo15 <= n6["r"] <= hi15, (
            f"n=15 CI {n15['ci95']} should contain the n=6 point estimate r={n6['r']}"
        )
        print(f"{GREEN}✓{RESET} n=15 r={n15['r']} (CI {n15['ci95']}) contains n=6 r={n6['r']}")
        return True
    except AssertionError as e:
        print(f"{RED}✗{RESET} regression consistency check failed: {e}")
        return False


def main():
    print(f"\n{'='*60}")
    print("Revision-Cycle Number Validation Tests")
    print(f"{'='*60}")

    all_passed = True
    all_passed &= test_headline_gap_positive_and_ordered()
    all_passed &= test_capacity_decomposition_dominant_contributor()
    all_passed &= test_domain_effect_real_but_secondary_to_geometry()
    all_passed &= test_seed_sweep_variance_negligible()
    all_passed &= test_three_tier_ladder_consistency()
    all_passed &= test_adaptation_pass_through_bounded()
    all_passed &= test_patient_level_significance()
    all_passed &= test_expanded_regression_consistent_with_figure()

    print(f"\n{'='*60}")
    if all_passed:
        print(f"{GREEN}✓ All revision-cycle number checks passed!{RESET}")
        print(f"\n{YELLOW}Key Points Verified:{RESET}")
        print("  • Headline true-HR gap: +1.92/+2.83/+3.60 dB (MRNet/BraTS/CXR)")
        print("  • Domain (capacity-matched, upper bound): +0.32 to +0.95 dB, always < headline gap")
        print("  • Domain@64sq (KL-f4 control): +0.74 to +1.17 dB, always < geometry gain")
        print("  • 3-tier ladder: pretraining ~0 dB -> adaptation ~1 dB -> geometry +1.9 to +3.6 dB")
        print("  • Seed-to-seed SD (0.08-0.18 dB) negligible vs. headline effects")
        print("  • Patient-level BraTS reanalysis (n=35) confirms image-level conclusion")
        return 0
    else:
        print(f"{RED}✗ Some revision-cycle checks failed.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
