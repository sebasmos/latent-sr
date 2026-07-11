#!/usr/bin/env python3
"""
TASK R2 -- Patient/subject-level reanalysis to address pseudoreplication.

The manuscript (Table `tab:stats`) reports per-IMAGE Wilcoxon signed-rank
p-values for MedVAE-SR vs. SD-VAE-SR PSNR. For BraTS the 700 "samples" are
20 axial slices from each of 35 subjects, so slices from the same brain are
not independent (pseudoreplication). This script:

  1. Reproduces the image-level paired Wilcoxon (sanity check vs. manuscript).
  2. Aggregates per-image PSNR (and MS-SSIM, LPIPS) to ONE mean value per
     subject/patient, then re-runs the MedVAE-vs-SD-VAE paired test at the
     patient level.
  3. Reports n_patients, patient-level p-value, effect size (Cohen's d_z and
     matched-pairs rank-biserial r), and whether the significance conclusion
     holds.

ID provenance (from per_image_metrics[*]['id']):
  BraTS : 'brats_BraTS-GLI-00021-001_z07'  -> subject = BraTS-GLI-00021-001
          => 35 subjects x 20 slices = 700  (PSEUDOREPLICATION present)
  MRNet : 'sagittal_1130_middle'           -> case 1130, exactly 1 slice/case
          => 120 images == 120 patients     (no aggregation needed)
  CXR   : 'cxr_00000'                       -> running index, NO patient linkage
          => patient IDs NOT recoverable from artifacts

Only PSNR arms that exist as paired per-image files under the *standard*
(non-truehr) evaluation are used, matching the manuscript's Table `tab:stats`.

Runs on CPU in <1s; scipy + numpy only.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_paths import repo_root

BASE = repo_root()

# Paired per-image result files used by the manuscript (Table tab:stats).
PAIRS = {
    "brats": {
        "medvae": BASE / "outputs/experiments/brats_medvae_s1_valid/diffusion_eval_results.json",
        "sdvae":  BASE / "outputs/experiments/brats_sdvae_valid/diffusion_eval_results.json",
        "subject_re": re.compile(r"^brats_(BraTS-GLI-\d+-\d+)_z\d+$"),
    },
    "mrnet": {
        "medvae": BASE / "outputs/step_ablation_s1/T1000/diffusion_eval_results.json",
        "sdvae":  BASE / "outputs/experiments/mrnet_sdvae/diffusion_eval_results.json",
        "subject_re": re.compile(r"^sagittal_(\d+)_middle$"),
    },
    "cxr": {
        "medvae": BASE / "outputs/experiments/cxr_medvae_s1/diffusion_eval_results.json",
        "sdvae":  BASE / "outputs/experiments/cxr_sdvae/diffusion_eval_results.json",
        "subject_re": re.compile(r"^cxr_(\d+)$"),  # index only -- NOT a patient id
    },
}

METRICS = ["psnr", "msssim", "lpips"]
# For these metrics, "MedVAE better" means:
HIGHER_IS_BETTER = {"psnr": True, "msssim": True, "lpips": False}


def load_pim(path):
    if not path.exists():
        return None
    d = json.load(open(path))
    pim = d.get("per_image_metrics", [])
    return pim if pim else None


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation for a paired Wilcoxon.
    r = (sum of ranks of |diff| where diff>0  -  where diff<0) / sum of all ranks.
    Zero differences are dropped (Wilcoxon convention). r in [-1, 1];
    positive => first arm (MedVAE) larger."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    return float((r_plus - r_minus) / ranks.sum())


def paired_report(med, sd, metric, label):
    """med, sd: 1-D arrays aligned pairwise (MedVAE, SD-VAE)."""
    med = np.asarray(med, float)
    sd = np.asarray(sd, float)
    diff = med - sd  # MedVAE - SD-VAE
    n = len(diff)
    higher = HIGHER_IS_BETTER[metric]
    # Direction in which MedVAE is expected to be *better*:
    alt = "greater" if higher else "less"
    try:
        w_two = stats.wilcoxon(med, sd, alternative="two-sided").pvalue
        w_dir = stats.wilcoxon(med, sd, alternative=alt).pvalue
    except ValueError:
        w_two = w_dir = float("nan")
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    # rank-biserial oriented so positive => MedVAE better
    rb = rank_biserial(diff if higher else -diff)
    medvae_better = (diff.mean() > 0) if higher else (diff.mean() < 0)
    return {
        "label": label,
        "metric": metric,
        "n": n,
        "medvae_mean": float(med.mean()),
        "sdvae_mean": float(sd.mean()),
        "delta_medvae_minus_sdvae": float(diff.mean()),
        "medvae_better": bool(medvae_better),
        "p_two_sided": float(w_two),
        "p_directional": float(w_dir),
        "cohens_dz": dz,
        "rank_biserial_r": rb,
    }


def aggregate_by_subject(pim, subject_re, metric):
    """Return dict subject -> mean(metric over that subject's slices)."""
    buckets = defaultdict(list)
    for e in pim:
        m = subject_re.match(e["id"])
        if not m:
            continue
        buckets[m.group(1)].append(e[metric])
    return {s: float(np.mean(v)) for s, v in buckets.items()}


def run_dataset(name, cfg):
    print("=" * 78)
    print(f"DATASET: {name.upper()}")
    print("=" * 78)
    med_pim = load_pim(cfg["medvae"])
    sd_pim = load_pim(cfg["sdvae"])
    out = {"dataset": name}

    if med_pim is None or sd_pim is None:
        # Determine which arm(s) lack per-image data.
        missing = []
        if med_pim is None:
            missing.append(f"MedVAE arm ({cfg['medvae'].name}: no per_image_metrics)")
        if sd_pim is None:
            missing.append(f"SD-VAE arm ({cfg['sdvae'].name}: no per_image_metrics)")
        print("  Cannot recompute paired test from artifacts:")
        for m in missing:
            print("   -", m)
        out["status"] = "paired_per_image_unavailable"
        out["missing"] = missing

    # ---- ID recoverability check ----
    subj_re = cfg["subject_re"]
    if med_pim is not None:
        subjects = {subj_re.match(e["id"]).group(1)
                    for e in med_pim if subj_re.match(e["id"])}
        n_img = len(med_pim)
        n_subj = len(subjects)
        slices_per = n_img / n_subj if n_subj else float("nan")
        out["n_images"] = n_img
        out["n_unique_id_groups"] = n_subj
        out["slices_per_group"] = slices_per
    else:
        n_img = n_subj = None
        slices_per = None

    # CXR: the id is a running index, not a patient/study id.
    if name == "cxr":
        out["patient_ids_recoverable"] = False
        out["reason"] = ("CXR image ids are sequential indices (cxr_00000...); "
                         "labels.csv columns are [filename, 5 pathology labels] "
                         "with NO subject_id/study_id. Original MIMIC-CXR "
                         "dicom_id->subject_id/study_id mapping was discarded "
                         "during preprocessing.")
        print("  Patient IDs NOT recoverable:", out["reason"])
        return out

    # MRNet: id groups == images (1 middle slice per case). No pseudoreplication.
    if name == "mrnet":
        out["patient_ids_recoverable"] = True
        if n_subj is not None:
            print(f"  n_images={n_img}, unique cases={n_subj}, "
                  f"slices/case={slices_per:.2f}")
            out["pseudoreplication"] = (slices_per > 1.0)
            print("  -> Each case contributes exactly one slice: image-level "
                  "test is ALREADY at patient granularity; no aggregation needed.")
        if sd_pim is None:
            print("  -> SD-VAE per-image PSNR absent in the standard eval file, "
                  "so the paired MedVAE-vs-SD-VAE test cannot be recomputed here; "
                  "but pseudoreplication does not apply (1 slice/patient).")
        return out

    # BraTS: real pseudoreplication + both arms available.
    out["patient_ids_recoverable"] = True
    if med_pim is None or sd_pim is None:
        return out

    med_by_id = {e["id"]: e for e in med_pim}
    sd_by_id = {e["id"]: e for e in sd_pim}
    common = [i for i in med_by_id if i in sd_by_id]
    common.sort()
    print(f"  Paired images: {len(common)}  |  subjects: {n_subj}  |  "
          f"slices/subject: {slices_per:.1f}")
    out["pseudoreplication"] = True

    out["image_level"] = {}
    out["patient_level"] = {}
    for metric in METRICS:
        med_img = [med_by_id[i][metric] for i in common]
        sd_img = [sd_by_id[i][metric] for i in common]
        img_rep = paired_report(med_img, sd_img, metric, "image-level")

        med_sub = aggregate_by_subject([med_by_id[i] for i in common], subj_re, metric)
        sd_sub = aggregate_by_subject([sd_by_id[i] for i in common], subj_re, metric)
        subs = sorted(med_sub)
        pat_rep = paired_report([med_sub[s] for s in subs],
                                [sd_sub[s] for s in subs], metric, "patient-level")

        out["image_level"][metric] = img_rep
        out["patient_level"][metric] = pat_rep

        print(f"\n  [{metric.upper()}]  MedVAE {img_rep['medvae_mean']:.3f} vs "
              f"SD-VAE {img_rep['sdvae_mean']:.3f}  (Delta={img_rep['delta_medvae_minus_sdvae']:+.3f})")
        print(f"    image-level  (n={img_rep['n']:3d}): p_two={img_rep['p_two_sided']:.2e} "
              f"p_dir={img_rep['p_directional']:.2e}  dz={img_rep['cohens_dz']:+.2f} "
              f"rrb={img_rep['rank_biserial_r']:+.2f}")
        print(f"    patient-lvl  (n={pat_rep['n']:3d}): p_two={pat_rep['p_two_sided']:.2e} "
              f"p_dir={pat_rep['p_directional']:.2e}  dz={pat_rep['cohens_dz']:+.2f} "
              f"rrb={pat_rep['rank_biserial_r']:+.2f}")

    # Conclusion: PSNR is the manuscript's headline metric.
    psnr_pat = out["patient_level"]["psnr"]
    holds = (psnr_pat["p_two_sided"] < 0.05) and psnr_pat["medvae_better"]
    out["psnr_conclusion_holds_patient_level"] = bool(holds)
    print(f"\n  PSNR conclusion (MedVAE > SD-VAE) holds at patient level: {holds} "
          f"(n_patients={psnr_pat['n']}, p_two={psnr_pat['p_two_sided']:.2e}, "
          f"dz={psnr_pat['cohens_dz']:+.2f})")
    return out


def main():
    results = {}
    for name, cfg in PAIRS.items():
        results[name] = run_dataset(name, cfg)
        print()
    out_dir = BASE / "outputs/statistical_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "patient_level_stats.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
