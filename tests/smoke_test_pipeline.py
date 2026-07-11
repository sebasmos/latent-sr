#!/usr/bin/env python3
"""
Smoke tests for the multi-dataset SR pipeline.

Tests each component independently without GPU:
1. Dataset prep scripts (MIMIC-CXR, BraTS)
2. Diffusion model (x0 and eps modes) — forward pass + sampling on CPU
3. YAML config loading
4. Latent dataset loading
5. Eval script imports
6. SLURM script structure

Run: python tests/smoke_test_pipeline.py
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def _load_train_module():
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "train_03",
        str(REPO_ROOT / "medvae_diffusion_pipeline" / "scripts" / "03_train_diffusion.py"),
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diffusion_model():
    print("\n=== Diffusion Model ===")
    import torch
    mod = _load_train_module()

    # x0 mode
    try:
        m = mod.LatentDiffusionSR(
            timesteps=10, hr_channels=3, hr_size=8, lr_channels=3, lr_size=2, prediction_mode="x0",
        )
        m.eval()
        lr = torch.randn(2, 3, 2, 2)
        lr_up = m._prepare_lr_condition(lr)
        report("x0 LR interpolation", lr_up.shape == (2, 3, 8, 8), f"shape={tuple(lr_up.shape)}")

        inp = torch.cat([torch.randn(2, 3, 8, 8), lr_up], dim=1)
        out = m.model(inp, torch.zeros(2, dtype=torch.long)).sample
        report("x0 U-Net forward", out.shape == (2, 3, 8, 8), f"shape={tuple(out.shape)}")
    except Exception as e:
        report("x0 forward", False, str(e))

    # eps mode
    try:
        m_eps = mod.LatentDiffusionSR(
            timesteps=10, hr_channels=3, hr_size=8, lr_channels=3, lr_size=2, prediction_mode="eps",
        )
        report("eps model creation", True)
    except Exception as e:
        report("eps model creation", False, str(e))

    # x0 sampling
    try:
        with torch.no_grad():
            sr = m.sample(lr, T=5)
        report("x0 sampling (T=5)", sr.shape == (2, 3, 8, 8), f"shape={tuple(sr.shape)}")
    except Exception as e:
        report("x0 sampling", False, str(e))

    # Schedule
    try:
        ac = m.alphas_cumprod
        assert ac.shape[0] == 10
        assert float(ac[0]) > float(ac[-1])
        report("noise schedule", True, f"alphas[0]={ac[0]:.4f} > alphas[-1]={ac[-1]:.4f}")
    except Exception as e:
        report("noise schedule", False, str(e))

    # prediction_mode stored
    report("prediction_mode x0", m.prediction_mode == "x0")
    report("prediction_mode eps", m_eps.prediction_mode == "eps")


def test_paired_latent_dataset():
    print("\n=== PairedLatentDataset ===")
    mod = _load_train_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            np.save(os.path.join(tmpdir, f"hr_{i}.npy"), np.random.randn(3, 8, 8).astype(np.float32))
            np.save(os.path.join(tmpdir, f"lr_{i}.npy"), np.random.randn(3, 2, 2).astype(np.float32))

        try:
            ds = mod.PairedLatentDataset(tmpdir)
            report("dataset len", len(ds) == 5)
            sample = ds[0]
            report("dataset getitem", sample["hr"].shape == (3, 8, 8))
            hr_s, lr_s = ds.shape_info()
            report("shape_info", hr_s == (3, 8, 8) and lr_s == (3, 2, 2))
        except Exception as e:
            report("dataset", False, str(e))

        # Mismatch
        np.save(os.path.join(tmpdir, "hr_extra.npy"), np.random.randn(3, 8, 8).astype(np.float32))
        try:
            mod.PairedLatentDataset(tmpdir)
            report("mismatch detection", False, "should have raised")
        except ValueError:
            report("mismatch detection", True)


def test_yaml_configs():
    print("\n=== YAML Configs ===")
    try:
        import yaml
    except ImportError:
        report("yaml import", False, "pyyaml not installed")
        return

    config_dir = REPO_ROOT / "configs"
    report("configs/ exists", config_dir.exists())

    for yf in sorted(config_dir.rglob("*.yaml")):
        try:
            cfg = yaml.safe_load(yf.read_text())
            assert isinstance(cfg, dict)
            report(f"parse {yf.relative_to(REPO_ROOT)}", True)
        except Exception as e:
            report(f"parse {yf.relative_to(REPO_ROOT)}", False, str(e))

    expected = [
        "configs/base.yaml",
        "configs/datasets/mrnet.yaml",
        "configs/datasets/mimic_cxr.yaml",
        "configs/datasets/brats.yaml",
        "configs/experiments/mrnet_medvae.yaml",
        "configs/experiments/mrnet_sdvae.yaml",
        "configs/experiments/cxr_medvae.yaml",
        "configs/experiments/cxr_sdvae.yaml",
        "configs/experiments/brats_medvae.yaml",
        "configs/experiments/brats_sdvae.yaml",
    ]
    for p in expected:
        report(f"exists {p}", (REPO_ROOT / p).exists())


def test_mimic_cxr_prep():
    print("\n=== MIMIC-CXR Prep ===")
    from scripts.prepare_mimic_cxr_sr import load_chexpert_labels, sample_balanced, TARGET_PATHOLOGIES

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "study_id"] + TARGET_PATHOLOGIES + ["No Finding"])
        for i in range(100):
            if i < 50:
                writer.writerow([str(10000000+i), str(50000000+i), "1.0", "0.0", "", "", "0.0", ""])
            else:
                writer.writerow([str(10000000+i), str(50000000+i), "", "", "", "", "", "1.0"])
        csv_path = f.name

    try:
        rows = load_chexpert_labels(csv_path)
        report("load labels", len(rows) == 100, f"n={len(rows)}")

        pos = sum(1 for r in rows if r["has_positive"])
        neg = sum(1 for r in rows if r["no_finding"])
        report("label parsing", pos == 50 and neg == 50, f"pos={pos} neg={neg}")

        sampled = sample_balanced(rows, 20, seed=42)
        report("sample_balanced", len(sampled) == 20)

        pos_s = sum(1 for r in sampled if r["has_positive"])
        report("balanced ratio", 8 <= pos_s <= 12, f"pos_in_sample={pos_s}")
    except Exception as e:
        report("mimic prep", False, str(e))
    finally:
        os.unlink(csv_path)


def test_brats_prep():
    print("\n=== BraTS Prep ===")
    try:
        from scripts.prepare_brats_sr import normalize_slice, extract_slices
        report("import", True)
    except Exception as e:
        report("import", False, str(e))
        return

    arr = np.random.randn(64, 64).astype(np.float32) * 100 + 50
    normed = normalize_slice(arr)
    report("normalize_slice", normed.dtype == np.uint8 and normed.min() == 0 and normed.max() == 255)

    const = np.ones((64, 64), dtype=np.float32) * 42
    report("normalize constant", np.all(normalize_slice(const) == 0))

    vol = np.random.randn(128, 128, 40).astype(np.float32)
    vol[:, :, 10:30] += 5
    seg = np.zeros_like(vol)
    seg[50:70, 50:70, 15:25] = 1
    slices, masks = extract_slices(vol, seg, [0.25, 0.75], 5, 42)
    report("extract_slices", len(slices) == 5 and len(masks) == 5)
    report("mask shape", masks[0].shape == (128, 128))


def test_eval_imports():
    print("\n=== Eval Imports ===")
    try:
        from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
        report("torchmetrics", True)
    except Exception as e:
        report("torchmetrics", False, str(e))

    report("eval script exists", (REPO_ROOT / "scripts" / "eval_diffusion_sr.py").exists())


def test_slurm_scripts():
    print("\n=== SLURM Scripts ===")
    scripts = [
        "slurm/run_experiment.sh",
        "slurm/run_all_experiments.sh",
        "slurm/pipeline_mimic_cxr.sh",
    ]
    for s in scripts:
        path = REPO_ROOT / s
        if not path.exists():
            report(f"exists {s}", False)
            continue
        content = path.read_text()
        checks = {
            "shebang": content.startswith("#!/bin/bash"),
            "PYTHONNOUSERSITE": "PYTHONNOUSERSITE" in content,
            "PYTHONPATH": "PYTHONPATH" in content,
        }
        failed = [k for k, v in checks.items() if not v]
        report(s, not failed, f"missing: {', '.join(failed)}" if failed else "OK")


def main():
    print("=" * 60)
    print("Smoke Tests — Multi-Dataset SR Pipeline")
    print("=" * 60)

    test_yaml_configs()
    test_paired_latent_dataset()
    test_diffusion_model()
    test_mimic_cxr_prep()
    test_brats_prep()
    test_eval_imports()
    test_slurm_scripts()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
