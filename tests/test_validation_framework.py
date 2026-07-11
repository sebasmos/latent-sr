#!/usr/bin/env python3
"""
Test Validation Framework

This script tests the validation framework functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from medvae_diffusion_pipeline.validation_framework import (
    ValidationConfig,
    MetricsCalculator,
    ValidationFramework,
)

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, n_samples=100, img_size=192):
        self.n_samples = n_samples
        self.img_size = img_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Generate random images in [0, 1] range for metrics
        hr = torch.rand(3, self.img_size, self.img_size)
        lr = torch.rand(3, self.img_size // 2, self.img_size // 2)
        return hr, lr


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

    def forward(self, x):
        return self.upscale(x)


def test_validation_config():
    """Test ValidationConfig creation."""
    print("\nTesting ValidationConfig...")

    try:
        config = ValidationConfig(
            validation_type="held_out",
            n_seeds=3,
            test_size=0.15,
            val_size=0.15,
            batch_size=8,
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert "validation_type" in config_dict
        assert config_dict["n_seeds"] == 3

        # Test from_dict
        config2 = ValidationConfig.from_dict(config_dict)
        assert config2.n_seeds == 3

        print(f"{GREEN}✓{RESET} ValidationConfig works correctly")
        return True
    except Exception as e:
        print(f"{RED}✗{RESET} ValidationConfig failed: {e}")
        return False


def test_metrics_calculator():
    """Test MetricsCalculator."""
    print("\nTesting MetricsCalculator...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = MetricsCalculator(device=device)

        # Create dummy tensors (MS-SSIM requires images >160px, LPIPS needs [0,1] range)
        pred = torch.rand(2, 3, 192, 192).to(device)  # torch.rand gives [0, 1]
        target = torch.rand(2, 3, 192, 192).to(device)

        # Compute metrics
        metrics = calc.compute_all(pred, target)

        # Check all expected metrics are present
        expected_metrics = ["psnr", "ssim", "ms_ssim", "lpips", "mse", "mae"]
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], float), f"{metric} should be float"

        print(f"{GREEN}✓{RESET} All metrics computed correctly:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} MetricsCalculator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_split():
    """Test dataset splitting."""
    print("\nTesting dataset splitting...")

    try:
        config = ValidationConfig(
            validation_type="held_out",
            test_size=0.2,
            val_size=0.1,
        )

        framework = ValidationFramework(config)
        dataset = DummyDataset(n_samples=100)

        # Split dataset
        train, val, test = framework.split_dataset(dataset, seed=42)

        # Check sizes
        total = len(train) + len(val) + len(test)
        assert total == 100, f"Total should be 100, got {total}"

        # Check approximate sizes
        assert 65 <= len(train) <= 75, f"Train size wrong: {len(train)}"
        assert 8 <= len(val) <= 12, f"Val size wrong: {len(val)}"
        assert 18 <= len(test) <= 22, f"Test size wrong: {len(test)}"

        print(f"{GREEN}✓{RESET} Dataset split correctly:")
        print(f"  - Train: {len(train)}")
        print(f"  - Val: {len(val)}")
        print(f"  - Test: {len(test)}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Dataset split failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_k_fold_split():
    """Test K-fold splitting."""
    print("\nTesting K-fold splitting...")

    try:
        config = ValidationConfig(
            validation_type="k_fold",
            n_folds=5,
        )

        framework = ValidationFramework(config)
        dataset = DummyDataset(n_samples=100)

        # Create K-fold splits
        folds = framework.k_fold_split(dataset, seed=42)

        assert len(folds) == 5, f"Should have 5 folds, got {len(folds)}"

        print(f"{GREEN}✓{RESET} K-fold split correctly:")
        for i, (train, val) in enumerate(folds):
            print(f"  - Fold {i+1}: Train={len(train)}, Val={len(val)}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} K-fold split failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_seed_reproducibility():
    """Test that same seed produces same results."""
    print("\nTesting seed reproducibility...")

    try:
        config = ValidationConfig(validation_type="held_out")
        framework = ValidationFramework(config)
        dataset = DummyDataset(n_samples=100)

        # Split with same seed twice
        train1, val1, test1 = framework.split_dataset(dataset, seed=42)
        train2, val2, test2 = framework.split_dataset(dataset, seed=42)

        # Check that indices are the same
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)

        print(f"{GREEN}✓{RESET} Seed reproducibility works")
        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Seed reproducibility failed: {e}")
        return False


def test_model_evaluation():
    """Test model evaluation."""
    print("\nTesting model evaluation...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = ValidationConfig(
            batch_size=4,
            device=device,
        )

        framework = ValidationFramework(config)

        # Create dummy dataset and model (MS-SSIM requires images larger than 160px)
        dataset = DummyDataset(n_samples=20, img_size=192)
        dataloader = DataLoader(dataset, batch_size=4)
        model = DummyModel(upscale_factor=2).to(device)

        # Evaluate
        metrics = framework.evaluate_model(model, dataloader, desc="Test")

        # Check metrics
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "ms_ssim" in metrics

        print(f"{GREEN}✓{RESET} Model evaluation works:")
        for metric, value in metrics.items():
            if not metric.endswith("_std"):
                print(f"  - {metric}: {value:.4f}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Model evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"\n{'='*60}")
    print("Validation Framework Tests")
    print(f"{'='*60}")

    all_passed = True

    all_passed &= test_validation_config()
    all_passed &= test_metrics_calculator()
    all_passed &= test_dataset_split()
    all_passed &= test_k_fold_split()
    all_passed &= test_seed_reproducibility()
    all_passed &= test_model_evaluation()

    print(f"\n{'='*60}")
    if all_passed:
        print(f"{GREEN}✓ All validation framework tests passed!{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
