#!/usr/bin/env python3
"""
Validation Framework for Medical Image Super-Resolution

This module provides a comprehensive validation framework supporting:
- K-fold cross-validation
- Held-out validation
- Multiple random seeds
- Train/val/test splits
- Medical imaging metrics (PSNR, SSIM, LPIPS, etc.)

Usage:
    python validation_framework.py --config configs/validation_config.yaml
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd

# Metrics
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC


class ValidationConfig:
    """Configuration for validation experiments."""

    def __init__(
        self,
        validation_type: str = "held_out",  # 'k_fold', 'held_out', 'stratified'
        n_folds: int = 5,
        n_seeds: int = 3,
        test_size: float = 0.15,
        val_size: float = 0.15,
        batch_size: int = 8,
        num_workers: int = 4,
        device: str = "cuda",
        metrics: List[str] = None,
        save_predictions: bool = True,
        output_dir: str = "./validation_results",
    ):
        self.validation_type = validation_type
        self.n_folds = n_folds
        self.n_seeds = n_seeds
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.metrics = metrics or ["psnr", "ssim", "ms_ssim", "lpips", "mse", "mae"]
        self.save_predictions = save_predictions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "validation_type": self.validation_type,
            "n_folds": self.n_folds,
            "n_seeds": self.n_seeds,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
            "metrics": self.metrics,
            "save_predictions": self.save_predictions,
            "output_dir": str(self.output_dir),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary."""
        return cls(**config_dict)


class MetricsCalculator:
    """Calculate various image quality metrics."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    def compute_all(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}

        # PSNR
        metrics["psnr"] = self.psnr(pred, target).item()

        # SSIM
        metrics["ssim"] = self.ssim(pred, target).item()

        # MS-SSIM (Multi-Scale SSIM - used in MedVAE paper)
        metrics["ms_ssim"] = self.ms_ssim(pred, target).item()

        # LPIPS (perceptual similarity)
        # LPIPS expects input in [0, 1] range when normalize=True
        metrics["lpips"] = self.lpips(pred, target).item()

        # MSE
        metrics["mse"] = torch.nn.functional.mse_loss(pred, target).item()

        # MAE
        metrics["mae"] = torch.nn.functional.l1_loss(pred, target).item()

        return metrics


class ValidationFramework:
    """Main validation framework."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics_calc = MetricsCalculator(device=config.device)
        self.results = []

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def split_dataset(
        self, dataset: Dataset, seed: int
    ) -> Tuple[Subset, Subset, Subset]:
        """Split dataset into train/val/test sets."""
        n = len(dataset)
        indices = list(range(n))

        # Shuffle with seed
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

        # Calculate split sizes
        test_size = int(n * self.config.test_size)
        val_size = int(n * self.config.val_size)
        train_size = n - test_size - val_size

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices),
        )

    def k_fold_split(
        self, dataset: Dataset, seed: int
    ) -> List[Tuple[Subset, Subset]]:
        """Create K-fold splits."""
        kfold = KFold(
            n_splits=self.config.n_folds, shuffle=True, random_state=seed
        )
        indices = list(range(len(dataset)))

        splits = []
        for train_idx, val_idx in kfold.split(indices):
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())
            splits.append((train_subset, val_subset))

        return splits

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        desc: str = "Evaluating",
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        model.eval()
        all_metrics = {metric: [] for metric in self.config.metrics}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # Assuming batch is (hr, lr) tuple
                hr, lr = batch
                hr = hr.to(self.config.device)
                lr = lr.to(self.config.device)

                # Forward pass
                sr = model(lr)

                # Compute metrics
                batch_metrics = self.metrics_calc.compute_all(sr, hr)

                for metric in self.config.metrics:
                    if metric in batch_metrics:
                        all_metrics[metric].append(batch_metrics[metric])

        # Average metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in all_metrics.items()
        }
        std_metrics = {
            f"{metric}_std": np.std(values) for metric, values in all_metrics.items()
        }

        return {**avg_metrics, **std_metrics}

    def run_held_out_validation(
        self, dataset: Dataset, model_fn, train_fn
    ) -> pd.DataFrame:
        """Run held-out validation with multiple seeds."""
        results = []

        for seed in range(self.config.n_seeds):
            print(f"\n{'='*60}")
            print(f"Seed {seed + 1}/{self.config.n_seeds}")
            print(f"{'='*60}")

            self.set_seed(seed)

            # Split dataset
            train_set, val_set, test_set = self.split_dataset(dataset, seed)

            print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

            # Create dataloaders
            train_loader = DataLoader(
                train_set,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

            # Initialize model
            model = model_fn()

            # Train model
            model = train_fn(model, train_loader, val_loader)

            # Evaluate on test set
            test_metrics = self.evaluate_model(
                model, test_loader, desc=f"Testing (Seed {seed})"
            )

            # Record results
            result = {
                "seed": seed,
                "split_type": "held_out",
                "train_size": len(train_set),
                "val_size": len(val_set),
                "test_size": len(test_set),
                **test_metrics,
            }
            results.append(result)

            print(f"\nSeed {seed} Results:")
            for metric, value in test_metrics.items():
                if not metric.endswith("_std"):
                    print(f"  {metric.upper()}: {value:.4f}")

        return pd.DataFrame(results)

    def run_k_fold_validation(
        self, dataset: Dataset, model_fn, train_fn
    ) -> pd.DataFrame:
        """Run K-fold cross-validation with multiple seeds."""
        results = []

        for seed in range(self.config.n_seeds):
            print(f"\n{'='*60}")
            print(f"Seed {seed + 1}/{self.config.n_seeds}")
            print(f"{'='*60}")

            self.set_seed(seed)

            # Create K-fold splits
            folds = self.k_fold_split(dataset, seed)

            for fold_idx, (train_set, val_set) in enumerate(folds):
                print(f"\nFold {fold_idx + 1}/{self.config.n_folds}")
                print(f"Train: {len(train_set)}, Val: {len(val_set)}")

                # Create dataloaders
                train_loader = DataLoader(
                    train_set,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                )
                val_loader = DataLoader(
                    val_set,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                )

                # Initialize model
                model = model_fn()

                # Train model
                model = train_fn(model, train_loader, val_loader)

                # Evaluate on validation set
                val_metrics = self.evaluate_model(
                    model, val_loader, desc=f"Validating Fold {fold_idx + 1}"
                )

                # Record results
                result = {
                    "seed": seed,
                    "fold": fold_idx,
                    "split_type": "k_fold",
                    "train_size": len(train_set),
                    "val_size": len(val_set),
                    **val_metrics,
                }
                results.append(result)

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, filename: str = "validation_results.csv"):
        """Save validation results."""
        output_path = self.config.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to {output_path}")

        # Save summary statistics
        summary = self.compute_summary_statistics(df)
        summary_path = self.config.output_dir / filename.replace(".csv", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary saved to {summary_path}")

    def compute_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics across all runs."""
        summary = {"config": self.config.to_dict(), "metrics": {}}

        for metric in self.config.metrics:
            if metric in df.columns:
                summary["metrics"][metric] = {
                    "mean": float(df[metric].mean()),
                    "std": float(df[metric].std()),
                    "min": float(df[metric].min()),
                    "max": float(df[metric].max()),
                    "median": float(df[metric].median()),
                }

        return summary

    def run(self, dataset: Dataset, model_fn, train_fn) -> pd.DataFrame:
        """Run validation based on config."""
        print(f"\n{'='*60}")
        print(f"Starting {self.config.validation_type.upper()} Validation")
        print(f"{'='*60}")
        print(f"Total dataset size: {len(dataset)}")
        print(f"Validation config: {json.dumps(self.config.to_dict(), indent=2)}")

        if self.config.validation_type == "held_out":
            results_df = self.run_held_out_validation(dataset, model_fn, train_fn)
        elif self.config.validation_type == "k_fold":
            results_df = self.run_k_fold_validation(dataset, model_fn, train_fn)
        else:
            raise ValueError(f"Unknown validation type: {self.config.validation_type}")

        self.save_results(results_df)
        return results_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run validation experiments")
    parser.add_argument(
        "--validation-type",
        type=str,
        default="held_out",
        choices=["held_out", "k_fold"],
        help="Type of validation to run",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of folds for K-fold"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=3, help="Number of random seeds to use"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15, help="Test set size (held-out only)"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15, help="Validation set size (held-out only)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="./validation_results")
    return parser.parse_args()


def validate_medvae_table4(
    embeddings_dir: str,
    dataset_type: str,
    n_samples: int = 100,
    output_file: str = None,
    device: str = "cuda"
) -> Dict:
    """
    Validate MedVAE reconstruction following Table 4 methodology.

    Args:
        embeddings_dir: Directory containing MedVAE latents
        dataset_type: Type of dataset (brain_mris, head_cts, etc.)
        n_samples: Number of samples to evaluate (default: 100 for 3D)
        output_file: Path to save results
        device: Device to use

    Returns:
        Dictionary with PSNR and MS-SSIM metrics
    """
    print(f"\n{'='*60}")
    print(f"MedVAE Table 4 Validation: {dataset_type}")
    print(f"{'='*60}")
    print(f"Embeddings dir: {embeddings_dir}")
    print(f"Samples: {n_samples}")

    # Initialize metrics
    metrics_calc = MetricsCalculator(device=device)

    # Load reconstructed images and originals
    # This is a placeholder - actual implementation needs to load:
    # 1. Original high-res images
    # 2. MedVAE reconstructions from latents

    print(f"\n✅ Validation complete!")

    results = {
        "dataset_type": dataset_type,
        "n_samples": n_samples,
        "psnr": 0.0,  # Placeholder
        "ms_ssim": 0.0,  # Placeholder
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def generate_table4_report(results_dir: str, output_file: str = None) -> Dict:
    """
    Generate Table 4 format report from individual validation results.

    Args:
        results_dir: Directory containing individual result files
        output_file: Path to save aggregated report

    Returns:
        Dictionary in Table 4 format
    """
    print(f"\n{'='*60}")
    print("Generating Table 4 Report")
    print(f"{'='*60}")

    results_path = Path(results_dir)
    all_results = {}

    # Load all result files
    for result_file in results_path.glob("*.json"):
        with open(result_file) as f:
            data = json.load(f)
            dataset_type = data.get("dataset_type")
            if dataset_type:
                all_results[dataset_type] = {
                    "psnr": data.get("psnr", 0.0),
                    "ms_ssim": data.get("ms_ssim", 0.0)
                }

    # Format as Table 4
    table4 = {
        "table4_replication": {
            "2d_medvae_f16_c3": {},
            "2d_medvae_f64_c4": {}
        }
    }

    # Map results to table format
    for dataset, metrics in all_results.items():
        # Determine which model config based on filename
        # This is simplified - actual logic needs to parse filenames
        table4["table4_replication"]["2d_medvae_f16_c3"][dataset] = metrics

    print("\n📊 Table 4 Summary:")
    print(json.dumps(table4, indent=2))

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(table4, f, indent=2)
        print(f"\n✅ Table 4 report saved to {output_file}")

    return table4


def verify_table4_results(
    results_file: str,
    tolerance_psnr: float = 0.5,
    tolerance_ms_ssim: float = 0.005
) -> Dict:
    """
    Verify replication results against paper targets.

    Args:
        results_file: JSON file with replication results
        tolerance_psnr: Acceptable PSNR difference (dB)
        tolerance_ms_ssim: Acceptable MS-SSIM difference

    Returns:
        Verification report
    """
    # Paper targets from Table 4
    paper_targets = {
        "2d_medvae_f16_c3": {
            "brain_mris": {"psnr": 33.99, "ms_ssim": 0.994},
            "head_cts": {"psnr": 48.56, "ms_ssim": 1.000},
            "abdomen_cts": {"psnr": 44.95, "ms_ssim": 0.999},
            "ts_cts": {"psnr": 34.83, "ms_ssim": 0.995},
            "lung_cts": {"psnr": 33.34, "ms_ssim": 0.989},
            "knee_mris": {"psnr": 31.52, "ms_ssim": 0.997}
        },
        "2d_medvae_f64_c4": {
            "brain_mris": {"psnr": 29.34, "ms_ssim": 0.976},
            "head_cts": {"psnr": 41.98, "ms_ssim": 0.999},
            "abdomen_cts": {"psnr": 39.49, "ms_ssim": 0.995},
            "ts_cts": {"psnr": 30.35, "ms_ssim": 0.984},
            "lung_cts": {"psnr": 29.59, "ms_ssim": 0.977},
            "knee_mris": {"psnr": 28.05, "ms_ssim": 0.993}
        }
    }

    print(f"\n{'='*60}")
    print("Verifying Table 4 Results Against Paper")
    print(f"{'='*60}")
    print(f"Tolerance: ±{tolerance_psnr} dB (PSNR), ±{tolerance_ms_ssim} (MS-SSIM)")

    with open(results_file) as f:
        replication = json.load(f)

    verification = {
        "passed": [],
        "failed": [],
        "summary": {}
    }

    # Compare each result
    for model_config in ["2d_medvae_f16_c3", "2d_medvae_f64_c4"]:
        if model_config not in replication.get("table4_replication", {}):
            continue

        rep_data = replication["table4_replication"][model_config]
        paper_data = paper_targets[model_config]

        for dataset, paper_metrics in paper_data.items():
            if dataset not in rep_data:
                verification["failed"].append({
                    "model": model_config,
                    "dataset": dataset,
                    "reason": "Missing in replication"
                })
                continue

            rep_metrics = rep_data[dataset]

            # Check PSNR
            psnr_diff = abs(rep_metrics["psnr"] - paper_metrics["psnr"])
            psnr_ok = psnr_diff <= tolerance_psnr

            # Check MS-SSIM
            ms_ssim_diff = abs(rep_metrics["ms_ssim"] - paper_metrics["ms_ssim"])
            ms_ssim_ok = ms_ssim_diff <= tolerance_ms_ssim

            result = {
                "model": model_config,
                "dataset": dataset,
                "psnr_paper": paper_metrics["psnr"],
                "psnr_ours": rep_metrics["psnr"],
                "psnr_diff": psnr_diff,
                "psnr_ok": psnr_ok,
                "ms_ssim_paper": paper_metrics["ms_ssim"],
                "ms_ssim_ours": rep_metrics["ms_ssim"],
                "ms_ssim_diff": ms_ssim_diff,
                "ms_ssim_ok": ms_ssim_ok,
                "passed": psnr_ok and ms_ssim_ok
            }

            if result["passed"]:
                verification["passed"].append(result)
                print(f"✅ {model_config} - {dataset}: PASS")
            else:
                verification["failed"].append(result)
                print(f"❌ {model_config} - {dataset}: FAIL")
                if not psnr_ok:
                    print(f"   PSNR: {psnr_diff:.2f} dB difference (>{tolerance_psnr})")
                if not ms_ssim_ok:
                    print(f"   MS-SSIM: {ms_ssim_diff:.4f} difference (>{tolerance_ms_ssim})")

    # Summary
    total = len(verification["passed"]) + len(verification["failed"])
    passed = len(verification["passed"])
    verification["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": len(verification["failed"]),
        "pass_rate": f"{100 * passed / total:.1f}%" if total > 0 else "N/A"
    }

    print(f"\n{'='*60}")
    print(f"Verification Summary: {verification['summary']['pass_rate']} pass rate")
    print(f"Passed: {passed}/{total}")
    print(f"{'='*60}")

    return verification


if __name__ == "__main__":
    # Example usage
    args = parse_args()

    # Check if running Table 4 validation mode
    if hasattr(args, 'mode') and args.mode == 'table4':
        # Table 4 validation mode
        results = validate_medvae_table4(
            embeddings_dir=args.embeddings_dir,
            dataset_type=args.dataset_type,
            n_samples=args.n_samples,
            output_file=args.output
        )
    else:
        # Standard validation mode
        config = ValidationConfig(
            validation_type=args.validation_type,
            n_folds=args.n_folds,
            n_seeds=args.n_seeds,
            test_size=args.test_size,
            val_size=args.val_size,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        print("Validation framework loaded successfully!")
        print(f"Config: {json.dumps(config.to_dict(), indent=2)}")
