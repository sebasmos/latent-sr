#!/usr/bin/env python3
"""
Test Environment Setup

This script verifies that all required packages are installed correctly
and that the environment matches the requirements.
"""

import sys
import importlib
from pathlib import Path

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"{GREEN}✓{RESET} {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"{RED}✗{RESET} {package_name or module_name}: {e}")
        return False


def test_version(module_name, expected_version=None):
    """Test module version."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")

        if expected_version and not version.startswith(expected_version):
            print(f"{YELLOW}⚠{RESET} {module_name}: v{version} (expected {expected_version}.x)")
            return True
        else:
            print(f"{GREEN}✓{RESET} {module_name}: v{version}")
            return True
    except Exception as e:
        print(f"{RED}✗{RESET} {module_name}: {e}")
        return False


def main():
    print(f"\n{'='*60}")
    print("Environment Setup Test")
    print(f"{'='*60}\n")

    all_passed = True

    # Test Python version
    print("Python Version:")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"{GREEN}✓{RESET} Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"{RED}✗{RESET} Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        all_passed = False

    # Test Core Dependencies
    print("\nCore Scientific Computing:")
    all_passed &= test_version("numpy", "2.2")
    all_passed &= test_version("scipy", "1.13")
    all_passed &= test_version("pandas", "2.2")
    all_passed &= test_version("sklearn", "1.6")

    # Test PyTorch
    print("\nPyTorch Ecosystem:")
    all_passed &= test_version("torch", "2.5")
    all_passed &= test_version("torchvision", "0.20")

    # Test CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'

        if cuda_available:
            print(f"{GREEN}✓{RESET} CUDA: {cuda_version} (GPU available)")
        else:
            print(f"{YELLOW}⚠{RESET} CUDA: {cuda_version} (No GPU available - CPU only)")
    except Exception as e:
        print(f"{RED}✗{RESET} CUDA check failed: {e}")
        all_passed = False

    # Test Deep Learning Libraries
    print("\nDeep Learning:")
    all_passed &= test_version("diffusers", "0.33")
    all_passed &= test_version("transformers", "4.52")
    all_passed &= test_version("accelerate", "1.7")
    all_passed &= test_import("torchmetrics")

    # Test Medical Imaging
    print("\nMedical Imaging:")
    all_passed &= test_import("monai")
    all_passed &= test_import("nibabel")
    all_passed &= test_import("SimpleITK")
    all_passed &= test_import("nilearn")
    all_passed &= test_import("dicom2nifti")

    # Test Image Processing
    print("\nImage Processing:")
    all_passed &= test_import("PIL", "Pillow")
    all_passed &= test_import("cv2", "opencv-python")
    all_passed &= test_import("matplotlib")
    all_passed &= test_import("seaborn")

    # Test Utilities
    print("\nUtilities:")
    all_passed &= test_import("tqdm")
    all_passed &= test_import("omegaconf")
    all_passed &= test_import("hydra")
    all_passed &= test_import("wandb")

    # Test Real-ESRGAN Dependencies (optional)
    print("\nReal-ESRGAN (optional):")
    test_import("basicsr")  # Don't fail if missing
    test_import("facexlib")
    test_import("gfpgan")

    # Test Metrics
    print("\nValidation Metrics:")
    try:
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
            MultiScaleStructuralSimilarityIndexMeasure,
            LearnedPerceptualImagePatchSimilarity,
        )
        print(f"{GREEN}✓{RESET} All image metrics available")

        from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
        print(f"{GREEN}✓{RESET} All classification metrics available")
    except ImportError as e:
        print(f"{RED}✗{RESET} Metrics import failed: {e}")
        all_passed = False

    # Final result
    print(f"\n{'='*60}")
    if all_passed:
        print(f"{GREEN}✓ All environment tests passed!{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some tests failed. Check the output above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
