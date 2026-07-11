#!/usr/bin/env python3
"""
Basic Test Suite

Tests that don't require all packages to be installed.
This verifies the core structure and files are correct.
"""

import sys
from pathlib import Path

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def test_python_version():
    """Test Python version."""
    print("\n1. Testing Python Version...")
    version = sys.version_info

    if version.major == 3 and version.minor >= 11:
        print(f"{GREEN}✓{RESET} Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"{RED}✗{RESET} Python {version.major}.{version.minor} (requires 3.11+)")
        return False


def test_directory_structure():
    """Test directory structure."""
    print("\n2. Testing Directory Structure...")

    base = Path(__file__).parent.parent
    required_dirs = [
        "medvae_diffusion_pipeline",
        "medvae_diffusion_pipeline/scripts",
        "tests",
        "Real-ESRGAN",
        "MedVAE",
        "basic_diffusion",
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = base / dir_path
        if full_path.exists():
            print(f"{GREEN}✓{RESET} {dir_path}/")
        else:
            print(f"{RED}✗{RESET} {dir_path}/ (missing)")
            all_exist = False

    return all_exist


def test_installation_files():
    """Test installation files exist."""
    print("\n3. Testing Installation Files...")

    base = Path(__file__).parent.parent
    required_files = [
        "requirements.txt",
        "environment.yml",
        "pyproject.toml",
        "setup.sh",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = base / file_path
        if full_path.exists():
            print(f"{GREEN}✓{RESET} {file_path}")
        else:
            print(f"{RED}✗{RESET} {file_path} (missing)")
            all_exist = False

    return all_exist


def test_validation_files():
    """Test validation framework files."""
    print("\n5. Testing Validation Framework Files...")

    base = Path(__file__).parent.parent
    required_files = [
        "medvae_diffusion_pipeline/validation_framework.py",
        "medvae_diffusion_pipeline/paper_validation_config.py",
        "medvae_diffusion_pipeline/scripts/04_validate_model.py",
        "medvae_diffusion_pipeline/scripts/05_paper_validation.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = base / file_path
        if full_path.exists():
            print(f"{GREEN}✓{RESET} {Path(file_path).name}")
        else:
            print(f"{RED}✗{RESET} {file_path} (missing)")
            all_exist = False

    return all_exist


def test_file_contents():
    """Test key file contents."""
    print("\n6. Testing File Contents...")

    base = Path(__file__).parent.parent

    # Test requirements.txt
    req_file = base / "requirements.txt"
    with open(req_file, 'r') as f:
        req_content = f.read()

    required_packages = ["torch", "diffusers", "monai", "basicsr", "torchmetrics"]
    missing = [pkg for pkg in required_packages if pkg not in req_content]

    if missing:
        print(f"{RED}✗{RESET} requirements.txt missing: {missing}")
        return False
    else:
        print(f"{GREEN}✓{RESET} requirements.txt has all key packages")

    # Test paper validation config structure
    config_file = base / "medvae_diffusion_pipeline/paper_validation_config.py"
    with open(config_file, 'r') as f:
        config_content = f.read()

    required_configs = ["CONFIG_2D_PERCEPTUAL", "CONFIG_3D_PERCEPTUAL", "CONFIG_CAD_TASKS"]
    missing = [cfg for cfg in required_configs if cfg not in config_content]

    if missing:
        print(f"{RED}✗{RESET} paper_validation_config.py missing: {missing}")
        return False
    else:
        print(f"{GREEN}✓{RESET} paper_validation_config.py has all configs")

    return True


def test_paper_methodology_documented():
    """Test paper methodology is documented."""
    print("\n7. Testing Paper Methodology Documentation...")

    base = Path(__file__).parent.parent
    paper_doc = base / "PAPER_VALIDATION.md"

    if not paper_doc.exists():
        print(f"{YELLOW}⊘{RESET} PAPER_VALIDATION.md not found (skipping)")
        return True

    with open(paper_doc, 'r') as f:
        content = f.read()

    # Check for key methodology details
    required_content = [
        "1000 images",  # 2D sample size
        "100 volumes",  # 3D sample size
        "4 seeds",  # 2D seeds
        "1 seed",  # 3D seeds
        "3 seeds",  # CAD seeds
        "MS-SSIM",  # Multi-scale SSIM
        "PSNR",  # Peak signal-to-noise ratio
        "AUROC",  # Area under ROC
    ]

    missing = [item for item in required_content if item not in content]

    if missing:
        print(f"{RED}✗{RESET} PAPER_VALIDATION.md missing methodology: {missing}")
        return False
    else:
        print(f"{GREEN}✓{RESET} Paper methodology fully documented")

    return True


def main():
    print(f"\n{'='*60}")
    print("Basic Test Suite")
    print("(Tests that don't require all packages installed)")
    print(f"{'='*60}")

    all_passed = True

    all_passed &= test_python_version()
    all_passed &= test_directory_structure()
    all_passed &= test_installation_files()
    # all_passed &= test_documentation_files()
    all_passed &= test_validation_files()
    all_passed &= test_file_contents()
    all_passed &= test_paper_methodology_documented()

    print(f"\n{'='*60}")
    if all_passed:
        print(f"{GREEN}✓ All basic tests passed!{RESET}")
        print(f"\n{YELLOW}Note:{RESET} To run full tests with package imports:")
        print(f"  1. Install environment: ./setup.sh")
        print(f"  2. Run: ./tests/run_all_tests.sh")
        return 0
    else:
        print(f"{RED}✗ Some basic tests failed.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
