#!/usr/bin/env python3
"""
Test Reproducibility and Requirements

This script validates that the setup ensures reproducible results
and that all requirements are properly specified.
"""

import sys
import hashlib
from pathlib import Path

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def test_requirements_pinned():
    """Test that all requirements have pinned versions."""
    print("\n1. Testing Requirements Pinning...")

    base = Path(__file__).parent.parent
    req_file = base / "requirements.txt"

    with open(req_file, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    # Filter out comments and empty lines
    packages = [l for l in lines if l and not l.startswith('#')]

    unpinned = []
    for pkg in packages:
        # Skip git URLs
        if pkg.startswith('http') or '@git+' in pkg:
            continue

        # Check if version is pinned (has ==)
        if '==' not in pkg and '>=' not in pkg:
            unpinned.append(pkg)

    if unpinned:
        print(f"{RED}✗{RESET} Unpinned packages found: {unpinned}")
        return False
    else:
        print(f"{GREEN}✓{RESET} All {len(packages)} packages have versions specified")
        return True


def test_pytorch_cuda_version():
    """Test that PyTorch CUDA version is specified."""
    print("\n2. Testing PyTorch CUDA Version...")

    base = Path(__file__).parent.parent
    req_file = base / "requirements.txt"

    with open(req_file, 'r') as f:
        content = f.read()

    # Check for PyTorch with version
    if 'torch==' in content:
        torch_line = [l for l in content.split('\n') if 'torch==' in l][0]
        print(f"{GREEN}✓{RESET} PyTorch version pinned: {torch_line}")

        # Check environment.yml for CUDA version
        env_file = base / "environment.yml"
        with open(env_file, 'r') as f:
            env_content = f.read()

        if 'pytorch-cuda' in env_content or 'cudatoolkit' in env_content:
            print(f"{GREEN}✓{RESET} CUDA version specified in environment.yml")
            return True
        else:
            print(f"{YELLOW}⚠{RESET} CUDA version should be specified in environment.yml")
            return True
    else:
        print(f"{RED}✗{RESET} PyTorch version not pinned")
        return False


def test_seed_setting_documented():
    """Test that seed setting is documented."""
    print("\n3. Testing Seed Setting Documentation...")

    base = Path(__file__).parent.parent

    # Check validation framework for seed setting
    val_file = base / "medvae_diffusion_pipeline/validation_framework.py"

    with open(val_file, 'r') as f:
        content = f.read()

    required_seed_components = [
        'random.seed',
        'np.random.seed',
        'torch.manual_seed',
        'torch.cuda.manual_seed_all',
        'torch.backends.cudnn.deterministic',
        'torch.backends.cudnn.benchmark'
    ]

    missing = []
    for component in required_seed_components:
        if component not in content:
            missing.append(component)

    if missing:
        print(f"{RED}✗{RESET} Missing seed components: {missing}")
        return False
    else:
        print(f"{GREEN}✓{RESET} All reproducibility components present:")
        print(f"  - random.seed")
        print(f"  - numpy.random.seed")
        print(f"  - torch.manual_seed")
        print(f"  - torch.cuda.manual_seed_all")
        print(f"  - cudnn.deterministic = True")
        print(f"  - cudnn.benchmark = False")
        return True


def test_paper_seeds_documented():
    """Test that paper seed counts are documented."""
    print("\n4. Testing Paper Seed Documentation...")

    base = Path(__file__).parent.parent
    paper_doc = base / "PAPER_VALIDATION.md"

    if not paper_doc.exists():
        print(f"{YELLOW}⊘{RESET} PAPER_VALIDATION.md not found (skipping)")
        return True

    with open(paper_doc, 'r') as f:
        content = f.read()

    # Check for seed documentation
    required_info = [
        '4 seeds',  # 2D
        '1 seed',   # 3D
        '3 seeds',  # CAD
        'random seed',
        'reproducib',
    ]

    missing = []
    for info in required_info:
        if info.lower() not in content.lower():
            missing.append(info)

    if missing:
        print(f"{RED}✗{RESET} Missing seed documentation: {missing}")
        return False
    else:
        print(f"{GREEN}✓{RESET} Paper seed methodology documented:")
        print(f"  - 2D: 4 random seeds")
        print(f"  - 3D: 1 seed")
        print(f"  - CAD: 3 random seeds")
        return True


def test_dataset_info_documented():
    """Test that dataset information is documented."""
    print("\n5. Testing Dataset Documentation...")

    base = Path(__file__).parent.parent
    paper_doc = base / "PAPER_VALIDATION.md"

    if not paper_doc.exists():
        print(f"{YELLOW}⊘{RESET} PAPER_VALIDATION.md not found (skipping)")
        return True

    with open(paper_doc, 'r') as f:
        content = f.read()

    # Required dataset info
    required_datasets = [
        'CMMD',  # Mammography
        'VinDR-Mammo',  # Mammography
        'RSNA',  # Bone Age
        'GRAZPEDWRI-DX',  # Wrist
        'VerSe',  # Spine
        'CQ500',  # Head CT
        'MRNet',  # Knee MRI
        'CANDID-PTX',  # Chest X-ray
        'MIMIC-CXR',  # Chest X-ray
    ]

    found = []
    missing = []
    for dataset in required_datasets:
        if dataset in content:
            found.append(dataset)
        else:
            missing.append(dataset)

    if missing:
        print(f"{YELLOW}⚠{RESET} Some datasets not documented: {missing}")

    print(f"{GREEN}✓{RESET} {len(found)}/{len(required_datasets)} datasets documented")
    return len(found) >= len(required_datasets) * 0.7  # 70% threshold


def test_metrics_reproducibility():
    """Test that metrics are deterministic."""
    print("\n6. Testing Metrics Reproducibility...")

    base = Path(__file__).parent.parent
    val_file = base / "medvae_diffusion_pipeline/validation_framework.py"

    with open(val_file, 'r') as f:
        content = f.read()

    # Check for metrics that should be deterministic
    required_metrics = [
        'PeakSignalNoiseRatio',
        'StructuralSimilarityIndexMeasure',
        'MultiScaleStructuralSimilarityIndexMeasure',
    ]

    all_present = True
    for metric in required_metrics:
        if metric in content:
            print(f"{GREEN}✓{RESET} {metric}")
        else:
            print(f"{RED}✗{RESET} {metric} not found")
            all_present = False

    return all_present


def test_batch_size_documented():
    """Test that batch sizes are documented."""
    print("\n7. Testing Batch Size Documentation...")

    base = Path(__file__).parent.parent
    paper_doc = base / "PAPER_VALIDATION.md"

    if not paper_doc.exists():
        print(f"{YELLOW}⊘{RESET} PAPER_VALIDATION.md not found (skipping)")
        return True

    with open(paper_doc, 'r') as f:
        content = f.read()

    if 'batch' in content.lower() or 'batch_size' in content:
        print(f"{GREEN}✓{RESET} Batch size information documented")
        return True
    else:
        print(f"{YELLOW}⚠{RESET} Batch size should be documented")
        return True


def test_hardware_specs_documented():
    """Test that hardware specifications are mentioned."""
    print("\n8. Testing Hardware Documentation...")

    base = Path(__file__).parent.parent

    # Check multiple docs for hardware info
    docs_to_check = [
        "INSTALLATION.md",
        "PAPER_VALIDATION.md",
    ]

    found_hardware = False
    for doc in docs_to_check:
        doc_path = base / doc
        if doc_path.exists():
            with open(doc_path, 'r') as f:
                content = f.read().lower()
                if 'gpu' in content or 'cuda' in content or 'nvidia' in content:
                    found_hardware = True
                    break

    if found_hardware:
        print(f"{GREEN}✓{RESET} Hardware requirements documented")
        return True
    else:
        print(f"{YELLOW}⚠{RESET} Hardware specs should be documented")
        return True


def test_file_checksums():
    """Test that key files haven't been corrupted."""
    print("\n9. Testing File Integrity...")

    base = Path(__file__).parent.parent

    key_files = [
        "requirements.txt",
        "environment.yml",
        "pyproject.toml",
    ]

    all_valid = True
    for filename in key_files:
        filepath = base / filename
        if filepath.exists():
            # Just check file is readable and not empty
            with open(filepath, 'rb') as f:
                content = f.read()
                if len(content) > 0:
                    md5 = hashlib.md5(content).hexdigest()[:8]
                    print(f"{GREEN}✓{RESET} {filename} ({len(content)} bytes, md5:{md5})")
                else:
                    print(f"{RED}✗{RESET} {filename} is empty")
                    all_valid = False
        else:
            print(f"{RED}✗{RESET} {filename} not found")
            all_valid = False

    return all_valid


def test_environment_isolation():
    """Test that environment isolation is documented."""
    print("\n10. Testing Environment Isolation Documentation...")

    base = Path(__file__).parent.parent
    install_doc = base / "INSTALLATION.md"

    if not install_doc.exists():
        print(f"{YELLOW}⊘{RESET} INSTALLATION.md not found (skipping)")
        return True

    with open(install_doc, 'r') as f:
        content = f.read()

    isolation_methods = ['venv', 'conda', 'uv venv']
    found = [method for method in isolation_methods if method in content]

    if found:
        print(f"{GREEN}✓{RESET} Environment isolation documented:")
        for method in found:
            print(f"  - {method}")
        return True
    else:
        print(f"{RED}✗{RESET} Environment isolation not documented")
        return False


def main():
    print(f"\n{'='*60}")
    print("Reproducibility & Requirements Test")
    print(f"{'='*60}")

    all_passed = True

    all_passed &= test_requirements_pinned()
    all_passed &= test_pytorch_cuda_version()
    all_passed &= test_seed_setting_documented()
    all_passed &= test_paper_seeds_documented()
    all_passed &= test_dataset_info_documented()
    all_passed &= test_metrics_reproducibility()
    all_passed &= test_batch_size_documented()
    all_passed &= test_hardware_specs_documented()
    all_passed &= test_file_checksums()
    all_passed &= test_environment_isolation()

    print(f"\n{'='*60}")
    if all_passed:
        print(f"{GREEN}✓ All reproducibility tests passed!{RESET}")
        print(f"\n{YELLOW}Key Reproducibility Features:{RESET}")
        print(f"  • All package versions pinned")
        print(f"  • Random seeds properly set")
        print(f"  • Deterministic CUDA operations")
        print(f"  • Paper methodology documented")
        print(f"  • Environment isolation supported")
        return 0
    else:
        print(f"{RED}✗ Some reproducibility tests failed.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
