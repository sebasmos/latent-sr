#!/usr/bin/env python3
"""
Test Installation Files

This script verifies that all installation files are present and valid.
"""

import sys
import json
import yaml
from pathlib import Path

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def test_file_exists(filepath, description):
    """Test if a file exists."""
    path = Path(filepath)
    if path.exists():
        print(f"{GREEN}✓{RESET} {description}: {path.name}")
        return True
    else:
        print(f"{RED}✗{RESET} {description}: {path.name} not found")
        return False


def test_requirements_txt():
    """Test requirements.txt exists and is valid."""
    print("\nTesting requirements.txt...")

    try:
        req_file = Path(__file__).parent.parent / "requirements.txt"
        if not req_file.exists():
            print(f"{RED}✗{RESET} requirements.txt not found")
            return False

        # Read and check content
        with open(req_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#')]

        # Check for key packages
        required_packages = [
            "torch",
            "torchvision",
            "diffusers",
            "transformers",
            "monai",
            "basicsr",
        ]

        content = '\n'.join(lines)
        missing = []
        for pkg in required_packages:
            if pkg not in content:
                missing.append(pkg)

        if missing:
            print(f"{RED}✗{RESET} Missing packages: {missing}")
            return False

        print(f"{GREEN}✓{RESET} requirements.txt valid ({len(lines)} packages)")
        return True
    except Exception as e:
        print(f"{RED}✗{RESET} requirements.txt test failed: {e}")
        return False


def test_environment_yml():
    """Test environment.yml exists and is valid."""
    print("\nTesting environment.yml...")

    try:
        env_file = Path(__file__).parent.parent / "environment.yml"
        if not env_file.exists():
            print(f"{RED}✗{RESET} environment.yml not found")
            return False

        # Parse YAML
        with open(env_file, 'r') as f:
            env = yaml.safe_load(f)

        # Check structure
        assert "name" in env, "Missing 'name' in environment.yml"
        assert "channels" in env, "Missing 'channels' in environment.yml"
        assert "dependencies" in env, "Missing 'dependencies' in environment.yml"

        # Check name
        assert env["name"] == "medvae-sr", "Environment name should be 'medvae-sr'"

        # Check Python version
        deps = env["dependencies"]
        python_found = any("python" in str(d) for d in deps)
        assert python_found, "Python version not specified"

        print(f"{GREEN}✓{RESET} environment.yml valid")
        print(f"  - Name: {env['name']}")
        print(f"  - Channels: {len(env['channels'])}")
        print(f"  - Dependencies: {len(deps)}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} environment.yml test failed: {e}")
        return False


def test_pyproject_toml():
    """Test pyproject.toml exists and is valid."""
    print("\nTesting pyproject.toml...")

    try:
        toml_file = Path(__file__).parent.parent / "pyproject.toml"
        if not toml_file.exists():
            print(f"{RED}✗{RESET} pyproject.toml not found")
            return False

        # Read content (basic validation)
        with open(toml_file, 'r') as f:
            content = f.read()

        # Check for required sections
        required_sections = ["[project]", "[build-system]", "[tool.uv]"]
        missing = [s for s in required_sections if s not in content]

        if missing:
            print(f"{RED}✗{RESET} Missing sections: {missing}")
            return False

        # Check project name
        assert "medvae-diffusion-sr" in content or "medvae" in content

        print(f"{GREEN}✓{RESET} pyproject.toml valid")
        return True
    except Exception as e:
        print(f"{RED}✗{RESET} pyproject.toml test failed: {e}")
        return False


def test_setup_sh():
    """Test setup.sh exists and is executable."""
    print("\nTesting setup.sh...")

    try:
        setup_file = Path(__file__).parent.parent / "setup.sh"
        if not setup_file.exists():
            print(f"{RED}✗{RESET} setup.sh not found")
            return False

        # Check if executable
        import os
        is_executable = os.access(setup_file, os.X_OK)

        if not is_executable:
            print(f"{YELLOW}⚠{RESET} setup.sh exists but not executable (run: chmod +x setup.sh)")
        else:
            print(f"{GREEN}✓{RESET} setup.sh valid and executable")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} setup.sh test failed: {e}")
        return False


def test_documentation_files():
    """Test documentation files exist."""
    print("\nTesting documentation files...")

    base_dir = Path(__file__).parent.parent

    # Check for README (required)
    readme_exists = test_file_exists(base_dir / "README.md", "Main README")

    # Optional docs (skip if not present)
    optional_docs = [
        ("INSTALLATION.md", "Installation guide"),
        ("VALIDATION.md", "Validation docs"),
        ("PAPER_VALIDATION.md", "Paper validation"),
        ("README_VALIDATION.md", "Validation README"),
    ]

    for doc_file, description in optional_docs:
        doc_path = base_dir / doc_file
        if doc_path.exists():
            print(f"{GREEN}✓{RESET} {description}: {doc_file}")
        else:
            print(f"{YELLOW}⊘{RESET} {description}: {doc_file} (optional, skipping)")

    return readme_exists


def test_validation_scripts():
    """Test validation scripts exist."""
    print("\nTesting validation scripts...")

    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "medvae_diffusion_pipeline"

    all_exist = True
    all_exist &= test_file_exists(
        scripts_dir / "validation_framework.py",
        "Validation framework"
    )
    all_exist &= test_file_exists(
        scripts_dir / "paper_validation_config.py",
        "Paper config"
    )
    all_exist &= test_file_exists(
        scripts_dir / "scripts/04_validate_model.py",
        "Validation script"
    )
    all_exist &= test_file_exists(
        scripts_dir / "scripts/05_paper_validation.py",
        "Paper validation script"
    )

    return all_exist


def main():
    print(f"\n{'='*60}")
    print("Installation Files Test")
    print(f"{'='*60}")

    all_passed = True

    all_passed &= test_requirements_txt()
    all_passed &= test_environment_yml()
    all_passed &= test_pyproject_toml()
    all_passed &= test_setup_sh()
    all_passed &= test_documentation_files()
    all_passed &= test_validation_scripts()

    print(f"\n{'='*60}")
    if all_passed:
        print(f"{GREEN}✓ All installation files are present and valid!{RESET}")
        return 0
    else:
        print(f"{RED}✗ Some files are missing or invalid.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
