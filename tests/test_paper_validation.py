#!/usr/bin/env python3
"""
Test Paper Validation Configuration

This script tests that the paper validation configuration matches
the methodology described in the MedVAE paper.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from medvae_diffusion_pipeline.paper_validation_config import (
    CONFIG_2D_PERCEPTUAL,
    CONFIG_3D_PERCEPTUAL,
    CONFIG_CAD_TASKS,
    SAMPLE_SIZES,
    IMAGE_TYPES_2D,
    IMAGE_TYPES_3D,
    CAD_TASKS,
    DOWNSIZING_FACTORS,
    LATENT_CHANNELS,
    get_paper_config,
)

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def test_2d_config():
    """Test 2D perceptual quality configuration."""
    print("\nTesting 2D Perceptual Config...")

    try:
        config = CONFIG_2D_PERCEPTUAL

        # Check methodology from paper
        assert config.n_seeds == 4, f"2D should use 4 seeds, got {config.n_seeds}"
        assert "psnr" in config.metrics, "2D should include PSNR"
        assert "ms_ssim" in config.metrics, "2D should include MS-SSIM"

        print(f"{GREEN}✓{RESET} 2D config matches paper:")
        print(f"  - Seeds: {config.n_seeds} (paper: 4)")
        print(f"  - Metrics: {config.metrics}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} 2D config failed: {e}")
        return False


def test_3d_config():
    """Test 3D perceptual quality configuration."""
    print("\nTesting 3D Perceptual Config...")

    try:
        config = CONFIG_3D_PERCEPTUAL

        # Check methodology from paper
        assert config.n_seeds == 1, f"3D should use 1 seed, got {config.n_seeds}"
        assert "psnr" in config.metrics, "3D should include PSNR"
        assert "ms_ssim" in config.metrics, "3D should include MS-SSIM"

        print(f"{GREEN}✓{RESET} 3D config matches paper:")
        print(f"  - Seeds: {config.n_seeds} (paper: 1)")
        print(f"  - Metrics: {config.metrics}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} 3D config failed: {e}")
        return False


def test_cad_config():
    """Test CAD task configuration."""
    print("\nTesting CAD Task Config...")

    try:
        config = CONFIG_CAD_TASKS

        # Check methodology from paper
        assert config.n_seeds == 3, f"CAD should use 3 seeds, got {config.n_seeds}"
        assert "auroc" in config.metrics, "CAD should include AUROC"

        print(f"{GREEN}✓{RESET} CAD config matches paper:")
        print(f"  - Seeds: {config.n_seeds} (paper: 3)")
        print(f"  - Metrics: {config.metrics}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} CAD config failed: {e}")
        return False


def test_sample_sizes():
    """Test sample sizes match paper."""
    print("\nTesting Sample Sizes...")

    try:
        assert SAMPLE_SIZES["2d_perceptual"] == 1000, "2D should use 1000 samples"
        assert SAMPLE_SIZES["3d_perceptual"] == 100, "3D should use 100 samples"
        assert SAMPLE_SIZES["reader_study"] == 50, "Reader study should use 50 samples"

        print(f"{GREEN}✓{RESET} Sample sizes match paper:")
        for key, size in SAMPLE_SIZES.items():
            print(f"  - {key}: {size}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Sample sizes failed: {e}")
        return False


def test_image_types():
    """Test image types are documented."""
    print("\nTesting Image Types...")

    try:
        # Check 2D image types
        expected_2d = ["mammograms", "chest_xrays", "musculoskeletal_xrays", "wrist_xrays_fg"]
        for img_type in expected_2d:
            assert img_type in IMAGE_TYPES_2D, f"Missing 2D type: {img_type}"

        # Check 3D image types
        expected_3d = ["brain_mris", "head_cts", "abdomen_cts", "ts_cts", "lung_cts", "knee_mris"]
        for img_type in expected_3d:
            assert img_type in IMAGE_TYPES_3D, f"Missing 3D type: {img_type}"

        print(f"{GREEN}✓{RESET} Image types documented:")
        print(f"  - 2D: {len(IMAGE_TYPES_2D)} types")
        print(f"  - 3D: {len(IMAGE_TYPES_3D)} types")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Image types failed: {e}")
        return False


def test_cad_tasks():
    """Test CAD tasks are documented."""
    print("\nTesting CAD Tasks...")

    try:
        assert "2d" in CAD_TASKS
        assert "3d" in CAD_TASKS

        # Check 2D tasks
        assert len(CAD_TASKS["2d"]) == 5, "Should have 5 2D CAD tasks"

        # Check 3D tasks
        assert len(CAD_TASKS["3d"]) == 3, "Should have 3 3D CAD tasks"

        print(f"{GREEN}✓{RESET} CAD tasks documented:")
        print(f"  - 2D tasks: {len(CAD_TASKS['2d'])}")
        print(f"  - 3D tasks: {len(CAD_TASKS['3d'])}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} CAD tasks failed: {e}")
        return False


def test_downsizing_factors():
    """Test downsizing factors."""
    print("\nTesting Downsizing Factors...")

    try:
        assert DOWNSIZING_FACTORS["2d"] == [16, 64], "2D should use f=16 and f=64"
        assert DOWNSIZING_FACTORS["3d"] == [64, 512], "3D should use f=64 and f=512"

        print(f"{GREEN}✓{RESET} Downsizing factors match paper:")
        print(f"  - 2D: {DOWNSIZING_FACTORS['2d']}")
        print(f"  - 3D: {DOWNSIZING_FACTORS['3d']}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Downsizing factors failed: {e}")
        return False


def test_latent_channels():
    """Test latent channel configurations."""
    print("\nTesting Latent Channels...")

    try:
        assert LATENT_CHANNELS["f16"] == [1, 3], "f=16 should use C=1 or C=3"
        assert LATENT_CHANNELS["f64"] == [1, 4], "f=64 should use C=1 or C=4"
        assert LATENT_CHANNELS["f512"] == [1], "f=512 should use C=1"

        print(f"{GREEN}✓{RESET} Latent channels match paper:")
        for f, channels in LATENT_CHANNELS.items():
            print(f"  - {f}: C={channels}")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Latent channels failed: {e}")
        return False


def test_get_paper_config():
    """Test get_paper_config function."""
    print("\nTesting get_paper_config()...")

    try:
        # Test all valid task types
        config_2d = get_paper_config("2d_perceptual")
        assert config_2d.n_seeds == 4

        config_3d = get_paper_config("3d_perceptual")
        assert config_3d.n_seeds == 1

        config_cad = get_paper_config("cad_tasks")
        assert config_cad.n_seeds == 3

        # Test invalid task type
        try:
            get_paper_config("invalid")
            print(f"{RED}✗{RESET} Should raise error for invalid task type")
            return False
        except ValueError:
            pass

        print(f"{GREEN}✓{RESET} get_paper_config() works correctly")
        return True
    except Exception as e:
        print(f"{RED}✗{RESET} get_paper_config failed: {e}")
        return False


def test_ms_ssim_metric():
    """Verify MS-SSIM is included (not regular SSIM)."""
    print("\nVerifying MS-SSIM metric...")

    try:
        # Check that configs use ms_ssim
        assert "ms_ssim" in CONFIG_2D_PERCEPTUAL.metrics, "2D should use MS-SSIM"
        assert "ms_ssim" in CONFIG_3D_PERCEPTUAL.metrics, "3D should use MS-SSIM"

        # Verify we can import it
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

        print(f"{GREEN}✓{RESET} MS-SSIM (not regular SSIM) is used")
        print(f"{YELLOW}ℹ{RESET} This is critical - paper uses Multi-Scale SSIM!")

        return True
    except Exception as e:
        print(f"{RED}✗{RESET} MS-SSIM verification failed: {e}")
        return False


def main():
    print(f"\n{'='*60}")
    print("Paper Validation Configuration Tests")
    print(f"{'='*60}")

    all_passed = True

    all_passed &= test_2d_config()
    all_passed &= test_3d_config()
    all_passed &= test_cad_config()
    all_passed &= test_sample_sizes()
    all_passed &= test_image_types()
    all_passed &= test_cad_tasks()
    all_passed &= test_downsizing_factors()
    all_passed &= test_latent_channels()
    all_passed &= test_get_paper_config()
    all_passed &= test_ms_ssim_metric()

    print(f"\n{'='*60}")
    if all_passed:
        print(f"{GREEN}✓ All paper configuration tests passed!{RESET}")
        print(f"\n{YELLOW}Key Points Verified:{RESET}")
        print(f"  • 2D: 1000 samples, 4 seeds, PSNR & MS-SSIM")
        print(f"  • 3D: 100 samples, 1 seed, PSNR & MS-SSIM")
        print(f"  • CAD: 3 seeds, AUROC")
        print(f"  • MS-SSIM (not SSIM) is used")
        return 0
    else:
        print(f"{RED}✗ Some tests failed.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
