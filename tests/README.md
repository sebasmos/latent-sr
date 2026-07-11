# Test Suite

This directory contains comprehensive tests to verify that all environment setup and validation framework components are working correctly.

## Quick Start

Run all tests:
```bash
./tests/run_all_tests.sh
```

Or run individual tests:
```bash
# Test installation files
python tests/test_installation.py

# Test environment setup
python tests/test_environment.py

# Test validation framework
python tests/test_validation_framework.py

# Test paper validation config
python tests/test_paper_validation.py

# Test revision-cycle numbers (capacity-matched controls, KL-f4 control,
# seed sweep, target-dataset adaptation, patient-level reanalysis, regression)
python tests/test_revision_validation.py
```

## Test Descriptions

### 1. `test_installation.py`
**Purpose:** Verify all installation files exist and are valid

**Tests:**
- ✓ `requirements.txt` exists and contains all required packages
- ✓ `environment.yml` is valid YAML with correct structure
- ✓ `pyproject.toml` has all required sections
- ✓ `setup.sh` exists and is executable
- ✓ All documentation files exist
- ✓ All validation scripts exist

**Run:**
```bash
python tests/test_installation.py
```

### 2. `test_environment.py`
**Purpose:** Test that the Python environment is correctly set up

**Tests:**
- ✓ Python version (3.11+)
- ✓ PyTorch and CUDA availability
- ✓ All core dependencies (numpy, scipy, pandas, etc.)
- ✓ Deep learning libraries (diffusers, transformers, accelerate)
- ✓ Medical imaging packages (MONAI, nibabel, SimpleITK)
- ✓ Image processing (Pillow, OpenCV)
- ✓ Real-ESRGAN dependencies
- ✓ Validation metrics (PSNR, SSIM, MS-SSIM, LPIPS, AUROC)

**Run:**
```bash
python tests/test_environment.py
```

**Expected Output:**
```
Python Version:
✓ Python 3.11.14

PyTorch Ecosystem:
✓ torch: v2.5.0
✓ torchvision: v0.20.0
✓ CUDA: 12.4 (GPU available)

Validation Metrics:
✓ All image metrics available
✓ All classification metrics available
```

### 3. `test_validation_framework.py`
**Purpose:** Test the validation framework functionality

**Tests:**
- ✓ `ValidationConfig` creation and serialization
- ✓ `MetricsCalculator` computes all metrics correctly
- ✓ Dataset splitting (train/val/test)
- ✓ K-fold cross-validation splitting
- ✓ Seed reproducibility
- ✓ Model evaluation pipeline

**Run:**
```bash
python tests/test_validation_framework.py
```

**Expected Output:**
```
Testing MetricsCalculator...
✓ All metrics computed correctly:
  - psnr: 12.3456
  - ssim: 0.7890
  - ms_ssim: 0.8123
  - lpips: 0.2345
  - mse: 0.0567
  - mae: 0.1234
```

### 4. `test_paper_validation.py`
**Purpose:** Verify paper validation configuration matches the methodology

**Tests:**
- ✓ 2D config uses 4 seeds (as per paper)
- ✓ 3D config uses 1 seed (as per paper)
- ✓ CAD config uses 3 seeds (as per paper)
- ✓ Sample sizes: 1000 for 2D, 100 for 3D
- ✓ Image types documented (4 for 2D, 6 for 3D)
- ✓ CAD tasks documented (5 for 2D, 3 for 3D)
- ✓ Downsizing factors (f=16,64 for 2D; f=64,512 for 3D)
- ✓ Latent channels configurations
- ✓ **MS-SSIM** is used (not regular SSIM)

**Run:**
```bash
python tests/test_paper_validation.py
```

**Expected Output:**
```
✓ All paper configuration tests passed!

Key Points Verified:
  • 2D: 1000 samples, 4 seeds, PSNR & MS-SSIM
  • 3D: 100 samples, 1 seed, PSNR & MS-SSIM
  • CAD: 3 seeds, AUROC
  • MS-SSIM (not SSIM) is used
```

### `test_revision_validation.py`
**Purpose:** Golden-number consistency checks for the major-revision results (transcribed
from the manuscript/supplementary, not re-run) — catches a mistyped digit before it
propagates into the README or a downstream script.

**Tests:**
- ✓ Headline true-HR PSNR gap is positive and CXR > BraTS > MRNet, all Cohen's d > 0.8
- ✓ Capacity-matched decomposition: dominant contributor is domain (MRNet) / resolution
  (BraTS) / capacity (CXR); domain stays < 1 dB even as an upper bound
- ✓ KL-f4 natural control: domain@64² is real (0.5-1.5 dB) but always smaller than the
  geometry gain
- ✓ Three-seed sweep SD is >=5x smaller than any headline effect
- ✓ Three-tier ladder (pretraining ~0 dB -> adaptation ~1 dB -> geometry) holds, with the
  documented MRNet exception (0.26 dB margin) checked explicitly
- ✓ Adaptation pass-through percentages are bounded in (0, 100); significance > 5σ
- ✓ Patient-level BraTS reanalysis (n=35) remains significant, delta favours MedVAE SR
- ✓ Expanded n=15 regression CI contains the original n=6 Fig. 3 point estimate

**Run:**
```bash
python tests/test_revision_validation.py
```

## Test Coverage

| Component | Test Coverage | Status |
|-----------|---------------|--------|
| Installation files | 100% | ✓ |
| Environment setup | 100% | ✓ |
| Validation config | 100% | ✓ |
| Metrics calculator | 100% | ✓ |
| Dataset splitting | 100% | ✓ |
| Model evaluation | 100% | ✓ |
| Paper methodology | 100% | ✓ |
| Revision-cycle numbers | 100% | ✓ |

## Troubleshooting

### Tests fail due to missing packages

Make sure you've installed the environment:
```bash
# With UV
uv pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml
conda activate medvae-sr
```

### CUDA tests show warnings

If you see:
```
⚠ CUDA: 12.4 (No GPU available - CPU only)
```

This is normal if you're running on a machine without a GPU. The tests will still pass.

### Import errors

Make sure you're running from the repository root:
```bash
cd /home/user/SR
python tests/test_validation_framework.py
```

## Expected Test Results

When all tests pass, you should see:

```
========================================
Installation Files Test
========================================
✓ requirements.txt valid (40+ packages)
✓ environment.yml valid
✓ pyproject.toml valid
✓ setup.sh valid and executable
✓ All documentation files exist
✓ All validation scripts exist

========================================
✓ All installation files are present and valid!

========================================
Environment Setup Test
========================================
✓ Python 3.11.14
✓ PyTorch: 2.5.0
✓ All metrics available

========================================
✓ All environment tests passed!

========================================
Validation Framework Tests
========================================
✓ All validation framework tests passed!

========================================
Paper Validation Configuration Tests
========================================
✓ All paper configuration tests passed!

Key Points Verified:
  • 2D: 1000 samples, 4 seeds, PSNR & MS-SSIM
  • 3D: 100 samples, 1 seed, PSNR & MS-SSIM
  • CAD: 3 seeds, AUROC
  • MS-SSIM (not SSIM) is used
```

## Adding New Tests

To add a new test:

1. Create `test_your_feature.py` in this directory
2. Follow the same structure as existing tests
3. Add it to `run_all_tests.sh`
4. Make it executable: `chmod +x tests/test_your_feature.py`

Example test structure:
```python
#!/usr/bin/env python3
import sys

def test_something():
    try:
        # Your test logic
        print("✓ Test passed")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    all_passed = test_something()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    ./tests/run_all_tests.sh
```

## Support

If tests fail:
1. Check the error messages
2. Verify environment is activated
3. Check installation files exist
4. Review the documentation in `INSTALLATION.md`
