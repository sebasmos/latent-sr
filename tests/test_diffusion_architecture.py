#!/usr/bin/env python3
"""
Source-level checks for the diffusion training architecture.

These tests avoid importing diffusers or requiring a GPU. They verify the current
training script keeps the LDM-style spatial conditioning path that replaced the
older cross-attention implementation.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CURRENT_SCRIPT = ROOT / "medvae_diffusion_pipeline" / "scripts" / "03_train_diffusion.py"
OLD_03B_COMMIT = "3c8e4bd"
OLD_03B_PATH = "medvae_diffusion_pipeline/scripts/03b_train_latent_diffusion.py"


def read_current_script() -> str:
    return CURRENT_SCRIPT.read_text()


def read_old_script() -> str | None:
    try:
        result = subprocess.run(
            ["git", "show", f"{OLD_03B_COMMIT}:{OLD_03B_PATH}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def assert_true(condition: bool, message: str) -> None:
    if condition:
        print(f"PASS: {message}")
        return
    raise AssertionError(f"FAIL: {message}")


def test_current_script_uses_spatial_concat() -> None:
    print("\n1. Testing current diffusion training architecture...")
    text = read_current_script()

    assert_true(
        "from diffusers import UNet2DModel" in text,
        "current script uses UNet2DModel",
    )
    assert_true(
        "UNet2DConditionModel" not in text,
        "current script does not use cross-attention conditioning",
    )
    assert_true(
        'in_channels=hr_channels + lr_channels' in text,
        "current script configures U-Net input as noise + LR channels",
    )
    assert_true(
        'model_input = torch.cat([noisy_hr, lr_upsampled], dim=1)' in text,
        "current script concatenates noisy HR with spatial LR latents",
    )
    assert_true(
        "cond_proj" not in text and "encoder_hidden_states" not in text,
        "current script has no flattened conditioning token path",
    )
    assert_true(
        "--prediction-mode" not in text and 'default="l1"' in text,
        "current script keeps the reverted x0-only training path as the default",
    )
    assert_true(
        'self.register_buffer("betas"' not in text and "def _ddim_sample(" not in text,
        "current script does not expose the reverted DDPM/DDIM epsilon path",
    )
    assert_true(
        '--beta-schedule' in text and '--use-ema' in text and '--use-snr-weighting' in text,
        "current script exposes low-risk LDM-style ablation flags",
    )
    assert_true(
        'default=(1, 2, 4, 4)' in text and 'default=(16,)' in text,
        "current script preserves the default architecture through explicit flags",
    )


def test_historical_03b_was_cross_attention_based() -> None:
    print("\n2. Testing historical 03b architecture...")
    text = read_old_script()
    if text is None:
        print("SKIP: could not load deleted 03b script from git history")
        return

    assert_true(
        "from diffusers import UNet2DConditionModel" in text,
        "historical 03b used UNet2DConditionModel",
    )
    assert_true(
        "self.cond_proj = torch.nn.Linear(" in text,
        "historical 03b projected LR latents through a linear layer",
    )
    assert_true(
        "cond_input.view(bs, -1)" in text and ".unsqueeze(1)" in text,
        "historical 03b flattened LR latents into a single token",
    )
    assert_true(
        "encoder_hidden_states=cond" in text,
        "historical 03b passed conditioning through cross-attention",
    )


def main() -> int:
    all_passed = True
    try:
        test_current_script_uses_spatial_concat()
        test_historical_03b_was_cross_attention_based()
    except AssertionError as exc:
        print(str(exc))
        all_passed = False

    print("\n========================================")
    if all_passed:
        print("PASS: Diffusion architecture checks passed")
        return 0
    print("FAIL: Diffusion architecture checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
