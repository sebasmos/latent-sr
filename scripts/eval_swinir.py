#!/usr/bin/env python3
"""
Evaluate SwinIR pretrained SR baseline against HR images.

SwinIR is a Transformer-based pixel-space SR model (Liang et al., 2021).
Uses Real-SR pretrained weights: 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth

No basicsr dependency — clones the SwinIR repo directly on first run.

Output JSON mirrors eval_baselines.py / eval_diffusion_sr.py so that
aggregate_results.py can consume it without schema changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from tqdm import tqdm

SWINIR_REPO_URL = "https://github.com/JingyunLiang/SwinIR.git"
SWINIR_WEIGHTS_URL = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
)
WINDOW_SIZE = 8
SCALE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SwinIR SR baseline")
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument("--lr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke runs.")
    parser.add_argument("--save-images", action="store_true", help="Save generated SR PNGs.")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Explicit path to SwinIR .pth weights file (skips download).",
    )
    return parser.parse_args()


def load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def match_pairs(hr_dir: Path, lr_dir: Path) -> list[tuple[Path, Path, str]]:
    hr_by_stem = {path.stem: path for path in sorted(hr_dir.glob("*.png"))}
    lr_by_stem = {path.stem: path for path in sorted(lr_dir.glob("*.png"))}
    common = sorted(set(hr_by_stem).intersection(lr_by_stem))
    if not common:
        raise RuntimeError(f"No matching PNG filenames found between {hr_dir} and {lr_dir}")
    return [(hr_by_stem[name], lr_by_stem[name], name) for name in common]


def ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    if tensor.shape[1] == 3:
        return tensor
    raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


class SwinIRRunner:
    def __init__(self, device: str, weights_path: Path | None = None) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        swinir_root = repo_root / "SwinIR"

        # Clone SwinIR repo if not present (no basicsr needed)
        if not (swinir_root / "models" / "network_swinir.py").exists():
            print(f"Cloning SwinIR to {swinir_root} ...")
            subprocess.run(
                ["git", "clone", "--depth=1", SWINIR_REPO_URL, str(swinir_root)],
                check=True,
            )

        sys.path.insert(0, str(swinir_root))

        # timm imports wandb in timm.utils.summary, and wandb has a numpy 2.0
        # incompatibility (np.float_ removed). Mock wandb before importing timm/SwinIR
        # so the import chain succeeds — we never call wandb at runtime.
        import types as _types  # noqa: PLC0415
        if "wandb" not in sys.modules:
            sys.modules["wandb"] = _types.ModuleType("wandb")

        from models.network_swinir import SwinIR  # noqa: PLC0415

        # Resolve weights
        if weights_path is None:
            weights_dir = swinir_root / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            weights_path = weights_dir / "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
            if not weights_path.exists():
                print(f"Downloading SwinIR weights to {weights_path} ...")
                torch.hub.download_url_to_file(SWINIR_WEIGHTS_URL, str(weights_path), progress=True)

        # Real-SR large variant (4x)
        self.model = SwinIR(
            upscale=SCALE,
            in_chans=3,
            img_size=64,
            window_size=WINDOW_SIZE,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="3conv",
        )
        pretrained = torch.load(weights_path, map_location="cpu")
        # Weights may be stored under 'params_ema', 'params', or at the top level
        state_dict = pretrained.get("params_ema", pretrained.get("params", pretrained))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(device)
        self.device = device
        print(f"SwinIR loaded on {device} from {weights_path}")

    def upscale(self, lr_path: Path) -> torch.Tensor:
        """Return a [1, H*4, W*4] float32 tensor in [0, 1]."""
        img = Image.open(lr_path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Replicate grayscale to 3-channel RGB for SwinIR
        arr_rgb = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
        tensor = torch.from_numpy(arr_rgb).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Pad to multiple of window_size
        _, _, h, w = tensor.shape
        pad_h = (WINDOW_SIZE - h % WINDOW_SIZE) % WINDOW_SIZE
        pad_w = (WINDOW_SIZE - w % WINDOW_SIZE) % WINDOW_SIZE
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            output = self.model(tensor)  # (1, 3, H*4, W*4)

        # Crop padding
        output = output[:, :, : h * SCALE, : w * SCALE]
        # Average RGB channels → grayscale
        gray = output.mean(dim=1, keepdim=True).squeeze(0).cpu()  # (1, H*4, W*4)
        return gray.clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    pairs = match_pairs(args.hr_dir, args.lr_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sr_dir = args.output_dir / "sr_images"
    if args.save_images:
        sr_dir.mkdir(parents=True, exist_ok=True)

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0)
    msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    try:
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    except Exception as exc:
        lpips_fn = None
        print(f"WARNING: LPIPS disabled: {exc}")

    runner = SwinIRRunner(device=args.device, weights_path=args.weights_path)

    per_image_metrics: list[dict] = []
    psnr_values: list[float] = []
    msssim_values: list[float] = []
    lpips_values: list[float] = []

    start_time = time.perf_counter()
    for hr_path, lr_path, image_id in tqdm(pairs, desc="SwinIR"):
        gt = load_image_tensor(hr_path)
        pred = runner.upscale(lr_path)

        # Resize to HR dimensions if shape mismatch (should not happen for 4x)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred_pil = Image.fromarray(
                (pred.squeeze().numpy() * 255.0).astype(np.uint8), mode="L"
            )
            pred = torch.from_numpy(
                np.asarray(
                    pred_pil.resize((gt.shape[-1], gt.shape[-2]), resample=Image.Resampling.BICUBIC),
                    dtype=np.float32,
                )
                / 255.0
            ).unsqueeze(0)

        gt_b = gt.unsqueeze(0)
        pred_b = pred.unsqueeze(0)

        psnr_value = psnr_fn(pred_b, gt_b).item()
        msssim_value = None
        if gt.shape[-1] >= 160 and gt.shape[-2] >= 160:
            msssim_value = msssim_fn(pred_b, gt_b).item()
            msssim_values.append(msssim_value)
        lpips_value = None
        if lpips_fn is not None:
            lpips_value = lpips_fn(
                ensure_three_channels(pred_b), ensure_three_channels(gt_b)
            ).item()

        psnr_values.append(psnr_value)
        if lpips_value is not None:
            lpips_values.append(lpips_value)
        per_image_metrics.append(
            {
                "id": image_id,
                "psnr": psnr_value,
                "psnr_grayscale": psnr_value,
                "msssim": msssim_value,
                "msssim_grayscale": msssim_value,
                "lpips": lpips_value,
            }
        )
        if args.save_images:
            save_image_tensor(pred, sr_dir / f"{image_id}.png")

    elapsed_seconds = time.perf_counter() - start_time
    psnr_mean, psnr_std = summarize(psnr_values)
    msssim_mean, msssim_std = summarize(msssim_values)
    lpips_mean, lpips_std = summarize(lpips_values)

    results: dict = {
        "n_samples": len(pairs),
        "timesteps": None,
        "backend": "swinir",
        "method": "swinir",
        "source": "baseline",
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / max(len(pairs), 1),
            "samples_per_second": len(pairs) / elapsed_seconds if elapsed_seconds else None,
        },
        "diffusion_sr": {
            "psnr_mean": psnr_mean,
            "psnr_std": psnr_std,
            "psnr_grayscale_mean": psnr_mean,
            "psnr_grayscale_std": psnr_std,
            "lpips_mean": lpips_mean,
            "lpips_std": lpips_std,
        },
        "per_image_metrics": per_image_metrics,
    }
    if msssim_mean is not None:
        results["diffusion_sr"]["msssim_mean"] = msssim_mean
        results["diffusion_sr"]["msssim_std"] = msssim_std
        results["diffusion_sr"]["msssim_grayscale_mean"] = msssim_mean
        results["diffusion_sr"]["msssim_grayscale_std"] = msssim_std

    outfile = args.output_dir / "diffusion_eval_results.json"
    with open(outfile, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved to {outfile}")
    print(f"PSNR:    {psnr_mean:.2f} ± {psnr_std:.2f} dB")
    if msssim_mean is not None:
        print(f"MS-SSIM: {msssim_mean:.4f} ± {msssim_std:.4f}")
    if lpips_mean is not None:
        print(f"LPIPS:   {lpips_mean:.4f} ± {lpips_std:.4f}")


if __name__ == "__main__":
    main()
