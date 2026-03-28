#!/usr/bin/env python3
"""
Evaluate VAE autoencoder reconstruction quality in isolation.

Loads HR PNGs, encodes them through the VAE latent space, decodes back to
pixel space, and compares the reconstruction against the original HR image.
This isolates the VAE reconstruction error from the diffusion prediction error.

Supported backends:
  - medvae: MedVAE encoder/decoder
  - sd-vae: Stable Diffusion VAE (stabilityai/sd-vae-ft-ema)

The output JSON intentionally mirrors `scripts/eval_diffusion_sr.py` so
`scripts/aggregate_results.py` can consume it without a separate schema.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VAE reconstruction quality (encode -> decode) on HR images"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="medvae",
        choices=["medvae", "sd-vae"],
        help="VAE backend: 'medvae' (default) or 'sd-vae' (Stable Diffusion VAE).",
    )
    parser.add_argument("--hr-dir", type=Path, required=True, help="Directory of HR PNG images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for results and optional images.")
    parser.add_argument("--medvae-model", type=str, default="medvae_4_3_2d", help="MedVAE model name.")
    parser.add_argument("--modality", type=str, default="mri", help="MedVAE modality.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save decoded grayscale images as PNGs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_hr_image(path: Path) -> torch.Tensor:
    """Load a grayscale HR PNG and return a [0, 1] float tensor of shape (1, H, W)."""
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def normalize_to_model_range(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] to [-1, 1] via (x - 0.5) / 0.5."""
    return (tensor - 0.5) / 0.5


def expand_to_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """Expand a (B, 1, H, W) tensor to (B, 3, H, W) by repeating the channel."""
    if tensor.shape[1] == 3:
        return tensor
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")


def ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has 3 channels for metrics that require it (e.g. LPIPS)."""
    if tensor.shape[1] == 3:
        return tensor
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")


def mean_to_grayscale(tensor: torch.Tensor) -> torch.Tensor:
    """Average across channels to produce a single-channel (B, 1, H, W) tensor."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] == 1:
        return tensor
    return tensor.mean(dim=1, keepdim=True)


def save_grayscale_png(tensor: torch.Tensor, path: Path) -> None:
    """Save a [0, 1] single-channel tensor as a uint8 grayscale PNG."""
    array = (tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def summarize_metric(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover HR images
    hr_paths = sorted(args.hr_dir.glob("*.png"))
    if not hr_paths:
        raise RuntimeError(f"No PNG files found in {args.hr_dir}")
    print(f"Found {len(hr_paths)} HR images in {args.hr_dir}")

    # Load VAE
    if args.backend == "sd-vae":
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        vae.eval()
        vae.requires_grad_(False)

        def encode_decode(img: torch.Tensor) -> torch.Tensor:
            z = vae.encode(img).latent_dist.sample()
            recon = vae.decode(z).sample
            return recon

        print("VAE backend: SD-VAE (stabilityai/sd-vae-ft-ema)")
    else:
        from medvae import MVAE

        vae = MVAE(args.medvae_model, args.modality).to(device)
        vae.eval()
        vae.requires_grad_(False)

        def encode_decode(img: torch.Tensor) -> torch.Tensor:
            z = vae.encode(img)
            recon = vae.decode(z)
            return recon

        print(f"VAE backend: MedVAE ({args.medvae_model})")

    # Initialize metrics
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    try:
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    except Exception as exc:
        lpips_fn = None
        print(f"WARNING: LPIPS disabled because initialization failed: {exc}")

    psnr_values: list[float] = []
    msssim_values: list[float] = []
    lpips_values: list[float] = []
    per_image_metrics: list[dict[str, float | str | None]] = []

    sr_img_dir = out_dir / "sr_images"
    if args.save_images:
        sr_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning VAE encode-decode ({len(hr_paths)} images, backend={args.backend})...\n")
    start_time = time.perf_counter()

    # Process in batches
    for batch_start in tqdm(range(0, len(hr_paths), args.batch_size), desc="Evaluating"):
        batch_end = min(batch_start + args.batch_size, len(hr_paths))
        batch_paths = hr_paths[batch_start:batch_end]

        # Load batch of HR images as grayscale [0,1], shape (B, 1, H, W)
        hr_tensors = [load_hr_image(p) for p in batch_paths]
        hr_batch = torch.stack(hr_tensors, dim=0).to(device)  # (B, 1, H, W)

        # Store original [0,1] grayscale for comparison
        gt_gray = hr_batch.clone()  # (B, 1, H, W), [0, 1]

        # Normalize to [-1, 1]
        hr_normalized = normalize_to_model_range(hr_batch)  # (B, 1, H, W), [-1, 1]

        # Determine input channels: 1-channel models (e.g. medvae_4_1_2d) take (B,1,H,W),
        # 3-channel models (medvae_4_3_2d, sd-vae) take (B,3,H,W)
        is_one_channel = (args.backend == "medvae" and "_1_" in args.medvae_model)
        if is_one_channel:
            vae_input = hr_normalized  # (B, 1, H, W)
        else:
            vae_input = expand_to_three_channels(hr_normalized)  # (B, 3, H, W)

        # Encode and decode
        with torch.no_grad():
            recon = encode_decode(vae_input)  # (B, C, H, W), approx [-1, 1]

        # Convert decoded output: [-1, 1] -> [0, 1], then average to grayscale
        recon_01 = ((recon + 1) / 2).clamp(0, 1)
        recon_gray = mean_to_grayscale(recon_01)  # (B, 1, H, W), [0, 1]

        # Compute per-image metrics
        for local_idx in range(gt_gray.shape[0]):
            global_idx = batch_start + local_idx
            image_id = batch_paths[local_idx].stem

            gt_i = gt_gray[local_idx : local_idx + 1]      # (1, 1, H, W)
            recon_i = recon_gray[local_idx : local_idx + 1]  # (1, 1, H, W)

            psnr_value = psnr_fn(recon_i, gt_i).item()
            psnr_values.append(psnr_value)

            msssim_value = None
            if gt_i.shape[-1] >= 160 and gt_i.shape[-2] >= 160:
                msssim_value = msssim_fn(recon_i, gt_i).item()
                msssim_values.append(msssim_value)

            lpips_value = None
            if lpips_fn is not None:
                gt_lpips = ensure_three_channels(gt_i)
                recon_lpips = ensure_three_channels(recon_i)
                lpips_value = lpips_fn(recon_lpips, gt_lpips).item()
                lpips_values.append(lpips_value)

            per_image_metrics.append(
                {
                    "id": image_id,
                    "psnr": psnr_value,
                    "msssim": msssim_value,
                    "lpips": lpips_value,
                }
            )

            if args.save_images:
                save_grayscale_png(recon_i, sr_img_dir / f"{image_id}.png")

    elapsed_seconds = time.perf_counter() - start_time
    psnr_mean, psnr_std = summarize_metric(psnr_values)
    msssim_mean, msssim_std = summarize_metric(msssim_values)
    lpips_mean, lpips_std = summarize_metric(lpips_values)

    results: dict[str, object] = {
        "n_samples": len(hr_paths),
        "timesteps": None,
        "backend": args.backend,
        "method": "vae_reconstruction",
        "source": "vae_reconstruction",
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / max(len(hr_paths), 1),
            "samples_per_second": len(hr_paths) / elapsed_seconds if elapsed_seconds else None,
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

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"VAE Reconstruction Results ({args.backend})")
    print(f"{'=' * 60}")
    if psnr_mean is not None:
        print(f"  PSNR:    {psnr_mean:.2f} +/- {psnr_std:.2f} dB")
    if msssim_mean is not None:
        print(f"  MS-SSIM: {msssim_mean:.4f} +/- {msssim_std:.4f}")
    if lpips_mean is not None:
        print(f"  LPIPS:   {lpips_mean:.4f} +/- {lpips_std:.4f}")
    print(f"  Elapsed: {elapsed_seconds:.2f}s for {len(hr_paths)} samples")
    print(f"  Speed:   {elapsed_seconds / max(len(hr_paths), 1):.3f}s per sample")

    outfile = out_dir / "diffusion_eval_results.json"
    with open(outfile, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
