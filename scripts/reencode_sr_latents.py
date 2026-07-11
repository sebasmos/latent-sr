#!/usr/bin/env python3
"""
reencode_sr_latents.py

Re-encode CXR validation SR pixel images through the MedVAE encoder to
produce proper validation-split SR latents.

Fixes GitHub issue #97: the existing sr_latents under
outputs/experiments/cxr_decoder_finetune/sr_latents/ are training-split
latents, not validation-split latents.

Input:
  SR pixel images: outputs/experiments/cxr_medvae_s1/sr_images/
                   Files: cxr_00000.png ... cxr_00999.png (1000 files)
  Each image is a grayscale 256x256 PNG.

Output:
  SR latents: outputs/experiments/cxr_sr_latents_valid/
              Files: sr_00000.npy ... sr_00999.npy
              Shape: (3, 64, 64) float32

Model:
  medvae_4_3_2d with modality xray.
  Uses model.encode(img) to get S1 (raw) latents — same convention as
  medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_paths import outputs_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-encode SR pixel images through MedVAE encoder to get validation SR latents"
    )
    parser.add_argument(
        "--sr-image-dir",
        type=Path,
        default=outputs_root() / "experiments/cxr_medvae_s1/sr_images",
        help="Directory containing validation SR pixel images (*.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs_root() / "experiments/cxr_sr_latents_valid",
        help="Directory to save re-encoded SR latents (sr_*.npy)",
    )
    parser.add_argument(
        "--medvae-model",
        type=str,
        default="medvae_4_3_2d",
        help="MedVAE model name",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="xray",
        help="MedVAE modality",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device",
    )
    return parser.parse_args()


def load_sr_image(path: Path) -> torch.Tensor:
    """
    Load a grayscale SR PNG, convert to RGB (3-channel), normalize to [-1, 1].
    Returns tensor of shape (3, H, W).
    """
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0  # [0, 1]
    tensor = torch.from_numpy(array).permute(2, 0, 1)     # (3, H, W)
    tensor = (tensor - 0.5) / 0.5                         # [-1, 1]
    return tensor


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Discover SR images and sort by filename
    sr_image_paths = sorted(args.sr_image_dir.glob("*.png"))
    if not sr_image_paths:
        raise RuntimeError(f"No PNG files found in {args.sr_image_dir}")
    print(f"Found {len(sr_image_paths)} SR images in {args.sr_image_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load MedVAE model
    print(f"\nLoading MedVAE model: {args.medvae_model} (modality={args.modality})...")
    from medvae import MVAE
    vae = MVAE(model_name=args.medvae_model, modality=args.modality).to(device)
    vae.requires_grad_(False)
    vae.eval()
    print(f"MedVAE loaded. Device: {device}")

    # Enable TF32 on Ampere GPUs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\nEncoding {len(sr_image_paths)} images (batch_size={args.batch_size})...")

    global_idx = 0
    n_batches = (len(sr_image_paths) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(sr_image_paths), args.batch_size), total=n_batches, desc="Encoding"):
            batch_end = min(batch_start + args.batch_size, len(sr_image_paths))
            batch_paths = sr_image_paths[batch_start:batch_end]

            # Load and stack batch: (B, 3, H, W), normalized to [-1, 1]
            tensors = [load_sr_image(p) for p in batch_paths]
            batch = torch.stack(tensors, dim=0).to(device)

            # Encode: model.encode() returns S1 (raw) latents, shape (B, C, H, W)
            latents = vae.encode(batch)

            # Ensure 4D: (B, C, H, W)
            if latents.ndim == 3:
                latents = latents.unsqueeze(0)

            latents_np = latents.cpu().numpy()  # (B, C, H, W)

            # Save each latent individually as (C, H, W) float32
            for i in range(latents_np.shape[0]):
                out_path = args.output_dir / f"sr_{global_idx:05d}.npy"
                np.save(str(out_path), latents_np[i].astype(np.float32))
                global_idx += 1

    print(f"\nDone. Saved {global_idx} SR latents to {args.output_dir}")

    # Sanity check first latent shape
    first_file = args.output_dir / "sr_00000.npy"
    if first_file.exists():
        shape = np.load(str(first_file)).shape
        print(f"Sample latent shape: {shape}  (expected: (3, 64, 64))")


if __name__ == "__main__":
    main()
