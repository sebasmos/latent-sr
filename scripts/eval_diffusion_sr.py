#!/usr/bin/env python3
"""
Evaluate diffusion SR for the supported x0-only checkpoint.

Loads the Stage 3 model, runs the iterative cold-diffusion sampler, decodes with
MedVAE or SD-VAE, and computes image-space metrics.

Metrics contract:
  - Aggregate metrics live under `diffusion_sr`
  - Per-image metrics live under `per_image_metrics`
  - Timing metadata lives under `timing`

The main PSNR/MS-SSIM fields preserve the existing decoded-tensor behavior for
backward compatibility. Grayscale-averaged variants are included to support the
MS-SSIM anomaly investigation and to align with the saved PNG outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
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
        description="Evaluate diffusion SR with the supported x0-only sampler"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--medvae-model", type=str, default="medvae_4_3_2d")
    parser.add_argument("--modality", type=str, default="mri")
    parser.add_argument(
        "--backend",
        type=str,
        default="medvae",
        choices=["medvae", "sd-vae"],
        help="Decoder backend: 'medvae' (default) or 'sd-vae' (Stable Diffusion VAE).",
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--attention-resolutions",
        type=str,
        default=None,
        help="Comma-separated attention resolutions to override checkpoint defaults "
             "(e.g., '8' for old SD-VAE checkpoints trained before architecture flag changes).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save grayscale-averaged SR images as PNGs for downstream eval.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_eval")
    parser.add_argument(
        "--training-script",
        type=str,
        default=None,
        help="Path to the training script module to load LatentDiffusionSR from. "
             "Default: medvae_diffusion_pipeline/scripts/03_train_diffusion.py. "
             "Use 03_train_diffusion_featloss.py for feature-loss checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned decoder .pt file produced by 04_finetune_decoder.py. "
             "If provided, replaces the MedVAE model's decoder + post_quant_conv weights "
             "before running inference (Issue #76 decoder adaptation).",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default="LatentDiffusionSR",
        help="Model class name to load from the training script "
             "(default: LatentDiffusionSR; use LatentDiffusionCFG for CFG checkpoints).",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Path to labels CSV (requires filenames.txt in --latent-dir). "
             "When set, enables CFG conditional sampling with --guidance-scale.",
    )
    parser.add_argument(
        "--label-columns",
        type=str,
        default=None,
        help="Comma-separated label column names from --labels-csv "
             "(e.g. 'acl,meniscus,abnormal' for MRNet).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="CFG guidance scale gamma (default: 1.0 = unconditional; "
             "only used when --labels-csv is set).",
    )
    return parser.parse_args()


def mean_to_grayscale(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] == 1:
        return tensor
    return tensor.mean(dim=1, keepdim=True)


def ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 3:
        return tensor
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")


def summarize_metric(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def load_filename_mapping(latent_dir: Path) -> list[str] | None:
    filenames_path = latent_dir / "filenames.txt"
    if not filenames_path.exists():
        return None
    with open(filenames_path) as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_image_id(
    global_index: int,
    hr_file: str,
    orig_filenames: list[str] | None,
) -> str:
    if orig_filenames and global_index < len(orig_filenames):
        return orig_filenames[global_index]
    return hr_file.replace("hr_", "").replace(".npy", "")


def load_label_map(
    latent_dir: Path,
    labels_csv: str,
    label_columns: list[str],
) -> dict[int, torch.Tensor]:
    """Build index → float tensor map from filenames.txt + labels CSV (same as PairedLatentDataset)."""
    import csv

    filenames_txt = latent_dir / "filenames.txt"
    if not filenames_txt.exists():
        raise FileNotFoundError(
            f"filenames.txt not found in {latent_dir}. Required for CFG label loading."
        )
    with open(filenames_txt) as f:
        raw_filenames = [line.strip() for line in f if line.strip()]

    csv_lookup: dict = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = Path(row["filename"]).stem
            csv_lookup[key] = row

    label_map: dict[int, torch.Tensor] = {}
    missing = 0
    for idx, fname in enumerate(raw_filenames):
        stem = Path(fname).stem
        if stem in csv_lookup:
            row = csv_lookup[stem]
            vals = []
            for col in label_columns:
                v = float(row.get(col, 0.0))
                if v < 0:
                    v = 0.0
                vals.append(v)
            label_map[idx] = torch.tensor(vals, dtype=torch.float32)
        else:
            missing += 1
            label_map[idx] = torch.zeros(len(label_columns), dtype=torch.float32)

    if missing > 0:
        print(f"[label_map] Warning: {missing}/{len(raw_filenames)} entries had no CSV match.")
    print(f"[label_map] Loaded {len(label_map)} labels ({len(label_columns)} cols: {label_columns})")
    return label_map


def load_training_module(training_script_path: str | None = None):
    repo_root = Path(__file__).resolve().parents[1]
    if training_script_path:
        train_script = Path(training_script_path)
        if not train_script.is_absolute():
            train_script = repo_root / train_script
    else:
        train_script = repo_root / "medvae_diffusion_pipeline" / "scripts" / "03_train_diffusion.py"
    sys.path.insert(0, str(train_script.parent))
    spec = spec_from_file_location("train_03", str(train_script))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save_grayscale_png(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_dir = Path(args.latent_dir)
    hr_files = sorted(file for file in os.listdir(latent_dir) if file.startswith("hr_"))
    lr_files = sorted(file for file in os.listdir(latent_dir) if file.startswith("lr_"))
    if len(hr_files) != len(lr_files):
        raise ValueError(f"HR/LR count mismatch: {len(hr_files)} vs {len(lr_files)}")
    if not hr_files:
        raise RuntimeError(f"No HR/LR latents found in {latent_dir}")

    orig_filenames = load_filename_mapping(latent_dir)
    sample_hr = np.load(latent_dir / hr_files[0], mmap_mode="r")
    sample_lr = np.load(latent_dir / lr_files[0], mmap_mode="r")
    print(f"Latent pairs: {len(hr_files)}")
    print(f"HR: {sample_hr.shape}  LR: {sample_lr.shape}")
    if orig_filenames:
        print(f"Filename mapping: {len(orig_filenames)} entries")

    module = load_training_module(args.training_script)
    load_kwargs = {}
    if args.attention_resolutions is not None:
        load_kwargs["attention_resolutions"] = tuple(
            int(x) for x in args.attention_resolutions.split(",")
        )
    model_cls = getattr(module, args.model_class)
    ldm = model_cls.load_from_checkpoint(
        args.checkpoint, map_location=device, strict=False, **load_kwargs
    )
    ldm = ldm.to(device)
    ldm.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.backend == "sd-vae":
        from diffusers.models import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        vae.eval()
        vae.requires_grad_(False)
        decode_fn = lambda z: vae.decode(z).sample
        print("Decoder: SD-VAE (stabilityai/sd-vae-ft-ema)")
    else:
        from medvae import MVAE

        vae = MVAE(args.medvae_model, args.modality).to(device)
        vae.eval()
        vae.requires_grad_(False)
        decode_fn = lambda z: vae.decode(z)
        print(f"Decoder: MedVAE ({args.medvae_model})")

    # Optionally load fine-tuned decoder weights (Issue #76)
    if args.decoder_checkpoint is not None:
        if args.backend != "medvae":
            raise ValueError("--decoder-checkpoint is only supported with --backend medvae")
        print(f"Loading fine-tuned decoder from: {args.decoder_checkpoint}")
        ckpt = torch.load(args.decoder_checkpoint, map_location=device)
        vae.model.decoder.load_state_dict(ckpt["decoder"])
        vae.model.post_quant_conv.load_state_dict(ckpt["post_quant_conv"])
        vae.eval()
        vae.requires_grad_(False)
        print(f"  Fine-tuned decoder loaded (variant={ckpt.get('variant', 'unknown')})")

    # CFG label loading (issue #52)
    label_map: dict = {}
    label_columns: list[str] = []
    if args.labels_csv is not None and args.label_columns is not None:
        label_columns = [c.strip() for c in args.label_columns.split(",") if c.strip()]
        label_map = load_label_map(latent_dir, args.labels_csv, label_columns)
        print(f"CFG guidance scale: {args.guidance_scale}")

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    try:
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    except Exception as exc:
        lpips_fn = None
        print(f"WARNING: LPIPS disabled because initialization failed: {exc}")

    psnr_values: list[float] = []
    psnr_gray_values: list[float] = []
    msssim_values: list[float] = []
    msssim_gray_values: list[float] = []
    lpips_values: list[float] = []
    per_image_metrics: list[dict[str, float | str | None]] = []

    sr_img_dir = out_dir / "sr_images"
    if args.save_images:
        sr_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference ({len(hr_files)} samples, T={args.timesteps})...\n")
    start_time = time.perf_counter()

    for start in tqdm(range(0, len(hr_files), args.batch_size), desc="Evaluating"):
        end = min(start + args.batch_size, len(hr_files))

        hr_lat = torch.stack(
            [torch.from_numpy(np.load(latent_dir / hr_files[idx])).float() for idx in range(start, end)]
        ).to(device)
        lr_lat = torch.stack(
            [torch.from_numpy(np.load(latent_dir / lr_files[idx])).float() for idx in range(start, end)]
        ).to(device)

        if label_map:
            labels_batch = torch.stack(
                [label_map.get(idx, torch.zeros(len(label_columns))) for idx in range(start, end)]
            ).to(device)
            pred_hr = ldm.sample(
                lr_lat, T=args.timesteps,
                class_labels=labels_batch,
                guidance_scale=args.guidance_scale,
            )
        else:
            pred_hr = ldm.sample(lr_lat, T=args.timesteps)

        gt_pix = decode_fn(hr_lat)
        sr_pix = decode_fn(pred_hr)

        gt_01 = ((gt_pix + 1) / 2).clamp(0, 1)
        sr_01 = ((sr_pix + 1) / 2).clamp(0, 1)
        if gt_01.ndim == 3:
            gt_01 = gt_01.unsqueeze(0)
            sr_01 = sr_01.unsqueeze(0)

        gt_gray = mean_to_grayscale(gt_01)
        sr_gray = mean_to_grayscale(sr_01)
        gt_lpips = ensure_three_channels(gt_gray)
        sr_lpips = ensure_three_channels(sr_gray)

        for local_idx in range(gt_01.shape[0]):
            global_idx = start + local_idx
            image_id = resolve_image_id(global_idx, hr_files[global_idx], orig_filenames)

            gt_full = gt_01[local_idx : local_idx + 1]
            sr_full = sr_01[local_idx : local_idx + 1]
            gt_gray_i = gt_gray[local_idx : local_idx + 1]
            sr_gray_i = sr_gray[local_idx : local_idx + 1]
            gt_lpips_i = gt_lpips[local_idx : local_idx + 1]
            sr_lpips_i = sr_lpips[local_idx : local_idx + 1]

            psnr_value = psnr_fn(sr_full, gt_full).item()
            gray_psnr_value = psnr_fn(sr_gray_i, gt_gray_i).item()
            lpips_value = None
            if lpips_fn is not None:
                lpips_value = lpips_fn(sr_lpips_i, gt_lpips_i).item()

            psnr_values.append(psnr_value)
            psnr_gray_values.append(gray_psnr_value)
            if lpips_value is not None:
                lpips_values.append(lpips_value)

            msssim_value = None
            if gt_full.shape[-1] >= 160 and gt_full.shape[-2] >= 160:
                msssim_value = msssim_fn(sr_full, gt_full).item()
                msssim_values.append(msssim_value)

            gray_msssim_value = None
            if gt_gray_i.shape[-1] >= 160 and gt_gray_i.shape[-2] >= 160:
                gray_msssim_value = msssim_fn(sr_gray_i, gt_gray_i).item()
                msssim_gray_values.append(gray_msssim_value)

            per_image_metrics.append(
                {
                    "id": image_id,
                    "psnr": psnr_value,
                    "psnr_grayscale": gray_psnr_value,
                    "msssim": msssim_value,
                    "msssim_grayscale": gray_msssim_value,
                    "lpips": lpips_value,
                }
            )

            if args.save_images:
                save_grayscale_png(sr_gray_i, sr_img_dir / f"{image_id}.png")

    elapsed_seconds = time.perf_counter() - start_time
    psnr_mean, psnr_std = summarize_metric(psnr_values)
    psnr_gray_mean, psnr_gray_std = summarize_metric(psnr_gray_values)
    msssim_mean, msssim_std = summarize_metric(msssim_values)
    msssim_gray_mean, msssim_gray_std = summarize_metric(msssim_gray_values)
    lpips_mean, lpips_std = summarize_metric(lpips_values)

    results: dict[str, object] = {
        "n_samples": len(hr_files),
        "timesteps": args.timesteps,
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_sample": elapsed_seconds / max(len(hr_files), 1),
            "samples_per_second": len(hr_files) / elapsed_seconds if elapsed_seconds else None,
        },
        "diffusion_sr": {
            "psnr_mean": psnr_mean,
            "psnr_std": psnr_std,
            "psnr_grayscale_mean": psnr_gray_mean,
            "psnr_grayscale_std": psnr_gray_std,
            "lpips_mean": lpips_mean,
            "lpips_std": lpips_std,
        },
        "per_image_metrics": per_image_metrics,
    }
    if msssim_mean is not None:
        results["diffusion_sr"]["msssim_mean"] = msssim_mean
        results["diffusion_sr"]["msssim_std"] = msssim_std
    if msssim_gray_mean is not None:
        results["diffusion_sr"]["msssim_grayscale_mean"] = msssim_gray_mean
        results["diffusion_sr"]["msssim_grayscale_std"] = msssim_gray_std

    print(f"\n{'=' * 60}")
    print("Results (pixel-space)")
    print(f"{'=' * 60}")
    print(f"  Diffusion SR PSNR: {psnr_mean:.2f} +/- {psnr_std:.2f} dB")
    if msssim_mean is not None:
        print(f"  Diffusion SR MS-SSIM: {msssim_mean:.4f}")
    if lpips_mean is not None:
        print(f"  Diffusion SR LPIPS: {lpips_mean:.4f} +/- {lpips_std:.4f}")
    print(f"  Elapsed: {elapsed_seconds:.2f}s for {len(hr_files)} samples")

    outfile = out_dir / "diffusion_eval_results.json"
    with open(outfile, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
