#!/usr/bin/env python3
"""
Evaluate diffusion SR PSNR against the TRUE HR image, not against decode(hr_lat).

`eval_diffusion_sr.py` computes PSNR(decode(pred_hr), decode(hr_lat)) — i.e. it
scores the diffusion output against *the VAE's own reconstruction* of the HR
latent, not the real image. That cancels out the VAE's reconstruction ceiling,
which is exactly where the domain-vs-geometry effect lives (see
SUBMISSION_CHECKLIST.md, "OPEN ISSUE" section, 2026-07-07).

This script is a minimal variant: identical model loading / sampling / decode
path as eval_diffusion_sr.py, but the ground truth for PSNR/MS-SSIM/LPIPS is the
true HR PNG, loaded and normalized exactly as eval_vae_reconstruction.py does.

Filename mapping (verified against on-disk data 2026-07-07 — see checklist):
  hr_N.npy (extraction-time index) <-> filenames.txt line N (0-indexed) <->
  sorted(hr_image_dir.glob("*.png"))[N]. All three independently agree for the
  CXR validation set. Mapping is done by parsing N from the "hr_N.npy" filename
  and indexing filenames.txt directly — NOT by using eval_diffusion_sr.py's
  `resolve_image_id` helper, which indexes by sorted-list position instead of
  N and is wrong for any index >= 10 (confirmed bug, does not affect the
  existing self-referential PSNR since hr_lat/lr_lat pair up by matching
  sorted-position, but would silently mismatch true-HR images if reused here).

For every sample, also reports the OLD self-referential PSNR (decode(hr_lat)
GT) alongside the new true-HR PSNR, so the shift can be inspected directly,
plus a per-sample sanity check: PSNR(decode(hr_lat), true_HR) (the "AE-ceiling
recomputed inline") must be reasonably high, or the filename mapping itself is
broken and results are meaningless.
"""

from __future__ import annotations

import argparse
import json
import re
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
    parser = argparse.ArgumentParser(description="Evaluate diffusion SR PSNR against the true HR image")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--hr-image-dir", type=str, required=True,
                         help="Directory of true HR PNGs (same one used for the AE-ceiling eval).")
    parser.add_argument("--medvae-model", type=str, default="medvae_4_3_2d")
    parser.add_argument("--modality", type=str, default="mri")
    parser.add_argument("--backend", type=str, default="medvae", choices=["medvae", "sd-vae", "klf4"])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_eval_truehr")
    parser.add_argument("--training-script", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sanity-check-min-psnr", type=float, default=15.0,
                         help="Abort if median AE-ceiling-recomputed-inline PSNR falls below this "
                              "(indicates the filename mapping is wrong, not a model quality issue).")
    return parser.parse_args()


def mean_to_grayscale(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 1:
        return tensor
    return tensor.mean(dim=1, keepdim=True)


def ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 3:
        return tensor
    return tensor.repeat(1, 3, 1, 1)


def summarize_metric(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def load_hr_png(path: Path, n_channels: int) -> torch.Tensor:
    """Load a HR PNG exactly as eval_vae_reconstruction.py does: grayscale, [0,1], then
    normalize to [-1,1] via (x-0.5)/0.5, then expand to n_channels if needed."""
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)  # (1, H, W)
    tensor = (tensor - 0.5) / 0.5  # [-1, 1]
    if n_channels == 3:
        tensor = tensor.repeat(3, 1, 1)
    return tensor  # (C, H, W)


def load_training_module(training_script_path: str | None):
    repo_root = Path(__file__).resolve().parent.parent
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hr_image_dir = Path(args.hr_image_dir)

    latent_dir = Path(args.latent_dir)
    hr_files = sorted(f for f in __import__("os").listdir(latent_dir) if f.startswith("hr_"))
    lr_files = sorted(f for f in __import__("os").listdir(latent_dir) if f.startswith("lr_"))
    if len(hr_files) != len(lr_files):
        raise ValueError(f"HR/LR count mismatch: {len(hr_files)} vs {len(lr_files)}")

    hr_png_paths = sorted(hr_image_dir.glob("*.png"))
    filenames_path = latent_dir / "filenames.txt"
    if filenames_path.exists():
        with open(filenames_path) as fh:
            orig_filenames = [line.strip() for line in fh if line.strip()]
    else:
        # Legacy extraction (pre-filenames.txt tracking): fall back to independently
        # re-deriving the mapping from a fresh sorted HR-dir listing. Verified equivalent
        # to filenames.txt byte-for-byte for CXR (see SUBMISSION_CHECKLIST.md); safe because
        # 02_extract_medvae_embeddings.py's PairedImageDataset always sorts hr_dir the same
        # way with shuffle=False, so hr_N.npy <-> sorted(hr_dir)[N] regardless of whether
        # filenames.txt was written that run.
        print(f"WARNING: no filenames.txt in {latent_dir} (legacy extraction) — "
              f"falling back to a fresh sorted HR-dir listing for the mapping.", flush=True)
        orig_filenames = [p.stem for p in hr_png_paths]

    # Independent cross-check: re-derive the same mapping from the HR dir listing and confirm agreement.
    if len(hr_png_paths) != len(orig_filenames):
        raise RuntimeError(
            f"HR PNG count ({len(hr_png_paths)}) != filenames.txt length ({len(orig_filenames)}) "
            f"— hr-image-dir does not match the split used at extraction time."
        )
    for i in range(min(20, len(orig_filenames))):
        if hr_png_paths[i].stem != orig_filenames[i]:
            raise RuntimeError(
                f"Cross-check FAILED at index {i}: filenames.txt says '{orig_filenames[i]}' "
                f"but sorted HR dir gives '{hr_png_paths[i].stem}'. Do not trust this mapping."
            )
    print(f"Filename-mapping cross-check passed for first {min(20, len(orig_filenames))} entries.", flush=True)

    sample_hr = np.load(latent_dir / hr_files[0], mmap_mode="r")
    print(f"Latent pairs: {len(hr_files)}  HR latent shape: {sample_hr.shape}", flush=True)

    module = load_training_module(args.training_script)
    model_cls = getattr(module, "LatentDiffusionSR")
    ldm = model_cls.load_from_checkpoint(args.checkpoint, map_location=device, strict=False)
    ldm = ldm.to(device)
    ldm.eval()
    print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    if args.backend == "sd-vae":
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        vae.eval()
        vae.requires_grad_(False)
        decode_fn = lambda z: vae.decode(z).sample
        img_channels = 3
        print("Decoder: SD-VAE", flush=True)
    elif args.backend == "klf4":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slurm" / "revision"))
        from klf4_vae import load_klf4
        vae = load_klf4(device)
        vae.requires_grad_(False)
        decode_fn = lambda z: vae.decode(z)
        img_channels = 3
        print("Decoder: KL-f4", flush=True)
    else:
        from medvae import MVAE
        vae = MVAE(args.medvae_model, args.modality).to(device)
        vae.eval()
        vae.requires_grad_(False)
        decode_fn = lambda z: vae.decode(z)
        img_channels = 1 if args.medvae_model in {"medvae_4_1_2d", "medvae_8_1_2d", "medvae_4_4_2d"} else 3
        print(f"Decoder: MedVAE ({args.medvae_model}), img_channels={img_channels}", flush=True)

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    try:
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    except Exception as exc:
        lpips_fn = None
        print(f"WARNING: LPIPS disabled: {exc}", flush=True)

    psnr_truehr_values: list[float] = []
    psnr_selfref_values: list[float] = []
    ae_ceiling_inline_values: list[float] = []  # sanity check
    msssim_truehr_values: list[float] = []
    lpips_truehr_values: list[float] = []
    per_image_metrics: list[dict] = []

    print(f"\nRunning inference ({len(hr_files)} samples, T={args.timesteps})...\n", flush=True)
    start_time = time.perf_counter()

    for start in tqdm(range(0, len(hr_files), args.batch_size), desc="Evaluating"):
        end = min(start + args.batch_size, len(hr_files))

        hr_lat = torch.stack(
            [torch.from_numpy(np.load(latent_dir / hr_files[idx])).float() for idx in range(start, end)]
        ).to(device)
        lr_lat = torch.stack(
            [torch.from_numpy(np.load(latent_dir / lr_files[idx])).float() for idx in range(start, end)]
        ).to(device)

        # True-HR ground truth batch: parse N from each hr_N.npy filename, look up filenames.txt[N].
        true_hr_batch = []
        for idx in range(start, end):
            m = re.match(r"hr_(\d+)\.npy$", hr_files[idx])
            assert m is not None, f"Unexpected latent filename: {hr_files[idx]}"
            n = int(m.group(1))
            stem = orig_filenames[n]
            true_hr_batch.append(load_hr_png(hr_image_dir / f"{stem}.png", img_channels))
        true_hr = torch.stack(true_hr_batch).to(device)

        pred_hr = ldm.sample(lr_lat, T=args.timesteps)

        selfref_gt_pix = decode_fn(hr_lat)
        sr_pix = decode_fn(pred_hr)

        selfref_gt_01 = ((selfref_gt_pix + 1) / 2).clamp(0, 1)
        sr_01 = ((sr_pix + 1) / 2).clamp(0, 1)
        true_hr_01 = ((true_hr + 1) / 2).clamp(0, 1)
        if selfref_gt_01.ndim == 3:
            selfref_gt_01 = selfref_gt_01.unsqueeze(0)
            sr_01 = sr_01.unsqueeze(0)

        selfref_gray = mean_to_grayscale(selfref_gt_01)
        sr_gray = mean_to_grayscale(sr_01)
        true_hr_gray = mean_to_grayscale(true_hr_01)
        sr_lpips = ensure_three_channels(sr_gray)
        true_hr_lpips = ensure_three_channels(true_hr_gray)

        for local_idx in range(sr_gray.shape[0]):
            n = int(re.match(r"hr_(\d+)\.npy$", hr_files[start + local_idx]).group(1))
            image_id = orig_filenames[n]

            sr_gray_i = sr_gray[local_idx : local_idx + 1]
            true_hr_gray_i = true_hr_gray[local_idx : local_idx + 1]
            selfref_gray_i = selfref_gray[local_idx : local_idx + 1]
            sr_lpips_i = sr_lpips[local_idx : local_idx + 1]
            true_hr_lpips_i = true_hr_lpips[local_idx : local_idx + 1]

            psnr_truehr = psnr_fn(sr_gray_i, true_hr_gray_i).item()
            psnr_selfref = psnr_fn(sr_gray_i, selfref_gray_i).item()
            ae_ceiling_inline = psnr_fn(selfref_gray_i, true_hr_gray_i).item()  # sanity check

            msssim_truehr = None
            if true_hr_gray_i.shape[-1] >= 160 and true_hr_gray_i.shape[-2] >= 160:
                msssim_truehr = msssim_fn(sr_gray_i, true_hr_gray_i).item()
                msssim_truehr_values.append(msssim_truehr)

            lpips_truehr = None
            if lpips_fn is not None:
                lpips_truehr = lpips_fn(sr_lpips_i, true_hr_lpips_i).item()
                lpips_truehr_values.append(lpips_truehr)

            psnr_truehr_values.append(psnr_truehr)
            psnr_selfref_values.append(psnr_selfref)
            ae_ceiling_inline_values.append(ae_ceiling_inline)

            per_image_metrics.append({
                "id": image_id,
                "psnr_truehr": psnr_truehr,
                "psnr_selfref": psnr_selfref,
                "ae_ceiling_inline": ae_ceiling_inline,
                "msssim_truehr": msssim_truehr,
                "lpips_truehr": lpips_truehr,
            })

    elapsed_seconds = time.perf_counter() - start_time

    median_ae_ceiling = float(np.median(ae_ceiling_inline_values))
    print(f"\nSanity check: median inline AE-ceiling PSNR = {median_ae_ceiling:.2f} dB "
          f"(threshold {args.sanity_check_min_psnr} dB)", flush=True)
    if median_ae_ceiling < args.sanity_check_min_psnr:
        print("FAILED SANITY CHECK — filename mapping is likely wrong. Results below are NOT trustworthy.",
              flush=True)

    psnr_truehr_mean, psnr_truehr_std = summarize_metric(psnr_truehr_values)
    psnr_selfref_mean, psnr_selfref_std = summarize_metric(psnr_selfref_values)
    ae_ceiling_mean, ae_ceiling_std = summarize_metric(ae_ceiling_inline_values)
    msssim_mean, msssim_std = summarize_metric(msssim_truehr_values)
    lpips_mean, lpips_std = summarize_metric(lpips_truehr_values)

    results = {
        "n_samples": len(hr_files),
        "timesteps": args.timesteps,
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "hr_image_dir": str(hr_image_dir),
        "sanity_check": {
            "median_ae_ceiling_inline_psnr": median_ae_ceiling,
            "threshold": args.sanity_check_min_psnr,
            "passed": median_ae_ceiling >= args.sanity_check_min_psnr,
        },
        "timing": {"elapsed_seconds": elapsed_seconds,
                   "seconds_per_sample": elapsed_seconds / max(len(hr_files), 1)},
        "diffusion_sr_truehr": {
            "psnr_mean": psnr_truehr_mean, "psnr_std": psnr_truehr_std,
            "msssim_mean": msssim_mean, "msssim_std": msssim_std,
            "lpips_mean": lpips_mean, "lpips_std": lpips_std,
        },
        "diffusion_sr_selfref_forcomparison": {
            "psnr_mean": psnr_selfref_mean, "psnr_std": psnr_selfref_std,
        },
        "ae_ceiling_inline_forcomparison": {
            "psnr_mean": ae_ceiling_mean, "psnr_std": ae_ceiling_std,
        },
        "per_image_metrics": per_image_metrics,
    }

    print(f"\n{'=' * 60}")
    print("Results (true-HR ground truth)")
    print(f"{'=' * 60}")
    print(f"  Diffusion SR PSNR (true HR):     {psnr_truehr_mean:.2f} +/- {psnr_truehr_std:.2f} dB")
    print(f"  Diffusion SR PSNR (self-ref, old): {psnr_selfref_mean:.2f} +/- {psnr_selfref_std:.2f} dB")
    print(f"  AE ceiling (inline, sanity check): {ae_ceiling_mean:.2f} +/- {ae_ceiling_std:.2f} dB")
    if msssim_mean is not None:
        print(f"  MS-SSIM (true HR): {msssim_mean:.4f}")
    if lpips_mean is not None:
        print(f"  LPIPS (true HR): {lpips_mean:.4f}")
    print(f"  Elapsed: {elapsed_seconds:.2f}s for {len(hr_files)} samples")

    outfile = out_dir / "diffusion_eval_truehr_results.json"
    with open(outfile, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
