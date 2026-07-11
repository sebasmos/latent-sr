#!/usr/bin/env python3
"""
Replicate MedVAE Paper Table 4 — Knee MRI reconstruction quality.

Matches the author's exact evaluation protocol (from GitHub issue #33 response):
  1. Load 3D MRNet validation volumes (120 total)
  2. Apply: ScaleIntensity [0,1] → Normalize [-1,1] → Pad to 160³ → CenterCrop 160³
  3. For 2D models: encode/decode each 2D slice independently, compute per-volume PSNR
  4. Sample 100 random volumes (fixed seed)
  5. Report mean PSNR, MSE, MS-SSIM

Reference targets:
  medvae_4_1_2d (f=16, C=1): PSNR=27.38, MSE=0.0079, MS-SSIM=0.990
  medvae_4_3_2d (f=16, C=3): PSNR=31.52, MSE=0.0033, MS-SSIM=0.997
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from tqdm import tqdm


def load_mri_3d_for_2d_eval(path: str, roi_size=(160, 160, 160)):
    """
    Load MRNet .npy volume and apply the MedVAE author's preprocessing.

    Pipeline (from MedVAE issue #33):
      LoadImage → EnsureChannelFirst → Orientation(RAS) →
      ScaleIntensity(channel_wise, 0→1) → Normalize(0.5, 0.5) →
      Pad(160³, value=-1) → CenterSpatialCrop(160³)

    MRNet volumes are already (S, 256, 256) uint8 sagittal, so
    Orientation(RAS) is a no-op for our purposes. We replicate
    the numeric transforms exactly.
    """
    vol = np.load(path).astype(np.float32)  # (S, H, W) uint8 → float32

    # EnsureChannelFirst: (S, H, W) → (1, S, H, W) for 3D
    vol = torch.from_numpy(vol).unsqueeze(0)  # (1, S, H, W)

    # ScaleIntensity channel_wise [0, 1]
    # For single-channel: (val - min) / (max - min)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = torch.zeros_like(vol)

    # MonaiNormalize: (x - 0.5) / 0.5 → [-1, 1]
    vol = (vol - 0.5) / 0.5

    # MonaiPad to roi_size with value=-1
    d, h, w = vol.shape[1], vol.shape[2], vol.shape[3]
    pad_d = max(roi_size[0] - d, 0)
    pad_h = max(roi_size[1] - h, 0)
    pad_w = max(roi_size[2] - w, 0)
    # F.pad order: (w_left, w_right, h_top, h_bot, d_front, d_back)
    vol = F.pad(
        vol,
        (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
        ),
        mode="constant",
        value=-1,
    )

    # CenterSpatialCrop to roi_size
    _, cd, ch, cw = vol.shape
    sd = (cd - roi_size[0]) // 2
    sh = (ch - roi_size[1]) // 2
    sw = (cw - roi_size[2]) // 2
    vol = vol[
        :,
        sd : sd + roi_size[0],
        sh : sh + roi_size[1],
        sw : sw + roi_size[2],
    ]

    return vol  # (1, 160, 160, 160) in [-1, 1]


def evaluate_volume_2d(model, vol, device, model_name):
    """
    Evaluate a single 3D volume slice-by-slice through the 2D VAE.

    Args:
        vol: (1, D, H, W) tensor in [-1, 1]
    Returns:
        per-volume PSNR, MSE, MS-SSIM
    """
    merge_channels = "1_2d" in model_name  # C=1 models need channel merge
    n_slices = vol.shape[1]

    all_gt = []
    all_recon = []

    for s in range(n_slices):
        slice_2d = vol[:, s, :, :]  # (1, H, W)

        # Include ALL slices (including padded) to match author's protocol.
        # Padded slices (value=-1) reconstruct trivially, which is how the
        # paper computes per-volume MSE → PSNR.

        # Expand to 3 channels for C=3 models, keep 1 for C=1
        if merge_channels:
            inp = slice_2d.unsqueeze(0).to(device)  # (1, 1, H, W)
        else:
            inp = slice_2d.expand(3, -1, -1).unsqueeze(0).to(device)  # (1, 3, H, W)

        with torch.no_grad():
            # Encode → decode
            latent = model.encode(inp)
            recon = model.decode(latent)

        all_gt.append(inp.cpu())
        all_recon.append(recon.cpu())

    if not all_gt:
        return None, None, None

    gt = torch.cat(all_gt, dim=0)  # (N, C, H, W)
    recon = torch.cat(all_recon, dim=0)

    # Normalize to [0, 1] for metrics
    gt_01 = (gt + 1) / 2
    recon_01 = (recon + 1) / 2

    # Per-volume MSE (average over all slices)
    mse = F.mse_loss(recon_01, gt_01).item()

    # PSNR from MSE
    psnr = -10 * np.log10(mse + 1e-10)

    # MS-SSIM (needs spatial >= 160)
    msssim = None
    if gt_01.shape[-1] >= 160 and gt_01.shape[-2] >= 160:
        msssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        try:
            msssim = msssim_fn(recon_01, gt_01).item()
        except Exception:
            pass

    return psnr, mse, msssim


def main():
    parser = argparse.ArgumentParser(description="Replicate MedVAE Table 4 (Knee MRI)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repro_paths import data_root
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(data_root() / "mrnetkneemris/MRNet-v1.0"),
    )
    parser.add_argument("--model-name", type=str, default="medvae_4_3_2d")
    parser.add_argument("--modality", type=str, default="mri")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/table4_replication")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)

    # Load MedVAE model
    from medvae import MVAE

    model = MVAE(args.model_name, args.modality).to(device)
    model.eval()
    print(f"Model: {args.model_name}, modality: {args.modality}")
    print(f"Device: {device}")

    # Collect validation volumes
    val_dir = Path(args.data_root) / "valid" / "sagittal"
    vol_files = sorted(val_dir.glob("*.npy"))
    print(f"Validation volumes: {len(vol_files)}")

    # Sample N volumes (author used 100 from validation set)
    if args.n_samples < len(vol_files):
        vol_files = random.sample(vol_files, args.n_samples)
        vol_files.sort()
    print(f"Evaluating {len(vol_files)} volumes (seed={args.seed})")

    # Evaluate
    psnrs, mses, msssims = [], [], []

    for vf in tqdm(vol_files, desc="Evaluating"):
        vol = load_mri_3d_for_2d_eval(str(vf))
        psnr, mse, msssim = evaluate_volume_2d(model, vol, device, args.model_name)
        if psnr is not None:
            psnrs.append(psnr)
            mses.append(mse)
            if msssim is not None:
                msssims.append(msssim)

    results = {
        "model": args.model_name,
        "n_samples": len(psnrs),
        "seed": args.seed,
        "PSNR": float(np.mean(psnrs)),
        "PSNR_std": float(np.std(psnrs)),
        "MSE": float(np.mean(mses)),
        "MS-SSIM": float(np.mean(msssims)) if msssims else None,
        "protocol": "3D volume, 160^3 center crop, 2D slice-by-slice encode/decode",
    }

    print(f"\n{'='*60}")
    print(f"Table 4 Replication: {args.model_name}")
    print(f"{'='*60}")
    print(f"  PSNR:    {results['PSNR']:.2f} ± {results['PSNR_std']:.2f} dB")
    print(f"  MSE:     {results['MSE']:.6f}")
    print(f"  MS-SSIM: {results['MS-SSIM']:.4f}" if results["MS-SSIM"] else "  MS-SSIM: N/A")
    print(f"  Samples: {results['n_samples']}")

    # Paper targets
    targets = {
        "medvae_4_1_2d": {"PSNR": 27.38, "MSE": 0.0079, "MS-SSIM": 0.990},
        "medvae_4_3_2d": {"PSNR": 31.52, "MSE": 0.0033, "MS-SSIM": 0.997},
    }
    if args.model_name in targets:
        t = targets[args.model_name]
        print(f"\n  Paper:   PSNR={t['PSNR']}, MSE={t['MSE']}, MS-SSIM={t['MS-SSIM']}")
        print(f"  Delta:   {results['PSNR'] - t['PSNR']:+.2f} dB")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"table4_{args.model_name}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
