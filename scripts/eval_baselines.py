#!/usr/bin/env python3
"""
Evaluate non-diffusion SR baselines against HR images.

Supported methods:
  - bicubic: deterministic CPU-safe resize baseline
  - realesrgan: optional pretrained Real-ESRGAN inference

The output JSON intentionally mirrors `scripts/eval_diffusion_sr.py` so
`scripts/aggregate_results.py` can consume it without a separate schema.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SR baselines")
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument("--lr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="bicubic",
        choices=["bicubic", "realesrgan"],
        help="Baseline method to evaluate.",
    )
    parser.add_argument(
        "--realesrgan-model",
        type=str,
        default="RealESRGAN_x4plus",
        help="Real-ESRGAN model name when --method realesrgan is selected.",
    )
    parser.add_argument(
        "--realesrgan-model-path",
        type=str,
        default=None,
        help="Optional explicit .pth path for Real-ESRGAN weights.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile", type=int, default=0, help="Tile size for Real-ESRGAN inference.")
    parser.add_argument("--fp32", action="store_true", help="Disable half precision for Real-ESRGAN.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke runs.")
    parser.add_argument("--save-images", action="store_true", help="Save generated SR PNGs.")
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


def bicubic_upscale(lr_path: Path, target_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(lr_path).convert("L")
    upsampled = image.resize(target_size, resample=Image.Resampling.BICUBIC)
    array = np.asarray(upsampled, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class RealESRGANRunner:
    def __init__(self, model_name: str, model_path: str | None, tile: int, fp32: bool, device: str) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        realesrgan_root = repo_root / "Real-ESRGAN"
        sys.path.insert(0, str(realesrgan_root))

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        self.netscale, model, model_path_resolved = self._resolve_model(
            model_name,
            model_path,
            realesrgan_root,
            RRDBNet,
            SRVGGNetCompact,
            load_file_from_url,
        )
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=model_path_resolved,
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=(device.startswith("cuda") and not fp32),
            gpu_id=0 if device.startswith("cuda") else None,
        )

    @staticmethod
    def _resolve_model(model_name, model_path, realesrgan_root, RRDBNet, SRVGGNetCompact, load_file_from_url):
        file_urls = None
        if model_name == "RealESRGAN_x4plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_urls = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            ]
        elif model_name == "RealESRNet_x4plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_urls = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            ]
        elif model_name == "RealESRGAN_x4plus_anime_6B":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_urls = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            ]
        elif model_name == "RealESRGAN_x2plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_urls = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            ]
        elif model_name == "realesr-general-x4v3":
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            netscale = 4
            file_urls = [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
            ]
        else:
            raise ValueError(f"Unsupported Real-ESRGAN model: {model_name}")

        if model_path is None:
            weights_dir = realesrgan_root / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            resolved = None
            for url in file_urls:
                resolved = load_file_from_url(url=url, model_dir=str(weights_dir), progress=True, file_name=None)
            assert resolved is not None
            model_path = resolved
        return netscale, model, model_path

    def upscale(self, lr_path: Path) -> torch.Tensor:
        import cv2

        image = cv2.imread(str(lr_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read {lr_path}")
        image_rgb = np.repeat(image[:, :, None], 3, axis=2)
        output, _ = self.upsampler.enhance(image_rgb, outscale=self.netscale)
        output_gray = output.mean(axis=2).astype(np.float32) / 255.0
        return torch.from_numpy(output_gray).unsqueeze(0)


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
        print(f"WARNING: LPIPS disabled because initialization failed: {exc}")

    runner = None
    if args.method == "realesrgan":
        runner = RealESRGANRunner(
            model_name=args.realesrgan_model,
            model_path=args.realesrgan_model_path,
            tile=args.tile,
            fp32=args.fp32,
            device=args.device,
        )

    per_image_metrics: list[dict[str, float | str | None]] = []
    psnr_values: list[float] = []
    msssim_values: list[float] = []
    lpips_values: list[float] = []

    start_time = time.perf_counter()
    for hr_path, lr_path, image_id in tqdm(pairs, desc=f"Evaluating {args.method}"):
        gt = load_image_tensor(hr_path)
        if args.method == "bicubic":
            pred = bicubic_upscale(lr_path, target_size=(gt.shape[-1], gt.shape[-2]))
        else:
            assert runner is not None
            pred = runner.upscale(lr_path)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred_image = Image.fromarray((pred.squeeze().numpy() * 255.0).astype(np.uint8), mode="L")
                pred = torch.from_numpy(
                    np.asarray(
                        pred_image.resize((gt.shape[-1], gt.shape[-2]), resample=Image.Resampling.BICUBIC),
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
            lpips_value = lpips_fn(ensure_three_channels(pred_b), ensure_three_channels(gt_b)).item()

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

    results: dict[str, object] = {
        "n_samples": len(pairs),
        "timesteps": None,
        "backend": args.method,
        "method": args.method,
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
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
