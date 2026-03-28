#!/usr/bin/env python3
"""
Extract MedVAE embeddings from HR/LR image pairs and save them as .npy files.
This replaces the Stable Diffusion VAE with MedVAE for medical imaging.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class PairedImageDataset(Dataset):
    """Loads matching HR-LR image pairs from hr/ and lr/ folders.

    HR and LR use separate transforms so LR can be encoded at native
    resolution (no upscaling), matching the LDM SR approach.
    """

    def __init__(self, hr_dir: str, lr_dir: str, hr_transform=None, lr_transform=None):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

        self.hr_images = sorted([p for p in self.hr_dir.iterdir() if p.is_file()])
        self.lr_images = sorted([p for p in self.lr_dir.iterdir() if p.is_file()])

        if len(self.hr_images) != len(self.lr_images):
            raise ValueError(f"HR and LR image counts must match: {len(self.hr_images)} vs {len(self.lr_images)}")

        if len(self.hr_images) == 0:
            raise ValueError(f"No images found in {self.hr_dir}")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # Use "L" (grayscale) for 1-channel models (e.g. medvae_4_1_2d),
        # "RGB" for 3+ channel models (e.g. medvae_4_3_2d, sd-vae)
        mode = getattr(self, '_img_mode', 'RGB')
        hr_img = Image.open(self.hr_images[idx]).convert(mode)
        lr_img = Image.open(self.lr_images[idx]).convert(mode)

        if self.hr_transform:
            hr_img = self.hr_transform(hr_img)
        if self.lr_transform:
            lr_img = self.lr_transform(lr_img)

        return hr_img, lr_img, self.hr_images[idx].stem


def encode_batch(model, images: torch.Tensor, backend: str) -> torch.Tensor:
    """
    Encode a batch of images to latent tensors with a normalized interface.
    Returns tensor of shape [B, C, H, W].
    """
    if backend == "medvae":
        # Use model.encode() to get S1 (raw) latents, NOT model() which returns
        # S2 (projected) latents via channel_proj(channel_ds(z) + z).
        # The decoder expects S1 latents — S2 latents cause pixel value shift.
        latents = model.encode(images)
    elif backend == "autoencoderkl":
        encoded = model.encode(images)
        latents = encoded.latent_dist.sample()
    else:
        raise ValueError(f"Unknown encoder backend: {backend}")

    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Encoder returned unsupported latent type: {type(latents)}")
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4:
        raise ValueError(f"Expected 4D latents [B,C,H,W], got shape {tuple(latents.shape)}")
    return latents


def encode_split(
    split: str,
    split_dir: Path,
    latent_split_dir: Path,
    model,
    encoder_backend: str,
    hr_transform,
    lr_transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    img_mode: str = "RGB",
):
    """Encode one split (train/val) and save .npy latents."""

    print(f"\n{'='*60}")
    print(f"Encoding {split} split")
    print(f"{'='*60}")

    dataset = PairedImageDataset(
        hr_dir=split_dir / "hr",
        lr_dir=split_dir / "lr",
        hr_transform=hr_transform,
        lr_transform=lr_transform,
    )
    # Set image loading mode (L for 1-channel models, RGB for 3-channel)
    dataset._img_mode = img_mode

    print(f"Found {len(dataset)} image pairs")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    latent_split_dir.mkdir(parents=True, exist_ok=True)
    img_index = 0
    all_filenames = []

    model.eval()
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for hr_img, lr_img, names in tqdm(loader, desc=f"Encoding {split}", unit="batch"):
            hr_img = hr_img.to(device)
            lr_img = lr_img.to(device)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                hr_lat = encode_batch(model, hr_img, encoder_backend).cpu()
                lr_lat = encode_batch(model, lr_img, encoder_backend).cpu()

            # Save each latent
            for i, (h, l) in enumerate(zip(hr_lat, lr_lat)):
                np.save(latent_split_dir / f"hr_{img_index+i}.npy", h.numpy())
                np.save(latent_split_dir / f"lr_{img_index+i}.npy", l.numpy())
                all_filenames.append(names[i])

            img_index += len(hr_lat)

    # Save index→filename mapping for SR image naming in eval
    with open(latent_split_dir / "filenames.txt", "w") as f:
        for name in all_filenames:
            f.write(name + "\n")

    print(f"✅ Encoded {img_index} pairs to {latent_split_dir}")
    return img_index


def main():
    parser = argparse.ArgumentParser(description="Extract MedVAE embeddings from HR/LR images")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/processed",
        help="Folder containing split sub-folders (train, val)",
    )
    parser.add_argument(
        "--latent-root",
        type=str,
        default="../embeddings/medvae_latents",
        help="Root folder where latent sub-folders will be created",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which dataset splits to encode",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--model-name",
        type=str,
        default="medvae_4_3_2d",
        help="MedVAE model name",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="xray",
        help="Medical imaging modality (xray, ct, mri, etc.)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="medvae",
        choices=["medvae", "sd-vae"],
        help="Encoder backend: 'medvae' (default) or 'sd-vae' (Stable Diffusion VAE for comparison)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"Embedding Extraction Pipeline")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_name}")
    print(f"Modality: {args.modality}")
    print(f"Data root: {args.data_root}")
    print(f"Output: {args.latent_root}")

    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load encoder model
    if args.backend == "sd-vae":
        from diffusers.models import AutoencoderKL
        print(f"\nLoading Stable Diffusion VAE (sd-vae-ft-ema)...")
        model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        model.eval()
        encoder_backend = "autoencoderkl"
        print("Encoder backend: autoencoderkl (SD-VAE)")
    else:
        print(f"\nLoading MedVAE model: {args.model_name}...")
        try:
            from medvae import MVAE
            model = MVAE(model_name=args.model_name, modality=args.modality).to(device)
            model.requires_grad_(False)
            model.eval()
            encoder_backend = "medvae"
            print("Encoder backend: medvae")
        except Exception as e:
            print(f"Error loading MedVAE: {e}")
            print("\nFalling back to SD-VAE...")
            from diffusers.models import AutoencoderKL
            model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
            model.eval()
            encoder_backend = "autoencoderkl"
            print("Encoder backend: autoencoderkl (fallback)")

    # Separate transforms for HR and LR
    # Determine number of channels from model name
    # medvae_4_1_2d → 1 channel (grayscale), medvae_4_3_2d → 3 channels (RGB), sd-vae → 3 channels
    if args.backend == "medvae" and "_1_" in args.model_name:
        n_channels = 1
        img_mode = "L"
    else:
        n_channels = 3
        img_mode = "RGB"

    print(f"Image mode: {img_mode} ({n_channels} channels)")

    # HR: resize to target size (e.g. 256x256)
    # LR: encode at native resolution (no resize) for proper SR conditioning
    normalize = transforms.Normalize([0.5] * n_channels, [0.5] * n_channels)

    hr_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])
    lr_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    print(f"HR transform: Resize({args.image_size}) + CenterCrop + ToTensor + Normalize")
    print(f"LR transform: ToTensor + Normalize (native resolution, no resize)")

    data_root = Path(args.data_root)
    latent_root = Path(args.latent_root)

    total_encoded = 0

    # Encode each split
    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"⚠️  Split '{split}' not found at {split_dir}, skipping...")
            continue

        num_encoded = encode_split(
            split=split,
            split_dir=split_dir,
            latent_split_dir=latent_root / f"{split}_latent",
            model=model,
            encoder_backend=encoder_backend,
            hr_transform=hr_transform,
            lr_transform=lr_transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            img_mode=img_mode,
        )
        total_encoded += num_encoded

    print(f"\n{'='*60}")
    print(f"✅ Embedding extraction complete!")
    print(f"{'='*60}")
    print(f"Total encoded: {total_encoded} image pairs")
    print(f"Saved to: {latent_root.absolute()}")


if __name__ == "__main__":
    main()
