#!/usr/bin/env python3
"""
KL-f4 (CompVis natural-image VAE, 3x64x64) AE reconstruction ceiling on the 3 datasets.
This is the natural-side matched-GEOMETRY control (vs medvae_4_3 at 3x64x64) requested in
index.md §1c/C1 — it separates DOMAIN from latent GEOMETRY at the large latent.

AE ceiling = encode (posterior mean) -> decode, PSNR vs HR. Pure forward passes, no training.
Loads the CompVis kl-f4 checkpoint using ldm.modules Encoder/Decoder + a minimal AutoencoderKL.
"""
import os, glob, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent-diffusion"))
from ldm.modules.diffusionmodules.model import Encoder, Decoder

_SHARED_ROOT = os.environ.get("LATENT_SR_SHARED_ROOT", "/orcd/pool/006/lceli_shared")
CKPT = f"{_SHARED_ROOT}/latent-sr-release/pretrained_vaes/kl-f4/model.ckpt"
DB = os.environ.get("LATENT_SR_DATA_ROOT", f"{_SHARED_ROOT}/DATASET")
HR_DIRS = {
    "MRNet": f"{DB}/mrnetkneemris/MRNet-v1.0-middle/valid/hr",
    "BraTS": f"{DB}/brats2023-sr/valid/hr",
    "CXR":   f"{DB}/mimic-cxr-sr/valid/hr",
}
# standard CompVis kl-f4 first-stage config (f=4 -> 256->64, z=3)
DD = dict(double_z=True, z_channels=3, resolution=256, in_channels=3, out_ch=3,
          ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)


class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim=3):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    @torch.no_grad()
    def reconstruct(self, x):
        moments = self.quant_conv(self.encoder(x))
        mean, _ = torch.chunk(moments, 2, dim=1)          # posterior mean (deterministic ceiling)
        return self.decoder(self.post_quant_conv(mean))


def load_model(device):
    m = AutoencoderKL(DD, embed_dim=3)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k: v for k, v in sd.items() if not k.startswith("loss")}   # drop discriminator
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  loaded kl-f4: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return m.to(device).eval()


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def eval_dir(model, hr_dir, device, limit=None):
    files = sorted(glob.glob(os.path.join(hr_dir, "*.png")) + glob.glob(os.path.join(hr_dir, "*.jpg")))
    if limit:
        files = files[:limit]
    ps = []
    for f in files:
        img = Image.open(f).convert("L").resize((256, 256))       # grayscale HR
        arr = np.asarray(img, np.float32) / 255.0                  # [0,1]
        x = torch.from_numpy(arr)[None, None].repeat(1, 3, 1, 1)   # RGB replicate
        x = (x * 2 - 1).to(device)                                 # [-1,1] CompVis convention
        rec = model.reconstruct(x).clamp(-1, 1)
        rec = ((rec + 1) / 2).mean(1, keepdim=True)[0, 0].cpu().numpy()  # back to [0,1] grayscale
        ps.append(psnr(arr, rec))
    return np.mean(ps), np.std(ps), len(ps)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  limit={limit}", flush=True)
    model = load_model(device)
    print("\n=== KL-f4 (natural, 3x64x64) AE reconstruction ceiling ===", flush=True)
    print(f"{'dataset':7} {'kl-f4 AE':>9} {'medvae_4_3':>11} {'SD-VAE':>8} {'domain@64²':>11}", flush=True)
    ref43 = {"MRNet": 27.85, "BraTS": 37.87, "CXR": 36.93}   # medical 3x64x64
    refsd = {"MRNet": 23.92, "BraTS": 31.39, "CXR": 32.02}   # natural 4x32x32
    for name, d in HR_DIRS.items():
        if not os.path.isdir(d):
            print(f"  {name}: HR dir missing {d}", flush=True); continue
        mean, std, n = eval_dir(model, d, device, limit)
        dom = ref43[name] - mean   # medvae_4_3 - kl-f4 : DOMAIN effect at matched 3x64x64
        print(f"{name:7} {mean:7.2f}±{std:.2f} {ref43[name]:11.2f} {refsd[name]:8.2f} {dom:+11.2f}  (n={n})", flush=True)
    print("\nInterpretation: domain@64² = medvae_4_3 - kl-f4 at IDENTICAL 3x64x64 geometry.", flush=True)
    print("  large positive -> geometry gain is DOMAIN-ENABLED (title rescued on SR tasks).", flush=True)
