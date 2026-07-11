"""
Shared KL-f4 (CompVis natural-image VAE, 3x64x64) backend for the revision SR pipeline.

Single source of the encode/decode convention so the extract, SR-eval and AE-ceiling
scripts cannot drift. KL-f4 is the natural-side matched-GEOMETRY control (vs medvae_4_3,
both 3x64x64) for index.md §1c/C1: it isolates DOMAIN from latent GEOMETRY at the large
latent. The AE-ceiling numbers came from eval_klf4_ae.py; this module extends the same
model to the full extract -> train -> SR path (the SR domain gap at 3x64x64).

Conventions (identical to eval_klf4_ae.py, and to how medvae_4_3 latents are treated):
  - encode(x): x is [B,3,256,256] in [-1,1]; returns the DETERMINISTIC posterior mean
    [B,3,64,64] (no sampling), matching medvae's deterministic S1 latents. Latents are
    stored RAW (no rescaling) so KL-f4 gets the exact same pipeline treatment as SD-VAE.
  - decode(z): z is [B,3,64,64] raw latent; returns [B,3,256,256] in [-1,1] (CompVis
    convention), which the SR-eval maps to [0,1] via (x+1)/2 then averages to grayscale.
"""
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent-diffusion"))
from ldm.modules.diffusionmodules.model import Encoder, Decoder  # noqa: E402

_SHARED_ROOT = os.environ.get("LATENT_SR_SHARED_ROOT", "/orcd/pool/006/lceli_shared")
CKPT = f"{_SHARED_ROOT}/latent-sr-release/pretrained_vaes/kl-f4/model.ckpt"

# Standard CompVis kl-f4 first-stage config (f=4 -> 256->64, z=3).
DDCONFIG = dict(
    double_z=True, z_channels=3, resolution=256, in_channels=3, out_ch=3,
    ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0,
)


class KLF4(nn.Module):
    """Minimal AutoencoderKL matching the CompVis kl-f4 first-stage checkpoint."""

    def __init__(self, ddconfig=DDCONFIG, embed_dim=3):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        moments = self.quant_conv(self.encoder(x))
        mean, _ = torch.chunk(moments, 2, dim=1)  # deterministic posterior mean
        return mean

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))


def load_klf4(device, ckpt=None):
    # DOMAIN SWITCH (revision R1): default = natural CompVis checkpoint. If the caller
    # passes ckpt, or exports env KLF4_CKPT, load the medical-fine-tuned kl-f4 instead
    # (same architecture/keys, produced by finetune_klf4.py). This lets the ENTIRE SR
    # path (02_extract / eval_diffusion_sr) run on medical-KL-f4 with zero code edits —
    # the natural-vs-medical contrast then changes only the training domain.
    import os
    ckpt = ckpt or os.environ.get("KLF4_CKPT") or CKPT
    m = KLF4()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k: v for k, v in sd.items() if not k.startswith("loss")}  # drop discriminator
    missing, unexpected = m.load_state_dict(sd, strict=False)
    tag = "medical" if ckpt != CKPT else "natural"
    print(f"  loaded kl-f4 [{tag}] {ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return m.to(device).eval()
