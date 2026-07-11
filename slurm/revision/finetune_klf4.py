#!/usr/bin/env python3
"""
REVISION R1 — medical-KL-f4: fine-tune the EXACT CompVis KL-f4 first-stage
autoencoder (3x64x64, f=4) on MEDICAL data, so the natural-KL-f4 vs medical-KL-f4
contrast changes ONLY the training DOMAIN. Architecture, latent geometry, checkpoint
initialisation and the deterministic-mean encode convention are held byte-identical to
klf4_vae.py / eval_klf4_ae.py (the natural-side control of index.md §1c/C1).

WHY A SELF-CONTAINED LOOP (see the R1 task report):
  This latent-diffusion checkout has NO autoencoder LightningModule (ldm/models/ is
  empty), NO first-stage config, and its CompVis loss (contperceptual.py) needs the
  `taming` package, which is not installed and whose src/ submodule is unchecked-out.
  pytorch_lightning is also not importable in the `medvae-sr` env (torchvision::nms).
  So there is no existing trainer/config to fine-tune through. This loop instead reuses
  the PROVEN, already-imported pieces (ldm Encoder/Decoder + the exact kl-f4 checkpoint,
  as in klf4_vae.py) with the standard AutoencoderKL reconstruction objective. No new
  deps, no pl, no taming, fp32 throughout — the lowest-risk faithful path.

RECIPE (held fixed vs the natural checkpoint's own training, up to what deps allow):
  loss = L1(recon, x) + kl_weight * KL,  kl_weight = 1e-6  (CompVis autoencoder_kl_64x64x3
  value), Adam(betas=(0.5,0.9)), fp32. Training samples z ~ posterior (reparameterised),
  exactly like CompVis AutoencoderKL.training_step. The metric (AE PSNR ceiling and the
  downstream SR latents) uses the DETERMINISTIC posterior MEAN — identical to
  eval_klf4_ae.py. CAVEAT (reported, not hidden): the released natural checkpoint was
  trained with additional LPIPS + PatchGAN terms that are unavailable here; those terms
  sharpen perceptual texture and are NOT part of a PSNR ceiling, so omitting them for the
  medical fine-tune keeps the comparison honest for the reference metrics we report.

Init from the natural CompVis kl-f4 checkpoint (fine-tune, not from scratch) so the ONLY
thing that changes relative to the natural control is the medical training signal.

Modes:
  train (default) : fine-tune on <data-root>/train/hr, checkpoint to <out-dir>/checkpoints/
                    last.ckpt (RESUMABLE), then run the valid AE ceiling and write results.
  --eval-only     : skip training, just run the AE ceiling for --init-ckpt on <data-root>/
                    valid/hr and write results (used by the cheap ceiling-eval SLURM job).

The saved checkpoint stores {"state_dict": {encoder.*, decoder.*, quant_conv.*,
post_quant_conv.*}} with the SAME key names as the CompVis ckpt, so the whole downstream
SR path (02_extract / eval_diffusion_sr via klf4_vae.load_klf4) can load the medical
weights simply by exporting KLF4_CKPT=<this file>.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "latent-diffusion"))
from ldm.modules.diffusionmodules.model import Encoder, Decoder  # noqa: E402
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution  # noqa: E402

_SHARED_ROOT = os.environ.get("LATENT_SR_SHARED_ROOT", "/orcd/pool/006/lceli_shared")
NATURAL_CKPT = f"{_SHARED_ROOT}/latent-sr-release/pretrained_vaes/kl-f4/model.ckpt"

# Standard CompVis kl-f4 first-stage config (f=4 -> 256->64, z=3) — identical to klf4_vae.py.
DDCONFIG = dict(
    double_z=True, z_channels=3, resolution=256, in_channels=3, out_ch=3,
    ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0,
)


class AutoencoderKL(nn.Module):
    """CompVis kl-f4 first-stage, trainable. Same module graph as klf4_vae.KLF4."""

    def __init__(self, ddconfig=DDCONFIG, embed_dim=3):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample_posterior=True):
        post = self.encode(x)
        z = post.sample() if sample_posterior else post.mode()
        return self.decode(z), post

    def last_layer(self):
        return self.decoder.conv_out.weight


def load_state(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k: v for k, v in sd.items() if not k.startswith("loss")}  # drop discriminator/LPIPS
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded {os.path.basename(ckpt_path)}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return ck


# ----------------------------- data -----------------------------
class HRDataset(Dataset):
    """Grayscale HR images -> [-1,1], resized 256, replicated to 3ch. EXACTLY matches
    eval_klf4_ae.py / 02_extract HR convention so the domain is the ONLY thing that moves."""

    def __init__(self, hr_dir, size=256, limit=None):
        files = sorted(glob.glob(os.path.join(hr_dir, "*.png")) + glob.glob(os.path.join(hr_dir, "*.jpg")))
        if not files:
            raise ValueError(f"No HR images in {hr_dir}")
        if limit:
            files = files[:limit]
        self.files = files
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("L").resize((self.size, self.size))
        arr = np.asarray(img, np.float32) / 255.0
        x = torch.from_numpy(arr)[None].repeat(3, 1, 1)  # [3,H,W]
        return x * 2 - 1  # [-1,1]


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


@torch.no_grad()
def eval_ceiling(model, hr_dir, device, limit=None):
    """AE reconstruction ceiling: encode -> posterior MEAN -> decode, PSNR vs HR grayscale.
    Identical maths to eval_klf4_ae.py."""
    model.eval()
    files = sorted(glob.glob(os.path.join(hr_dir, "*.png")) + glob.glob(os.path.join(hr_dir, "*.jpg")))
    if limit:
        files = files[:limit]
    ps = []
    for f in files:
        img = Image.open(f).convert("L").resize((256, 256))
        arr = np.asarray(img, np.float32) / 255.0
        x = torch.from_numpy(arr)[None, None].repeat(1, 3, 1, 1)
        x = (x * 2 - 1).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            z = model.encode(x.float()).mode()          # deterministic mean
            rec = model.decode(z).clamp(-1, 1)
        rec = ((rec + 1) / 2).mean(1, keepdim=True)[0, 0].cpu().numpy()
        ps.append(psnr(arr, rec))
    return float(np.mean(ps)), float(np.std(ps)), len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", required=True, help="dir with train/hr and valid/hr")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-ckpt", default=NATURAL_CKPT, help="checkpoint to init from (or eval)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    # NOTE (2026-07-09): default lowered 1e-6 -> 1e-8 after diagnosing posterior collapse.
    # CompVis's 1e-6 was tuned for a reconstruction loss containing LPIPS + PatchGAN
    # (magnitude ~O(1)). Our L1-only loss is ~50-100x smaller, so 1e-6 made the KL term
    # COMPARABLE TO OR LARGER THAN the reconstruction term (BraTS: kl_term/L1 = 1.13 at
    # epoch 1). The optimiser then minimised loss by collapsing q(z|x) onto the prior
    # N(0,1) -- measured on the resulting BraTS ckpt: mu std 10.99 -> 0.41, sigma 0.00 ->
    # 0.90 -- destroying the latent and dropping the AE ceiling 37.2 -> 16.6 dB.
    # 1e-8 puts the KL term at ~1% of L1 on the worst-case dataset: a regulariser, not a
    # competing objective.
    ap.add_argument("--kl-weight", type=float, default=1e-8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-images", type=int, default=None, help="smoke: cap train set")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke: cap optimiser steps")
    ap.add_argument("--max-val-images", type=int, default=None, help="smoke: cap ceiling eval")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")

    print(f"=== finetune_klf4 | dataset={args.dataset} device={device} eval_only={args.eval_only} ===", flush=True)
    model = AutoencoderKL().to(device)

    valid_hr = os.path.join(args.data_root, "valid", "hr")

    if args.eval_only:
        load_state(model, args.init_ckpt)
        mean, std, n = eval_ceiling(model, valid_hr, device, args.max_val_images)
        out = {"dataset": args.dataset, "mode": "eval_only", "init_ckpt": args.init_ckpt,
               "ae_psnr_mean": mean, "ae_psnr_std": std, "n": n}
        print(f"[ceiling] {args.dataset}: AE PSNR {mean:.2f}±{std:.2f} dB (n={n})", flush=True)
        with open(os.path.join(args.out_dir, "ceiling.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        return

    # ---- train ----
    train_hr = os.path.join(args.data_root, "train", "hr")
    ds = HRDataset(train_hr, size=256, limit=args.max_train_images)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, pin_memory=True)
    print(f"  train images: {len(ds)}  batches/epoch: {len(dl)}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))

    start_epoch = 0
    if os.path.exists(last_ckpt):
        ck = load_state(model, last_ckpt)
        if "optimizer" in ck:
            try:
                opt.load_state_dict(ck["optimizer"])
            except Exception as e:
                print(f"  optimizer state not restored ({e}); continuing", flush=True)
        start_epoch = int(ck.get("epoch", 0))
        print(f"  RESUME from {last_ckpt} at epoch {start_epoch}", flush=True)
    else:
        load_state(model, args.init_ckpt)
        print(f"  INIT from natural checkpoint {args.init_ckpt}", flush=True)

    def save(epoch, path=last_ckpt):
        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "epoch": epoch, "dataset": args.dataset}, path)

    # ---- guard against the posterior-collapse failure mode (see --kl-weight note) ----
    # Evaluate the INIT ceiling as a baseline, then re-evaluate every epoch and keep the
    # best checkpoint. Fine-tuning can then never leave us worse off than the natural init,
    # and a collapse shows up immediately in the per-epoch log instead of only at the end.
    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
    init_mean, _, init_n = eval_ceiling(model, valid_hr, device, args.max_val_images)
    print(f"[ceiling] INIT ({'resumed' if start_epoch else 'natural'}) {args.dataset}: "
          f"AE PSNR {init_mean:.2f} dB (n={init_n})  <- baseline to beat", flush=True)
    best_psnr = init_mean
    save(start_epoch, best_ckpt)  # so best.ckpt is always valid, even if no epoch improves

    step = 0
    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run_rec = run_kl = 0.0
        for x in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=False):
                rec, post = model(x.float(), sample_posterior=True)
                rec_loss = F.l1_loss(rec, x)
                kl_loss = post.kl().mean()
                loss = rec_loss + args.kl_weight * kl_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_rec += float(rec_loss.detach())
            run_kl += float(kl_loss.detach())
            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        nb = max(1, min(len(dl), step if epoch == start_epoch else len(dl)))
        # Per-epoch validation ceiling: the quantity we actually care about, and the only
        # signal that catches posterior collapse (train L1 keeps *decreasing* through it).
        ep_mean, _, _ = eval_ceiling(model, valid_hr, device, args.max_val_images)
        kl_term = args.kl_weight * (run_kl / nb)
        flag = ""
        if ep_mean > best_psnr:
            best_psnr = ep_mean
            save(epoch + 1, best_ckpt)
            flag = "  <- best"
        elif ep_mean < init_mean - 1.0:
            flag = "  <- WARNING: below init, possible posterior collapse"
        print(f"  epoch {epoch+1}/{args.epochs} | L1 {run_rec/nb:.4f} | KL {run_kl/nb:.1f} "
              f"| kl_term/L1 {kl_term/max(run_rec/nb,1e-9):.3f} | ceiling {ep_mean:.2f} dB "
              f"| {time.time()-t0:.0f}s | steps {step}{flag}", flush=True)
        save(epoch + 1)
        if stop:
            print("  --max-steps reached (smoke) -> stop", flush=True)
            break

    # Final ceiling: report the LAST epoch, but the checkpoint we hand downstream is BEST.
    last_mean, last_std, n = eval_ceiling(model, valid_hr, device, args.max_val_images)
    print(f"[ceiling] medical-klf4 {args.dataset} (last epoch): "
          f"AE PSNR {last_mean:.2f}±{last_std:.2f} dB (n={n})", flush=True)

    # Reload best and re-measure, so the reported number matches the shipped weights.
    load_state(model, best_ckpt)
    mean, std, n = eval_ceiling(model, valid_hr, device, args.max_val_images)
    improved = mean > init_mean
    print(f"[ceiling] medical-klf4 {args.dataset} (BEST, shipped): "
          f"AE PSNR {mean:.2f}±{std:.2f} dB (n={n}) | init was {init_mean:.2f} dB "
          f"-> {'+' if improved else ''}{mean-init_mean:.2f} dB "
          f"{'(fine-tuning helped)' if improved else '(NO IMPROVEMENT over natural init)'}",
          flush=True)
    with open(os.path.join(args.out_dir, "ceiling.json"), "w") as fh:
        json.dump({"dataset": args.dataset, "mode": "train", "epochs": args.epochs,
                   "kl_weight": args.kl_weight, "lr": args.lr,
                   "ckpt": best_ckpt, "last_ckpt": last_ckpt,
                   "ae_psnr_mean": mean, "ae_psnr_std": std, "n": n,
                   "ae_psnr_init": init_mean, "ae_psnr_last_epoch": last_mean,
                   "delta_vs_init": mean - init_mean}, fh, indent=2)
    print(f"=== done. medical checkpoint (best): {best_ckpt} ===", flush=True)


if __name__ == "__main__":
    main()
