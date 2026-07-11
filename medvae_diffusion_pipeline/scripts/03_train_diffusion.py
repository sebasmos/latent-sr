#!/usr/bin/env python3
"""
Train latent diffusion for super-resolution using channel concatenation.

Matches the LDM (Rombach et al.) SR conditioning approach:
  - LR latent is interpolated to HR spatial size and concatenated along channels
  - U-Net input: [noise + LR_interp] = 2*C channels
  - U-Net output: C channels (predicted clean HR latent)

This replaces the cross-attention conditioning used in 03a/03b, which
crushed all LR spatial information into a 4-dim vector.

Pipeline:
  HR (256x256) → MedVAE (f=4) → HR latent (3, 64, 64)
  LR  (64x64) → MedVAE (f=4) → LR latent (3, 16, 16) → interp to (3, 64, 64)
  concat [noise, LR_interp] = (6, 64, 64) → U-Net → predicted HR (3, 64, 64)
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader, Dataset


# ---------- Dataset ------------------------------------------------ #
class PairedLatentDataset(Dataset):
    """Load paired HR/LR latents from .npy files."""

    def __init__(self, latent_dir, augment=False):
        self.dir = Path(latent_dir)
        self.augment = augment
        self.hr = sorted(f for f in os.listdir(self.dir) if f.startswith("hr_"))
        self.lr = sorted(f for f in os.listdir(self.dir) if f.startswith("lr_"))

        if len(self.hr) != len(self.lr):
            raise ValueError(f"HR/LR count mismatch: {len(self.hr)} vs {len(self.lr)}")
        if not self.hr:
            raise RuntimeError(f"No hr_*.npy files in {latent_dir}")

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, i):
        h = torch.from_numpy(np.load(self.dir / self.hr[i])).float()
        l = torch.from_numpy(np.load(self.dir / self.lr[i])).float()
        if self.augment:
            # Random horizontal flip (dim 2) applied consistently to HR and LR
            if torch.rand(1).item() > 0.5:
                h = torch.flip(h, dims=[2])
                l = torch.flip(l, dims=[2])
            # Random vertical flip (dim 1)
            if torch.rand(1).item() > 0.5:
                h = torch.flip(h, dims=[1])
                l = torch.flip(l, dims=[1])
        return {"hr": h, "lr": l}

    def shape_info(self):
        hr = np.load(self.dir / self.hr[0], mmap_mode="r")
        lr = np.load(self.dir / self.lr[0], mmap_mode="r")
        return tuple(hr.shape), tuple(lr.shape)


# ---------- Lightning Module --------------------------------------- #
class LatentDiffusionSR(pl.LightningModule):
    """
    Latent diffusion for SR with channel concatenation conditioning.
    LR latent is interpolated to HR size and concatenated with noisy HR.
    """

    def __init__(
        self,
        learning_rate=1e-4,
        timesteps=1000,
        hr_channels=3,
        hr_size=64,
        lr_channels=3,
        lr_size=16,
        beta_schedule="cosine",
        loss_type="l1",
        use_snr_weighting=False,
        use_ema=False,
        ema_decay=0.9999,
        base_channels=128,
        channel_mult=(1, 2, 4, 4),
        attention_resolutions=(16,),
        layers_per_block=2,
        zlr_init=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.hr_channels = hr_channels
        self.hr_size = hr_size
        self.lr_channels = lr_channels
        self.lr_size = lr_size
        self.beta_schedule = beta_schedule
        self.loss_type = loss_type
        self.use_snr_weighting = use_snr_weighting
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.attention_resolutions = tuple(attention_resolutions)
        self.layers_per_block = layers_per_block
        self.zlr_init = zlr_init

        from diffusers import UNet2DModel

        block_out_channels = tuple(base_channels * mult for mult in self.channel_mult)
        down_block_types, up_block_types = self._build_block_types()

        # U-Net: input = noise (C) + interpolated LR (C) = 2*C channels
        # Output = C channels (predicted clean HR)
        self.model = UNet2DModel(
            sample_size=hr_size,
            in_channels=hr_channels + lr_channels,  # concat: noise + LR
            out_channels=hr_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )

        if beta_schedule == "linear":
            schedule = self._linear_schedule(timesteps)
        else:
            schedule = self._cosine_schedule(timesteps)
        self.register_buffer("alphas", schedule)

        self.model_ema = None
        if use_ema:
            import importlib.util

            ema_path = Path(__file__).resolve().parents[2] / "latent-diffusion" / "ldm" / "modules" / "ema.py"
            spec = importlib.util.spec_from_file_location("ldm_ema", ema_path)
            ema_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(ema_module)
            self.model_ema = ema_module.LitEma(self.model, decay=ema_decay, use_num_upates=True)

    def _cosine_schedule(self, T, s=0.008):
        x = torch.linspace(0, T + 1, T + 1)
        a = torch.cos(((x / (T + 1)) + s) / (1 + s) * torch.pi * 0.5) ** 2
        return torch.flip((a / a[0])[1:], (0,))

    def _linear_schedule(self, T, beta_start=1e-4, beta_end=2e-2):
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = torch.cumprod(1.0 - betas, dim=0)
        return alphas

    def _build_block_types(self):
        down_block_types = []
        current_resolution = self.hr_size
        for _mult in self.channel_mult:
            block_type = "AttnDownBlock2D" if current_resolution in self.attention_resolutions else "DownBlock2D"
            down_block_types.append(block_type)
            current_resolution = max(current_resolution // 2, 1)

        up_block_types = []
        current_resolution = max(self.hr_size // (2 ** (len(self.channel_mult) - 1)), 1)
        for idx, _mult in enumerate(reversed(self.channel_mult)):
            block_type = "AttnUpBlock2D" if current_resolution in self.attention_resolutions else "UpBlock2D"
            up_block_types.append(block_type)
            if idx < len(self.channel_mult) - 1:
                current_resolution *= 2
        return tuple(down_block_types), tuple(up_block_types)

    def _prepare_lr_condition(self, lr):
        """Interpolate LR latent to HR spatial size."""
        return F.interpolate(
            lr,
            size=(self.hr_size, self.hr_size),
            mode="bilinear",
            align_corners=False,
        )

    def _diffusion_step(self, hr, lr):
        """Shared forward for training and validation."""
        bs = hr.size(0)
        t = torch.randint(0, self.timesteps, (bs,), device=self.device)

        # Interpolate LR latent to HR size (used for conditioning and optionally as noise prior)
        lr_upsampled = self._prepare_lr_condition(lr)

        if self.zlr_init:
            # Issue #56: use z_lr as noise prior instead of randn.
            # Forward diffusion: z_t = alpha[t]*z_hr + (1-alpha[t])*z_lr
            # Model learns to refine from z_lr toward z_hr (residual latent refinement).
            noise = lr_upsampled.detach()
        else:
            noise = torch.randn_like(hr)

        # Forward diffusion: add noise to HR latent
        a = self.alphas[t].view(bs, 1, 1, 1)
        noisy_hr = a.sqrt() * hr + (1 - a).sqrt() * noise

        # Concatenate noisy HR with LR condition
        model_input = torch.cat([noisy_hr, lr_upsampled], dim=1)

        # Predict clean HR latent
        pred = self.model(model_input, t).sample
        return self._compute_loss(pred, hr, t)

    def _compute_loss(self, pred, target, t):
        reduction = "none" if self.use_snr_weighting else "mean"
        if self.loss_type == "l2":
            loss_map = F.mse_loss(pred, target, reduction=reduction)
        else:
            loss_map = F.l1_loss(pred, target, reduction=reduction)

        if not self.use_snr_weighting:
            return loss_map

        snr = self.alphas[t] / (1 - self.alphas[t] + 1e-8)
        weights = snr / (snr + 1.0)
        while weights.ndim < loss_map.ndim:
            weights = weights.unsqueeze(-1)
        return (loss_map.mean(dim=(1, 2, 3)) * weights.view(weights.shape[0])).mean()

    def training_step(self, batch, batch_idx):
        loss = self._diffusion_step(batch["hr"], batch["lr"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._diffusion_step(batch["hr"], batch["lr"])
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.model_ema is not None:
            self.model_ema(self.model)

    def _maybe_swap_to_ema(self):
        if self.model_ema is None:
            return False
        self.model_ema.store(self.model.parameters())
        self.model_ema.copy_to(self.model)
        return True

    def _restore_from_ema(self):
        if self.model_ema is not None:
            self.model_ema.restore(self.model.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    @torch.no_grad()
    def sample(self, lr, T=None):
        """Cold diffusion sampling: noise → predicted HR conditioned on LR.

        Schedule note: this code uses a FLIPPED cosine schedule where
        alpha[0] ≈ 0 (most noisy) and alpha[T-1] ≈ 1 (cleanest). The
        correct denoising direction is therefore 0 → T-1, i.e. the loop
        runs i = 0, 1, ..., T-2 with a final step at T-1.

        Issue #58 investigation (2026-03-22): Two hypotheses tested and
        refuted — (1) reversed loop direction (T-1→0) made T=1000 PSNR
        catastrophic (5.24 dB); (2) std=1.0 init made T=1000 worse (23.59
        vs 25.25 dB). The original code (0→T-1 loop + 0.5*randn init) is
        empirically optimal. Root cause of T=50>T=1000 remains unexplained.

        Issue #56: when zlr_init=True the initial latent at t=0 is the
        bilinearly upsampled z_lr instead of random noise, consistent with
        the modified training prior (noise = z_lr at t=0).
        """
        T = T or self.timesteps
        lr_up = self._prepare_lr_condition(lr)

        if self.zlr_init:
            # Start from z_lr at t=0 — matches training prior (noise = z_lr)
            z = lr_up.clone()
        else:
            # Empirically validated: 0.5*randn outperforms std=1.0 at all T
            # (see issue #58 investigation, jobs 10813228 vs original baseline)
            z = 0.5 * torch.randn(
                lr.size(0), self.hr_channels, self.hr_size, self.hr_size,
                device=self.device,
            )

        using_ema = self._maybe_swap_to_ema()

        try:
            # Forward loop: 0 → T-2 (denoising direction for flipped schedule)
            for i in range(T - 1):
                t = torch.full((z.size(0),), i, device=self.device, dtype=torch.long)
                model_input = torch.cat([z, lr_up], dim=1)
                x0 = self.model(model_input, t).sample
                eps = (z - self.alphas[i].sqrt() * x0) / ((1 - self.alphas[i]).sqrt() + 1e-4)
                z = z + (self.alphas[i + 1].sqrt() - self.alphas[i].sqrt()) * x0 + \
                    ((1 - self.alphas[i + 1]).sqrt() - (1 - self.alphas[i]).sqrt()) * eps

            # Final prediction at t=T-1 (cleanest timestep, alpha[T-1] ≈ 1)
            t = torch.full((z.size(0),), T - 1, device=self.device, dtype=torch.long)
            model_input = torch.cat([z, lr_up], dim=1)
            return self.model(model_input, t).sample
        finally:
            if using_ema:
                self._restore_from_ema()


# ---------- CLI ---------------------------------------------------- #
def parse_args():
    def parse_int_list(value):
        return tuple(int(part) for part in value.split(",") if part)

    p = argparse.ArgumentParser(
        description=(
            "Train latent diffusion SR with the supported x0-prediction "
            "channel-concat setup."
        )
    )
    p.add_argument("--train-latent-dir", type=str, required=True)
    p.add_argument("--val-latent-dir", type=str, default=None,
                    help="Validation latent dir (default: use train)")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta-schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--loss-type", choices=["l1", "l2"], default="l1")
    p.add_argument("--use-snr-weighting", action="store_true")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--channel-mult", type=parse_int_list, default=(1, 2, 4, 4))
    p.add_argument("--attention-resolutions", type=parse_int_list, default=(16,))
    p.add_argument("--layers-per-block", type=int, default=2)
    p.add_argument("--augment", action="store_true",
                   help="Apply random H/V flips to latents during training (issue #46)")
    p.add_argument("--zlr-init", action="store_true",
                   help="Use z_lr as noise prior instead of randn (issue #56): "
                        "trains residual latent refinement z_lr→z_hr instead of "
                        "denoising pure Gaussian noise.")
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    print(f"\n{'='*60}")
    print("Latent Diffusion SR (Channel Concatenation)")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Train: {args.train_latent_dir}")
    print(f"Val: {args.val_latent_dir or '(same as train)'}")
    print(f"Output: {args.output_dir}")
    print(f"Beta schedule: {args.beta_schedule}")
    print(f"Loss: {args.loss_type} | SNR weighting: {args.use_snr_weighting}")
    print(f"EMA: {args.use_ema} | decay={args.ema_decay}")
    print(f"Augment (H/V flip): {args.augment}")
    print(f"z_lr init (issue #56): {args.zlr_init}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stage the latent directories to node-local disk before training. With
    # num_workers=0 every batch fetch is a SYNCHRONOUS np.load() in the main
    # thread, so a single slow/contended read on the network LCELI mount
    # stalls the whole pipeline. Observed exactly this: diagnostic callbacks
    # showed a resumed run clearing checkpoint restore + sanity check + even
    # the first training batch (on_train_batch_end fired), then GPU utilization
    # went bursty (33%->0%->0%) and stalled again a few batches later - not a
    # fixed deadlock point, but per-batch I/O stalls under variable filesystem
    # load (confirmed separately: staging the same 404 MB checkpoint file took
    # anywhere from 0.2s to 91s). Copying the whole (small, ~0.5-1 GB) latent
    # dataset once upfront removes the network FS from the per-batch hot path
    # entirely for the rest of the run. Falls back to the network path if
    # staging fails for any reason.
    stage_root = os.environ.get("SLURM_TMPDIR") or "/tmp"
    local_latent_dir = Path(stage_root) / f"latents_{os.environ.get('SLURM_JOB_ID', os.getpid())}"

    def stage_dir(src, label):
        src = Path(src)
        dst = local_latent_dir / label
        try:
            print(f"Staging {label} latents: {src} -> {dst}", flush=True)
            t0 = time.time()
            shutil.copytree(src, dst)
            n = len(list(dst.glob("hr_*.npy")))
            print(f"  staged {n} pairs in {time.time() - t0:.1f}s", flush=True)
            return dst
        except Exception as e:
            print(f"  latent staging failed ({e}); falling back to network path", flush=True)
            return src

    train_latent_dir = stage_dir(args.train_latent_dir, "train")
    val_latent_src = args.val_latent_dir or args.train_latent_dir
    if val_latent_src == args.train_latent_dir:
        val_latent_dir = train_latent_dir  # same source, reuse the local copy
    else:
        val_latent_dir = stage_dir(val_latent_src, "val")

    # Load datasets
    train_ds = PairedLatentDataset(train_latent_dir, augment=args.augment)
    hr_shape, lr_shape = train_ds.shape_info()
    print(f"\nTrain: {len(train_ds)} pairs")
    print(f"HR latent: {hr_shape}")
    print(f"LR latent: {lr_shape}")

    val_ds = PairedLatentDataset(val_latent_dir) if Path(val_latent_dir).exists() else train_ds
    print(f"Val: {len(val_ds)} pairs")

    hr_c, hr_h, hr_w = hr_shape
    lr_c, lr_h, lr_w = lr_shape

    print(f"\nU-Net input channels: {hr_c} (noise) + {lr_c} (LR interp) = {hr_c + lr_c}")
    print(f"LR spatial: {lr_h}x{lr_w} → interpolate to {hr_h}x{hr_w}")

    # DataLoaders
    # pin_memory=False: with num_workers=0, cudaHostRegister pinning happens
    # SYNCHRONOUSLY in the main thread on the first batch fetch. Observed a
    # reproducible hang (D-state, 0% GPU) at exactly that point on checkpoint
    # RESUME (never on a fresh start) across 7 different physical nodes, right
    # after Lightning's RNG/precision-plugin state restore touches the CUDA
    # context — consistent with a known class of pinned-memory/driver-lock
    # contention on busy shared GPU nodes. Throughput cost is negligible with
    # num_workers=0 (no background prefetch to overlap the transfer anyway).
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)

    # Model
    model = LatentDiffusionSR(
        learning_rate=args.lr,
        timesteps=args.timesteps,
        hr_channels=hr_c, hr_size=hr_h,
        lr_channels=lr_c, lr_size=lr_h,
        beta_schedule=args.beta_schedule,
        loss_type=args.loss_type,
        use_snr_weighting=args.use_snr_weighting,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        base_channels=args.base_channels,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions,
        layers_per_block=args.layers_per_block,
        zlr_init=args.zlr_init,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    checkpoint_cb = ModelCheckpoint(
        dirpath=out / "checkpoints",
        filename="ldm-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss", mode="min", save_top_k=3, save_last=True,
    )

    # DIAGNOSTIC (temporary): the resume path deadlocks (D-state, 0% GPU) right
    # after Lightning restores checkpoint state, reproduced 11x across 8+ nodes.
    # Neither staging the checkpoint to local disk nor pin_memory=False fixed it,
    # and CPU-time barely ticks during the hang (not a slow-but-real skip-ahead
    # read loop) - it's a genuine block on one call, not merely slow I/O. These
    # hooks pinpoint exactly which lifecycle stage is never reached, so the next
    # fix targets the actual bottleneck instead of guessing again.
    class DiagCallback(pl.Callback):
        def on_sanity_check_start(self, trainer, pl_module):
            print("[DIAG] on_sanity_check_start", flush=True)
        def on_sanity_check_end(self, trainer, pl_module):
            print("[DIAG] on_sanity_check_end", flush=True)
        def on_train_epoch_start(self, trainer, pl_module):
            print(f"[DIAG] on_train_epoch_start epoch={trainer.current_epoch}", flush=True)
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch_idx == 0:
                print(f"[DIAG] on_train_batch_start epoch={trainer.current_epoch} batch_idx=0", flush=True)
        def on_before_backward(self, trainer, pl_module, loss):
            if trainer.global_step < 2:
                print(f"[DIAG] on_before_backward step={trainer.global_step} loss={loss.item():.4f}", flush=True)
        def on_after_backward(self, trainer, pl_module):
            if trainer.global_step < 2:
                print(f"[DIAG] on_after_backward step={trainer.global_step}", flush=True)
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            if trainer.global_step < 2:
                print(f"[DIAG] on_before_optimizer_step step={trainer.global_step}", flush=True)
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx == 0:
                print(f"[DIAG] on_train_batch_end epoch={trainer.current_epoch} batch_idx=0", flush=True)
        def on_validation_start(self, trainer, pl_module):
            print("[DIAG] on_validation_start", flush=True)
        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            if batch_idx == 0:
                print("[DIAG] on_validation_batch_start batch_idx=0", flush=True)
        def on_fit_start(self, trainer, pl_module):
            print("[DIAG] on_fit_start", flush=True)

    trainer = Trainer(
        default_root_dir=str(out),
        max_epochs=args.epochs,
        accelerator="auto",
        devices=args.gpus if torch.cuda.is_available() else "auto",
        callbacks=[checkpoint_cb, LearningRateMonitor("epoch"), DiagCallback()],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Resume from last checkpoint if it exists (supports preemption/timeout).
    # Stage the checkpoint to node-LOCAL disk before loading: reading a large
    # pickle directly off the shared network FS (LCELI, heavily loaded by
    # concurrent jobs) was observed to deadlock (D-state, 0% GPU) reproducibly
    # across 5 different physical nodes, always right after Lightning restores
    # state. Copying first isolates the read from network-FS contention/RPC
    # stalls regardless of the exact internal cause. New checkpoints still
    # save to the network path (ModelCheckpoint dirpath is unchanged) — only
    # the resume READ is redirected to local scratch.
    last_ckpt = out / "checkpoints" / "last.ckpt"
    ckpt_path = None
    local_ckpt_dir = None
    if last_ckpt.exists():
        stage_root = os.environ.get("SLURM_TMPDIR") or "/tmp"
        local_ckpt_dir = Path(stage_root) / f"resume_ckpt_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        local_ckpt = local_ckpt_dir / "last.ckpt"
        try:
            print(f"\nStaging checkpoint to local disk: {last_ckpt} -> {local_ckpt}")
            t0 = time.time()
            shutil.copy2(last_ckpt, local_ckpt)
            print(f"  staged {local_ckpt.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f}s")
            ckpt_path = str(local_ckpt)
        except Exception as e:
            print(f"  local staging failed ({e}); falling back to network path")
            ckpt_path = str(last_ckpt)
        print(f"Resuming from checkpoint: {ckpt_path}")

    try:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    finally:
        if local_ckpt_dir is not None:
            shutil.rmtree(local_ckpt_dir, ignore_errors=True)
        shutil.rmtree(local_latent_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Checkpoints: {out / 'checkpoints'}")


if __name__ == "__main__":
    main()
