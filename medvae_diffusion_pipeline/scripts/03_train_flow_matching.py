#!/usr/bin/env python3
"""
Train latent flow matching (rectified flow) for super-resolution.

Implements Issue #77 — Path A: Rectified Flow in MedVAE latent space.

References: FlowSR (Xu et al., ICCV 2025), ELIR (BMVC 2025).
MedVAE latent application is novel.

Mathematical formulation
------------------------
DDPM (old): z_t = sqrt(alpha_t)*z_HR + sqrt(1-alpha_t)*noise  →  predict x̂₀
Rectified flow (new, this script):
  t ~ Uniform(0, 1)
  z_t = (1 - t)*noise + t*z_HR        (linear interpolation)
  v* = z_HR - noise                   (constant target velocity along straight path)
  model predicts: v_θ(z_t, t_int)
  loss: L1 or MSE  ||v_θ - v*||

Euler ODE sampling (inference):
  Start from z ~ N(0,I) at t=0.
  For i = 0, ..., steps-1:
      t = i / steps
      v = model(z, t_int)
      z = z + v * (1/steps)
  Return z at t=1  → predicted HR latent.

Conditioning (same as DDPM):
  LR latent is bilinearly interpolated to HR spatial size,
  concatenated channel-wise with z_t → 2*C input channels.

Compatibility:
  eval_diffusion_sr.py calls model.sample(lr, T=timesteps).
  LatentFlowMatchingSR.sample() accepts T as a keyword arg.

Pipeline:
  HR (256×256) → MedVAE (f=4) → HR latent (3, 64, 64)
  LR (64×64)   → MedVAE (f=4) → LR latent (3, 16, 16) → interp (3, 64, 64)
  concat [z_t, LR_interp] = (6, 64, 64) → U-Net → velocity (3, 64, 64)
"""

import argparse
import os
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
    """Load paired HR/LR latents from .npy files.

    Copied verbatim from 03_train_diffusion.py — same file format.
    """

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
    Latent flow matching SR with channel concatenation conditioning.

    Named LatentDiffusionSR for eval_diffusion_sr.py compatibility:
      module.LatentDiffusionSR.load_from_checkpoint(...)
      ldm.sample(lr, T=timesteps)

    Internally this is a rectified flow model, not DDPM.
    The class alias LatentFlowMatchingSR is also exported at module level.

    Architecture:
      - UNet2DModel from diffusers (same as DDPM baseline)
      - Input: z_t (C channels) concatenated with LR_interp (C channels) = 2*C
      - Output: predicted velocity v (C channels)
      - Integer timestep t_int = int(t * (T-1))  passed to UNet

    Training:
      t ~ Uniform(0,1), z_t = (1-t)*noise + t*z_HR, target v* = z_HR - noise
      Loss: L1 or MSE  ||v_θ(z_t, t_int) - v*||

    Inference (Euler ODE):
      z_0 = randn(); dt = 1/steps
      for i in range(steps): z += v_θ(z, i/steps) * dt
    """

    def __init__(
        self,
        learning_rate=1e-4,
        timesteps=1000,
        hr_channels=3,
        hr_size=64,
        lr_channels=3,
        lr_size=16,
        loss_type="l1",
        num_sample_steps=16,
        use_ema=False,
        ema_decay=0.9999,
        base_channels=128,
        channel_mult=(1, 2, 4, 4),
        attention_resolutions=(16,),
        layers_per_block=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.hr_channels = hr_channels
        self.hr_size = hr_size
        self.lr_channels = lr_channels
        self.lr_size = lr_size
        self.loss_type = loss_type
        self.num_sample_steps = num_sample_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.attention_resolutions = tuple(attention_resolutions)
        self.layers_per_block = layers_per_block

        from diffusers import UNet2DModel

        block_out_channels = tuple(base_channels * mult for mult in self.channel_mult)
        down_block_types, up_block_types = self._build_block_types()

        # U-Net: input = z_t (C) + interpolated LR (C) = 2*C channels
        # Output = C channels (predicted velocity)
        self.model = UNet2DModel(
            sample_size=hr_size,
            in_channels=hr_channels + lr_channels,  # concat: z_t + LR_interp
            out_channels=hr_channels,               # velocity prediction
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )

        self.model_ema = None
        if use_ema:
            import importlib.util

            ema_path = (
                Path(__file__).resolve().parents[2]
                / "latent-diffusion" / "ldm" / "modules" / "ema.py"
            )
            spec = importlib.util.spec_from_file_location("ldm_ema", ema_path)
            ema_module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(ema_module)
            self.model_ema = ema_module.LitEma(self.model, decay=ema_decay, use_num_upates=True)

    def _build_block_types(self):
        down_block_types = []
        current_resolution = self.hr_size
        for _mult in self.channel_mult:
            block_type = (
                "AttnDownBlock2D"
                if current_resolution in self.attention_resolutions
                else "DownBlock2D"
            )
            down_block_types.append(block_type)
            current_resolution = max(current_resolution // 2, 1)

        up_block_types = []
        current_resolution = max(self.hr_size // (2 ** (len(self.channel_mult) - 1)), 1)
        for idx, _mult in enumerate(reversed(self.channel_mult)):
            block_type = (
                "AttnUpBlock2D"
                if current_resolution in self.attention_resolutions
                else "UpBlock2D"
            )
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

    def _flow_step(self, hr, lr):
        """
        Rectified flow forward pass for training and validation.

        Samples t ~ Uniform(0,1), constructs the linear interpolation
        z_t = (1-t)*noise + t*z_HR, and regresses the constant velocity
        v* = z_HR - noise.
        """
        bs = hr.size(0)

        # Sample continuous t in [0, 1)
        t_frac = torch.rand(bs, device=self.device)       # (B,)

        noise = torch.randn_like(hr)

        # Rectified flow interpolation: z_t = (1-t)*noise + t*z_HR
        t_view = t_frac.view(bs, 1, 1, 1)
        z_t = (1.0 - t_view) * noise + t_view * hr

        # Target velocity (constant along straight paths)
        v_target = hr - noise                              # (B, C, H, W)

        # Convert to integer timestep for UNet conditioning
        t_int = (t_frac * (self.timesteps - 1)).long()    # (B,)

        # Condition on LR: interpolate and concatenate
        lr_upsampled = self._prepare_lr_condition(lr)
        model_input = torch.cat([z_t, lr_upsampled], dim=1)

        # Predict velocity
        v_pred = self.model(model_input, t_int).sample    # (B, C, H, W)

        # Flow matching loss
        if self.loss_type == "l2":
            return F.mse_loss(v_pred, v_target)
        return F.l1_loss(v_pred, v_target)

    def training_step(self, batch, batch_idx):
        loss = self._flow_step(batch["hr"], batch["lr"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._flow_step(batch["hr"], batch["lr"])
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @torch.no_grad()
    def sample(self, lr, T=None, steps=None):
        """
        Euler ODE sampling: noise at t=0 → HR latent at t=1.

        Parameters
        ----------
        lr : torch.Tensor  (B, C_lr, H_lr, W_lr)
            Low-resolution latent batch.
        T : int, optional
            Backward-compat alias for steps (eval_diffusion_sr.py passes T=timesteps).
            When T is provided it is used as the number of Euler steps, NOT as the
            diffusion timestep count — keep T small (e.g. 16) for fast inference.
        steps : int, optional
            Number of Euler integration steps. Default: self.num_sample_steps.

        Returns
        -------
        torch.Tensor  (B, C_hr, H_hr, W_hr)
            Predicted HR latent.
        """
        steps = steps or T or self.num_sample_steps
        bs = lr.size(0)

        # Start from pure noise at t=0
        z = torch.randn(
            bs, self.hr_channels, self.hr_size, self.hr_size,
            device=self.device,
        )
        lr_up = self._prepare_lr_condition(lr)
        using_ema = self._maybe_swap_to_ema()

        dt = 1.0 / steps
        try:
            for i in range(steps):
                t_frac = i / steps                         # current t in [0,1)
                t_int = int(t_frac * (self.timesteps - 1))
                t_tensor = torch.full(
                    (bs,), t_int, device=self.device, dtype=torch.long
                )
                model_input = torch.cat([z, lr_up], dim=1)
                v = self.model(model_input, t_tensor).sample   # predicted velocity
                z = z + v * dt                             # Euler step toward data
        finally:
            if using_ema:
                self._restore_from_ema()

        return z   # predicted HR latent at t=1


# Alias: keep the original name for internal use and PLAN.md references
LatentFlowMatchingSR = LatentDiffusionSR


# ---------- CLI ---------------------------------------------------- #
def parse_args():
    def parse_int_list(value):
        return tuple(int(part) for part in value.split(",") if part)

    p = argparse.ArgumentParser(
        description=(
            "Train latent flow matching (rectified flow) SR — Issue #77 Path A. "
            "Replaces DDPM with straight-line ODE paths in MedVAE latent space."
        )
    )
    p.add_argument("--train-latent-dir", type=str, required=True)
    p.add_argument(
        "--val-latent-dir", type=str, default=None,
        help="Validation latent dir (default: use train dir)",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--timesteps", type=int, default=1000,
        help=(
            "Integer timestep grid size for UNet conditioning. "
            "Continuous t in [0,1] is mapped to integers via t_int = int(t*(T-1)). "
            "Larger T gives finer conditioning granularity. Default: 1000."
        ),
    )
    p.add_argument(
        "--num-sample-steps", type=int, default=16,
        help="Number of Euler ODE steps at inference (default: 16). "
             "16 steps is typically sufficient for rectified flow.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss-type", choices=["l1", "l2"], default="l1")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--channel-mult", type=parse_int_list, default=(1, 2, 4, 4))
    p.add_argument("--attention-resolutions", type=parse_int_list, default=(16,))
    p.add_argument("--layers-per-block", type=int, default=2)
    p.add_argument(
        "--augment", action="store_true",
        help="Apply random H/V flips to latents during training.",
    )
    # TODO (Issue #77 Path B): consistency distillation from an existing DDPM checkpoint.
    # To implement: load LatentDiffusionSR from --distill-from, then train the flow model
    # with a combined flow-matching + consistency loss against the teacher's x0 predictions.
    p.add_argument(
        "--distill-from", type=str, default=None,
        help="(FUTURE) Path to a LatentDiffusionSR DDPM checkpoint for "
             "consistency distillation (Path B). Not yet implemented.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    if args.distill_from is not None:
        # TODO: implement Path B consistency distillation here.
        # Load LatentDiffusionSR from args.distill_from, iterate over
        # teacher predictions, add consistency loss term.
        raise NotImplementedError(
            "--distill-from (Path B consistency distillation) is not yet implemented. "
            "Run without --distill-from for Path A rectified flow training."
        )

    print(f"\n{'='*60}")
    print("Latent Flow Matching SR (Rectified Flow — Issue #77 Path A)")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    print(f"Train: {args.train_latent_dir}")
    print(f"Val: {args.val_latent_dir or '(same as train)'}")
    print(f"Output: {args.output_dir}")
    print(f"Loss: {args.loss_type}")
    print(f"Timestep grid T: {args.timesteps}")
    print(f"Euler sample steps: {args.num_sample_steps}")
    print(f"EMA: {args.use_ema} | decay={args.ema_decay}")
    print(f"Augment (H/V flip): {args.augment}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = PairedLatentDataset(args.train_latent_dir, augment=args.augment)
    hr_shape, lr_shape = train_ds.shape_info()
    print(f"\nTrain: {len(train_ds)} pairs")
    print(f"HR latent: {hr_shape}")
    print(f"LR latent: {lr_shape}")

    val_dir = args.val_latent_dir or args.train_latent_dir
    val_ds = PairedLatentDataset(val_dir) if Path(val_dir).exists() else train_ds
    print(f"Val: {len(val_ds)} pairs")

    hr_c, hr_h, hr_w = hr_shape
    lr_c, lr_h, lr_w = lr_shape

    print(f"\nU-Net input channels: {hr_c} (z_t) + {lr_c} (LR interp) = {hr_c + lr_c}")
    print(f"LR spatial: {lr_h}x{lr_w} → interpolate to {hr_h}x{hr_w}")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model (LatentDiffusionSR alias makes this compatible with eval_diffusion_sr.py)
    model = LatentDiffusionSR(
        learning_rate=args.lr,
        timesteps=args.timesteps,
        hr_channels=hr_c, hr_size=hr_h,
        lr_channels=lr_c, lr_size=lr_h,
        loss_type=args.loss_type,
        num_sample_steps=args.num_sample_steps,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        base_channels=args.base_channels,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions,
        layers_per_block=args.layers_per_block,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=out / "checkpoints",
        filename="flow-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss", mode="min", save_top_k=3, save_last=True,
    )

    trainer = Trainer(
        default_root_dir=str(out),
        max_epochs=args.epochs,
        accelerator="auto",
        devices=args.gpus if torch.cuda.is_available() else "auto",
        callbacks=[checkpoint_cb, LearningRateMonitor("epoch")],
        log_every_n_steps=10,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Resume from last checkpoint (supports SLURM preemption / timeout)
    last_ckpt = out / "checkpoints" / "last.ckpt"
    ckpt_path = str(last_ckpt) if last_ckpt.exists() else None
    if ckpt_path:
        print(f"\nResuming from checkpoint: {ckpt_path}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Checkpoints: {out / 'checkpoints'}")


if __name__ == "__main__":
    main()
