# Latent Geometry Shapes Fidelity in Diffusion-Based Medical Image Super-Resolution

**Reproducibility Package**

[![arXiv](https://img.shields.io/badge/arXiv-2604.12152-b31b1b.svg)](https://arxiv.org/abs/2604.12152)

> Submitted to Nature *Scientific Reports* (major revision). Originally submitted as
> *"Domain-Specific Latent Representations Improve the Fidelity of Diffusion-Based Medical
> Image Super-Resolution."*

> **Core finding:** Replacing SD-VAE with MedVAE in an otherwise identical latent-diffusion
> pipeline yields **+1.92 to +3.60 dB** true-HR PSNR across knee MRI, brain MRI, and chest
> X-ray (n=1,820; Cohen's d = 1.17–1.61, all p < 10⁻²⁰). Capacity-matched controls and a
> natural-image VAE at MedVAE's identical latent geometry (KL-f4) show this advantage is
> driven primarily by **latent geometry** (resolution and capacity): medical-domain
> pretraining contributes a real but secondary +0.3 to +1.0 dB at the smaller matched
> geometry, and is within ±0.14 dB of zero at the larger one.

---

## What's new since the original submission (major-revision cycle)

The original submission scored SR output against each VAE's own decoded HR latent
(self-referential). The revision switched to scoring against the **true HR image**
throughout, and added four new controls that decompose the headline gap:

| Addition | What it answers | Key result |
|---|---|---|
| **True-HR evaluation convention** | Is the self-referential metric inflating the gap? | Headline gap under true-HR: +1.92/+2.83/+3.60 dB (MRNet/BraTS/CXR) |
| **Capacity-matched controls** (`medvae_8_4_2d`, `medvae_4_1_2d`) | Is the gain domain-specific or just higher latent capacity? | Domain (upper bound) +0.32–+0.95 dB; dominant contributor is domain on MRNet, resolution on BraTS, capacity on CXR |
| **KL-f4 natural-VAE control** (CompVis KL-f4, 3×64×64) | Domain vs. geometry at MedVAE's *exact* geometry | Domain@64² real but modest (+0.74–+1.17 dB); geometry alone accounts for 70–89% of the AE-ceiling gap |
| **Three-seed reproducibility sweep** (seeds 42/43/44) | Are single-run numbers representative? | Seed-to-seed SD (0.08–0.18 dB) is 1–2 orders of magnitude below the headline effect |
| **Target-dataset AE adaptation control** | Domain pretraining vs. in-distribution adaptation | Adaptation raises SR PSNR +0.96 to +1.65 dB; geometry (+1.9 to +3.6 dB) still dominates on 2 of 3 datasets |
| **Patient-level reanalysis** (BraTS) | Pseudoreplication from scoring 20 slices/patient as independent | Patient-level (n=35) confirms the image-level conclusion: +2.905 dB, p=2.5×10⁻⁷ |

See `slurm/revision/` and `scripts/*_truehr.py` for the code, and
`tests/test_revision_validation.py` for golden-number consistency checks on every
number above.

---

## Directory layout

```
latent-sr/
├── README.md                        ← this file
├── reproduce_all.sh                 ← master end-to-end script (SLURM)
├── environment.yml                  ← conda environment
├── requirements.txt                 ← pip packages
├── configs/                         ← model / training configuration files
│
├── medvae_diffusion_pipeline/       ← core extraction + training package
│   ├── paper_validation_config.py
│   ├── validation_framework.py
│   └── scripts/
│       ├── 02_extract_medvae_embeddings.py  ← cache VAE latents (medvae/sdvae/klf4 backends)
│       ├── 03_train_diffusion.py            ← LDM UNet training (x0-pred, L1, cosine beta)
│       └── 03_train_flow_matching.py        ← rectified-flow variant (Supp Table 3)
│
├── scripts/                         ← evaluation, statistics, figure generation
│   ├── Data prep
│   │   ├── prep_mrnet_lr_64.py
│   │   ├── prepare_brats_sr.py
│   │   ├── prepare_mimic_cxr_sr.py
│   │   └── create_lr_images.py
│   │
│   ├── Evaluation (pre-revision, self-referential convention)
│   │   ├── eval_diffusion_sr.py     ← main SR: PSNR / MS-SSIM / LPIPS (Table 1)
│   │   ├── eval_vae_reconstruction.py   ← AE ceiling (Supp Table 11)
│   │   ├── eval_baselines.py        ← bicubic + ESRGAN
│   │   ├── eval_swinir.py           ← SwinIR baseline
│   │   ├── eval_multiscale_sr.py    ← multi-scale bicubic (2×/4×/8×)
│   │   ├── gen_bicubic_multiscale.py ← 2×/8× bicubic SR image generation (Supp Table 4)
│   │   ├── eval_frequency_analysis.py   ← Haar wavelet + FFT power spectrum (Supp Table 6)
│   │   ├── eval_hallucination_quantification.py  ← pixel-level decomposition (Supp Table 7)
│   │   ├── eval_multiresolution_embedding.py     ← SR-vs-HR latent cosine sim (Supp Table 8)
│   │   ├── compute_roi_metrics.py   ← BraTS tumour-vs-background ROI PSNR/SSIM (Supp Table 10)
│   │   ├── compute_fid.py           ← Fréchet Inception Distance (Supp Table 12)
│   │   ├── eval_sr_hr_diffmaps.py   ← signed difference maps
│   │   ├── eval_multit_embedding.py ← multi-T cosine sim (Supp Table 13)
│   │   ├── reencode_sr_latents.py   ← re-encode CXR SR images through MedVAE (R2.11)
│   │   └── replicate_table4.py      ← MedVAE-paper Table 4 protocol (provenance/sanity check)
│   │
│   ├── Evaluation (revision, true-HR convention)
│   │   ├── eval_diffusion_sr_truehr.py    ← true-HR PSNR/MS-SSIM/LPIPS + AE-ceiling recomputed inline
│   │   ├── recompute_regression_truehr.py ← n=15 AE-ceiling→SR regression, single consistent metric
│   │   ├── patient_level_stats.py         ← BraTS pseudoreplication check (image- vs. patient-level)
│   │   ├── run_hallucination_stat_test.py ← Wilcoxon + two-proportion z-test on hallucination rates
│   │   └── make_hallucination_figures.sh  ← regenerate + reproducibility-check the hallucination figures
│   │
│   ├── Statistical analysis
│   │   ├── compute_effect_sizes.py  ← Cohen's d + bootstrap CI (Table 2 / Supp Table 5)
│   │   ├── compute_pvalues.py       ← Wilcoxon signed-rank
│   │   └── compute_bland_altman.py  ← Bland-Altman intensity agreement (Supp Table 9)
│   │
│   └── Figure generation
│       ├── generate_paper1_fig1.py / generate_paper1_fig1_truehr.py  ← Fig 1: PSNR/LPIPS bar chart
│       ├── plot_ae_ceiling_correlation.py ← AE ceiling scatter
│       ├── generate_freq_figures.py      ← wavelet + FFT panels
│       ├── generate_visual_comparisons.py ← visual + diffmap panels
│       ├── generate_paper1_composite_figs.py ← embedding composite
│       ├── generate_paper1_fig6.py / generate_paper1_fig6_truehr.py ← stats (BA + forest + violin)
│       ├── plot_perception_distortion.py ← Supp Fig 1: perception-distortion
│       └── collect_paper1_figures.sh     ← copies all figures to figures-paper-1/
│
├── slurm/                           ← SLURM job scripts (original submission)
│   ├── _env.sh                      ← shared env setup, sourced by every job script
│   ├── prep_brats.sh
│   ├── run_brats_medvae.sh / run_cxr_medvae.sh
│   ├── run_brats_sdvae.sh / run_cxr_sdvae.sh
│   ├── run_mrnet_eval.sh
│   ├── run_baselines.sh / run_swinir.sh
│   ├── run_frequency_analysis.sh / run_hallucination.sh
│   ├── run_multiresolution_embedding.sh / run_sr_hr_diffmaps.sh
│   ├── run_step_ablation.sh
│   ├── run_t16_{mrnet,brats,cxr}.sh
│   ├── run_flow_matching_{mrnet,brats,cxr}.sh
│   └── reeval_for_pvalues.sh
│   │
│   └── revision/                    ← SLURM jobs added during the major-revision cycle
│       ├── klf4_vae.py              ← shared KL-f4 encode/decode backend
│       ├── eval_klf4_ae.py          ← KL-f4 AE-ceiling control (natural-side, matched geometry)
│       ├── finetune_klf4.py         ← target-dataset AE adaptation (kl_weight=1e-8, best-ckpt guard)
│       ├── run_matched_vae.sh       ← generic capacity-matched-VAE training job
│       ├── run_mrnet_medvae_4_1.sh / run_cxr_medvae_4_1.sh / run_brats_medvae_4_1_valid.sh
│       ├── run_klf4_ae.sh / run_klf4_sr.sh / run_klf4_sr_seedsweep.sh
│       ├── run_klf4med_ceiling.sh / run_klf4med_finetune.sh / run_klf4med_sr.sh / run_klf4med_smoke.sh
│       └── run_truehr_eval_{brats_mrnet,capacity,cxr}.sh
│
├── tests/                           ← environment, installation, and results validation
│   ├── test_installation.py / test_environment.py
│   ├── test_validation_framework.py / test_paper_validation.py
│   ├── test_diffusion_architecture.py / test_reproducibility.py
│   ├── test_revision_validation.py  ← golden-number checks for every number in "What's new" above
│   └── run_all_tests.sh
│
└── figures/                         ← pre-generated paper figures
    ├── fig_pipeline.{svg,png,pdf}
    ├── fig1.{png,pdf}
    ├── fig2{a,b,c}_{mrnet,brats,cxr}_{visual,diffmap}.png
    ├── fig3{a,b,c}_{mrnet,brats,cxr}_freq.png
    ├── fig4.png, fig4{a,b,c}_{mrnet,brats,cxr}_hallucination.png
    ├── fig5.png, fig6.{png,pdf}
    ├── fig_ae_ceiling_correlation.png
    └── fig_perception_distortion.png
```

---

## Hardware requirements

| Resource | Spec | Used for |
|----------|------|----------|
| GPU | NVIDIA L40S | Headline training runs (MedVAE SR, SD-VAE SR, KL-f4 SR — 100 epochs, cached latents, 0.4–3.0 h each) |
| GPU (alt) | NVIDIA A100 80 GB / H200 130 GB | Ablations, AE-ceiling sweeps, and other revision-cycle controls |
| CPU | 32 GB RAM | Analysis, statistics, figures |
| Storage | ~200 GB | Checkpoints + latent caches |
| Time | 18–24 h total wall-clock | Full pipeline |

Wall-clock varies 0.4–3.0 h across runs primarily due to node contention on a shared
cluster, not latent geometry; the per-step computational cost is asymmetric by
architectural construction (3×64×64 attends over 4× the spatial tokens of 4×32×32 per
denoising step), so the fidelity gain is obtained at a higher inference cost per step, not
for free. See the manuscript's "Compute resources" paragraph for the full statement.

---

## Model weights & embeddings (Hugging Face)

Trained diffusion-SR checkpoints and cached VAE latents are derived from the credentialed
source datasets (MRNet, BraTS, MIMIC-CXR — see *Dataset access* below); redistribution
requires holding the corresponding source-data credentials.

- **Diffusion checkpoints (live):**
  [`sebasmos/latent-sr-weights`](https://huggingface.co/sebasmos/latent-sr-weights) — 21
  trained diffusion UNet checkpoints (7 latent geometries × 3 datasets). Each checkpoint
  contains the diffusion UNet only; the VAE encoder/decoder is loaded separately at
  inference time from its own source (see below).
- **Cached VAE latents:** not yet uploaded.
- **Encoders used off-the-shelf, already public:**
  [`stanfordmimi/MedVAE`](https://huggingface.co/stanfordmimi/MedVAE) (all `medvae-*`
  geometries) and [`stabilityai/sd-vae-ft-ema`](https://huggingface.co/stabilityai/sd-vae-ft-ema)
  (SD-VAE, 4×32×32).
- **Naming convention** for released artifacts, every one is `<vae>_<dataset>`:

  | VAE id | Domain | Latent (C×H×W) | Budget | Role |
  |--------|--------|-----------------|--------|------|
  | `sdvae` | natural | 4×32×32 | 4,096 | generic baseline |
  | `kl-f4` | natural | 3×64×64 | 12,288 | natural VAE @ MedVAE geometry (control) |
  | `medvae-8-1` | medical | 1×32×32 | 1,024 | ceiling-regression point |
  | `medvae-8-4` | medical | 4×32×32 | 4,096 | = SD-VAE geometry (domain-only control) |
  | `medvae-4-1` | medical | 1×64×64 | 4,096 | capacity-matched to SD-VAE |
  | `medvae-4-3` | medical | 3×64×64 | 12,288 | paper method ("ours") |
  | `medvae-4-4` | medical | 4×64×64 | 16,384 | ceiling-regression point |

  `<dataset>` ∈ `{mrnet, brats, cxr}`.

---

## Quick start (pre-trained weights — reproduce figures only)

If you have access to the shared checkpoints and cached latents, point the repo at
that storage root first:

```bash
cd /path/to/latent-sr
export LATENT_SR_SHARED_ROOT=/path/to/shared-root
conda activate medvae-sr

# Run all CPU-only analysis + figure generation
bash reproduce_all.sh --skip-train --skip-eval

# Or generate figures only (all results already in outputs/)
bash reproduce_all.sh --figures-only
```

---

## Full reproduction from scratch

### §1 — Environment setup

```bash
conda env create -f environment.yml
conda activate medvae-sr
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### §1.1 — Path configuration

All core scripts resolve dataset, embedding, weight, and output roots from environment
variables (`scripts/repro_paths.py`) so the repo runs outside the original ORCD checkout.

```bash
# Required only if your data is not in the original ORCD layout.
export LATENT_SR_SHARED_ROOT=/path/to/shared-root

# Optional explicit overrides.
export LATENT_SR_DATA_ROOT="$LATENT_SR_SHARED_ROOT/DATASET"
export LATENT_SR_MRI_UGANDA_ROOT="$LATENT_SR_SHARED_ROOT/mri-uganda"
export LATENT_SR_EMBEDDINGS_ROOT="$LATENT_SR_MRI_UGANDA_ROOT/embeddings"
export LATENT_SR_WEIGHTS_ROOT="$LATENT_SR_MRI_UGANDA_ROOT/weights"
export LATENT_SR_OUTPUTS_ROOT="$PWD/outputs"
export LATENT_SR_CONDA_ENV=medvae-sr
```

### §2 — Dataset access

| Dataset | Source | Preprocessing |
|---------|--------|---------------|
| **MRNet** | [stanfordmlgroup.github.io/competitions/mrnet](https://stanfordmlgroup.github.io/competitions/mrnet/) | Extract axial slices → 256×256; generate 2× LR via `prep_mrnet_lr_64.py` |
| **BraTS 2023** | [synapse.org/#!Synapse:syn51156910](https://www.synapse.org/#!Synapse:syn51156910) | Extract FLAIR axial slices → 256×256; generate 4× LR via `prepare_brats_sr.py`; or run `sbatch slurm/prep_brats.sh` |
| **MIMIC-CXR** | [physionet.org/content/mimic-cxr-jpg](https://physionet.org/content/mimic-cxr-jpg/) (credentialed) | Frontal view → 256×256; generate 4× LR via `prepare_mimic_cxr_sr.py` |

### §3 — Extract VAE latents (skip if using shared cached latents)

```bash
python medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \
  --data-root "${LATENT_SR_DATA_ROOT}/mrnetkneemris/MRNet-v1.0-middle" \
  --latent-root "${LATENT_SR_EMBEDDINGS_ROOT}/medvae_4_3_2d_mrnet" \
  --model-name medvae_4_3_2d --modality mri --backend medvae \
  --splits train valid test --batch-size 8

# Other backends: --backend sdvae (no --model-name needed), --backend klf4 (natural KL-f4 control)
# Other geometries: --model-name medvae_8_4_2d | medvae_4_1_2d | medvae_8_1_2d | medvae_4_4_2d
```

### §4 — Train LDM UNet (skip if using pre-trained checkpoints)

Each model trains for 100 epochs on a single L40S GPU (~0.4–3.0 h, dominated by cluster
contention, see *Hardware requirements* above).

```bash
# Via SLURM (recommended)
sbatch slurm/run_brats_medvae.sh
sbatch slurm/run_cxr_medvae.sh
sbatch slurm/run_brats_sdvae.sh
sbatch slurm/run_cxr_sdvae.sh
# MRNet uses pre-trained weights — no training needed

# Training hyperparameters (identical across all VAE backends):
#   optimizer: Adam, lr=1e-4, batch_size=8, epochs=100
#   objective: x0-prediction, L1 loss
#   noise schedule: cosine beta, T=1000
```

### §5 — Evaluate SR quality (Table 1, self-referential convention)

```bash
# MedVAE SR — all 3 datasets
python scripts/eval_diffusion_sr.py \
  --checkpoint "${LATENT_SR_WEIGHTS_ROOT}/diffusion_medvae_mrnet_x0/checkpoints/last.ckpt" \
  --latent-dir "${LATENT_SR_EMBEDDINGS_ROOT}/medvae_4_3_2d_v2/phase2/valid_latent" \
  --backend medvae --medvae-model medvae_4_3_2d --modality mri \
  --timesteps 100 --batch-size 4 \
  --output-dir outputs/experiments/mrnet_medvae_s1

# AE ceiling (encode + decode only, no diffusion)
python scripts/eval_vae_reconstruction.py \
  --dataset mrnet --backend medvae \
  --output-dir outputs/experiments/mrnet_medvae_ae

# Baselines (bicubic + ESRGAN), SwinIR
python scripts/eval_baselines.py --dataset mrnet --output-dir outputs/experiments/mrnet_bicubic
python scripts/eval_swinir.py --dataset mrnet --output-dir outputs/experiments/mrnet_swinir
```

Results are written to `outputs/experiments/<name>/diffusion_eval_results.json`.

### §5.1 — True-HR evaluation + revision controls

```bash
# Score against the true HR image (not the VAE's own decoded HR) — the convention used
# throughout the revised manuscript.
sbatch slurm/revision/run_truehr_eval_brats_mrnet.sh brats medvae_4_3   # {brats|mrnet} {klf4|medvae_4_3|sdvae}
sbatch slurm/revision/run_truehr_eval_cxr.sh klf4                        # {klf4|medvae_4_3|sdvae}
sbatch slurm/revision/run_truehr_eval_capacity.sh medvae_8_4 brats       # {medvae_8_4|medvae_4_1} {brats|cxr|mrnet}

# Or call the script directly (see --help for the full flag list; requires --checkpoint,
# --latent-dir, --hr-image-dir, --backend):
python scripts/eval_diffusion_sr_truehr.py \
  --checkpoint "${LATENT_SR_WEIGHTS_ROOT}/diffusion_medvae_mrnet_s1_x0/checkpoints/last.ckpt" \
  --latent-dir "${LATENT_SR_EMBEDDINGS_ROOT}/medvae_mrnet_s1/valid_latent" \
  --hr-image-dir "${LATENT_SR_DATA_ROOT}/mrnetkneemris/MRNet-v1.0-middle/valid/hr" \
  --backend medvae --medvae-model medvae_4_3_2d \
  --output-dir outputs/experiments/truehr_mrnet_medvae

# Capacity-matched VAE controls: identical LDM pipeline, controlled latent geometry
sbatch slurm/revision/run_mrnet_medvae_4_1.sh   # 1x64x64, matched to SD-VAE's budget
sbatch slurm/revision/run_matched_vae.sh medvae_8_4_2d brats   # 4x32x32, SD-VAE's exact geometry

# KL-f4 natural-VAE control (AE ceiling + full SR pipeline, 3x64x64)
sbatch slurm/revision/run_klf4_ae.sh
sbatch slurm/revision/run_klf4_sr.sh mrnet

# Three-seed reproducibility sweep
sbatch slurm/revision/run_klf4_sr_seedsweep.sh mrnet 42
sbatch slurm/revision/run_klf4_sr_seedsweep.sh mrnet 43
sbatch slurm/revision/run_klf4_sr_seedsweep.sh mrnet 44

# Target-dataset AE adaptation (fine-tune KL-f4 on the target dataset's own training split)
sbatch slurm/revision/run_klf4med_finetune.sh brats
sbatch slurm/revision/run_klf4med_sr.sh brats

# Expanded n=15 AE-ceiling -> SR regression, single consistent (true-HR) metric
python scripts/recompute_regression_truehr.py

# Patient-level reanalysis (BraTS pseudoreplication check)
python scripts/patient_level_stats.py
```

### §6 — New analyses (Fig 3–5, Supp Figs 2–6)

```bash
# Wavelet frequency analysis (CPU-only, ~5 min per dataset)
python scripts/eval_frequency_analysis.py --dataset mrnet

# Hallucination quantification (regenerate figures + reproducibility-check aggregate metrics)
bash scripts/make_hallucination_figures.sh
python scripts/run_hallucination_stat_test.py

# Multi-resolution latent embedding (CPU-only — uses pre-saved .npy latents)
python scripts/eval_multiresolution_embedding.py \
  --dataset mrnet --sr-latent-dir outputs/experiments/mrnet_decoder_finetune/sr_latents

# SR vs HR difference maps
python scripts/eval_sr_hr_diffmaps.py --dataset mrnet
```

### §6.1 — Supplementary-table analyses

```bash
# Supp Table 4 — multi-scale (2×/8×) bicubic baselines (BraTS, CXR)
python scripts/gen_bicubic_multiscale.py

# Supp Table 10 — BraTS tumour-vs-background ROI PSNR/SSIM (needs seg masks)
python scripts/compute_roi_metrics.py

# Supp Table 12 — Fréchet Inception Distance across all methods
sbatch slurm/run_fid.sh          # or: python scripts/compute_fid.py

# R2.11 — re-encode CXR SR images through MedVAE, then re-run the embedding analysis
sbatch slurm/run_reencode_sr_latents_cxr.sh

# Provenance/sanity check — replicate the original MedVAE paper's Table 4 protocol
sbatch slurm/run_table4_replication.sh
```

### §7 — Ablation studies (Supp Tables 1–5)

```bash
sbatch slurm/run_step_ablation.sh                    # T=50,100,250,500,1000
sbatch slurm/run_flow_matching_mrnet.sh               # rectified flow, T=16 Euler steps
sbatch slurm/run_t16_mrnet.sh                         # T=16 DDPM inference
```

### §8 — Statistical analysis (Table 2, Supp Tables 5, 9)

```bash
python scripts/compute_effect_sizes.py    # Cohen's d + 95% bootstrap CI, metrics_summary.json
python scripts/compute_pvalues.py         # Wilcoxon signed-rank
python scripts/compute_bland_altman.py    # Bland-Altman intensity agreement (Supp Table 9)
```

### §9 — Generate all paper figures

```bash
python scripts/generate_paper1_fig1_truehr.py    # Fig 1: PSNR / LPIPS bar chart (true-HR)
python scripts/plot_ae_ceiling_correlation.py    # AE ceiling scatter
python scripts/generate_freq_figures.py          # Wavelet + FFT frequency analysis
python scripts/generate_visual_comparisons.py    # Visual comparison + SR-HR diffmaps
python scripts/generate_paper1_composite_figs.py # Multi-resolution latent embedding composite
python scripts/generate_paper1_fig6_truehr.py    # Bland-Altman + Cohen's d forest + violin plots
python scripts/plot_perception_distortion.py     # Supp Fig 1: perception-distortion tradeoff
bash scripts/collect_paper1_figures.sh           # Collect into figures-paper-1/
```

All figures are generated at **300 DPI** in `outputs/figures/`.

### §10 — Run the test suite

```bash
bash tests/run_all_tests.sh
# or individually:
python tests/test_revision_validation.py   # golden-number checks for the revision cycle
```

---

## Expected results

### Table 1 — Headline SR PSNR, true-HR convention (MedVAE SR vs. SD-VAE SR)

| Dataset | n | SD-VAE SR (dB) | **MedVAE SR (dB)** | Δ PSNR | 95% CI | Cohen's d | p-value |
|---------|---|-----------------|---------------------|--------|--------|-----------|---------|
| MRNet | 120 | 22.08 ± 1.57 | **24.00 ± 1.70** | +1.92 | [+1.51, +2.33] | 1.17 | 2.0×10⁻²¹ |
| BraTS | 700 | 23.40 ± 1.82 | **26.23 ± 2.05** | +2.83 | [+2.63, +3.03] | 1.46 | 2.9×10⁻¹¹⁶ |
| CXR | 1,000 | 25.13 ± 2.06 | **28.73 ± 2.41** | +3.60 | [+3.40, +3.80] | 1.61 | 3.3×10⁻¹⁶⁵ |

Plain bicubic interpolation exceeds MedVAE SR on PSNR and MS-SSIM on 2 of 3 datasets
(BraTS, CXR) — MedVAE's advantage is perceptual (LPIPS) and geometric, not a uniform
pixel-fidelity superiority. See the manuscript's Table 1 for the full baseline set
(bicubic, ESRGAN, SwinIR, SD-VAE SR, MedVAE SR, MedVAE AE ceiling).

### Table 2 — Domain vs. geometry decomposition (revision controls)

| Dataset | Domain (capacity-matched, upper bound) | Resolution | Capacity | Domain@64² (KL-f4) | Geometry gain (AE ceiling) |
|---------|------------------------------------------|------------|----------|----------------------|------------------------------|
| MRNet | +0.95 dB | +0.26 dB | +0.72 dB | +1.17 dB | +2.76 dB |
| BraTS | +0.32 dB | +1.85 dB | +0.66 dB | +0.74 dB | +5.74 dB |
| CXR | +0.89 dB | +0.56 dB | +2.16 dB | +0.94 dB | +3.97 dB |

### Table 3 — Three-tier ladder (Supplementary Note 5, target-dataset AE adaptation)

| Dataset | General pretraining (SR) | Target-dataset adaptation (SR) | Latent geometry (SR) |
|---------|---------------------------|----------------------------------|------------------------|
| MRNet | +0.01 dB (≈0) | +1.65 dB | +1.91 dB |
| BraTS | +0.14 dB (≈0) | +0.96 dB | +2.69 dB |
| CXR | −0.03 dB (≈0) | +1.00 dB | +3.63 dB |

Geometry dominates clearly on BraTS and CXR; on MRNet the margin over adaptation is only
0.26 dB, so this is a dataset-dependent ladder, not a general ordering.

### Key analysis results

| Analysis | Key number | Location |
|----------|-----------|----------|
| AE-ceiling → SR regression (n=15, true-HR, primary) | r = 0.83, R² = 0.69, p = 1.3×10⁻⁴, 95% CI [0.55, 0.94] | `outputs/statistical_tests/regression_truehr.json` |
| AE-ceiling → SR regression (n=6, Fig. 3 subset) | r = 0.87, R² = 0.76, p = 0.024 | `outputs/statistical_tests/ae_ceiling_correlation.json` |
| Three-seed sweep SD (seeds 42/43/44) | 0.08–0.18 dB (MRNet/BraTS/CXR) | `outputs/experiments/klf4_seedsweep_*` |
| Patient-level reanalysis (BraTS, n=35) | +2.905 dB, p=2.48×10⁻⁷, Cohen's dz=+10.27 | `outputs/statistical_tests/patient_level_stats.json` |
| HH₁ wavelet gain (MRNet/BraTS/CXR) | +1.18 / +1.41 / +0.70 dB | `outputs/experiments/frequency_analysis_*` |
| Hallucination rate (BraTS) | MedVAE 12.9% vs. SD-VAE 13.3% | `outputs/experiments/hallucination_brats/` |
| Hallucination rate (CXR) | MedVAE 3.3% vs. SD-VAE 3.4% | `outputs/experiments/hallucination_cxr/` |

---

## Output file structure

After running all stages, `outputs/` will contain:

```
outputs/
├── experiments/
│   ├── mrnet_medvae_s1/ mrnet_sdvae/ mrnet_medvae_ae/    ← self-referential convention
│   ├── truehr_mrnet_medvae/ truehr_brats_klf4/ ...        ← true-HR convention
│   ├── revision_capacity/medvae_8_4_*/ medvae_4_1_*/      ← capacity-matched controls
│   ├── klf4_ae/ klf4_sr_*/ klf4_seedsweep_*/               ← KL-f4 controls + seed sweep
│   ├── klf4med_*/                                          ← target-dataset AE adaptation
│   ├── frequency_analysis_{mrnet,brats,cxr}/
│   ├── hallucination_{mrnet,brats,cxr}/
│   └── multiresolution_embedding_{mrnet,brats,cxr}/
│
├── statistical_tests/
│   ├── metrics_summary.json         ← canonical Table 1 values
│   ├── effect_sizes.json            ← Cohen's d + 95% bootstrap CI
│   ├── wilcoxon_results.json
│   ├── bland_altman_stats.json
│   ├── regression_truehr.json       ← n=15 AE-ceiling -> SR regression
│   ├── patient_level_stats.json     ← BraTS pseudoreplication check
│   └── hallucination_stat_tests/
│
└── figures/
    ├── paper1_fig1.{png,pdf} / paper1_fig1_truehr.{png,pdf}
    ├── paper1_fig6.{png,pdf} / paper1_fig6_truehr.{png,pdf}
    └── perception_distortion_*.png
```

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{cajas2026latentgeometry,
  title   = {Latent Geometry Shapes Fidelity in Diffusion-Based Medical Image Super-Resolution},
  author  = {Sebastian Cajas and Ashaba Judith and Rahul Gorijavolu and Sahil Kapadia and
             Hillary Clinton Kasimbazi and Leo Kinyera and Emmanuel Paul Kwesiga and
             Sri Sri Jaithra Varma Manthena and Luis Filipe Nakayama and Ninsiima Doreen and
             Leo Anthony Celi},
  year    = {2026},
  eprint  = {2604.12152},
  archivePrefix = {arXiv},
  note    = {Submitted to Nature Scientific Reports (major revision)}
}
```

---

## Contact

- Code: [github.com/sebasmos/latent-sr](https://github.com/sebasmos/latent-sr)
- Paper: [arxiv.org/abs/2604.12152](https://arxiv.org/abs/2604.12152)
- Data: MRNet (Stanford), BraTS 2023 (Synapse), MIMIC-CXR (PhysioNet)
- GPU allocation: MIT ORCD (L40S / A100 80 GB / H200 130 GB)
