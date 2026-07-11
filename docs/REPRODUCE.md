# Reproduction guide

Full instructions to reproduce every table and figure in the paper from this repository.
All reported numbers are in the paper; this document only covers *how to regenerate them*.

- Paper: https://arxiv.org/abs/2604.12152
- Weights: https://huggingface.co/sebasmos/latent-sr-weights
- Embeddings: https://huggingface.co/datasets/sebasmos/latent-sr-embeddings

---

## Contents by paper artifact

| Paper artifact | Script(s) |
|----------------|-----------|
| Table 1 (SR quality) | `eval_diffusion_sr.py`, `eval_diffusion_sr_truehr.py`, `eval_baselines.py`, `eval_swinir.py`, `eval_vae_reconstruction.py` |
| Table 2 / effect sizes | `compute_effect_sizes.py`, `compute_pvalues.py` |
| Capacity-matched decomposition | `slurm/revision/run_matched_vae.sh`, `run_mrnet_medvae_4_1.sh`, `run_cxr_medvae_4_1.sh`, `run_brats_medvae_4_1_valid.sh` |
| KL-f4 natural-VAE control | `slurm/revision/run_klf4_ae.sh`, `run_klf4_sr.sh` |
| Three-seed sweep | `slurm/revision/run_klf4_sr_seedsweep.sh` |
| Target-dataset AE adaptation | `slurm/revision/run_klf4med_finetune.sh`, `run_klf4med_sr.sh` |
| True-HR evaluation | `eval_diffusion_sr_truehr.py`, `slurm/revision/run_truehr_eval_*.sh` |
| AE-ceiling → SR regression (n=15) | `recompute_regression_truehr.py` |
| Patient-level reanalysis | `patient_level_stats.py` |
| Supp Table 4 (multi-scale bicubic) | `gen_bicubic_multiscale.py` |
| Supp Table 6 (wavelet) | `eval_frequency_analysis.py` |
| Supp Table 7 (hallucination) | `eval_hallucination_quantification.py`, `run_hallucination_stat_test.py` |
| Supp Table 8 (embedding) | `eval_multiresolution_embedding.py` |
| Supp Table 9 (Bland-Altman) | `compute_bland_altman.py` |
| Supp Table 10 (BraTS ROI) | `compute_roi_metrics.py` |
| Supp Table 11 (AE ceiling) | `eval_vae_reconstruction.py` |
| Supp Table 12 (FID) | `compute_fid.py` |
| Supp Table 13 (multi-T embedding) | `eval_multit_embedding.py` |
| R2.11 (CXR re-encoding) | `reencode_sr_latents.py` |
| Provenance / MedVAE Table 4 | `replicate_table4.py` |

---

## Directory layout

```
latent-sr/
├── README.md                        ← short project blurb
├── docs/REPRODUCE.md                ← this file
├── reproduce_all.sh                 ← master end-to-end script (SLURM)
├── environment.yml / requirements.txt
├── configs/                         ← model / training configuration files
│
├── medvae_diffusion_pipeline/       ← core extraction + training package
│   ├── paper_validation_config.py / validation_framework.py
│   └── scripts/
│       ├── 02_extract_medvae_embeddings.py  ← cache VAE latents (medvae/sdvae/klf4 backends)
│       ├── 03_train_diffusion.py            ← LDM UNet training (x0-pred, L1, cosine beta)
│       └── 03_train_flow_matching.py        ← rectified-flow variant
│
├── scripts/                         ← evaluation, statistics, figure generation
│   ├── repro_paths.py               ← env-var path resolution (used by every script)
│   ├── Data prep: prep_mrnet_lr_64.py, prepare_brats_sr.py, prepare_mimic_cxr_sr.py, create_lr_images.py
│   ├── Evaluation: eval_diffusion_sr.py, eval_vae_reconstruction.py, eval_baselines.py,
│   │   eval_swinir.py, eval_multiscale_sr.py, gen_bicubic_multiscale.py, eval_frequency_analysis.py,
│   │   eval_hallucination_quantification.py, eval_multiresolution_embedding.py, compute_roi_metrics.py,
│   │   compute_fid.py, eval_sr_hr_diffmaps.py, eval_multit_embedding.py, reencode_sr_latents.py,
│   │   replicate_table4.py
│   ├── Evaluation (true-HR): eval_diffusion_sr_truehr.py, recompute_regression_truehr.py,
│   │   patient_level_stats.py, run_hallucination_stat_test.py, make_hallucination_figures.sh
│   ├── Statistics: compute_effect_sizes.py, compute_pvalues.py, compute_bland_altman.py
│   └── Figures: generate_paper1_fig1_truehr.py, plot_ae_ceiling_correlation.py, generate_freq_figures.py,
│       generate_visual_comparisons.py, generate_paper1_composite_figs.py, generate_paper1_fig6_truehr.py,
│       plot_perception_distortion.py, collect_paper1_figures.sh
│
├── slurm/                           ← SLURM job scripts (_env.sh is sourced by every job)
│   └── revision/                    ← the expanded controls (capacity-matched, KL-f4, seed sweep,
│                                       adaptation, true-HR eval)
│
├── tests/                           ← environment, installation, and golden-number validation
└── figures/                         ← pre-generated paper figures
```

---

## Hardware

| Resource | Spec | Used for |
|----------|------|----------|
| GPU | NVIDIA L40S | Headline training runs (MedVAE SR, SD-VAE SR, KL-f4 SR — 100 epochs, cached latents, 0.4–3.0 h each) |
| GPU (alt) | NVIDIA A100 80 GB / H200 130 GB | Ablations, AE-ceiling sweeps, other controls |
| CPU | 32 GB RAM | Analysis, statistics, figures |
| Storage | ~200 GB | Checkpoints + latent caches |

Wall-clock varies 0.4–3.0 h across runs primarily due to node contention on a shared
cluster, not latent geometry; the per-step cost is asymmetric by construction (3×64×64
attends over 4× the spatial tokens of 4×32×32 per denoising step).

---

## Weights, embeddings, and encoders (Hugging Face)

Derived artifacts are gated by the credentialing terms of the source datasets; redistribution
requires holding the corresponding source-data credentials.

- **Diffusion checkpoints:** [`sebasmos/latent-sr-weights`](https://huggingface.co/sebasmos/latent-sr-weights)
  — 21 trained diffusion UNet checkpoints (7 geometries × 3 datasets). UNet only; the VAE is
  loaded separately at inference.
- **Cached VAE latents:** [`sebasmos/latent-sr-embeddings`](https://huggingface.co/datasets/sebasmos/latent-sr-embeddings)
  — dataset repo with all 21 `<vae>_<dataset>` latent caches (HR/LR `.npy` per split).
- **Encoders (off-the-shelf):** [`stanfordmimi/MedVAE`](https://huggingface.co/stanfordmimi/MedVAE)
  (all `medvae-*`) and [`stabilityai/sd-vae-ft-ema`](https://huggingface.co/stabilityai/sd-vae-ft-ema) (SD-VAE).

Naming convention — every artifact is `<vae>_<dataset>`:

| VAE id | Domain | Latent (C×H×W) | Budget | Role |
|--------|--------|-----------------|--------|------|
| `sdvae` | natural | 4×32×32 | 4,096 | generic baseline |
| `kl-f4` | natural | 3×64×64 | 12,288 | natural VAE @ MedVAE geometry (control) |
| `medvae-8-1` | medical | 1×32×32 | 1,024 | ceiling-regression point |
| `medvae-8-4` | medical | 4×32×32 | 4,096 | = SD-VAE geometry (domain-only control) |
| `medvae-4-1` | medical | 1×64×64 | 4,096 | capacity-matched to SD-VAE |
| `medvae-4-3` | medical | 3×64×64 | 12,288 | main method |
| `medvae-4-4` | medical | 4×64×64 | 16,384 | ceiling-regression point |

`<dataset>` ∈ `{mrnet, brats, cxr}`.

---

## Quick start (pre-trained weights — figures only)

```bash
cd /path/to/latent-sr
export LATENT_SR_SHARED_ROOT=/path/to/shared-root
conda activate medvae-sr

bash reproduce_all.sh --skip-train --skip-eval   # CPU-only analysis + figures
bash reproduce_all.sh --figures-only             # figures only (results already in outputs/)
```

---

## Full reproduction from scratch

### 1. Environment

```bash
conda env create -f environment.yml
conda activate medvae-sr
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 1.1 Path configuration

All scripts resolve dataset/embedding/weight/output roots from environment variables
(`scripts/repro_paths.py`) so the repo runs outside the original checkout.

```bash
export LATENT_SR_SHARED_ROOT=/path/to/shared-root     # required if data is not in the original layout
# optional explicit overrides:
export LATENT_SR_DATA_ROOT="$LATENT_SR_SHARED_ROOT/DATASET"
export LATENT_SR_MRI_UGANDA_ROOT="$LATENT_SR_SHARED_ROOT/mri-uganda"
export LATENT_SR_EMBEDDINGS_ROOT="$LATENT_SR_MRI_UGANDA_ROOT/embeddings"
export LATENT_SR_WEIGHTS_ROOT="$LATENT_SR_MRI_UGANDA_ROOT/weights"
export LATENT_SR_OUTPUTS_ROOT="$PWD/outputs"
export LATENT_SR_CONDA_ENV=medvae-sr
```

### 2. Dataset access

| Dataset | Source | Preprocessing |
|---------|--------|---------------|
| MRNet | [stanfordmlgroup.github.io/competitions/mrnet](https://stanfordmlgroup.github.io/competitions/mrnet/) | Axial slices → 256×256; 2× LR via `prep_mrnet_lr_64.py` |
| BraTS 2023 | [synapse.org/#!Synapse:syn51156910](https://www.synapse.org/#!Synapse:syn51156910) | FLAIR axial slices → 256×256; 4× LR via `prepare_brats_sr.py` (or `sbatch slurm/prep_brats.sh`) |
| MIMIC-CXR | [physionet.org/content/mimic-cxr-jpg](https://physionet.org/content/mimic-cxr-jpg/) (credentialed) | Frontal view → 256×256; 4× LR via `prepare_mimic_cxr_sr.py` |

### 3. Extract VAE latents (skip if using the cached latents from HF)

```bash
python medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \
  --data-root "${LATENT_SR_DATA_ROOT}/mrnetkneemris/MRNet-v1.0-middle" \
  --latent-root "${LATENT_SR_EMBEDDINGS_ROOT}/medvae_4_3_2d_mrnet" \
  --model-name medvae_4_3_2d --modality mri --backend medvae \
  --splits train valid test --batch-size 8
# Other backends: --backend sdvae (no --model-name), --backend klf4
# Other geometries: --model-name medvae_8_4_2d | medvae_4_1_2d | medvae_8_1_2d | medvae_4_4_2d
```

### 4. Train the LDM UNet (skip if using the pre-trained checkpoints from HF)

100 epochs per model on a single L40S GPU (~0.4–3.0 h).

```bash
sbatch slurm/run_brats_medvae.sh
sbatch slurm/run_cxr_medvae.sh
sbatch slurm/run_brats_sdvae.sh
sbatch slurm/run_cxr_sdvae.sh
# MRNet uses pre-trained weights — no training needed
# Hyperparameters (identical across backends): Adam lr=1e-4, batch=8, 100 epochs,
#   x0-prediction, L1 loss, cosine beta, T=1000
```

### 5. Evaluate SR quality (self-referential convention)

```bash
python scripts/eval_diffusion_sr.py \
  --checkpoint "${LATENT_SR_WEIGHTS_ROOT}/diffusion_medvae_mrnet_x0/checkpoints/last.ckpt" \
  --latent-dir "${LATENT_SR_EMBEDDINGS_ROOT}/medvae_4_3_2d_v2/phase2/valid_latent" \
  --backend medvae --medvae-model medvae_4_3_2d --modality mri \
  --timesteps 100 --batch-size 4 --output-dir outputs/experiments/mrnet_medvae_s1

python scripts/eval_vae_reconstruction.py --dataset mrnet --backend medvae \
  --output-dir outputs/experiments/mrnet_medvae_ae
python scripts/eval_baselines.py --dataset mrnet --output-dir outputs/experiments/mrnet_bicubic
python scripts/eval_swinir.py    --dataset mrnet --output-dir outputs/experiments/mrnet_swinir
```

### 5.1 True-HR evaluation + expanded controls

```bash
# Score against the true HR image (not the VAE's own decoded HR).
sbatch slurm/revision/run_truehr_eval_brats_mrnet.sh brats medvae_4_3   # {brats|mrnet} {klf4|medvae_4_3|sdvae}
sbatch slurm/revision/run_truehr_eval_cxr.sh klf4                        # {klf4|medvae_4_3|sdvae}
sbatch slurm/revision/run_truehr_eval_capacity.sh medvae_8_4 brats       # {medvae_8_4|medvae_4_1} {brats|cxr|mrnet}

# Capacity-matched VAE controls
sbatch slurm/revision/run_mrnet_medvae_4_1.sh                 # 1x64x64, matched to SD-VAE's budget
sbatch slurm/revision/run_matched_vae.sh medvae_8_4_2d brats  # 4x32x32, SD-VAE's exact geometry

# KL-f4 natural-VAE control
sbatch slurm/revision/run_klf4_ae.sh
sbatch slurm/revision/run_klf4_sr.sh mrnet

# Three-seed sweep
sbatch slurm/revision/run_klf4_sr_seedsweep.sh mrnet 42
sbatch slurm/revision/run_klf4_sr_seedsweep.sh mrnet 43
sbatch slurm/revision/run_klf4_sr_seedsweep.sh mrnet 44

# Target-dataset AE adaptation
sbatch slurm/revision/run_klf4med_finetune.sh brats
sbatch slurm/revision/run_klf4med_sr.sh brats

# Expanded n=15 AE-ceiling → SR regression, and patient-level reanalysis
python scripts/recompute_regression_truehr.py
python scripts/patient_level_stats.py
```

### 6. Additional analyses

```bash
python scripts/eval_frequency_analysis.py --dataset mrnet          # wavelet + FFT (Supp Table 6)
bash   scripts/make_hallucination_figures.sh                        # hallucination figs (Supp Table 7)
python scripts/run_hallucination_stat_test.py
python scripts/eval_multiresolution_embedding.py --dataset mrnet \
  --sr-latent-dir outputs/experiments/mrnet_decoder_finetune/sr_latents  # (Supp Table 8)
python scripts/eval_sr_hr_diffmaps.py --dataset mrnet

# Supplementary-table analyses
python scripts/gen_bicubic_multiscale.py                            # Supp Table 4
python scripts/compute_roi_metrics.py                              # Supp Table 10 (needs seg masks)
sbatch slurm/run_fid.sh                                             # Supp Table 12 (or run compute_fid.py)
sbatch slurm/run_reencode_sr_latents_cxr.sh                        # R2.11 CXR re-encoding
sbatch slurm/run_table4_replication.sh                            # MedVAE Table 4 provenance check
```

### 7. Ablation studies

```bash
sbatch slurm/run_step_ablation.sh          # T=50,100,250,500,1000
sbatch slurm/run_flow_matching_mrnet.sh    # rectified flow, T=16 Euler steps
sbatch slurm/run_t16_mrnet.sh              # T=16 DDPM inference
```

### 8. Statistical analysis

```bash
python scripts/compute_effect_sizes.py     # Cohen's d + 95% bootstrap CI
python scripts/compute_pvalues.py          # Wilcoxon signed-rank
python scripts/compute_bland_altman.py     # Bland-Altman intensity agreement (Supp Table 9)
```

### 9. Generate figures

```bash
python scripts/generate_paper1_fig1_truehr.py    # PSNR / LPIPS bar chart (true-HR)
python scripts/plot_ae_ceiling_correlation.py    # AE ceiling scatter
python scripts/generate_freq_figures.py          # wavelet + FFT
python scripts/generate_visual_comparisons.py    # visual comparison + diffmaps
python scripts/generate_paper1_composite_figs.py # multi-resolution embedding composite
python scripts/generate_paper1_fig6_truehr.py    # Bland-Altman + forest + violin
python scripts/plot_perception_distortion.py     # perception-distortion tradeoff
bash   scripts/collect_paper1_figures.sh
```

Figures are written at 300 DPI to `outputs/figures/`.

### 10. Test suite

```bash
bash tests/run_all_tests.sh
python tests/test_revision_validation.py   # golden-number consistency checks vs. the paper
```

---

## Output layout

```
outputs/
├── experiments/          ← per-run metric JSONs + SR images (self-referential, true-HR, controls)
├── statistical_tests/    ← effect sizes, Wilcoxon, Bland-Altman, regression, patient-level
└── figures/              ← generated figures
```
