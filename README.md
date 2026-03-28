# Paper 1 — Reproducibility Package

**"Domain-Specific Latent Representations Improve Diffusion-Based Medical Image Super-Resolution"**

> **Core finding:** Replacing SD-VAE with MedVAE in an otherwise identical LDM pipeline yields +2.91–3.29 dB PSNR across three medical imaging modalities (Cohen's d = 1.37–1.86; all p < 10⁻²⁰, Wilcoxon signed-rank).

---

## Directory layout

```
latent-sr/
├── README.md                   ← this file
├── reproduce_all.sh            ← master end-to-end script (SLURM)
├── environment.yml             ← conda environment
├── requirements.txt            ← pip packages
├── configs/                    ← model / training configuration files
│
├── scripts/                    ← all Python scripts (eval + figures + stats)
│   ├── Data prep
│   │   ├── prep_mrnet_lr_64.py
│   │   ├── prepare_brats_sr.py
│   │   ├── prepare_mimic_cxr_sr.py
│   │   └── create_lr_images.py
│   │
│   ├── Training
│   │   ├── extract_embeddings.py    ← cache VAE latents before training
│   │   └── train_ldm.py             ← LDM UNet training
│   │
│   ├── Evaluation
│   │   ├── eval_diffusion_sr.py     ← main SR: PSNR / MS-SSIM / LPIPS
│   │   ├── eval_vae_reconstruction.py   ← AE ceiling (encode+decode only)
│   │   ├── eval_baselines.py        ← bicubic + ESRGAN
│   │   ├── eval_bicubic_baseline.py
│   │   ├── eval_swinir.py           ← SwinIR baseline
│   │   ├── eval_multiscale_sr.py    ← multi-scale bicubic (2×/4×/8×)
│   │   ├── eval_frequency_analysis.py   ← Haar wavelet + FFT power spectrum
│   │   ├── eval_hallucination_quantification.py  ← pixel-level decomposition
│   │   ├── eval_multiresolution_embedding.py     ← SR-vs-HR latent cosine sim
│   │   ├── eval_sr_hr_diffmaps.py   ← signed difference maps
│   │   └── eval_multit_embedding.py ← multi-T cosine sim (Supp Table 13)
│   │
│   ├── Statistical analysis
│   │   ├── compute_effect_sizes.py  ← Cohen's d + bootstrap CI
│   │   ├── compute_pvalues.py       ← Wilcoxon signed-rank
│   │   └── compute_bland_altman.py  ← Bland-Altman intensity agreement
│   │
│   └── Figure generation
│       ├── generate_paper1_fig1.py       ← Fig 1: PSNR/LPIPS bar chart
│       ├── plot_ae_ceiling_correlation.py ← Fig 2: AE ceiling scatter (r=0.82)
│       ├── generate_freq_figures.py      ← Fig 3: wavelet + FFT panels
│       ├── generate_visual_comparisons.py ← Fig 4: visual + diffmap panels
│       ├── generate_paper1_composite_figs.py ← Fig 5: embedding composite
│       ├── generate_paper1_fig6.py       ← Fig 6: stats (BA + forest + violin)
│       ├── plot_perception_distortion.py ← Supp Fig 1: perception-distortion
│       └── collect_paper1_figures.sh     ← copies all figures to figures-paper-1/
│
└── slurm/                      ← SLURM job scripts
│   ├── prep_brats.sh
│   ├── run_brats_medvae.sh     ← train MedVAE SR on BraTS
│   ├── run_cxr_medvae.sh       ← train MedVAE SR on CXR
│   ├── run_mrnet_eval.sh       ← eval MedVAE SR on MRNet (pre-trained ckpt)
│   ├── run_brats_sdvae.sh      ← train SD-VAE SR on BraTS
│   ├── run_cxr_sdvae.sh        ← train SD-VAE SR on CXR
│   ├── run_baselines.sh
│   ├── run_swinir.sh
│   ├── run_frequency_analysis.sh
│   ├── run_hallucination.sh
│   ├── run_multiresolution_embedding.sh
│   ├── run_sr_hr_diffmaps.sh
│   ├── run_step_ablation.sh
│   ├── run_t16_{mrnet,brats,cxr}.sh
│   ├── run_flow_matching_{mrnet,brats,cxr}.sh
│   └── reeval_for_pvalues.sh
│
└── figures/                    ← pre-generated paper figures (upload to Overleaf)
    ├── fig_pipeline.{svg,png,pdf}
    ├── fig1.{png,pdf}                ← Fig 1 bar chart
    ├── fig2a_mrnet_visual.png        ← Fig 2 visual comparison (MRNet)
    ├── fig2a_mrnet_diffmap.png       ← Fig 2 difference map (MRNet)
    ├── fig2b_brats_visual.png        ← Fig 2 visual comparison (BraTS)
    ├── fig2b_brats_diffmap.png       ← Fig 2 difference map (BraTS)
    ├── fig2c_cxr_visual.png          ← Fig 2 visual comparison (CXR)
    ├── fig2c_cxr_diffmap.png         ← Fig 2 difference map (CXR)
    ├── fig3a_mrnet_freq.png          ← Fig 3 wavelet + FFT (MRNet)
    ├── fig3b_brats_freq.png          ← Fig 3 wavelet + FFT (BraTS)
    ├── fig3c_cxr_freq.png            ← Fig 3 wavelet + FFT (CXR)
    ├── fig4.png                      ← Fig 4 composite
    ├── fig4a_mrnet_hallucination.png ← Fig 4 hallucination (MRNet)
    ├── fig4b_brats_hallucination.png ← Fig 4 hallucination (BraTS)
    ├── fig4c_cxr_hallucination.png   ← Fig 4 hallucination (CXR)
    ├── fig5.png                      ← Fig 5 embedding comparison
    ├── fig6.{png,pdf}                ← Fig 6 statistical analysis
    ├── fig_ae_ceiling_correlation.png
    └── fig_perception_distortion.png
```

---

## Hardware requirements

| Resource | Spec | Used for |
|----------|------|----------|
| GPU | NVIDIA A100 80 GB | Training + SR inference |
| GPU (alt) | NVIDIA H200 130 GB | Fast inference (T=16 jobs) |
| CPU | 32 GB RAM | Analysis, statistics, figures |
| Storage | ~200 GB | Checkpoints + latent caches |
| Time | 18–24 h total wall-clock | Full pipeline |

---

## Quick start (pre-trained weights — reproduce figures only)

If you have access to the pre-trained checkpoints at `/orcd/pool/006/lceli_shared/`:

```bash
cd /orcd/pool/005/sebasmos/code/latent-sr
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
export PYTHONPATH="/orcd/pool/005/sebasmos/code/latent-sr:$PYTHONPATH"
```

### §2 — Dataset access

| Dataset | Source | Preprocessing |
|---------|--------|---------------|
| **MRNet** | [stanfordmlgroup.github.io/competitions/mrnet](https://stanfordmlgroup.github.io/competitions/mrnet/) | Extract axial slices → 256×256; generate 2× LR via `prep_mrnet_lr_64.py` |
| **BraTS 2023** | [synapse.org/#!Synapse:syn51156910](https://www.synapse.org/#!Synapse:syn51156910) | Extract FLAIR axial slices → 256×256; generate 4× LR via `prepare_brats_sr.py`; or run `sbatch slurm/prep_brats.sh` |
| **MIMIC-CXR** | [physionet.org/content/mimic-cxr-jpg](https://physionet.org/content/mimic-cxr-jpg/) (credentialed) | Frontal view → 256×256; generate 4× LR via `prepare_mimic_cxr_sr.py` |

**Shared pre-processed data location (ORCD):**
```
/orcd/pool/006/lceli_shared/mri-uganda/
  embeddings/medvae_4_3_2d_v2/         ← MedVAE latents (cached)
  embeddings/sdvae/                     ← SD-VAE latents (cached)
  weights/diffusion_medvae_mrnet_x0/    ← MedVAE SR checkpoint (MRNet)
  weights/diffusion_medvae_brats_s1/    ← MedVAE SR checkpoint (BraTS)
  weights/diffusion_medvae_cxr_s1/      ← MedVAE SR checkpoint (CXR)
  weights/diffusion_sdvae_mrnet_x0/     ← SD-VAE SR checkpoint (MRNet)
  weights/diffusion_sdvae_brats_s1/     ← SD-VAE SR checkpoint (BraTS)
  weights/diffusion_sdvae_cxr_s1/       ← SD-VAE SR checkpoint (CXR)
```

### §3 — Extract VAE latents (skip if using shared cached latents)

MedVAE and SD-VAE latents are cached in the shared directory above. To re-extract:

```bash
python scripts/extract_embeddings.py \
  --dataset mrnet \
  --vae medvae_4_3_2d \
  --output-dir outputs/embeddings/medvae_mrnet/
```

### §4 — Train LDM UNet (skip if using pre-trained checkpoints)

Each model trains for 100 epochs on a single A100 80 GB (~6–8 h).

```bash
# Via SLURM (recommended)
sbatch slurm/run_brats_medvae.sh
sbatch slurm/run_cxr_medvae.sh
sbatch slurm/run_brats_sdvae.sh
sbatch slurm/run_cxr_sdvae.sh
# MRNet uses pre-trained weights — no training needed

# Training hyperparameters (same for both VAEs):
#   optimizer: Adam, lr=1e-4, batch_size=8, epochs=100
#   objective: x0-prediction, L1 loss
#   noise schedule: cosine beta, T=1000
#   hardware: 1× A100 80 GB
```

### §5 — Evaluate SR quality (Table 1)

```bash
# MedVAE SR — all 3 datasets
python scripts/eval_diffusion_sr.py \
  --checkpoint /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_medvae_mrnet_x0/checkpoints/last.ckpt \
  --latent-dir /orcd/pool/006/lceli_shared/mri-uganda/embeddings/medvae_4_3_2d_v2/phase2/valid_latent \
  --backend medvae --medvae-model medvae_4_3_2d --modality mri \
  --timesteps 100 --batch-size 4 \
  --output-dir outputs/experiments/mrnet_medvae_s1

# SD-VAE SR — BraTS example
python scripts/eval_diffusion_sr.py \
  --checkpoint /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_sdvae_brats_s1/checkpoints/last.ckpt \
  --latent-dir /orcd/pool/006/lceli_shared/mri-uganda/embeddings/sdvae/brats/valid_latent \
  --backend sdvae --modality mri \
  --timesteps 100 --batch-size 4 \
  --output-dir outputs/experiments/brats_sdvae_valid

# AE ceiling (encode + decode only, no diffusion)
python scripts/eval_vae_reconstruction.py \
  --dataset mrnet --backend medvae \
  --output-dir outputs/experiments/mrnet_medvae_ae

# Baselines (bicubic + ESRGAN)
python scripts/eval_baselines.py \
  --dataset mrnet --output-dir outputs/experiments/mrnet_bicubic

# SwinIR
python scripts/eval_swinir.py \
  --dataset mrnet --output-dir outputs/experiments/mrnet_swinir
```

Results are written to `outputs/experiments/<name>/diffusion_eval_results.json`.

### §6 — New analyses

#### Wavelet frequency analysis (Fig 3, Supp Table 6)
```bash
# CPU-only, ~5 min per dataset
python scripts/eval_frequency_analysis.py \
  --sr-dir outputs/experiments/mrnet_medvae_s1/sr_images \
  --sr-dir-baseline outputs/experiments/mrnet_sdvae/sr_images \
  --hr-dir /path/to/mrnet/hr_images \
  --output-dir outputs/experiments/frequency_analysis_mrnet
```

#### Hallucination quantification (Fig 4, Supp Figs 2–4, Supp Table 7)
```bash
python scripts/eval_hallucination_quantification.py \
  --sr-medvae outputs/experiments/mrnet_medvae_s1/sr_images \
  --sr-sdvae  outputs/experiments/mrnet_sdvae/sr_images \
  --ae-dir    outputs/experiments/mrnet_medvae_ae/ae_images \
  --hr-dir    /path/to/mrnet/hr_images \
  --output-dir outputs/experiments/hallucination_mrnet
```

#### Multi-resolution latent embedding (Fig 5, Supp Table 8)
```bash
# CPU-only — uses pre-saved .npy latents
python scripts/eval_multiresolution_embedding.py \
  --sr-latent-dir  outputs/experiments/mrnet_decoder_finetune/sr_latents \
  --hr-latent-dir  /orcd/pool/006/lceli_shared/mri-uganda/embeddings/medvae_4_3_2d_v2/phase2/valid_latent \
  --output-dir     outputs/experiments/multiresolution_embedding_mrnet
```

#### SR vs HR difference maps (Supp Figs 5–6)
```bash
python scripts/eval_sr_hr_diffmaps.py \
  --sr-dir outputs/experiments/mrnet_medvae_s1/sr_images \
  --hr-dir /path/to/mrnet/hr_images \
  --output-dir outputs/experiments/sr_hr_diffmaps_mrnet
```

### §7 — Ablation studies (Supp Tables 1–5)

```bash
# Step ablation T=50,100,250,500,1000
sbatch slurm/run_step_ablation.sh

# Flow matching (rectified flow, T=16 Euler steps)
sbatch slurm/run_flow_matching_mrnet.sh
sbatch slurm/run_flow_matching_brats.sh
sbatch slurm/run_flow_matching_cxr.sh

# T=16 DDPM inference (Q2 reviewer response)
sbatch slurm/run_t16_mrnet.sh
sbatch slurm/run_t16_brats.sh
sbatch slurm/run_t16_cxr.sh
```

### §8 — Statistical analysis (Table 2, Supp Tables 5, 9)

```bash
# Cohen's d + 95% bootstrap CI (n=10,000)
python scripts/compute_effect_sizes.py
# → outputs/statistical_tests/effect_sizes.json

# Wilcoxon signed-rank p-values
python scripts/compute_pvalues.py
# → outputs/statistical_tests/wilcoxon_results.json

# Bland-Altman intensity agreement (Supp Table 9)
python scripts/compute_bland_altman.py
# → outputs/statistical_tests/bland_altman_stats.json
```

### §9 — Generate all paper figures

```bash
cd /orcd/pool/005/sebasmos/code/latent-sr
conda activate medvae-sr

# Fig 1: PSNR / LPIPS bar chart (reads outputs/experiments/*/diffusion_eval_results.json)
python scripts/generate_paper1_fig1.py

# Fig 2: AE ceiling scatter (r=0.82 across 6 method-dataset pairs)
python scripts/plot_ae_ceiling_correlation.py

# Fig 3: Wavelet + FFT frequency analysis (3-panel, one per dataset)
python scripts/generate_freq_figures.py

# Fig 4: Visual comparison + SR-HR diffmaps
python scripts/generate_visual_comparisons.py

# Fig 5: Multi-resolution latent embedding composite
python scripts/generate_paper1_composite_figs.py

# Fig 6: Bland-Altman + Cohen's d forest + violin plots
python scripts/generate_paper1_fig6.py

# Supp Fig 1: Perception-distortion tradeoff
python scripts/plot_perception_distortion.py

# Collect all into figures-paper-1/ (→ upload to Overleaf as figures/)
bash scripts/collect_paper1_figures.sh
```

All figures are generated at **300 DPI** in `outputs/figures/` and collected into `figures-paper-1/` and `paper1/figures/`.

---

## Expected results

### Table 1 — SR reconstruction quality

| Dataset | Bicubic | ESRGAN | SwinIR | SD-VAE SR | **MedVAE SR** | MedVAE AE |
|---------|---------|--------|--------|-----------|---------------|-----------|
| MRNet PSNR | 23.79 | 23.28 | 22.48 | 22.34 | **25.26** | 27.85 |
| BraTS PSNR | 29.91 | 27.45 | 28.38 | 23.51 | **26.42** | 38.42 |
| CXR PSNR | 30.47 | 27.71 | 29.21 | 25.58 | **28.87** | 36.93 |
| MRNet LPIPS | 0.541 | 0.425 | 0.446 | 0.173 | **0.135** | 0.109 |
| BraTS LPIPS | 0.218 | 0.098 | 0.077 | 0.097 | **0.088** | 0.014 |
| CXR LPIPS | 0.330 | 0.175 | 0.138 | 0.167 | **0.127** | 0.029 |

### Table 2 — Statistical significance

| Dataset | n | ΔPSNR | 95% CI | Cohen's d | p-value |
|---------|---|--------|--------|-----------|---------|
| MRNet | 120 | +2.91 dB | [+2.52, +3.30] | 1.858 | 9.9×10⁻²² |
| BraTS | 700 | +2.91 dB | [+2.69, +3.12] | 1.421 | 7.8×10⁻¹²⁰ |
| CXR | 1,000 | +3.29 dB | [+3.09, +3.51] | 1.370 | 2.3×10⁻¹¹⁷ |

### Key analysis results

| Analysis | Key number | Location |
|----------|-----------|----------|
| AE ceiling predictor | r = 0.82, R² = 0.67 | outputs/statistical_tests/ae_ceiling_correlation.json |
| HH₁ wavelet gain (MRNet) | +1.18 dB | outputs/experiments/frequency_analysis_mrnet/ |
| HH₁ wavelet gain (BraTS) | +1.41 dB | outputs/experiments/frequency_analysis_brats/ |
| HH₁ wavelet gain (CXR) | +0.70 dB | outputs/experiments/frequency_analysis_cxr/ |
| LL₃ wavelet diff | ≤ 0.05 dB (all) | same |
| Hallucination rate (BraTS) | MedVAE 12.9% vs SD-VAE 13.3% | outputs/experiments/hallucination_brats/ |
| Hallucination rate (CXR) | MedVAE 3.3% vs SD-VAE 3.4% | outputs/experiments/hallucination_cxr/ |
| Cosine sim @64×64 (MRNet) | 0.69–0.90 | outputs/experiments/multiresolution_embedding_mrnet/ |
| T=16 LPIPS (MRNet) | 0.218 vs T=100 0.135 | outputs/experiments/mrnet_medvae_t16/ |
| T=16 LPIPS (BraTS) | 0.086 vs T=100 0.088 | outputs/experiments/brats_medvae_t16/ |
| T=16 LPIPS (CXR) | 0.301 vs T=100 0.127 | outputs/experiments/cxr_medvae_t16/ |

---

## Output file structure

After running all stages, `outputs/` will contain:

```
outputs/
├── experiments/
│   ├── mrnet_medvae_s1/              ← MedVAE SR on MRNet
│   │   ├── diffusion_eval_results.json
│   │   └── sr_images/
│   ├── mrnet_sdvae/                  ← SD-VAE SR on MRNet
│   ├── mrnet_medvae_ae/              ← MedVAE AE ceiling
│   ├── mrnet_bicubic/ mrnet_swinir/ mrnet_realesrgan/
│   ├── brats_medvae_s1_valid/
│   ├── brats_sdvae_valid/
│   ├── brats_medvae_ae_valid/
│   ├── cxr_medvae_s1/
│   ├── cxr_sdvae/
│   ├── cxr_medvae_ae/
│   ├── frequency_analysis_{mrnet,brats,cxr}/
│   ├── hallucination_{mrnet,brats,cxr}/
│   ├── multiresolution_embedding_{mrnet,brats,cxr}/
│   ├── sr_hr_diffmaps_{mrnet,brats,cxr}/
│   └── mrnet_medvae_t16/ brats_medvae_t16/ cxr_medvae_t16/
│
├── statistical_tests/
│   ├── effect_sizes.json
│   ├── wilcoxon_results.json
│   └── bland_altman_stats.json
│
└── figures/
    ├── paper1_fig1.{png,pdf}
    ├── paper1_fig6.{png,pdf}
    ├── perception_distortion_*.png
    └── ...
```

---

## Citation

If you use this code or results, please cite:

```bibtex
@software{cajas2026latentSR,
  title   = {Domain-Specific Latent Representations Improve Diffusion-Based Medical Image Super-Resolution},
  author  = {Sebastian Cajas and Hillary Clinton Kasimbazi and Leo Anthony Celi},
  year    = {2026}
}
```

---

## Contact

- Code: [github.com/sebasmos/latent-sr](https://github.com/sebasmos/latent-sr)
- Data: MRNet (Stanford), BraTS 2023 (Synapse), MIMIC-CXR (PhysioNet)
- GPU allocation: MIT ORCD (A100 80 GB / H200 130 GB)
