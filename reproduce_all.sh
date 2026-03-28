#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh — Full pipeline to reproduce Paper 1
#
# Paper: "Domain-Specific Latent Representations Improve Diffusion-Based
#         Medical Image Super-Resolution"
# Venue: Nature (submitted 2026)
#
# Runtime:  ~18–24 h on A100 80 GB (SLURM cluster)
# Hardware: Single NVIDIA A100 80 GB per job; CPU for analysis/figures
#
# Usage:
#   bash reproduce_all.sh [--skip-train] [--skip-eval] [--figures-only]
#
# Each stage submits SLURM jobs and waits for them. Set SLURM=0 to run
# locally (slow — use only for testing individual stages).
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
SCRIPTS="$ROOT/scripts"
SLURM_DIR="$ROOT/slurm"
OUTPUTS="$REPO/outputs"
FIGURES="$ROOT/figures"

SKIP_TRAIN=0
SKIP_EVAL=0
FIGURES_ONLY=0

for arg in "$@"; do
  case $arg in
    --skip-train)   SKIP_TRAIN=1 ;;
    --skip-eval)    SKIP_EVAL=1  ;;
    --figures-only) FIGURES_ONLY=1; SKIP_TRAIN=1; SKIP_EVAL=1 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# STAGE 0: Environment check
# ---------------------------------------------------------------------------
log "=== STAGE 0: Environment ==="
if ! conda info --envs | grep -q medvae-sr; then
  log "ERROR: conda env 'medvae-sr' not found. Run:"
  log "  conda env create -f $ROOT/environment.yml"
  exit 1
fi
log "Environment OK: medvae-sr"

# ---------------------------------------------------------------------------
# STAGE 1: Data preparation
# ---------------------------------------------------------------------------
log "=== STAGE 1: Data Preparation ==="
if [ "$SKIP_TRAIN" -eq 0 ] && [ "$FIGURES_ONLY" -eq 0 ]; then
  log "MRNet: expects standard MRNet split at /orcd/pool/006/lceli_shared/mri-uganda/"
  log "BraTS: run prep_brats.sh to extract FLAIR slices"
  log "CXR:   expects MIMIC-CXR-JPG pre-processed at /orcd/pool/006/lceli_shared/"
  log ""
  log "  sbatch $SLURM_DIR/prep_brats.sh"
  log "  (MRNet and CXR data prep is preprocessing-free — see README §2)"
fi

# ---------------------------------------------------------------------------
# STAGE 2: Extract MedVAE embeddings (cache latents for fast training)
# ---------------------------------------------------------------------------
log "=== STAGE 2: Extract Embeddings (cached at shared path — skip if present) ==="
EMBED_PATH="/orcd/pool/006/lceli_shared/mri-uganda/embeddings"
if [ -d "$EMBED_PATH/medvae_4_3_2d_v2" ]; then
  log "  Embeddings already cached at $EMBED_PATH — skipping extraction"
else
  log "  Run: python $SCRIPTS/extract_embeddings.py  (see README §3)"
fi

# ---------------------------------------------------------------------------
# STAGE 3: Train MedVAE SR and SD-VAE SR (main models)
# ---------------------------------------------------------------------------
log "=== STAGE 3: Training (6–8 h per model on A100) ==="
if [ "$SKIP_TRAIN" -eq 0 ]; then
  log "Submitting MedVAE SR training jobs..."
  sbatch "$SLURM_DIR/run_brats_medvae.sh"
  sbatch "$SLURM_DIR/run_cxr_medvae.sh"
  sbatch "$SLURM_DIR/run_mrnet_eval.sh"   # MRNet uses pre-trained weights
  log ""
  log "Submitting SD-VAE SR training jobs..."
  sbatch "$SLURM_DIR/run_brats_sdvae.sh"
  sbatch "$SLURM_DIR/run_cxr_sdvae.sh"
  log ""
  log "Wait for all jobs to complete (use: squeue -u \$USER), then continue."
  log "Pre-trained checkpoint paths:"
  log "  MedVAE MRNet: /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_medvae_mrnet_x0/checkpoints/last.ckpt"
  log "  MedVAE BraTS: /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_medvae_brats_s1/checkpoints/last.ckpt"
  log "  MedVAE CXR:   /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_medvae_cxr_s1/checkpoints/last.ckpt"
  log "  SD-VAE MRNet: /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_sdvae_mrnet_x0/checkpoints/last.ckpt"
  log "  SD-VAE BraTS: /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_sdvae_brats_s1/checkpoints/last.ckpt"
  log "  SD-VAE CXR:   /orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_sdvae_cxr_s1/checkpoints/last.ckpt"
fi

# ---------------------------------------------------------------------------
# STAGE 4: Evaluate baselines
# ---------------------------------------------------------------------------
log "=== STAGE 4: Evaluate Baselines ==="
if [ "$SKIP_EVAL" -eq 0 ]; then
  sbatch "$SLURM_DIR/run_baselines.sh"
  sbatch "$SLURM_DIR/run_swinir.sh"
fi

# ---------------------------------------------------------------------------
# STAGE 5: Core SR evaluations (PSNR / MS-SSIM / LPIPS)
# ---------------------------------------------------------------------------
log "=== STAGE 5: Core SR Evaluations ==="
if [ "$SKIP_EVAL" -eq 0 ]; then
  # (Run after checkpoints from Stage 3 are available)
  log "Run eval_diffusion_sr.py for each checkpoint + dataset combo."
  log "See README §5 or individual slurm scripts for exact commands."
  sbatch "$SLURM_DIR/reeval_for_pvalues.sh"
fi

# ---------------------------------------------------------------------------
# STAGE 6: New analyses (freq / hallucination / embedding / diffmaps)
# ---------------------------------------------------------------------------
log "=== STAGE 6: Paper 1 Analyses ==="
if [ "$SKIP_EVAL" -eq 0 ]; then
  sbatch "$SLURM_DIR/run_frequency_analysis.sh"
  sbatch "$SLURM_DIR/run_hallucination.sh"
  sbatch "$SLURM_DIR/run_multiresolution_embedding.sh"
  sbatch "$SLURM_DIR/run_sr_hr_diffmaps.sh"
fi

# ---------------------------------------------------------------------------
# STAGE 7: Ablations (step count T, flow matching, T=16)
# ---------------------------------------------------------------------------
log "=== STAGE 7: Ablation Studies ==="
if [ "$SKIP_EVAL" -eq 0 ]; then
  sbatch "$SLURM_DIR/run_step_ablation.sh"
  sbatch "$SLURM_DIR/run_flow_matching_mrnet.sh"
  sbatch "$SLURM_DIR/run_flow_matching_brats.sh"
  sbatch "$SLURM_DIR/run_flow_matching_cxr.sh"
  sbatch "$SLURM_DIR/run_t16_mrnet.sh"
  sbatch "$SLURM_DIR/run_t16_brats.sh"
  sbatch "$SLURM_DIR/run_t16_cxr.sh"
fi

# ---------------------------------------------------------------------------
# STAGE 8: Statistical tests
# ---------------------------------------------------------------------------
log "=== STAGE 8: Statistical Analysis (CPU — runs locally) ==="
conda run -n medvae-sr python "$SCRIPTS/compute_effect_sizes.py"
conda run -n medvae-sr python "$SCRIPTS/compute_pvalues.py"
conda run -n medvae-sr python "$SCRIPTS/compute_bland_altman.py"
log "Statistical outputs → $OUTPUTS/statistical_tests/"

# ---------------------------------------------------------------------------
# STAGE 9: Generate all figures
# ---------------------------------------------------------------------------
log "=== STAGE 9: Generate Figures ==="
cd "$REPO"
conda run -n medvae-sr python "$SCRIPTS/generate_paper1_fig1.py"
conda run -n medvae-sr python "$SCRIPTS/plot_ae_ceiling_correlation.py"
conda run -n medvae-sr python "$SCRIPTS/generate_freq_figures.py"
conda run -n medvae-sr python "$SCRIPTS/generate_visual_comparisons.py"
conda run -n medvae-sr python "$SCRIPTS/generate_paper1_composite_figs.py"
conda run -n medvae-sr python "$SCRIPTS/generate_paper1_fig6.py"
conda run -n medvae-sr python "$SCRIPTS/plot_perception_distortion.py"
conda run -n medvae-sr bash   "$SCRIPTS/collect_paper1_figures.sh"
log "Figures → $REPO/figures-paper-1/"

# ---------------------------------------------------------------------------
# STAGE 10: Copy final figures into figures/
# ---------------------------------------------------------------------------
log "=== STAGE 10: Collect Final Figures ==="
cp -r "$REPO/figures-paper-1/"* "$FIGURES/"
log "All figures collected in $FIGURES/"
log ""
log "=== REPRODUCTION COMPLETE ==="
log "Upload figures/ as the 'figures/' folder in Overleaf."
