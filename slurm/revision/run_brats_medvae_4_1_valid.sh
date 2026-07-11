#!/bin/bash
#SBATCH --job-name=rev_brats41v
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# ============================================================
# REVISION Item 1: BraTS + MedVAE 4_1_2d re-evaluated on the VALIDATION split
# for apples-to-apples comparison with the paper's 4_3 valid numbers (n=700).
# The existing brats_medvae_4_1_s1 result was on the TEST split (n=720).
# Uses the ALREADY-TRAINED checkpoint + already-extracted latents. No training.
# ============================================================

source "$(dirname "$0")/../_env.sh"

DATA_BASE="/orcd/pool/006/lceli_shared"
OUT_BASE="${DATA_BASE}/mri-uganda"
DATA_ROOT="${DATA_BASE}/DATASET/brats2023-sr"

LATENT_DIR="${OUT_BASE}/embeddings/medvae_4_1_brats_s1"
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_medvae_4_1_brats_s1_x0"
EVAL_DIR="outputs/experiments/revision_capacity/brats_medvae_4_1_s1_valid"
AE_DIR="outputs/experiments/revision_capacity/brats_medvae_4_1_ae_valid"

EXPERIMENT=rev_brats_medvae_4_1_valid
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${EVAL_DIR}" "${AE_DIR}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT} (valid-split re-eval, no training)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="

# SR eval on valid split
echo "=== SR eval (valid, T=1000) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${WEIGHT_DIR}/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --backend medvae \
    --medvae-model medvae_4_1_2d \
    --modality mri \
    --timesteps 1000 \
    --output-dir "${EVAL_DIR}"

# AE ceiling on valid HR
echo "=== AE ceiling (valid) ==="
python -u scripts/eval_vae_reconstruction.py \
    --backend medvae \
    --medvae-model medvae_4_1_2d \
    --modality mri \
    --hr-dir "${DATA_ROOT}/valid/hr" \
    --output-dir "${AE_DIR}"

echo "========================================="
echo "Experiment ${EXPERIMENT} complete at: $(date)"
echo "========================================="
