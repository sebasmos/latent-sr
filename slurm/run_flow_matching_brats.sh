#!/bin/bash
#SBATCH --job-name=flow_brats
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# Issue #77 Path A — Rectified Flow (MedVAE) — BraTS dataset
# Replaces DDPM with straight-line ODE (flow matching) in MedVAE latent space.
# Target: +0.5–2 dB PSNR (from 27.13 dB baseline), BraTS Dice >0.30, 10–50× speedup.
# 16 Euler steps vs 1000 DDPM steps.

source "$(dirname "$0")/_env.sh"

DATA_BASE="${LATENT_SR_SHARED_ROOT}"
OUT_BASE="${LATENT_SR_MRI_UGANDA_ROOT}"

LATENT_DIR="${OUT_BASE}/embeddings/medvae_brats_s1"
WEIGHT_DIR="${OUT_BASE}/weights/flow_matching_medvae_brats"
EVAL_DIR="outputs/experiments/brats_flow_matching"

LOG_FILE="slurm/experiments/flow_matching_brats_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "BraTS Flow Matching SR — Issue #77 Path A"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

########################################################################
# Step 1: Train rectified flow model on BraTS MedVAE latents
########################################################################
echo ""
echo "=== Training BraTS MedVAE rectified flow model ==="
python -u medvae_diffusion_pipeline/scripts/03_train_flow_matching.py \
    --train-latent-dir "${LATENT_DIR}/train_latent" \
    --val-latent-dir "${LATENT_DIR}/valid_latent" \
    --output-dir "${WEIGHT_DIR}" \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --timesteps 1000 \
    --num-sample-steps 16 \
    --loss-type l1 \
    --augment

CKPT="${WEIGHT_DIR}/checkpoints/last.ckpt"

########################################################################
# Step 2: Evaluate SR quality with 16 Euler steps
########################################################################
echo ""
echo "=== Evaluating SR quality (flow matching, 16 Euler steps) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${CKPT}" \
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --medvae-model medvae_4_3_2d \
    --modality mri \
    --backend medvae \
    --timesteps 16 \
    --training-script medvae_diffusion_pipeline/scripts/03_train_flow_matching.py \
    --output-dir "${EVAL_DIR}"

echo ""
echo "========================================="
echo "BraTS flow matching complete at: $(date)"
echo "Results: ${EVAL_DIR}/diffusion_eval_results.json"
echo "========================================="
