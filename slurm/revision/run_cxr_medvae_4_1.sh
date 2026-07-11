#!/bin/bash
#SBATCH --job-name=rev_cxr41
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# ============================================================
# REVISION (Scientific Reports major revision, Item 1: capacity-vs-domain)
# CXR + MedVAE 4_1_2d (1-channel latent, 1x64x64 = 4096 floats).
# Capacity-matched medical control vs SD-VAE (4x32x32 = 4096 floats).
# Reference-based metrics only. No downstream (Leo constraint).
# CXR training ~9h for 100 epochs; 6h limit -> auto-resumes from last.ckpt.
# Resubmit if it times out (training continues from checkpoint).
# ============================================================

source "$(dirname "$0")/../_env.sh"

DATA_BASE="/orcd/pool/006/lceli_shared"
OUT_BASE="${DATA_BASE}/mri-uganda"
DATA_ROOT="${DATA_BASE}/DATASET/mimic-cxr-sr"

LATENT_DIR="${OUT_BASE}/embeddings/medvae_4_1_cxr_s1"
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_medvae_4_1_cxr_s1_x0"
EVAL_DIR="outputs/experiments/revision_capacity/cxr_medvae_4_1_s1"
AE_DIR="outputs/experiments/revision_capacity/cxr_medvae_4_1_ae"

EXPERIMENT=rev_cxr_medvae_4_1
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${EVAL_DIR}" "${AE_DIR}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT} (capacity-matched, 1-ch latent)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

# Step 1: Extract S1 latents with medvae_4_1_2d (1 channel)
if [ ! -d "${LATENT_DIR}/train_latent" ] || [ "$(ls ${LATENT_DIR}/train_latent/hr_*.npy 2>/dev/null | wc -l)" -eq 0 ]; then
    echo ""
    echo "=== Step 1: Extract MedVAE 4_1_2d S1 latents (1 channel) ==="
    python -u medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \
        --data-root "${DATA_ROOT}" \
        --latent-root "${LATENT_DIR}" \
        --model-name medvae_4_1_2d \
        --modality xray \
        --backend medvae \
        --splits train valid \
        --batch-size 8
else
    echo "=== Step 1: S1 latents already exist, skipping ==="
fi

# Step 2: Train diffusion on S1 latents (auto-resumes from last.ckpt)
echo ""
echo "=== Step 2: Train diffusion SR (x0-prediction, 1-ch latents) ==="
python -u medvae_diffusion_pipeline/scripts/03_train_diffusion.py \
    --train-latent-dir "${LATENT_DIR}/train_latent" \
    --val-latent-dir "${LATENT_DIR}/valid_latent" \
    --output-dir "${WEIGHT_DIR}" \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --timesteps 1000 \
    --seed 42

# Step 3: Evaluate SR quality on validation set (n=1000, matches paper)
echo ""
echo "=== Step 3: Evaluate SR (valid, T=1000) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${WEIGHT_DIR}/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --backend medvae \
    --medvae-model medvae_4_1_2d \
    --modality xray \
    --timesteps 1000 \
    --output-dir "${EVAL_DIR}"

# Step 4: VAE autoencoder ceiling (encode->decode vs HR, no diffusion)
echo ""
echo "=== Step 4: VAE autoencoder ceiling (medvae_4_1_2d) ==="
python -u scripts/eval_vae_reconstruction.py \
    --backend medvae \
    --medvae-model medvae_4_1_2d \
    --modality xray \
    --hr-dir "${DATA_ROOT}/valid/hr" \
    --output-dir "${AE_DIR}"

echo ""
echo "========================================="
echo "Experiment ${EXPERIMENT} complete at: $(date)"
echo "========================================="
