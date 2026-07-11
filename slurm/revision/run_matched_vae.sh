#!/bin/bash
#SBATCH --job-name=rev_vae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# ============================================================
# REVISION Item 1 (capacity-vs-domain) — parametrized MedVAE geometry sweep.
# Usage:  sbatch [--partition=P] run_matched_vae.sh <medvae_model> <dataset>
#   <medvae_model> : medvae_8_4_2d (4x32x32, EXACT SD-VAE geometry) |
#                    medvae_8_1_2d (1x32x32=1024) | medvae_4_4_2d (4x64x64=16384) | medvae_4_1_2d ...
#   <dataset>      : mrnet | brats | cxr
# Feeds: (a) spatial-matched domain control vs SD-VAE, (b) AE-ceiling regression points.
# Reference metrics only. No downstream (Leo constraint). Namespaced under revision_capacity/.
# ============================================================

MODEL="${1:?usage: run_matched_vae.sh <medvae_model> <dataset>}"
DATASET="${2:?usage: run_matched_vae.sh <medvae_model> <dataset>}"

source "$(dirname "$0")/../_env.sh"

DATA_BASE="/orcd/pool/006/lceli_shared"
OUT_BASE="${DATA_BASE}/mri-uganda"

case "$DATASET" in
  mrnet) DATA_ROOT="${DATA_BASE}/DATASET/mrnetkneemris/MRNet-v1.0-middle"; MODALITY=mri ;;
  brats) DATA_ROOT="${DATA_BASE}/DATASET/brats2023-sr";                    MODALITY=mri ;;
  cxr)   DATA_ROOT="${DATA_BASE}/DATASET/mimic-cxr-sr";                    MODALITY=xray ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

TAG="${MODEL//medvae_/}"; TAG="${TAG//_2d/}"   # e.g. 8_4
LATENT_DIR="${OUT_BASE}/embeddings/medvae_${TAG}_${DATASET}_s1"
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_medvae_${TAG}_${DATASET}_s1_x0"
EVAL_DIR="outputs/experiments/revision_capacity/${DATASET}_medvae_${TAG}_s1"
AE_DIR="outputs/experiments/revision_capacity/${DATASET}_medvae_${TAG}_ae"

EXPERIMENT="rev_${DATASET}_medvae_${TAG}"
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${EVAL_DIR}" "${AE_DIR}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT} | model=${MODEL} dataset=${DATASET}"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

# Step 1: Extract latents with the chosen MedVAE geometry
if [ ! -d "${LATENT_DIR}/train_latent" ] || [ "$(ls ${LATENT_DIR}/train_latent/hr_*.npy 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "=== Step 1: Extract ${MODEL} latents ==="
    python -u medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \
        --data-root "${DATA_ROOT}" --latent-root "${LATENT_DIR}" \
        --model-name "${MODEL}" --modality "${MODALITY}" --backend medvae \
        --splits train valid --batch-size 8
else
    echo "=== Step 1: latents exist, skipping ==="
fi

# Step 2: Train diffusion (auto-resumes from last.ckpt)
echo "=== Step 2: Train diffusion SR (x0) ==="
python -u medvae_diffusion_pipeline/scripts/03_train_diffusion.py \
    --train-latent-dir "${LATENT_DIR}/train_latent" \
    --val-latent-dir "${LATENT_DIR}/valid_latent" \
    --output-dir "${WEIGHT_DIR}" \
    --epochs 100 --batch-size 8 --lr 1e-4 --timesteps 1000 --seed 42

# Step 3: Evaluate SR (valid split)
echo "=== Step 3: Evaluate SR (valid, T=1000) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${WEIGHT_DIR}/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --backend medvae --medvae-model "${MODEL}" --modality "${MODALITY}" \
    --timesteps 1000 --output-dir "${EVAL_DIR}"

# Step 4: AE ceiling (encode->decode vs HR, no diffusion)
echo "=== Step 4: AE ceiling (${MODEL}) ==="
python -u scripts/eval_vae_reconstruction.py \
    --backend medvae --medvae-model "${MODEL}" --modality "${MODALITY}" \
    --hr-dir "${DATA_ROOT}/valid/hr" --output-dir "${AE_DIR}"

echo "========================================="
echo "Experiment ${EXPERIMENT} complete at: $(date)"
echo "========================================="
