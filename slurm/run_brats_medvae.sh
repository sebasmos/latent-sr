#!/bin/bash
#SBATCH --job-name=brats_med
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# BraTS + MedVAE: 4K training slices (200 subjects × 20 slices)
# Latents NOT yet extracted. Full pipeline: extract → train → eval → downstream.
# Training should be fast (~15min for 100 epochs with small dataset).

source "$(dirname "$0")/_env.sh"

DATA_BASE="${LATENT_SR_SHARED_ROOT}"
OUT_BASE="${LATENT_SR_MRI_UGANDA_ROOT}"

EXPERIMENT=brats_medvae
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT}"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

LATENT_DIR="${OUT_BASE}/embeddings/medvae_4_3_2d_brats"
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_medvae_brats_x0"
EVAL_DIR="outputs/experiments/brats_medvae"
DATA_ROOT="${LATENT_SR_DATA_ROOT}/brats2023-sr"

# Step 1: Extract latents
if [ ! -d "${LATENT_DIR}/test_latent" ] || [ "$(ls ${LATENT_DIR}/test_latent/hr_*.npy 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "=== Step 1: Extract MedVAE latents ==="
    python -u medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \
        --data-root "${DATA_ROOT}" \
        --latent-root "${LATENT_DIR}" \
        --model-name medvae_4_3_2d \
        --modality mri \
        --backend medvae \
        --splits train valid test \
        --batch-size 8
else
    echo "=== Step 1: Latents already exist ($(ls ${LATENT_DIR}/train_latent/hr_*.npy | wc -l) files), skipping ==="
fi

# Step 2: Train diffusion
echo ""
echo "=== Step 2: Train diffusion SR (x0-prediction) ==="
python -u medvae_diffusion_pipeline/scripts/03_train_diffusion.py \
    --train-latent-dir "${LATENT_DIR}/train_latent" \
    --val-latent-dir "${LATENT_DIR}/valid_latent" \
    --output-dir "${WEIGHT_DIR}" \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --timesteps 1000 \
    --seed 42

# Step 3: Evaluate SR quality
echo ""
echo "=== Step 3: Evaluate SR ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${WEIGHT_DIR}/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_DIR}/test_latent" \
    --backend medvae \
    --medvae-model medvae_4_3_2d \
    --modality mri \
    --timesteps 1000 \
    --save-images \
    --output-dir "${EVAL_DIR}"

# Step 4: Downstream segmentation
echo ""
echo "=== Step 4: Downstream segmentation ==="
python -u scripts/eval_downstream.py \
    --task segmentation \
    --hr-dir "${DATA_ROOT}/test/hr" \
    --lr-dir "${DATA_ROOT}/test/lr" \
    --sr-dir "${EVAL_DIR}/sr_images" \
    --seg-dir "${DATA_ROOT}/test/seg_masks" \
    --output-dir "${EVAL_DIR}"

echo ""
echo "========================================="
echo "Experiment ${EXPERIMENT} complete at: $(date)"
echo "========================================="
