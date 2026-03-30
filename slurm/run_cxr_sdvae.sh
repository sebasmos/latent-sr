#!/bin/bash
#SBATCH --job-name=cxr_sd
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# CXR + SD-VAE: 8K training images → needs ~9h for 100 epochs.
# mit_normal_gpu has 6h limit, so this may need 2 submissions.
# Training auto-resumes from last.ckpt on resubmit.
# Latents already extracted (7992 files), so Step 1 will be skipped.

source "$(dirname "$0")/_env.sh"

DATA_BASE="${LATENT_SR_SHARED_ROOT}"
OUT_BASE="${LATENT_SR_MRI_UGANDA_ROOT}"

EXPERIMENT=cxr_sdvae
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT}"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

LATENT_DIR="${OUT_BASE}/embeddings/sd_vae_cxr"
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_sdvae_cxr_x0"
EVAL_DIR="outputs/experiments/cxr_sdvae"
DATA_ROOT="${LATENT_SR_DATA_ROOT}/mimic-cxr-sr"

# Step 1: Extract latents (skip if already exist)
if [ ! -d "${LATENT_DIR}/train_latent" ] || [ "$(ls ${LATENT_DIR}/train_latent/hr_*.npy 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "=== Step 1: Extract SD-VAE latents ==="
    python -u medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \
        --data-root "${DATA_ROOT}" \
        --latent-root "${LATENT_DIR}" \
        --model-name stabilityai/sd-vae-ft-ema \
        --modality xray \
        --backend sd-vae \
        --splits train valid \
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
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --backend sd-vae \
    --medvae-model stabilityai/sd-vae-ft-ema \
    --modality xray \
    --timesteps 1000 \
    --save-images \
    --output-dir "${EVAL_DIR}"

# Step 4: Downstream classification
echo ""
echo "=== Step 4: Downstream classification ==="
python -u scripts/eval_downstream.py \
    --task classification \
    --hr-dir "${DATA_ROOT}/test/hr" \
    --lr-dir "${DATA_ROOT}/test/lr" \
    --sr-dir "${EVAL_DIR}/sr_images" \
    --labels-csv "${DATA_ROOT}/test/labels.csv" \
    --train-hr-dir "${DATA_ROOT}/train/hr" \
    --train-labels-csv "${DATA_ROOT}/train/labels.csv" \
    --label-columns "Atelectasis,Cardiomegaly,Edema,Pleural Effusion,Pneumothorax" \
    --output-dir "${EVAL_DIR}"

echo ""
echo "========================================="
echo "Experiment ${EXPERIMENT} complete at: $(date)"
echo "========================================="
