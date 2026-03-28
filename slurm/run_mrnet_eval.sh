#!/bin/bash
#SBATCH --job-name=mrnet_ev
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# MRNet eval-only: SR eval on valid set (120 samples) + downstream classification
# Training already done — just runs Steps 3+4 for both backends.

cd /orcd/home/002/sebasmos/orcd/pool/code/latent-sr
module load miniforge/24.3.0-0
conda activate medvae-sr
export PYTHONNOUSERSITE=1
export PYTHONPATH="/orcd/home/002/sebasmos/orcd/pool/code/latent-sr:$PYTHONPATH"

DATA_BASE="/orcd/pool/006/lceli_shared"
OUT_BASE="${DATA_BASE}/mri-uganda"
DATA_ROOT="${DATA_BASE}/DATASET/mrnetkneemris/MRNet-v1.0-middle"

LOG_FILE="slurm/experiments/mrnet_eval_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "MRNet Eval (valid set, both backends)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

# --- MedVAE ---
echo ""
echo "=== MedVAE: SR eval on valid set (120 samples) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${OUT_BASE}/weights/diffusion_medvae_mrnet_x0/checkpoints/last.ckpt" \
    --latent-dir "${OUT_BASE}/embeddings/medvae_4_3_2d_v2/phase2/valid_latent" \
    --backend medvae \
    --medvae-model medvae_4_3_2d \
    --modality mri \
    --timesteps 1000 \
    --save-images \
    --output-dir "outputs/experiments/mrnet_medvae"

echo ""
echo "=== MedVAE: Downstream classification ==="
python -u scripts/eval_downstream.py \
    --task classification \
    --hr-dir "${DATA_ROOT}/valid/hr" \
    --lr-dir "${DATA_ROOT}/valid/lr" \
    --sr-dir "outputs/experiments/mrnet_medvae/sr_images" \
    --labels-csv "${DATA_ROOT}/valid/labels.csv" \
    --train-hr-dir "${DATA_ROOT}/train/hr" \
    --train-labels-csv "${DATA_ROOT}/train/labels.csv" \
    --label-columns "acl,meniscus,abnormal" \
    --output-dir "outputs/experiments/mrnet_medvae"

# --- SD-VAE ---
echo ""
echo "=== SD-VAE: SR eval on valid set (120 samples) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${OUT_BASE}/weights/diffusion_sdvae_mrnet_x0/checkpoints/last.ckpt" \
    --latent-dir "${OUT_BASE}/embeddings/sd_vae_v2/phase2/valid_latent" \
    --backend sd-vae \
    --medvae-model stabilityai/sd-vae-ft-ema \
    --modality mri \
    --timesteps 1000 \
    --save-images \
    --output-dir "outputs/experiments/mrnet_sdvae"

echo ""
echo "=== SD-VAE: Downstream classification ==="
python -u scripts/eval_downstream.py \
    --task classification \
    --hr-dir "${DATA_ROOT}/valid/hr" \
    --lr-dir "${DATA_ROOT}/valid/lr" \
    --sr-dir "outputs/experiments/mrnet_sdvae/sr_images" \
    --labels-csv "${DATA_ROOT}/valid/labels.csv" \
    --train-hr-dir "${DATA_ROOT}/train/hr" \
    --train-labels-csv "${DATA_ROOT}/train/labels.csv" \
    --label-columns "acl,meniscus,abnormal" \
    --output-dir "outputs/experiments/mrnet_sdvae"

echo ""
echo "========================================="
echo "MRNet eval complete at: $(date)"
echo "========================================="
