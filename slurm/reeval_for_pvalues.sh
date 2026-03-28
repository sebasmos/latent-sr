#!/bin/bash
#SBATCH --job-name=pval_ev
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# Re-evaluate SD-VAE and BraTS MedVAE S1 with Sahil's updated eval script
# to capture per-image metrics needed for Wilcoxon p-values.

cd /orcd/home/002/sebasmos/orcd/pool/code/latent-sr
module load miniforge/24.3.0-0
conda activate medvae-sr
export PYTHONNOUSERSITE=1
export PYTHONPATH="/orcd/home/002/sebasmos/orcd/pool/code/latent-sr:$PYTHONPATH"

DATA_BASE="/orcd/pool/006/lceli_shared"
OUT_BASE="${DATA_BASE}/mri-uganda"

LOG_FILE="slurm/experiments/reeval_pvalues_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Re-eval for p-values (per-image metrics)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="

# 1. MRNet SD-VAE re-eval (120 samples, ~5 min)
echo ""
echo "=== MRNet SD-VAE re-eval ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${OUT_BASE}/weights/diffusion_sdvae_mrnet_x0/checkpoints/last.ckpt" \
    --latent-dir "${OUT_BASE}/embeddings/sd_vae_v2/phase2/valid_latent" \
    --backend sd-vae \
    --modality mri \
    --timesteps 1000 \
    --output-dir "outputs/experiments/mrnet_sdvae_v2"

# 2. BraTS SD-VAE re-eval (720 samples, ~30 min)
echo ""
echo "=== BraTS SD-VAE re-eval ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${OUT_BASE}/weights/diffusion_sdvae_brats_x0/checkpoints/last.ckpt" \
    --latent-dir "${OUT_BASE}/embeddings/sd_vae_brats/test_latent" \
    --backend sd-vae \
    --modality mri \
    --timesteps 1000 \
    --output-dir "outputs/experiments/brats_sdvae_v2"

# 3. BraTS MedVAE S1 re-eval (720 samples, ~30 min)
echo ""
echo "=== BraTS MedVAE S1 re-eval ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${OUT_BASE}/weights/diffusion_medvae_brats_s1_x0/checkpoints/last.ckpt" \
    --latent-dir "${OUT_BASE}/embeddings/medvae_brats_s1/test_latent" \
    --backend medvae \
    --medvae-model medvae_4_3_2d \
    --modality mri \
    --timesteps 1000 \
    --output-dir "outputs/experiments/brats_medvae_s1_v2"

# 4. Compute p-values
echo ""
echo "=== Computing p-values ==="
python -u scripts/compute_pvalues.py

echo ""
echo "========================================="
echo "Re-eval for p-values complete at: $(date)"
echo "========================================="
