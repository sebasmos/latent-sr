#!/bin/bash
#SBATCH --job-name=t16_brats
#SBATCH --output=slurm/experiments/t16_brats_%j.log
#SBATCH --error=slurm/experiments/t16_brats_%j.log
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source "$(dirname "$0")/_env.sh"

mkdir -p slurm/experiments
echo "========================================="
echo "T=16 inference: BraTS"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${LATENT_SR_WEIGHTS_ROOT}/diffusion_medvae_brats_x0/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_SR_EMBEDDINGS_ROOT}/medvae_4_3_2d_brats/test_latent" \
    --backend medvae \
    --medvae-model medvae_4_3_2d \
    --modality mri \
    --timesteps 16 \
    --batch-size 4 \
    --save-images \
    --output-dir "outputs/experiments/brats_medvae_t16"

echo ""
echo "========================================="
echo "T=16 BraTS inference complete at: $(date)"
echo "========================================="
