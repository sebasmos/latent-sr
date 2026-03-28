#!/bin/bash
#SBATCH --job-name=t16_mrnet
#SBATCH --output=slurm/experiments/t16_mrnet_%j.log
#SBATCH --error=slurm/experiments/t16_mrnet_%j.log
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd /orcd/home/002/sebasmos/orcd/pool/code/latent-sr
module load miniforge/24.3.0-0
conda activate medvae-sr
export PYTHONNOUSERSITE=1
export PYTHONPATH="/orcd/home/002/sebasmos/orcd/pool/code/latent-sr:$PYTHONPATH"

mkdir -p slurm/experiments
echo "========================================="
echo "T=16 inference: MRNet"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

python -u scripts/eval_diffusion_sr.py \
    --checkpoint "/orcd/pool/006/lceli_shared/mri-uganda/weights/diffusion_medvae_mrnet_x0/checkpoints/last.ckpt" \
    --latent-dir "/orcd/pool/006/lceli_shared/mri-uganda/embeddings/medvae_4_3_2d_v2/phase2/valid_latent" \
    --backend medvae \
    --medvae-model medvae_4_3_2d \
    --modality mri \
    --timesteps 16 \
    --batch-size 4 \
    --save-images \
    --output-dir "outputs/experiments/mrnet_medvae_t16"

echo ""
echo "========================================="
echo "T=16 MRNet inference complete at: $(date)"
echo "========================================="
