#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=prep_brats
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

### Environment Setup
cd /orcd/home/002/sebasmos/orcd/pool/code/latent-sr
module load miniforge/24.3.0-0
conda activate medvae-sr
export PYTHONNOUSERSITE=1
export PYTHONPATH="/orcd/home/002/sebasmos/orcd/pool/code/latent-sr:$PYTHONPATH"

LOG_FILE="slurm/prep_brats_${SLURM_JOB_ID}.log"
mkdir -p slurm
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "BraTS 2023 Data Preparation"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="

DATA_BASE="/orcd/pool/006/lceli_shared"

python -u scripts/prepare_brats_sr.py \
    --data-root "${DATA_BASE}/DATASET/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" \
    --output-dir "${DATA_BASE}/DATASET/brats2023-sr" \
    --mri-sequence t2w \
    --slice-range 0.25 0.75 \
    --slices-per-volume 20 \
    --n-train 200 \
    --n-val 35 \
    --n-test 36 \
    --seed 42

echo "========================================="
echo "BraTS prep complete at: $(date)"
echo "========================================="
