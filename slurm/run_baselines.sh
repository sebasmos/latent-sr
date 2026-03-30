#!/bin/bash
#SBATCH --job-name=sr_base
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Usage:
#   sbatch --export=DATASET=mrnet,HR_DIR=...,LR_DIR=...,OUT_ROOT=outputs/experiments,MODEL=RealESRGAN_x4plus slurm/run_baselines.sh

source "$(dirname "$0")/_env.sh"

DATASET="${DATASET:?DATASET is required}"
HR_DIR="${HR_DIR:?HR_DIR is required}"
LR_DIR="${LR_DIR:?LR_DIR is required}"
OUT_ROOT="${OUT_ROOT:-outputs/experiments}"
MODEL="${MODEL:-RealESRGAN_x4plus}"

LOG_FILE="slurm/experiments/${DATASET}_baselines_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Dataset: ${DATASET}"
echo "HR_DIR:  ${HR_DIR}"
echo "LR_DIR:  ${LR_DIR}"
echo "OUT_ROOT:${OUT_ROOT}"
echo "========================================="

python -u scripts/eval_baselines.py \
  --method bicubic \
  --hr-dir "${HR_DIR}" \
  --lr-dir "${LR_DIR}" \
  --save-images \
  --output-dir "${OUT_ROOT}/${DATASET}_bicubic"

python -u scripts/eval_baselines.py \
  --method realesrgan \
  --realesrgan-model "${MODEL}" \
  --hr-dir "${HR_DIR}" \
  --lr-dir "${LR_DIR}" \
  --save-images \
  --output-dir "${OUT_ROOT}/${DATASET}_realesrgan"

echo "Baseline evaluation complete."
