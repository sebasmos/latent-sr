#!/bin/bash
#SBATCH --job-name=sr_fid
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

source "$(dirname "$0")/_env.sh"

LOG_FILE="slurm/experiments/fid_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "FID Computation — all methods / datasets"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

python -u scripts/compute_fid.py

echo ""
echo "FID computation complete — $(date)"
