#!/bin/bash
#SBATCH --job-name=table4
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Replicate MedVAE Table 4 using author's exact protocol (issue #33):
# - 3D volumes with 160³ center crop
# - 2D slice-by-slice encode/decode
# - 100 random validation volumes, fixed seed
# - Both C=1 and C=3 models

source "$(dirname "$0")/_env.sh"

LOG_FILE="slurm/experiments/table4_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Table 4 Replication (MedVAE author protocol)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

DATA_ROOT="${LATENT_SR_DATA_ROOT}/mrnetkneemris/MRNet-v1.0"

echo ""
echo "=== medvae_4_3_2d (f=16, C=3) — Paper target: 31.52 dB ==="
python -u scripts/replicate_table4.py \
    --data-root "${DATA_ROOT}" \
    --model-name medvae_4_3_2d \
    --modality mri \
    --n-samples 100 \
    --seed 42 \
    --output-dir outputs/table4_replication

echo ""
echo "=== medvae_4_1_2d (f=16, C=1) — Paper target: 27.38 dB ==="
python -u scripts/replicate_table4.py \
    --data-root "${DATA_ROOT}" \
    --model-name medvae_4_1_2d \
    --modality mri \
    --n-samples 100 \
    --seed 42 \
    --output-dir outputs/table4_replication

echo ""
echo "========================================="
echo "Table 4 replication complete at: $(date)"
echo "========================================="
