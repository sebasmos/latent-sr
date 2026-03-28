#!/bin/bash
#SBATCH --job-name=sr_swinir
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Run SwinIR pretrained baseline for one dataset.
# Issue #49 — PLAN.md §12 row 57
#
# Usage (submit once per dataset):
#   sbatch --export=DATASET=mrnet slurm/run_swinir.sh
#   sbatch --export=DATASET=brats slurm/run_swinir.sh
#   sbatch --export=DATASET=cxr   slurm/run_swinir.sh

cd /orcd/home/002/sebasmos/orcd/pool/code/latent-sr
module load miniforge/24.3.0-0
conda activate medvae-sr
export PYTHONNOUSERSITE=1
export PYTHONPATH="/orcd/home/002/sebasmos/orcd/pool/code/latent-sr:$PYTHONPATH"

DATASET="${DATASET:?DATASET env var is required (mrnet|brats|cxr)}"
DATA_BASE="/orcd/pool/006/lceli_shared/DATASET"
OUT_ROOT="outputs/experiments"

case "${DATASET}" in
  mrnet)
    HR_DIR="${DATA_BASE}/mrnetkneemris/MRNet-v1.0-middle/valid/hr"
    LR_DIR="${DATA_BASE}/mrnetkneemris/MRNet-v1.0-middle/valid/lr"
    ;;
  brats)
    HR_DIR="${DATA_BASE}/brats2023-sr/test/hr"
    LR_DIR="${DATA_BASE}/brats2023-sr/test/lr"
    ;;
  cxr)
    HR_DIR="${DATA_BASE}/mimic-cxr-sr/test/hr"
    LR_DIR="${DATA_BASE}/mimic-cxr-sr/test/lr"
    ;;
  *)
    echo "ERROR: unknown DATASET '${DATASET}'. Use mrnet|brats|cxr." >&2
    exit 1
    ;;
esac

LOG_FILE="slurm/experiments/${DATASET}_swinir_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Job: SwinIR baseline — ${DATASET}"
echo "Job ID: ${SLURM_JOB_ID} | Node: $(hostname) | Date: $(date)"
echo "HR_DIR:  ${HR_DIR}"
echo "LR_DIR:  ${LR_DIR}"
echo "OUT:     ${OUT_ROOT}/${DATASET}_swinir"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

# Install timm if not present (SwinIR dependency for DropPath / trunc_normal_)
python -c "import timm" 2>/dev/null || {
    echo "timm not found — installing ..."
    pip install timm --quiet
}

python -u scripts/eval_swinir.py \
    --hr-dir  "${HR_DIR}" \
    --lr-dir  "${LR_DIR}" \
    --save-images \
    --output-dir "${OUT_ROOT}/${DATASET}_swinir"

echo "========================================="
echo "SwinIR ${DATASET} complete at: $(date)"
echo "========================================="
