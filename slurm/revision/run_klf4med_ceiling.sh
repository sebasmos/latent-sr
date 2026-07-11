#!/bin/bash
#SBATCH --job-name=rev_klf4med_ceil
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# REVISION R1 — medical-KL-f4 AE reconstruction CEILING (cheap, forward passes only).
# Evaluates the fine-tuned medical kl-f4 checkpoint on <dataset>/valid/hr (deterministic
# posterior mean -> decode -> PSNR). Same maths as eval_klf4_ae.py; the natural-side
# ceiling is eval_klf4_ae.py, so this is the medical half of the domain contrast.
#
# Usage:  sbatch run_klf4med_ceiling.sh <dataset>   # brats|cxr|mrnet
DATASET="${1:?usage: run_klf4med_ceiling.sh <dataset>  (brats|cxr|mrnet)}"

source "$(dirname "$0")/../_env.sh"
export PYTHONPATH="${LATENT_SR_REPO_ROOT}/latent-diffusion:$PYTHONPATH"

DATA_BASE="/orcd/pool/006/lceli_shared"
OUT_BASE="${DATA_BASE}/mri-uganda"
case "$DATASET" in
  mrnet) DATA_ROOT="${DATA_BASE}/DATASET/mrnetkneemris/MRNet-v1.0-middle" ;;
  brats) DATA_ROOT="${DATA_BASE}/DATASET/brats2023-sr" ;;
  cxr)   DATA_ROOT="${DATA_BASE}/DATASET/mimic-cxr-sr" ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

MED_CKPT="${OUT_BASE}/weights/klf4med_${DATASET}/checkpoints/last.ckpt"
OUT_DIR="outputs/experiments/revision_capacity/klf4med_${DATASET}_ceiling"
LOG="slurm/experiments/rev_klf4med_ceil_${DATASET}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${OUT_DIR}"
exec > >(tee -a "$LOG") 2>&1

echo "=== medical-KL-f4 ceiling | dataset=${DATASET} | ckpt=${MED_CKPT} | $(date) ==="
if [ ! -f "${MED_CKPT}" ]; then echo "MISSING medical ckpt ${MED_CKPT} — run run_klf4med_finetune.sh first"; exit 2; fi
nvidia-smi 2>/dev/null | head -3

python -u slurm/revision/finetune_klf4.py --eval-only \
    --dataset "${DATASET}" --data-root "${DATA_ROOT}" \
    --init-ckpt "${MED_CKPT}" --out-dir "${OUT_DIR}"

echo "=== done ${DATASET} at $(date) | results: ${OUT_DIR}/ceiling.json ==="
