#!/bin/bash
#SBATCH --job-name=rev_klf4med_ft
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# ============================================================
# REVISION R1 — medical-KL-f4 FINE-TUNE (domain-isolation control).
# Fine-tune the EXACT CompVis kl-f4 first-stage AE (3x64x64) on a medical dataset,
# starting from the natural checkpoint. Architecture / geometry / init / encode
# convention held identical to klf4_vae.py & eval_klf4_ae.py; ONLY the domain changes.
# Also writes the medical AE reconstruction ceiling (valid) at the end.
#
# Usage:  sbatch [--time=T] run_klf4med_finetune.sh <dataset>   # brats|cxr|mrnet
# RESUMABLE: finetune_klf4.py auto-resumes from checkpoints/last.ckpt; --requeue safe.
# ============================================================
DATASET="${1:?usage: run_klf4med_finetune.sh <dataset> [epochs] [tag]  (brats|cxr|mrnet)}"
EPOCHS="${2:-20}"
# TAG isolates a rerun from a previous run's checkpoints (auto-resume would otherwise pick
# up the old last.ckpt). v2 = the kl_weight=1e-8 + best-ckpt-guard recipe (2026-07-09).
TAG="${3:-}"

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

OUT_DIR="${OUT_BASE}/weights/klf4med_${DATASET}${TAG}"
LOG="slurm/experiments/rev_klf4med_ft_${DATASET}${TAG}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${OUT_DIR}"
exec > >(tee -a "$LOG") 2>&1

echo "=== medical-KL-f4 fine-tune | dataset=${DATASET} tag='${TAG}' | Job ${SLURM_JOB_ID} | $(date) ==="
nvidia-smi 2>/dev/null | head -5

# kl-weight intentionally NOT passed: use finetune_klf4.py's corrected default (1e-8).
# Passing 1e-6 here is what caused the BraTS posterior collapse (see finetune_klf4.py note).
python -u slurm/revision/finetune_klf4.py \
    --dataset "${DATASET}" --data-root "${DATA_ROOT}" --out-dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" --batch-size 4 --lr 1.5e-5 --seed 42

echo "=== done ${DATASET} at $(date) | ckpt: ${OUT_DIR}/checkpoints/last.ckpt ==="
