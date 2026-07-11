#!/bin/bash
#SBATCH --job-name=rev_klf4sr_seed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# ============================================================
# REVISION R3 — SEED SWEEP for the KL-f4 SR tables (repeatability / mean+-std).
# Reviewers flagged the KL-f4 SR decomposition tables as single runs with no
# variance. This reruns the SAME KL-f4 SR pipeline as run_klf4_sr.sh for a given
# random SEED so that n=3 seeds/dataset give mean +- std.
#
# WHAT CHANGES vs run_klf4_sr.sh (nothing else):
#   * takes a SEED positional arg and passes it to 03_train_diffusion (--seed).
#     Verified: --seed -> pl.seed_everything(seed, workers=True) reseeds torch/
#     numpy/random/DataLoader-workers; seed 42 vs 43 give different RNG streams.
#   * WEIGHT_DIR and EVAL_DIR are made SEED-SPECIFIC so the 3 seeds do NOT collide
#     on checkpoints/last.ckpt (auto-resume) or overwrite each other's results.
#   * LATENT_DIR is SHARED across seeds (klf4 VAE latents are deterministic /
#     seed-independent) -> extract once, reuse -> no wasted recompute.
#
# Usage:  sbatch [--time=T] run_klf4_sr_seedsweep.sh <dataset> <seed> [epochs]
#            dataset: mrnet|brats|cxr    seed: e.g. 42|43|44    epochs: default 100 (set 1 for smoke)
#
# FULLY RESUMABLE (same guards as run_klf4_sr.sh): extract per-split completeness
# guard, 03_train auto-resume from last.ckpt, idempotent eval, --requeue.
# ============================================================

DATASET="${1:?usage: run_klf4_sr_seedsweep.sh <dataset> <seed> [epochs]   (mrnet|brats|cxr)}"
SEED="${2:?usage: run_klf4_sr_seedsweep.sh <dataset> <seed> [epochs]   (e.g. 42|43|44)}"
EPOCHS="${3:-100}"

source "$(dirname "$0")/../_env.sh"
export PYTHONPATH="${LATENT_SR_REPO_ROOT}/latent-diffusion:$PYTHONPATH"

DATA_BASE="/orcd/pool/006/lceli_shared"
OUT_BASE="${DATA_BASE}/mri-uganda"

case "$DATASET" in
  mrnet) DATA_ROOT="${DATA_BASE}/DATASET/mrnetkneemris/MRNet-v1.0-middle"; MODALITY=mri ;;
  brats) DATA_ROOT="${DATA_BASE}/DATASET/brats2023-sr";                    MODALITY=mri ;;
  cxr)   DATA_ROOT="${DATA_BASE}/DATASET/mimic-cxr-sr";                    MODALITY=xray ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac

# Latents are deterministic -> SHARED across seeds (same dir as run_klf4_sr.sh).
LATENT_DIR="${OUT_BASE}/embeddings/klf4_${DATASET}_s1"
# Weights + eval are SEED-SPECIFIC -> no collision between seeds.
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_klf4_${DATASET}_seed${SEED}_x0"
EVAL_DIR="outputs/experiments/revision_capacity/${DATASET}_klf4_seed${SEED}"

EXPERIMENT="rev_klf4sr_${DATASET}_seed${SEED}"
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${EVAL_DIR}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT} | backend=klf4 dataset=${DATASET} seed=${SEED} epochs=${EPOCHS}"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

# --- Step 1: Extract KL-f4 latents (SHARED dir; per-split completeness guard) ---
NEED_SPLITS=()
for split in train valid; do
    fn="${LATENT_DIR}/${split}_latent/filenames.txt"
    nsrc=$(ls "${DATA_ROOT}/${split}/hr"/*.png "${DATA_ROOT}/${split}/hr"/*.jpg 2>/dev/null | wc -l)
    if [ -f "$fn" ]; then nlat=$(wc -l < "$fn"); else nlat=0; fi
    if [ "$nsrc" -gt 0 ] && [ "$nlat" -eq "$nsrc" ]; then
        echo "=== Step 1: split '${split}' complete (${nlat}/${nsrc}) -> skip ==="
    else
        echo "=== Step 1: split '${split}' incomplete (${nlat}/${nsrc}) -> extract ==="
        NEED_SPLITS+=("$split")
    fi
done
if [ ${#NEED_SPLITS[@]} -gt 0 ]; then
    python -u medvae_diffusion_pipeline/scripts/02_extract_medvae_embeddings.py \
        --data-root "${DATA_ROOT}" --latent-root "${LATENT_DIR}" \
        --model-name klf4 --modality "${MODALITY}" --backend klf4 \
        --splits "${NEED_SPLITS[@]}" --batch-size 8 --image-size 256
fi

# --- Step 2: Train diffusion SR (SEED-SPECIFIC out; auto-resumes from last.ckpt) ---
echo "=== Step 2: Train diffusion SR (x0) seed=${SEED} — resumes if last.ckpt exists ==="
python -u medvae_diffusion_pipeline/scripts/03_train_diffusion.py \
    --train-latent-dir "${LATENT_DIR}/train_latent" \
    --val-latent-dir "${LATENT_DIR}/valid_latent" \
    --output-dir "${WEIGHT_DIR}" \
    --epochs "${EPOCHS}" --batch-size 8 --lr 1e-4 --timesteps 1000 --seed "${SEED}"

# --- Step 3: Evaluate SR (valid split, decode with KL-f4) ---
echo "=== Step 3: Evaluate SR (valid, T=1000, klf4 decode) seed=${SEED} ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${WEIGHT_DIR}/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --backend klf4 --modality "${MODALITY}" \
    --timesteps 1000 --output-dir "${EVAL_DIR}"

echo "========================================="
echo "Experiment ${EXPERIMENT} (seed=${SEED}) complete at: $(date)"
echo "========================================="
