#!/bin/bash
#SBATCH --job-name=rev_klf4sr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# ============================================================
# REVISION C1 (SR side) — KL-f4 natural VAE (3x64x64) full SR pipeline.
# The last unmeasured cell of index.md §1c: the SR domain gap at matched LARGE
# geometry. AE ceiling already showed domain@64² = +0.74..+1.17 dB; SR amplifies
# domain (MRNet: +2.47 SR vs +0.94 AE), so this measures whether the SR domain gap
# at 3x64x64 is larger than the AE gap — the one result that could move the verdict
# back toward the domain title.
#
# IDENTICAL pipeline to medvae_4_3 / SD-VAE (swap only the VAE): raw deterministic
# latents, same UNet, same 100 epochs / T=1000. Reference metrics only (Leo).
#
# Usage:  sbatch --partition=P [--time=T] run_klf4_sr.sh <dataset>   # dataset: mrnet|brats|cxr
#
# FULLY RESUMABLE (jobs break often on preemptable):
#   * extract  -> skips a split only if its filenames.txt count matches the source
#                 image count; a partial/interrupted split is re-extracted from scratch.
#   * training -> 03_train_diffusion auto-resumes from checkpoints/last.ckpt.
#   * SR eval  -> idempotent (overwrites results json).
#   * --requeue -> on preemption SLURM restarts this script; the above make it a no-op
#                 up to the last completed stage.
# ============================================================

DATASET="${1:?usage: run_klf4_sr.sh <dataset>   (mrnet|brats|cxr)}"

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

LATENT_DIR="${OUT_BASE}/embeddings/klf4_${DATASET}_s1"
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_klf4_${DATASET}_s1_x0"
EVAL_DIR="outputs/experiments/revision_capacity/${DATASET}_klf4_s1"

EXPERIMENT="rev_klf4sr_${DATASET}"
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${EVAL_DIR}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT} | backend=klf4 dataset=${DATASET}"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

# --- Step 1: Extract KL-f4 latents (per-split completeness guard = resumable) ---
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

# Sanity: KL-f4 latent scale vs the pipeline's tuning target (SD-VAE/MedVAE std~8).
python3 - "$LATENT_DIR" "$DATASET" <<'PY'
import sys, glob, numpy as np
ld, ds = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{ld}/train_latent/hr_*.npy"))[:32]
if fs:
    # upcast to float64 so std() does not overflow if latents were stored as fp16
    a = np.stack([np.load(f) for f in fs]).astype(np.float64)
    print(f"[latent-scale] klf4 {ds}: dtype={np.load(fs[0]).dtype} shape={a.shape[1:]} "
          f"mean={a.mean():+.3f} std={a.std():.3f} range=[{a.min():.1f},{a.max():.1f}] "
          f"inf={np.isinf(a).sum()} nan={np.isnan(a).sum()}  "
          f"(SD-VAE/MedVAE ref std~8; large deviation => scale confound)", flush=True)
PY

# --- Step 2: Train diffusion SR (auto-resumes from last.ckpt) ---
echo "=== Step 2: Train diffusion SR (x0) — resumes if last.ckpt exists ==="
python -u medvae_diffusion_pipeline/scripts/03_train_diffusion.py \
    --train-latent-dir "${LATENT_DIR}/train_latent" \
    --val-latent-dir "${LATENT_DIR}/valid_latent" \
    --output-dir "${WEIGHT_DIR}" \
    --epochs 100 --batch-size 8 --lr 1e-4 --timesteps 1000 --seed 42

# --- Step 3: Evaluate SR (valid split, decode with KL-f4) ---
echo "=== Step 3: Evaluate SR (valid, T=1000, klf4 decode) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${WEIGHT_DIR}/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --backend klf4 --modality "${MODALITY}" \
    --timesteps 1000 --output-dir "${EVAL_DIR}"

echo "========================================="
echo "Experiment ${EXPERIMENT} complete at: $(date)"
echo "========================================="
