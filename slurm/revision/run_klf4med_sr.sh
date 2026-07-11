#!/bin/bash
#SBATCH --job-name=rev_klf4medsr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# ============================================================
# REVISION R1 (SR side) — medical-KL-f4 full SR pipeline (domain-isolation control).
# BYTE-IDENTICAL to run_klf4_sr.sh (extract -> train diffusion -> SR eval) EXCEPT it
# exports KLF4_CKPT so klf4_vae.load_klf4 uses the MEDICAL fine-tuned kl-f4 weights for
# BOTH latent extraction and SR decoding. Natural run_klf4_sr.sh vs this script therefore
# differ in ONE thing only: the training domain of the kl-f4 autoencoder. Same UNet, raw
# deterministic latents, 100 epochs, T=1000, reference metrics only.
#
# PREREQUISITE: run_klf4med_finetune.sh <dataset> must have produced the medical ckpt.
# Usage:  sbatch --partition=P [--time=T] run_klf4med_sr.sh <dataset>   # mrnet|brats|cxr
# FULLY RESUMABLE (identical guards to run_klf4_sr.sh).
# ============================================================
DATASET="${1:?usage: run_klf4med_sr.sh <dataset> [tag]   (mrnet|brats|cxr)}"
# TAG must match the finetune run's tag. It isolates the encoder checkpoint AND the derived
# latents/weights/eval dirs -- without it, a rerun would silently reuse latents extracted by
# the previous (collapsed) encoder, because the extract step is guarded by dir existence.
TAG="${2:-}"

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

# --- THE domain switch: route the whole kl-f4 path through the MEDICAL checkpoint ---
# Prefer best.ckpt (best validation AE ceiling) over last.ckpt: the ceiling is the quantity
# this experiment measures, and last-epoch weights can be worse than an earlier epoch's.
FT_DIR="${OUT_BASE}/weights/klf4med_${DATASET}${TAG}"
if   [ -f "${FT_DIR}/checkpoints/best.ckpt" ]; then export KLF4_CKPT="${FT_DIR}/checkpoints/best.ckpt"
elif [ -f "${FT_DIR}/checkpoints/last.ckpt" ]; then export KLF4_CKPT="${FT_DIR}/checkpoints/last.ckpt"
else
  echo "MISSING medical ckpt in ${FT_DIR}/checkpoints — run run_klf4med_finetune.sh ${DATASET} <epochs> '${TAG}' first"; exit 2
fi

LATENT_DIR="${OUT_BASE}/embeddings/klf4med_${DATASET}${TAG}_s1"
WEIGHT_DIR="${OUT_BASE}/weights/diffusion_klf4med_${DATASET}${TAG}_s1_x0"
EVAL_DIR="outputs/experiments/revision_capacity/${DATASET}_klf4med${TAG}_s1"

EXPERIMENT="rev_klf4medsr_${DATASET}${TAG}"
LOG_FILE="slurm/experiments/${EXPERIMENT}_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${EVAL_DIR}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Experiment: ${EXPERIMENT} | backend=klf4(MEDICAL) dataset=${DATASET}"
echo "KLF4_CKPT=${KLF4_CKPT}"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

# --- Step 1: Extract MEDICAL KL-f4 latents (per-split completeness guard = resumable) ---
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

# Sanity: medical KL-f4 latent scale vs the pipeline's tuning target (SD-VAE/MedVAE std~8).
python3 - "$LATENT_DIR" "$DATASET" <<'PY'
import sys, glob, numpy as np
ld, ds = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(f"{ld}/train_latent/hr_*.npy"))[:32]
if fs:
    a = np.stack([np.load(f) for f in fs]).astype(np.float64)
    print(f"[latent-scale] klf4med {ds}: dtype={np.load(fs[0]).dtype} shape={a.shape[1:]} "
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

# --- Step 3: Evaluate SR (valid split, decode with MEDICAL KL-f4) ---
echo "=== Step 3: Evaluate SR (valid, T=1000, klf4 MEDICAL decode) ==="
python -u scripts/eval_diffusion_sr.py \
    --checkpoint "${WEIGHT_DIR}/checkpoints/last.ckpt" \
    --latent-dir "${LATENT_DIR}/valid_latent" \
    --backend klf4 --modality "${MODALITY}" \
    --timesteps 1000 --output-dir "${EVAL_DIR}"

echo "========================================="
echo "Experiment ${EXPERIMENT} complete at: $(date)"
echo "========================================="
