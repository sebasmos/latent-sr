#!/bin/bash
#SBATCH --job-name=reencode_cxr
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

# Issue #97: Re-encode CXR validation SR images through MedVAE encoder.
# The existing sr_latents in cxr_decoder_finetune/sr_latents/ are training-split
# latents. This job produces proper validation-split SR latents from the 1000
# validation SR pixel images in outputs/experiments/cxr_medvae_s1/sr_images/.
#
# Output: outputs/experiments/cxr_sr_latents_valid/sr_00000.npy ... sr_00999.npy

source "$(dirname "$0")/_env.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

LOG_FILE="slurm/experiments/reencode_cxr_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments outputs/experiments/cxr_sr_latents_valid
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "CXR SR Latent Re-encode (Issue #97)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

echo ""
echo "=== Step 1: Re-encode CXR validation SR images through MedVAE ==="
python -u scripts/reencode_sr_latents.py \
    --sr-image-dir outputs/experiments/cxr_medvae_s1/sr_images \
    --output-dir outputs/experiments/cxr_sr_latents_valid \
    --medvae-model medvae_4_3_2d \
    --modality xray \
    --batch-size 16

echo ""
echo "=== Step 2: Verify output count ==="
N=$(ls outputs/experiments/cxr_sr_latents_valid/sr_*.npy 2>/dev/null | wc -l)
echo "SR latents saved: ${N}  (expected: 1000)"

echo ""
echo "=== Step 3: Re-run multiresolution embedding analysis for CXR ==="
python -u scripts/eval_multiresolution_embedding.py \
    --dataset cxr \
    --sr-latent-dir outputs/experiments/cxr_sr_latents_valid

echo ""
echo "========================================="
echo "Re-encode + embedding analysis complete at: $(date)"
echo "SR latents: outputs/experiments/cxr_sr_latents_valid/"
echo "Results:    outputs/experiments/multiresolution_embedding_cxr/results.json"
echo "Figure:     outputs/figures/multiresolution_embedding_cxr.png"
echo "========================================="
