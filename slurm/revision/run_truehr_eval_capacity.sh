#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=truehr_cap
#SBATCH --output=slurm/experiments/truehr_cap_%x_%j.log
#SBATCH --error=slurm/experiments/truehr_cap_%x_%j.log

set -e
source "$(dirname "$0")/../_env.sh"
mkdir -p slurm/experiments
MRIU="/orcd/pool/006/lceli_shared/mri-uganda"

# $1 = vae: medvae_8_4 | medvae_4_1     $2 = dataset: brats | cxr | mrnet
VAE="$1"; DS="$2"
case "$DS" in
  brats) HR="/orcd/pool/006/lceli_shared/DATASET/brats2023-sr/valid/hr"; MOD="mri" ;;
  mrnet) HR="/orcd/pool/006/lceli_shared/DATASET/mrnetkneemris/MRNet-v1.0-middle/valid/hr"; MOD="mri" ;;
  cxr)   HR="/orcd/pool/006/lceli_shared/DATASET/mimic-cxr-sr/valid/hr"; MOD="xray" ;;
  *) echo "bad dataset $DS"; exit 1 ;;
esac
MVM="${VAE}_2d"   # medvae_8_4_2d | medvae_4_1_2d
CKPT="${MRIU}/weights/diffusion_${VAE}_${DS}_s1_x0/checkpoints/last.ckpt"
LAT="${MRIU}/embeddings/${VAE}_${DS}_s1/valid_latent"

python -u scripts/eval_diffusion_sr_truehr.py \
  --checkpoint "${CKPT}" --latent-dir "${LAT}" --hr-image-dir "${HR}" \
  --backend medvae --medvae-model "${MVM}" --modality "${MOD}" --timesteps 1000 \
  --output-dir "outputs/experiments/truehr_${DS}_${VAE}"

echo "Done ${VAE} ${DS} at: $(date)"
