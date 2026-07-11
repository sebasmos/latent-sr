#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=truehr_eval
#SBATCH --output=slurm/experiments/truehr_eval_%x_%j.log
#SBATCH --error=slurm/experiments/truehr_eval_%x_%j.log

set -e
source "$(dirname "$0")/../_env.sh"

mkdir -p slurm/experiments
MRIU="/orcd/pool/006/lceli_shared/mri-uganda"

# $1 = dataset: brats | mrnet
# $2 = model: klf4 | medvae_4_3 | sdvae
case "$1" in
  brats)
    HR_DIR="/orcd/pool/006/lceli_shared/DATASET/brats2023-sr/valid/hr"
    MODALITY="mri"
    case "$2" in
      klf4)       CKPT="${MRIU}/weights/diffusion_klf4_brats_s1_x0/checkpoints/last.ckpt"; LAT="${MRIU}/embeddings/klf4_brats_s1/valid_latent"; BACKEND="klf4"; MVM="" ;;
      medvae_4_3) CKPT="${MRIU}/weights/diffusion_medvae_brats_s1_x0/checkpoints/last.ckpt"; LAT="${MRIU}/embeddings/medvae_brats_s1/valid_latent"; BACKEND="medvae"; MVM="medvae_4_3_2d" ;;
      sdvae)      CKPT="${MRIU}/weights/diffusion_sdvae_brats_x0/checkpoints/last.ckpt"; LAT="${MRIU}/embeddings/sd_vae_brats/valid_latent"; BACKEND="sd-vae"; MVM="" ;;
    esac
    ;;
  mrnet)
    HR_DIR="/orcd/pool/006/lceli_shared/DATASET/mrnetkneemris/MRNet-v1.0-middle/valid/hr"
    MODALITY="mri"
    case "$2" in
      klf4)       CKPT="${MRIU}/weights/diffusion_klf4_mrnet_s1_x0/checkpoints/last.ckpt"; LAT="${MRIU}/embeddings/klf4_mrnet_s1/valid_latent"; BACKEND="klf4"; MVM="" ;;
      medvae_4_3) CKPT="${MRIU}/weights/diffusion_medvae_mrnet_s1_x0/checkpoints/last.ckpt"; LAT="${MRIU}/embeddings/medvae_mrnet_s1/valid_latent"; BACKEND="medvae"; MVM="medvae_4_3_2d" ;;
      sdvae)      CKPT="${MRIU}/weights/diffusion_sdvae_mrnet_x0/checkpoints/last.ckpt"; LAT="${MRIU}/embeddings/sd_vae/phase2/valid_latent"; BACKEND="sd-vae"; MVM="" ;;
    esac
    ;;
  *)
    echo "Usage: sbatch run_truehr_eval_brats_mrnet.sh {brats|mrnet} {klf4|medvae_4_3|sdvae}"
    exit 1
    ;;
esac

EXTRA=""
if [ -n "$MVM" ]; then EXTRA="--medvae-model ${MVM}"; fi

python -u scripts/eval_diffusion_sr_truehr.py \
  --checkpoint "${CKPT}" \
  --latent-dir "${LAT}" \
  --hr-image-dir "${HR_DIR}" \
  --backend "${BACKEND}" ${EXTRA} --modality "${MODALITY}" --timesteps 1000 \
  --output-dir "outputs/experiments/truehr_${1}_${2}"

echo "Done at: $(date)"
