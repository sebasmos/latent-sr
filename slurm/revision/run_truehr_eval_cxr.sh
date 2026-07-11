#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=truehr_eval
#SBATCH --output=slurm/experiments/truehr_eval_%x_%j.log
#SBATCH --error=slurm/experiments/truehr_eval_%x_%j.log

set -e
source "$(dirname "$0")/../_env.sh"

mkdir -p slurm/experiments

HR_DIR="/orcd/pool/006/lceli_shared/DATASET/mimic-cxr-sr/valid/hr"
MRIU="/orcd/pool/006/lceli_shared/mri-uganda"

# $1 = which model: klf4 | medvae_4_3 | sdvae
case "$1" in
  klf4)
    python -u scripts/eval_diffusion_sr_truehr.py \
      --checkpoint "${MRIU}/weights/diffusion_klf4_cxr_s1_x0/checkpoints/last.ckpt" \
      --latent-dir "${MRIU}/embeddings/klf4_cxr_s1/valid_latent" \
      --hr-image-dir "${HR_DIR}" \
      --backend klf4 --modality xray --timesteps 1000 \
      --output-dir "outputs/experiments/truehr_cxr_klf4"
    ;;
  medvae_4_3)
    python -u scripts/eval_diffusion_sr_truehr.py \
      --checkpoint "${MRIU}/weights/diffusion_medvae_cxr_s1_x0/checkpoints/last.ckpt" \
      --latent-dir "${MRIU}/embeddings/medvae_cxr_s1/valid_latent" \
      --hr-image-dir "${HR_DIR}" \
      --backend medvae --medvae-model medvae_4_3_2d --modality xray --timesteps 1000 \
      --output-dir "outputs/experiments/truehr_cxr_medvae_4_3"
    ;;
  sdvae)
    python -u scripts/eval_diffusion_sr_truehr.py \
      --checkpoint "${MRIU}/weights/diffusion_sdvae_cxr_x0/checkpoints/last.ckpt" \
      --latent-dir "${MRIU}/embeddings/sd_vae_cxr/valid_latent" \
      --hr-image-dir "${HR_DIR}" \
      --backend sd-vae --modality xray --timesteps 1000 \
      --output-dir "outputs/experiments/truehr_cxr_sdvae"
    ;;
  *)
    echo "Usage: sbatch run_truehr_eval_cxr.sh {klf4|medvae_4_3|sdvae}"
    exit 1
    ;;
esac

echo "Done at: $(date)"
