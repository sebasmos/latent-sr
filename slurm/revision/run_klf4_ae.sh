#!/bin/bash
#SBATCH --job-name=rev_klf4ae
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# REVISION C1: KL-f4 (natural VAE, 3x64x64) AE reconstruction ceiling vs medvae_4_3.
# The natural-side matched-geometry control (index.md §1c). Forward passes only.

source "$(dirname "$0")/../_env.sh"
export PYTHONPATH="${LATENT_SR_REPO_ROOT}/latent-diffusion:$PYTHONPATH"

LOG=slurm/experiments/rev_klf4_ae_${SLURM_JOB_ID}.log
mkdir -p slurm/experiments outputs/experiments/revision_capacity/klf4_ae
exec > >(tee -a "$LOG") 2>&1

echo "=== KL-f4 AE ceiling | Job $SLURM_JOB_ID | $(date) ==="
nvidia-smi 2>/dev/null | head -3
# full validation sets (no limit)
python -u slurm/revision/eval_klf4_ae.py | tee outputs/experiments/revision_capacity/klf4_ae/results.txt
echo "=== done $(date) ==="
