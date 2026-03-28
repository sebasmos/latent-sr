#!/bin/bash
#SBATCH --job-name=sr_hr_diffmaps
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/experiments/sr_hr_diffmaps_%j.log
#SBATCH --error=slurm/experiments/sr_hr_diffmaps_%j.log

# Compute |SR - HR| and (SR - HR) difference maps for all 3 datasets.
# Compares MedVAE SR, SD-VAE SR, and AE all vs HR ground truth.
# CPU-only job.

cd /orcd/home/002/sebasmos/orcd/pool/code/latent-sr
module load miniforge/24.3.0-0
conda activate medvae-sr
export PYTHONNOUSERSITE=1
export PYTHONPATH="/orcd/home/002/sebasmos/orcd/pool/code/latent-sr:$PYTHONPATH"

echo "========================================="
echo "SR-HR Difference Maps (vs ground truth)"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="

echo ""
echo "=== MRNet ==="
python -u scripts/eval_sr_hr_diffmaps.py --dataset mrnet

echo ""
echo "=== BraTS ==="
python -u scripts/eval_sr_hr_diffmaps.py --dataset brats

echo ""
echo "=== CXR ==="
python -u scripts/eval_sr_hr_diffmaps.py --dataset cxr

echo ""
echo "========================================="
echo "SR-HR diffmaps complete at: $(date)"
echo "Results:"
echo "  outputs/experiments/sr_hr_diffmaps_mrnet/results.json"
echo "  outputs/experiments/sr_hr_diffmaps_brats/results.json"
echo "  outputs/experiments/sr_hr_diffmaps_cxr/results.json"
echo "========================================="
