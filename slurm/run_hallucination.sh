#!/bin/bash
#SBATCH --job-name=hallucination
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/experiments/hallucination_%j.log
#SBATCH --error=slurm/experiments/hallucination_%j.log

# Quantify hallucinated vs lost content for MedVAE SR and SD-VAE SR
# relative to HR ground truth, using AE as noise-floor baseline.
# CPU-only job.

source "$(dirname "$0")/_env.sh"

echo "========================================="
echo "Hallucination Quantification"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="

echo ""
echo "=== MRNet ==="
python -u scripts/eval_hallucination_quantification.py --dataset mrnet

echo ""
echo "=== BraTS ==="
python -u scripts/eval_hallucination_quantification.py --dataset brats

echo ""
echo "=== CXR ==="
python -u scripts/eval_hallucination_quantification.py --dataset cxr

echo ""
echo "========================================="
echo "Hallucination quantification complete at: $(date)"
echo "Results:"
echo "  outputs/experiments/hallucination_mrnet/results.json"
echo "  outputs/experiments/hallucination_brats/results.json"
echo "  outputs/experiments/hallucination_cxr/results.json"
echo "========================================="
