#!/bin/bash
#SBATCH --job-name=freq_analysis
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/experiments/freq_analysis_%j.log
#SBATCH --error=slurm/experiments/freq_analysis_%j.log

# Multi-resolution frequency analysis (wavelet + radial FFT) for all 3 datasets.
# CPU-only job. Compares MedVAE SR vs SD-VAE SR vs HR ground truth.

source "$(dirname "$0")/_env.sh"

echo "========================================="
echo "Frequency Analysis — Wavelet + Radial FFT"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="

echo ""
echo "=== MRNet ==="
python -u scripts/eval_frequency_analysis.py --dataset mrnet

echo ""
echo "=== BraTS ==="
python -u scripts/eval_frequency_analysis.py --dataset brats

echo ""
echo "=== CXR ==="
python -u scripts/eval_frequency_analysis.py --dataset cxr

echo ""
echo "========================================="
echo "Frequency analysis complete at: $(date)"
echo "Results:"
echo "  outputs/experiments/frequency_analysis_mrnet/results.json"
echo "  outputs/experiments/frequency_analysis_brats/results.json"
echo "  outputs/experiments/frequency_analysis_cxr/results.json"
echo "Figures:"
echo "  outputs/figures/frequency_analysis_mrnet.png"
echo "  outputs/figures/frequency_analysis_brats.png"
echo "  outputs/figures/frequency_analysis_cxr.png"
echo "========================================="
