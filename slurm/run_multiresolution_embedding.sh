#!/bin/bash
#SBATCH --job-name=multires_embed
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/experiments/multires_embed_%j.log
#SBATCH --error=slurm/experiments/multires_embed_%j.log

# Multi-resolution latent embedding comparison: SR latents vs HR latents
# at spatial pooling scales 64, 32, 16, 8, 4, 2, 1.
# CPU-only job (latents loaded from disk). 32GB RAM for large latent arrays.

cd /orcd/home/002/sebasmos/orcd/pool/code/latent-sr
module load miniforge/24.3.0-0
conda activate medvae-sr
export PYTHONNOUSERSITE=1
export PYTHONPATH="/orcd/home/002/sebasmos/orcd/pool/code/latent-sr:$PYTHONPATH"

echo "========================================="
echo "Multi-resolution Latent Embedding Analysis"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="

echo ""
echo "=== MRNet ==="
python -u scripts/eval_multiresolution_embedding.py --dataset mrnet

echo ""
echo "=== BraTS ==="
python -u scripts/eval_multiresolution_embedding.py --dataset brats

echo ""
echo "=== CXR ==="
python -u scripts/eval_multiresolution_embedding.py --dataset cxr

echo ""
echo "========================================="
echo "Multi-resolution embedding analysis complete at: $(date)"
echo "Results:"
echo "  outputs/experiments/multiresolution_embedding_mrnet/results.json"
echo "  outputs/experiments/multiresolution_embedding_brats/results.json"
echo "  outputs/experiments/multiresolution_embedding_cxr/results.json"
echo "Figures:"
echo "  outputs/figures/multiresolution_embedding_mrnet.png"
echo "  outputs/figures/multiresolution_embedding_brats.png"
echo "  outputs/figures/multiresolution_embedding_cxr.png"
echo "========================================="
