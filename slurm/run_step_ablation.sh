#!/bin/bash
#SBATCH --job-name=step_abl
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue

# Issue #14: Inference step ablation T={50,100,250,500,1000} vs quality
# Uses MRNet S1 model (already trained). Runs 5 evals on valid set (120 samples).
# Each T=1000 eval takes ~5 min, T=50 takes ~15 sec. Total ~15 min.

source "$(dirname "$0")/_env.sh"

DATA_BASE="${LATENT_SR_SHARED_ROOT}"
OUT_BASE="${LATENT_SR_MRI_UGANDA_ROOT}"
CHECKPOINT="${OUT_BASE}/weights/diffusion_medvae_mrnet_s1_x0/checkpoints/last.ckpt"
LATENT_DIR="${OUT_BASE}/embeddings/medvae_mrnet_s1/valid_latent"

LOG_FILE="slurm/experiments/step_ablation_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "Issue #14: Inference Step Ablation"
echo "Model: MRNet MedVAE S1 | Backend: medvae"
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "========================================="
nvidia-smi 2>/dev/null | head -5

RESULTS_FILE="outputs/experiments/step_ablation/ablation_results.json"
mkdir -p outputs/experiments/step_ablation

echo '{"experiment": "step_ablation", "model": "mrnet_medvae_s1", "results": {}}' > "$RESULTS_FILE"

for T in 50 100 250 500 1000; do
    echo ""
    echo "============================================"
    echo "=== T=${T} steps ==="
    echo "============================================"
    START_TIME=$(date +%s)

    python -u scripts/eval_diffusion_sr.py \
        --checkpoint "${CHECKPOINT}" \
        --latent-dir "${LATENT_DIR}" \
        --backend medvae \
        --medvae-model medvae_4_3_2d \
        --modality mri \
        --timesteps ${T} \
        --output-dir "outputs/experiments/step_ablation/T${T}"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "T=${T}: Wall time = ${ELAPSED}s"
done

echo ""
echo "========================================="
echo "Step Ablation Summary"
echo "========================================="
for T in 50 100 250 500 1000; do
    if [ -f "outputs/experiments/step_ablation/T${T}/diffusion_eval_results.json" ]; then
        echo "T=${T}:"
        python -c "
import json
with open('outputs/experiments/step_ablation/T${T}/diffusion_eval_results.json') as f:
    r = json.load(f)
sr = r['diffusion_sr']
psnr = sr['psnr_mean']
msssim = sr.get('msssim_mean', 'N/A')
print(f'  PSNR: {psnr:.2f} ± {sr[\"psnr_std\"]:.2f} dB')
if msssim != 'N/A':
    print(f'  MS-SSIM: {msssim:.4f}')
"
    fi
done

echo ""
echo "========================================="
echo "Step ablation complete at: $(date)"
echo "========================================="
