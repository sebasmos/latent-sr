#!/bin/bash
#SBATCH --job-name=rev_klf4med_smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ============================================================
# REVISION R1 — SMOKE TEST (gate the full runs on this). Proves, on GPU, in <15 min:
#   (1) the medical-KL-f4 fine-tune loop trains a few steps (BraTS, 64 imgs, 10 steps),
#   (2) it saves a checkpoint and runs the AE ceiling on a tiny valid subset,
#   (3) --eval-only reloads that checkpoint,
#   (4) the downstream wiring resolves it: klf4_vae.load_klf4 picks up KLF4_CKPT and
#       loads the medical weights with 0 missing/unexpected keys (what 02_extract /
#       eval_diffusion_sr do). This is the whole domain-switch contract.
# NOTHING here is a scientific result — tiny data / 1 epoch / 10 steps only.
# ============================================================
source "$(dirname "$0")/../_env.sh"
export PYTHONPATH="${LATENT_SR_REPO_ROOT}/latent-diffusion:$PYTHONPATH"

DATA_ROOT="${LATENT_SR_DATA_ROOT}/brats2023-sr"
OUT_DIR="/tmp/klf4med_smoke_${SLURM_JOB_ID}"      # throwaway; not the real weights dir
LOG="slurm/experiments/rev_klf4med_smoke_${SLURM_JOB_ID}.log"
mkdir -p slurm/experiments "${OUT_DIR}"
exec > >(tee -a "$LOG") 2>&1

echo "=== R1 SMOKE | Job ${SLURM_JOB_ID} | $(date) ==="
nvidia-smi 2>/dev/null | head -5

echo "--- (1)+(2) fine-tune loop (10 steps) + ceiling ---"
python -u slurm/revision/finetune_klf4.py \
    --dataset brats --data-root "${DATA_ROOT}" --out-dir "${OUT_DIR}" \
    --epochs 1 --batch-size 4 --max-train-images 64 --max-steps 10 --max-val-images 16 \
    --lr 1.5e-5 --kl-weight 1e-6 --seed 42 || { echo "SMOKE FAIL: train loop"; exit 1; }

SMOKE_CKPT="${OUT_DIR}/checkpoints/last.ckpt"
echo "--- (3) --eval-only reload of smoke ckpt ---"
python -u slurm/revision/finetune_klf4.py --eval-only \
    --dataset brats --data-root "${DATA_ROOT}" \
    --init-ckpt "${SMOKE_CKPT}" --out-dir "${OUT_DIR}" --max-val-images 16 \
    || { echo "SMOKE FAIL: eval-only"; exit 1; }

echo "--- (4) downstream wiring: klf4_vae.load_klf4 picks up KLF4_CKPT ---"
KLF4_CKPT="${SMOKE_CKPT}" python -u - <<'PY' || { echo "SMOKE FAIL: downstream load"; exit 1; }
import os, sys, torch
sys.path.insert(0, os.path.join(os.environ["LATENT_SR_REPO_ROOT"], "slurm", "revision"))
from klf4_vae import load_klf4
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = load_klf4(dev)                       # must print [medical] and load cleanly
x = torch.randn(1, 3, 256, 256, device=dev)
z = m.encode(x); r = m.decode(z)
assert tuple(z.shape) == (1, 3, 64, 64), z.shape
assert tuple(r.shape) == (1, 3, 256, 256), r.shape
print("downstream OK: medical kl-f4 encodes/decodes at 3x64x64")
PY

rm -rf "${OUT_DIR}"
echo "=== SMOKE PASS at $(date) — safe to launch full runs ==="
