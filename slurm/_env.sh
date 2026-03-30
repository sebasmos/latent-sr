#!/bin/bash

LATENT_SR_REPO_ROOT="${LATENT_SR_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if command -v module >/dev/null 2>&1; then
  module load miniforge/24.3.0-0 >/dev/null 2>&1 || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
  conda activate "${LATENT_SR_CONDA_ENV:-medvae-sr}" >/dev/null 2>&1 || true
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${LATENT_SR_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LATENT_SR_SHARED_ROOT="${LATENT_SR_SHARED_ROOT:-/orcd/pool/006/lceli_shared}"
export LATENT_SR_DATA_ROOT="${LATENT_SR_DATA_ROOT:-${LATENT_SR_SHARED_ROOT}/DATASET}"
export LATENT_SR_MRI_UGANDA_ROOT="${LATENT_SR_MRI_UGANDA_ROOT:-${LATENT_SR_SHARED_ROOT}/mri-uganda}"
export LATENT_SR_EMBEDDINGS_ROOT="${LATENT_SR_EMBEDDINGS_ROOT:-${LATENT_SR_MRI_UGANDA_ROOT}/embeddings}"
export LATENT_SR_WEIGHTS_ROOT="${LATENT_SR_WEIGHTS_ROOT:-${LATENT_SR_MRI_UGANDA_ROOT}/weights}"

cd "${LATENT_SR_REPO_ROOT}" || exit 1
