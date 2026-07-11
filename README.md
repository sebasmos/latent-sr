# Latent Geometry Shapes Fidelity in Diffusion-Based Medical Image Super-Resolution

**Reproducibility Package**

[![arXiv](https://img.shields.io/badge/arXiv-2604.12152-b31b1b.svg)](https://arxiv.org/abs/2604.12152)
[![HF Weights](https://img.shields.io/badge/🤗%20Weights-latent--sr--weights-yellow.svg)](https://huggingface.co/sebasmos/latent-sr-weights)
[![HF Embeddings](https://img.shields.io/badge/🤗%20Embeddings-latent--sr--embeddings-yellow.svg)](https://huggingface.co/datasets/sebasmos/latent-sr-embeddings)

Latent diffusion models for medical image super-resolution inherit variational autoencoders
designed for natural photographs. In a controlled single-variable comparison — holding the UNet,
objective, schedule, and evaluation protocol fixed and swapping only the VAE — replacing the
Stable Diffusion VAE with the domain-specific MedVAE improves true-HR PSNR by **+1.92 to +3.60 dB**
across knee MRI, brain MRI, and chest X-ray. Capacity-matched controls and a natural-image VAE at
MedVAE's identical latent geometry (KL-f4) show the advantage is driven primarily by **latent
geometry** (resolution and capacity), with medical-domain pretraining a real but secondary benefit.

This repository contains the code, tests, and job scripts to reproduce every table and figure in
the paper. Trained weights and cached latents are released on the Hugging Face Hub (badges above).

## Reproduce

See **[docs/REPRODUCE.md](docs/REPRODUCE.md)** for the full step-by-step guide: environment setup,
dataset access, weight/latent download, and the exact commands for every table and figure.

```bash
conda env create -f environment.yml && conda activate medvae-sr
export LATENT_SR_SHARED_ROOT=/path/to/shared-root
bash reproduce_all.sh --figures-only          # regenerate figures from released results
bash tests/run_all_tests.sh                    # sanity + golden-number checks
```

## 🙏 Acknowledgements

This work used computational resources from MIT ORCD (NVIDIA L40S / A100 80 GB / H200 130 GB).

This work was supported by the Google Cloud Research Credits program under award number GCP19980904.

## 📚 Citation

```bibtex
@misc{cajas2026latentgeometry,
  title  = {Latent Geometry Shapes Fidelity in Diffusion-Based Medical Image Super-Resolution},
  author = {Sebastian Cajas and Ashaba Judith and Rahul Gorijavolu and Sahil Kapadia and
            Hillary Clinton Kasimbazi and Leo Kinyera and Emmanuel Paul Kwesiga and
            Sri Sri Jaithra Varma Manthena and Luis Filipe Nakayama and Ninsiima Doreen and
            Leo Anthony Celi},
  year   = {2026},
  eprint = {2604.12152},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url    = {https://arxiv.org/abs/2604.12152}
}
```
