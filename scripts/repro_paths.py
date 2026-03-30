from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARED_ROOT = Path("/orcd/pool/006/lceli_shared")
DATASET_SUBDIRS = {
    "mrnet": Path("mrnetkneemris/MRNet-v1.0-middle"),
    "brats": Path("brats2023-sr"),
    "cxr": Path("mimic-cxr-sr"),
}


def _dataset_key(dataset: str) -> str:
    key = dataset.lower()
    if key not in DATASET_SUBDIRS:
        raise KeyError(f"Unsupported dataset: {dataset}")
    return key


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def repo_root() -> Path:
    return REPO_ROOT


def shared_root() -> Path:
    return env_path("LATENT_SR_SHARED_ROOT", DEFAULT_SHARED_ROOT)


def data_root() -> Path:
    return env_path("LATENT_SR_DATA_ROOT", shared_root() / "DATASET")


def mri_uganda_root() -> Path:
    return env_path("LATENT_SR_MRI_UGANDA_ROOT", shared_root() / "mri-uganda")


def embeddings_root() -> Path:
    return env_path("LATENT_SR_EMBEDDINGS_ROOT", mri_uganda_root() / "embeddings")


def weights_root() -> Path:
    return env_path("LATENT_SR_WEIGHTS_ROOT", mri_uganda_root() / "weights")


def outputs_root() -> Path:
    return env_path("LATENT_SR_OUTPUTS_ROOT", repo_root() / "outputs")


def figures_root() -> Path:
    return env_path("LATENT_SR_FIGURES_ROOT", repo_root() / "figures")


def paper_figures_root() -> Path:
    return env_path("LATENT_SR_PAPER_FIGURES_ROOT", repo_root() / "figures-paper-1")


def dataset_dir(dataset: str) -> Path:
    return data_root() / DATASET_SUBDIRS[_dataset_key(dataset)]


def dataset_hr_dir(dataset: str, split: str = "valid") -> Path:
    return dataset_dir(dataset) / split / "hr"


def dataset_lr_dir(dataset: str, split: str = "valid") -> Path:
    return dataset_dir(dataset) / split / "lr"
