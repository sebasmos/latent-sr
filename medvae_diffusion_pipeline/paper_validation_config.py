"""
Paper-Specific Validation Configuration for MedVAE + Diffusion SR

This configuration matches the exact validation methodology from:
"MedVAE: A Family of Large-Scale Medical Image Autoencoders" (arXiv:2502.14753)

Key Methodology from Paper:
========================

1. **2D Perceptual Quality Evaluation:**
   - Random sample of 1000 images per image type
   - 4 different random seeds
   - Metrics: PSNR and MS-SSIM
   - Report: mean ± std across 4 runs

2. **3D Perceptual Quality Evaluation:**
   - Single random sample of 100 images per image type
   - Single run (no multiple seeds)
   - Metrics: PSNR and MS-SSIM

3. **CAD Task Evaluation:**
   - 3 different random seeds
   - Metric: AUROC
   - Report: mean ± std across 3 runs
   - Data splits vary by task (e.g., 95/5 train/test)

4. **Manual Reader Study:**
   - 3 expert radiologists
   - 50 unique images (randomly sampled)
   - 5-point Likert scale (-2 to 2)
   - Report: mean scores with 95% confidence intervals
"""

from medvae_diffusion_pipeline.validation_framework import ValidationConfig


# Configuration for 2D Perceptual Quality Evaluation (matching paper)
CONFIG_2D_PERCEPTUAL = ValidationConfig(
    validation_type="held_out",
    n_seeds=4,  # Paper uses 4 random seeds for 2D
    test_size=0.0,  # No separate test set for perceptual quality
    val_size=0.0,  # Evaluate on 1000 random samples directly
    batch_size=8,
    num_workers=4,
    device="cuda",
    metrics=["psnr", "ms_ssim"],  # Paper uses PSNR and MS-SSIM
    save_predictions=True,
    output_dir="./validation_results/2d_perceptual",
)

# Configuration for 3D Perceptual Quality Evaluation (matching paper)
CONFIG_3D_PERCEPTUAL = ValidationConfig(
    validation_type="held_out",
    n_seeds=1,  # Paper uses single run for 3D
    test_size=0.0,
    val_size=0.0,  # Evaluate on 100 random samples directly
    batch_size=4,  # Smaller batch for 3D volumes
    num_workers=4,
    device="cuda",
    metrics=["psnr", "ms_ssim"],
    save_predictions=True,
    output_dir="./validation_results/3d_perceptual",
)

# Configuration for CAD Tasks (matching paper)
CONFIG_CAD_TASKS = ValidationConfig(
    validation_type="held_out",
    n_seeds=3,  # Paper uses 3 random seeds for CAD tasks
    test_size=0.05,  # Example: 95/5 split from paper
    val_size=0.0,  # No separate validation set mentioned
    batch_size=20,  # Paper uses batch size 20 for latents, 10 for originals
    num_workers=4,
    device="cuda",
    metrics=["auroc"],  # Classification tasks use AUROC
    save_predictions=False,
    output_dir="./validation_results/cad_tasks",
)

# Sample sizes from paper
SAMPLE_SIZES = {
    "2d_perceptual": 1000,  # 1000 images per image type for 2D
    "3d_perceptual": 100,  # 100 images per image type for 3D
    "reader_study": 50,  # 50 images for manual reader study
}

# Image types evaluated in paper
IMAGE_TYPES_2D = [
    "mammograms",  # FFDM
    "chest_xrays",  # Chest X-rays
    "musculoskeletal_xrays",  # Musculoskeletal X-rays
    "wrist_xrays_fg",  # Wrist X-rays (fine-grained)
]

IMAGE_TYPES_3D = [
    "brain_mris",  # Brain MRIs
    "head_cts",  # Head CTs
    "abdomen_cts",  # Abdomen CTs
    "ts_cts",  # TotalSegmentator CTs (various anatomies)
    "lung_cts",  # Lung CTs
    "knee_mris",  # Knee MRIs
]

# CAD tasks from paper
CAD_TASKS = {
    "2d": [
        "malignancy_detection",  # CMMD
        "calcification_detection",  # CMMD
        "birads_classification",  # VinDR-Mammo (5 classes)
        "bone_age_prediction",  # RSNA Bone Age (20 classes)
        "wrist_fracture_detection",  # GRAZPEDWRI-DX
    ],
    "3d": [
        "spine_fracture_detection",  # VerSe
        "head_fracture_detection",  # CQ500
        "acl_meniscal_tear_detection",  # MRNet
    ],
}

# Downsizing factors tested in paper
DOWNSIZING_FACTORS = {
    "2d": [16, 64],  # f=16 and f=64 for 2D
    "3d": [64, 512],  # f=64 and f=512 for 3D
}

# Latent channels tested in paper
LATENT_CHANNELS = {
    "f16": [1, 3],  # C=1 and C=3 for f=16
    "f64": [1, 4],  # C=1 and C=4 for f=64
    "f512": [1],  # C=1 for f=512
}


def get_paper_config(task_type: str = "2d_perceptual"):
    """
    Get validation configuration matching the paper methodology.

    Args:
        task_type: One of "2d_perceptual", "3d_perceptual", or "cad_tasks"

    Returns:
        ValidationConfig matching paper methodology
    """
    configs = {
        "2d_perceptual": CONFIG_2D_PERCEPTUAL,
        "3d_perceptual": CONFIG_3D_PERCEPTUAL,
        "cad_tasks": CONFIG_CAD_TASKS,
    }

    if task_type not in configs:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Must be one of {list(configs.keys())}"
        )

    return configs[task_type]


# Example usage:
if __name__ == "__main__":
    import json

    print("=" * 80)
    print("Paper Validation Configurations")
    print("=" * 80)

    for task_type in ["2d_perceptual", "3d_perceptual", "cad_tasks"]:
        config = get_paper_config(task_type)
        print(f"\n{task_type.upper()}:")
        print(json.dumps(config.to_dict(), indent=2))

    print("\n" + "=" * 80)
    print("Sample Sizes:")
    print(json.dumps(SAMPLE_SIZES, indent=2))

    print("\n" + "=" * 80)
    print("Evaluated Image Types:")
    print(f"\n2D: {IMAGE_TYPES_2D}")
    print(f"\n3D: {IMAGE_TYPES_3D}")

    print("\n" + "=" * 80)
    print("CAD Tasks:")
    print(json.dumps(CAD_TASKS, indent=2))
