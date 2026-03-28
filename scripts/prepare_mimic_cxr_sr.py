#!/usr/bin/env python3
"""
Prepare MIMIC-CXR images for SR pipeline with CheXpert pathology labels.

Creates HR (256x256) / LR (64x64) pairs and a labels CSV for downstream
classification evaluation. Uses the official CheXpert label file to sample
images WITH pathology diversity (not just "No Finding").

Pipeline: image JPGs + CheXpert labels → HR/LR pairs + labels.csv
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# 5 most common and clinically relevant CheXpert pathologies
TARGET_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
    "Pneumothorax",
]


def find_image_path(image_root, subject_id, study_id):
    """Find any JPG image for a given subject/study."""
    prefix = f"p{str(subject_id)[:2]}"
    study_dir = image_root / prefix / f"p{subject_id}" / f"s{study_id}"
    if not study_dir.exists():
        return None
    jpgs = list(study_dir.glob("*.jpg"))
    return jpgs[0] if jpgs else None


def load_chexpert_labels(chexpert_csv):
    """Load CheXpert labels, return list of dicts with subject_id, study_id, labels."""
    rows = []
    with open(chexpert_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = {}
            has_positive = False
            for p in TARGET_PATHOLOGIES:
                val = row[p].strip()
                if val == "1.0":
                    labels[p] = 1
                    has_positive = True
                elif val == "0.0":
                    labels[p] = 0
                else:
                    labels[p] = -1  # uncertain or missing
            rows.append({
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "labels": labels,
                "has_positive": has_positive,
                "no_finding": row.get("No Finding", "").strip() == "1.0",
            })
    return rows


def sample_balanced(rows, n_total, seed=42):
    """Sample with pathology diversity: ~50% positive, ~50% no finding."""
    random.seed(seed)
    positives = [r for r in rows if r["has_positive"]]
    negatives = [r for r in rows if r["no_finding"] and not r["has_positive"]]

    n_pos = min(n_total // 2, len(positives))
    n_neg = min(n_total - n_pos, len(negatives))

    sampled = random.sample(positives, n_pos) + random.sample(negatives, n_neg)
    random.shuffle(sampled)
    return sampled


def process_samples(samples, image_root, output_dir, split_name):
    """Process sampled studies: find images, resize, save HR/LR + labels."""
    hr_dir = output_dir / split_name / "hr"
    lr_dir = output_dir / split_name / "lr"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    label_rows = []
    saved = 0

    for row in tqdm(samples, desc=f"{split_name}"):
        img_path = find_image_path(image_root, row["subject_id"], row["study_id"])
        if img_path is None:
            continue

        img = Image.open(img_path).convert("L")  # grayscale
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        hr = img.resize((256, 256), Image.LANCZOS)
        lr = img.resize((64, 64), Image.LANCZOS)

        fname = f"cxr_{saved:05d}.png"
        hr.save(hr_dir / fname)
        lr.save(lr_dir / fname)

        label_row = {"filename": fname, **row["labels"]}
        label_rows.append(label_row)
        saved += 1

    # Save labels CSV
    labels_path = output_dir / split_name / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + TARGET_PATHOLOGIES)
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"  {split_name}: {saved} images saved, labels at {labels_path}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Prepare MIMIC-CXR for SR + classification")
    parser.add_argument("--chexpert-csv", type=Path, required=True,
                        help="Path to mimic-cxr-2.0.0-chexpert.csv")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Root of MIMIC-CXR JPG files (contains p10/ ... p19/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for HR/LR pairs + labels")
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading CheXpert labels...")
    all_rows = load_chexpert_labels(args.chexpert_csv)
    print(f"  Total studies: {len(all_rows)}")
    print(f"  With pathology: {sum(1 for r in all_rows if r['has_positive'])}")
    print(f"  No finding only: {sum(1 for r in all_rows if r['no_finding'] and not r['has_positive'])}")

    # Shuffle and split deterministically
    random.seed(args.seed)
    random.shuffle(all_rows)

    n_total = args.n_train + args.n_val + args.n_test
    pool = all_rows[:n_total * 3]  # oversample pool to account for missing images

    train_samples = sample_balanced(pool[:len(pool)//2], args.n_train, seed=args.seed)
    val_samples = sample_balanced(pool[len(pool)//2:len(pool)*3//4], args.n_val, seed=args.seed+1)
    test_samples = sample_balanced(pool[len(pool)*3//4:], args.n_test, seed=args.seed+2)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in [("train", train_samples), ("valid", val_samples), ("test", test_samples)]:
        process_samples(samples, args.image_root, args.output_dir, split_name)

    # Save summary
    summary = {
        "dataset": "MIMIC-CXR",
        "pathologies": TARGET_PATHOLOGIES,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "hr_size": 256,
        "lr_size": 64,
        "seed": args.seed,
    }
    with open(args.output_dir / "dataset_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Dataset ready at {args.output_dir}")


if __name__ == "__main__":
    main()
