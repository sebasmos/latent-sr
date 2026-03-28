#!/usr/bin/env python3
"""
generate_paper1_composite_figs.py
Assemble composite 3-row figures for Figs 2-5 in Paper 1.

Each figure stacks existing per-dataset outputs vertically:
  Row 1 = MRNet, Row 2 = BraTS, Row 3 = CXR

Outputs:
  outputs/figures/paper1_fig2_composite.png          -- combined visual + diffmap
  outputs/figures/paper1_fig2a_visual_composite.png  -- visual comparison only
  outputs/figures/paper1_fig2b_diffmap_composite.png -- difference maps only
  outputs/figures/paper1_fig3_composite.png
  outputs/figures/paper1_fig4_composite.png
  outputs/figures/paper1_fig4_{ds}_hallucination.png -- per-dataset copies
  outputs/figures/paper1_fig5_composite.png

Usage:
  python scripts/generate_paper1_composite_figs.py

Dependencies: Pillow (PIL)

# NOTE #120: CXR visual comparison may include Bicubic + MedVAE AE panels
# not shown in MRNet/BraTS. TODO: regenerate with consistent method set.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
EXP_DIR = ROOT / "outputs" / "experiments"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Per-figure input paths
# ---------------------------------------------------------------------------

FIG2_VISUAL_INPUTS = {
    "MRNet": FIG_DIR / "visual_comparison_mrnet.png",
    "BraTS": FIG_DIR / "visual_comparison_brats.png",
    "CXR":   FIG_DIR / "cxr_visual_comparison.png",
}

FIG2_DIFFMAP_INPUTS = {
    "MRNet": EXP_DIR / "sr_hr_diffmaps_mrnet" / "summary_mean_diffmaps.png",
    "BraTS": EXP_DIR / "sr_hr_diffmaps_brats" / "summary_mean_diffmaps.png",
    "CXR":   EXP_DIR / "sr_hr_diffmaps_cxr"   / "summary_mean_diffmaps.png",
}

FIG3_INPUTS = {
    "MRNet": FIG_DIR / "frequency_analysis_mrnet.png",
    "BraTS": FIG_DIR / "frequency_analysis_brats.png",
    "CXR":   FIG_DIR / "frequency_analysis_cxr.png",
}

FIG4_INPUTS = {
    "MRNet": EXP_DIR / "hallucination_mrnet" / "summary_bar_chart.png",
    "BraTS": EXP_DIR / "hallucination_brats" / "summary_bar_chart.png",
    "CXR":   EXP_DIR / "hallucination_cxr"   / "summary_bar_chart.png",
}

FIG5_INPUTS = {
    "MRNet": FIG_DIR / "multiresolution_embedding_mrnet.png",
    "BraTS": FIG_DIR / "multiresolution_embedding_brats.png",
    "CXR":   FIG_DIR / "multiresolution_embedding_cxr.png",
}

FIG5_PLACEHOLDER_TEXT = "CXR data pending"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_WIDTH  = 80   # px for the dataset label strip on the left
LABEL_BG     = (240, 240, 240)
LABEL_FG     = (50, 50, 50)
GAP          = 4    # vertical gap (px) between rows
GAP_COLOR    = (200, 200, 200)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_image(path: pathlib.Path) -> Image.Image:
    """Load an RGB image from disk."""
    return Image.open(path).convert("RGB")


def make_placeholder(width: int, height: int, text: str) -> Image.Image:
    """Create a grey placeholder image with centred text."""
    img = Image.new("RGB", (width, height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    # Try to load a basic font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", size=28)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Centre the text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=(60, 60, 60), font=font)
    return img


def make_label_strip(height: int, label: str) -> Image.Image:
    """Create a vertical label strip for a dataset row."""
    strip = Image.new("RGB", (LABEL_WIDTH, height), color=LABEL_BG)
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Bold.ttf", size=20)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Draw text rotated 90 degrees for vertical labels
    # We create a temporary image, write the text, rotate, and paste
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tmp = Image.new("RGB", (tw + 4, th + 4), color=LABEL_BG)
    tmp_draw = ImageDraw.Draw(tmp)
    tmp_draw.text((2, 2), label, fill=LABEL_FG, font=font)
    tmp_rot = tmp.rotate(90, expand=True)

    # Paste centred in the strip
    px = (LABEL_WIDTH - tmp_rot.width) // 2
    py = (height - tmp_rot.height) // 2
    strip.paste(tmp_rot, (max(0, px), max(0, py)))

    return strip


def stack_rows(
    inputs: dict[str, pathlib.Path | None],
    placeholder_text: str = "Pending",
    target_width: int | None = None,
) -> Image.Image:
    """
    Stack rows vertically.

    inputs: ordered dict {label: path_or_None}
    Returns composite PIL image with label strips on the left.
    """
    labels = list(inputs.keys())
    paths  = list(inputs.values())

    # --- Load images (or create placeholders) ---
    images: list[Image.Image] = []
    for path in paths:
        if path is not None and path.exists():
            images.append(load_image(path))
        else:
            # Placeholder; use a default size if we have no reference
            ref_w = target_width or 1200
            ref_h = 400
            images.append(make_placeholder(ref_w, ref_h, placeholder_text))

    # --- Determine common width ---
    if target_width is None:
        common_w = max(img.width for img in images)
    else:
        common_w = target_width

    total_content_width = LABEL_WIDTH + common_w

    # --- Resize images to common width (preserve aspect ratio) ---
    resized: list[Image.Image] = []
    for img in images:
        if img.width != common_w:
            scale = common_w / img.width
            new_h = int(img.height * scale)
            img = img.resize((common_w, new_h), Image.LANCZOS)
        resized.append(img)

    # --- Compute total height ---
    total_h = sum(img.height for img in resized) + GAP * (len(resized) - 1)

    canvas = Image.new("RGB", (total_content_width, total_h), color=(255, 255, 255))

    y_offset = 0
    for i, (label, img) in enumerate(zip(labels, resized)):
        # Label strip
        strip = make_label_strip(img.height, label)
        canvas.paste(strip, (0, y_offset))
        # Image
        canvas.paste(img, (LABEL_WIDTH, y_offset))
        y_offset += img.height
        if i < len(resized) - 1:
            # Gap
            gap_strip = Image.new("RGB", (total_content_width, GAP), color=GAP_COLOR)
            canvas.paste(gap_strip, (0, y_offset))
            y_offset += GAP

    return canvas


def stack_visual_and_diffmap(
    visual_inputs: dict[str, pathlib.Path | None],
    diffmap_inputs: dict[str, pathlib.Path | None],
    placeholder_text: str = "Pending",
    target_width: int | None = None,
) -> Image.Image:
    """
    For each dataset (label), stack visual_comparison (top) and diffmap (bottom)
    as a pair, then stack all dataset pairs vertically with a label strip on the left.

    The result has 6 sub-rows: visual+diffmap for MRNet, visual+diffmap for BraTS,
    visual+diffmap for CXR (in insertion order).
    """
    labels = list(visual_inputs.keys())

    # --- Load all images (or create placeholders) ---
    def load_or_placeholder(path: pathlib.Path | None, ref_w: int, ref_h: int) -> Image.Image:
        if path is not None and path.exists():
            return load_image(path)
        return make_placeholder(ref_w, ref_h, placeholder_text)

    # First pass: load to find common width
    vis_imgs:  list[Image.Image] = []
    diff_imgs: list[Image.Image] = []
    for label in labels:
        vis_imgs.append(load_or_placeholder(visual_inputs[label], 1200, 400))
        diff_imgs.append(load_or_placeholder(diffmap_inputs[label], 1200, 300))

    # --- Determine common content width ---
    if target_width is None:
        common_w = max(img.width for img in vis_imgs + diff_imgs)
    else:
        common_w = target_width

    total_content_width = LABEL_WIDTH + common_w

    # --- Resize all images to common width (preserve aspect ratio) ---
    def resize_to_width(img: Image.Image, w: int) -> Image.Image:
        if img.width != w:
            scale = w / img.width
            new_h = int(img.height * scale)
            img = img.resize((w, new_h), Image.LANCZOS)
        return img

    vis_imgs  = [resize_to_width(img, common_w) for img in vis_imgs]
    diff_imgs = [resize_to_width(img, common_w) for img in diff_imgs]

    # --- Build dataset pairs (visual on top, diffmap below, inner gap) ---
    INNER_GAP = 2   # small gap between visual and diffmap within a dataset pair
    OUTER_GAP = GAP  # gap between dataset pairs

    pair_heights = [
        v.height + INNER_GAP + d.height
        for v, d in zip(vis_imgs, diff_imgs)
    ]
    total_h = sum(pair_heights) + OUTER_GAP * (len(labels) - 1)

    canvas = Image.new("RGB", (total_content_width, total_h), color=(255, 255, 255))

    y_offset = 0
    for i, label in enumerate(labels):
        pair_h = pair_heights[i]

        # Label strip spanning the whole dataset pair (visual + inner gap + diffmap)
        strip = make_label_strip(pair_h, label)
        canvas.paste(strip, (0, y_offset))

        # Visual comparison sub-row
        canvas.paste(vis_imgs[i], (LABEL_WIDTH, y_offset))
        y_offset += vis_imgs[i].height

        # Inner gap (grey line between visual and diffmap)
        inner_gap_strip = Image.new("RGB", (total_content_width, INNER_GAP), color=GAP_COLOR)
        canvas.paste(inner_gap_strip, (0, y_offset))
        y_offset += INNER_GAP

        # Diffmap sub-row
        canvas.paste(diff_imgs[i], (LABEL_WIDTH, y_offset))
        y_offset += diff_imgs[i].height

        # Outer gap between dataset pairs
        if i < len(labels) - 1:
            outer_gap_strip = Image.new("RGB", (total_content_width, OUTER_GAP), color=GAP_COLOR)
            canvas.paste(outer_gap_strip, (0, y_offset))
            y_offset += OUTER_GAP

    return canvas


# ---------------------------------------------------------------------------
# Generate all four composites
# ---------------------------------------------------------------------------

print("Generating Fig 2 composite (visual comparison + diffmaps)...")
fig2 = stack_visual_and_diffmap(FIG2_VISUAL_INPUTS, FIG2_DIFFMAP_INPUTS)
out2 = FIG_DIR / "paper1_fig2_composite.png"
fig2.save(str(out2), dpi=(150, 150))
print(f"  -> saved {out2}  ({fig2.width} x {fig2.height} px)")

# Visual-only composite
print("Generating Fig 2a visual-only composite...")
fig2_visual = stack_rows(FIG2_VISUAL_INPUTS)
fig2_visual.save(str(FIG_DIR / "paper1_fig2a_visual_composite.png"), dpi=(150, 150))
print(f"  -> saved {FIG_DIR / 'paper1_fig2a_visual_composite.png'}  ({fig2_visual.width} x {fig2_visual.height} px)")

# Diffmap-only composite
print("Generating Fig 2b diffmap-only composite...")
fig2_diffmap = stack_rows(FIG2_DIFFMAP_INPUTS)
fig2_diffmap.save(str(FIG_DIR / "paper1_fig2b_diffmap_composite.png"), dpi=(150, 150))
print(f"  -> saved {FIG_DIR / 'paper1_fig2b_diffmap_composite.png'}  ({fig2_diffmap.width} x {fig2_diffmap.height} px)")

print("Generating Fig 3 composite (frequency analysis)...")
fig3 = stack_rows(FIG3_INPUTS)
out3 = FIG_DIR / "paper1_fig3_composite.png"
fig3.save(str(out3), dpi=(150, 150))
print(f"  -> saved {out3}  ({fig3.width} x {fig3.height} px)")

print("Generating Fig 4 composite (hallucination)...")
fig4 = stack_rows(FIG4_INPUTS)
out4 = FIG_DIR / "paper1_fig4_composite.png"
fig4.save(str(out4), dpi=(150, 150))
print(f"  -> saved {out4}  ({fig4.width} x {fig4.height} px)")

# Per-dataset hallucination figures
import shutil
print("Copying per-dataset hallucination figures...")
for ds, src in FIG4_INPUTS.items():
    if src.exists():
        shutil.copy(src, FIG_DIR / f"paper1_fig4_{ds.lower()}_hallucination.png")
        print(f"  -> copied {ds} hallucination figure")

print("Generating Fig 5 composite (multiresolution embedding)...")
fig5 = stack_rows(FIG5_INPUTS, placeholder_text=FIG5_PLACEHOLDER_TEXT)
out5 = FIG_DIR / "paper1_fig5_composite.png"
fig5.save(str(out5), dpi=(150, 150))
print(f"  -> saved {out5}  ({fig5.width} x {fig5.height} px)")

print("\nDone. All composite figures generated.")
