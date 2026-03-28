"""
Plot AE ceiling PSNR vs SR PSNR scatter plot.
Demonstrates the ceiling principle as a quantitative predictor.
Closes GitHub issue #209.
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import pearsonr

# ── Nature style settings ────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Data ─────────────────────────────────────────────────────────────────────
data = [
    # (vae,     dataset,  ae_ceiling,  sr_psnr)
    ("SD-VAE",  "MRNet",  23.92,       22.34),
    ("SD-VAE",  "BraTS",  31.39,       23.51),
    ("SD-VAE",  "CXR",    32.02,       25.58),
    ("MedVAE",  "MRNet",  27.85,       25.26),
    ("MedVAE",  "BraTS",  37.87,       26.42),
    ("MedVAE",  "CXR",    36.93,       28.87),
]

vae_names   = [d[0] for d in data]
ds_names    = [d[1] for d in data]
ae_ceiling  = np.array([d[2] for d in data])
sr_psnr     = np.array([d[3] for d in data])

# ── Pearson r and R² ─────────────────────────────────────────────────────────
r_val, p_val = pearsonr(ae_ceiling, sr_psnr)
r2_val = r_val ** 2
print(f"Pearson r = {r_val:.4f},  R² = {r2_val:.4f},  p = {p_val:.4g}")

# ── OLS regression ────────────────────────────────────────────────────────────
coeffs = np.polyfit(ae_ceiling, sr_psnr, 1)
poly   = np.poly1d(coeffs)
x_line = np.linspace(ae_ceiling.min() - 1, ae_ceiling.max() + 1, 200)
y_line = poly(x_line)

# 95 % CI via bootstrapping (n = 5000)
rng = np.random.default_rng(42)
n_boot = 5000
y_boot = np.zeros((n_boot, len(x_line)))
for i in range(n_boot):
    idx  = rng.integers(0, len(ae_ceiling), len(ae_ceiling))
    c_b  = np.polyfit(ae_ceiling[idx], sr_psnr[idx], 1)
    y_boot[i] = np.polyval(c_b, x_line)
ci_lo = np.percentile(y_boot, 2.5,  axis=0)
ci_hi = np.percentile(y_boot, 97.5, axis=0)

# ── Visual encoding ───────────────────────────────────────────────────────────
vae_colors  = {"MedVAE": "#1565C0", "SD-VAE": "#C62828"}
ds_markers  = {"MRNet": "o", "BraTS": "s", "CXR": "^"}
MARKER_SIZE = 120

# ── Figure ────────────────────────────────────────────────────────────────────
fig_w_cm, fig_h_cm = 9.0, 7.0
fig, ax = plt.subplots(
    figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54),
    constrained_layout=True,
)

# regression line + CI band
ax.fill_between(x_line, ci_lo, ci_hi, alpha=0.15, color="grey", linewidth=0)
ax.plot(x_line, y_line, color="black", linewidth=1.0, zorder=2)

# scatter points
for vae, ds, ae, sr in data:
    ax.scatter(
        ae, sr,
        c=vae_colors[vae],
        marker=ds_markers[ds],
        s=MARKER_SIZE,
        zorder=5,
        linewidths=0.5,
        edgecolors="white",
    )

# annotation
ax.text(
    0.04, 0.97,
    f"$R^2 = {r2_val:.2f}$,  $r = {r_val:.2f}$",
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="left",
)

# axes labels
ax.set_xlabel("AE Ceiling PSNR (dB)  \u2192", labelpad=3)
ax.set_ylabel("SR Output PSNR (dB)  \u2192", labelpad=3)
ax.set_title(None)

# ── Legend ────────────────────────────────────────────────────────────────────
# VAE colour patches
vae_handles = [
    mpatches.Patch(color=vae_colors["MedVAE"], label="MedVAE SR"),
    mpatches.Patch(color=vae_colors["SD-VAE"],  label="SD-VAE SR"),
]
# Dataset shape handles
ds_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
           markersize=6, label="MRNet"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
           markersize=6, label="BraTS"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
           markersize=6, label="CXR"),
]
leg = ax.legend(
    handles=vae_handles + ds_handles,
    fontsize=7,
    loc="lower right",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.8",
    handlelength=1.4,
)

ax.tick_params(axis="both", which="major", length=3, pad=2)
ax.spines[["top", "right"]].set_visible(False)

# ── Output ────────────────────────────────────────────────────────────────────
out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "outputs", "figures"
)
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)

png_path = os.path.join(out_dir, "ae_ceiling_correlation.png")
pdf_path = os.path.join(out_dir, "ae_ceiling_correlation.pdf")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved  {png_path}")
print(f"Saved  {pdf_path}")

# copy to figures-paper-1/
paper_fig_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures-paper-1")
)
os.makedirs(paper_fig_dir, exist_ok=True)
dest = os.path.join(paper_fig_dir, "fig_ae_ceiling_correlation.png")
shutil.copy2(png_path, dest)
print(f"Copied {dest}")

# ── Print values for LaTeX substitution ──────────────────────────────────────
print(f"\nFor LaTeX:  R^2 = {r2_val:.2f},  r = {r_val:.2f}")
