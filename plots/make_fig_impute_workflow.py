"""Schematic of the imputation experiments (Section 2.6).

Reviewer 1 asked for a workflow figure that makes the whole section followable:
where each method sits, how AGs become a reference panel, how direct imputation
bypasses that, and how the three settings (general, population-specific,
array-based) differ. Panel (a) is the pipeline; panel (b) is what changes
between settings.

    python plots/make_fig_impute_workflow.py
    -> overleaf/figs/fig_impute_workflow.pdf (+ .png)
"""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DARK = "#333333"
RED = "#C23F38"
NAVY = "#1A1A2E"
TEAL = "#289D8F"
TEAL_D = "#1B6B62"
GRAY = "#555555"

MINT = "#F1FBF8"
BLUSH = "#FDF1F0"
SLATE = "#EFF1F5"
CREAM = "#FBF6EC"


def box(ax, x, y, w, h, title, sub=None, fc="white", ec=DARK, tc=NAVY,
        fs=11, subfs=9.0, lw=1.5, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.22",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3))
    if sub:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", zorder=4)
        ax.text(x, y - h * 0.21, sub, ha="center", va="center", color=GRAY,
                fontsize=subfs, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", zorder=4)
    return (x, y, w, h)


def arrow(ax, p, q, color=DARK, lw=1.8, style="solid", label=None,
          lx=0.0, ly=0.45, rad=0.0, fs=9.0, italic=True):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=15, linewidth=lw, color=color,
        linestyle=style, connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1, zorder=2))
    if label:
        mx, my = (p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", color=color,
                fontsize=fs, style="italic" if italic else "normal", zorder=4)


def right(b):
    x, y, w, h = b
    return (x + w / 2, y)


def left(b):
    x, y, w, h = b
    return (x - w / 2, y)


def top(b):
    x, y, w, h = b
    return (x, y + h / 2)


def bottom(b):
    x, y, w, h = b
    return (x, y - h / 2)


def main():
    fig, ax = plt.subplots(figsize=(19.0, 9.5))
    ax.set_xlim(0, 53.5)
    ax.set_ylim(-10.6, 15.9)
    ax.axis("off")
    ax.set_aspect("equal")

    # ============================================  (a) the pipeline, by phase
    ax.text(0.2, 15.2, "(a)", ha="left", va="center", color=NAVY,
            fontsize=14, fontweight="bold")
    ax.text(2.0, 15.2, "Imputation pipeline, by phase",
            ha="left", va="center", color=NAVY, fontsize=13.5, fontweight="bold")

    # One column per phase. Columns are spaced 1.8 units apart so every
    # connector has a visible stem, not just an arrowhead.
    P1, P2, P3, P4, P5, P6 = 4.6, 14.1, 22.9, 32.3, 42.1, 50.2
    phases = [(P1, 8.0, "Training data"), (P2, 7.4, "Model"),
              (P3, 6.6, "Generation"), (P4, 8.6, "Reference panel"),
              (P5, 7.4, "Imputation"), (P6, 5.2, "Evaluation")]
    for x, w, name in phases:
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, 13.7), w, 0.95,
            boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=SLATE, edgecolor="none", zorder=1))
        ax.text(x, 14.18, name, ha="center", va="center", color=NAVY,
                fontsize=10.5, fontweight="bold", zorder=2)

    # -- the main row: every phase of the AG route, left to right
    b_train = box(ax, P1, 8.5, 8.0, 2.3, "Training haplotypes",
                  "80% split; private or\npopulation-specific", fc=MINT, ec=TEAL_D)
    b_models = box(ax, P2, 8.5, 7.4, 2.6, "Generative models",
                   "WGAN   RBM\nGPC   HMM", fc="white")
    b_ags = box(ax, P3, 8.5, 6.6, 2.3, "Artificial genomes",
                "ancestral sampling", fc="white")
    b_panel = box(ax, P4, 8.5, 8.6, 2.3, "Reference panel",
                  "AGs, real genomes,\nor AGs + real EUR", fc="white")
    b_imp = box(ax, P5, 8.5, 7.4, 2.3, "Impute5", fc=SLATE)
    b_eval = box(ax, P6, 5.4, 5.2, 3.4, "Accuracy",
                 "$r^2$ vs. true\ngenotype, by MAF\n(10 bootstraps)",
                 fc="white", ec=NAVY)

    # -- real genomes are a panel input, not a separate pipeline
    b_real = box(ax, P4, 12.4, 8.6, 2.0, "Real reference genomes",
                 "public EUR, or ancestry-matched", fc=SLATE)

    # -- the genotypes being imputed, feeding both imputation routes
    b_test = box(ax, P5, 5.0, 7.4, 1.9, "Test haplotypes",
                 "20% split; target SNP(s) masked", fc=CREAM, ec="#B08A3E",
                 fs=10.5, subfs=8.5)

    # -- the direct route: straight from the model to imputation
    b_direct = box(ax, P5, 1.45, 7.4, 2.4, "Direct imputation",
                   "$P(X_{\\mathrm{miss}} \\mid X_{\\mathrm{obs}})$   "
                   "exact conditional", fc=MINT, ec=TEAL_D, tc=TEAL_D,
                   fs=10.5, subfs=8.5)

    # -- wiring: one solid arrow style throughout
    arrow(ax, right(b_train), left(b_models), color=TEAL_D)
    arrow(ax, right(b_models), left(b_ags))
    arrow(ax, right(b_ags), left(b_panel))
    arrow(ax, right(b_panel), left(b_imp))
    arrow(ax, (45.8, 8.1), (47.6, 6.9))

    # real genomes drop straight down into the panel
    arrow(ax, bottom(b_real), top(b_panel))

    # test genotypes feed Impute5 above and the direct route below
    arrow(ax, top(b_test), bottom(b_imp))
    arrow(ax, bottom(b_test), top(b_direct))

    # the bypass: model straight to imputation, skipping generation and panel
    ax.add_patch(FancyArrowPatch(
        (P2, 7.2), (38.4, 1.45), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.8, color=TEAL_D,
        connectionstyle="angle,angleA=-90,angleB=0,rad=0.8",
        shrinkA=1, shrinkB=1, zorder=2))
    ax.text(26.0, 0.75, "no AGs, no Impute5; GPC and HMM only",
            ha="center", va="center", color=TEAL_D, fontsize=9.5, style="italic")

    arrow(ax, (45.8, 1.85), (47.6, 3.6), color=TEAL_D)

    # route labels
    ax.text(22.9, 10.6, "AG route", ha="center", va="center", color=DARK,
            fontsize=10.5, fontweight="bold")
    ax.text(9.2, 4.4, "direct route", ha="center", va="center", color=TEAL_D,
            fontsize=10.5, fontweight="bold")

    # =============================================  (b) what differs by setting
    ax.text(0.2, -1.6, "(b)", ha="left", va="center", color=NAVY,
            fontsize=14, fontweight="bold")
    ax.text(2.0, -1.6, "The three settings differ only in the data splits and "
                       "which SNPs are masked",
            ha="left", va="center", color=NAVY, fontsize=13.5, fontweight="bold")

    cols = [(6.0, 10.0, "Setting"), (18.0, 12.0, "Training data"),
            (30.5, 11.0, "Test data"), (43.5, 13.0, "SNPs imputed")]
    rows = [
        (-4.9, MINT, "General",
         "80% of a cosmopolitan\npanel (1KG, UKBB)",
         "20% held-out,\nsame ancestries",
         "one SNP at a time,\nall others observed"),
        (-6.9, BLUSH, "Population-specific",
         "80% of the target population\n(African / non-European),\noptionally + real EUR",
         "20% held-out of the\ntarget population",
         "one SNP at a time,\nall others observed"),
        (-8.9, CREAM, "Array-based",
         "80% of high-coverage 1KG",
         "20% held-out",
         "12,551 non-array SNPs\njointly, from 2,119 typed"),
    ]

    for x, w, name in cols:
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, -3.7), w, 1.0,
            boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=NAVY, edgecolor="none", zorder=3))
        ax.text(x, -3.2, name, ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=4)

    for y, fc, *cells in rows:
        for (x, w, _), cell in zip(cols, cells):
            ax.add_patch(FancyBboxPatch(
                (x - w / 2, y - 0.85), w, 1.7,
                boxstyle="round,pad=0,rounding_size=0.18",
                facecolor=fc, edgecolor="#D8DDE4", linewidth=1.0, zorder=3))
            ax.text(x, y, cell, ha="center", va="center", color=NAVY,
                    fontsize=9.2, zorder=4,
                    fontweight="bold" if cell is cells[0] else "normal")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "overleaf", "figs", "fig_impute_workflow")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{out}.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"wrote {os.path.normpath(out)}.pdf / .png")


if __name__ == "__main__":
    main()
