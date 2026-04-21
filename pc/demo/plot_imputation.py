"""
plot_imputation.py — Plot imputation R^2 vs MAF.

Reads CSVs from <run-dir>/imputation/ and writes imputation_r2.pdf alongside.

Notes:
  - SEM per MAF bin is s / sqrt(n_SNPs) over per-SNP R^2 (no bootstrap).
  - Multi-SNP runs only score a random fraction of SNPs, so rare-MAF bins can
    be very noisy; bins with < --min-bin-count SNPs are hidden (counts printed).

Example:
    python3 plot_imputation.py --run-dir out/1K
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FONTSIZE_LABEL  = 18
FONTSIZE_TICK   = 16
FONTSIZE_LEGEND = 14


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default="out/1K")
    p.add_argument("--bins", type=int, default=6,
                   help="Number of log-spaced MAF bins")
    p.add_argument("--maf-min", type=float, default=1e-3)
    p.add_argument("--min-bin-count", type=int, default=3,
                   help="Hide bins with fewer than this many SNPs "
                        "(counts still appear in the summary)")
    return p.parse_args()


def bin_stats(df, edges):
    # mean R^2 and SEM (s / sqrt(n)) of per-SNP R^2 within each MAF bin.
    df = df.dropna(subset=["r2", "maf"])
    df = df[df["maf"] > 0]
    centers = np.sqrt(edges[:-1] * edges[1:])
    means  = np.full(len(centers), np.nan)
    sems   = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        lo, hi = edges[i], edges[i + 1]
        mask = (df["maf"] >= lo) & (df["maf"] < hi)
        r = df.loc[mask, "r2"].values
        counts[i] = len(r)
        if len(r) > 0:
            means[i] = r.mean()
            sems[i] = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    return centers, means, sems, counts


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    impute_dir = run_dir / "imputation"

    meta = json.loads((impute_dir / "meta.json").read_text()) \
           if (impute_dir / "meta.json").exists() else {"mask_rates": [0.3, 0.5, 0.8]}

    entries = []
    single_path = impute_dir / "r2_single.csv"
    if single_path.exists():
        entries.append(("Single-SNP", single_path, "tab:blue", "o", "-"))

    rate_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(meta["mask_rates"])))
    multi_markers = ["s", "^", "D", "v", "X", "P"]
    for i, (rate, color) in enumerate(zip(meta["mask_rates"], rate_colors)):
        tag = f"{int(round(rate * 100)):02d}"
        path = impute_dir / f"r2_multi_{tag}.csv"
        if path.exists():
            marker = multi_markers[i % len(multi_markers)]
            entries.append((f"Multi-SNP ({int(round(rate * 100))}% missing)",
                            path, color, marker, "--"))

    if not entries:
        raise SystemExit(f"No imputation CSVs found in {impute_dir}")

    edges = np.logspace(np.log10(args.maf_min), np.log10(0.5), args.bins + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    bin_count_report = []
    for label, path, color, marker, ls in entries:
        df = pd.read_csv(path)
        x, mean, sem, counts = bin_stats(df, edges)
        bin_count_report.append((label, counts))
        plot_mask = counts >= args.min_bin_count
        ax.errorbar(x[plot_mask], mean[plot_mask], yerr=sem[plot_mask],
                    marker=marker, linestyle=ls, linewidth=2.0,
                    capsize=3, color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=label)

    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-2, 1e-1])
    ax.set_xticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"],
                       fontsize=FONTSIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.set_xlabel("MAF", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r"Average $r^2$", fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND)

    out = impute_dir / "imputation_r2.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

    # Per-bin SNP counts (useful for diagnosing noisy bins like 30% multi-SNP).
    print("\nSNP counts per MAF bin:")
    print("  bin edges: " + " ".join(f"{e:.3g}" for e in edges))
    for label, counts in bin_count_report:
        hidden = (counts < args.min_bin_count) & (counts > 0)
        tag = "".join("!" if h else " " for h in hidden)
        print(f"  {label:28s}  {counts}   {tag}  "
              f"(! = hidden in plot, < {args.min_bin_count})")

    # Overall + MAF-subset summary per setting.
    summary = []
    print("\nMean R^2 summary:")
    for label, path, _, _, _ in entries:
        df = pd.read_csv(path).dropna(subset=["r2", "maf"])
        df = df[df["maf"] > 0]
        row = {"setting": label, "n_snps": int(len(df))}
        for bucket, mask in [
            ("all",      df["maf"] > 0),
            ("low_freq", df["maf"] < 0.01),
            ("rare",     df["maf"] < 0.001),
        ]:
            sub = df.loc[mask, "r2"]
            if len(sub) > 0:
                row[f"mean_r2_{bucket}"] = float(sub.mean())
                row[f"n_{bucket}"] = int(len(sub))
            else:
                row[f"mean_r2_{bucket}"] = np.nan
                row[f"n_{bucket}"] = 0
        summary.append(row)
        print(f"  {label:28s}  all={row['mean_r2_all']:.4f} (n={row['n_all']})  "
              f"low-freq={row['mean_r2_low_freq']:.4f} (n={row['n_low_freq']})  "
              f"rare={row['mean_r2_rare']:.4f} (n={row['n_rare']})")

    pd.DataFrame(summary).to_csv(impute_dir / "imputation_summary.csv", index=False)
    print(f"\nSummary: {impute_dir/'imputation_summary.csv'}")


if __name__ == "__main__":
    main()
