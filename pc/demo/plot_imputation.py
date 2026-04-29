"""
plot_imputation.py — Plot imputation R^2 vs MAF.

Reads CSVs from <run-dir>/imputation/ and writes imputation_r2.pdf alongside.

Notes:
  - CI bands are mean ± 1.96 * (s / sqrt(n)) per MAF bin, where s is the
    sample std of per-SNP R^2 within the bin (no bootstrap over individuals).
  - Multi-SNP runs score only a random fraction of SNPs, so rare-MAF bins
    may be noisy; SNP counts per bin are printed for inspection.

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
    p.add_argument("--bins", type=int, default=9)
    return p.parse_args()


def bin_r2(df, bins, num_bins):
    """Per-bin mean R^2 and 95% CI (mean ± 1.96 * s/sqrt(n))."""
    df = df.dropna(subset=["r2", "maf"])
    df = df[df["maf"] > 0]
    bin_indices = np.digitize(df["maf"].values, bins) - 1
    means   = np.full(num_bins, np.nan)
    ci_low  = np.full(num_bins, np.nan)
    ci_high = np.full(num_bins, np.nan)
    counts  = np.zeros(num_bins, dtype=int)
    for i in range(num_bins):
        in_bin = bin_indices == i
        r = df.loc[in_bin, "r2"].values
        counts[i] = len(r)
        if len(r) == 0:
            continue
        m = r.mean()
        s = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
        means[i]   = m
        ci_low[i]  = max(m - 1.96 * s, 0.0)
        ci_high[i] = min(m + 1.96 * s, 1.0)
    return means, ci_low, ci_high, counts


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

    rate_colors  = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(meta["mask_rates"])))
    multi_markers = ["s", "^", "D", "v", "X", "P"]
    for i, (rate, color) in enumerate(zip(meta["mask_rates"], rate_colors)):
        tag  = f"{int(round(rate * 100)):02d}"
        path = impute_dir / f"r2_multi_{tag}.csv"
        if path.exists():
            entries.append((f"Multi-SNP ({int(round(rate * 100))}% missing)",
                            path, color, multi_markers[i % len(multi_markers)], "--"))

    if not entries:
        raise SystemExit(f"No imputation CSVs found in {impute_dir}")

    num_bins    = args.bins
    bins        = np.logspace(-4, np.log10(0.5), num_bins + 1)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    fig, ax = plt.subplots(figsize=(8, 6))
    print("SNP counts per MAF bin:")
    print("  bin edges: " + " ".join(f"{e:.2g}" for e in bins))

    for label, path, color, marker, ls in entries:
        df = pd.read_csv(path)
        means, ci_low, ci_high, counts = bin_r2(df, bins, num_bins)
        ok = ~np.isnan(means)
        print(f"  {label:30s} {counts}")
        ax.plot(bin_centers[ok], means[ok],
                marker=marker, linestyle=ls, linewidth=2.0,
                color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.5,
                label=label)
        ax.fill_between(bin_centers[ok], ci_low[ok], ci_high[ok],
                        color=color, alpha=0.2)

    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"],
                       fontsize=FONTSIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.set_xlabel("MAF", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r"Average $r^2$", fontsize=FONTSIZE_LABEL)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = impute_dir / "imputation_r2.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

    # Summary table.
    summary = []
    print("\nMean R^2 summary:")
    for label, path, *_ in entries:
        df = pd.read_csv(path).dropna(subset=["r2", "maf"])
        df = df[df["maf"] > 0]
        row = {"setting": label, "n_snps": int(len(df))}
        for bucket, mask in [("all",      df["maf"] > 0),
                              ("low_freq", df["maf"] < 0.01),
                              ("rare",     df["maf"] < 0.001)]:
            sub = df.loc[mask, "r2"]
            row[f"mean_r2_{bucket}"] = float(sub.mean()) if len(sub) else np.nan
            row[f"n_{bucket}"]       = int(len(sub))
        summary.append(row)
        print(f"  {label:30s}  all={row['mean_r2_all']:.4f} (n={row['n_all']})  "
              f"low-freq={row['mean_r2_low_freq']:.4f} (n={row['n_low_freq']})  "
              f"rare={row['mean_r2_rare']:.4f} (n={row['n_rare']})")

    pd.DataFrame(summary).to_csv(impute_dir / "imputation_summary.csv", index=False)
    print(f"Summary: {impute_dir/'imputation_summary.csv'}")


if __name__ == "__main__":
    main()