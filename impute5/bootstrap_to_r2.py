"""Assemble per-SNP Impute5 dosages into the R2 CSV read by plot.ipynb.

Same job as bootstrap_info.py, but it takes the bootstrap resampling indices
from test_bootstraps/indices_*.txt instead of re-parsing every bootstrap VCF for
every SNP, and it emits the columns in legend order with MAF already attached --
so the output drops straight into plots/impute/results/r2/ with no follow-up
attach_maf.py step.

Usage:
    python bootstrap_to_r2.py 1KG:afr --dosage-dir results/bootstrap/10K_hmm_afr_dosages
    python bootstrap_to_r2.py 1KG:afr --dosage-dir ..._combined_dosages --combined
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pc"))
from hmm_config import resolve, r2_csv  # noqa: E402
from assemble_r2 import r2_per_snp  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Impute5 dosages -> R2/bootstrap CSV.")
    p.add_argument("config", help="dataset/split key, e.g. 1KG:afr")
    p.add_argument("--dosage-dir", required=True)
    p.add_argument("--combined", action="store_true",
                   help="panel was merged with real EUR (affects the output file name)")
    p.add_argument("--out", default=None)
    p.add_argument("--n-bootstraps", type=int, default=10)
    args = p.parse_args()

    cfg = resolve(args.config)
    out_path = args.out or r2_csv(cfg, "ag", combined=args.combined)

    truth = np.loadtxt(cfg["valid_path"], dtype=np.int8, delimiter=' ')
    legend = pd.read_csv(cfg["legend_maf"], sep="\t", header=None, names=["SNP Set", "MAF"])
    assert len(legend) == truth.shape[1], \
        f"{len(legend)} SNPs in {cfg['legend_maf']} vs {truth.shape[1]} in the data"

    # Dosages are one file per SNP; missing ones stay NaN so gaps are visible
    # rather than silently scoring 0.
    pred = np.full(truth.shape, np.nan, dtype=np.float32)
    missing = []
    for j, snp in enumerate(legend["SNP Set"]):
        f = os.path.join(args.dosage_dir, f"{snp}.txt")
        if not os.path.exists(f):
            missing.append(snp)
            continue
        d = np.loadtxt(f)
        if d.shape[0] != truth.shape[0]:
            raise SystemExit(f"{f}: {d.shape[0]} dosages vs {truth.shape[0]} test haplotypes")
        pred[:, j] = d

    if missing:
        print(f"WARNING: {len(missing)}/{len(legend)} SNPs have no dosage file "
              f"(first few: {missing[:5]}) -- rerun impute5_loo_local.py to fill them in")

    done = ~np.isnan(pred[0])
    filled = np.nan_to_num(pred, nan=0.0)

    out = legend[["SNP Set"]].copy()
    base = r2_per_snp(truth, filled)
    base[~done] = np.nan
    out["R2"] = base

    for b in range(1, args.n_bootstraps + 1):
        idx_file = f"{cfg['bootstrap_dir']}/indices_{b}.txt"
        if not os.path.exists(idx_file):
            raise SystemExit(f"missing {idx_file}; run aux/scripts/recover_bootstrap_indices.py first")
        idx = np.loadtxt(idx_file, dtype=int)
        col = r2_per_snp(truth[idx], filled[idx])
        col[~done] = np.nan
        out[f"R2_boot_{b}"] = col

    out["MAF"] = legend["MAF"].values

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    boot_cols = [f"R2_boot_{b}" for b in range(1, args.n_bootstraps + 1)]
    keep = out[(out["MAF"] > 0) & out["R2"].notna()]
    means = keep[boot_cols].mean()
    print(f"wrote {out_path}  ({done.sum()}/{len(out)} SNPs scored)")
    print(f"  mean R2 (MAF>0)      {keep['R2'].mean():.4f}")
    print(f"  bootstrap mean       {means.mean():.4f} +/- {1.96*means.std(ddof=1):.4f}")
    print(f"  low-freq  (<1%)      {keep[keep['MAF'] < 0.01][boot_cols].mean().mean():.4f}")
    print(f"  rare      (<0.1%)    {keep[keep['MAF'] < 0.001][boot_cols].mean().mean():.4f}")


if __name__ == "__main__":
    main()
