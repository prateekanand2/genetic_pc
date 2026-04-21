"""
impute_demo.py — Imputation benchmark for a trained GPC.

Two modes:
  * single — mask one SNP at a time (leave-one-out per-SNP R^2).
  * multi  — mask a random fraction of SNPs simultaneously (30/50/80%).

Notes:
  - SNPs monomorphic in train (MAF == 0) are excluded from the reported R^2.
  - Error bars in plot_imputation.py are SEM across SNPs within a MAF bin
    (s / sqrt(n), no bootstrap over individuals).

Example:
    python3 impute_demo.py --run-dir out/1K --mask-rates 0.3 0.5 0.8 --seed 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import pyjuice as juice


def r2_per_feature(truth, pred):
    """Vectorized squared Pearson correlation per column. Returns shape (F,)."""
    t = truth.astype(np.float64)
    p = pred.astype(np.float64)
    tc = t - t.mean(axis=0, keepdims=True)
    pc = p - p.mean(axis=0, keepdims=True)
    num = (tc * pc).sum(axis=0)
    den = np.sqrt((tc ** 2).sum(axis=0) * (pc ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return r ** 2


def summarize(tag, maf, r2):
    # Print mean R^2 + SEM for All / Low-freq (<1%) / Rare (<0.1%) MAF buckets.
    m = np.asarray(maf)
    r = np.asarray(r2)
    cats = [
        ("All        (MAF > 0)    ", m > 0),
        ("Low-freq   (MAF < 1%)   ", m < 0.01),
        ("Rare       (MAF < 0.1%) ", m < 0.001),
    ]
    print(f"  {tag}")
    for name, mask in cats:
        mask = mask & (m > 0) & ~np.isnan(r)
        if mask.sum() == 0:
            print(f"    {name}: (no SNPs)")
            continue
        rr = r[mask]
        s = rr.std(ddof=1) / np.sqrt(len(rr)) if len(rr) > 1 else 0.0
        print(f"    {name}: mean R^2 = {rr.mean():.4f}  SEM = {s:.4f}   "
              f"(n = {len(rr)})")


def impute_single(pc, data, device, batch_size):
    """Mask one SNP at a time; return (n_samples, n_features) expected values."""
    n, f = data.shape
    ds = TensorDataset(torch.tensor(data, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size)
    pred = np.zeros((n, f), dtype=np.float32)
    mask = torch.zeros(f, dtype=torch.bool, device=device)

    cursor = 0
    for (batch,) in loader:
        x = batch.to(device)
        bs = x.size(0)
        for pos in tqdm(range(f), desc="Single-SNP imputation", leave=False):
            mask[pos] = True
            lls = juice.queries.conditional(pc, data=x, missing_mask=mask)
            pred[cursor:cursor + bs, pos] = lls[:, pos, 1].detach().cpu().numpy()
            mask[pos] = False
        cursor += bs
    return pred


def impute_multi(pc, data, device, batch_size, mask_indices):
    """Mask a fixed subset simultaneously; return predictions for those indices only."""
    n, f = data.shape
    ds = TensorDataset(torch.tensor(data, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size)
    mask = torch.zeros(f, dtype=torch.bool, device=device)
    mask[mask_indices] = True

    pred = np.zeros((n, len(mask_indices)), dtype=np.float32)
    cursor = 0
    for (batch,) in tqdm(loader, desc="Multi-SNP imputation", leave=False):
        x = batch.to(device)
        bs = x.size(0)
        lls = juice.queries.conditional(pc, data=x, missing_mask=mask)
        pred[cursor:cursor + bs] = lls[:, mask_indices, 1].detach().cpu().numpy()
        cursor += bs
    return pred


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default="out/1K")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--skip-single", action="store_true")
    p.add_argument("--skip-multi", action="store_true")
    p.add_argument("--mask-rates", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "gpc_best.jpc"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = np.loadtxt(run_dir / "train.txt", dtype=np.int8, delimiter=" ")
    test  = np.loadtxt(run_dir / "test.txt",  dtype=np.int8, delimiter=" ")
    n, f = test.shape
    print(f"Test: {n} haplotypes x {f} SNPs  |  Loading {ckpt}")

    pc = juice.compile(juice.load(str(ckpt))).to(device)

    af = train.mean(axis=0)
    maf = np.minimum(af, 1 - af)  # MAF folded to [0, 0.5]
    mono = maf == 0
    print(f"  Monomorphic SNPs in train (MAF = 0): {int(mono.sum())} / {f} "
          f"(excluded from reported R^2)")

    out_dir = run_dir / "imputation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_single:
        print("Running single-SNP imputation (leave-one-out, all SNPs) ...")
        pred = impute_single(pc, test, device, args.batch_size)
        r2_full = r2_per_feature(test, pred)
        keep = ~mono
        pd.DataFrame({
            "snp": np.arange(f)[keep],
            "maf": maf[keep],
            "r2":  r2_full[keep],
        }).to_csv(out_dir / "r2_single.csv", index=False)
        print(f"  -> {out_dir/'r2_single.csv'}   "
              f"({int(keep.sum())} non-monomorphic SNPs)")
        summarize("Single-SNP", maf[keep], r2_full[keep])

    if not args.skip_multi:
        for rate in args.mask_rates:
            k = max(1, int(round(rate * f)))
            idx = np.sort(rng.choice(f, size=k, replace=False))
            print(f"Running multi-SNP imputation at {rate*100:.0f}% "
                  f"({k}/{f} SNPs) ...")
            pred = impute_multi(pc, test, device, args.batch_size, idx)
            r2 = r2_per_feature(test[:, idx], pred)
            keep = ~mono[idx]
            tag = f"{int(round(rate * 100)):02d}"
            pd.DataFrame({
                "snp": idx[keep],
                "maf": maf[idx][keep],
                "r2":  r2[keep],
            }).to_csv(out_dir / f"r2_multi_{tag}.csv", index=False)
            print(f"  -> {out_dir/f'r2_multi_{tag}.csv'}   "
                  f"({int(keep.sum())} non-monomorphic SNPs in mask)")
            summarize(f"Multi-SNP {int(round(rate*100))}%",
                      maf[idx][keep], r2[keep])

    with open(out_dir / "meta.json", "w") as fp:
        json.dump({"mask_rates": list(args.mask_rates), "n_snps": int(f)}, fp, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
