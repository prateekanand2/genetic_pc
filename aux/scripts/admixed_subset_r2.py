"""Imputation accuracy restricted to admixed individuals in the 1KG test set.

Reviewer 1 (comment 7) asked how GPC performs in admixed populations. Rather
than training new models, this re-scores the existing experiments on the subset
of the held-out test individuals whose 1KG population is admixed, so every
method is evaluated on exactly the models and reference panels already reported.

Admixed is taken as the four American populations (CLM, MXL, PEL, PUR), which
are admixed by construction, plus the two recently admixed African-ancestry
populations (ASW, ACB). In the 1KG 80/20 test split this is 212 of the 1,002
haplotypes (106 of 501 individuals, 21.2%).

Recovering which individual each test haplotype belongs to is possible because
the split is reproducible: aux/scripts/split.py uses
train_test_split(..., random_state=42) at the individual level. That recovers
the set exactly, though not the row order, so rows are matched back to sample
IDs by hashing the haplotype (all 5,008 are unique over 10,000 SNPs).

    python admixed_subset_r2.py --dosage-dir ../../impute5/results/admixed
"""

import argparse
import collections
import csv
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Admixed by construction (American) plus the recently admixed African-ancestry
# populations. ACB/ASW carry substantial European and Native American ancestry.
ADMIXED = {"CLM", "MXL", "PEL", "PUR", "ASW", "ACB"}


def _hash(row):
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()


def test_haplotype_ancestry():
    """Population code for each row of 8020_test.txt, in file order."""
    hap = pd.read_csv(f"{ROOT}/aux/10K_real.hapt", sep=r"\s+", header=None)
    ids = hap.iloc[:, 1].tolist()
    panel = hap.iloc[:, 2:].to_numpy(dtype=np.int8)

    # Confirm the recorded split is the one we can reproduce, then map by hash.
    individuals = pd.Series(ids).str.replace(r"_[AB]$", "", regex=True)
    _, test_inds = train_test_split(individuals.unique(), test_size=0.2, random_state=42)
    expected = int(individuals.isin(test_inds).sum())

    lut = collections.defaultdict(list)
    for i, row in enumerate(panel):
        lut[_hash(row)].append(i)
    assert all(len(v) == 1 for v in lut.values()), "haplotypes are not unique; cannot map by hash"

    test = np.loadtxt(f"{ROOT}/results/1KG/8020/data/8020_test.txt", dtype=np.int8, delimiter=' ')
    assert len(test) == expected, f"test set has {len(test)} rows, split predicts {expected}"

    meta = {r["Sample name"]: r["Population code"]
            for r in csv.DictReader(open(f"{ROOT}/aux/10K_igsr_samples.tsv"), delimiter="\t")}
    pops = []
    for row in test:
        hit = lut.get(_hash(row))
        assert hit, "a test haplotype does not occur in the full panel"
        pops.append(meta[ids[hit[0]].rsplit("_", 1)[0]])
    return test, np.array(pops)


def r2_per_snp(truth, pred):
    t = truth.astype(np.float64)
    p = pred.astype(np.float64)
    tc = t - t.mean(axis=0)
    pc_ = p - p.mean(axis=0)
    num = (tc * pc_).sum(axis=0)
    den = np.sqrt((tc ** 2).sum(axis=0) * (pc_ ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0) ** 2


def load_dosages(dosage_dir, snp_ids, n_hap):
    """(n_hap x n_snp) matrix from the per-SNP files impute5_loo_local.py writes."""
    out = np.full((n_hap, len(snp_ids)), np.nan, dtype=np.float32)
    missing = 0
    for j, snp in enumerate(snp_ids):
        f = os.path.join(dosage_dir, f"{snp}.txt")
        if not os.path.exists(f):
            missing += 1
            continue
        out[:, j] = np.loadtxt(f)
    return out, missing


def summarize(truth, pred, maf, mask, label):
    """Mean r^2 over SNPs, for one subset of haplotypes, in three MAF bins."""
    t, p = truth[mask], pred[mask]
    scored = ~np.isnan(p[0])
    r2 = np.full(t.shape[1], np.nan)
    r2[scored] = r2_per_snp(t[:, scored], np.nan_to_num(p[:, scored]))
    out = {}
    for name, hi in (("all", 1.0), ("<1%", 0.01), ("<0.1%", 0.001)):
        sel = scored & (maf > 0) & (maf < hi)
        out[name] = float(np.nanmean(r2[sel])) if sel.any() else float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dosage-dir", default=f"{ROOT}/impute5/results/admixed",
                   help="parent of {method}_dosages, as written by impute5_loo_local.py")
    p.add_argument("--methods", nargs="+",
                   default=["real_8020", "hclt_8020", "rbm_8020", "wgan_8020", "hmm_8020"])
    p.add_argument("--direct", nargs="*", default=[],
                   help="extra direct arms as name=path/to/dosages.npy")
    p.add_argument("--out", default=f"{ROOT}/plots/impute/results/r2/admixed_subset_summary.csv")
    args = p.parse_args()

    test, pops = test_haplotype_ancestry()
    adm = np.isin(pops, list(ADMIXED))
    print(f"test set: {len(test)} haplotypes; admixed {adm.sum()} "
          f"({100*adm.mean():.1f}%), non-admixed {(~adm).sum()}")
    for pop, n in sorted(collections.Counter(pops[adm]).items()):
        print(f"    {pop}: {n}")

    legend = pd.read_csv(f"{ROOT}/aux/10K_legend.maf.txt", sep="\t", header=None,
                         names=["SNP", "MAF"])
    snp_ids, maf = legend["SNP"].tolist(), legend["MAF"].to_numpy()

    sources = [(m, os.path.join(args.dosage_dir, f"{m}_dosages"), "ag") for m in args.methods]
    sources += [(s.split("=", 1)[0], s.split("=", 1)[1], "npy") for s in args.direct]

    rows = []
    for name, path, kind in sources:
        if kind == "ag":
            if not os.path.isdir(path):
                print(f"  {name}: no dosages yet at {path}, skipping")
                continue
            pred, miss = load_dosages(path, snp_ids, len(test))
            if miss:
                print(f"  {name}: {miss} SNPs still missing, reporting on the rest")
        else:
            if not os.path.exists(path):
                print(f"  {name}: {path} not found, skipping")
                continue
            pred = np.load(path)
        for label, mask in (("admixed", adm), ("non-admixed", ~adm), ("all", np.ones_like(adm))):
            r = summarize(test, pred, maf, mask, label)
            rows.append(dict(method=name, subset=label, n_haplotypes=int(mask.sum()), **r))
            print(f"  {name:12s} {label:12s} n={int(mask.sum()):4d}  "
                  f"all={r['all']:.4f}  <1%={r['<1%']:.4f}  <0.1%={r['<0.1%']:.4f}")

    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
