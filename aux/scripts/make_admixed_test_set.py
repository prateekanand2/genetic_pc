"""Build an admixed-only version of the 1KG 80/20 test set.

Reviewer 1 (comment 7) asked how GPC performs in admixed populations. Rather
than retraining anything, we re-run the existing imputation experiments -- same
models, same reference panels -- against a test set restricted to the admixed
individuals already present in the held-out split. Only the target file changes.

Admixed is taken as the four American populations (CLM, MXL, PEL, PUR), which
are admixed by construction, plus the two recently admixed African-ancestry
populations (ASW, ACB). In this split that is 212 of the 1,002 test haplotypes,
i.e. 106 of the 501 individuals; both haplotypes of an individual are always
kept together.

The test files carry anonymized sample names, so ancestry is recovered by
reproducing the split -- aux/scripts/split.py uses
train_test_split(..., random_state=42) at the individual level -- and then
matching rows back to sample IDs by hashing the haplotype. All 5,008 haplotypes
in the panel are unique over the 10,000 SNPs, so the mapping is exact.

Writes, under results/1KG/8020/data/:
    admixed_test.txt / .vcf / .vcf.gz      the 212-haplotype target set
    admixed_test_bootstraps/bootstrap_N.vcf, indices_N.txt

    python make_admixed_test_set.py
"""

import collections
import csv
import gzip
import hashlib
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The 1KG and high-coverage 1KG (b38) experiments use the same 2,504 individuals
# and the same 80/20 split, verified by mapping both test sets back to sample IDs,
# so the admixed subset is the same 106 individuals in both.
PANELS = {"1KG": ("aux/10K_real.hapt", "results/1KG/8020/data"),
          "b38": ("aux/b38_real.hapt", "results/b38/8020/data")}
BCFTOOLS = os.environ.get("BCFTOOLS", "/scratch2/prateek/bcftools/bcftools")
ADMIXED = {"CLM", "MXL", "PEL", "PUR", "ASW", "ACB"}
N_BOOT = 10
SEED = 42


def _hash(row):
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()


def admixed_mask(panel_hapt, data_dir):
    """Boolean mask over the rows of 8020_test.txt, plus per-row population."""
    hap = pd.read_csv(f"{ROOT}/{panel_hapt}", sep=r"\s+", header=None)
    ids = hap.iloc[:, 1].tolist()
    panel = hap.iloc[:, 2:].to_numpy(dtype=np.int8)

    individuals = pd.Series(ids).str.replace(r"_[AB]$", "", regex=True)
    _, test_inds = train_test_split(individuals.unique(), test_size=0.2, random_state=SEED)
    expected = int(individuals.isin(test_inds).sum())

    lut = collections.defaultdict(list)
    for i, row in enumerate(panel):
        lut[_hash(row)].append(i)
    assert all(len(v) == 1 for v in lut.values()), "haplotypes not unique; cannot map by hash"

    test = np.loadtxt(f"{data_dir}/8020_test.txt", dtype=np.int8, delimiter=' ')
    assert len(test) == expected, f"test has {len(test)} rows, split predicts {expected}"

    meta = {r["Sample name"]: r["Population code"]
            for r in csv.DictReader(open(f"{ROOT}/aux/10K_igsr_samples.tsv"), delimiter="\t")}
    pops, sample_ids = [], []
    for row in test:
        hit = lut.get(_hash(row))
        assert hit, "a test haplotype does not occur in the full panel"
        sid = ids[hit[0]]
        sample_ids.append(sid)
        pops.append(meta[sid.rsplit("_", 1)[0]])
    pops = np.array(pops)
    mask = np.isin(pops, list(ADMIXED))

    # Both haplotypes of an individual must be kept or dropped together.
    per_ind = collections.Counter(sample_ids[i].rsplit("_", 1)[0] for i in np.where(mask)[0])
    assert all(v == 2 for v in per_ind.values()), "an individual was split across the subset"
    return test, mask, pops, sample_ids


def read_vcf(path):
    op = gzip.open if path.endswith(".gz") else open
    header, chrom_line, body = [], None, []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                header.append(line)
            elif line.startswith("#CHROM"):
                chrom_line = line.rstrip("\n").split("\t")
            else:
                body.append(line.rstrip("\n").split("\t"))
    return header, chrom_line, body


def write_vcf(path, header, chrom_line, body, cols, rename=False):
    """Keep only the genotype columns in `cols` (0-based over samples).

    Bootstrap replicates sample columns with replacement, so the same column can
    appear more than once; carrying the original names over would produce a
    duplicated sample name and bcftools refuses to parse the header. Fresh
    sequential names are emitted instead, matching create_bootstraps.py.
    """
    names = ([f"Sample_r{i + 1}" for i in range(len(cols))] if rename
             else [chrom_line[9 + c] for c in cols])
    with open(path, "w") as f:
        f.writelines(header)
        f.write("\t".join(chrom_line[:9] + names) + "\n")
        for rec in body:
            f.write("\t".join(rec[:9] + [rec[9 + c] for c in cols]) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="1KG", choices=sorted(PANELS))
    args = ap.parse_args()
    panel_hapt, rel = PANELS[args.dataset]
    DATA = f"{ROOT}/{rel}"
    print(f"[{args.dataset}] panel {panel_hapt} -> {rel}")
    test, mask, pops, sample_ids = admixed_mask(panel_hapt, DATA)
    keep = np.where(mask)[0]
    print(f"test set: {len(test)} haplotypes -> admixed {len(keep)} "
          f"({len(keep)//2} individuals, {100*mask.mean():.1f}%)")
    for pop, n in sorted(collections.Counter(pops[mask]).items()):
        print(f"    {pop}: {n} haplotypes")

    # --- haplotype matrix, for the direct-imputation scripts ---
    np.savetxt(f"{DATA}/admixed_test.txt", test[keep], fmt="%d")
    print(f"\nwrote {DATA}/admixed_test.txt  {test[keep].shape}")

    # --- target VCF: subset the existing one so the format is bit-for-bit the same ---
    src_vcf = f"{DATA}/8020_test.vcf.gz"
    if not os.path.exists(src_vcf):
        src_vcf = f"{DATA}/8020_test.vcf"
    header, chrom_line, body = read_vcf(src_vcf)
    assert len(chrom_line) - 9 == len(test), "VCF sample count does not match the test matrix"
    out_vcf = f"{DATA}/admixed_test.vcf"
    write_vcf(out_vcf, header, chrom_line, body, keep)
    subprocess.run([BCFTOOLS, "view", "-Oz", "-o", f"{out_vcf}.gz", out_vcf], check=True)
    subprocess.run([BCFTOOLS, "index", "-f", f"{out_vcf}.gz"], check=True)
    print(f"wrote {out_vcf}(.gz)  {len(keep)} samples, {len(body)} SNPs")

    # --- bootstrap replicates over the admixed haplotypes only ---
    boot_dir = f"{DATA}/admixed_test_bootstraps"
    os.makedirs(boot_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for b in range(1, N_BOOT + 1):
        idx = rng.choice(len(keep), size=len(keep), replace=True)
        write_vcf(f"{boot_dir}/bootstrap_{b}.vcf", header, chrom_line, body, keep[idx], rename=True)
        np.savetxt(f"{boot_dir}/indices_{b}.txt", idx, fmt="%d")
    print(f"wrote {N_BOOT} bootstrap replicates + indices to {boot_dir}")

    # --- record which haplotype is which, for the methods section ---
    pd.DataFrame({"row": keep,
                  "sample": [sample_ids[i] for i in keep],
                  "population": pops[keep]}).to_csv(
        f"{DATA}/admixed_test_samples.tsv", sep="\t", index=False)
    print(f"wrote {DATA}/admixed_test_samples.tsv")


if __name__ == "__main__":
    main()
