#!/usr/bin/env python3

import numpy as np
import sys

# Usage:
# python make_maf_legend.py train.txt test.txt legend.txt output.legend

train_file = sys.argv[1]
test_file = sys.argv[2]
legend_file = sys.argv[3]
output_file = sys.argv[4]

print("Loading train data...")
train = np.loadtxt(train_file, dtype=np.float64)

print("Loading test data...")
test = np.loadtxt(test_file, dtype=np.float64)

print("Combining datasets...")
combined = np.vstack([train, test])  # shape: (num_haplotypes, num_snps)

num_haps, num_snps = combined.shape
print(f"Total haplotypes: {num_haps}")
print(f"Total SNPs: {num_snps}")

print("Computing allele frequencies...")
# Since rows are haplotypes (0/1), allele frequency = mean across rows
af = combined.mean(axis=0)

# Minor allele frequency
maf = np.minimum(af, 1.0 - af)

print("Reading legend file...")
with open(legend_file, "r") as f:
    header = f.readline()  # skip header
    legend_lines = f.readlines()

if len(legend_lines) != num_snps:
    raise ValueError(
        f"Mismatch: legend has {len(legend_lines)} SNPs but data has {num_snps}"
    )

print("Writing new legend with MAF...")
with open(output_file, "w") as out:
    for i, line in enumerate(legend_lines):
        parts = line.strip().split()
        snp_id = parts[0]      # e.g. chr2:118367466_A_G
        pos = parts[1]         # 118367466

        # Extract chromosome from snp_id
        chrom = snp_id.split(":")[0]  # chr2
        chrom = chrom.replace("chr", "")  # 2

        out.write(f"{chrom}:{pos}\t{maf[i]:.6f}\n")

print("Done.")