"""Turn HMM artificial genomes into Impute5 reference panels.

Produces, for one dataset/split:
    {sample_prefix}.vcf.gz(+.csi)           the AG-only panel
    {sample_prefix}_combined.vcf.gz(+.csi)  AG panel merged with real EUR

Genotypes are written diploid-homozygous ("0|0"/"1|1"), matching
aux/scripts/create_vcf.py, because Impute5 is run in --haploid mode against a
diploid panel. The combined panel is the bcftools merge used by
aux/scripts/combine_vcf.sh, i.e. the same construction as
10K_hclt_afr_samples_final_combined.vcf.gz.

Usage:
    python make_hmm_panels.py 1KG:afr [--combined]
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pc"))
from hmm_config import resolve  # noqa: E402

BCFTOOLS = os.environ.get("BCFTOOLS", "/scratch2/prateek/bcftools/bcftools")


def read_legend(path):
    sites = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            snp_id = parts[0].split("_")[0]
            chrom = snp_id.split(":")[0]
            if chrom.lower().startswith("chr"):
                chrom = chrom[3:]
                snp_id = snp_id.replace("chr", "", 1)
            sites.append((chrom, parts[1], snp_id, parts[2], parts[3]))
    return sites


def write_panel_vcf(hap, sites, out_vcf, chrom, reference="hg38"):
    """hap is (n_samples, n_snps); the VCF is transposed on the fly."""
    n = hap.shape[0]
    assert hap.shape[1] == len(sites), f"{hap.shape[1]} SNPs vs {len(sites)} legend rows"
    # "Sample_h{i}", matching the published AG panels (10K_hclt_afr_samples_final
    # uses Sample_h1..Sample_h5008). The h prefix is what keeps these distinct
    # from the real-EUR panel's Sample_1..Sample_1004, so that the combined
    # panel can be built with a plain `bcftools merge` -- without it the merge
    # aborts on duplicate sample names.
    samples = [f"Sample_h{i+1}" for i in range(n)]

    # Precompute the two possible genotype strings; the panel is homozygous.
    gt = np.array(["0|0", "1|1"])

    with open(out_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=my_haplotype_data\n")
        f.write(f"##reference={reference}\n")
        f.write(f"##contig=<ID={chrom}>\n")
        f.write('##INFO=<ID=PR,Number=0,Type=Flag,Description="Variant is '
                'present in reference panel">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(samples) + "\n")
        for j, (c, pos, snp_id, ref, alt) in enumerate(sites):
            f.write(f"{c}\t{pos}\t{snp_id}\t{ref}\t{alt}\t.\t.\tPR\tGT\t"
                    + "\t".join(gt[hap[:, j]]) + "\n")


def bgzip_index(vcf):
    gz = f"{vcf}.gz"
    subprocess.run([BCFTOOLS, "view", "-Oz", "-o", gz, vcf], check=True)
    subprocess.run([BCFTOOLS, "index", "-f", gz], check=True)
    # Keep the uncompressed copy: impute5_multi.py takes a prefix and expects
    # "{prefix}.vcf", falling back to plink2 if it is absent.
    return gz


def main():
    p = argparse.ArgumentParser(description="Build HMM Impute5 reference panels.")
    p.add_argument("config", help="dataset/split key, e.g. 1KG:afr")
    p.add_argument("--combined", action="store_true",
                   help="also merge with the real EUR panel")
    p.add_argument("--samples", default=None, help="override the AG .txt path")
    args = p.parse_args()

    cfg = resolve(args.config)
    txt = args.samples or f"{cfg['sample_prefix']}.txt"
    if not os.path.exists(txt):
        raise SystemExit(f"missing AGs: {txt} (run pc/generate_hmm.py first)")

    hap = np.loadtxt(txt, dtype=np.int8)
    sites = read_legend(cfg["snp_legend"])
    print(f"{txt}: {hap.shape} -> {len(sites)} sites")

    vcf = f"{cfg['sample_prefix']}.vcf"
    write_panel_vcf(hap, sites, vcf, cfg["chrom"])
    gz = bgzip_index(vcf)
    print(f"wrote {gz}")

    if args.combined:
        out = f"{cfg['sample_prefix']}_combined.vcf.gz"
        # Real EUR first, then the AGs -- the sample order of
        # 10K_hclt_afr_samples_final_combined.vcf.gz (1004 real, then 5008 AG).
        # Order does not affect imputation, but keeps the panels comparable.
        subprocess.run([BCFTOOLS, "merge", cfg["real_eur_vcf"], gz,
                        "-Oz", "-o", out], check=True)
        subprocess.run([BCFTOOLS, "index", "-f", out], check=True)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
