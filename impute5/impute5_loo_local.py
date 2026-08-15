"""Local (multi-core) version of the leave-one-SNP-out Impute5 sweep.

impute5.sh / impute5_single.py drive this sweep as an SGE array job on the
cluster; this script does the same work on one multi-core box, which is where
the HMM panels are being produced. The protocol is unchanged: one SNP dropped
from the test VCF per Impute5 call, imputed against the artificial-genome
reference panel over the full buffer region, with the ALT dosage written to

    {outdir}/{method}_dosages/{chrom}:{pos}.txt

which is exactly what impute5/bootstrap_info.py consumes.

Example:
    python impute5_loo_local.py \
        --panel   ../results/1KG/8020/hmm/10K_hmm_8020_samples.vcf.gz \
        --test    ../results/1KG/8020/data/8020_test.vcf.gz \
        --chr     15 \
        --map     /scratch2/prateek/b37_recombination_maps/chr15.b37.gmap.gz \
        --method  10K_hmm_8020 \
        --workers 30
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

BCFTOOLS = os.environ.get("BCFTOOLS", "/scratch2/prateek/bcftools/bcftools")
IMPUTE5 = os.environ.get(
    "IMPUTE5", "/scratch2/prateek/impute5_v1.2.0/impute5_v1.2.0_static")

# Filled in per worker process by _init.
CTX = {}


def run(cmd, **kw):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, **kw)


def read_vcf_lines(path):
    """(header_lines, body_lines) of a possibly gzipped VCF."""
    opener = gzip.open if path.endswith(".gz") else open
    header, body = [], []
    with opener(path, "rt") as f:
        for line in f:
            (header if line.startswith("#") else body).append(line)
    return header, body


def build_panel(panel_vcf, workdir):
    """Panel -> _AC.bcf + index, done once and shared by every worker."""
    stem = os.path.join(workdir, "panel")
    ac = f"{stem}_AC.bcf"
    if not os.path.exists(ac):
        run([BCFTOOLS, "view", "-Ou", "-o", f"{stem}.bcf", panel_vcf])
        run([BCFTOOLS, "+fill-tags", f"{stem}.bcf", "-Ou", "-o", ac, "--", "-t", "AN,AC"])
        run([BCFTOOLS, "index", "-f", ac])
    return ac


def _init(header, body, panel_ac, chrom, gmap, region, outdir):
    CTX.update(header=header, body=body, panel_ac=panel_ac, chrom=chrom,
               gmap=gmap, region=region, outdir=outdir,
               region_end=int(region.split("-")[1]))


def impute_one(k):
    """Drop body line k, impute it, save the ALT dosage. Returns (snp, ok)."""
    fields = CTX["body"][k].split("\t")
    chrom, pos, snp_id = fields[0], int(fields[1]), fields[2]
    snp = f"{chrom}:{pos}"
    out_file = os.path.join(CTX["outdir"], f"{snp}.txt")
    if os.path.exists(out_file):
        return snp, True

    tmp = tempfile.mkdtemp(prefix=f"loo{k}_")
    try:
        mod = os.path.join(tmp, "modified_test.vcf")
        with open(mod, "w") as f:
            f.writelines(CTX["header"])
            f.writelines(CTX["body"][:k])
            f.writelines(CTX["body"][k + 1:])

        run([BCFTOOLS, "view", "-Ou", "-o", f"{tmp}/m.bcf", mod])
        run([BCFTOOLS, "+fill-tags", f"{tmp}/m.bcf", "-Ou", "-o",
             f"{tmp}/m_AC.bcf", "--", "-t", "AN,AC"])
        run([BCFTOOLS, "index", "-f", f"{tmp}/m_AC.bcf"])

        # Impute5 needs the target window inside the buffer. For the final SNP
        # pos+1 falls past the buffer end, so shift the window one base left --
        # the same edge case impute5_single.py handles on the cluster.
        lo, hi = pos, pos + 1
        if hi > CTX["region_end"]:
            lo, hi = lo - 1, hi - 1

        cmd = [IMPUTE5,
               "--h", CTX["panel_ac"],
               "--g", f"{tmp}/m_AC.bcf",
               "--r", f"{CTX['chrom']}:{lo}-{hi}",
               "--buffer-region", CTX["region"],
               "--o", f"{tmp}/imputed.vcf",
               "--l", f"{tmp}/imputed.log",
               "--haploid",
               "--threads", "1"]
        if CTX["gmap"]:
            cmd[3:3] = ["--m", CTX["gmap"]]
        run(cmd)

        dosage = None
        with open(f"{tmp}/imputed.vcf") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                g = line.rstrip("\n").split("\t")
                if f"{g[0]}:{g[1]}" != snp:
                    continue
                dosage = [float(s.split(":")[2].split(",")[1]) for s in g[9:]]
                break

        if dosage is None:
            return snp, False

        np.savetxt(out_file, np.asarray(dosage), fmt="%.5f")
        return snp, True
    except Exception:
        return snp, False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description="Leave-one-SNP-out Impute5 sweep, locally parallel.")
    p.add_argument("--panel", required=True, help="reference-panel VCF (diploid-homozygous AGs)")
    p.add_argument("--test", required=True, help="haploid test VCF")
    p.add_argument("--chr", required=True, type=int)
    p.add_argument("--map", default=None, help="recombination map (omit for b38 runs, as in impute5_multi.py)")
    p.add_argument("--method", required=True, help="e.g. 10K_hmm_8020; names the dosage directory")
    p.add_argument("--outdir", default="results/bootstrap", help="parent of {method}_dosages")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--workdir", default=None, help="scratch for the shared panel BCF")
    p.add_argument("--limit", type=int, default=None, help="only the first N SNPs (smoke test)")
    args = p.parse_args()

    outdir = os.path.join(args.outdir, f"{args.method}_dosages")
    os.makedirs(outdir, exist_ok=True)

    workdir = args.workdir or tempfile.mkdtemp(prefix="impute5_panel_")
    os.makedirs(workdir, exist_ok=True)

    print(f"Preparing panel {args.panel} ...", flush=True)
    panel_ac = build_panel(args.panel, workdir)

    header, body = read_vcf_lines(args.test)
    positions = [int(l.split("\t", 2)[1]) for l in body]
    region = f"{args.chr}:{min(positions)}-{max(positions)}"
    print(f"{len(body)} SNPs, buffer region {region}, {args.workers} workers -> {outdir}", flush=True)

    todo = list(range(len(body)))
    if args.limit:
        todo = todo[:args.limit]

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(header, body, panel_ac, args.chr,
                                       args.map, region, outdir)) as ex:
        futures = {ex.submit(impute_one, k): k for k in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="snp"):
            snp, ok = fut.result()
            if not ok:
                failures.append(snp)

    print(f"\ndone: {len(todo) - len(failures)}/{len(todo)} SNPs")
    if failures:
        fail_file = os.path.join(args.outdir, f"{args.method}_failed.txt")
        with open(fail_file, "w") as f:
            f.write("\n".join(failures) + "\n")
        print(f"{len(failures)} failed, listed in {fail_file} (rerun to retry; "
              f"completed SNPs are skipped)")
        sys.exit(1)


if __name__ == "__main__":
    main()
