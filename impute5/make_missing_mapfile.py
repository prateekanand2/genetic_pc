"""Build the job-array mapfile of SNPs that still need imputing.

Replaces the log-parsing half of extract.py. extract.py decides which SNPs are
done by parsing the SGE stdout logs in out/impute_{method}/ for the two printed
r^2 values; this decides from results/bootstrap/{method}_dosages/, which is the
artifact bootstrap_to_r2.py actually consumes. That matters in both directions:

  * a task that wrote its dosage file but had its stdout truncated (killed node,
    full disk, log rotation) is counted missing by extract.py and re-run for
    nothing;
  * a task that printed its r^2 values but died before np.savetxt is counted
    DONE by extract.py, so the gap is never noticed and bootstrap_to_r2.py
    silently sees fewer SNPs than it should.

It is also dataset-agnostic -- extract.py hardcodes aux/10K_SNP.legend and the
four original method names, so it cannot be pointed at UKBB or the HMM arms
without editing.

The emitted indices are 1-based positions into the test VCF body, which is what
impute5_single.py indexes (batch_idx = k = argv[1] - 1, snp = rs_ids[k]).

Usage:
    python make_missing_mapfile.py --method 10K_hmm_afr \
        --test ../results/1KG/afr/data/afr_test.vcf \
        [--outdir results/bootstrap] [--rerun-dir rerun/bootstrap]
"""

import argparse
import gzip
import os


def read_body(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return [l for l in fh if not l.startswith("#")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, help="e.g. 10K_hmm_afr")
    p.add_argument("--test", required=True, help="test VCF (.vcf or .vcf.gz)")
    p.add_argument("--outdir", default="results/bootstrap",
                   help="parent of {method}_dosages")
    p.add_argument("--rerun-dir", default="rerun/bootstrap")
    p.add_argument("--all", action="store_true",
                   help="emit every SNP index regardless of which dosage files already "
                        "exist. Use to force a deliberate full re-run -- e.g. re-running a "
                        "sweep that was produced by a different pipeline -- without having "
                        "to delete the existing results first. Note impute_one()/the array "
                        "task will still skip a SNP whose dosage file is present, so move "
                        "or archive the old {method}_dosages directory as well.")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    body = read_body(a.test)
    dos = os.path.join(a.outdir, f"{a.method}_dosages")
    have = set() if a.all else (set(os.listdir(dos)) if os.path.isdir(dos) else set())

    missing = []
    for i, line in enumerate(body, 1):          # 1-based, matches SGE_TASK_ID
        chrom, pos = line.split("\t", 2)[:2]
        if f"{chrom}:{pos}.txt" not in have:
            missing.append(i)

    os.makedirs(a.rerun_dir, exist_ok=True)
    out = os.path.join(a.rerun_dir, f"impute_{a.method}_correct_map.txt")
    with open(out, "w") as fh:
        if missing:
            fh.write("\n".join(map(str, missing)) + "\n")

    if not a.quiet:
        print(f"{a.method}: {len(body)} SNPs, {len(body)-len(missing)} done, "
              f"{len(missing)} missing -> {out}")
    return 0 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
