# HMM ablation — single-SNP imputation handoff to the cluster

**Purpose of this document.** It describes everything needed to run the remaining
leave-one-SNP-out Impute5 sweeps for the HMM ablation as SGE job arrays on
`/u/scratch/p/panand2/genetic_pc`, and to bring the results back. It is written
to be handed to Claude on the cluster machine, which will need to reconcile the
script differences described in §5 — the pipeline files there are the same ones
but may be named or arranged differently.

Last updated: 2026-08-05.

---

## 0. Checklist

In order. Sections in brackets have the detail.

1. Copy the ten reference panels and gunzip them [§4a]. **Rename the combined
   ones** `..._samples_combined` → `..._combined_samples` [§4a naming caveat].
2. Check whether the six test VCFs are already on the cluster; copy only what is
   missing [§4b].
3. Copy `make_missing_mapfile.py` — **it is a new file and is not on the cluster**
   [§4d]. No mapfiles need copying [§4c].
4. Fix the `BCFTOOLS_PLUGINS` bug in `impute5_single.py` — it fails *silently*
   and will corrupt every panel if missed [§5d].
5. Reconcile `impute5.sh` (script name, and parameterise dataset / split / test
   prefix / chrnum) and `submit_bootstraps.sh` (method list, SNP counts) [§5a–c].
6. Run the submit → check → resubmit loop for all ten sweeps [§6]. Sweeps 1 and 6
   (`10K_hmm_8020`, `UKBB_hmm_8020`) are seeded with `--all` because they have
   pre-existing workstation output that is being discarded [§3].
7. Verify line counts, then copy the dosage directories back to the workstation
   [§6 verification, §7].

Everything else — model training, AG sampling, panel construction, the direct
imputation arm, and final CSV assembly — happens on the workstation.

---

## 1. What this work is

Reviewer 2, comment 2 asks for HMM arms in the imputation benchmarks. The HMM is
not an external baseline: it is the same GPC/PyJuice framework with the latent
graph swapped from the learned Chow-Liu tree to a chain, so it isolates the
contribution of the tree topology. Two arms are reported per config, mirroring
how GPC is already reported:

| arm | what it is | where it runs |
|---|---|---|
| **HMM (direct)** | conditional imputation straight from the chain model, no artificial genomes and no Impute5 | **workstation, GPU** — nothing needed from the cluster |
| **HMM (AG panel)** | artificial genomes used as an Impute5 reference panel, leave-one-SNP-out | **cluster** — this document |

The direct arm is being handled locally and is not the cluster's concern.

## 2. Division of labour

**Workstation** (`/scratch2/prateek/genetic_pc_github`): trains the models,
samples the artificial genomes, builds the reference panel VCFs, runs the direct
arm, and assembles the final R² CSVs.

**Cluster**: the ten leave-one-SNP-out sweeps below — about **99,100 SNP-jobs**
at ~390 s each, roughly **10,700 CPU-hours**. This is why it belongs on a job
array: it is ~12 days wall-clock on the 32-core workstation and the per-SNP cost
is dominated by Impute5 itself (~86% of each job), not by pipeline overhead, so
the only real lever is core count.

## 3. The ten sweeps

`chr` is 15 for 1KG, 22 for UKBB. All ten sweeps are to be run **in full on the
cluster**, including the two that already have workstation output — see the
consistency note below. Track what is left with `make_missing_mapfile.py` (§6)
before each submission round.

| # | method (`METHOD_FULL`) | panel file | test VCF | chr | SNPs |
|---|---|---|---|---|---|
| 1 | `10K_hmm_8020` | `10K_hmm_8020_samples.vcf` | `1KG/8020/data/8020_test.vcf` | 15 | 10000 |
| 2 | `10K_hmm_afr` | `10K_hmm_afr_samples.vcf` | `1KG/afr/data/afr_test.vcf` | 15 | 10000 |
| 3 | `10K_hmm_afr_combined` | `10K_hmm_afr_combined_samples.vcf` | `1KG/afr/data/afr_test.vcf` | 15 | 10000 |
| 4 | `10K_hmm_noneur` | `10K_hmm_noneur_samples.vcf` | `1KG/noneur/data/noneur_test.vcf` | 15 | 10000 |
| 5 | `10K_hmm_noneur_combined` | `10K_hmm_noneur_combined_samples.vcf` | `1KG/noneur/data/noneur_test.vcf` | 15 | 10000 |
| 6 | `UKBB_hmm_8020` | `UKBB_hmm_8020_samples.vcf` | `UKBB/8020/data/8020_test.vcf` | 22 | 9820 |
| 7 | `UKBB_hmm_afr` | `UKBB_hmm_afr_samples.vcf` | `UKBB/afr/data/afr_test.vcf` | 22 | 9820 |
| 8 | `UKBB_hmm_afr_combined` | `UKBB_hmm_afr_combined_samples.vcf` | `UKBB/afr/data/afr_test.vcf` | 22 | 9820 |
| 9 | `UKBB_hmm_noneur` | `UKBB_hmm_noneur_samples.vcf` | `UKBB/noneur/data/noneur_test.vcf` | 22 | 9820 |
| 10 | `UKBB_hmm_noneur_combined` | `UKBB_hmm_noneur_combined_samples.vcf` | `UKBB/noneur/data/noneur_test.vcf` | 22 | 9820 |

**Total: 99,100 SNP-jobs** (5 × 10000 + 5 × 9820).

`_combined` means the AG panel merged with the real European haplotypes, the
same construction as `combine_vcf.sh` / `10K_hclt_afr_samples_final_combined.vcf.gz`.

### Why sweeps 1 and 6 are re-run from scratch

Two sweeps already have workstation output, produced by `impute5_loo_local.py`:
`10K_hmm_8020` complete at 10000/10000, and `UKBB_hmm_8020` partial at 4002/9820.
Both are being **discarded and re-run here**, so that every number in the paper's
HMM AG-panel arm comes from one pipeline on one machine.

That is a deliberate choice, not a correctness fix. The two code paths were
checked and are equivalent — same map, same `--haploid`, same buffer region, same
`GP[1]` extraction, same `%.5f` output (§7). What was *not* verified is that the
two machines produce bit-identical numbers: the workstation ran Impute5 v1.2.0
with bcftools 1.21-38-g4f8bf539, and the cluster's builds may differ. Re-running
removes that variable.

For `UKBB_hmm_8020` the argument is stronger: leaving it as-is would make a
single sweep a 41%/59% mix of two pipelines, which is harder to defend than one
sweep from a different machine.

**Operationally this means**: do not use the plain `make_missing_mapfile.py`
invocation to seed these two — it would report 0 and 5818 missing respectively.
Use `--all` (§6), and archive the workstation dosage directories so the array
tasks do not skip SNPs whose files already exist.

## 4. Files to copy to the cluster

Panels are gzipped for transfer (a few MB each). **`process_plink_data()` in
`impute5_single.py` reads an uncompressed `{prefix}.vcf`**, so gunzip after
copying — otherwise it will try to run `plink2 --bfile` and fail.

### 4a. Reference panels — the actual deliverable from the workstation

Ten panels, one per sweep. From `/scratch2/prateek/genetic_pc_github/results/`:

```
1KG/8020/hmm/10K_hmm_8020_samples.vcf.gz                    3.0M   ready
1KG/afr/hmm/10K_hmm_afr_samples.vcf.gz                      3.7M   ready
1KG/afr/hmm/10K_hmm_afr_samples_combined.vcf.gz             4.6M   ready
UKBB/8020/hmm/UKBB_hmm_8020_samples.vcf.gz                  5.0M   ready
1KG/noneur/hmm/10K_hmm_noneur_samples.vcf.gz                       pending training
1KG/noneur/hmm/10K_hmm_noneur_samples_combined.vcf.gz              pending training
UKBB/afr/hmm/UKBB_hmm_afr_samples.vcf.gz                           pending training
UKBB/afr/hmm/UKBB_hmm_afr_samples_combined.vcf.gz                  pending training
UKBB/noneur/hmm/UKBB_hmm_noneur_samples.vcf.gz                     pending training
UKBB/noneur/hmm/UKBB_hmm_noneur_samples_combined.vcf.gz            pending training
```

The two `_8020` panels are the ones the finished/partial workstation sweeps used;
they are included because sweeps 1 and 6 are being re-run here (§3).

(The `.csi` indexes are not needed — the cluster rebuilds `.bcf`/`_AC.bcf` itself.)

> **Naming caveat.** `impute5.sh` derives the panel path as
> `${METHOD_FULL}_samples`, so for `METHOD_FULL=10K_hmm_afr_combined` it will
> look for `10K_hmm_afr_combined_samples.vcf` — but the workstation writes
> `10K_hmm_afr_samples_combined.vcf.gz`. **Rename the combined panels on arrival**
> (`..._samples_combined` → `..._combined_samples`), or change the path
> construction in `impute5.sh`. Renaming is less invasive.

> **Panel sample naming.** AG haplotypes are named `Sample_h1..Sample_h{N}` and
> the real-EUR block `Sample_1..Sample_1004`, so a combined panel is
> `Sample_1..Sample_1004` followed by `Sample_h1..Sample_h5008` — byte-for-byte
> the same layout as the published `10K_hclt_afr_samples_final_combined.vcf.gz`
> (6012 samples). This matters: `make_hmm_panels.py` originally emitted plain
> `Sample_N` for the AGs, which collides with the real-EUR names and makes
> `bcftools merge` abort with *"Duplicate sample names (Sample_1)"*. It has been
> fixed on the workstation. Panel sample names do not affect imputation output,
> so the already-finished `10K_hmm_8020` sweep (built under the old naming) is
> unaffected.

### 4b. Test VCFs — only if not already on the cluster

These are the real held-out haplotypes and are unchanged from the HCLT/RBM/WGAN
runs, so they are probably already there. Verify before copying (the UKBB ones
are large):

```
1KG/8020/data/8020_test.vcf.gz        904K
1KG/afr/data/afr_test.vcf.gz          264K
1KG/noneur/data/noneur_test.vcf.gz    736K
UKBB/8020/data/8020_test.vcf          102M
UKBB/afr/data/afr_test.vcf             38M
UKBB/noneur/data/noneur_test.vcf       38M
```

### 4c. Mapfiles

None need copying. Every sweep — including the two being re-run — is seeded on
the cluster with `make_missing_mapfile.py` (§6); for sweeps 1 and 6 use `--all`.

The workstation's `impute5/rerun/bootstrap/impute_UKBB_hmm_8020_correct_map.txt`
(5,818 lines) is now **obsolete**: it was the resume list for continuing that
sweep, which is no longer the plan. Ignore it.

### 4d. Scripts — one new file to copy

```
impute5/make_missing_mapfile.py
```

**This file is new and does not exist on the cluster.** It is written for this
handoff and replaces the log-parsing half of `extract.py` (see §6 for why).
Standard library only — no pandas, no numpy — so it runs under any Python 3
without loading a conda env. Copy it next to `extract.py`.

Nothing else is needed. Model checkpoints, training data, the `.txt`/`.hapt` AG
files, and all of `pc/` stay on the workstation.

### 4e. Workstation changes you will not see reflected on the cluster

Two files were edited here during this work. Neither needs to be copied — both
are workstation-side only — but they explain the provenance of what you are
receiving:

- `aux/scripts/make_hmm_panels.py` — AG sample naming changed to `Sample_h{i}`
  and the combined merge reordered to real-EUR-first, so the combined panels
  match the published HCLT layout (see the naming note in §4a). This is why the
  panels you get are structurally identical to
  `10K_hclt_afr_samples_final_combined.vcf.gz`.
- `pc/hmm.py`, `pc/hmm_queue.py`, `pc/impute5_queue.py` — training and panel-prep
  orchestration. Entirely local; irrelevant to the cluster.

`impute5/impute5_loo_local.py` is the workstation's local-parallel implementation
of the same sweep. It is what produced the finished `10K_hmm_8020` results and
the partial `UKBB_hmm_8020` ones. **Do not run it on the cluster** — it is a
`ProcessPoolExecutor` design meant for one fat node, not a job array. It is
useful only as the reference for the output contract in §7.

## 5. Reconciling the cluster scripts

The existing pipeline is sound and its **output is already byte-compatible** with
what the workstation expects (§7). These are the deltas to fix.

**a. `impute5.sh` calls a script that may be named differently.** It runs
`python3 impute5.py ...`; the workstation copy is `impute5_single.py`. Confirm
which name exists on the cluster. The CLI contract is
`argv = [k, threads, train_prefix, test_prefix, chrnum, method_full]`, `k`
1-based.

**b. `impute5.sh` hardcodes the 1KG 8020 paths.** Currently:

```sh
train=results/1KG/8020/${METHOD}/${METHOD_FULL}_samples
test=../results/1KG/8020/data/8020_test
chrnum=15
```

These must become per-sweep. `METHOD=$(echo "$b" | cut -d'_' -f2)` already
yields `hmm` for every method name in §3, so only the **dataset directory**
(`1KG`/`UKBB`), the **split directory** (`8020`/`afr`/`noneur`), the **test
prefix** and **chrnum** need parameterising. Pass them through `qsub -v`
alongside `METHOD`/`METHOD_FULL`/`MAPFILE`.

**c. `submit_bootstraps.sh` hardcodes the old methods and SNP count.** It loops
over `10K_real_8020 10K_hclt_8020 10K_rbm_8020 10K_wgan_8020` and seeds mapfiles
with `seq 1 10000`. Replace the list with the §3 methods and use 9820 for UKBB.
Keep the `-t 1-${num_lines}:100` step: `impute5.sh` walks 100 indices per task
internally, so step and inner loop must stay in sync.

**d. Bug: `BCFTOOLS_PLUGINS` points at a mistyped local path.** In
`impute5_single.py`:

```python
os.environ['BCFTOOLS_PLUGINS'] = '/scrach2/prateek/bcftools/plugins'
```

Note `scrach2` (typo) and that it is a *workstation* path. On the cluster this
leaves the plugin path unresolvable, so `bcftools +fill-tags` fails. Because
`process_plink_data()` does not check the return code, the failure is silent and
you get an `_AC.bcf` without AN/AC rather than an error. Set this to the
cluster's plugin directory, and consider adding `check=True` to those
`subprocess.run` calls.

**e. Everything else can stay.** The map path is already parameterised by
`chrnum` (`chr{chrnum}.b37.gmap.gz`), `--haploid` is correct, and the buffer
region is derived from the test VCF's min/max positions.

## 6. Running it — the submit / check / resubmit loop

Your existing loop is *submit → `extract.py` → resubmit the missing → repeat*,
because tasks fail randomly. **That loop is the right shape** — with tens of
thousands of array tasks, some always die, and iterating to convergence is the
practical answer. Keep it. One change is worth making to how "done" is decided.

`extract.py` infers completion by parsing the SGE stdout logs in
`out/impute_{method}/` for the two printed r² values. That is a proxy for the
thing that matters, and it fails in both directions:

- a task that **wrote its dosage file** but had stdout truncated (killed node,
  full disk, log rotation) is counted missing and re-run for nothing;
- a task that **printed its r² values but died before `np.savetxt`** is counted
  done, so the gap is never noticed and `bootstrap_to_r2.py` silently sees fewer
  SNPs than it should. This is the dangerous one — it produces a quietly
  incomplete result rather than a loud failure.

`extract.py` also `os.remove()`s log files it considers duplicates, which
destroys the evidence you would need to diagnose a systematic failure, and it
hardcodes `aux/10K_SNP.legend` plus the four original method names, so it cannot
be pointed at UKBB or the HMM arms without editing.

**Use `make_missing_mapfile.py` instead** — the new file from §4d, which you need
to copy over first; it is not on the cluster yet. It decides from
`results/bootstrap/{method}_dosages/` — the artifact `bootstrap_to_r2.py`
actually consumes — so "done" means the dosage file exists. It is
dataset-agnostic, standard-library only, and emits the same 1-based test-VCF
indices the array expects. It writes
`{--rerun-dir}/impute_{method}_correct_map.txt`, exactly the path and format
`submit_bootstraps.sh` already reads, so it is a drop-in for the mapfile half of
`extract.py`.

The same command **seeds** a fresh sweep (every index is missing, so you get all
10000 / 9820 lines) and **refreshes** a partial one — there is no separate
"first submission" path, and `submit_bootstraps.sh`'s `seq 1 10000` fallback
becomes unnecessary.

**Sweeps 1 and 6 need `--all` on the first pass.** `10K_hmm_8020` and
`UKBB_hmm_8020` arrive with workstation dosage files already present (10000 and
4002), so the plain invocation would report 0 and 5818 missing and you would not
re-run them as intended (§3). Archive the old directory and seed with `--all`:

```sh
mv results/bootstrap/10K_hmm_8020_dosages results/bootstrap/10K_hmm_8020_dosages.workstation
python3 make_missing_mapfile.py --method 10K_hmm_8020 \
    --test ../results/1KG/8020/data/8020_test.vcf --all      # -> 10000 lines
```

The `mv` matters as much as the `--all`: each array task returns early if its
dosage file already exists, so leaving the old directory in place would make
every task a no-op regardless of what the mapfile says. After the first pass,
drop `--all` and use the normal loop below — otherwise each round would resubmit
all 10000 SNPs instead of just the failures.

Worked example for one sweep, start to finish. Adjust the `--test` path and the
`qsub -v` variables to the cluster's layout (§5b):

```sh
cd /u/scratch/p/panand2/genetic_pc/impute5

M=UKBB_hmm_afr
TEST=../results/UKBB/afr/data/afr_test.vcf

while :; do
    # 1. what is still missing? (seeds on the first pass)
    python3 make_missing_mapfile.py --method $M --test $TEST
    MAP=rerun/bootstrap/impute_${M}_correct_map.txt
    N=$(wc -l < $MAP)
    [ "$N" -eq 0 ] && { echo "$M complete"; break; }

    # 2. submit; impute5.sh walks 100 indices per task, hence the :100 step
    echo "submitting $N remaining SNPs for $M"
    qsub -v METHOD=hmm,METHOD_FULL=$M,MAPFILE=$MAP,\
DATASET=UKBB,SPLIT=afr,TESTPREFIX=../results/UKBB/afr/data/afr_test,CHRNUM=22 \
        -t 1-${N}:100 impute5.sh

    # 3. wait for the array to drain, then loop
    while qstat -u $USER 2>/dev/null | grep -q impute_; do sleep 300; done
done
```

Expect it to take 2–4 rounds. If a round makes no progress at all (`N` unchanged),
something systematic is wrong — check a task log in `out/impute_${M}/` rather
than resubmitting; the most likely cause is the `BCFTOOLS_PLUGINS` bug in §5d.

`extract.py` is still useful for the per-SNP r²/r²-geno CSV it writes from the
logs, if you want that diagnostic. Just don't use it to decide what to re-run.

### Verifying a sweep before you ship it back

`make_missing_mapfile.py` only checks that each dosage file *exists*. A truncated
write would pass. Before declaring a sweep done, confirm every file has the
expected number of lines (§7 lists them per method):

```sh
M=UKBB_hmm_afr; EXPECT=2000
find results/bootstrap/${M}_dosages -name '*.txt' \
  | xargs -P 8 -n 100 wc -l \
  | awk -v e=$EXPECT '$2!="total" && $1!=e {print; n++} END {print (n?n:0)" bad files"}'
```

Delete any that come back wrong and run one more round — they will be picked up
as missing.

Note the index mapping is safe: `aux/10K_SNP.legend` and `aux/UKBB_SNP.legend`
were both verified to be in the **same order** as their test VCFs (all three UKBB
test sets included), so legend-derived and VCF-derived indices agree.

## 7. Output contract — what must come back

Both the cluster path and the workstation path write **the same files**, which is
what makes the already-finished `10K_hmm_8020` sweep and the cluster sweeps
mixable. Verified equivalences:

| | cluster (`impute5_single.py`) | workstation (`impute5_loo_local.py`) |
|---|---|---|
| output dir | `results/bootstrap/{method_full}_dosages/` | same |
| filename | `{rs_id}.txt`, rs_id = VCF ID column | `{chrom}:{pos}.txt` |
| value | `genotype.split(':')[2]` → `gp.split(',')[1]` | `split(':')[2].split(',')[1]` |
| format | `np.savetxt(..., fmt="%.5f")` | same |

The ID column in every test VCF is already `chrom:pos` (e.g. `15:27379578`), so
the two filename conventions coincide exactly.

**Each file is one line per test haplotype**, the imputed P(allele=1). Expected
line counts: 1KG afr 266, 1KG noneur 802, UKBB 8020 5384, UKBB afr 2000, UKBB
noneur 2000.

To bring results back, copy the dosage directories to the workstation at
`impute5/results/bootstrap/{method}_dosages/`. The workstation then runs
`bootstrap_to_r2.py` to produce the R² CSVs that `plots/impute/plot.ipynb` reads.

## 8. Where this sits in the overall deliverable

20 result CSVs total: 10 direct + 10 AG-panel.

The **10 direct CSVs** are entirely workstation work; two are done
(`pc_hmm_8020_8020`, `UKBB_pc_hmm_8020_bootstraps`) and the rest follow from the
GPU queue there. Nothing about the direct arm depends on the cluster.

The **10 AG CSVs** all come from the §3 sweeps. `hmm_8020_8020.csv` already exists
on the workstation from the local run, but it will be **regenerated** from the
cluster's sweep-1 dosages once they come back, so treat all ten as pending.

(The reviewer-response doc says "14 new result files" — that is an undercount;
the actual count implied by the stated scope, 2 datasets × {general, non-European,
African} × {baseline, combined} × 2 arms, is 20. Worth reconciling before the
response letter goes out.)
