"""Shared configuration for the HMM ablation (chain-structured GPC).

One entry per (dataset, split). Every downstream HMM script -- training
(``hmm.py``), sampling (``generate_hmm.py``) and direct imputation
(``predict_hmm.py``) -- reads its paths from here so that the file layout
matches what already exists on disk for HCLT/RBM/WGAN.

Naming conventions (taken from the files already produced for 1KG/UKBB 8020):
    model  results/{data}/{dir}/hmm/pc_{tag}_{chunk}_{split}_hmm_{n}-128_{E}epochs_ps0.005.jpc
    log    results/{data}/{dir}/hmm/{tag}_{chunk}_{split}_hmm_{n}_128_{E}epochs_ps0.005.log
    AGs    results/{data}/{dir}/hmm/{sample_tag}_hmm_{dir}_samples.txt
"""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Per-dataset constants.
DATASETS = {
    "1KG": dict(tag="10K", sample_tag="10K", epochs=5000, num_snps=10000, num_chunks=4,
                legend_maf=f"{ROOT}/aux/10K_legend.maf.txt",
                real_eur_vcf=f"{ROOT}/aux/10K_real_eur.vcf.gz",
                snp_legend=f"{ROOT}/aux/10K_SNP.legend",
                chrom=15, num_ags=5008),
    "UKBB": dict(tag="ukbb", sample_tag="UKBB", epochs=2000, num_snps=9820, num_chunks=4,
                 legend_maf=f"{ROOT}/aux/UKBB_legend.maf.txt",
                 real_eur_vcf=f"{ROOT}/aux/UKBB_real_eur.vcf.gz",
                 snp_legend=f"{ROOT}/aux/UKBB_SNP.legend",
                 chrom=22, num_ags=10000),
    # High-coverage 1KG (build 38). 14,670 SNPs is not divisible by 4, so this
    # dataset uses 6 chunks of 2,445 -- almost exactly the UKBB chunk width
    # (2,455), which keeps per-chunk model capacity comparable across datasets.
    "b38": dict(tag="14670", sample_tag="b38", epochs=5000, num_snps=14670, num_chunks=6,
                legend_maf=f"{ROOT}/aux/b38_legend.maf.txt",
                real_eur_vcf=f"{ROOT}/aux/b38_real_eur.vcf.gz",
                snp_legend=f"{ROOT}/aux/b38_SNP.legend",
                chrom=15, num_ags=5008),
}

# (dataset, split) -> training file, number of training haplotypes, and the
# split directory / test set the model is evaluated against.
#
# ``dir`` is the results sub-directory (8020 / afr / noneur); ``split`` is the
# token that goes into the model filename. For the "combined" direct-imputation
# models the training set is real EUR + the target population, mirroring the
# GPC models pc_10K_eur_and_afr_train_2060-128... etc.
_SPLITS = {
    ("1KG", "8020"):                  dict(dir="8020",   n_train=4006, test="8020_test"),
    ("1KG", "afr"):                   dict(dir="afr",    n_train=1056, test="afr_test"),
    ("1KG", "noneur"):                dict(dir="noneur", n_train=3202, test="noneur_test"),
    ("1KG", "eur_and_afr_train"):     dict(dir="afr",    n_train=2060, test="afr_test"),
    ("1KG", "eur_and_noneur_train"):  dict(dir="noneur", n_train=4206, test="noneur_test"),

    # 50/50 split used for the AATS privacy evaluation (Section 2.7); the AG
    # count matches the other methods' *_aats_samples.txt, which hold two
    # blocks of num_syn haplotypes (one scored against train, one against test).
    ("1KG", "aats"):                  dict(dir="aats",   n_train=2504,  test="aats_test"),

    ("UKBB", "8020"):                 dict(dir="8020",   n_train=21540, test="8020_test"),
    ("UKBB", "afr"):                  dict(dir="afr",    n_train=8000,  test="afr_test"),
    ("UKBB", "noneur"):               dict(dir="noneur", n_train=8000,  test="noneur_test"),
    ("UKBB", "eur_and_afr_train"):    dict(dir="afr",    n_train=18000, test="afr_test"),
    ("UKBB", "eur_and_noneur_train"): dict(dir="noneur", n_train=18000, test="noneur_test"),

    ("UKBB", "aats"):                 dict(dir="aats",   n_train=13462, test="aats_test"),

    ("b38", "8020"):                  dict(dir="8020",   n_train=4006, test="8020_test"),
    ("b38", "afr"):                   dict(dir="afr",    n_train=1056, test="afr_test"),
    ("b38", "noneur"):                dict(dir="noneur", n_train=3202, test="noneur_test"),
    ("b38", "eur_and_afr_train"):     dict(dir="afr",    n_train=2060, test="afr_test",
                                           train_override="aux/b38_real_eur_and_afr_train.txt"),
    ("b38", "eur_and_noneur_train"):  dict(dir="noneur", n_train=4206, test="noneur_test",
                                           train_override="aux/b38_real_eur_and_noneur_train.txt"),
}


def _build(dataset, split, spec):
    ds = DATASETS[dataset]
    data_dir = f"{ROOT}/results/{dataset}/{spec['dir']}/data"
    model_dir = f"{ROOT}/results/{dataset}/{spec['dir']}/hmm"

    # 8020/afr/noneur train on "{split}_train.txt"; the combined models train on
    # the pre-built "eur_and_*_train.txt" files that live in the same data dir.
    train_name = split if split.endswith("_train") else f"{split}_train"
    train_path = (f"{ROOT}/{spec['train_override']}" if spec.get("train_override")
                  else f"{data_dir}/{train_name}.txt")

    return dict(
        key=f"{dataset}:{split}",
        dataset=dataset,
        split=split,
        dir=spec["dir"],
        tag=ds["tag"],
        sample_tag=ds["sample_tag"],
        epochs=ds["epochs"],
        num_snps=ds["num_snps"],
        num_chunks=ds.get("num_chunks", 4),
        num_ags=ds["num_ags"],
        chrom=ds["chrom"],
        legend_maf=ds["legend_maf"],
        real_eur_vcf=ds["real_eur_vcf"],
        snp_legend=ds["snp_legend"],
        n_train=spec["n_train"],
        data_dir=data_dir,
        model_dir=model_dir,
        train_path=train_path,
        valid_path=f"{data_dir}/{spec['test']}.txt",
        test_prefix=f"{data_dir}/{spec['test']}",
        bootstrap_dir=f"{data_dir}/test_bootstraps",
        # The population-only models keep the original {tag}_hmm_{dir}_samples
        # naming, which the existing AG panels and R2 files already use. The
        # combined (eur_and_*) models get their own prefix keyed on the split:
        # they live in the same results directory as their population-only
        # counterpart, so sharing a prefix would silently overwrite that
        # config's AGs and panels.
        sample_prefix=(
            f"{model_dir}/{ds['sample_tag']}_hmm_{spec['dir']}_samples"
            if not split.endswith("_train")
            else f"{model_dir}/{ds['sample_tag']}_hmm_{split}_samples"),
    )


CONFIGS = {f"{d}:{s}": _build(d, s, spec) for (d, s), spec in _SPLITS.items()}


def resolve(key):
    if key not in CONFIGS:
        raise SystemExit(f"unknown config '{key}'. choose from:\n  " +
                         "\n  ".join(sorted(CONFIGS)))
    return CONFIGS[key]


R2_DIR = f"{ROOT}/plots/impute/results/r2"

# The notebook keys off file names, and the two datasets use different suffixes:
#   1KG   hclt_afr_afr.csv          pc_real_afr_afr.csv
#   UKBB  UKBB_hclt_afr_bootstraps.csv   UKBB_pc_real_afr_bootstraps.csv
# The HMM files mirror that, with hclt -> hmm and pc_real -> pc_hmm.


def r2_csv(cfg, arm, combined=False):
    """Destination CSV for an HMM result. arm is 'ag' (Impute5 panel) or 'direct'."""
    assert arm in ("ag", "direct")
    stem = "hmm" if arm == "ag" else "pc_hmm"
    name = f"{stem}_{cfg['split']}" + ("_combined" if combined else "")
    if cfg["dataset"] == "1KG":
        return f"{R2_DIR}/{name}_{cfg['dir']}.csv"
    if cfg["dataset"] == "UKBB":
        return f"{R2_DIR}/UKBB_{name}_bootstraps.csv"
    # b38 results are joint (array-based) imputation and live alongside the
    # other multi-SNP results: {dir}_multi_{method}_b38_hum5_chr15_results.csv,
    # where method is hclt/pc for GPC, so hmm/pc_hmm for the chain.
    method = ("hmm" if arm == "ag" else "pc_hmm") + ("_combined" if combined else "")
    return f"{ROOT}/plots/impute/results/multi/{cfg['dir']}_multi_{method}_b38_hum5_chr15_results.csv"


def model_paths(cfg, num_chunks=None, epochs=None):
    """Checkpoint path for each chunk, in genomic order."""
    E = cfg["epochs"] if epochs is None else epochs
    num_chunks = cfg.get("num_chunks", 4) if num_chunks is None else num_chunks
    return [
        f"{cfg['model_dir']}/pc_{cfg['tag']}_{c}_{cfg['split']}_hmm_"
        f"{cfg['n_train']}-128_{E}epochs_ps0.005.jpc"
        for c in range(num_chunks)
    ]
