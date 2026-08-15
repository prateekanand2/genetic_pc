#!/bin/bash
# Driver for the HMM ablation (chain-structured GPC) in the imputation benchmarks.
#
# Stages, in order:
#   train     4-chunk HMM per dataset/split                       (GPU)
#   sample    artificial genomes from the population-only models  (GPU)
#   panels    AGs -> Impute5 reference panels (+ merged with real EUR)
#   direct    leave-one-SNP-out conditionals -> dosage matrices    (GPU)
#   loo       leave-one-SNP-out Impute5 sweep over the AG panels   (CPU)
#   csv       assemble the R2 CSVs read by plots/impute/plot.ipynb
#
# Usage:  ./hmm_pipeline.sh <stage> [config ...]
# e.g.    ./hmm_pipeline.sh train 1KG:afr 1KG:noneur
#         ./hmm_pipeline.sh sample 1KG:afr
#
# GPU stages honour CUDA_VISIBLE_DEVICES; run several configs in parallel by
# launching one invocation per device.

set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(cd .. && pwd)

export BCFTOOLS_PLUGINS=${BCFTOOLS_PLUGINS:-/scratch2/prateek/bcftools/plugins}
R2DIR="$ROOT/plots/impute/results/r2"
WORKERS=${WORKERS:-30}

# Models the AG arm needs (population-only). The eur_and_*_train configs are
# direct-imputation only, mirroring pc_realeur_and_afr_train_afr for GPC.
AG_CONFIGS="1KG:8020 1KG:afr 1KG:noneur UKBB:8020 UKBB:afr UKBB:noneur"
ALL_CONFIGS="$AG_CONFIGS 1KG:eur_and_afr_train 1KG:eur_and_noneur_train UKBB:eur_and_afr_train UKBB:eur_and_noneur_train"

STAGE=${1:?stage required: train|sample|panels|direct|loo|csv}
shift || true
CONFIGS=${*:-$ALL_CONFIGS}

# Recombination map + chromosome per dataset.
map_for() {
    case "$1" in
        1KG:*)  echo "/scratch2/prateek/b37_recombination_maps/chr15.b37.gmap.gz" ;;
        # UKBB genotypes here are b37, which is why impute5.sh passed a map.
        UKBB:*) echo "${UKBB_MAP:-/scratch2/prateek/b37_recombination_maps/chr22.b37.gmap.gz}" ;;
    esac
}

# Name used for the dosage directory and the final CSV, matching the existing
# files: 1KG -> hmm_8020_8020.csv, UKBB -> UKBB_hmm_8020_bootstraps.csv
csv_for() {
    local ds=${1%%:*} split=${1##*:} suffix=${2:-}
    case "$ds" in
        1KG)  echo "$R2DIR/hmm_${split}${suffix}_${split}.csv" ;;
        UKBB) echo "$R2DIR/UKBB_hmm_${split}${suffix}_bootstraps.csv" ;;
    esac
}

for cfg in $CONFIGS; do
    ds=${cfg%%:*}; split=${cfg##*:}
    dir=$(python -c "from hmm_config import resolve; print(resolve('$cfg')['dir'])")
    prefix=$(python -c "from hmm_config import resolve; print(resolve('$cfg')['sample_prefix'])")
    test_prefix=$(python -c "from hmm_config import resolve; print(resolve('$cfg')['test_prefix'])")
    chrom=$(python -c "from hmm_config import resolve; print(resolve('$cfg')['chrom'])")

    echo "=============== $STAGE  $cfg ==============="
    case "$STAGE" in

    train)
        # Resume at chunk granularity. A chunk counts as done only if its log
        # reaches the final epoch -- checkpoints are written every 500 epochs,
        # so the .jpc existing is not by itself evidence of a finished run.
        missing=$(python - <<PY
import os
from hmm_config import resolve, model_paths
c = resolve("$cfg")
E = c["epochs"]
todo = []
for i, ckpt in enumerate(model_paths(c)):
    log = os.path.join(c["model_dir"],
                       f"{c['tag']}_{i}_{c['split']}_hmm_{c['n_train']}_128_{E}epochs_ps0.005.log")
    done = os.path.exists(ckpt) and os.path.exists(log) and \
        f"[Epoch {E}/{E}]" in open(log).read()[-400:]
    if not done:
        todo.append(str(i))
print(" ".join(todo))
PY
)
        if [ -z "$missing" ]; then
            echo "all chunks already trained, skipping"
            continue
        fi
        # Guard against two streams training the same config: they would
        # interleave writes to the same .log and .jpc files.
        mkdir -p "$ROOT/logs/locks"
        lock="$ROOT/logs/locks/${cfg//:/_}.lock"
        exec 9>"$lock"
        if ! flock -n 9; then
            echo "another process already holds the lock for $cfg -- skipping"
            exec 9>&-
            continue
        fi
        echo "training chunks: $missing (batch size ${HMM_BATCH:-256})"
        python -u hmm.py "$cfg" --chunks $missing --batch-size "${HMM_BATCH:-256}"
        exec 9>&-
        ;;

    sample)
        case " $AG_CONFIGS " in *" $cfg "*) ;; *) echo "skip (direct-only config)"; continue ;; esac
        python -u generate_hmm.py "$cfg"
        ;;

    panels)
        case " $AG_CONFIGS " in *" $cfg "*) ;; *) echo "skip (direct-only config)"; continue ;; esac
        extra=""
        [ "$split" != "8020" ] && extra="--combined"
        python -u ../aux/scripts/make_hmm_panels.py "$cfg" $extra
        ;;

    direct)
        python -u predict_hmm.py "$cfg"
        ;;

    loo)
        case " $AG_CONFIGS " in *" $cfg "*) ;; *) echo "skip (direct-only config)"; continue ;; esac
        test_vcf="$test_prefix.vcf"; [ -f "$test_vcf" ] || test_vcf="$test_prefix.vcf.gz"
        for variant in "" "_combined"; do
            [ -z "$variant" ] || [ "$split" != "8020" ] || continue
            panel="${prefix}${variant}.vcf.gz"
            [ -f "$panel" ] || { echo "missing panel $panel"; continue; }
            method="$(basename "$prefix" _samples)${variant}"
            python -u ../impute5/impute5_loo_local.py \
                --panel "$panel" --test "$test_vcf" --chr "$chrom" \
                --map "$(map_for "$cfg")" --method "$method" \
                --outdir "$ROOT/impute5/results/bootstrap" --workers "$WORKERS"
        done
        ;;

    csv)
        # Direct arm: dosage matrix -> CSV.
        out=$(python -c "from hmm_config import resolve, r2_csv; print(r2_csv(resolve('$cfg'), 'direct'))")
        python -u assemble_r2.py "$cfg" --out "$out"

        # AG arm: per-SNP Impute5 dosages -> CSV, then attach MAF.
        case " $AG_CONFIGS " in *" $cfg "*) ;; *) continue ;; esac
        for variant in "" "_combined"; do
            [ -z "$variant" ] || [ "$split" != "8020" ] || continue
            method="$(basename "$prefix" _samples)${variant}"
            dosages="$ROOT/impute5/results/bootstrap/${method}_dosages"
            [ -d "$dosages" ] || { echo "no dosages for $method yet"; continue; }
            combined_flag=$([ -n "$variant" ] && echo --combined || true)
            python -u ../impute5/bootstrap_to_r2.py "$cfg" \
                --dosage-dir "$dosages" $combined_flag
        done
        ;;

    *) echo "unknown stage $STAGE"; exit 1 ;;
    esac
done
