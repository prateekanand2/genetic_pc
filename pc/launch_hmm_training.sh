#!/bin/bash
# Launch the HMM training runs as parallel streams on one GPU.
#
# A single training process only reaches ~11% GPU utilisation: each epoch is
# dominated by the EM parameter update walking all 2,501 circuit layers, not by
# the data, so epoch time barely depends on the training-set size and extra
# batches are nearly free. That makes the GPU launch-bound, and running several
# configs side by side costs far less than running them back to back.
#
# Streams are balanced to ~33 h each. Each gets its own Triton cache so that
# concurrent pyjuice compiles cannot race, and starts are staggered so six
# compiles do not land at once.
#
# Safe to re-run: hmm_pipeline.sh skips any chunk whose log already reached the
# final epoch, so an interrupted stream resumes at chunk granularity.
#
# Usage:  ./launch_hmm_training.sh [gpu_id]

set -euo pipefail
cd "$(dirname "$0")"

GPU=${1:-3}
LOGS=../logs
CACHE=/scratch2/prateek/tmp/triton_hmm
mkdir -p "$LOGS" "$CACHE"

# Four streams, not more. Measured under load: one job alone runs 5.6 s/epoch,
# but with five concurrent the same jobs ran 6-20 s/epoch -- aggregate
# throughput only about doubles, and six jobs drove GPU memory to 23.7/24.5 GB,
# which would have OOM'd once the last streams began training. Four jobs sit at
# roughly 14 GB and capture most of the available speedup.
#
# Each stream does its 1KG config first (needed before UKBB) then its UKBB one,
# so the two AG-route models -- 1KG:afr and 1KG:noneur -- land earliest and
# unblock the CPU-bound Impute5 sweeps.
STREAMS=(
  "afr:1KG:afr UKBB:afr"
  "noneur:1KG:noneur UKBB:noneur"
  "eur_and_afr:1KG:eur_and_afr_train UKBB:eur_and_afr_train"
  "eur_and_noneur:1KG:eur_and_noneur_train UKBB:eur_and_noneur_train"
)

delay=0
for entry in "${STREAMS[@]}"; do
    name=${entry%%:*}
    configs=${entry#*:}
    log="$LOGS/hmm_train_${name}.log"
    # setsid puts each stream in its own session so it survives the launching
    # shell exiting -- a plain "( ... ) &" gets SIGHUP'd and dies.
    setsid nohup bash -c "
        sleep $delay
        CUDA_VISIBLE_DEVICES=$GPU \
        TRITON_CACHE_DIR=$CACHE/$name \
        ./hmm_pipeline.sh train $configs
    " > "$log" 2>&1 &
    disown || true
    echo "stream $name (pid $!, +${delay}s) -> $log   [$configs]"
    delay=$((delay + 25))
done

echo
echo "6 streams launched on GPU $GPU. Watch with:"
echo "  tail -f $LOGS/hmm_train_*.log"
echo "  ./hmm_progress.sh"
