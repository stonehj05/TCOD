#!/bin/bash
# avg@N eval of every saved checkpoint of one run, one after another, each on all TPU chips
# (see evaluate_checkpoint_avg4.sh). Summary: $PROBE_DIR/data/avg4_<LABEL_PREFIX>_all.log
# Usage: eval_all_checkpoints.sh RUN_DIR LABEL_PREFIX [STEP ...]   (default: all global_step_*)
set -uo pipefail
RUN=$1; PREFIX=$2; shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || mapfile -t STEPS < <(ls -d "$RUN"/global_step_* | sed 's/.*global_step_//' | sort -rn)
PROBE_DIR=${PROBE_DIR:-$HOME/alfworld_ts_probe}
mkdir -p "$PROBE_DIR/data"
SUMMARY=$PROBE_DIR/data/avg4_${PREFIX}_all.log
echo "[$(date -u +%F' '%T)] evaluating steps ${STEPS[*]} of $RUN" >> "$SUMMARY"
for step in "${STEPS[@]}"; do
    ckpt=$RUN/global_step_$step/actor/huggingface; label=${PREFIX}_step$step
    [ -f "$ckpt/model.safetensors" ] || { echo "[$(date -u +%T)] step $step: no checkpoint, skipped" >> "$SUMMARY"; continue; }
    echo "[$(date -u +%T)] step $step: started" >> "$SUMMARY"
    bash "$HERE/evaluate_checkpoint_avg4.sh" "$ckpt" "$label" > "$PROBE_DIR/data/avg4_$label.log" 2>&1
    errs=$(grep -oE "errors [0-9]+" "$PROBE_DIR/data/avg4_$label.log" | awk '{s+=$2} END {print s+0}')
    echo "[$(date -u +%T)] step $step: done, errored games $errs" >> "$SUMMARY"
    grep -E "avg@" "$PROBE_DIR/data/avg4_$label.log" | sed "s/^/    step $step  /" >> "$SUMMARY"
    [ "$errs" -gt 12 ] && echo "    WARNING: step $step has $errs errored games (counted as failures); check data/avg4_$label.log" >> "$SUMMARY"
done
echo "[$(date -u +%F' '%T)] ALL DONE" >> "$SUMMARY"
