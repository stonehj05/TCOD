#!/bin/bash
# avg@N eval of every saved checkpoint of one run, one after another, each on all TPU chips
# (see evaluate_checkpoint_avg4.sh). Summary: $RESULT_DIR/avg4_<LABEL_PREFIX>_all.log
# Usage: eval_all_checkpoints.sh RUN_DIR LABEL_PREFIX [STEP ...]   (default: all global_step_*)
set -uo pipefail
RUN=$1; PREFIX=$2; shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || mapfile -t STEPS < <(ls -d "$RUN"/global_step_* | sed 's/.*global_step_//' | sort -rn)
REPO=${REPO:-$HOME/TCOD}
export RESULT_DIR=${RESULT_DIR:-$REPO/checkpoints/eval_results}
mkdir -p "$RESULT_DIR"
SUMMARY=$RESULT_DIR/avg4_${PREFIX}_all.log
echo "[$(date -u +%F' '%T)] evaluating steps ${STEPS[*]} of $RUN" >> "$SUMMARY"
for step in "${STEPS[@]}"; do
    ckpt=$RUN/global_step_$step/actor/huggingface; label=${PREFIX}_step$step
    [ -f "$ckpt/model.safetensors" ] || { echo "[$(date -u +%T)] step $step: no checkpoint, skipped" >> "$SUMMARY"; continue; }
    echo "[$(date -u +%T)] step $step: started" >> "$SUMMARY"
    bash "$HERE/evaluate_checkpoint_avg4.sh" "$ckpt" "$label" > "$RESULT_DIR/avg4_$label.log" 2>&1
    errs=$(grep -oE "errors [0-9]+" "$RESULT_DIR/avg4_$label.log" | awk '{s+=$2} END {print s+0}')
    echo "[$(date -u +%T)] step $step: done, errored games $errs" >> "$SUMMARY"
    grep -E "avg@" "$RESULT_DIR/avg4_$label.log" | sed "s/^/    step $step  /" >> "$SUMMARY"
    [ "$errs" -gt 12 ] && echo "    WARNING: step $step has $errs errored games (counted as failures); check data/avg4_$label.log" >> "$SUMMARY"
done
echo "[$(date -u +%F' '%T)] ALL DONE" >> "$SUMMARY"
