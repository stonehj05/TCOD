#!/bin/bash
# One ALFWorld eval job on ONE TPU chip of this VM: start a vLLM-TPU server on the chip, run
# 12_evaluate_checkpoint.py --full-memory on games [START, END) of SPLIT, then stop the server
# and free the chip. Called by evaluate_checkpoint_avg4.sh (possibly over ssh).
#
# The evaluation client is client/ next to this script (12_evaluate_checkpoint.py and its
# helpers); set $PROBE_DIR to use another copy.
# Task files: $TASK_DIR/{test,test_unseen,train_hard}.jsonl (default ~/alf-data/tcod_tasks,
# written by scripts/tpu/setup_node.sh).
#
# Usage: run_eval_job.sh MODEL_PATH SERVED_NAME CHIP REP SPLIT START END OUT_PREFIX
set -uo pipefail
MODEL_PATH=$1 SERVED_NAME=$2 CHIP=$3 REP=$4 SPLIT=$5 START=$6 END=$7 OUT=$8
# Eval protocol (TCOD paper): do not change these when comparing against existing results.
WORKERS=${WORKERS:-8} MAX_ENV_STEPS=30 TEMPERATURE=0.4
# Server sizing: lower these for a model that does not fit one chip's memory at 0.85.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85} MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
PROBE_DIR=${PROBE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/client}
TASK_DIR=${TASK_DIR:-$HOME/alf-data/tcod_tasks}
VENV=${VENV:-$HOME/venv-vllm-tpu}
cd "$PROBE_DIR" || { echo "ERROR: PROBE_DIR $PROBE_DIR not found"; exit 1; }
# ALFWorld copies its planner library into a temp dir for every game. Keep that off the root
# disk (a full disk turns every game into an "error" = failure) by using tmpfs.
export TMPDIR=/dev/shm/tmp; mkdir -p "$TMPDIR"
PY=$VENV/bin/python VLLM=$VENV/bin/vllm
PORT=$(( 8081 + CHIP )) TPU_PORT=$(( 8477 + CHIP ))
LOG=/dev/shm/vllm_evaljob_chip${CHIP}.log

start_server() {  # $1: extra LIBTPU_INIT_ARGS
    TPU_VISIBLE_CHIPS=$CHIP TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_PORT=$TPU_PORT TPU_PROCESS_ADDRESSES=localhost:$TPU_PORT LIBTPU_INIT_ARGS="$1" \
        nohup $VLLM serve "$MODEL_PATH" --port "$PORT" --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" \
        > "$LOG" 2>&1 &
    PID=$!
    for i in $(seq 1 90); do
        id=$(curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | $PY -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        [ "$id" = "$SERVED_NAME" ] && return 0
        kill -0 $PID 2>/dev/null || return 1   # server process died
        sleep 10
    done
    return 1
}

free_chip() {
    kill $PID 2>/dev/null   # SIGTERM: vLLM also stops its EngineCore child
    for i in $(seq 1 30); do [ -z "$(lsof -t /dev/accel$CHIP 2>/dev/null)" ] && return; sleep 2; done
    for p in $(lsof -t /dev/accel$CHIP 2>/dev/null); do kill -9 $p; done
}

echo "[$(hostname) chip $CHIP] rep$REP $SPLIT games [$START,$END) -> $OUT"
# Metrics on 8431 + chip (for scripts/tpu/tpu_usage.sh); fall back without it if the server won't start.
if ! start_server "--runtime_metric_service_port=$(( 8431 + CHIP ))"; then
    echo "server failed with per-chip metrics port; retrying without it (see $LOG)"
    free_chip; start_server "" || { echo "ERROR: server failed; see $LOG"; free_chip; exit 1; }
fi
$PY 12_evaluate_checkpoint.py --base-url "http://localhost:${PORT}/v1" --model "$SERVED_NAME" \
    --checkpoint-label "${SERVED_NAME}_rep${REP}" --full-memory --split "$SPLIT" --workers "$WORKERS" \
    --task-start "$START" --task-end "$END" \
    --max-env-steps "$MAX_ENV_STEPS" --temperature "$TEMPERATURE" \
    --seen-jsonl "$TASK_DIR/test.jsonl" \
    --unseen-jsonl "$TASK_DIR/test_unseen.jsonl" \
    --hard-jsonl "$TASK_DIR/train_hard.jsonl" \
    --output "$OUT.jsonl" --summary "$OUT.summary.json"
rc=$?
free_chip
echo "[$(hostname) chip $CHIP] rep$REP $SPLIT [$START,$END) finished (exit $rc)"
exit $rc
