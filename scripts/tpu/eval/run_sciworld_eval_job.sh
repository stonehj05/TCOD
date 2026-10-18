#!/bin/bash
# One ScienceWorld eval job on ONE TPU chip of this VM: start a vLLM-TPU server on the chip, run
# client/13_evaluate_sciworld.py on test games [START, END), then stop the server and free the
# chip. The ScienceWorld counterpart of run_eval_job.sh (same server setup); called by
# evaluate_sciworld_avg4.sh (possibly over ssh). Needs Java and the scienceworld package (see
# TCOD_examples/scienceworld/opd_gated_student_teacher_lookahead_t07_s03_deferred_tpu.yaml).
#
# Usage: run_sciworld_eval_job.sh MODEL_PATH SERVED_NAME CHIP REP START END OUT_PREFIX
set -uo pipefail
MODEL_PATH=$1 SERVED_NAME=$2 CHIP=$3 REP=$4 START=$5 END=$6 OUT=$7
# Eval protocol (TCOD paper): do not change these when comparing against existing results.
WORKERS=${WORKERS:-8} MAX_ENV_STEPS=30 TEMPERATURE=0.4 MAX_TOKENS=4096
# Server sizing: lower these for a model that does not fit one chip's memory at 0.85.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85} MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
PROBE_DIR=${PROBE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/client}
TASK_DIR=${TASK_DIR:-$HOME/sci-data/tcod_tasks}
VENV=${VENV:-$HOME/venv-vllm-tpu}
cd "$PROBE_DIR" || { echo "ERROR: PROBE_DIR $PROBE_DIR not found"; exit 1; }
export TMPDIR=/dev/shm/tmp; mkdir -p "$TMPDIR"
PY=$VENV/bin/python VLLM=$VENV/bin/vllm
PORT=$(( 8081 + CHIP )) TPU_PORT=$(( 8477 + CHIP ))
LOG=/dev/shm/vllm_sciworld_evaljob_chip${CHIP}.log

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

echo "[$(hostname) chip $CHIP] rep$REP test games [$START,$END) -> $OUT"
# Metrics on 8431 + chip (for scripts/tpu/tpu_usage.sh); fall back without it if the server won't start.
if ! start_server "--runtime_metric_service_port=$(( 8431 + CHIP ))"; then
    echo "server failed with per-chip metrics port; retrying without it (see $LOG)"
    free_chip; start_server "" || { echo "ERROR: server failed; see $LOG"; free_chip; exit 1; }
fi
$PY 13_evaluate_sciworld.py --base-url "http://localhost:${PORT}/v1" --model "$SERVED_NAME" \
    --checkpoint-label "${SERVED_NAME}_rep${REP}" ${EVAL_TEMPLATE:+--template "$EVAL_TEMPLATE"} --workers "$WORKERS" \
    --tasks-jsonl "$TASK_DIR/test.jsonl" --task-start "$START" --task-end "$END" \
    --max-env-steps "$MAX_ENV_STEPS" --temperature "$TEMPERATURE" --max-tokens "$MAX_TOKENS" \
    --output "$OUT.jsonl" --summary "$OUT.summary.json"
rc=$?
free_chip
echo "[$(hostname) chip $CHIP] rep$REP test [$START,$END) finished (exit $rc)"
exit $rc
