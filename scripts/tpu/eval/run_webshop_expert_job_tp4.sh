#!/bin/bash
# One WebShop expert-collection job on ALL 4 TPU chips of this VM (tensor parallel 4): serves the teacher
# (the server setup of run_webshop_eval_job_tp4.sh, unchanged) and runs client/15_collect_webshop_experts.py
# on training sessions [START, END). Called by collect_webshop_experts.sh (possibly over ssh).
#
# Usage: run_webshop_expert_job_tp4.sh MODEL_PATH SERVED_NAME START END OUT_JSONL
set -uo pipefail
MODEL_PATH=$1 SERVED_NAME=$2 START=$3 END=$4 OUT=$5
CHIP=0
# Client workers: 32 games at once for one TP4 server, as the TP4 eval jobs.
WORKERS=${WORKERS:-32}
# Server sizing: lower these for a model that does not fit one chip's memory at 0.85.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85} MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
PROBE_DIR=${PROBE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/client}
TASK_DIR=${TASK_DIR:-$HOME/TCOD/TCOD_examples/webshop/webshop_data}
VENV=${VENV:-$HOME/venv-vllm-tpu}
cd "$PROBE_DIR" || { echo "ERROR: PROBE_DIR $PROBE_DIR not found"; exit 1; }
export TMPDIR=/dev/shm/tmp; mkdir -p "$TMPDIR"
export JAVA_HOME=${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}  # pyserini's JVM
PY=$VENV/bin/python VLLM=$VENV/bin/vllm
PORT=$(( 8081 + CHIP )) TPU_PORT=$(( 8477 + CHIP ))
LOG=/dev/shm/vllm_webshop_expertjob_tp4.log

start_server() {  # $1: extra LIBTPU_INIT_ARGS
    TPU_VISIBLE_CHIPS=0,1,2,3 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_PORT=$TPU_PORT TPU_PROCESS_ADDRESSES=localhost:$TPU_PORT LIBTPU_INIT_ARGS="$1" \
        nohup $VLLM serve "$MODEL_PATH" --port "$PORT" --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size 4 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" \
        > "$LOG" 2>&1 &
    PID=$!
    for i in $(seq 1 180); do
        id=$(curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | $PY -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
        [ "$id" = "$SERVED_NAME" ] && return 0
        kill -0 $PID 2>/dev/null || return 1   # server process died
        sleep 10
    done
    return 1
}

free_chip() {
    kill $PID 2>/dev/null   # SIGTERM: vLLM also stops its EngineCore child
    for i in $(seq 1 30); do [ -z "$(lsof -t /dev/accel* 2>/dev/null)" ] && return; sleep 2; done
    for p in $(lsof -t /dev/accel* 2>/dev/null); do kill -9 $p; done
}

echo "[$(hostname) TP4] expert collection, training sessions [$START,$END) -> $OUT"
if ! start_server "--runtime_metric_service_port=$(( 8431 + CHIP ))"; then
    echo "server failed with per-chip metrics port; retrying without it (see $LOG)"
    free_chip; start_server "" || { echo "ERROR: server failed; see $LOG"; free_chip; exit 1; }
fi
$PY 15_collect_webshop_experts.py --base-url "http://localhost:${PORT}/v1" --model "$SERVED_NAME" \
    --workers "$WORKERS" --tasks-jsonl "$TASK_DIR/train.jsonl" --task-start "$START" --task-end "$END" \
    --output "$OUT"
rc=$?
free_chip
echo "[$(hostname) TP4] expert collection [$START,$END) finished (exit $rc)"
exit $rc
