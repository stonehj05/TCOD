#!/bin/bash
# One WebShop eval job on ALL 4 TPU chips of this VM (tensor parallel 4): run_webshop_eval_job.sh
# for a model that does not fit one chip (the 30B-A3B teacher). Same protocol and client; only the
# server spans the host (the TPU env of trinity/common/models/tpu_env.py for 4 chips on v4). Called by
# evaluate_webshop_avg4_tp4.sh (possibly over ssh). CHIP must be 0 (the whole host). Needs Java and WebShop's data, index and packages (see
# TCOD_examples/webshop/opd_gated_student_teacher_lookahead_t07_s03_dashtpl_deferred_tpu.yaml).
#
# Usage: run_webshop_eval_job_tp4.sh MODEL_PATH SERVED_NAME CHIP REP START END OUT_PREFIX
set -uo pipefail
MODEL_PATH=$1 SERVED_NAME=$2 CHIP=$3 REP=$4 START=$5 END=$6 OUT=$7
# Client workers: 32 for one TP4 server (8 per chip, as the 1-chip jobs).
# Eval protocol (DASH-OPD's WebShop benchmark): do not change these when comparing against existing results.
WORKERS=${WORKERS:-32} MAX_ENV_STEPS=15 TEMPERATURE=0.4 MAX_TOKENS=4096
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
LOG=/dev/shm/vllm_webshop_evaljob_tp4.log

# Gemma 4 checkpoints carry vision / audio towers: serve text-only (empty for every other model)
MM_ARGS=()
$PY -c "import json,sys; sys.exit(0 if json.load(open('$MODEL_PATH/config.json'))['architectures'][0].startswith('Gemma4') else 1)" 2>/dev/null \
    && MM_ARGS=(--limit-mm-per-prompt '{"image":0,"audio":0}')
# Gemma 4 MoE (26B-A4B): vLLM model path + gmm_v2 tile patch, as for training (tpu_env.moe_env_vars; empty
# otherwise). Passed to the server only: with the patch dir on PYTHONPATH every python start imports
# tpu_inference (slow, and it logs to stdout), which would break this script's own small python calls.
MOE_ENV=($(cd "$PROBE_DIR/../../../.." && $PY -c "from trinity.common.models.tpu_env import moe_env_vars
print(' '.join(f'{k}={v}' for k, v in moe_env_vars('$MODEL_PATH').items()))" 2>/dev/null))

start_server() {  # $1: extra LIBTPU_INIT_ARGS
    TPU_VISIBLE_CHIPS=0,1,2,3 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_PROCESS_PORT=$TPU_PORT TPU_PROCESS_ADDRESSES=localhost:$TPU_PORT LIBTPU_INIT_ARGS="$1" \
        nohup env "${MOE_ENV[@]}" $VLLM serve "$MODEL_PATH" --port "$PORT" --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size 4 --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" "${MM_ARGS[@]}" \
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

echo "[$(hostname) TP4] rep$REP test sessions [$START,$END) -> $OUT"
# Metrics on 8431 + chip (for scripts/tpu/tpu_usage.sh); fall back without it if the server won't start.
if ! start_server "--runtime_metric_service_port=$(( 8431 + CHIP ))"; then
    echo "server failed with per-chip metrics port; retrying without it (see $LOG)"
    free_chip; start_server "" || { echo "ERROR: server failed; see $LOG"; free_chip; exit 1; }
fi
$PY 14_evaluate_webshop.py --base-url "http://localhost:${PORT}/v1" --model "$SERVED_NAME" \
    --checkpoint-label "${SERVED_NAME}_rep${REP}" --workers "$WORKERS" \
    --tasks-jsonl "$TASK_DIR/test.jsonl" --task-start "$START" --task-end "$END" \
    --max-env-steps "$MAX_ENV_STEPS" --temperature "$TEMPERATURE" --max-tokens "$MAX_TOKENS" \
    --output "$OUT.jsonl" --summary "$OUT.summary.json"
rc=$?
free_chip
echo "[$(hostname) TP4] rep$REP test [$START,$END) finished (exit $rc)"
exit $rc
