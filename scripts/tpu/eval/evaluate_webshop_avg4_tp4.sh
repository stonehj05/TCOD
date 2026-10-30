#!/bin/bash
# avg@N WebShop eval with ONE tensor-parallel-4 server per worker (run on worker 0):
# evaluate_webshop_avg4.sh for a model that does not fit one chip (the 30B-A3B teacher), with
# run_webshop_eval_job_tp4.sh as the job. Same protocol, split, merge and output files.
# Run it only on workers that hold MODEL_PATH (WORKER_IPS), e.g. the teacher VMs' tmpfs copy.
# Protocol: TCOD's test sessions (get_webshop_data.py: 4096-4195, 100 sessions; = DASH-OPD's),
# client/14_evaluate_webshop.py (the training rollout: DASH-OPD's prompts, full conversation),
# temperature 0.4, 15 game steps, max_tokens 4096, thinking off, 40960 context, 32 games at once per server.
# Reports success rate (score 1) and mean score, each mean +/- std over the
# N_REPS reps; per-rep files eval_<LABEL>.rep<R>.test.{jsonl,summary.json} in $RESULT_DIR.
#
# With C workers the test split is cut into PARTS = max(1, C / N_REPS) parts (2 workers: 4 reps x
# 1 part = two waves of 2 jobs). Part files go to $OUT_ROOT (default $REPO/checkpoints/eval_runs,
# the NFS share); the model path must be readable from every worker.
#
# Usage: WORKER_IPS="ip1 ip2" evaluate_webshop_avg4_tp4.sh MODEL_PATH LABEL
set -uo pipefail
MODEL_PATH=$1
LABEL=$2
N_REPS=${N_REPS:-4}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$HOME/TCOD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
PROBE_DIR=${PROBE_DIR:-$HERE/client}
RESULT_DIR=${RESULT_DIR:-$REPO/checkpoints/eval_results}
TASK_DIR=${TASK_DIR:-$REPO/TCOD_examples/webshop/webshop_data}
OUT_ROOT=${OUT_ROOT:-$REPO/checkpoints/eval_runs}
PY=$VENV/bin/python
SSH="ssh -i ${SSH_KEY:-$HOME/.ssh/google_compute_engine} -o BatchMode=yes"
OUT_DIR=$OUT_ROOT/$LABEL
mkdir -p "$OUT_DIR" "$RESULT_DIR"
[ -f "$PROBE_DIR/14_evaluate_webshop.py" ] || { echo "missing $PROBE_DIR/14_evaluate_webshop.py" >&2; exit 1; }

# Workers: $WORKER_IPS ("ip0 ip1 ..."), else TPU-VM metadata, else just this VM.
md() { curl -sf -m 5 -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null || true; }
if [ -n "${WORKER_IPS:-}" ]; then read -r -a IPS <<< "$WORKER_IPS"
else mapfile -t IPS < <(md attributes/worker-network-endpoints | tr ',' '\n' | awk -F: 'NF>=3 {print $3}'); fi
[ "${#IPS[@]}" -gt 0 ] || IPS=(localhost)
is_local() { [ "$1" = localhost ] || hostname -I | tr ' ' '\n' | grep -qx "$1"; }  # this VM: run here, not over ssh

# One slot per worker that has 4 chips: "worker_index 0". Also ship the eval client to the other workers.
SLOTS=()
for w in "${!IPS[@]}"; do
    if is_local "${IPS[$w]}"; then n=$(ls /dev/accel* 2>/dev/null | wc -l)
    else
        rsync -a -e "$SSH" "$HERE/" "${IPS[$w]}:$HERE/" || exit 1
        [ "$PROBE_DIR" = "$HERE/client" ] || rsync -a -e "$SSH" --exclude data/ "$PROBE_DIR/" "${IPS[$w]}:$PROBE_DIR/" || exit 1
        n=$($SSH "${IPS[$w]}" "ls /dev/accel* 2>/dev/null | wc -l")
    fi
    [ "$n" = 4 ] && SLOTS+=("$w 0") || echo "worker $w (${IPS[$w]}): $n chips, skipped" >&2
done
C=${#SLOTS[@]}
[ "$C" -gt 0 ] || { echo "no worker with 4 TPU chips found" >&2; exit 1; }
PARTS=$(( C / N_REPS )); [ "$PARTS" -ge 1 ] || PARTS=1
echo "[$LABEL] $C TP4 servers on ${#IPS[@]} worker(s); $N_REPS reps x $PARTS part(s) = $(( N_REPS * PARTS )) jobs"

n=$(wc -l < "$TASK_DIR/test.jsonl")
job=0
for rep in $(seq 1 "$N_REPS"); do
    for part in $(seq 0 $((PARTS - 1))); do
        start=$(( part * n / PARTS )); end=$(( (part + 1) * n / PARTS ))
        read -r w chip <<< "${SLOTS[$(( job % C ))]}"
        out="$OUT_DIR/eval_${LABEL}.rep${rep}.test.part${part}"
        cmd="GPU_MEM_UTIL='${GPU_MEM_UTIL:-0.85}' PROBE_DIR='$PROBE_DIR' TASK_DIR='$TASK_DIR' VENV='$VENV' bash '$HERE/run_webshop_eval_job_tp4.sh' '$MODEL_PATH' '$LABEL' $chip $rep $start $end '$out'"
        if is_local "${IPS[$w]}"; then bash -c "$cmd" > "$out.log" 2>&1 &
        else $SSH "${IPS[$w]}" "$cmd" > "$out.log" 2>&1 & fi
        echo "job $job: worker $w TP4 rep$rep test [$start,$end)"
        job=$(( job + 1 ))
        [ $(( job % C )) = 0 ] && wait   # every server is busy: finish this wave first
    done
done
wait
echo "[$LABEL] all $job jobs done at $(date -u +%T)"

# Merge parts -> per-rep files, then avg@N of success rate and mean score.
$PY - "$PROBE_DIR" "$OUT_DIR" "$LABEL" "$N_REPS" "$PARTS" "$RESULT_DIR" <<'PYEOF'
import importlib.util, json, os, statistics, sys
here, out_dir, label, n, parts, data = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("ev", os.path.join(here, "14_evaluate_webshop.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
rates, scores = [], []
for rep in range(1, n + 1):
    records, ok = [], True
    for part in range(parts):
        f = f"{out_dir}/eval_{label}.rep{rep}.test.part{part}.jsonl"
        if not os.path.exists(f):
            print(f"test rep{rep}: missing {f}"); ok = False; continue
        records += [json.loads(l) for l in open(f) if l.strip()]
    if not ok:
        continue
    records.sort(key=lambda r: r["index"])
    base = f"{data}/eval_{label}.rep{rep}.test"
    with open(base + ".jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    d = ev._write_summary(records, base + ".summary.json", f"{label}_rep{rep}")["per_split"]["test"]
    rates.append(d["success_rate"]); scores.append(d["mean_score"])
    print(f"test rep{rep}: success {d['n_success']}/{d['n_tasks']} = {d['success_rate']:.1%}, "
          f"mean score {d['mean_score']:.3f}, errors {d['n_errors']}, ended on a full context {d['n_context_overflow']}, mean turns {d['mean_env_rounds']:.2f}")
if rates:
    sd = lambda x: statistics.stdev(x) if len(x) > 1 else 0.0
    print(f"test: avg@{len(rates)} success rate = {statistics.mean(rates):.1%} +/- {sd(rates):.1%}  reps={[round(r, 3) for r in rates]}")
    print(f"test: avg@{len(scores)} mean score = {statistics.mean(scores):.3f} +/- {sd(scores):.3f}  reps={[round(s, 3) for s in scores]}")
PYEOF
