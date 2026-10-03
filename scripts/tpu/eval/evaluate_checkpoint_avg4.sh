#!/bin/bash
# avg@N ALFWorld eval of one checkpoint on every TPU chip of the slice (run on worker 0).
# Protocol: 12_evaluate_checkpoint.py --full-memory, temperature 0.4, max_env_steps 30,
# max_tokens 4096, 40960 context, 8 client workers per server; N_REPS reps x {unseen, seen}.
#
# The work is cut into jobs of one (rep, split, part) each, one vLLM-TPU server per chip:
# with C chips in total, each split is cut into PARTS = max(1, C / (2 * N_REPS)) parts, and the
# jobs run in waves of C. Games are independent, so the parts are merged back into the usual
# per-rep files eval_<LABEL>.rep<R>.<split>.{jsonl,summary.json} (written to $RESULT_DIR,
# default $REPO/checkpoints/eval_results) and aggregated as mean +/- std over reps.
# Verified on 16 chips (4 VMs x 4): ~12 min.
#
# Uses the evaluation client in client/ (override with $PROBE_DIR) and a directory visible at
# the same path on all workers for the part files ($OUT_ROOT, default
# $REPO/checkpoints/eval_runs = the NFS share).
# The model path must also be readable from every worker (a checkpoint under
# $REPO/checkpoints is; a Hugging Face repo id works too).
#
# Usage: evaluate_checkpoint_avg4.sh MODEL_PATH LABEL
set -uo pipefail
MODEL_PATH=$1
LABEL=$2
N_REPS=${N_REPS:-4}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$HOME/TCOD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
PROBE_DIR=${PROBE_DIR:-$HERE/client}
RESULT_DIR=${RESULT_DIR:-$REPO/checkpoints/eval_results}
TASK_DIR=${TASK_DIR:-$HOME/alf-data/tcod_tasks}
OUT_ROOT=${OUT_ROOT:-$REPO/checkpoints/eval_runs}
PY=$VENV/bin/python
SSH="ssh -i ${SSH_KEY:-$HOME/.ssh/google_compute_engine} -o BatchMode=yes"
OUT_DIR=$OUT_ROOT/$LABEL
mkdir -p "$OUT_DIR" "$RESULT_DIR"
[ -f "$PROBE_DIR/12_evaluate_checkpoint.py" ] || { echo "missing $PROBE_DIR/12_evaluate_checkpoint.py" >&2; exit 1; }

# Workers: $WORKER_IPS ("ip0 ip1 ..."), else TPU-VM metadata, else just this VM.
md() { curl -sf -m 5 -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null || true; }
if [ -n "${WORKER_IPS:-}" ]; then read -r -a IPS <<< "$WORKER_IPS"
else mapfile -t IPS < <(md attributes/worker-network-endpoints | tr ',' '\n' | awk -F: 'NF>=3 {print $3}'); fi
[ "${#IPS[@]}" -gt 0 ] || IPS=(localhost)

# One slot per chip: "worker_index chip_index". Also ship the eval client to the other workers.
SLOTS=()
for w in "${!IPS[@]}"; do
    if [ "$w" = 0 ]; then n=$(ls /dev/accel* 2>/dev/null | wc -l)
    else
        rsync -a -e "$SSH" "$HERE/" "${IPS[$w]}:$HERE/" || exit 1
        [ "$PROBE_DIR" = "$HERE/client" ] || rsync -a -e "$SSH" --exclude data/ "$PROBE_DIR/" "${IPS[$w]}:$PROBE_DIR/" || exit 1
        n=$($SSH "${IPS[$w]}" "ls /dev/accel* 2>/dev/null | wc -l")
    fi
    for c in $(seq 0 $((n - 1))); do SLOTS+=("$w $c"); done
done
C=${#SLOTS[@]}
[ "$C" -gt 0 ] || { echo "no TPU chips found" >&2; exit 1; }
PARTS=$(( C / (2 * N_REPS) )); [ "$PARTS" -ge 1 ] || PARTS=1
echo "[$LABEL] $C chips on ${#IPS[@]} worker(s); $N_REPS reps x 2 splits x $PARTS part(s) = $(( 2 * N_REPS * PARTS )) jobs"

n_games() { wc -l < "$TASK_DIR/$([ "$1" = seen ] && echo test || echo test_unseen).jsonl"; }
job=0
for rep in $(seq 1 "$N_REPS"); do
    for split in unseen seen; do
        n=$(n_games $split)
        for part in $(seq 0 $((PARTS - 1))); do
            start=$(( part * n / PARTS )); end=$(( (part + 1) * n / PARTS ))
            read -r w chip <<< "${SLOTS[$(( job % C ))]}"
            out="$OUT_DIR/eval_${LABEL}.rep${rep}.${split}.part${part}"
            cmd="PROBE_DIR='$PROBE_DIR' TASK_DIR='$TASK_DIR' VENV='$VENV' bash '$HERE/run_eval_job.sh' '$MODEL_PATH' '$LABEL' $chip $rep $split $start $end '$out'"
            if [ "$w" = 0 ]; then bash -c "$cmd" > "$out.log" 2>&1 &
            else $SSH "${IPS[$w]}" "$cmd" > "$out.log" 2>&1 & fi
            echo "job $job: worker $w chip $chip rep$rep $split [$start,$end)"
            job=$(( job + 1 ))
            [ $(( job % C )) = 0 ] && wait   # every chip is busy: finish this wave first
        done
    done
done
wait
echo "[$LABEL] all $job jobs done at $(date -u +%T)"

# Merge parts -> per-rep files (same format as the GPU pipeline), then avg@N.
$PY - "$PROBE_DIR" "$OUT_DIR" "$LABEL" "$N_REPS" "$PARTS" "$RESULT_DIR" <<'PYEOF'
import importlib.util, json, os, statistics, sys
here, out_dir, label, n, parts, data = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, here)
spec = importlib.util.spec_from_file_location("ev", os.path.join(here, "12_evaluate_checkpoint.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
for split in ("unseen", "seen"):
    rates = []
    for rep in range(1, n + 1):
        records, ok = [], True
        for part in range(parts):
            f = f"{out_dir}/eval_{label}.rep{rep}.{split}.part{part}.jsonl"
            if not os.path.exists(f):
                print(f"{split} rep{rep}: missing {f}"); ok = False; continue
            records += [json.loads(l) for l in open(f) if l.strip()]
        if not ok:
            continue
        records.sort(key=lambda r: (r["split"], r["game_file"]))
        base = f"{data}/eval_{label}.rep{rep}.{split}"
        with open(base + ".jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)
        s = ev._write_summary(records, base + ".summary.json", f"{label}_rep{rep}")
        rates.append(s["per_split"][split]["success_rate"])
        d = s["per_split"][split]
        print(f"{split} rep{rep}: {d['n_success']}/{d['n_tasks']} = {d['success_rate']:.1%}, errors {d['n_errors']}")
    if rates:
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        print(f"{split}: avg@{len(rates)} = {statistics.mean(rates):.1%} +/- {sd:.1%}  reps={[round(r, 3) for r in rates]}")
PYEOF
