#!/bin/bash
# Teacher expert actions for TCOD-B2F on WebShop (run on worker 0): one greedy rollout of the teacher per
# training session (TCOD's 4096; client/15_collect_webshop_experts.py), one TP4 teacher server per worker in
# WORKER_IPS ("ip:model_path ..."), the sessions split evenly. Parts go to $OUT_DIR (the NFS share) and can
# be resumed: a rerun skips the sessions a part already has. Then merged, in TCOD's row order, into
# $EXPERT_DIR/train.jsonl ({"task_id", "target", "actions"}; actions = [] where the teacher did not succeed),
# with TCOD's test.jsonl / dataset_dict.json copied next to it.
#
# Usage: WORKER_IPS="10.130.15.215:/dev/shm/models/Qwen3-30B-A3B ..." collect_webshop_experts.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$HOME/TCOD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
TASK_DIR=${TASK_DIR:-$REPO/TCOD_examples/webshop/webshop_data}
EXPERT_DIR=${EXPERT_DIR:-$REPO/TCOD_examples/webshop/webshop_data_expert}
OUT_DIR=${OUT_DIR:-$REPO/checkpoints/eval_runs/webshop_experts}
SSH="ssh -i ${SSH_KEY:-$HOME/.ssh/google_compute_engine} -o BatchMode=yes"
read -r -a SLOTS <<< "${WORKER_IPS:?set WORKER_IPS=\"ip:model_path ...\"}"
C=${#SLOTS[@]}
mkdir -p "$OUT_DIR"
is_local() { [ "$1" = localhost ] || hostname -I | tr ' ' '\n' | grep -qx "$1"; }
n=$(wc -l < "$TASK_DIR/train.jsonl")
echo "[experts] $n training sessions on $C TP4 teacher server(s)"

run_parts() {
    for w in $(seq 0 $((C - 1))); do
        ip=${SLOTS[$w]%%:*} model=${SLOTS[$w]#*:}
        start=$(( w * n / C )); end=$(( (w + 1) * n / C ))
        out="$OUT_DIR/part$w.jsonl"
        have=$( [ -f "$out" ] && grep -c . "$out" || echo 0 )
        [ "$have" -ge $(( end - start )) ] && { echo "part $w complete ($have)"; continue; }
        cmd="TASK_DIR='$TASK_DIR' VENV='$VENV' bash '$HERE/run_webshop_expert_job_tp4.sh' '$model' teacher $start $end '$out'"
        if is_local "$ip"; then bash -c "$cmd" >> "$out.log" 2>&1 &
        else rsync -a -e "$SSH" "$HERE/" "$ip:$HERE/" && $SSH "$ip" "$cmd" >> "$out.log" 2>&1 & fi
        echo "part $w: $ip sessions [$start,$end) ($have already)"
    done
    wait
}
run_parts
run_parts   # a second pass resumes any part that was cut off

$VENV/bin/python - "$TASK_DIR" "$OUT_DIR" "$EXPERT_DIR" "$C" <<'PYEOF'
import json, os, shutil, sys
task_dir, out_dir, expert_dir, c = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(f"{task_dir}/train.jsonl") if l.strip()]
got = {}
for w in range(c):
    for l in open(f"{out_dir}/part{w}.jsonl"):
        if l.strip():
            r = json.loads(l); got[r["index"]] = r
missing = [i for i in range(len(rows)) if i not in got]
if missing:
    sys.exit(f"MERGE FAILED: {len(missing)} sessions missing (first {missing[:5]})")
os.makedirs(expert_dir, exist_ok=True)
with open(f"{expert_dir}/train.jsonl", "w") as f:
    for i, row in enumerate(rows):
        f.write(json.dumps({**row, "actions": got[i]["actions"]}) + "\n")
for name in ("test.jsonl", "dataset_dict.json"):
    shutil.copy(f"{task_dir}/{name}", f"{expert_dir}/{name}")
recs = list(got.values())
kept = [r for r in recs if r["actions"]]
print(f"EXPERTS DONE: {len(kept)}/{len(recs)} sessions with expert actions "
      f"({sum(r['score'] >= 1 for r in recs)} teacher successes, {sum(r['score'] >= 1 and not r['replay_ok'] for r in recs)} failed replay, "
      f"{sum(1 for r in recs if r['error'])} errors); mean teacher score {sum(r['score'] for r in recs) / len(recs):.3f}; "
      f"mean expert length {sum(len(r['actions']) for r in kept) / max(1, len(kept)):.2f} -> {expert_dir}/train.jsonl")
PYEOF
