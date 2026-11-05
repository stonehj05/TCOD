#!/bin/bash
# Bring up a Ray cluster on TWO workers of the slice only, for one run, while another run uses the other workers (e.g.
# two 8-chip runs on the v4-32, each with a cluster of its own):
#   PAIR_IPS="<trainer host> <inference host>" bash start_cluster_pair.sh
# Run it from any worker with passwordless ssh to both hosts (worker 0: workers 1-3 have no ssh key); either host may
# be this one. The first host gets the Ray label trainer_tpu and is the head (start the run there), the second
# explorer_tpu. There is no teacher_tpu label, so students and teachers share the second host (one chip each).
#
# Ray is stopped and restarted on these two hosts only. Nothing else is touched: start_cluster.sh must have run once
# since the last reboot (worker 0's NFS share of checkpoints/ and /dev/shm/tcod_sync on every worker, and the host
# firewall for all workers of the slice), and this script only checks that both hosts see the shared directories.
# Model weights are not downloaded here (point the config at a local copy on both hosts).
set -euo pipefail

REPO=${REPO:-$HOME/TCOD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
OBJECT_STORE_BYTES=${OBJECT_STORE_BYTES:-20000000000}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/google_compute_engine}
SSH="ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
SYNC=/dev/shm/tcod_sync
CKPT=$REPO/checkpoints
RAY=$VENV/bin/ray

read -r -a IPS <<< "${PAIR_IPS:?set PAIR_IPS=\"<trainer host> <inference host>\"}"
[ "${#IPS[@]}" = 2 ] || { echo "PAIR_IPS needs exactly 2 addresses" >&2; exit 1; }
HEAD=${IPS[0]}; OTHER=${IPS[1]}
# on HOST CMD: run CMD on HOST (locally if HOST is this VM)
on() { if hostname -I | tr ' ' '\n' | grep -qx "$1"; then bash -c "$2"; else $SSH "$1" "$2"; fi; }
echo "pair: $HEAD trainer, $OTHER explorer"

# 1. Both hosts must see the same checkpoints/ and /dev/shm/tcod_sync (NFS from worker 0)
for d in "$CKPT" "$SYNC"; do
    probe="$d/.pair_probe_$$"
    on "$HEAD" "touch '$probe'"
    on "$OTHER" "[ -f '$probe' ]" || { on "$HEAD" "rm -f '$probe'"; echo "$OTHER does not see $d of $HEAD (run start_cluster.sh first)" >&2; exit 1; }
    on "$HEAD" "rm -f '$probe'"
done
echo "shared dirs: OK"

# 2. Node-local tmpfs folders
for ip in "$HEAD" "$OTHER"; do on "$ip" 'mkdir -p /dev/shm/tmp /dev/shm/ray_tmp /dev/shm/tcod_buffers'; done

# 3. Ray on the two hosts (label = <role>_tpu, sized to the host's chip count)
n0=$(on "$HEAD" 'ls /dev/accel* 2>/dev/null | wc -l')
n1=$(on "$OTHER" 'ls /dev/accel* 2>/dev/null | wc -l')
for ip in "$HEAD" "$OTHER"; do on "$ip" "$RAY stop --force >/dev/null 2>&1 || true"; done
on "$HEAD" "setsid $RAY start --head --port=6379 --temp-dir=/dev/shm/ray_tmp --include-dashboard=false \
    --object-store-memory=$OBJECT_STORE_BYTES --resources='{\"trainer_tpu\": $n0}' > /dev/shm/ray_start.log 2>&1 < /dev/null"
echo "$HEAD: trainer ($n0 chips)"
on "$OTHER" "setsid $RAY start --address=$HEAD:6379 --temp-dir=/dev/shm/ray_tmp \
    --object-store-memory=$OBJECT_STORE_BYTES --resources='{\"explorer_tpu\": $n1}' > /dev/shm/ray_start.log 2>&1 < /dev/null"
echo "$OTHER: explorer ($n1 chips)"
sleep 8
on "$HEAD" "$RAY status | sed -n '/Resources/,/Demands/p' | grep -E 'TPU|_tpu'"
nodes=$(on "$HEAD" "$VENV/bin/python -c \"import ray; ray.init(address='auto', logging_level='ERROR'); print('ALIVE_NODES=%d' % sum(n['Alive'] for n in ray.nodes()))\" 2>/dev/null" | grep -o 'ALIVE_NODES=[0-9]*' | tail -1)
[ "$nodes" = ALIVE_NODES=2 ] || { echo "expected 2 alive Ray nodes, got '$nodes'" >&2; exit 1; }
echo "ray: 2 nodes"
