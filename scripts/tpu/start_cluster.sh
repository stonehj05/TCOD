#!/bin/bash
# Bring up the multi-VM runtime for TPU training. Run on worker 0 after every (re)start of
# the slice; everything it creates (NFS mounts, tmpfs dirs, Ray) is lost on reboot.
#   - NFS: worker 0 exports $REPO/checkpoints (run + full checkpoints) and /dev/shm/tcod_sync
#     (per-step weight-sync checkpoints); every other worker mounts both at the same paths.
#   - Ray: worker 0 is head and holds the trainer chips (resource trainer_tpu); the other
#     workers hold engine chips (explorer_tpu). Object store capped so tmpfs has headroom
#     for the sync checkpoints / teacher weights.
# Needs passwordless ssh to the other workers (see README: gcloud pushes the key once).
set -euo pipefail

REPO=${REPO:-$HOME/TCOD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
OBJECT_STORE_BYTES=${OBJECT_STORE_BYTES:-60000000000}
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
SYNC=/dev/shm/tcod_sync
CKPT=$REPO/checkpoints

md() { curl -s -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
# "worker-id:?:ip,..." in worker order; worker 0 is this VM.
mapfile -t IPS < <(md attributes/worker-network-endpoints | tr ',' '\n' | awk -F: '{print $3}')
HEAD=${IPS[0]}; OTHERS=("${IPS[@]:1}")
[ "$(md attributes/agent-worker-number)" = "0" ] || { echo "run this on worker 0" >&2; exit 1; }
echo "head $HEAD, workers ${OTHERS[*]}"

# 1. NFS server on worker 0, exported only to the slice's workers
mkdir -p "$CKPT" "$SYNC" /dev/shm/tmp /dev/shm/ray_tmp
dpkg -s nfs-kernel-server >/dev/null 2>&1 || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q nfs-kernel-server
clients=""; for ip in "${OTHERS[@]}"; do clients+=" $ip(rw,sync,no_subtree_check,no_root_squash)"; done
sync_clients=""; for ip in "${OTHERS[@]}"; do sync_clients+=" $ip(rw,sync,no_subtree_check,no_root_squash,fsid=17)"; done
sudo sed -i "\#^$CKPT #d; \#^$SYNC #d" /etc/exports
echo "$CKPT$clients" | sudo tee -a /etc/exports >/dev/null
echo "$SYNC$sync_clients" | sudo tee -a /etc/exports >/dev/null   # tmpfs needs an explicit fsid
sudo exportfs -ra

# 2. Mount on the other workers
for ip in "${OTHERS[@]}"; do
    $SSH "$ip" "set -e
        dpkg -s nfs-common >/dev/null 2>&1 || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q nfs-common
        mkdir -p $CKPT $SYNC /dev/shm/tmp /dev/shm/ray_tmp
        mountpoint -q $CKPT || sudo mount -t nfs $HEAD:$CKPT $CKPT
        mountpoint -q $SYNC || sudo mount -t nfs $HEAD:$SYNC $SYNC
        echo \"\$(hostname): NFS mounted\""
done

# 3. Ray cluster
RAY=$VENV/bin/ray
$RAY stop --force >/dev/null 2>&1 || true
$RAY start --head --port=6379 --temp-dir=/dev/shm/ray_tmp --include-dashboard=false \
    --object-store-memory="$OBJECT_STORE_BYTES" --resources='{"trainer_tpu": 4}' >/dev/null
for ip in "${OTHERS[@]}"; do
    $SSH "$ip" "$RAY stop --force >/dev/null 2>&1 || true
        setsid $RAY start --address=$HEAD:6379 --temp-dir=/dev/shm/ray_tmp \
            --object-store-memory=$OBJECT_STORE_BYTES --resources='{\"explorer_tpu\": 4}' \
            > /dev/shm/ray_start.log 2>&1 < /dev/null"
done
sleep 8
$RAY status | sed -n '/Resources/,/Demands/p' | grep -E "TPU|_tpu"
