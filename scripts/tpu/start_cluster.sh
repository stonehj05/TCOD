#!/bin/bash
# Bring up the runtime for TPU training. Run on worker 0 after every (re)start of the TPU
# slice; everything it creates (NFS mounts, tmpfs dirs, Ray) is lost on reboot.
#
#   - Workers: taken from the TPU-VM metadata (multi-host slices), or from $WORKER_IPS
#     ("ip0 ip1 ...", worker 0 = this VM first), or just this VM if neither is available.
#   - NFS (only with more than one worker): worker 0 exports $REPO/checkpoints (run + full
#     checkpoints) and /dev/shm/tcod_sync (per-step weight-sync checkpoints); every other
#     worker mounts both at the same paths.
#   - Ray: worker 0 is the head. Each worker gets a Ray label for its role, sized to the
#     number of TPU chips found on it (ls /dev/accel*):
#         trainer  -> trainer_tpu    (the Tunix trainer; must be worker 0)
#         explorer -> explorer_tpu   (rollout / student engines)
#         teacher  -> teacher_tpu    (teacher engines; optional, see below)
#     WORKER_ROLES lists one role per worker in worker order. Default: worker 0 trainer, all
#     others explorer. Engines fall back gracefully: with no teacher_tpu label teachers go to
#     explorer_tpu nodes, and with no explorer_tpu label (single VM) engines just take free
#     chips. The object store is capped so tmpfs keeps room for sync checkpoints / weights.
#   - TEACHER_MODEL (optional, e.g. Qwen/Qwen3-30B-A3B): downloaded into TEACHER_DIR (tmpfs)
#     on every teacher worker (on worker 0 if there is no teacher role). tmpfs is wiped on
#     reboot; the download is free and takes about a minute.
#
# Needs passwordless ssh to the other workers (README: gcloud pushes the key once).
# Examples:
#   bash start_cluster.sh                                      # trainer + explorers
#   WORKER_ROLES="trainer teacher teacher explorer" TEACHER_MODEL=Qwen/Qwen3-30B-A3B bash start_cluster.sh
set -euo pipefail

REPO=${REPO:-$HOME/TCOD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
OBJECT_STORE_BYTES=${OBJECT_STORE_BYTES:-60000000000}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/google_compute_engine}
SSH="ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
SYNC=/dev/shm/tcod_sync
CKPT=$REPO/checkpoints
TEACHER_MODEL=${TEACHER_MODEL:-}
TEACHER_DIR=${TEACHER_DIR:-/dev/shm/models/${TEACHER_MODEL##*/}}

md() { curl -sf -m 5 -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null || true; }
if [ -n "${WORKER_IPS:-}" ]; then
    read -r -a IPS <<< "$WORKER_IPS"
else
    # "worker-id:?:ip,..." in worker order; worker 0 is this VM.
    mapfile -t IPS < <(md attributes/worker-network-endpoints | tr ',' '\n' | awk -F: 'NF>=3 {print $3}')
    num=$(md attributes/agent-worker-number)
    [ -z "$num" ] || [ "$num" = "0" ] || { echo "run this on worker 0" >&2; exit 1; }
fi
[ "${#IPS[@]}" -gt 0 ] || IPS=("$(hostname -I | awk '{print $1}')")   # single VM
HEAD=${IPS[0]}; OTHERS=("${IPS[@]:1}")

if [ -n "${WORKER_ROLES:-}" ]; then read -r -a ROLES <<< "$WORKER_ROLES"
else ROLES=(trainer); for _ in "${OTHERS[@]}"; do ROLES+=(explorer); done; fi
[ "${#ROLES[@]}" = "${#IPS[@]}" ] || { echo "WORKER_ROLES needs ${#IPS[@]} entries (workers: ${IPS[*]})" >&2; exit 1; }
[ "${ROLES[0]}" = trainer ] || { echo "worker 0 must be the trainer" >&2; exit 1; }
echo "head $HEAD, other workers: ${OTHERS[*]:-none}; roles: ${ROLES[*]}"

mkdir -p "$CKPT" "$SYNC" /dev/shm/tmp /dev/shm/ray_tmp

if [ "${#OTHERS[@]}" -gt 0 ]; then
    # 1. NFS server on worker 0, exported only to the slice's workers
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
fi

# 3. Teacher weights on tmpfs: on the teacher workers, or on worker 0 if no worker has that role
if [ -n "$TEACHER_MODEL" ]; then
    has_teacher=0; for r in "${ROLES[@]}"; do [ "$r" = teacher ] && has_teacher=1; done
    fetch="$VENV/bin/hf download $TEACHER_MODEL --local-dir $TEACHER_DIR >/dev/null"   # idempotent
    for i in "${!IPS[@]}"; do
        if [ "${ROLES[$i]}" = teacher ] || { [ $has_teacher = 0 ] && [ "$i" = 0 ]; }; then
            if [ "$i" = 0 ]; then bash -c "$fetch"; else $SSH "${IPS[$i]}" "$fetch"; fi
            echo "worker $i: $TEACHER_DIR ready"
        fi
    done
fi

# 3b. Host firewall: Ray has no authentication, and its ports listen on every interface. On a
# VM with a public address anyone who can reach them can run code on the whole cluster (a
# crypto miner was installed here that way, 2026-10-02). Accept inbound TCP only from the
# workers themselves and loopback, plus SSH; needs passwordless sudo. Not persistent across
# reboots, which is fine: this script runs after every reboot. Set TCOD_NO_FIREWALL=1 to skip.
if [ -z "${TCOD_NO_FIREWALL:-}" ]; then
    guard="sudo -n true 2>/dev/null || { echo \"\$(hostname): no passwordless sudo, Ray ports NOT firewalled\" >&2; exit 0; }
        sudo iptables -N TCOD_GUARD 2>/dev/null || sudo iptables -F TCOD_GUARD
        sudo iptables -A TCOD_GUARD -i lo -j RETURN
        sudo iptables -A TCOD_GUARD -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
        for ip in ${IPS[*]}; do sudo iptables -A TCOD_GUARD -s \$ip -j RETURN; done
        sudo iptables -A TCOD_GUARD -p tcp --dport 22 -j RETURN
        sudo iptables -A TCOD_GUARD -p tcp -j DROP
        sudo iptables -C INPUT -j TCOD_GUARD 2>/dev/null || sudo iptables -I INPUT 1 -j TCOD_GUARD"
    bash -c "$guard"
    for ip in "${OTHERS[@]}"; do $SSH "$ip" "$guard"; done
    echo "host firewall: inbound TCP limited to ${IPS[*]} and SSH"
fi

# 3c. Node-local tmpfs folder for the trainer-input buffer copy (configs point
# trainer_input.experience_buffer.path at it). A relative sqlite path lands in the home
# directory of whichever worker hosts the queue actor and can fill that worker's disk.
mkdir -p /dev/shm/tcod_buffers
for ip in "${OTHERS[@]}"; do $SSH "$ip" 'mkdir -p /dev/shm/tcod_buffers'; done

# 4. Ray cluster (label = <role>_tpu, sized to the worker's chip count)
RAY=$VENV/bin/ray
chips() { if [ "$1" = 0 ]; then ls /dev/accel* 2>/dev/null | wc -l; else $SSH "${IPS[$1]}" 'ls /dev/accel* 2>/dev/null | wc -l'; fi; }
$RAY stop --force >/dev/null 2>&1 || true
n0=$(chips 0)
$RAY start --head --port=6379 --temp-dir=/dev/shm/ray_tmp --include-dashboard=false \
    --object-store-memory="$OBJECT_STORE_BYTES" --resources="{\"trainer_tpu\": $n0}" >/dev/null
echo "$HEAD: trainer ($n0 chips)"
for i in "${!OTHERS[@]}"; do
    ip=${OTHERS[$i]}; role=${ROLES[$((i + 1))]}; n=$(chips $((i + 1)))
    case $role in explorer|teacher) ;; *) echo "unknown role '$role' (use trainer/explorer/teacher)" >&2; exit 1 ;; esac
    $SSH "$ip" "$RAY stop --force >/dev/null 2>&1 || true
        setsid $RAY start --address=$HEAD:6379 --temp-dir=/dev/shm/ray_tmp \
            --object-store-memory=$OBJECT_STORE_BYTES --resources='{\"${role}_tpu\": $n}' \
            > /dev/shm/ray_start.log 2>&1 < /dev/null"
    echo "$ip: $role ($n chips)"
done
sleep 8
$RAY status | sed -n '/Resources/,/Demands/p' | grep -E "TPU|_tpu"
