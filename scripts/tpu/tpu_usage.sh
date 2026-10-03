#!/bin/bash
# TPU usage on every VM of the slice: which process holds each chip, HBM used, duty cycle.
# Run on worker 0:  bash ~/TCOD/scripts/tpu/tpu_usage.sh
#
# `tpu-info` only reads libtpu's metrics server on localhost:8431, which covers one process per
# VM. TPU processes launched by TCOD (vLLM-TPU engines, the eval servers) serve metrics on
# 8431 + their first chip, so this queries one port per chip and reports chip = (port - 8431) +
# device index. A process holding several chips (e.g. the 4-chip trainer on 8431) reports all
# of them from its one port. Processes started without a per-chip port all share 8431; then
# only one of them is visible.
VENV=${VENV:-$HOME/venv-vllm-tpu}
SSH="ssh -i ${SSH_KEY:-$HOME/.ssh/google_compute_engine} -o BatchMode=yes -o ConnectTimeout=10"

# Workers: $WORKER_IPS ("ip0 ip1 ..."), else TPU-VM metadata, else just this VM.
md() { curl -sf -m 5 -H Metadata-Flavor:Google "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null || true; }
if [ -n "${WORKER_IPS:-}" ]; then read -r -a IPS <<< "$WORKER_IPS"
else mapfile -t IPS < <(md attributes/worker-network-endpoints | tr ',' '\n' | awk -F: 'NF>=3 {print $3}'); fi
[ "${#IPS[@]}" -gt 0 ] || IPS=(localhost)

read -r -d '' PER_VM <<'EOF'
source "$VENV/bin/activate"
tpu-info -p 2>/dev/null | grep -vE '^\s*$'
python - <<'PY' 2>/dev/null
from tpu_info import device, metrics
import glob
chip_type, _ = device.get_local_chips()
n_chips = len(glob.glob("/dev/accel*")) or 4
rows = {}
for port in range(8431, 8431 + n_chips):
    try:
        usage = metrics.get_chip_usage(chip_type, addr=f"localhost:{port}")
    except Exception:
        continue
    for u in usage:
        rows.setdefault((port - 8431) + u.device_id, (port, u))
print(f"{'chip':>4}  {'port':>5}  {'HBM used / total':>22}  {'duty cycle':>10}")
for chip in range(n_chips):
    if chip in rows:
        port, u = rows[chip]
        print(f"{chip:>4}  {port:>5}  {u.memory_usage / 2**30:8.2f} / {u.total_memory / 2**30:5.2f} GiB  {u.duty_cycle_pct:9.2f}%")
    else:
        print(f"{chip:>4}  {'-':>5}  {'no metrics server':>22}  {'-':>10}")
PY
EOF

for i in "${!IPS[@]}"; do
    ip=${IPS[$i]}
    echo "=================== worker $i ($ip) ==================="
    if [ "$i" = 0 ]; then VENV=$VENV bash -c "$PER_VM"; else $SSH "$ip" "VENV=$VENV bash -c $(printf %q "$PER_VM")"; fi
done
