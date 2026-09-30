#!/bin/bash
# One-time setup of ONE TPU VM (run it on every worker of the slice, e.g. via
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "bash ~/TCOD/scripts/tpu/setup_node.sh"
# after the repo is on each worker). Idempotent: safe to re-run.
#
# Produces: ~/venv-vllm-tpu (exact pinned env, see requirements-tpu.txt), the TPU v4 kernel
# patch applied, ALFWorld data in ~/alf-data with remapped TCOD task files, and the HF models.
set -euo pipefail

REPO=${REPO:-$HOME/TCOD}
VENV=${VENV:-$HOME/venv-vllm-tpu}
ALF=${ALF:-$HOME/alf-data}
MODELS=${MODELS:-"Qwen/Qwen3-1.7B Qwen/Qwen3-8B"}
HERE=$REPO/scripts/tpu

# 1. uv + Python 3.12.14 + venv
command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
[ -x "$VENV/bin/python" ] || { "$UV" python install 3.12.14; "$UV" venv --python 3.12.14 "$VENV"; }

# 2. Exact package set. --no-deps on purpose: the tested env uses qwix 0.1.8 (needed by
#    google-tunix) although tpu-inference pins 0.1.2, so a normal resolve would fail.
VIRTUAL_ENV=$VENV "$UV" pip install --no-deps -r "$HERE/requirements-tpu.txt"
VIRTUAL_ENV=$VENV "$UV" pip install --no-deps -e "$REPO"

# 3. vLLM-TPU ragged-paged-attention v3 kernel: add TPU v4 tile sizes (v4 has 16 MiB VMEM).
SP=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
K=$SP/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py
if ! grep -q "case 4:  # added locally" "$K"; then
    cp "$K" "$K.orig"
    patch -p1 -d "$SP" < "$HERE/tpu_inference_rpa_v3_tpu_v4.patch"
fi

# 4. ALFWorld games + TCOD task files with game_file paths pointing at $ALF
[ -d "$ALF/json_2.1.1/valid_unseen" ] || "$VENV/bin/alfworld-download" --data-dir "$ALF"
mkdir -p "$ALF/tcod_tasks"
for f in train test test_unseen train_hard; do
    sed "s#/data/hs2352/LookAheadOPD/alf-data/#$ALF/#" "$REPO/TCOD_examples/alfworld/alfworld_data/$f.jsonl" > "$ALF/tcod_tasks/$f.jsonl"
done
"$VENV/bin/python" - "$ALF" <<'EOF'
import json, os, sys
alf = sys.argv[1]
for f in ["train", "test", "test_unseen"]:
    rows = [json.loads(l) for l in open(f"{alf}/tcod_tasks/{f}.jsonl")]
    missing = sum(not os.path.exists(r["game_file"]) for r in rows)
    print(f"{f}: {len(rows)} tasks, {missing} missing game files")
    assert missing == 0, "ALFWorld download incomplete or task paths wrong"
EOF

# 5. Models (Hugging Face cache)
for m in $MODELS; do "$VENV/bin/hf" download "$m" >/dev/null; echo "downloaded $m"; done

# 6. Scratch dirs used by the configs / Ray
mkdir -p /dev/shm/tmp /dev/shm/ray_tmp /dev/shm/tcod_sync "$REPO/checkpoints"
echo "setup_node: done on $(hostname)"
