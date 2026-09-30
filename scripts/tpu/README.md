# TCOD on Cloud TPU (v4-32): from a fresh slice to a training run

Tested on a v4-32 (4 workers x 4 chips) in us-central2-b. Worker 0 is where you type
everything below; it also serves the NFS shares and is the Ray head.

## 0. Once per new slice: ssh between workers

`gcloud` pushes your key to the slice's metadata the first time; after that the scripts use
plain ssh over internal IPs.

```bash
TPU_NAME=my-tpu-v4 ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command true
```

## 1. Once per new slice: code + environment on every worker (~20-30 min)

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command '
  [ -d ~/TCOD ] || git clone -b tpu-port https://github.com/stonehj05/TCOD ~/TCOD
  bash ~/TCOD/scripts/tpu/setup_node.sh'
```

`setup_node.sh` installs the exact tested package set (`requirements-tpu.txt`, installed with
`--no-deps`), applies the TPU v4 attention-kernel patch to vLLM-TPU, downloads ALFWorld into
`~/alf-data` (task files remapped into `~/alf-data/tcod_tasks/`) and the Qwen3-1.7B/8B weights.

## 2. After every (re)start of the slice: NFS + Ray

```bash
bash ~/TCOD/scripts/tpu/start_cluster.sh
```

Expect `16.0 TPU`, `12.0 explorer_tpu`, `4.0 trainer_tpu`.

## 3. Train (detached)

```bash
cd ~/TCOD && PATH=$HOME/venv-vllm-tpu/bin:$PATH setsid nohup \
  trinity run --config TCOD_examples/alfworld/opd_tpu_repro_1.7b_8b_tpu16_shmsync.yaml \
  > ~/train.log 2>&1 < /dev/null &
```

Logs: `~/train.log`, per-role logs in `checkpoints/ALFWORLD_TCOD/<run>/log/`.

## Notes

- Code changes must reach every worker before a launch: each VM imports its own copy
  (`git pull` on all workers, or `rsync -a --exclude checkpoints/ ~/TCOD/ <ip>:TCOD/`).
- Chip split lives in the yaml: trainer = `node_num x gpu_per_node` - engine chips, and must
  be 4 (one whole VM, the one labelled `trainer_tpu`).
- A process holding all 4 chips of a VM needs explicit single-host TPU bounds (handled in
  `trinity/trainer/tunix_trainer.py`), or libtpu waits for the whole slice.
