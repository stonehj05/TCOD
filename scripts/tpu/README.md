# TCOD on Cloud TPU: setup and operations guide

This guide is written for an engineer or an agent who has to get TCOD's on-policy
distillation (OPD) training running on Cloud TPU, possibly on a **different TPU
configuration** from the one it was developed on. Read sections 1-3 before touching anything;
section 5 is the part to adapt when your chip counts differ.

Contents

1. [What this is](#1-what-this-is)
2. [What was verified, and on what hardware](#2-what-was-verified-and-on-what-hardware)
3. [How it works](#3-how-it-works)
4. [Setup from a fresh slice](#4-setup-from-a-fresh-slice)
5. [Adapting to a different TPU configuration](#5-adapting-to-a-different-tpu-configuration)
6. [Pre-flight checks](#6-pre-flight-checks)
7. [Launching, monitoring, stopping](#7-launching-monitoring-stopping)
8. [Evaluation](#8-evaluation)
9. [Resuming a run](#9-resuming-a-run)
10. [TPU-specific config keys](#10-tpu-specific-config-keys)
11. [Known issues and gotchas](#11-known-issues-and-gotchas)
12. [File map](#12-file-map)

---

## 1. What this is

TCOD trains with Trinity-RFT: an *explorer* plays ALFWorld games with a student model served
by vLLM, a teacher model scores (and, in the gated variants, judges) each turn, and a *trainer*
updates the student from those turns. Upstream, the engines are vLLM on CUDA and the trainer is
verl (PyTorch), with NCCL weight sync.

The `tpu-port` branch adds TPU backends and leaves Trinity's orchestration (async explorer and
trainer, scheduler, workflows, buffer, staleness control) unchanged:

| Piece | GPU | TPU |
|---|---|---|
| Rollout / teacher engines | `engine_type: vllm` | `engine_type: vllm_tpu` (vLLM's TPU backend, `tpu-inference`) |
| Trainer | `trainer_type: verl` | `trainer_type: tunix` (JAX / Tunix) |
| Weight sync | NCCL | `sync_method: checkpoint`: trainer writes an HF checkpoint, engines load it in place |
| Placement | GPU placement groups | Ray `TPU` resource plus role labels |

GPU behavior is untouched: every TPU code path is behind `vllm_tpu` / `tunix`.

## 2. What was verified, and on what hardware

Development hardware: **one Cloud TPU v4-32 slice in us-central2-b: 4 VMs ("workers"), 4 chips
each, 30.75 GiB HBM per chip, 400 GB RAM and a 97 GB boot disk per VM.**

Verified end to end on that slice:

| Run | Result |
|---|---|
| Qwen3-1.7B student / Qwen3-8B teacher, `OPD_alfworld_workflow_fullmemory`, ~100 steps | 5 h 15 min on 16 chips |
| Qwen3-4B student / Qwen3-30B-A3B teacher, `OPD_gated_alfworld_workflow_lookahead_soft`, 250 steps | 28 h 10 min on 16 chips |
| Qwen3-4B / Qwen3-30B-A3B, `OPD_gated_alfworld_workflow_agree_lookahead`, 250 steps | ~37 h on 16 chips (projected: at step 238 of 250 when this was written) |
| avg@4 ALFWorld eval of one checkpoint (seen + unseen) | ~12 min on 16 chips |

Verified in isolation: the JAX loss and advantage equal Trinity's PyTorch `PPOPolicyLossFn` /
`MultiTurnOpdAdvantage` (`tests/tpu/test_tunix_loss_parity.py`); the HF checkpoint export is
bit-exact; exact-state save/restore continues a run bit-identically (CPU test).

**Not verified** (treat as untested code paths):

- Any TPU generation other than v4 (v5e, v6e, ...), any chips-per-host count other than 4, and
  single-VM slices. The scripts detect chip counts instead of assuming 4, but only the 4x4
  layout has actually run.
- `scripts/tpu/setup_node.sh` and the current `start_cluster.sh` as whole scripts on a fresh
  slice. Every step in them was run by hand during bring-up; the scripts are those steps
  collected. Expect to fix small things on first use.
- Resuming a run on TPU (section 9): saving the state works on TPU, loading it was tested on
  CPU only.
- Models other than dense Qwen2/Qwen3 as the *student*. (The MoE teacher Qwen3-30B-A3B works:
  teachers are only served, never trained.)

## 3. How it works

```
 worker 0 (head)                      other workers
 +-----------------------------+      +--------------------------------------+
 | Trinity launcher (driver)   |      | vLLM-TPU engines (Ray actors)        |
 | Tunix trainer worker        |      |   students: 1 chip each              |
 |   fp32 weights + AdamW      |      |   teacher(s): 1..N chips each        |
 |   on N chips of this host   |      | WorkflowRunners (CPU): ALFWorld envs |
 +--------------+--------------+      +-------------------+------------------+
                |  HF checkpoint (bf16) per sync                |
                +--> /dev/shm/tcod_sync  --- NFS, same path --->+  engines reload in place
                +--> checkpoints/        --- NFS, same path     (full checkpoints, logs)
```

- **Engines** (`trinity/common/models/vllm_tpu_model.py`): one Ray actor per engine, holding
  `tensor_parallel_size` chips. Weight reload happens inside the running engine
  (`vllm_tpu_worker.py`), no restart.
- **Trainer** (`trinity/trainer/tunix_trainer.py`, `trinity/trainer/tunix/`): a Ray actor that
  keeps fp32 master weights and AdamW state sharded across its chips, and reproduces verl's OPD
  step: recompute the student's log-probs, advantage `kl_coef * (teacher - old)`, PPO loss,
  gradient clip, update. One row per chip per micro-batch, with gradient accumulation.
- **Weight sync**: after a training step the trainer writes a bf16 HF checkpoint to
  `trainer.sync_checkpoint_dir`; the Synchronizer publishes its path; engines load it. Old sync
  checkpoints are deleted automatically.
- **Shared paths**: every worker must see `checkpoints/` and the sync directory at the same
  path. `start_cluster.sh` exports them from worker 0 over NFS. On a single VM nothing needs
  sharing.
- **Placement**: Ray exposes chips as the `TPU` resource. `start_cluster.sh` labels each worker
  by role, and the code requests those labels only if they exist in the cluster.

## 4. Setup from a fresh slice

All commands run on **worker 0** unless stated. Paths assume the repo at `~/TCOD` and the
environment at `~/venv-vllm-tpu` on every worker (both can be overridden with `REPO` / `VENV`).

### 4.0 SSH between workers (once per slice)

`gcloud` pushes your key to the slice the first time; afterwards the scripts use plain ssh over
internal IPs with `~/.ssh/google_compute_engine` (override with `SSH_KEY`).

```bash
TPU_NAME=<your-tpu> ZONE=<zone>
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command true
```

Check you are on quota that is free for your project before creating slices; TPU grants are
tied to specific zones and to spot vs on-demand.

### 4.1 Code, environment, data, student model on every worker (once per slice, ~20-30 min)

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command '
  [ -d ~/TCOD ] || git clone -b tpu-port https://github.com/stonehj05/TCOD ~/TCOD
  bash ~/TCOD/scripts/tpu/setup_node.sh'
```

`setup_node.sh` (idempotent) does, per VM:

1. Installs `uv`, Python 3.12.14 and the venv.
2. Installs the **exact** tested package set (`requirements-tpu.txt`, 321 pins) with
   `--no-deps`, then the repo as an editable install with `--no-deps`.
   **Do not** `pip install -r` normally: `tpu-inference` pins `qwix==0.1.2` while `google-tunix`
   needs `0.1.8`, so dependency resolution fails. Key versions: `vllm-tpu` / `tpu-inference`
   0.29.0, `jax` 0.11.0, `libtpu` 0.0.44, `google-tunix` 0.1.7, `ray` 2.58.0.
3. Applies `tpu_inference_rpa_v3_tpu_v4.patch`: vLLM-TPU's attention kernel has no tile sizes
   for TPU v4. The patch adds a `case 4` and is harmless on other generations.
4. Downloads ALFWorld (`alfworld-download`) to `~/alf-data` and writes the task lists to
   `~/alf-data/tcod_tasks/*.jsonl` with `game_file` paths rewritten for this machine. **The
   training yaml must point at these files** (`buffer.explorer_input.taskset.path`).
5. Downloads `$MODELS` to the Hugging Face cache (default `Qwen/Qwen3-4B`).

### 4.2 Runtime: shared directories and Ray (after every reboot of the slice)

```bash
# default: worker 0 = trainer, all other workers = student engines
bash ~/TCOD/scripts/tpu/start_cluster.sh

# with dedicated teacher workers and a large teacher kept on tmpfs
WORKER_ROLES="trainer teacher teacher explorer" TEACHER_MODEL=Qwen/Qwen3-30B-A3B \
  bash ~/TCOD/scripts/tpu/start_cluster.sh
```

It prints the resulting resources, for example `16.0 TPU`, `4.0 trainer_tpu`,
`8.0 teacher_tpu`, `4.0 explorer_tpu`. See the header of the script for every option
(`WORKER_IPS` for slices without TPU-VM metadata, `OBJECT_STORE_BYTES`, `TEACHER_DIR`).
It restarts Ray, so **never run it while a training is in progress**.

### 4.3 Train

```bash
cd ~/TCOD && PATH=$HOME/venv-vllm-tpu/bin:$PATH setsid nohup \
  trinity run --config TCOD_examples/alfworld/opd_gated_lookahead_soft_tpu.yaml \
  > ~/train.log 2>&1 < /dev/null &
```

Use `setsid nohup` so the run survives a dropped connection.

## 5. Adapting to a different TPU configuration

Nothing in the Python code assumes 16 chips. What you must decide and set is the **chip
layout**, and it lives in two places that have to agree: the roles given to
`start_cluster.sh`, and the engine counts in the training yaml.

### 5.1 Find out what you have

```bash
ls /dev/accel* | wc -l                       # chips on this VM
~/venv-vllm-tpu/bin/tpu-info                 # generation, HBM per chip
bash ~/TCOD/scripts/tpu/tpu_usage.sh         # the same for every worker
```

You need: number of workers (VMs), chips per worker, HBM per chip. On v4 that is 4 chips per
VM and 30.75 GiB per chip. Other generations differ (v5e chips have roughly half that memory),
so re-derive every memory setting below instead of copying it.

### 5.2 The layout rules

Let `T` = chips for the trainer, `S` = number of student engines (1 chip each unless you set
`tensor_parallel_size`), `K` = chips per teacher engine, `M` = number of teacher engines.

1. `cluster.node_num * cluster.gpu_per_node` must equal the chips you intend to use, and Trinity
   derives the trainer's chips as **`T = node_num * gpu_per_node - (S + M*K)`**. Set `node_num`
   and `gpu_per_node` to make `T` come out right; they need not match the physical VM count if
   you leave chips unused.
2. **The trainer must sit on one VM**: `T <= chips per VM`. (A trainer spanning VMs is not
   implemented.) Trinity also requires `T` to be a whole number of "nodes" when it exceeds
   `gpu_per_node`, so keep `T <= gpu_per_node`.
3. **`buffer.train_batch_size` must be divisible by `T`** (one row per chip per micro-batch).
4. **A multi-chip engine must sit on one VM**: `K <= chips per VM`.
5. Give the trainer and each multi-chip teacher a VM where their chips are guaranteed free.
   That is what the role labels are for: without them Ray may scatter 1-chip students over
   every VM and leave no VM with `T` or `K` free chips, and the run hangs at startup.
6. The teacher is usually the bottleneck in gated workflows (it both generates judgments and
   scores every turn); students are cheap. Each explore step plays `buffer.batch_size` games
   using `S * explorer.runner_per_model` runners, so more students than
   `batch_size / runner_per_model` adds nothing.

Examples (v4, 4 chips per VM):

| Slice | Roles for `start_cluster.sh` | yaml |
|---|---|---|
| 4 VMs, small teacher (1 chip) | `trainer explorer explorer explorer` | `node_num: 4`, `gpu_per_node: 4`, students 4, teachers 8 x 1 chip, trainer 4 |
| 4 VMs, 30B teacher (4 chips) | `trainer teacher teacher explorer` | students 4, teachers 2 x TP 4, trainer 4 |
| 2 VMs, small teacher | `trainer explorer` | `node_num: 2`, students 3, teacher 1, trainer 4 |
| 1 VM with 4 chips (smoke tests) | `trainer` | `node_num: 1`, `gpu_per_node: 4`, student 1, teacher 1, trainer 2 |

On one VM no label except `trainer_tpu` exists, so engines simply take free chips; this was
verified with 1 student + 1 teacher + 2 trainer chips (1.7B / 8B).

### 5.3 Whole-host processes

A process that takes **all** chips of a VM (a 4-chip trainer or a `tensor_parallel_size: 4`
teacher on v4) needs explicit single-host TPU bounds, otherwise libtpu waits forever for the
other VMs of the slice to join ("TPU backend initialization is taking more than 60 seconds.
Did you run your code on all TPU hosts?"). `trinity/common/models/tpu_env.py` sets them
automatically from the chips-per-host count Ray reports. The bounds table there
(`1 -> 1,1,1`, `2 -> 1,2,1`, `4 -> 2,2,1`, `8 -> 2,4,1`) is **verified only for 4**. On other
hardware, run `checks/check_placement.py` (section 6): it starts placeholder actors with these
settings and fails fast if the TPU does not initialize. Override the detected count with
`TCOD_TPU_CHIPS_PER_HOST` if needed.

### 5.4 Memory budgets (measured on v4, 30.75 GiB per chip)

| What | Setting | Measured |
|---|---|---|
| Trainer, Qwen3-1.7B, 2 chips, prompts to 10,240 tokens | - | 15.8 GiB peak per chip, 84 s/step |
| Trainer, Qwen3-4B, 4 chips, prompts to 10,240 tokens | - | 19.2 GiB peak per chip (synthetic), ~24.5 GiB in a real run, ~110 s/step |
| Student engine, Qwen3-4B, 1 chip | `gpu_memory_utilization: 0.7` | 21.5 GiB steady, 23.0 GiB during a weight reload |
| Teacher, Qwen3-8B, 1 chip | `gpu_memory_utilization: 0.75` | works; 0.9 runs out of memory when scoring long prompts |
| Teacher, Qwen3-30B-A3B, 4 chips (TP 4) | `gpu_memory_utilization: 0.8` | 57 GB of weights, 321k-token KV cache, scores a 14k-token prompt in 2.4 s |

Rules of thumb for other hardware:

- Trainer state is fp32 weights + two Adam moments + one gradient buffer = **16 bytes per
  parameter**, sharded over the trainer's chips, plus roughly 4-6 GiB per chip of activations
  at 10k-token sequences. If it does not fit, give the trainer more chips or lower
  `model.max_prompt_tokens`.
- An engine needs its bf16 weights (2 bytes per parameter) inside
  `gpu_memory_utilization * HBM`, and the *rest* of the chip for compiled programs and
  temporaries. A teacher that scores long prompts needs several GiB outside the reservation
  (each 1,024-token chunk of prompt log-probs is a 0.6 GiB buffer). Lower the utilization
  if you see `RESOURCE_EXHAUSTED` in a teacher.
- Disk on worker 0: every full checkpoint is ~2 bytes per parameter (8 GB for 4B); budget
  `total_steps / save_interval` of them. tmpfs on worker 0: about 3 sync checkpoints plus,
  for resumability, 12 bytes per parameter (45 GB for 4B).
- tmpfs on a teacher VM: the teacher's weights if you use `TEACHER_MODEL`, plus Ray's object
  store (capped at 60 GB by `start_cluster.sh`). tmpfs counts against RAM.

### 5.5 Other things that change with the hardware

- `scripts/tpu/tpu_inference_rpa_v3_tpu_v4.patch` only matters on v4.
- Paths in the example yamls are for user `haojun_shi` (`/home/haojun_shi/alf-data/...`,
  `/dev/shm/models/...`). Change `taskset.path` and any local `model_path`.
- Example yamls set `node_num: 4`, `gpu_per_node: 4` and their engine counts for 16 chips.
- A VM without TPU-VM metadata: pass `WORKER_IPS="ip0 ip1 ..."` to the scripts.

## 6. Pre-flight checks

Run these, in order, on a new slice or after changing the layout. Each takes the training
yaml; `--plan` prints what it would do without using any chip.

```bash
cd ~/TCOD && PY=~/venv-vllm-tpu/bin/python CFG=TCOD_examples/alfworld/<your>.yaml

# 0. loss parity with the PyTorch implementation (CPU, seconds)
JAX_PLATFORMS=cpu $PY tests/tpu/test_tunix_loss_parity.py

# 1. placement: placeholder actors with the run's exact resource requests (no models loaded)
$PY scripts/tpu/checks/check_placement.py $CFG

# 2. engines: load every student and teacher, one student turn, teacher generation + scoring,
#    in-place weight reload (a few minutes)
$PY scripts/tpu/checks/check_engines.py $CFG

# 3. trainer memory at full sequence length (run on the trainer host, chips free, ~15 min)
$PY scripts/tpu/checks/check_trainer_memory.py $CFG
```

Then do a short real run before a long one: copy your yaml with `buffer.total_steps: 6`,
`trainer.total_steps: 6`, `trainer.save_interval: 3`, small `batch_size` / `train_batch_size`,
and confirm the console shows `Loaded [N] tensors from ...` (engines picked up trained
weights) and a `global_step_3` checkpoint.

These scripts are adapted from the checks used during bring-up. Their `--plan` modes and
config handling were run; the chip-using parts of these exact files were not re-run after the
adaptation.

## 7. Launching, monitoring, stopping

**Logs** (run directory = `checkpoints/<project>/<name>/`):

| What | Where |
|---|---|
| Console (everything) | wherever you redirected `trinity run` |
| Trainer: one line per step with metrics | `<run>/log/trainer.log` |
| Explorer: rollout metrics, weight loads | `<run>/log/explorer.log` |
| Tensorboard | `<run>/monitor/` |
| Full checkpoints | `<run>/global_step_<N>/actor/huggingface` |

```bash
tail -f <run>/log/trainer.log | grep -E "Training at step|Error"
python3 scripts/tpu/rollout_speed.py <run> 12     # explore-step timing, turns/min, gate rates
bash scripts/tpu/tpu_usage.sh                     # per-chip HBM and duty cycle on every worker
```

Things that look wrong but are not:

- The trainer spends most of its time in `Sample data for step N started`: it is waiting for
  rollouts. The explorer is the bottleneck, by design of staleness control.
- Only ~15% of generated turns are trained on. Each explore step yields ~400 turns, the
  trainer uses `train_batch_size` (64), and the rest go stale under `max_staleness`. This is a
  property of the config, the same on GPU.
- `Error in Trainer: ... StopAsyncIteration` at the very end: the explorer finished and closed
  the queue. The run still saves and exits 0.
- `RESOURCE_EXHAUSTED ... attempting to defragment and retry` in a teacher, with no
  `EngineDeadError` afterwards: the retry succeeded.
- A run ends when **either** side reaches its `total_steps`. To get exactly N training steps
  set `buffer.total_steps` (explorer) above `trainer.total_steps`, e.g. 300 vs 250.

**Stopping**: kill the `trinity run` process (find it with `pgrep -f "bin/trinity run"`); Ray
tears down the actors. Check with `tpu_usage.sh` that no process still holds a chip.

## 8. Evaluation

`scripts/tpu/eval/` runs the ALFWorld avg@N protocol on every chip of the slice:

```bash
# one checkpoint
bash scripts/tpu/eval/evaluate_checkpoint_avg4.sh <run>/global_step_250/actor/huggingface my_label
# every checkpoint of a run, one after another
bash scripts/tpu/eval/eval_all_checkpoints.sh <run> my_label_prefix
```

- The evaluation client is `scripts/tpu/eval/client/`: `12_evaluate_checkpoint.py` and its
  helpers, copied unchanged from the project's GPU evaluation code so that TPU and GPU numbers
  come from the same client. It talks to any OpenAI-compatible server; the launchers start one
  vLLM-TPU server per chip for it. Its built-in default data paths point at the original GPU
  machine and are never used here, because the launchers always pass the task files
  (`$TASK_DIR`, default `~/alf-data/tcod_tasks`). Set `PROBE_DIR` to use a different copy.
- Quick check of the client without any TPU (mock model, two games):
  `cd scripts/tpu/eval/client && python 12_evaluate_checkpoint.py --base-url http://localhost:1/v1
  --model mock --mock-model --full-memory --split unseen --task-end 2 --workers 2
  --unseen-jsonl ~/alf-data/tcod_tasks/test_unseen.jsonl --output /tmp/m.jsonl --summary /tmp/m.json`
- Protocol: `--full-memory`, temperature 0.4, 30 env steps, 4096 max tokens, 4 reps of
  seen (140 games) and unseen (134 games); the result is mean +/- std over reps.
- Results go to `$RESULT_DIR` (default `checkpoints/eval_results/`, shared across workers and
  git-ignored): `eval_<label>.rep<R>.<split>.{jsonl,summary.json}`; the final lines printed by
  the launcher are the avg@N summary (redirect its output to keep them, as
  `eval_all_checkpoints.sh` does in `avg4_<label>.log` and `avg4_<prefix>_all.log`).
- The work is split into one job per chip and merged afterwards; with fewer chips than jobs
  it runs in waves. Verified on 16 chips.
- Do not run it while training holds the chips.
- Check the `errors` count in the summary. A handful of games per eval fail inside ALFWorld
  (`IndexError: Cannot choose from an empty sequence`) and count as failures; that is normal.
  Many errors mean an environment problem (see "disk" in section 11).

## 9. Resuming a run

At every full checkpoint the trainer also saves its exact state (fp32 weights, Adam state,
step) to `<sync_checkpoint_dir>/<project>/<name>_resume/global_step_<N>` (latest only). To
resume a stopped or crashed run from its last full checkpoint, set
`continue_from_checkpoint: true` and launch the same command.

Limits: the state lives on tmpfs and **does not survive a reboot**; the explorer resumes from
its saved position in the task list and loses games that were in flight; and **loading the
state has only been tested on CPU**, so try it on a short run before depending on it. Without
the saved state the trainer refuses to resume, because the bf16 checkpoint alone would round
the weights and reset the optimizer.

`scripts/tpu/export_live_state.py CONFIG STEP` writes the same state from a trainer that is
currently running (it executes inside the live trainer actor between two steps).

## 10. TPU-specific config keys

Start from `TCOD_examples/alfworld/opd_gated_lookahead_soft_tpu.yaml`; its header lists every
difference from the GPU config. In short:

```yaml
continue_from_checkpoint: false          # true = resume (section 9)
cluster:
  node_num: 4                            # see 5.2: these two set the trainer's chip count
  gpu_per_node: 4
buffer:
  total_steps: 300                       # explorer steps; keep above trainer.total_steps
  explorer_input:
    taskset:
      path: /home/<user>/alf-data/tcod_tasks/train.jsonl
explorer:
  rollout_model:
    engine_type: vllm_tpu
    engine_num: 4
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.7
  auxiliary_models:
    - model_path: /dev/shm/models/Qwen3-30B-A3B   # or a Hugging Face id
      engine_type: vllm_tpu
      engine_num: 2
      tensor_parallel_size: 4
      gpu_memory_utilization: 0.8
synchronizer:
  sync_method: 'checkpoint'              # required; NCCL is CUDA-only
trainer:
  trainer_type: tunix
  total_steps: 250
  save_interval: 50                      # mind worker 0's disk (5.4)
  sync_checkpoint_dir: /dev/shm/tcod_sync   # tmpfs, shared by start_cluster.sh
monitor:
  monitor_type: tensorboard
```

The tunix trainer supports `algorithm_type: on_policy_distill` with a constant learning rate
and reads `algorithm.optimizer` (lr, betas, weight_decay), `algorithm.advantage_fn_args.kl_coef`,
`algorithm.policy_loss_fn_args` and `trainer.grad_clip`. verl-only keys (`use_dynamic_bsz`,
`ulysses_sequence_parallel_size`, ...) are not used.

## 11. Known issues and gotchas

Environment

- Install with `--no-deps` from the pinned file (4.1). A normal resolve fails on `qwix`.
- Trinity is installed `--no-deps`: the TPU environment has no verl. Code on this branch does
  not import verl on the TPU path.
- Every worker imports its **own copy of the repo**. After changing code, copy it to all
  workers before launching: `rsync -a --exclude checkpoints/ ~/TCOD/ <ip>:TCOD/`.

TPU runtime

- First use of every compiled program is slow: the first training step takes ~8-10 minutes,
  engine start a few minutes, first calls of each kind tens of seconds.
- `logprobs=0` returns no log-probs on vLLM-TPU; the engine maps it to 1.
- The student must be trained with fp32 master weights; bf16 loses updates at `lr=1e-6`.
- Processes that take a whole VM need the bounds in `tpu_env.py` (5.3).
- `tpu-info` reads only port 8431. Engines serve metrics on 8431 + their first chip, and
  `tpu_usage.sh` queries each port.
- Stop a vLLM server with SIGTERM, not `kill -9`: the latter orphans its `VLLM::EngineCore`
  child, which keeps the chip.

Memory and storage

- In-place weight reload frees each old tensor as soon as its replacement is loaded. Before
  that fix a 4B student ran out of HBM on reload.
- A full root disk on any VM breaks ALFWorld: it copies a planner library into the temp dir
  for every game, and every game then "fails". Training configs set `TMPDIR=/dev/shm/tmp`;
  the eval scripts do the same.
- tmpfs and NFS mounts disappear on reboot: rerun `start_cluster.sh`, and the teacher weights
  are fetched again.
- `checkpoints/` is on worker 0's disk and is small (97 GB on v4 VMs). Move finished runs'
  checkpoints to another worker's disk before starting a new run.

Operating

- With `pkill -f <pattern>`, a pattern that appears in your own command line kills your shell.
  Put such commands in a script file, or kill by PID.
- Do not name scratch scripts like importable modules: Python may import them.
- Config validation (`check_and_update`) creates the run directory; if one already exists and
  `continue_from_checkpoint` is false, the new run gets a timestamp suffix.

## 12. File map

| Path | What |
|---|---|
| `trinity/common/models/vllm_tpu_model.py` | `vllm_tpu` engine (subclass of the GPU `vLLMRolloutModel`) |
| `trinity/common/models/vllm_tpu_worker.py` | in-place weight reload inside the engine |
| `trinity/common/models/tpu_env.py` | whole-host TPU bounds, chips-per-host detection |
| `trinity/common/models/__init__.py` | `_create_tpu_inference_models`: engine placement |
| `trinity/trainer/tunix_trainer.py` | `tunix` trainer wrapper (config mapping, checkpoints, resume) |
| `trinity/trainer/tunix/worker.py` | the training step (JAX) |
| `trinity/trainer/tunix/logps.py` | completion log-probs and the OPD/PPO loss |
| `trinity/trainer/tunix/hf_io.py` | HF checkpoint load / export |
| `trinity/trainer/tunix/resume.py` | exact-state save / restore |
| `trinity/manager/synchronizer.py` | publishes sync checkpoints (`_find_tunix_latest_state_dict`), cleanup |
| `trinity/common/workflows/envs/TCOD/alfworld/OPD_gated_workflow_agree_lookahead.py` | combined agreement + look-ahead gate with batched teacher calls |
| `TCOD_examples/alfworld/*_tpu*.yaml` | TPU configs |
| `scripts/tpu/setup_node.sh`, `requirements-tpu.txt`, `tpu_inference_rpa_v3_tpu_v4.patch` | per-VM environment |
| `scripts/tpu/start_cluster.sh` | NFS shares, teacher weights, Ray cluster with role labels |
| `scripts/tpu/checks/` | pre-flight checks (section 6) |
| `scripts/tpu/eval/` | avg@N evaluation on all chips; `client/` is the evaluation client |
| `scripts/tpu/tpu_usage.sh`, `rollout_speed.py`, `export_live_state.py` | monitoring and tools |
| `tests/tpu/`, `tests/workflow/` | loss parity test; gate logic test for the combined workflow |
