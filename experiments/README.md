# ALFWorld OPD experiments: evaluation results and training settings

Record of every evaluated checkpoint as of 2026-10-10, with the settings each training run
used. Student is plain `Qwen/Qwen3-4B` and teacher plain `Qwen/Qwen3-30B-A3B` (not Instruct)
unless a section says otherwise.

Each method below was trained **once**. Within a single run, pooled scores move by 4-8 points
between checkpoints with no trend, so differences of 2-3 points between methods are not
established by these numbers.

## How to read the tables

- **Evaluation protocol** (`scripts/tpu/eval/`, client in `scripts/tpu/eval/client/`):
  `12_evaluate_checkpoint.py --full-memory`, temperature 0.4, max_tokens 4096, 40960-token
  context, at most 30 environment steps, 8 client workers per server. 4 repetitions over the
  140 seen (`test.jsonl`) and 134 unseen (`test_unseen.jsonl`) tasks. Prompts are this
  repo's ALFWorld template (`trinity/common/workflows/envs/TCOD/alfworld/utils.py`).
- **Unseen / Seen / Pooled**: success rate in %, mean ± sample standard deviation over the
  repetitions. Pooled = (seen + unseen successes) / 274, computed per repetition.
- **Best rep**: the single repetition with the highest pooled score, with that repetition's
  unseen / seen. It is optimistic by construction (about 1-4 points above the mean).
- **Errored games**: games that raised an error, summed over all repetitions (out of 1096).
- **Where the raw files are**, on TPU worker 0: `~/TCOD/checkpoints/eval_results/` (runs
  trained on this slice from the agree + look-ahead run on, and the external model),
  `~/alfworld_ts_probe/data/` (GPU checkpoints, zero-shot models, the first TPU runs) and
  `~/TCOD/deferred_plain_run/eval_results/` (the run trained on a second TPU slice). Per
  repetition: `eval_<label>.rep<R>.<split>.jsonl` and `.summary.json`.

## Settings shared by all TPU training runs

| Setting | Value |
|---|---|
| Algorithm | `on_policy_distill`, advantage `multi_turn_opd`, `kl_coef` 1.0 |
| Loss | PPO-style policy loss, clip range 0.2, `clip_ratio_c` 3.0, token-mean |
| Optimizer | AdamW, lr 1e-6 constant, betas (0.9, 0.999), weight decay 0.01, grad clip 1.0 |
| Batches | train batch 64 turns; rollout batch 16 games per explore step unless noted |
| Sampling | `staleness_control`, `max_staleness` 2 (turns taken in arrival order) |
| Lengths | prompt 10240 tokens, response 512 tokens, `max_env_steps` 30 |
| Rollouts | full-memory conversation, temperature 1.0, thinking mode off, seed 42 |
| Hardware | Cloud TPU v4-32: 4 VMs x 4 chips. Worker 0 trainer (4 chips, `tunix`), workers 1-2 one teacher each (4 chips), worker 3 four students (1 chip each) |
| Weight sync | `checkpoint` method, after every training step (`dynamic_by_explorer`, interval 1) unless noted |
| Training data | `train.jsonl`, 3553 games |

"Deferred" below means `defer_teacher: true`: the explorer only plays the games and the trainer
asks the teacher for the 64 turns of each training batch (`scripts/tpu/README.md`, section
10.1). "Explorer-side" means the teacher gates and scores every turn during rollout.

## Reference models (no training here)

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| Qwen3-4B (student) | 37.7 ± 2.8 | 31.6 ± 1.9 | 34.6 ± 0.9 | 35.8 (38.8 / 32.9) | 4 | 3 |
| Qwen3-30B-A3B plain (teacher) | 39.7 ± 1.5 | 42.1 ± 0.8 | 41.0 ± 0.3 | 41.2 (41.0 / 41.4) | 4 | 4 |
| Qwen3-30B-A3B Instruct | 32.1 ± 2.2 | 41.4 ± 2.1 | 36.9 ± 0.9 | 38.0 (32.1 / 43.6) | 4 | 3 |
| DASH-OPD-Qwen3-4B-ALFWorld (external) | 39.4 ± 0.7 | 42.9 ± 2.7 | 41.1 ± 1.6 | 43.1 (39.6 / 46.4) | 4 | 2 |

`DASH-OPD-Qwen3-4B-ALFWorld` is `Lucian1115/DASH-OPD-Qwen3-4B-ALFWorld` from Hugging Face
(revision 122e6eb8), evaluated unchanged; its training settings are not documented in its
model card.

## GPU runs (trained elsewhere, evaluated with the same protocol)

These checkpoints were trained on GPU with the original verl / vLLM stack and only evaluated
here. The training settings are not recorded in this repo beyond the config names in
`TCOD_examples/alfworld/` (`opd_fullmemory.yaml`, `opd_gated_fullmemory.yaml`,
`opd_gated_random.yaml`, `opd_gated_reverse.yaml`); the step behind `trained` is 250 for
`opd_fullmem_trained` and `opd_gated_fullmem_reasoning_trained` and unknown for the others.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| `opd_fullmem_trained` (vanilla OPD, step 250) | 36.0 ± 1.5 | 38.2 ± 1.7 | 37.1 ± 1.4 | 39.1 (37.3 / 40.7) | 4 | 4 |
| `opd_fullmem_step375` | 37.3 ± 2.0 | 39.1 ± 1.2 | 38.2 ± 1.4 | 39.4 (38.1 / 40.7) | 4 | 1 |
| `opd_fullmem_step450` | 40.9 ± 2.6 | 40.7 ± 1.7 | 40.8 ± 0.9 | 41.2 (39.6 / 42.9) | 4 | 1 |
| `opd_fullmem_step500` | 43.8 ± 3.9 | 40.9 ± 1.2 | 42.3 ± 2.3 | 45.3 (48.5 / 42.1) | 4 | 3 |
| `opd_2step_trained` | 30.4 ± 2.1 | 33.8 ± 0.7 | 32.1 ± 1.1 | 33.2 (32.1 / 34.3) | 4 | 2 |
| `opd_gated_fullmem_reasoning_trained` (agreement gate + reasoning, step 250) | 48.5 ± 3.8 | 39.3 ± 1.9 | 43.8 ± 2.2 | 46.4 (52.2 / 40.7) | 4 | 3 |
| `opd_gated_fullmem_reasoning_step150` | 38.6 ± 2.0 | 35.0 ± 2.3 | 36.8 ± 1.1 | 38.3 (40.3 / 36.4) | 4 | 4 |
| `opd_gated_fullmem_trained` | 31.7 ± 2.2 | 32.5 ± 2.2 | 32.1 ± 1.3 | 33.6 (31.3 / 35.7) | 4 | 3 |
| `opd_gated_step200` | 38.4 ± 3.7 | 36.4 ± 2.5 | 37.4 ± 3.0 | 41.2 (42.5 / 40.0) | 4 | 3 |
| `opd_gated_2step_trained` | 39.0 ± 2.3 | 40.9 ± 3.5 | 40.0 ± 1.7 | 41.6 (40.3 / 42.9) | 4 | 7 |
| `opd_gated_random_step250` (control) | 31.3 ± 1.6 | 31.4 ± 2.8 | 31.4 ± 1.3 | 32.5 (30.6 / 34.3) | 4 | 3 |
| `opd_gated_reverse_step250` (control) | 29.7 ± 1.4 | 34.6 ± 2.4 | 32.2 ± 1.5 | 34.3 (30.6 / 37.9) | 4 | 4 |

## TPU runs

### 1. Pipeline tests: vanilla OPD, Qwen3-1.7B student / Qwen3-8B teacher

Configs `opd_tpu_repro_1.7b_8b_tpu16.yaml` and `..._tpu16_shmsync.yaml` (same run with sync
checkpoints on tmpfs). Workflow `OPD_alfworld_workflow_fullmemory`, explorer-side, 100 steps.
Checkpoints deleted.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| `tpu16_step100` | 19.6 ± 2.8 | 19.1 ± 2.8 | 19.3 ± 1.1 | 20.4 (18.7 / 22.1) | 4 | 1 |
| `tpu16_shmsync_step97` | 22.2 ± 0.7 | 18.9 ± 2.6 | 20.5 ± 1.4 | 21.9 (22.4 / 21.4) | 4 | 1 |

### 2. Look-ahead soft (explorer-side)

Config `opd_gated_lookahead_soft_tpu.yaml`, workflow `OPD_gated_alfworld_workflow_lookahead_soft`.
Progress question on the student's next 5 steps; weight 1.0 if not making progress, 0.5
(`progress_downweight_factor`) if making progress. 250 steps, about 28 h. Checkpoints deleted.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 38.8 ± 4.6 | 34.6 ± 3.6 | 36.7 ± 4.1 | 42.7 (45.5 / 40.0) | 4 | 3 |
| step 100 | 38.4 ± 5.6 | 38.0 ± 2.0 | 38.2 ± 3.3 | 42.7 (45.5 / 40.0) | 4 | 3 |
| step 150 | 37.3 ± 3.4 | 35.7 ± 1.3 | 36.5 ± 1.7 | 38.3 (39.6 / 37.1) | 4 | 2 |
| step 200 | 38.2 ± 2.1 | 38.9 ± 0.4 | 38.6 ± 1.1 | 39.8 (40.3 / 39.3) | 4 | 0 |
| step 250 | 38.1 ± 1.4 | 34.6 ± 1.4 | 36.3 ± 0.5 | 36.9 (37.3 / 36.4) | 4 | 3 |

### 3. Agree + look-ahead, "sum" (explorer-side)

Config `opd_gated_agree_lookahead_tpu.yaml`, workflow `OPD_gated_alfworld_workflow_agree_lookahead`,
`gate_mode: sum`: weight = 0.5 x [teacher disagrees] + 0.5 x [student's next 5 steps not making
progress]. 250 steps, about 37 h. Checkpoints deleted (copied off the slice by the user).

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 39.2 ± 1.3 | 34.1 ± 2.3 | 36.6 ± 1.0 | 38.0 (39.6 / 36.4) | 4 | 3 |
| step 100 | 37.9 ± 1.5 | 37.3 ± 1.1 | 37.6 ± 0.9 | 38.7 (39.6 / 37.9) | 4 | 2 |
| step 150 | 39.7 ± 1.9 | 36.6 ± 2.1 | 38.1 ± 1.9 | 39.8 (41.8 / 37.9) | 4 | 1 |
| step 200 | 37.5 ± 0.7 | 37.7 ± 2.4 | 37.6 ± 1.2 | 38.7 (38.1 / 39.3) | 4 | 1 |
| step 250 | 43.3 ± 1.5 | 36.4 ± 3.2 | 39.8 ± 1.5 | 41.6 (42.5 / 40.7) | 4 | 4 |

### 4. Student look-ahead, disagreement required (explorer-side)

Workflow `OPD_gated_alfworld_workflow_agree_lookahead`, `gate_mode: disagree_required`: teacher
agrees -> 0; disagrees and the student's next 5 steps make no progress -> 1.0; disagrees and
they make progress -> 0.5.

**Rollout batch 8** (`opd_gated_disagree_lookahead_tpu.yaml`; stopped at step 50 by a
checkpoint-saving bug and resumed from the exact state). Checkpoints deleted.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 37.9 ± 2.2 | 33.6 ± 1.5 | 35.7 ± 1.3 | 37.2 (40.3 / 34.3) | 4 | 3 |
| step 100 | 33.4 ± 2.7 | 34.1 ± 1.6 | 33.8 ± 1.9 | 35.4 (35.8 / 35.0) | 4 | 3 |
| step 150 | 36.8 ± 2.8 | 35.0 ± 1.3 | 35.9 ± 1.0 | 36.9 (40.3 / 33.6) | 4 | 3 |
| step 200 | 36.6 ± 0.6 | 38.4 ± 2.3 | 37.5 ± 0.9 | 38.0 (36.6 / 39.3) | 4 | 4 |
| step 250 | 38.1 ± 1.6 | 35.5 ± 2.9 | 36.8 ± 2.1 | 39.1 (38.8 / 39.3) | 4 | 4 |

**Rollout batch 16** (`opd_gated_disagree_lookahead_b16_tpu.yaml`), 250 steps in 33.6 h.
Checkpoints deleted from the slice (copied off by the user).

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 41.8 ± 3.0 | 35.4 ± 1.7 | 38.5 ± 2.3 | 40.1 (44.0 / 36.4) | 4 | 4 |
| step 100 | 43.3 ± 2.2 | 34.1 ± 2.5 | 38.6 ± 0.5 | 39.1 (41.0 / 37.1) | 4 | 3 |
| step 150 | 47.2 ± 0.7 | 37.5 ± 1.9 | 42.2 ± 1.1 | 43.4 (47.0 / 40.0) | 4 | 5 |
| step 200 | 38.6 ± 3.6 | 30.0 ± 1.3 | 34.2 ± 1.3 | 35.8 (42.5 / 29.3) | 4 | 3 |
| step 250 | 42.7 ± 1.3 | 36.8 ± 1.7 | 39.7 ± 0.9 | 40.9 (44.0 / 37.9) | 4 | 3 |

Data pattern of this run (and of every explorer-side run): each explore batch generated
with model version 2k fed two training steps, 2k+1 (its first 64 turns, 1 version old) and
2k+2 (the next 64 turns, 2 versions old). Turns are written in the order games finish, so
the first 64 are mostly short solved games (54% of turns) and the next 64 mostly failed ones.

### 5. Vanilla OPD, deferred

Config `opd_fullmemory_deferred_tpu.yaml`, workflow `OPD_alfworld_workflow_fullmemory` with
`defer_teacher: true`, free-running explorer. 250 steps in 7 h 23 min, final checkpoint only.
Checkpoint deleted. GPU reference: `opd_fullmem_trained` above.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 250 | 38.1 ± 1.8 | 35.7 ± 2.5 | 36.9 ± 2.0 | 38.0 (39.6 / 36.4) | 4 | 2 |

### 6. Student look-ahead, disagreement required, deferred

Same gate and weights as section 4, rollout batch 16, teacher on the trainer side.

**Plain deferred** (free-running explorer: `dynamic_by_explorer`, interval 1). Trained on a
second TPU v4-32 slice with the same layout, code at commit `50fd022`, config
`opd_gated_disagree_lookahead_deferred_tpu.yaml` as of that commit (copy in
`deferred_plain_run/`, not tracked). Every training step takes the first 64 turns of a new
batch, 2 versions old. Eval labels `disagree_la_deferred_step*`.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 41.6 ± 1.8 | 37.0 ± 2.1 | 39.2 ± 0.9 | 40.1 (40.3 / 40.0) | 4 | 4 |
| step 100 | 36.9 ± 3.3 | 33.4 ± 2.9 | 35.1 ± 2.8 | 38.7 (41.8 / 35.7) | 4 | 2 |
| step 150 | 38.4 ± 2.2 | 33.9 ± 0.9 | 36.1 ± 0.9 | 36.9 (41.0 / 32.9) | 4 | 3 |
| step 200 | 41.4 ± 1.6 | 35.2 ± 1.2 | 38.2 ± 1.1 | 39.8 (43.3 / 36.4) | 4 | 3 |
| step 250 | 43.7 ± 3.3 | 36.1 ± 1.7 | 39.8 ± 1.9 | 42.3 (48.5 / 36.4) | 4 | 4 |

**Fixed sync** (`sync_style: fixed`, `sync_interval: 2`; config
`opd_gated_disagree_lookahead_deferred_fixed2_s150_tpu.yaml`). Reproduces the 1-old / 2-old
pattern of the explorer-side run. Ran to step 150 (8 h 16 min), then continued from the exact
step-150 state to 250. Scores decline after step 100; cause not found. Checkpoints deleted.
Not recommended.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 40.7 ± 1.0 | 32.0 ± 2.1 | 36.2 ± 1.4 | 37.6 (41.8 / 33.6) | 4 | 3 |
| step 100 | 44.0 ± 2.4 | 34.6 ± 2.1 | 39.2 ± 1.1 | 40.5 (47.0 / 34.3) | 4 | 1 |
| step 150 | 34.1 ± 0.9 | 34.1 ± 2.0 | 34.1 ± 1.3 | 35.8 (35.1 / 36.4) | 4 | 1 |
| step 200 | 32.8 ± 4.2 | 33.8 ± 1.5 | 33.3 ± 2.4 | 36.5 (38.1 / 35.0) | 4 | 3 |
| step 250 | 30.0 ± 3.7 | 31.4 ± 0.6 | 30.7 ± 1.8 | 32.5 (33.6 / 31.4) | 4 | 3 |

A third variant, weight 0.8 instead of 0.5 (`opd_gated_disagree_lookahead_w08_deferred_tpu.yaml`,
plain deferred), stopped at step 200 when tmpfs filled and was deleted without evaluation.

### 7. Student + teacher look-ahead, deferred

Workflow `OPD_gated_alfworld_workflow_student_teacher_lookahead`, plain deferred, rollout
batch 16. Teacher agrees -> 0. Where it disagrees, two look-aheads run: the progress question
on the student's next 5 steps, and the teacher playing 5 steps itself from that position in a
replayed game and judging them. With S = student not making progress and T = teacher making
progress: S and T -> `both_weight`, exactly one -> `one_weight`, neither -> `neither_weight`.

**Weights 1.0 / 0.5 / 0.0** (`opd_gated_student_teacher_lookahead_deferred_tpu.yaml`). Stopped
at step 241 of 250 when tmpfs filled; the step-241 checkpoint is complete. Checkpoints 50-241
archived on worker 2 (`~/ckpt_archive/`).

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 41.6 ± 3.7 | 37.5 ± 1.7 | 39.5 ± 1.7 | 41.6 (45.5 / 37.9) | 4 | 2 |
| step 100 | 44.4 ± 2.5 | 35.9 ± 3.3 | 40.1 ± 1.2 | 40.9 (47.8 / 34.3) | 4 | 3 |
| step 150 | 39.9 ± 1.0 | 38.6 ± 1.2 | 39.2 ± 1.1 | 40.5 (41.0 / 40.0) | 4 | 2 |
| step 200 | 42.9 ± 3.9 | 38.9 ± 3.3 | 40.9 ± 3.2 | 44.9 (47.8 / 42.1) | 4 | 3 |
| step 241 | 44.6 ± 5.1 | 40.0 ± 2.1 | 42.2 ± 2.7 | 46.0 (52.2 / 40.0) | 4 | 1 |

**Weights 1.0 / 0.5 / 0.2** (`opd_gated_student_teacher_lookahead_w02_deferred_tpu.yaml`),
250 steps in 11 h 16 min. Checkpoints deleted.

| Checkpoint | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|
| step 50 | 39.7 ± 2.6 | 35.5 ± 1.2 | 37.6 ± 1.5 | 39.1 (41.0 / 37.1) | 4 | 4 |
| step 100 | 39.4 ± 2.1 | 36.8 ± 1.4 | 38.0 ± 1.0 | 39.4 (41.0 / 37.9) | 4 | 2 |
| step 150 | 42.2 ± 2.3 | 40.4 ± 2.5 | 41.2 ± 2.4 | 43.8 (44.8 / 42.9) | 4 | 3 |
| step 200 | 36.2 ± 2.2 | 35.9 ± 2.2 | 36.0 ± 2.2 | 37.6 (37.3 / 37.9) | 4 | 1 |
| step 250 | 38.1 ± 1.5 | 36.2 ± 3.7 | 37.1 ± 2.2 | 39.1 (40.3 / 37.9) | 4 | 3 |

### 8. FutureBridge-OPD (FTB)

Config `ftb_qwen3_30b_to_4b_tpu.yaml`, workflow `FutureBridgeAlfworldWorkflow`, copied from
https://github.com/ChenChiShui/FutureBridge-OPD (commit e73603a) with the release's prompt
templates (`ftb_release_utils.py`), curriculum and defaults; explorer-side; 200 steps;
training data `train_expert.jsonl`. Launched 2026-10-10, stopped at step 37 when worker 1's
disk filled, resumed from the exact step-37 state. **In progress when this file was written:
no evaluation yet.** Note that the evaluation client uses this repo's prompt template, not
the release's `<think>` template the run is trained on.

## Pooled scores side by side (4B student, 30B-A3B teacher)

| Step | Explorer-side, batch 16 (4) | Plain deferred (6) | Fixed sync (6) | Student + teacher look-ahead (7) | Same, weight 0.2 (7) |
|---|---|---|---|---|---|
| 50 | 38.5 | 39.2 | 36.2 | 39.5 | 37.6 |
| 100 | 38.6 | 35.1 | 39.2 | 40.1 | 38.0 |
| 150 | 42.2 | 36.1 | 34.1 | 39.2 | 41.2 |
| 200 | 34.2 | 38.2 | 33.3 | 40.9 | 36.0 |
| 250 | 39.7 | 39.8 | 30.7 | 42.2 (step 241) | 37.1 |
| Mean | 38.6 | 37.7 | 34.7 | 40.4 | 38.0 |

References on the same scale: Qwen3-4B zero-shot 34.6, Qwen3-30B-A3B zero-shot 41.0, GPU
vanilla OPD step 250 37.1, GPU agreement gate + reasoning 43.8.
