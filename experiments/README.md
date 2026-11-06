# OPD experiments on TPU: evaluation results and training settings

Record of every evaluated checkpoint as of 2026-11-05 (ALFWorld, ScienceWorld, WebShop), with
the settings each training run used and the training time to each checkpoint. Section headers
name the student / teacher pair; the ALFWorld sections up to section 12 use plain
`Qwen/Qwen3-4B` and `Qwen/Qwen3-30B-A3B` (not Instruct).

Each method below was trained **once**. Within a single run, scores move by several points
between checkpoints with no trend, so differences of 2-3 points between methods are not
established by these numbers.

## How to read the tables

- **Train h**: hours of training up to that checkpoint, from the trainer log
  (`<run>/log/trainer.log`, "Training at step N finished"). Counted from the end of step 1.
  A step that took more than 4x the median step (a crash, a pause before a resume, the
  wait for an evaluation) counts as one median step. Runs that alternate short and long
  steps (explorer-side runs, where every explore batch feeds two training steps) use 4x the
  75th percentile as the threshold and the mean step as the fill. Steps after an exact
  resume continue the count. `n/a`: the trainer log is gone.
- **ALFWorld evaluation protocol** (`scripts/tpu/eval/`, client in `scripts/tpu/eval/client/`):
  `12_evaluate_checkpoint.py --full-memory`, temperature 0.4, max_tokens 4096, 40960-token
  context, at most 30 environment steps, 8 client workers per server. 4 repetitions over the
  140 seen (`test.jsonl`) and 134 unseen (`test_unseen.jsonl`) tasks. Prompts are this
  repo's ALFWorld template (`trinity/common/workflows/envs/TCOD/alfworld/utils.py`) unless a
  table names another one.
- **Unseen / Seen / Pooled**: success rate in %, mean ± sample standard deviation over the
  repetitions. Pooled = (seen + unseen successes) / 274, computed per repetition.
- **Best rep**: the single repetition with the highest pooled score, with that repetition's
  unseen / seen. It is optimistic by construction (about 1-4 points above the mean).
- **Errored games**: games that raised an error, summed over all repetitions (out of 1096 on
  ALFWorld, 5232 on ScienceWorld, 400 on WebShop). They count as failures.
- **ScienceWorld protocol** (`scripts/tpu/eval/eval_all_sciworld_checkpoints.sh`): avg@4 on all
  1308 test games of TCOD's split. **Success** = % of games with score 100; **mean score** =
  best score / 100, averaged. Mean ± std over the 4 repetitions.
- **WebShop protocol** (`scripts/tpu/eval/eval_all_webshop_checkpoints.sh`): avg@4 on TCOD's
  100 test sessions (4096-4195) on the full product set, human goals, 15 environment steps.
  **Success** = % of sessions with reward 1; **mean score** = mean reward.
- **Template rule** (ScienceWorld and WebShop): every run trained on DASH-OPD's prompts is
  evaluated with them. For a baseline with its own prompt (TCOD, FutureBridge, UOPD), its last
  checkpoint was evaluated with both its own template and DASH-OPD's, and the template with the
  higher success rate was used for its other checkpoints. Both are shown where both were run.
- **Where the raw files are**, on TPU worker 0: `~/TCOD/checkpoints/eval_results/`
  (`avg4_<run>[_<template>]_all.log` per run, per-repetition `eval_<label>.rep<R>.*.jsonl`),
  plus `~/alfworld_ts_probe/data/` (GPU checkpoints, zero-shot models, the first TPU runs).
  Most checkpoints have been deleted to free tmpfs; the remaining ones are on the workers'
  `/dev/shm`.

## Settings shared by all TPU training runs

| Setting | Value |
|---|---|
| Algorithm | `on_policy_distill`, advantage `multi_turn_opd`, `kl_coef` 1.0 (DASH-OPD: `guided_opd`) |
| Loss | PPO-style policy loss, clip range 0.2, `clip_ratio_c` 3.0, token-mean |
| Optimizer | AdamW, lr 1e-6 constant, betas (0.9, 0.999), weight decay 0.01, grad clip 1.0 |
| Batches | train batch 64 turns; rollout batch 16 games per explore step unless noted |
| Sampling | `staleness_control`, `max_staleness` 2 (turns taken in arrival order) |
| Lengths | prompt 10240 tokens, response 512 tokens, `max_env_steps` 30 (WebShop 15) |
| Rollouts | full-memory conversation, temperature 1.0, thinking mode off, seed 42 |
| Hardware (4B / 30B) | Cloud TPU v4-32: 4 VMs x 4 chips. Worker 0 trainer (4 chips, `tunix`), workers 1-2 one teacher each (4 chips), worker 3 four students (1 chip each) |
| Hardware (1.7B / 8B) | 8-chip pairs (one trainer VM + one VM with 2 students and 2 teachers, TP1), two runs at a time; the ScienceWorld 1.7B runs use all 16 chips (trainer VM + 12 TP1 engines on workers 1-3) |
| Weight sync | `checkpoint` method, after every training step (`dynamic_by_explorer`, interval 1) unless noted |
| Training data | ALFWorld `train.jsonl`, 3553 games; ScienceWorld and WebShop: TCOD's train splits |

"Deferred" below means `defer_teacher: true`: the explorer only plays the games and the trainer
asks the teacher for the 64 turns of each training batch (`scripts/tpu/README.md`, section
10.1). "Explorer-side" means the teacher gates and scores every turn during rollout.

**Our method** ("gated t07_s03", sections 9 and 13-19) is the student + teacher
look-ahead gate of section 7 with weights: teacher agrees -> 0; it disagrees and S = the
student's next 5 steps make no progress, T = the teacher's own 5-step continuation (played at
temperature 0 in a replayed copy of the game) makes progress -> S and T 1.0, T only 0.7,
S only 0.3, neither 0.0. A continuation that ends the game counts as progress when it wins
(WebShop `p05`: when the purchase scores at least 0.5); otherwise the teacher is asked.

## ALFWorld reference models (no training here)

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

## ALFWorld TPU runs, Qwen3-4B student / Qwen3-30B-A3B teacher

### 1. Pipeline tests: vanilla OPD, Qwen3-1.7B student / Qwen3-8B teacher

Configs `opd_tpu_repro_1.7b_8b_tpu16.yaml` and `..._tpu16_shmsync.yaml` (same run with sync
checkpoints on tmpfs). Workflow `OPD_alfworld_workflow_fullmemory`, explorer-side, 100 steps.
Checkpoints deleted.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| `tpu16_step100` | n/a | 19.6 ± 2.8 | 19.1 ± 2.8 | 19.3 ± 1.1 | 20.4 (18.7 / 22.1) | 4 | 1 |
| `tpu16_shmsync_step97` | n/a | 22.2 ± 0.7 | 18.9 ± 2.6 | 20.5 ± 1.4 | 21.9 (22.4 / 21.4) | 4 | 1 |

### 2. Look-ahead soft (explorer-side)

Config `opd_gated_lookahead_soft_tpu.yaml`, workflow `OPD_gated_alfworld_workflow_lookahead_soft`.
Progress question on the student's next 5 steps; weight 1.0 if not making progress, 0.5
(`progress_downweight_factor`) if making progress. 250 steps, about 28 h. Checkpoints deleted.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | 5.4 | 38.8 ± 4.6 | 34.6 ± 3.6 | 36.7 ± 4.1 | 42.7 (45.5 / 40.0) | 4 | 3 |
| step 100 | 11.0 | 38.4 ± 5.6 | 38.0 ± 2.0 | 38.2 ± 3.3 | 42.7 (45.5 / 40.0) | 4 | 3 |
| step 150 | 16.6 | 37.3 ± 3.4 | 35.7 ± 1.3 | 36.5 ± 1.7 | 38.3 (39.6 / 37.1) | 4 | 2 |
| step 200 | 22.1 | 38.2 ± 2.1 | 38.9 ± 0.4 | 38.6 ± 1.1 | 39.8 (40.3 / 39.3) | 4 | 0 |
| step 250 | 27.7 | 38.1 ± 1.4 | 34.6 ± 1.4 | 36.3 ± 0.5 | 36.9 (37.3 / 36.4) | 4 | 3 |

### 3. Agree + look-ahead, "sum" (explorer-side)

Config `opd_gated_agree_lookahead_tpu.yaml`, workflow `OPD_gated_alfworld_workflow_agree_lookahead`,
`gate_mode: sum`: weight = 0.5 x [teacher disagrees] + 0.5 x [student's next 5 steps not making
progress]. 250 steps, about 37 h. Checkpoints deleted (copied off the slice by the user).

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | 7.2 | 39.2 ± 1.3 | 34.1 ± 2.3 | 36.6 ± 1.0 | 38.0 (39.6 / 36.4) | 4 | 3 |
| step 100 | 14.8 | 37.9 ± 1.5 | 37.3 ± 1.1 | 37.6 ± 0.9 | 38.7 (39.6 / 37.9) | 4 | 2 |
| step 150 | 22.4 | 39.7 ± 1.9 | 36.6 ± 2.1 | 38.1 ± 1.9 | 39.8 (41.8 / 37.9) | 4 | 1 |
| step 200 | 29.9 | 37.5 ± 0.7 | 37.7 ± 2.4 | 37.6 ± 1.2 | 38.7 (38.1 / 39.3) | 4 | 1 |
| step 250 | 37.4 | 43.3 ± 1.5 | 36.4 ± 3.2 | 39.8 ± 1.5 | 41.6 (42.5 / 40.7) | 4 | 4 |

### 4. Student look-ahead, disagreement required (explorer-side)

Workflow `OPD_gated_alfworld_workflow_agree_lookahead`, `gate_mode: disagree_required`: teacher
agrees -> 0; disagrees and the student's next 5 steps make no progress -> 1.0; disagrees and
they make progress -> 0.5.

**Rollout batch 8** (`opd_gated_disagree_lookahead_tpu.yaml`; stopped at step 50 by a
checkpoint-saving bug and resumed from the exact state). Checkpoints deleted.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | 3.6 | 37.9 ± 2.2 | 33.6 ± 1.5 | 35.7 ± 1.3 | 37.2 (40.3 / 34.3) | 4 | 3 |
| step 100 | 7.6 | 33.4 ± 2.7 | 34.1 ± 1.6 | 33.8 ± 1.9 | 35.4 (35.8 / 35.0) | 4 | 3 |
| step 150 | 11.5 | 36.8 ± 2.8 | 35.0 ± 1.3 | 35.9 ± 1.0 | 36.9 (40.3 / 33.6) | 4 | 3 |
| step 200 | 15.4 | 36.6 ± 0.6 | 38.4 ± 2.3 | 37.5 ± 0.9 | 38.0 (36.6 / 39.3) | 4 | 4 |
| step 250 | 19.4 | 38.1 ± 1.6 | 35.5 ± 2.9 | 36.8 ± 2.1 | 39.1 (38.8 / 39.3) | 4 | 4 |

**Rollout batch 16** (`opd_gated_disagree_lookahead_b16_tpu.yaml`), 250 steps in 33.6 h.
Checkpoints deleted from the slice (copied off by the user).

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | 6.2 | 41.8 ± 3.0 | 35.4 ± 1.7 | 38.5 ± 2.3 | 40.1 (44.0 / 36.4) | 4 | 4 |
| step 100 | 12.9 | 43.3 ± 2.2 | 34.1 ± 2.5 | 38.6 ± 0.5 | 39.1 (41.0 / 37.1) | 4 | 3 |
| step 150 | 19.3 | 47.2 ± 0.7 | 37.5 ± 1.9 | 42.2 ± 1.1 | 43.4 (47.0 / 40.0) | 4 | 5 |
| step 200 | 26.1 | 38.6 ± 3.6 | 30.0 ± 1.3 | 34.2 ± 1.3 | 35.8 (42.5 / 29.3) | 4 | 3 |
| step 250 | 32.7 | 42.7 ± 1.3 | 36.8 ± 1.7 | 39.7 ± 0.9 | 40.9 (44.0 / 37.9) | 4 | 3 |

Data pattern of this run (and of every explorer-side run): each explore batch generated
with model version 2k fed two training steps, 2k+1 (its first 64 turns, 1 version old) and
2k+2 (the next 64 turns, 2 versions old). Turns are written in the order games finish, so
the first 64 are mostly short solved games (54% of turns) and the next 64 mostly failed ones.

### 5. Vanilla OPD, deferred

Config `opd_fullmemory_deferred_tpu.yaml`, workflow `OPD_alfworld_workflow_fullmemory` with
`defer_teacher: true`, free-running explorer. 250 steps in 7 h 23 min, final checkpoint only.
Checkpoint deleted. GPU reference: `opd_fullmem_trained` above.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 250 | 7.1 | 38.1 ± 1.8 | 35.7 ± 2.5 | 36.9 ± 2.0 | 38.0 (39.6 / 36.4) | 4 | 2 |

### 6. Student look-ahead, disagreement required, deferred

Same gate and weights as section 4, rollout batch 16, teacher on the trainer side.

**Plain deferred** (free-running explorer: `dynamic_by_explorer`, interval 1). Trained on a
second TPU v4-32 slice with the same layout, code at commit `50fd022`, config
`opd_gated_disagree_lookahead_deferred_tpu.yaml` as of that commit (copy in
`deferred_plain_run/`, not tracked). Every training step takes the first 64 turns of a new
batch, 2 versions old. Eval labels `disagree_la_deferred_step*`.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | n/a | 41.6 ± 1.8 | 37.0 ± 2.1 | 39.2 ± 0.9 | 40.1 (40.3 / 40.0) | 4 | 4 |
| step 100 | n/a | 36.9 ± 3.3 | 33.4 ± 2.9 | 35.1 ± 2.8 | 38.7 (41.8 / 35.7) | 4 | 2 |
| step 150 | n/a | 38.4 ± 2.2 | 33.9 ± 0.9 | 36.1 ± 0.9 | 36.9 (41.0 / 32.9) | 4 | 3 |
| step 200 | n/a | 41.4 ± 1.6 | 35.2 ± 1.2 | 38.2 ± 1.1 | 39.8 (43.3 / 36.4) | 4 | 3 |
| step 250 | n/a | 43.7 ± 3.3 | 36.1 ± 1.7 | 39.8 ± 1.9 | 42.3 (48.5 / 36.4) | 4 | 4 |

**Fixed sync** (`sync_style: fixed`, `sync_interval: 2`; config
`opd_gated_disagree_lookahead_deferred_fixed2_s150_tpu.yaml`). Reproduces the 1-old / 2-old
pattern of the explorer-side run. Ran to step 150 (8 h 16 min), then continued from the exact
step-150 state to 250. Scores decline after step 100; cause not found. Checkpoints deleted.
Not recommended.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | 2.6 | 40.7 ± 1.0 | 32.0 ± 2.1 | 36.2 ± 1.4 | 37.6 (41.8 / 33.6) | 4 | 3 |
| step 100 | 5.2 | 44.0 ± 2.4 | 34.6 ± 2.1 | 39.2 ± 1.1 | 40.5 (47.0 / 34.3) | 4 | 1 |
| step 150 | 7.8 | 34.1 ± 0.9 | 34.1 ± 2.0 | 34.1 ± 1.3 | 35.8 (35.1 / 36.4) | 4 | 1 |
| step 200 | 10.5 | 32.8 ± 4.2 | 33.8 ± 1.5 | 33.3 ± 2.4 | 36.5 (38.1 / 35.0) | 4 | 3 |
| step 250 | 13.1 | 30.0 ± 3.7 | 31.4 ± 0.6 | 30.7 ± 1.8 | 32.5 (33.6 / 31.4) | 4 | 3 |

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

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | 2.3 | 41.6 ± 3.7 | 37.5 ± 1.7 | 39.5 ± 1.7 | 41.6 (45.5 / 37.9) | 4 | 2 |
| step 100 | 4.4 | 44.4 ± 2.5 | 35.9 ± 3.3 | 40.1 ± 1.2 | 40.9 (47.8 / 34.3) | 4 | 3 |
| step 150 | 6.4 | 39.9 ± 1.0 | 38.6 ± 1.2 | 39.2 ± 1.1 | 40.5 (41.0 / 40.0) | 4 | 2 |
| step 200 | 8.5 | 42.9 ± 3.9 | 38.9 ± 3.3 | 40.9 ± 3.2 | 44.9 (47.8 / 42.1) | 4 | 3 |
| step 241 | 10.2 | 44.6 ± 5.1 | 40.0 ± 2.1 | 42.2 ± 2.7 | 46.0 (52.2 / 40.0) | 4 | 1 |

**Weights 1.0 / 0.5 / 0.2** (`opd_gated_student_teacher_lookahead_w02_deferred_tpu.yaml`),
250 steps in 11 h 16 min. Checkpoints deleted.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---||---|---|---|---|---|---|---|
| step 50 | 2.2 | 39.7 ± 2.6 | 35.5 ± 1.2 | 37.6 ± 1.5 | 39.1 (41.0 / 37.1) | 4 | 4 |
| step 100 | 4.4 | 39.4 ± 2.1 | 36.8 ± 1.4 | 38.0 ± 1.0 | 39.4 (41.0 / 37.9) | 4 | 2 |
| step 150 | 6.6 | 42.2 ± 2.3 | 40.4 ± 2.5 | 41.2 ± 2.4 | 43.8 (44.8 / 42.9) | 4 | 3 |
| step 200 | 8.7 | 36.2 ± 2.2 | 35.9 ± 2.2 | 36.0 ± 2.2 | 37.6 (37.3 / 37.9) | 4 | 1 |
| step 250 | 10.9 | 38.1 ± 1.5 | 36.2 ± 3.7 | 37.1 ± 2.2 | 39.1 (40.3 / 37.9) | 4 | 3 |

### 8. FutureBridge-OPD (FTB)

Config `ftb_qwen3_30b_to_4b_tpu.yaml`, workflow `FutureBridgeAlfworldWorkflow`, copied from
https://github.com/ChenChiShui/FutureBridge-OPD (commit e73603a) with the release's prompt
templates (`ftb_release_utils.py`), curriculum and defaults; explorer-side; 200 steps;
training data `train_expert.jsonl`. Stopped at step 37 when worker 1's disk filled, resumed
from the exact step-37 state (the step-37 checkpoint is the one saved at the stop).

With this repo's template (default):

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 37 | 2.6 | 41.6 ± 0.7 | 35.9 ± 0.7 | 38.7 ± 0.5 | 39.4 (42.5 / 36.4) | 4 | 3 |
| step 50 | 3.7 | 39.8 ± 2.1 | 37.9 ± 1.6 | 38.8 ± 1.0 | 40.1 (42.5 / 37.9) | 4 | 3 |
| step 100 | 8.0 | 39.5 ± 2.1 | 39.5 ± 1.0 | 39.5 ± 0.8 | 40.2 (39.6 / 40.7) | 4 | 0 |
| step 150 | 12.3 | 37.0 ± 0.8 | 40.0 ± 3.8 | 38.5 ± 2.1 | 40.6 (38.1 / 42.9) | 4 | 2 |
| step 200 | 17.0 | 39.0 ± 3.0 | 40.2 ± 4.2 | 39.6 ± 3.0 | 43.8 (41.0 / 46.4) | 4 | 0 |

With the release's `<think>` template (`EVAL_TEMPLATE=release_think`), 1.6-5.1 points lower:

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 37 | 2.6 | 38.8 ± 0.6 | 35.4 ± 1.3 | 37.0 ± 0.6 | 37.9 (38.8 / 37.1) | 4 | 1 |
| step 50 | 3.7 | 34.3 ± 1.6 | 35.9 ± 2.4 | 35.1 ± 0.8 | 35.8 (32.8 / 38.6) | 4 | 2 |
| step 100 | 8.0 | 36.8 ± 1.9 | 38.2 ± 2.1 | 37.5 ± 1.3 | 38.7 (39.6 / 37.9) | 4 | 1 |
| step 150 | 12.3 | 32.5 ± 0.4 | 34.4 ± 1.6 | 33.5 ± 0.8 | 34.3 (32.8 / 35.7) | 4 | 4 |
| step 200 | 17.0 | 35.1 ± 2.7 | 39.1 ± 3.0 | 37.1 ± 0.5 | 37.6 (36.6 / 38.6) | 4 | 3 |

### 9. Our method: student + teacher look-ahead, weights 1.0 / 0.7 / 0.3 / 0.0, deferred

Config `opd_gated_student_teacher_lookahead_t07_s03_deferred_tpu.yaml` (workflow of section 7,
weights in the settings section above), plain deferred, rollout batch 16, 250 steps.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 2.1 | 47.4 ± 3.5 | 36.0 ± 0.7 | 41.6 ± 1.8 | 43.0 (50.7 / 35.7) | 4 | 0 |
| step 100 | 4.1 | 42.4 ± 3.7 | 33.8 ± 1.2 | 38.0 ± 1.3 | 39.8 (47.8 / 32.1) | 4 | 4 |
| step 150 | 6.3 | 41.2 ± 1.2 | 35.7 ± 1.8 | 38.4 ± 1.0 | 39.8 (41.8 / 37.9) | 4 | 2 |
| step 200 | 8.5 | 47.4 ± 2.5 | 40.2 ± 1.7 | 43.7 ± 1.4 | 44.9 (50.0 / 40.0) | 4 | 2 |
| step 250 | 10.6 | 44.6 ± 2.6 | 42.5 ± 2.1 | 43.5 ± 2.2 | 46.4 (47.8 / 45.0) | 4 | 2 |

**Weights 1.0 / 0.6 / 0.4 / 0.0** (`opd_gated_student_teacher_lookahead_t06_s04_deferred_tpu.yaml`).
Step 250's evaluation was merged from two partial runs of the evaluation script.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 2.1 | 35.8 ± 2.7 | 34.7 ± 1.3 | 35.2 ± 1.0 | 36.5 (38.1 / 35.0) | 4 | 5 |
| step 100 | 4.3 | 37.9 ± 4.0 | 34.3 ± 1.6 | 36.0 ± 2.2 | 38.7 (42.5 / 35.0) | 4 | 4 |
| step 150 | 6.6 | 35.5 ± 0.8 | 33.4 ± 1.6 | 34.4 ± 0.8 | 35.4 (35.1 / 35.7) | 4 | 2 |
| step 200 | 8.7 | 39.0 ± 0.7 | 38.2 ± 2.2 | 38.6 ± 1.1 | 39.8 (39.6 / 40.0) | 4 | 0 |
| step 250 | 10.8 | 41.6 ± 3.8 | 36.4 ± 1.2 | 39.0 ± 1.5 | 40.5 (46.3 / 35.0) | 4 | n/a |

### 10. Ablation: disagreement gate only, deferred

Config `opd_gated_disagree_only_deferred_tpu.yaml`, `gate_mode: disagree_only`: weight 1 where
the teacher disagrees, else 0; no look-ahead. 250 steps; last three checkpoints evaluated.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 150 | 5.3 | 34.3 ± 2.5 | 32.3 ± 0.4 | 33.3 ± 1.3 | 34.6 (37.3 / 32.1) | 4 | 0 |
| step 200 | 7.1 | 48.7 ± 2.4 | 38.4 ± 1.2 | 43.4 ± 1.7 | 46.0 (52.2 / 40.0) | 4 | 1 |
| step 250 | 8.9 | 42.0 ± 3.8 | 36.4 ± 1.5 | 39.1 ± 1.1 | 40.2 (46.3 / 34.3) | 4 | 1 |

### 11. TCOD (B2F)

Config `tcod_b2f_qwen3_30b_to_4b_tpu.yaml`, TCOD's B2F workflow (`TCOD_b2f_workflow.py`),
explorer-side, 250 steps, training data `train_expert.jsonl`. Reported with TCOD's own eval
setup (`EVAL_TEMPLATE=tcod_train EVAL_MAX_TOKENS=512`: the with-history prompt from turn 3).

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 2.6 | 37.1 ± 2.5 | 33.9 ± 1.3 | 35.5 ± 1.3 | 37.2 (40.3 / 34.3) | 4 | 2 |
| step 100 | 5.6 | 37.5 ± 2.5 | 33.6 ± 2.0 | 35.5 ± 1.9 | 37.2 (38.1 / 36.4) | 4 | 3 |
| step 150 | 8.6 | 37.7 ± 2.4 | 34.8 ± 0.7 | 36.2 ± 1.4 | 38.3 (41.0 / 35.7) | 4 | 2 |
| step 200 | 11.7 | 37.3 ± 3.2 | 40.9 ± 1.9 | 39.2 ± 1.7 | 40.9 (38.1 / 43.6) | 4 | 2 |
| step 250 | 14.8 | 36.2 ± 2.3 | 35.6 ± 2.7 | 35.9 ± 0.6 | 36.5 (35.1 / 37.9) | 4 | 4 |

With this repo's template: 50 36.8, 100 36.1, 150 35.2, 200 40.8, 250 36.5 pooled.

### 12. UOPD

Config `uopd_qwen3_30b_to_4b_tpu.yaml`, UOPD's ALFWorld workflow copied unchanged
(`trinity/common/workflows/envs/UOPD/`), the release's `<think>` prompt and two-step memory
(one user message with the last 2 steps), student prompt limit 2048 tokens, 250 steps.
Evaluated the same way (`EVAL_TEMPLATE=uopd EVAL_MAX_TOKENS=512`, 30 steps).

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 3.0 | 24.6 ± 3.2 | 26.4 ± 0.6 | 25.5 ± 1.7 | 27.7 (29.1 / 26.4) | 4 | 0 |
| step 100 | 6.0 | 29.9 ± 2.1 | 29.5 ± 1.4 | 29.7 ± 0.4 | 30.3 (32.8 / 27.9) | 4 | 0 |
| step 150 | 8.7 | 28.0 ± 2.3 | 28.0 ± 3.0 | 28.0 ± 2.6 | 31.4 (31.3 / 31.4) | 4 | 0 |
| step 200 | 11.3 | 28.3 ± 2.9 | 28.6 ± 2.0 | 28.5 ± 1.1 | 30.0 (29.9 / 30.0) | 4 | 0 |
| step 250 | 13.9 | 33.6 ± 2.1 | 30.0 ± 1.4 | 31.8 ± 1.5 | 33.6 (35.1 / 32.1) | 4 | 0 |

### ALFWorld pooled scores side by side (4B student, 30B-A3B teacher)

Pooled %, with training hours in brackets.

| Step | Explorer-side, batch 16 (4) | Plain deferred (6) | Fixed sync (6) | S + T look-ahead 1.0/0.5/0.0 (7) | Same, 0.2 (7) | **Ours t07_s03 (9)** | t06_s04 (9) | Disagree only (10) | FTB (8) | TCOD (11) | UOPD (12) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 38.5 [6.2] | 39.2 | 36.2 [2.6] | 39.5 [2.3] | 37.6 [2.2] | **41.6** [2.1] | 35.2 [2.1] | | 38.8 [3.7] | 35.5 [2.6] | 25.5 [3.0] |
| 100 | 38.6 [12.9] | 35.1 | 39.2 [5.2] | 40.1 [4.4] | 38.0 [4.4] | **38.0** [4.1] | 36.0 [4.3] | | 39.5 [8.0] | 35.5 [5.6] | 29.7 [6.0] |
| 150 | 42.2 [19.3] | 36.1 | 34.1 [7.8] | 39.2 [6.4] | 41.2 [6.6] | **38.4** [6.3] | 34.4 [6.6] | 33.3 [5.3] | 38.5 [12.3] | 36.2 [8.6] | 28.0 [8.7] |
| 200 | 34.2 [26.1] | 38.2 | 33.3 [10.5] | 40.9 [8.5] | 36.0 [8.7] | **43.7** [8.5] | 38.6 [8.7] | 43.4 [7.1] | 39.6 [17.0] | 39.2 [11.7] | 28.5 [11.3] |
| 250 | 39.7 [32.7] | 39.8 | 30.7 [13.1] | 42.2 (241) [10.2] | 37.1 [10.9] | **43.5** [10.6] | 39.0 [10.8] | 39.1 [8.9] | | 35.9 [14.8] | 31.8 [13.9] |

References on the same scale: Qwen3-4B zero-shot 34.6, Qwen3-30B-A3B zero-shot 41.0, GPU
vanilla OPD step 250 37.1, GPU agreement gate + reasoning 43.8, TPU vanilla OPD deferred
(5) step 250 36.9 [7.1], DASH-OPD-Qwen3-4B-ALFWorld (external) 41.1.

### 13. ALFWorld, Qwen3-1.7B student / Qwen3-8B teacher (8-chip pairs)

The comparison point is our run's step 200 (11.7 h of training); baseline checkpoints with
more training time than that are listed but not part of the comparison.

References (no training):

| Model | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps |
|---|---|---|---|---|---|
| Qwen3-1.7B (student) | 2.6 ± 1.0 | 3.2 ± 0.9 | 2.9 ± 0.3 | 3.3 (3.7 / 2.9) | 4 |
| Qwen3-8B (teacher) | 42.0 ± 2.1 | 42.9 ± 2.6 | 42.4 ± 1.7 | 44.1 (42.5 / 45.7) | 4 |

**Ours, gated t07_s03** (`opd_gated_qwen3_1.7b_8b_t07_s03_deferred_8chip_tpu.yaml`), 250 steps:

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 3.6 | 12.3 ± 3.8 | 12.5 ± 2.4 | 12.4 ± 2.8 | 15.3 (14.9 / 15.7) | 4 | 0 |
| step 100 | 6.5 | 18.1 ± 2.2 | 16.8 ± 3.2 | 17.4 ± 0.6 | 18.2 (14.9 / 21.4) | 4 | 0 |
| step 150 | 9.0 | 26.3 ± 2.1 | 21.4 ± 0.6 | 23.8 ± 1.2 | 24.8 (28.4 / 21.4) | 4 | 0 |
| step 200 | 11.7 | 31.9 ± 3.0 | 26.6 ± 2.7 | 29.2 ± 2.6 | 31.8 (35.1 / 28.6) | 4 | 0 |
| step 250 | 14.1 | 30.8 ± 3.7 | 26.6 ± 1.6 | 28.7 ± 2.5 | 32.1 (35.8 / 28.6) | 4 | 2 |

With the training prompt (40960 prompt / 4096 response tokens): step 200 28.3 ± 0.8
(31.7 / 25.0), step 250 28.9 ± 1.7 (31.4 / 26.6).

**Vanilla OPD** (`opd_qwen3_1.7b_8b_fullmemory_8chip_tpu.yaml`, `OPD_alfworld_workflow_fullmemory`,
explorer-side; stopped at step 198). An earlier attempt was deleted after NaN teacher log-probs
made its weights NaN; this run has the NaN fix.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 4.7 | 12.3 ± 2.2 | 7.8 ± 2.0 | 10.0 ± 1.2 | 11.0 (14.2 / 7.9) | 4 | 0 |
| step 100 | 9.7 | 14.9 ± 1.9 | 14.8 ± 1.6 | 14.9 ± 1.3 | 16.4 (17.2 / 15.7) | 4 | 0 |
| step 150 | 14.6 | 23.9 ± 3.1 | 21.3 ± 1.5 | 22.5 ± 1.4 | 24.1 (28.4 / 20.0) | 4 | 1 |

**DASH-OPD** (`dash_opd_qwen3_8b_to_1.7b_8chip_tpu.yaml`, the release's workflow, controller,
`guided_opd` advantage and loss; evaluated with DASH's template). Step 160 comes from a
continuation of the step-150 state (`..._from150.yaml`) that saved at step 160.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 150 | 10.5 | 29.7 ± 2.8 | 24.8 ± 2.4 | 27.2 ± 2.2 | 28.8 (32.1 / 25.7) | 4 | 0 |
| step 160 | 11.1 | 30.0 ± 4.2 | 23.9 ± 0.9 | 26.9 ± 2.4 | 29.2 (34.3 / 24.3) | 4 | 0 |
| step 200 | 14.1 | 33.2 ± 2.8 | 27.9 ± 2.1 | 30.5 ± 1.5 | 32.1 (37.3 / 27.1) | 4 | 0 |

**TCOD B2F** (`tcod_b2f_qwen3_8b_to_1.7b_8chip_tpu.yaml`, TCOD's eval setup; crashed at step 111
when worker 3's root disk filled and resumed; trained to step 150):

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 100 | 8.8 | 27.1 ± 2.3 | 22.3 ± 2.0 | 24.6 ± 2.1 | 26.3 (28.4 / 24.3) | 4 | 2 |
| step 150 | 13.4 | 27.6 ± 4.3 | 22.7 ± 1.6 | 25.1 ± 2.9 | 27.7 (31.3 / 24.3) | 4 | 0 |

**FutureBridge** (`ftb_qwen3_8b_to_1.7b_8chip_tpu.yaml`, stopped at step 164). Step 100 plays
`look` in loops with both templates.

| Checkpoint | Train h | Template | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|---|
| step 50 | 5.6 | ours | 6.0 ± 0.6 | 7.8 ± 0.9 | 6.9 ± 0.5 | 7.7 (6.7 / 8.6) | 4 | 0 |
| step 100 | 11.5 | ours | 0.2 ± 0.4 | 2.3 ± 0.4 | 1.3 ± 0.2 | 1.5 (0.0 / 2.9) | 4 | 0 |
| step 150 | 17.1 | ours | 0.7 ± 0.0 | 4.0 ± 1.2 | 2.4 ± 0.6 | 2.9 (0.7 / 5.0) | 4 | 1 |
| step 50 | 5.6 | release `<think>` | 11.7 ± 2.1 | 10.7 ± 1.9 | 11.2 ± 1.6 | 13.1 (14.9 / 11.4) | 4 | 11 |
| step 100 | 11.5 | release `<think>` | 0.0 ± 0.0 | 2.7 ± 1.5 | 1.4 ± 0.8 | 2.2 (0.0 / 4.3) | 4 | 0 |
| step 150 | 17.1 | release `<think>` | 25.2 ± 3.1 | 28.2 ± 1.7 | 26.7 ± 1.8 | 29.2 (29.1 / 29.3) | 4 | 6 |

**UOPD** (`uopd_qwen3_8b_to_1.7b_8chip_tpu.yaml`, release algorithm and 2-step memory, 500 steps,
UOPD's template). An earlier attempt (think-tag collapse) was discarded.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 100 | 3.1 | 9.7 ± 2.5 | 9.8 ± 0.7 | 9.8 ± 1.0 | 10.9 (11.9 / 10.0) | 4 | 0 |
| step 200 | 4.9 | 6.0 ± 0.6 | 8.4 ± 0.3 | 7.2 ± 0.2 | 7.3 (6.0 / 8.6) | 4 | 0 |
| step 300 | 6.8 | 4.5 ± 1.4 | 7.0 ± 1.7 | 5.7 ± 0.4 | 6.2 (3.0 / 9.3) | 4 | 0 |
| step 400 | 8.6 | 5.4 ± 1.7 | 8.2 ± 2.4 | 6.8 ± 1.8 | 8.8 (6.0 / 11.4) | 4 | 0 |
| step 500 | 10.4 | 8.6 ± 0.8 | 9.7 ± 1.3 | 9.1 ± 0.7 | 10.2 (9.0 / 11.4) | 4 | 0 |

At about equal training time (<= 11.7 h): ours 29.2 (step 200, 11.7 h), DASH-OPD 26.9 (step 160,
11.1 h), TCOD about 24.6 (step 100, 8.8 h; no checkpoint near 11.7 h), FTB 1.3 / 1.4 (step 100,
11.5 h), vanilla OPD 14.9 (step 100, 9.7 h), UOPD 9.1 (step 500, 10.4 h).

### 14. ALFWorld, gemma-4-E2B-it student / gemma-4-26B-A4B-it teacher

Deferred, 26B teacher served with `MODEL_IMPL_TYPE=vllm` and the GMM tile override. Both runs
were extended past 250 by exact resume; vanilla was continued to step 850 to match the gated
run's training time at step 350.

| Model | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps |
|---|---|---|---|---|---|
| gemma-4-E2B-it (student) | 7.8 ± 1.8 | 10.6 ± 2.5 | 9.2 ± 1.7 | 11.7 (9.7 / 13.6) | 4 |
| gemma-4-26B-A4B-it (teacher) | 39.2 ± 3.0 | 40.9 ± 1.0 | 40.0 ± 1.7 | 41.6 (41.0 / 42.1) | 4 |

**Ours, gated t07_s03** (`opd_gated_gemma4_e2b_26b_t07_s03_deferred_tpu.yaml`; teacher prompt
limit 32768, gate max_tokens 1024). Crashed at steps 96 and 144 and resumed (at 144 the last
layer's Adam second moment was filled from another run's state; exact from 150 on).

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 3.5 | 2.8 ± 0.7 | 5.9 ± 0.3 | 4.4 ± 0.4 | 4.7 (3.0 / 6.4) | 4 | 0 |
| step 100 | 6.3 | 12.5 ± 1.4 | 9.3 ± 0.6 | 10.9 ± 0.8 | 11.7 (13.4 / 10.0) | 4 | 0 |
| step 150 | 9.0 | 11.3 ± 0.7 | 17.5 ± 0.5 | 14.5 ± 0.5 | 15.0 (11.9 / 17.9) | 4 | 0 |
| step 200 | 11.1 | 12.8 ± 0.7 | 18.1 ± 0.4 | 15.5 ± 0.4 | 15.7 (12.7 / 18.6) | 4 | 0 |
| step 250 | 13.2 | 27.1 ± 1.3 | 28.4 ± 1.8 | 27.8 ± 0.9 | 28.8 (26.9 / 30.7) | 4 | 4 |
| step 300 | 15.3 | 25.4 ± 2.4 | 25.7 ± 0.8 | 25.5 ± 1.3 | 26.6 (26.9 / 26.4) | 4 | 4 |
| step 350 | 17.2 | 34.0 ± 1.8 | 36.8 ± 0.8 | 35.4 ± 1.2 | 36.9 (35.8 / 37.9) | 4 | 4 |

**Vanilla OPD** (`opd_gemma4_e2b_26b_fullmemory_deferred_tpu.yaml`; died at step 215 when worker
1's tmpfs filled, resumed exactly). Step 200 had 33 server errors.

| Checkpoint | Train h | Unseen | Seen | Pooled | Best rep: pooled (unseen / seen) | Reps | Errored games |
|---|---|---|---|---|---|---|---|
| step 50 | 1.2 | 3.5 ± 0.7 | 7.1 ± 1.0 | 5.4 ± 0.4 | 5.9 (3.0 / 8.6) | 4 | 0 |
| step 100 | 2.2 | 4.3 ± 0.4 | 5.2 ± 0.3 | 4.7 ± 0.3 | 5.1 (4.5 / 5.7) | 4 | 4 |
| step 150 | 3.2 | 14.5 ± 1.0 | 13.2 ± 1.5 | 13.9 ± 1.2 | 15.3 (15.7 / 15.0) | 4 | 0 |
| step 200 | 4.3 | 13.6 ± 2.5 | 12.7 ± 1.1 | 13.1 ± 1.3 | 13.9 (14.2 / 13.6) | 4 | 33 |
| step 215 | 4.6 | 19.0 ± 0.9 | 18.2 ± 1.3 | 18.6 ± 0.7 | 19.3 (20.1 / 18.6) | 4 | 0 |
| step 250 | 5.3 | 26.3 ± 3.4 | 20.2 ± 1.8 | 23.2 ± 1.5 | 24.5 (29.1 / 20.0) | 4 | 0 |
| step 300 | 6.3 | 16.4 ± 0.9 | 16.2 ± 1.2 | 16.3 ± 0.6 | 16.8 (15.7 / 17.9) | 4 | 0 |
| step 350 | 7.3 | 17.0 ± 0.4 | 16.8 ± 0.4 | 16.9 ± 0.2 | 17.1 (17.2 / 17.1) | 4 | 0 |
| step 850 | 17.0 | 20.0 ± 1.2 | 17.3 ± 0.7 | 18.6 ± 0.9 | 19.7 (21.6 / 17.9) | 4 | 0 |

## ScienceWorld

### 15. ScienceWorld, Qwen3-4B student / Qwen3-30B-A3B teacher

| Model | Success | Mean score |
|---|---|---|
| Qwen3-4B (student) | 15.1 ± 0.5 | 0.385 ± 0.007 |
| Qwen3-30B-A3B (teacher) | 23.9 ± 0.6 | 0.523 ± 0.008 |
| DASH-OPD release, Qwen3-4B (external) | 21.6 ± 0.8 | 0.500 ± 0.005 |

There is no vanilla OPD row: the run of DASH-OPD's own OPD config was deleted at the user's
request (its score was far above the paper's).

**Ours, gated t07_s03, DASH-OPD's prompts** (`opd_gated_student_teacher_lookahead_t07_s03_dashtpl_deferred_tpu.yaml`):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 150 | 7.7 | 22.6 ± 0.6 | 0.518 ± 0.003 | 0 |
| step 200 | 9.7 | 17.6 ± 0.7 | 0.415 ± 0.004 | 0 |
| step 250 | 11.7 | 26.2 ± 0.8 | 0.532 ± 0.005 | 0 |

**Same, TCOD's prompts** (`opd_gated_student_teacher_lookahead_t07_s03_deferred_tpu.yaml`). The
evaluation hit server errors (counted as failures), so these numbers are low by an unknown amount.

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 150 | 8.2 | 16.8 ± 0.7 | 0.458 ± 0.006 | 177 |
| step 200 | 11.1 | 16.9 ± 1.0 | 0.437 ± 0.012 | 284 |
| step 250 | 13.7 | 18.5 ± 0.8 | 0.466 ± 0.003 | 209 |

**Disagreement gate only** (`opd_gated_disagree_only_dashtpl_deferred_tpu.yaml`):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 150 | 4.9 | 20.3 ± 1.5 | 0.441 ± 0.010 | 0 |
| step 200 | 6.4 | 15.6 ± 0.6 | 0.421 ± 0.006 | 0 |
| step 250 | 8.1 | 14.3 ± 0.6 | 0.366 ± 0.002 | 0 |

**FutureBridge** (`ftb_qwen3_30b_to_4b_tpu.yaml`, 200 steps; DASH's template, which beat the
release template at step 200: 17.5 vs 16.3 / 0.409):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 100 | 4.9 | 19.9 ± 0.3 | 0.460 ± 0.007 | 0 |
| step 150 | 7.5 | 19.9 ± 0.4 | 0.446 ± 0.006 | 0 |
| step 200 | 10.5 | 17.5 ± 1.2 | 0.418 ± 0.007 | 0 |

**TCOD B2F** (`tcod_b2f_qwen3_30b_to_4b_tpu.yaml`, simulator gold paths as expert actions; DASH's
template, which beat TCOD's at step 250: 16.3 vs 16.2 / 0.433):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 100 | 4.2 | 17.0 ± 0.4 | 0.425 ± 0.008 | 0 |
| step 150 | 6.5 | 15.3 ± 0.5 | 0.413 ± 0.003 | 0 |
| step 200 | 8.8 | 19.8 ± 0.9 | 0.469 ± 0.006 | 0 |
| step 250 | 11.2 | 16.3 ± 0.8 | 0.415 ± 0.007 | 0 |

**UOPD** (`uopd_qwen3_30b_to_4b_tpu.yaml`: UOPD's ALFWorld algorithm with TCOD's ScienceWorld
prompt and UOPD's 2-step memory; UOPD's template, which beat DASH's at step 250: 17.5 vs 16.1 / 0.405):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 150 | 7.5 | 17.4 ± 1.0 | 0.503 ± 0.011 | 0 |
| step 200 | 9.7 | 17.9 ± 0.3 | 0.510 ± 0.005 | 0 |
| step 250 | 11.9 | 17.5 ± 0.6 | 0.508 ± 0.009 | 0 |

### 16. ScienceWorld, Qwen3-1.7B student / Qwen3-8B teacher (16 chips)

DASH-OPD's prompts, `prompt_truncation_keep_recent: true` for students and teachers (an over-long
prompt keeps its last 10240 tokens and still generates instead of returning an empty masked
turn; an earlier 8-chip attempt without it was stopped at step 70). References, one repetition each: Qwen3-1.7B
1.7% / 0.108, Qwen3-8B 25.3% / 0.510.

**Ours, gated t07_s03** (`qwen3_1.7b_8b_opd_gated_t07_s03_dashtpl_keeprecent_deferred_16chip_tpu.yaml`;
2 students + 10 teachers; median step 2.68 min):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 150 | 7.5 | 6.5 ± 0.3 | 0.217 ± 0.006 | 0 |
| step 200 | 9.7 | 7.3 ± 0.5 | 0.257 ± 0.009 | 0 |
| step 250 | 12.0 | 10.0 ± 1.0 | 0.307 ± 0.006 | 0 |
| step 300 | 14.0 | 10.1 ± 0.5 | 0.304 ± 0.003 | 0 |
| step 350 | 16.0 | 9.1 ± 0.9 | 0.296 ± 0.009 | 0 |

**DASH-OPD** (`dash_opd_qwen3_8b_to_1.7b_16chip_tpu.yaml`, following the release's
`configurations/scienceworld/dash_opd.yaml`; 6 students + 6 teachers; median step 2.87 min).

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 250 | 11.8 | 10.1 ± 0.7 | 0.344 ± 0.007 | 0 |
| step 300 | 14.1 | 11.6 ± 1.1 | 0.391 ± 0.007 | 0 |
| step 350 | 16.5 | 9.8 ± 0.8 | 0.315 ± 0.005 | 0 |

Steps 300 and 350 of both runs come from an exact resume of the step-250 state with
`total_steps: 350` (`..._resume350.yaml`, save every 50); DASH's teacher-start schedule keeps
`workflow_args.total_steps: 250`. Checkpoints before step 250 were deleted unevaluated for
DASH-OPD (150, 200).

## WebShop

### 17. WebShop, Qwen3-4B student / Qwen3-30B-A3B teacher

DASH-OPD's prompts for training and evaluation, full memory, 15 environment steps, 150 steps
with a checkpoint every 25 (baselines: every 50); task and experience shuffle as in DASH-OPD
(seed 42).

| Model | Success | Mean score |
|---|---|---|
| Qwen3-4B (student) | 10.5 ± 1.7 | 0.280 ± 0.020 |
| Qwen3-30B-A3B (teacher) | 16.0 ± 0.0 | 0.401 ± 0.022 |
| DASH-OPD release, Qwen3-4B (external) | 14.5 ± 1.3 | 0.415 ± 0.007 |

**Ours, gated t07_s03, `progress_score: 0.5`** (`opd_gated_student_teacher_lookahead_t07_s03_dashtpl_p05_deferred_tpu.yaml`):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 25 | 1.3 | 13.5 ± 2.5 | 0.337 ± 0.026 | 0 |
| step 50 | 2.4 | 22.2 ± 1.7 | 0.548 ± 0.022 | 0 |
| step 75 | 3.1 | 20.2 ± 1.5 | 0.507 ± 0.012 | 0 |
| step 100 | 4.2 | 16.2 ± 1.5 | 0.431 ± 0.009 | 0 |
| step 125 | 5.3 | 17.8 ± 1.5 | 0.382 ± 0.011 | 0 |
| step 150 | 6.5 | 16.5 ± 2.1 | 0.417 ± 0.013 | 0 |

**Same, progress only on a score-1 purchase** (`..._t07_s03_dashtpl_deferred_tpu.yaml`):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 25 | 1.3 | 14.8 ± 2.5 | 0.378 ± 0.039 | 0 |
| step 50 | 2.6 | 12.5 ± 1.3 | 0.307 ± 0.023 | 0 |
| step 75 | 3.9 | 11.8 ± 1.0 | 0.359 ± 0.008 | 0 |
| step 100 | 5.2 | 12.5 ± 1.3 | 0.353 ± 0.026 | 0 |
| step 125 | 6.6 | 13.0 ± 1.2 | 0.353 ± 0.017 | 0 |
| step 150 | 8.0 | 9.8 ± 1.7 | 0.294 ± 0.020 | 0 |

**Vanilla OPD** (`opd_dashtpl_deferred_tpu.yaml`, the gated run's setup with every turn weight 1):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 25 | 0.7 | 11.8 ± 1.3 | 0.344 ± 0.020 | 0 |
| step 50 | 1.2 | 15.0 ± 1.4 | 0.411 ± 0.023 | 0 |
| step 75 | 1.8 | 15.5 ± 1.7 | 0.397 ± 0.020 | 0 |
| step 100 | 2.4 | 14.3 ± 1.3 | 0.356 ± 0.023 | 0 |
| step 125 | 3.0 | 18.0 ± 0.8 | 0.426 ± 0.006 | 0 |
| step 150 | 3.5 | 14.5 ± 2.4 | 0.377 ± 0.020 | 0 |

**Disagreement gate only** (`opd_gated_disagree_only_dashtpl_deferred_tpu.yaml`):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 50 | 1.3 | 17.8 ± 3.6 | 0.454 ± 0.036 | 0 |
| step 100 | 2.7 | 14.3 ± 1.0 | 0.411 ± 0.016 | 0 |
| step 150 | 4.1 | 12.8 ± 2.1 | 0.377 ± 0.007 | 0 |

Baselines, each on our environment (full catalogue, human goals, TCOD's sessions) with its own
release algorithm, prompt and memory; evaluated with its own template, which won at its last
checkpoint (the other template's score at that checkpoint in the last column).

| Run (config) | Checkpoint | Train h | Success | Mean score | Other template at last ckpt |
|---|---|---|---|---|---|
| UOPD, 2-step (`uopd_qwen3_30b_to_4b_tpu.yaml`) | step 50 | 1.6 | 6.0 ± 1.4 | 0.215 ± 0.012 | |
| | step 100 | 3.0 | 8.0 ± 1.4 | 0.240 ± 0.030 | |
| | step 150 | 4.5 | 7.0 ± 2.7 | 0.234 ± 0.030 | DASH 5.0 / 0.133 |
| UOPD, full memory (`uopd_fullmemory_qwen3_30b_to_4b_tpu.yaml`)* | step 50 | 2.1 | 8.5 ± 1.0 | 0.203 ± 0.019 | |
| | step 100 | 3.9 | 7.0 ± 1.4 | 0.187 ± 0.012 | |
| | step 150 | 5.6 | 9.2 ± 2.6 | 0.214 ± 0.034 | own 8.5 / 0.256 |
| FutureBridge (`ftb_qwen3_30b_to_4b_tpu.yaml`) | step 50 | 2.5 | 11.8 ± 1.0 | 0.347 ± 0.011 | |
| | step 100 | 4.9 | 13.2 ± 0.5 | 0.338 ± 0.009 | |
| | step 150 | 7.5 | 11.0 ± 1.4 | 0.288 ± 0.034 | |
| | step 200 | 10.0 | 14.5 ± 1.7 | 0.332 ± 0.014 | DASH 12.0 / 0.361 |
| TCOD B2F, teacher experts (`tcod_b2f_qwen3_30b_to_4b_tpu.yaml`) | step 50 | 1.6 | 12.5 ± 2.4 | 0.355 ± 0.027 | |
| | step 100 | 3.2 | 14.0 ± 1.4 | 0.357 ± 0.002 | |
| | step 150 | 4.9 | 14.8 ± 1.3 | 0.353 ± 0.027 | DASH 13.0 / 0.353 |

\* evaluated with DASH's template (it won at step 150).

### 18. WebShop, gemma-4-E2B-it student / Gemma 4 teachers

Same WebShop setup as section 17. The run directories were deleted, so there are no
per-checkpoint training times; median steps were about 6.5 min (31B), 3.4 min (26B gated) and
82 s (26B vanilla).

| Model | Success | Mean score |
|---|---|---|
| gemma-4-E2B-it (student) | 14.8 ± 1.0 | 0.439 ± 0.022 |
| gemma-4-26B-A4B-it (teacher) | 20.0 ± 1.2 | 0.427 ± 0.013 |
| gemma-4-31B-it (teacher) | 19.8 ± 2.2 | 0.502 ± 0.020 |

Success % (mean score) per checkpoint:

| Run (config) | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|
| Ours p05, 31B teacher (`opd_gated_gemma4_e2b_31b_p05_deferred_tpu.yaml`) | 15.0 (0.431) | 18.8 (0.442) | 15.8 (0.354) | 14.5 (0.332) | 13.0 (0.327) | 13.5 (0.329) |
| Ours p05, 26B teacher (`opd_gated_gemma4_e2b_26b_p05_deferred_tpu.yaml`) | 14.3 (0.395) | 12.8 (0.331) | 13.5 (0.314) | 10.2 (0.238) | 7.8 (0.149) | 2.2 (0.048) |
| Same, teacher template fix (`..._26b_p05_tmplfix_deferred_tpu.yaml`) | 14.5 (0.401) | 11.2 (0.311) | 10.8 (0.282) | 10.8 (0.281) | 14.5 (0.325) | 13.2 (0.305) |
| Vanilla OPD, 26B teacher (`opd_gemma4_e2b_26b_dashtpl_deferred_tpu.yaml`) | 14.2 (0.407) | 14.8 (0.341) | 12.5 (0.296) | 13.0 (0.308) | 13.0 (0.321) | 12.8 (0.278) |

The 26B gated run without the template fix scored the teacher in thinking mode (the student's
prompt has no empty thought block); the fix inserts the teacher's marker.

### 19. WebShop, Qwen3-1.7B student / Qwen3-8B teacher (8-chip pairs)

References: Qwen3-1.7B 2.2 ± 1.0 / 0.037 ± 0.012, Qwen3-8B 17.0 ± 1.6 / 0.421 ± 0.014.

**Ours, gated t07_s03 p05** (`qwen3_1.7b_8b_opd_gated_t07_s03_dashtpl_p05_deferred_8chip_tpu.yaml`, 100 steps):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 25 | 2.1 | 14.3 ± 1.0 | 0.444 ± 0.020 | 0 |
| step 50 | 2.9 | 14.3 ± 1.0 | 0.517 ± 0.005 | 0 |
| step 75 | 4.2 | 16.0 ± 0.8 | 0.464 ± 0.016 | 0 |
| step 100 | 5.8 | 21.0 ± 1.8 | 0.543 ± 0.011 | 0 |

**Vanilla OPD** (`opd_dashtpl_qwen3_1.7b_8b_8chip_tpu.yaml`, 250 steps; step 230 from a
continuation of step 200, `..._from200.yaml`). Only step 200 is kept.

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 25 | 0.6 | 9.2 ± 2.5 | 0.376 ± 0.021 | 0 |
| step 50 | 1.0 | 8.5 ± 1.7 | 0.422 ± 0.009 | 0 |
| step 75 | 1.6 | 8.0 ± 0.8 | 0.369 ± 0.031 | 0 |
| step 100 | 2.3 | 10.8 ± 2.4 | 0.382 ± 0.023 | 0 |
| step 150 | 3.6 | 9.5 ± 1.9 | 0.372 ± 0.013 | 0 |
| step 200 | 5.0 | 10.5 ± 1.3 | 0.392 ± 0.005 | 0 |
| step 230 | 5.7 | 8.0 ± 0.8 | 0.350 ± 0.017 | 0 |
| step 250 | 6.3 | 9.0 ± 0.8 | 0.335 ± 0.016 | 0 |

**DASH-OPD** (`dash_opd_qwen3_8b_to_1.7b_8chip_tpu.yaml`, 250 steps):

| Checkpoint | Train h | Success | Mean score | Errored games |
|---|---|---|---|---|
| step 25 | 0.6 | 9.5 ± 2.6 | 0.374 ± 0.010 | 0 |
| step 50 | 1.2 | 8.5 ± 0.6 | 0.374 ± 0.014 | 0 |
| step 75 | 1.9 | 8.2 ± 3.0 | 0.343 ± 0.030 | 0 |
| step 100 | 2.6 | 6.0 ± 1.4 | 0.306 ± 0.032 | 0 |
| step 150 | 4.1 | 9.0 ± 2.8 | 0.307 ± 0.037 | 0 |
| step 200 | 5.6 | 11.0 ± 0.8 | 0.345 ± 0.023 | 0 |
| step 250 | 7.1 | 10.5 ± 2.6 | 0.314 ± 0.032 | 0 |

**Other baselines** (template choice as in section 17; the other template at the reference
checkpoint in the last column):

| Run (config) | Checkpoint | Train h | Success | Mean score | Other template |
|---|---|---|---|---|---|
| FutureBridge (`ftb_qwen3_8b_to_1.7b_8chip_tpu.yaml`, stopped at 140), DASH template | step 25 | 1.3 | 10.8 ± 3.9 | 0.348 ± 0.033 | |
| | step 50 | 2.2 | 13.2 ± 0.5 | 0.482 ± 0.003 | |
| | step 75 | 3.3 | 13.2 ± 2.5 | 0.464 ± 0.013 | |
| | step 100 | 4.4 | 9.8 ± 1.3 | 0.486 ± 0.021 | FTB 9.8 / 0.487 |
| | step 140 | 5.5 | 15.2 ± 1.0 | 0.460 ± 0.018 | FTB 15.0 / 0.439 |
| TCOD B2F (`tcod_b2f_qwen3_8b_to_1.7b_8chip_tpu.yaml`), DASH template | step 25 | 0.6 | 8.0 ± 0.8 | 0.282 ± 0.047 | |
| | step 50 | 2.0 | 9.5 ± 1.7 | 0.336 ± 0.036 | |
| | step 75 | 3.4 | 13.8 ± 1.9 | 0.327 ± 0.035 | |
| | step 100 | 4.2 | 0.0 ± 0.0 | 0.005 ± 0.003 | TCOD 0.0 / 0.009 |
| | step 150 | 6.5 | 7.5 ± 1.9 | 0.181 ± 0.031 | TCOD 8.2 / 0.149 |
| UOPD, 2-step (`uopd_qwen3_8b_to_1.7b_8chip_tpu.yaml`, 100 steps), DASH template | step 25 | 0.4 | 7.8 ± 1.9 | 0.265 ± 0.026 | |
| | step 50 | 0.7 | 12.0 ± 1.4 | 0.413 ± 0.022 | |
| | step 75 | 1.0 | 11.2 ± 1.3 | 0.474 ± 0.008 | |
| | step 100 | 1.3 | 14.5 ± 0.6 | 0.502 ± 0.011 | UOPD 0.0 / 0.000 |
| UOPD, full memory (`uopd_fullmemory_qwen3_8b_to_1.7b_8chip_tpu.yaml`, 250 steps), DASH template | step 200 | 4.9 | 16.2 ± 0.5 | 0.506 ± 0.011 | own 13.5 / 0.486 |
| | step 250 | 5.7 | 16.0 ± 0.8 | 0.530 ± 0.014 | own 14.5 / 0.442 |
| Same, own template | step 100 | 2.9 | 9.2 ± 1.0 | 0.490 ± 0.014 | |
| | step 150 | 3.9 | 10.2 ± 1.0 | 0.484 ± 0.011 | |
