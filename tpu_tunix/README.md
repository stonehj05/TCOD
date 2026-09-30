# TCOD on TPU (JAX / Tunix)

A TPU port of the multi-turn on-policy distillation (OPD) pipeline for ALFWorld.
It replaces the CUDA-only parts of the stack (verl FSDP trainer, vLLM, NCCL) with
[Tunix](https://github.com/google/tunix) on JAX. The ALFWorld prompt templates and
env helpers are imported from `trinity/common/workflows/envs/TCOD/alfworld/utils.py`,
so prompts match the GPU pipeline exactly.

Status: **OPD only**, single host (4 chips). TCOD-b2f / f2b and multi-host are not
wired up yet (`AlfworldEpisode` already takes `expert_actions/start_step` and
`max_turns` for them).

## Mapping to the Trinity/verl implementation

| Trinity / verl (GPU) | Here (TPU) |
|---|---|
| `OPD_alfworld_workflow`, one Experience per turn | `OPDTrainer.run_episodes`: all episodes stepped in lockstep, one batched generate per turn, one training row per turn |
| student vLLM rollout | Tunix vanilla sampler, sharing the trainer's weights (no weight sync) |
| teacher `logprobs_async` (vLLM auxiliary model) | teacher loaded as Tunix's REFERENCE model, prefill-only `get_ref_per_token_logps` |
| `multi_turn_opd` advantage: `kl_coef * (teacher_lp - student_lp)` | same, per token (`OPDTrainer.update`) |
| PPO loss (dual clip, `token-mean`) | Tunix `grpo_loss_fn` with 2D advantages, `beta=0` |
| verl FSDP, fp32 master weights | NNX model sharded on an `(fsdp, tp)` mesh, student in fp32 |

Differences worth knowing:
- **Batching:** a step plays `batch_size` episodes and then uses every collected turn
  once, one optimizer update per `mini_batch_size` turns (the last mini-batch is
  filled by re-sampling real turns). Trinity instead streams turns from an async
  explorer with staleness control. Step counts are not directly comparable.
- **Stop token:** the sampler drops the stop token, so `<|im_end|>` is re-appended to
  every response that stopped, so it still gets a distillation signal.
- **Prompt overflow:** a turn whose prompt exceeds `max_prompt_length` ends the
  episode (`rollout/prompt_truncated`) instead of being truncated.
- The student and teacher must share a tokenizer (checked at startup).

## Setup

```bash
python3.11 -m venv ~/venv-tunix
~/venv-tunix/bin/pip install -r requirements.txt
~/venv-tunix/bin/alfworld-download --data-dir ~/alf-data
```

The task lists in `TCOD_examples/alfworld/alfworld_data/*.jsonl` point to game files on
another machine; `game_path_prefix_from/to` in the config remaps them.

## Run

```bash
cd tpu_tunix
source env.sh           # single-host mode: use this VM's 4 chips only
~/venv-tunix/bin/python train_opd.py --config configs/smoke.yaml            # ~5 min check
~/venv-tunix/bin/python tests/check_kl_decreases.py --config configs/smoke.yaml lr=1e-5
~/venv-tunix/bin/python train_opd.py --config configs/alfworld_opd.yaml teacher=/path/to/teacher
```

Any config field can be overridden as `key=value`. Metrics are written to
`<log_dir>/metrics.jsonl` (rollout success rate, KL, timings) and Tunix's trainer
metrics to TensorBoard under `<log_dir>/tunix`.

## Files

- `opd.py`: `OPDConfig` and `OPDTrainer` (rollout, teacher scoring, PPO update, eval)
- `alfworld_env.py`: `AlfworldEpisode`, a turn-by-turn wrapper using TCOD's templates
- `models.py`: loads HF checkpoints (hub id or local dir) as Tunix models
- `train_opd.py`: CLI entry point
- `tests/check_kl_decreases.py`: repeated updates on one batch must shrink KL(student‖teacher)
