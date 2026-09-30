# Trinity-RFT OPD Pipeline: Launch-to-Gradient Call Flow

Reference for reimplementing this training pipeline (or an equivalent one) on a
TPU machine. Traced against `trinity run --config TCOD_examples/alfworld/opd.yaml`
(GPU/Ray/vLLM/verl-FSDP stack); correctness target for the port is
`TCOD_examples/alfworld/opd_tpu_repro_1.7b_8b.yaml` (Qwen3-1.7B student /
Qwen3-8B teacher, `OPD_alfworld_workflow_fullmemory`, 100 steps, tensorboard
logging — see that file's header comment for why those specific choices).

Every code reference below is `path/from/TCOD/root.py:line`.

---

## 0. Two processes, three concurrency layers

`trinity run` starts exactly two long-lived Ray actors that matter for
training: **Explorer** (generates rollouts) and **Trainer** (consumes them,
updates weights). They run concurrently and only meet at two points: the
experience buffer (Explorer writes, Trainer reads) and the weight
synchronizer (Trainer writes updated weights, Explorer pulls them).

Inside the Explorer there are two more layers of concurrency nested inside
each other:

- **Scheduler**: fans a batch of tasks out across a pool of `WorkflowRunner`
  actors (`runner_num = engine_num * runner_per_model` of them), one task per
  idle runner.
- **WorkflowRunner**: if a single task has `repeat_times > 1`, further
  parallelizes those repeats with `asyncio.gather` inside one runner.

```
                         ┌────────────────────────────────────────────┐
                         │                 Explorer                    │
                         │  ┌────────────────────────────────────┐   │
 trinity run ──▶ both() ─┼─▶│ Scheduler → N × WorkflowRunner actor │   │
                         │  │   → Task.to_workflow() → run_async() │   │
                         │  └──────────────┬───────────────────────┘   │
                         │      writes exps│(ExperiencePipeline)        │
                         └──────────────────┼───────────────────────────┘
                                            ▼
                              SQLite experience_buffer (queue)
                                            │
                         ┌──────────────────┼───────────────────────────┐
                         │                 Trainer                       │
                         │   sample_strategy.sample() reads it ◀─────────┘
                         │   → to_data_proto → advantage_fn → update_actor │
                         └──────────────────┬───────────────────────────┘
                                            │ NCCL broadcast (Synchronizer rendezvous)
                                            ▼
                                  Explorer's vLLM engines
```

---

## Phase A — Launch (`trinity/cli/launcher.py`)

1. `main()` (`launcher.py:314`) parses `run --config opd.yaml` → `run(config_path, dlc, plugin_dir)`.
2. `run()` (`launcher.py:167`) — `load_config()` parses YAML into `Config`; `config.check_and_update()` fills/validates defaults (GPU allocation split between Explorer and Trainer — see `config_validator.py`'s `_set_gpu_allocation_info`); calls `run_stage(config)`.
3. `run_stage()` (`launcher.py:147`) — `ray.init(...)`, then `MODE_MAP[config.mode](config)`. Default `mode` is `"both"` → `both(config)` (`launcher.py:76`).
4. `both()` — creates both actors, `.prepare.remote()` on each, `.sync_weight.remote()` on each (initial weight load), then fires `explorer.explore.remote()` and `trainer.train.remote()` **concurrently** and waits on whichever finishes first.

**TPU porting note:** this whole layer is orchestration-only (Ray actor lifecycle, argparse, YAML config) — nothing here is GPU-specific. Reusable as-is; a TPU port only needs `Config` fields for whatever TPU-specific device/mesh settings replace `gpu_per_node`/`tensor_parallel_size`.

---

## Phase B — Explorer setup (`trinity/explorer/explorer.py`)

5. `Explorer.__init__()` (`explorer.py:41`) — `create_inference_models(config)` stands up the **vLLM** engines for student (`self.models`) and teacher (`self.auxiliary_models`); `get_taskset_scheduler(...)` opens the training taskset (JSONL file reader).
6. `Explorer.prepare()` (`explorer.py:378`) — after models ready, `self.scheduler = Scheduler(...)`, `await self.scheduler.start()`.
7. `Scheduler.start()` (`explorer/scheduler.py:437`) — spawns `runner_num` `RunnerWrapper`s, each wrapping a Ray-actor `WorkflowRunner` (`scheduler.py:147`), and starts the background `_scheduler_loop()` (polls every 10ms).

**TPU porting note — this is the single biggest rewrite surface.** `create_inference_models` → vLLM is the GPU-specific rollout-serving engine. A TPU port needs an equivalent async chat/logprobs-serving interface (e.g. a JAX/PyTorch-XLA serving stack, or a TPU-compatible vLLM backend if available at build time) that implements the same two methods the rest of the pipeline depends on: `chat_async(messages, **rollout_args) -> List[Experience]` (with `.logprobs`, `.tokens`, `.prompt_length` populated) and `logprobs_async(tokens, temperature) -> Tensor` (`common/models/model.py:409-417`). Everything downstream of this interface (Scheduler, WorkflowRunner, the workflow classes themselves) is backend-agnostic as long as that contract is preserved — this is the layer to design first.

---

## Phase C — The explore loop: reading tasks, building `Task` objects

8. `Explorer.explore()` (`explorer.py:418`) — outer loop: `explore_step()` → `need_eval()`/`eval()` → `need_sync()`/`sync_weight()`.
9. `Explorer.explore_step()` (`explorer.py:449`) — `tasks = await self.taskset.read_async()`.
10. Resolves to `TaskFileReader` (`buffer/reader/file_reader.py:184`) → `TaskFormatter` (`buffer/schema/formatter.py:22`), constructed with `self.default_workflow_cls = WORKFLOWS.get(config.default_workflow_type)`.
11. `WORKFLOWS` (`common/workflows/__init__.py`) is a `Registry` (`utils/registry.py:38`) that dynamically imports the workflow class string from config (e.g. `OPD_alfworld_workflow_fullmemory` → `trinity.common.workflows.envs.TCOD.alfworld.OPD_workflow_fullmemory.OnPolicyDistillVerlAgentAlfworldWorkflowFullMemory`) — resolves the **class**, not yet instantiated.
12. `TaskFormatter.format(sample)` (`formatter.py:42`) — per JSONL row, builds `Task(workflow=<resolved class>, raw_task=sample, workflow_args={'max_env_steps': 30, ...}, ...)`.

**TPU porting note:** fully backend-agnostic (file I/O, dynamic class resolution, dataclass construction). Reuse as-is.

---

## Phase D — Scheduling tasks onto runners

13. `Explorer.explore_step()`: `self.scheduler.schedule(tasks, batch_id=step)` (`explorer.py:467`) — **non-blocking**.
14. `Scheduler.schedule()` → `_split_and_submit_tasks()` (`scheduler.py:475`, `488`) — wraps each `Task` in a `TaskWrapper`, pushes onto `pending_tasks[batch_id]`. If `max_repeat_times_per_runner` is set, one logical task's `repeat_times` gets split into multiple sub-task chunks distributed across different runners (reassembled later by `sub_task_num` bookkeeping).
15. Background `_scheduler_loop()` → `_schedule_pending_tasks()` (`scheduler.py:361`) — pops a task, grabs an idle runner, fires `asyncio.create_task(RunnerWrapper.run_with_retry(...))`.
16. `RunnerWrapper.run_with_retry()` (`scheduler.py:169`) — timeout/retry wrapper around the actual Ray RPC: `await self.runner.run_task.remote(task=..., repeat_times=..., run_id_base=...)`.

**TPU porting note:** fully backend-agnostic (pure asyncio task-queue logic, no tensor ops). Reuse as-is. The one config knob to re-tune for TPU is `runner_per_model` × `engine_num` — sized for GPU vLLM-replica memory budgets; a TPU serving setup will have different concurrency economics (TPU chips are typically fewer, larger, and pooled differently than discrete GPUs).

---

## Phase E — Inside `WorkflowRunner`: instantiating and running the workflow

17. `WorkflowRunner.run_task()` (`explorer/workflow_runner.py:267`) — entry point on the actor side.
18. `_run_task()` (`workflow_runner.py:133`) — branches on `task.workflow.can_repeat`. OPD workflows are `can_repeat = False` → routes to `concurrent_run_fn` (default `"asynchronous"` → `_asynchronous_run`, `workflow_runner.py:183`).
19. Inside `_asynchronous_run`'s `run_single(i)`: **`workflow = task.to_workflow(self.model_wrapper, self.auxiliary_model_wrappers)`**.
20. `Task.to_workflow()` (`common/workflows/workflow.py:40`) — `return self.workflow(model=model, task=self, auxiliary_models=auxiliary_models)`. **This is where the class resolved in step 11 is finally instantiated** — the workflow's `__init__` runs here.
21. `_run_workflow()` (`workflow_runner.py:126`) — checks `workflow_instance.asynchronous` → `await workflow_instance.run_async()`. **This is where your environment/episode logic actually executes** — for AlfWorld, `_run_episode()`: student samples turn-by-turn with logprobs, teacher rescoring pass, returns `List[Experience]`.
22. Back in `run_task()`: each returned `Experience` gets stamped with `eid.batch/task`, `info["model_version"/"use_count"/"task_index"]` (`workflow_runner.py:282-291`).

**TPU porting note:** entirely backend-agnostic *given* the model interface from Phase B — the `Workflow` base class, the env-driving loop, and the `Experience` dataclass have no GPU/vLLM assumptions baked in beyond calling `self.model.chat_async(...)` and `self.teacher_model.logprobs_async(...)`. Reuse the workflow classes (`OPD_workflow_fullmemory.py` etc.) unmodified once Phase B's serving interface exists.

---

## Phase F — Experience handoff: Explorer → SQLite buffer → Trainer

23. `Explorer._finish_explore_step()` (`explorer.py:621`) — only reached when `sync_weight()` fires (gated by `synchronizer.sync_interval`), NOT every `explore_step()`. This is where rollout generation genuinely runs ahead of result-collection.
24. `scheduler.get_results(batch_id=step)` (`scheduler.py:519`) — blocks (with over-rollout straggler tolerance via `min_wait_num`) until that batch's episodes are done, drains `completed_tasks[batch_id]`.
25. `self.experience_pipeline.process.remote(exps)` (`buffer/pipelines/experience_pipeline.py:112`) — runs configured `operators` (empty list for `opd.yaml`/`opd_tpu_repro_1.7b_8b.yaml` — pure pass-through for this algorithm, since `OnPolicyDistillAlgorithm.compute_advantage_in_trainer = True` means no operator gets auto-injected here), then `await self.output.write_async(exps)` — writes into `sqlite:///alfworld_opd_tpu_repro_1.7b_8b_buffer.db`.
26. On the Trainer side: `StalenessControlSampleStrategy.sample()` (`algorithm/sample_strategy/sample_strategy.py:82`) — `exp_list = await self.exp_buffer.read_async(min_model_version=max(step - max_staleness, 0))`. This is the actual **on-policy-ness enforcement point**: experiences tagged with a stale `info["model_version"]` get filtered out.

**TPU porting note:** fully backend-agnostic — SQLite queue, dataclass serialization, no tensor compute. Reuse as-is. `max_staleness` (2 in both configs) is a knob worth keeping identical between the GPU reference run and the TPU port for a fair correctness comparison — it directly affects how much rollout/train weight-drift is tolerated.

---

## Phase G — Trainer: sampling and the gradient step

27. `Trainer.train()` (`trainer/trainer.py:73`) — outer loop: fires `_sample_data()` as a background task, polls `need_sync()` every second *while waiting* for it (so a weight sync can happen even if the buffer doesn't have enough data yet — avoids an Explorer/Trainer deadlock under NCCL sync), then `train_step(exps)`.
28. `train_step()` (`trainer.py:111`) → `self.engine.train_step(exps)` — `self.engine` is `VerlPPOTrainerWrapper` (`trainer/verl_trainer.py:184`), chosen via `get_trainer_wrapper()` (`trainer.py:269`) from `config.trainer.trainer_type` (`"verl"` for both reference configs; `"tinker"` is the other built-in option — worth looking at `trinity/trainer/tinker_trainer.py` as a second existing example of a *different* backend already plugged into this same `TrainEngineWrapper` interface).

### `VerlPPOTrainerWrapper.train_step()` body (`verl_trainer.py:451-587`)

29. `batch = to_data_proto(batch_exps, pad_token_id, logger)` (`trainer/verl/utils.py:24`) — `Experience` list → `DataProto` tensors. **This is the field-name contract the rest of the trainer depends on**: `input_ids`/`responses`/`attention_mask`/`response_mask` from tokens+prompt_length+action_mask, plus (critical for OPD) `batch_dict["teacher_logprobs"] = gather_response_attrs(experiences, "teacher_logprobs", ...)` (`utils.py:82-84`) — this is literally `Experience.teacher_logprobs`, set by the workflow in step 21, now a batched tensor.
30. `_balance_batch()` (inherited from verl `RayPPOTrainer`, `ray_trainer.py:1106`) — reorders rows (index-only, no value change) so each data-parallel rank gets a similar *total token count*, not just row count.
31. Old-log-prob branch (`verl_trainer.py:486-508`, taken since neither config sets `algorithm.rollout_correction`): `old_log_prob, _ = self._compute_old_log_prob(batch)` — dispatches to `self.actor_rollout_wg.compute_log_prob(batch)`, which every distributed worker runs **under `torch.no_grad()`, `self.actor_module.eval()`** (`verl/workers/actor/dp_actor.py:354-380`) — a fresh no-grad forward pass of the *current* actor weights over the student's own rollout tokens.
32. `use_reference` / `use_critic` branches — **both `False`** for `OnPolicyDistillAlgorithm` (`algorithm/algorithm.py:486-515`): no separate reference-policy KL pass, no critic/value function. The teacher plays the reference-policy role; the advantage below *is* the whole signal.
33. Advantage computation (`verl_trainer.py:526-532`, only reached because `compute_advantage_in_trainer=True`):
    ```python
    batch, kl_metrics = self.kl_fn.apply_kl_penalty_to_reward(batch)   # DummyKLFn: no-op (kl_penalty_fn: none)
    batch, _ = self.advantage_fn(batch)                                 # MultiTurnOpdAdvantage
    ```
    `MultiTurnOpdAdvantage.__call__` (`algorithm/advantage_fn/on_policy_distill_advantage.py:130`) sets `batch.batch["advantages"] = kl_coef * (teacher_logprobs - old_log_probs)`, elementwise per response token. **Both input tensors are detached** (teacher_logprobs from a separate no-grad vLLM/serving call; old_log_probs from step 31's `torch.no_grad()` pass) — `advantages` therefore carries no gradient; it's a fixed per-token coefficient, exactly like a reward-derived advantage in standard PPO/GRPO.
34. `_update_actor(batch)` (inherited, `ray_trainer.py:1206`) → `self.actor_rollout_wg.update_actor(batch)` — Ray RPC fanning out to every FSDP/Megatron worker process, each running **this repo's own** `DataParallelPPOActor.update_policy()` (`trainer/verl/dp_actor.py:66`).

### `DataParallelPPOActor.update_policy()` — the actual backward pass (`dp_actor.py:66-225`)

35. `mini_batches = data.split(ppo_mini_batch_size)`; for each `ppo_epochs` × `mini_batch` × `micro_batch` (gradient-accumulation chunk, `ppo_micro_batch_size_per_gpu` or dynamic-bsz via `max_token_len_per_gpu`):
    ```python
    entropy, log_prob = self._forward_micro_batch(model_inputs, temperature, calculate_entropy)  # WITH grad, self.actor_module.train()
    pg_loss, pg_loss_metrics = self.policy_loss_fn(logprob=log_prob, **model_inputs)               # PPOPolicyLossFn
    entropy_loss, _ = self.entropy_loss_fn(...)   # DummyEntropyLossFn: no-op (entropy_loss_fn: none)
    kl_loss, _ = self.kl_loss_fn.calculate_kl_loss(...)   # DummyKLFn: no-op (kl_loss_fn: none)
    loss = (pg_loss - entropy_loss + kl_loss) * loss_scale
    loss.backward()
    ```
    `log_prob` here is the *live* forward pass — same tokens as `old_log_probs`/`teacher_logprobs`, but at the actor's current (mid-update) weights, with gradients enabled. This is the only tensor in the whole advantage/loss chain that carries gradient.
36. `PPOPolicyLossFn.__call__` (`algorithm/policy_loss_fn/ppo_policy_loss.py:45`):
    ```python
    ratio = exp(clamp(log_prob - old_logprob, -20, 20))
    pg_losses = max(-advantages*ratio, -advantages*clip(ratio, 1-ε_lo, 1+ε_hi))   # (min-branch when advantages<0)
    pg_loss = aggregate_loss(pg_losses, action_mask, loss_agg_mode="token-mean")
    ```
    **Why this is gradient descent on reverse KL(student‖teacher):** the workflow samples response tokens from the *student's own* rollout policy (Phase E, step 21), then scores that exact sequence under both teacher and student — that's the sampling structure for `KL(π_θ‖π_teacher) = E_{x∼π_θ}[log π_θ(x) − log π_teacher(x)]`, the *reverse* KL (as opposed to standard forward-KL/SFT distillation, which samples from the teacher). The score-function identity `∇_θ E_{x∼π_θ}[f_θ(x)] = E_{x∼π_θ}[∇_θ log π_θ(x)·f_θ(x)]` (the companion term `E[∇_θ log π_θ(x)]` is always exactly zero) applied to `f_θ(x) = log π_θ(x) − log π_teacher(x)` gives `∇_θ KL(π_θ‖π_teacher) = E_{x∼π_θ}[∇_θ log π_θ(x)·(log π_θ(x) − log π_teacher(x))]`. Negate and relabel: `advantages = kl_coef·(log π_teacher − log π_θ_old)` is exactly `kl_coef` times the *negative* of that per-token log-ratio — so maximizing `E[advantages · log π_θ(x)]` (what the PPO surrogate above does, with `ratio ≈ 1 + Δlog_prob` to first order right after a sync) **is** gradient descent on the reverse KL. PPO's clip/ratio machinery is a stability/trust-region device around this — it doesn't change the underlying objective.
    **Caveat worth carrying into the TPU port's correctness checking:** the code applies `advantages[t]` as a *purely local, per-token* quantity — no return-to-go/cumulative sum over future tokens' log-ratios. The textbook policy-gradient theorem for the full-sequence reverse KL would credit token `t` with `Σ_{t'≥t} δ_t'`, not just `δ_t` alone; dropping that is a standard low-variance simplification used throughout the token-level-RLHF/distillation literature (this project's docstrings cite the Tinker library's on-policy distillation as the design reference), not a bug — but it does mean the loss is not an *unbiased* estimator of the exact sequence-level reverse-KL gradient, only of a myopic, per-position local version of it. Match this behavior exactly in the TPU port (i.e. don't "fix" it into a return-to-go form) or the two implementations will target different objectives.
37. `grad_norm = self._optimizer_step()` (grad-clipped by `trainer.grad_clip: 1.0`) — once per mini-batch, after all its micro-batches' `.backward()` calls have accumulated.

**TPU porting note:** this whole phase (29-37) is where FSDP/Megatron-specific and NCCL-specific code lives (`self.actor_rollout_wg` is a verl `RayWorkerGroup` wrapping FSDP or Megatron-parallel workers). The **algorithmic** content — `MultiTurnOpdAdvantage`, `PPOPolicyLossFn`, `DummyKLFn`/`DummyEntropyLossFn`, and the mini/micro-batch/epoch loop structure in `update_policy` — is pure tensor math with no distributed-training-framework dependency baked into the *formulas themselves*; only the sharding/dispatch (`data.split()`, `.to(get_device_id())`, the worker-group RPC) is FSDP/Megatron-specific. A TPU port most naturally replaces `self.actor_rollout_wg` (the worker-group abstraction) with a JAX/PyTorch-XLA SPMD-sharded equivalent that exposes the same three methods this pipeline actually calls on it: `compute_log_prob(batch)`, `update_actor(batch)`, `sync_weight()`/`save_state_dict()`/`upload_state_dict()`. Everything upstream (`to_data_proto`, `MultiTurnOpdAdvantage`, `PPOPolicyLossFn`) can be reused verbatim since it operates on plain tensors, not on any GPU-specific API.

---

## Phase H — Weight synchronization: closing the loop

38. `Trainer.need_sync()` (`trainer.py:138`) — for `sync_method: nccl`, checks `Synchronizer.get_explorer_status_counts()` for `RunningStatus.WAITING_SYNC`.
39. `Trainer.sync_weight()` (`trainer.py:163`) → `self.synchronizer.ready_to_nccl_sync.remote("trainer", train_step_num)` — a **rendezvous**: `Synchronizer.ready_to_nccl_sync()` (`manager/synchronizer.py:369`) only returns a model version once *both* Trainer and Explorer (`explorer.py:360 _nccl_weights_update()`, called from the Explorer's own `need_sync()`) have called it.
40. Once both sides are ready: `self.engine.sync_weight()` → `self.actor_rollout_wg.sync_weight()` (`verl_trainer.py:695`) broadcasts updated weights over the **NCCL** process group set up once at startup (`Explorer.setup_weight_sync_group()` / `actor_rollout_wg.setup_weight_sync_group()` in `VerlPPOTrainerWrapper.prepare()`). The Explorer's vLLM engines pick up the new weights for the next batch of rollouts.

**TPU porting note:** this is the second major GPU-specific surface (NCCL is CUDA-only). TPU has its own high-bandwidth interconnect and collective libraries (e.g. XLA/JAX collectives over ICI), so this broadcast mechanism needs a TPU-native replacement. Two lower-effort alternatives already exist elsewhere in this same codebase and don't require reimplementing a custom collective at all: `sync_method: checkpoint` (`Trainer.sync_weight()`'s `SyncMethod.CHECKPOINT` branch → `self.engine.save_state_dict()`, Explorer reloads from disk) and `sync_method: memory` (`SyncMethod.MEMORY` → `self.engine.upload_state_dict()`, weights pass through the `Synchronizer` actor's own object store rather than a raw collective). Either is a reasonable first TPU-port milestone before attempting a genuine device-to-device broadcast — check correctness against `opd_tpu_repro_1.7b_8b.yaml`'s numbers with `checkpoint` sync first, then optimize.

---

## Condensed end-to-end call graph

```
main() → run() → run_stage() → both()
  → Explorer.get_actor().prepare()      [starts Scheduler + N WorkflowRunner actors; boots rollout-serving engines]
  → Trainer.get_actor().prepare()       [starts actor_rollout_wg distributed workers; loads checkpoint]
  → explorer.explore.remote() ∥ trainer.train.remote()     [concurrent, only meet via buffer + Synchronizer]

Explorer.explore() loop:
  explore_step()
    → taskset.read_async() → TaskFileReader → TaskFormatter.format()
          → WORKFLOWS.get(default_workflow_type)                    [resolve workflow class]
    → Scheduler.schedule() → _scheduler_loop() → _schedule_pending_tasks()
    → RunnerWrapper.run_with_retry() → runner.run_task.remote()     [→ WorkflowRunner actor]
        → WorkflowRunner.run_task() → _run_task() → _asynchronous_run()
            → Task.to_workflow()                                     [INSTANTIATE workflow]
            → workflow_instance.run_async()                          [ENV/ROLLOUT LOGIC RUNS]
  need_sync() → sync_weight() → save_checkpoint(sync_weight=True)
    → _finish_steps() → _finish_explore_step()
        → scheduler.get_results(batch_id=step)
        → experience_pipeline.process.remote(exps) → SQLite experience_buffer

Trainer.train() loop:
  _sample_data() → StalenessControlSampleStrategy.sample() → exp_buffer.read_async(min_model_version=...)
  train_step(exps) → VerlPPOTrainerWrapper.train_step(exps)
    → to_data_proto(exps)                          [Experience.teacher_logprobs → batch["teacher_logprobs"]]
    → _balance_batch()
    → _compute_old_log_prob(batch)                  [no-grad forward @ current weights → batch["old_log_probs"]]
    → kl_fn.apply_kl_penalty_to_reward(batch)        [no-op for OPD]
    → advantage_fn(batch)                            [MultiTurnOpdAdvantage: advantages = kl_coef*(teacher-student), detached]
    → _update_actor(batch) → actor_rollout_wg.update_actor(batch)
        → DataParallelPPOActor.update_policy(data)   [per mini/micro-batch/epoch:]
            → _forward_micro_batch()                  [WITH grad → fresh log_prob]
            → PPOPolicyLossFn(logprob, old_logprob, advantages)   [clipped surrogate ≈ reverse-KL policy gradient]
            → loss.backward() → _optimizer_step()
  need_sync() → sync_weight()
    → Synchronizer.ready_to_nccl_sync("trainer", step)   [rendezvous with Explorer's matching call]
    → actor_rollout_wg.sync_weight()                      [NCCL broadcast → Explorer's rollout engines]
```

---

## Summary table: what to keep vs. what to rebuild for TPU

| Layer | Files | GPU/CUDA-specific? | TPU-port action |
|---|---|---|---|
| CLI/config/Ray orchestration | `cli/launcher.py`, `common/config*.py` | No | Reuse as-is |
| Task reading, `Task`/`Workflow` abstraction, `WORKFLOWS` registry | `buffer/reader/*`, `common/workflows/workflow.py`, `common/workflows/__init__.py` | No | Reuse as-is |
| Scheduler (task queue, runner pool, retry/timeout) | `explorer/scheduler.py` | No | Reuse as-is |
| Rollout-serving engine (student + teacher inference) | `common/models/vllm_model.py`, `create_inference_models` | **Yes** (vLLM/CUDA) | **Rebuild.** Must expose `chat_async`/`logprobs_async` with the same `Experience` contract |
| Workflow logic (env loop, prompt building, reward) | `common/workflows/envs/TCOD/alfworld/OPD_workflow_fullmemory.py` | No | Reuse as-is once serving engine exists |
| Experience buffer (SQLite queue) | `buffer/pipelines/experience_pipeline.py`, `buffer/storage/*` | No | Reuse as-is |
| Sample strategy / staleness control | `algorithm/sample_strategy/*` | No | Reuse as-is |
| `to_data_proto`, advantage_fn, policy_loss_fn, kl_fn, entropy_loss_fn formulas | `trainer/verl/utils.py`, `algorithm/advantage_fn/*`, `algorithm/policy_loss_fn/*` | No (pure tensor math) | Reuse as-is |
| Distributed training dispatch (worker groups, sharding) | `trainer/verl/fsdp_workers.py`, `trainer/verl/megatron_*.py`, `trainer/verl_trainer.py`'s `actor_rollout_wg` calls | **Yes** (FSDP/Megatron/CUDA) | **Rebuild.** Needs a JAX/PyTorch-XLA SPMD equivalent exposing `compute_log_prob`/`update_actor`/`sync_weight`/`save_state_dict` |
| Weight synchronization | `manager/synchronizer.py`'s NCCL path, `setup_weight_sync_group` | **Yes** (NCCL) | **Rebuild**, or fall back to `sync_method: checkpoint`/`memory` (already implemented, backend-agnostic) as a first milestone |

**Suggested validation path:** get the two "rebuild" rows working with `sync_method: checkpoint` first (lowest-risk sync path, already implemented), run `opd_tpu_repro_1.7b_8b.yaml`'s exact config (Qwen3-1.7B/8B, `OPD_alfworld_workflow_fullmemory`, 100 steps) on both GPU and TPU, and diff the tensorboard curves (`kl/mean`, `kl/trajectory_mean`, `pg_loss`, eval success rate) before attempting the NCCL-equivalent broadcast path.
