"""Multi-turn on-policy distillation (OPD) for ALFWorld on TPU, built on Tunix.

JAX/Tunix counterpart of Trinity's `on_policy_distill` algorithm with the
`multi_turn_opd` advantage and `OPD_alfworld_workflow`:

  1. The student plays a batch of ALFWorld episodes. All live episodes are
     stepped in lockstep, one batched generate call per turn.
  2. Every turn is one training row: (self-contained prompt, student response).
  3. The teacher scores each row with a prefill-only forward pass (Tunix's
     REFERENCE role holds the teacher).
  4. Per-token advantage = kl_coef * (teacher_logp - student_logp), optimised
     with the dual-clip PPO loss (Tunix's grpo_loss_fn with 2D advantages).

Trainer, sampler and teacher are colocated on one mesh; the sampler shares the
trainer's weights, so no weight sync is needed between steps.
"""

import dataclasses
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import TrainExample
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.utils.mesh import create_mesh

from alfworld_env import AlfworldEpisode
from logps import make_scoring_fn, opd_loss
from models import load_model

Role, Mode = rl_cluster_lib.Role, rl_cluster_lib.Mode
log = logging.getLogger("tcod_opd")


@dataclasses.dataclass
class OPDConfig:
    # Models. `*_model_name` is Tunix's config name, only needed when it can't
    # be inferred from the hub id (e.g. local checkpoint dirs).
    student: str = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher: str = "Qwen/Qwen2.5-7B-Instruct"
    student_model_name: Optional[str] = None
    teacher_model_name: Optional[str] = None
    # fp32 master weights for the trained student (like verl/FSDP). With bf16
    # weights, lr~1e-6 Adam steps are mostly below bf16 resolution and are lost.
    # The colocated sampler shares these weights, so it also runs in fp32.
    student_dtype: str = "float32"
    # Flash (splash) attention for the teacher's scoring passes; needed for long
    # prompts. Not used for the student: Tunix's vanilla sampler produces
    # garbage with flash attention on (verified on Qwen3-4B).
    teacher_flash_attention: bool = True
    flash_block_size: int = 512  # must divide max_prompt_length + max_response_tokens
    student_remat: str = "NONE"  # NONE | BLOCK | DECODER (DECODER needed at 10k tokens)
    enable_thinking: bool = False  # Qwen3 chat template flag; Trinity's default is False

    # Data (jsonl with a `game_file` field, as in TCOD_examples/alfworld).
    train_file: str = ""
    eval_file: Optional[str] = None
    game_path_prefix_from: Optional[str] = None  # rewrite game_file prefixes
    game_path_prefix_to: Optional[str] = None

    # Rollout.
    batch_size: int = 16  # episodes per training step (= generate batch)
    max_env_steps: int = 30
    max_prompt_length: int = 1536
    max_response_tokens: int = 512
    temperature: float = 1.0
    eval_temperature: float = 0.4

    # Optimisation. One optimizer update per `mini_batch_size` turns; every
    # turn collected in a step is used once (one PPO epoch).
    total_steps: int = 250
    mini_batch_size: int = 64
    micro_batch_size: int = 2  # train rows per grad-accumulation step (logits dominate HBM)
    logps_micro_batch_size: int = 8  # rows per teacher/student scoring pass (no backward)
    lr: float = 1e-6
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    kl_coef: float = 1.0
    clip_range_low: float = 0.2
    clip_range_high: float = 0.2
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"

    # Mesh over all visible devices: (fsdp, tp).
    mesh_fsdp: int = -1  # -1: all devices
    mesh_tp: int = 1

    eval_interval: int = 5
    eval_episodes: int = 32
    log_dir: str = "./tcod_tpu_logs"
    seed: int = 42

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "OPDConfig":
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        out = {}
        for k, v in values.items():
            # YAML 1.1 reads "1e-6" as a string; coerce numeric fields.
            if isinstance(v, str) and fields[k].type in ("float", float):
                v = float(v)
            elif isinstance(v, str) and fields[k].type in ("int", int):
                v = int(v)
            out[k] = v
        return cls(**out)


def _load_tasks(path: str, cfg: OPDConfig) -> List[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            if cfg.game_path_prefix_from and t["game_file"].startswith(cfg.game_path_prefix_from):
                t["game_file"] = cfg.game_path_prefix_to + t["game_file"][len(cfg.game_path_prefix_from):]
            tasks.append(t)
    missing = [t["game_file"] for t in tasks if not os.path.exists(t["game_file"])]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(tasks)} game files in {path} do not exist, e.g. {missing[0]}. "
            "Set game_path_prefix_from/to to remap them.")
    return tasks


class OPDTrainer:

    def __init__(self, cfg: OPDConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        os.makedirs(cfg.log_dir, exist_ok=True)
        self._metrics_file = open(os.path.join(cfg.log_dir, "metrics.jsonl"), "a")
        self._samples_file = open(os.path.join(cfg.log_dir, "samples.jsonl"), "a")

        n_dev = jax.device_count()
        fsdp = n_dev // cfg.mesh_tp if cfg.mesh_fsdp == -1 else cfg.mesh_fsdp
        self.mesh = create_mesh((fsdp, cfg.mesh_tp), ("fsdp", "tp"))
        log.info("mesh %s", dict(self.mesh.shape))

        t0 = time.time()
        student_overrides = {}
        if cfg.student_remat != "NONE":
            from tunix.models.qwen3 import model as qwen3_lib
            student_overrides["remat_config"] = getattr(qwen3_lib.RematConfig, cfg.student_remat)
        student, self.tok, _ = load_model(cfg.student, self.mesh, cfg.student_model_name,
                                          dtype=jnp.dtype(cfg.student_dtype), **student_overrides)
        teacher_overrides = {}
        if cfg.teacher_flash_attention:
            if (cfg.max_prompt_length + cfg.max_response_tokens) % cfg.flash_block_size:
                raise ValueError("flash_block_size must divide max_prompt_length + max_response_tokens")
            teacher_overrides = dict(use_flash_attention=True, flash_attention_block_size=cfg.flash_block_size)
        teacher, teacher_tok, _ = load_model(cfg.teacher, self.mesh, cfg.teacher_model_name,
                                             **teacher_overrides)
        log.info("loaded student+teacher in %.1fs", time.time() - t0)
        if self.tok.get_vocab() != teacher_tok.get_vocab():
            raise ValueError("Student and teacher tokenizers differ; token-level OPD needs a shared vocab.")

        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        eos_tokens = sorted({self.eos_id, self.tok.eos_token_id, self.pad_id})

        rollout_cfg = base_rollout.RolloutConfig(
            max_tokens_to_generate=cfg.max_response_tokens,
            max_prompt_length=cfg.max_prompt_length,
            kv_cache_size=cfg.max_prompt_length + cfg.max_response_tokens + 256,
            temperature=cfg.temperature,
            top_p=1.0,
            top_k=None,
            eos_tokens=eos_tokens,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adamw(cfg.lr, b1=0.9, b2=0.999, weight_decay=cfg.weight_decay),
        )
        training_config = rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=10**9,
            max_steps=10**9,
            mini_batch_size=cfg.mini_batch_size,
            train_micro_batch_size=cfg.micro_batch_size,
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                log_dir=os.path.join(cfg.log_dir, "tunix"), flush_every_n_steps=1),
        )
        self.cluster = rl_cluster_lib.RLCluster(
            actor=student,
            reference=teacher,
            tokenizer=self.tok,
            cluster_config=rl_cluster_lib.ClusterConfig(
                role_to_mesh={Role.ACTOR: self.mesh, Role.REFERENCE: self.mesh, Role.ROLLOUT: self.mesh},
                rollout_engine="vanilla",
                offload_to_cpu=False,
                training_config=training_config,
                rollout_config={
                    Mode.TRAIN: rollout_cfg,
                    Mode.EVAL: dataclasses.replace(rollout_cfg, temperature=cfg.eval_temperature),
                },
            ),
        )

        loss_kwargs = dict(
            pad_id=self.pad_id, eos_id=self.eos_id, temperature=cfg.temperature,
            clip_low=cfg.clip_range_low, clip_high=cfg.clip_range_high,
            clip_ratio_c=cfg.clip_ratio_c, loss_agg_mode=cfg.loss_agg_mode)
        trainer = self.cluster.actor_trainer
        # Completion-only LM head (logps.py): full-sequence logits don't fit at 10k tokens.
        trainer.with_loss_fn(lambda model, train_example: opd_loss(model, train_example, **loss_kwargs),
                             has_aux=True)
        self._score_fn = make_scoring_fn()
        trainer.with_gen_model_input_fn(lambda x: {"train_example": x})
        trainer.with_rl_metrics_to_log({
            k: np.mean for k in ["pg_clipfrac", "pg_clipfrac_lower", "ppo_kl", "entropy",
                                 "advantage/abs_mean", "is_ratio/max", "is_ratio/min"]})
        # We drive trainer.train() once per step ourselves; without this it
        # skips "already seen" examples on every call after the first.
        trainer.is_managed_externally = True

        self.train_tasks = _load_tasks(cfg.train_file, cfg)
        self.eval_tasks = _load_tasks(cfg.eval_file, cfg) if cfg.eval_file else []
        self._gen_calls = 0

    # ------------------------------------------------------------------ rollout

    def _chat_prompt(self, messages: List[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.cfg.enable_thinking)

    def _generate(self, prompts: List[str], mode: Mode) -> base_rollout.RolloutOutput:
        # Fresh seed per call; the sampler otherwise reuses PRNGKey(0).
        rc = self.cluster.cluster_config.rollout_config
        self._gen_calls += 1
        rc[mode] = dataclasses.replace(rc[mode], seed=self.cfg.seed * 1_000_003 + self._gen_calls)
        return self.cluster.generate(prompts, mode=mode)

    def run_episodes(self, tasks: List[dict], mode: Mode) -> Dict[str, Any]:
        """Plays `tasks` to completion in lockstep; returns turns + episode stats."""
        cfg = self.cfg
        # Sequential on purpose: textworld's PDDL parser (tatsu) is not thread-safe.
        episodes = [AlfworldEpisode(t["game_file"], max_env_steps=cfg.max_env_steps) for t in tasks]
        turns = []  # (episode index, prompt row [P], completion tokens, user msg, response text)
        try:
            while True:
                live = []
                for i, ep in enumerate(episodes):
                    if not ep.active:
                        continue
                    prompt = self._chat_prompt(ep.messages())
                    if len(self.tok(prompt).input_ids) > cfg.max_prompt_length:
                        ep.truncated, ep.done = True, True
                        continue
                    live.append((i, prompt))
                if not live:
                    break
                prompts = [p for _, p in live]
                # Fixed generate batch (= batch_size) keeps one compiled program;
                # filler slots repeat a live prompt and are discarded.
                n_pad = max(0, cfg.batch_size - len(prompts))
                out = self._generate(prompts + [prompts[0]] * n_pad, mode)
                for k, (i, _) in enumerate(live):
                    completion = np.asarray(out.tokens[k], dtype=np.int32)
                    if len(completion) < cfg.max_response_tokens:
                        # The sampler drops the stop token; keep it so the
                        # student learns (from the teacher) when to stop.
                        completion = np.append(completion, self.eos_id)
                    turns.append((i, np.asarray(out.left_padded_prompt_tokens[k]), completion,
                                  episodes[i].user_content(), out.text[k]))
                for k, (i, _) in enumerate(live):
                    episodes[i].step(out.text[k])
        finally:
            for ep in episodes:
                ep.close()
        return {
            "turns": turns,
            "success": [ep.reward for ep in episodes],
            "env_rounds": [ep.step_idx for ep in episodes],
            "truncated": [float(ep.truncated) for ep in episodes],
        }

    # ------------------------------------------------------------------ training

    def _pad_completions(self, comps: List[np.ndarray]) -> np.ndarray:
        out = np.full((len(comps), self.cfg.max_response_tokens), self.pad_id, dtype=np.int32)
        for r, c in enumerate(comps):
            out[r, :len(c)] = c
        return out

    def build_batch(self, turns: List[tuple]) -> Dict[str, Any]:
        """Stacks turns into fixed-shape arrays.

        The turn count is padded up to a whole number of mini-batches by
        re-using random real turns (no all-pad rows: they would dilute the
        token-mean loss and all-masked attention rows can produce NaNs).
        """
        cfg = self.cfg
        n_real = len(turns)
        n_total = -(-n_real // cfg.mini_batch_size) * cfg.mini_batch_size
        order = list(range(n_real))
        self.rng.shuffle(order)
        order += [self.rng.randrange(n_real) for _ in range(n_total - n_real)]
        completion_ids = self._pad_completions([turns[j][2] for j in order])
        return {
            "n_real": n_real,
            "ep_idx": np.array([turns[j][0] for j in order]),
            "prompt_ids": jnp.asarray(np.stack([turns[j][1] for j in order])),
            "completion_ids": jnp.asarray(completion_ids),
            "mask_np": completion_ids != self.pad_id,
            "prompt_lens": np.array([int((turns[j][1] != self.pad_id).sum()) for j in order[:n_real]]),
        }

    def score(self, batch: Dict[str, Any]) -> tuple:
        """Returns (teacher_logps, student_logps) under the current policy."""
        mb = self.cfg.logps_micro_batch_size
        prompt, comp = batch["prompt_ids"], batch["completion_ids"]
        out = []
        with self.mesh:
            for model in (self.cluster.inference_worker.get_model("reference"), self.cluster.rollout.model()):
                graphdef, state = nnx.split(model)
                out.append(jnp.concatenate([
                    self._score_fn(graphdef, state, prompt[a:a + mb], comp[a:a + mb],
                                   self.pad_id, self.eos_id, self.cfg.temperature)
                    for a in range(0, prompt.shape[0], mb)], axis=0))
        return out[0], out[1]

    def update(self, batch: Dict[str, Any], teacher_logps, old_logps) -> None:
        """One PPO epoch over the batch: one optimizer update per mini-batch."""
        cfg = self.cfg
        prompt_ids, completion_ids = batch["prompt_ids"], batch["completion_ids"]
        completion_mask = jnp.asarray(batch["mask_np"])
        advantages = cfg.kl_coef * (teacher_logps - old_logps) * completion_mask
        n_total = prompt_ids.shape[0]
        examples = [
            TrainExample(
                prompt_ids=prompt_ids[s],
                prompt_mask=prompt_ids[s] != self.pad_id,
                completion_ids=completion_ids[s],
                completion_mask=completion_mask[s],
                advantages=advantages[s],
                ref_per_token_logps=None,
                old_per_token_logps=old_logps[s],
            )
            for s in (slice(a, a + cfg.micro_batch_size) for a in range(0, n_total, cfg.micro_batch_size))
        ]
        self.cluster.update_actor(iter(examples), None)
        self.cluster.global_steps += 1  # sampler shares the actor's weights: no sync

    def train_step(self, step: int) -> Dict[str, float]:
        cfg = self.cfg
        t0 = time.time()
        tasks = self.rng.sample(self.train_tasks, cfg.batch_size)
        roll = self.run_episodes(tasks, Mode.TRAIN)
        t_roll = time.time() - t0
        if not roll["turns"]:
            log.warning("step %d produced no turns", step)
            return {}
        batch = self.build_batch(roll["turns"])
        for ep_i, _, _, user, text in self.rng.sample(roll["turns"], min(4, len(roll["turns"]))):
            self._samples_file.write(json.dumps({"step": step, "episode": int(ep_i), "user": user,
                                                 "response": text}) + "\n")
        self._samples_file.flush()
        t1 = time.time()
        # Student logps from the pre-update policy (= the sampling policy) are
        # both the advantage's student term and the PPO ratio's denominator.
        teacher_logps, old_logps = self.score(batch)
        t_score = time.time() - t1
        t2 = time.time()
        self.update(batch, teacher_logps, old_logps)
        t_train = time.time() - t2
        n_real, ep_idx, mask_np = batch["n_real"], batch["ep_idx"], batch["mask_np"]
        n_total = len(ep_idx)

        # Metrics (real turns only). kl = student_logp - teacher_logp.
        real = np.arange(n_total) < n_real
        kl_tok = np.asarray(old_logps - teacher_logps) * mask_np
        kl_turn = kl_tok.sum(-1)
        traj_kl = {}
        for k in np.flatnonzero(real):
            traj_kl[ep_idx[k]] = traj_kl.get(ep_idx[k], 0.0) + float(kl_turn[k])
        resp_len = mask_np[real].sum(-1)
        return {
            "step": step,
            "rollout/success_rate": float(np.mean(roll["success"])),
            "rollout/env_rounds": float(np.mean(roll["env_rounds"])),
            "rollout/prompt_truncated": float(np.mean(roll["truncated"])),
            "rollout/num_turns": n_real,
            "rollout/response_len_mean": float(resp_len.mean()),
            "rollout/prompt_len_max": int(batch["prompt_lens"].max()),
            "rollout/response_clip_ratio": float(np.mean(resp_len >= cfg.max_response_tokens)),
            "kl/token_mean": float(kl_tok[real].sum() / max(mask_np[real].sum(), 1)),
            "kl/trajectory_mean": float(np.mean(list(traj_kl.values()))),
            "train/optimizer_updates": n_total // cfg.mini_batch_size,
            "time/rollout_s": t_roll,
            "time/score_s": t_score,
            "time/train_s": t_train,
        }

    def evaluate(self) -> Dict[str, float]:
        if not self.eval_tasks:
            return {}
        tasks = self.eval_tasks[: self.cfg.eval_episodes]
        success, rounds = [], []
        for a in range(0, len(tasks), self.cfg.batch_size):
            roll = self.run_episodes(tasks[a:a + self.cfg.batch_size], Mode.EVAL)
            success += roll["success"]
            rounds += roll["env_rounds"]
        return {"eval/success_rate": float(np.mean(success)), "eval/env_rounds": float(np.mean(rounds))}

    def _log(self, metrics: Dict[str, float]) -> None:
        log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
        self._metrics_file.write(json.dumps(metrics) + "\n")
        self._metrics_file.flush()

    def train(self) -> None:
        cfg = self.cfg
        for step in range(1, cfg.total_steps + 1):
            metrics = self.train_step(step)
            if cfg.eval_interval and step % cfg.eval_interval == 0:
                metrics.update(self.evaluate())
            self._log(metrics)
        self.cluster.close()
