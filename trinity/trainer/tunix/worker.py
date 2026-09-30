"""JAX/Tunix actor worker for the `tunix` trainer: the TPU counterpart of verl's
actor worker group (`compute_log_prob` + `update_actor` + checkpoint export).

Runs as a Ray actor holding `TPU` chips; the trainer wrapper sends it batches as
plain numpy arrays. One `train_step` reproduces VerlPPOTrainerWrapper.train_step
for algorithms with `compute_advantage_in_trainer` (on_policy_distill):

  old_log_probs  = no-grad forward of the current weights        (_compute_old_log_prob)
  advantages     = kl_coef * (teacher_logprobs - old_log_probs)   (MultiTurnOpdAdvantage)
  update_policy  = ppo_epochs x mini-batches x micro-batches with
                   PPOPolicyLossFn, micro loss scaled by rows/ppo_mini_batch_size
                   (verl's use_dynamic_bsz scaling), grad clip, AdamW step.
"""

import math
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, struct

from trinity.trainer.tunix import hf_io
from trinity.trainer.tunix.logps import completion_logps, opd_loss


@struct.dataclass
class _MicroBatch:
    prompt_ids: jax.Array
    completion_ids: jax.Array
    completion_mask: jax.Array
    old_per_token_logps: jax.Array
    advantages: jax.Array


class TunixActorWorker:
    def __init__(self, cfg: Dict[str, Any]):
        from tunix.models.qwen3 import model as qwen3_lib
        from tunix.utils.mesh import create_mesh

        self.cfg = cfg
        n = jax.device_count()
        self.mesh = create_mesh((n, 1), ("fsdp", "tp"))
        self.model_dir = hf_io.resolve_model_dir(cfg["model_path"])
        overrides = dict(
            use_flash_attention=True,
            flash_attention_block_size=cfg["flash_block_size"],
            remat_config=getattr(qwen3_lib.RematConfig, cfg["remat"]),
        )
        with self.mesh:
            model = hf_io.load_model(self.model_dir, self.mesh, dtype=jnp.float32, **overrides)
        # Functional state: fp32 master params (sharded over `fsdp`) + optax state.
        self.graphdef, self.params, self.rest = nnx.split(model, nnx.Param, ...)
        del model
        self.param_shardings = jax.tree.map(lambda x: x.sharding, self.params)
        opt = cfg["optimizer"]
        # torch.optim.AdamW (verl's default) == optax.adamw with the same betas/eps/wd.
        self.tx = optax.chain(
            optax.clip_by_global_norm(opt["clip_grad"]),
            optax.adamw(opt["lr"], b1=opt["betas"][0], b2=opt["betas"][1], eps=1e-8,
                        weight_decay=opt["weight_decay"]),
        )
        with self.mesh:
            self.opt_state = self.tx.init(self.params)  # zeros_like keeps the param sharding
        self.pad_id = cfg["pad_token_id"]
        self.eos_id = cfg["eos_token_id"]
        graphdef, rest = self.graphdef, self.rest
        loss_kwargs = dict(
            pad_id=self.pad_id, eos_id=self.eos_id, temperature=cfg["temperature"],
            clip_low=cfg["clip_range_low"], clip_high=cfg["clip_range_high"],
            clip_ratio_c=cfg["clip_ratio_c"], loss_agg_mode=cfg["loss_agg_mode"])
        pad_id, eos_id, temperature = self.pad_id, self.eos_id, cfg["temperature"]

        def score(params, prompt, comp):
            model = nnx.merge(graphdef, params, rest)
            return jax.lax.stop_gradient(completion_logps(model, prompt, comp, pad_id, eos_id, temperature))

        def grad_step(params, mb, scale, acc):
            def loss_fn(p):
                loss, aux = opd_loss(nnx.merge(graphdef, p, rest), mb, **loss_kwargs)
                return loss * scale, aux

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            return loss, aux, jax.tree.map(jnp.add, acc, grads)

        def apply_grads(params, opt_state, grads):
            grad_norm = optax.global_norm(grads)
            updates, opt_state = self.tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, grad_norm

        ps = self.param_shardings
        self._score = jax.jit(score)
        # The gradient accumulator is donated and pinned to the param sharding so it
        # stays one sharded fp32 copy (unconstrained grads come out replicated).
        self._grad_step = jax.jit(grad_step, donate_argnums=(3,), out_shardings=(None, None, ps))
        self._apply_grads = jax.jit(apply_grads, donate_argnums=(0, 1, 2))
        self._zeros = jax.jit(lambda p: jax.tree.map(jnp.zeros_like, p), out_shardings=ps)
        self.step = 0

    # ------------------------------------------------------------------ batching

    def _bucket(self, total_len: int) -> int:
        for b in self.cfg["seq_buckets"]:
            if total_len <= b:
                return b
        raise ValueError(f"sequence of {total_len} tokens exceeds the largest bucket {self.cfg['seq_buckets'][-1]}")

    def _micro_batches(self, rows: List[Dict[str, np.ndarray]], rows_per_micro: int):
        """Group rows of similar length; pad each group to one (prompt, response) bucket."""
        c = self.cfg["max_response_tokens"]
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
        groups = [order[a:a + rows_per_micro] for a in range(0, len(order), rows_per_micro)]
        for g in groups:
            # pad the group to a device-divisible row count with fully-masked copies
            while len(g) % jax.device_count():
                g = g + [g[-1]]
            p_len = max(len(rows[i]["prompt"]) for i in g)
            total = self._bucket(p_len + c)
            p = total - c
            prompt = np.full((len(g), p), self.pad_id, np.int32)
            comp = np.full((len(g), c), self.pad_id, np.int32)
            mask = np.zeros((len(g), c), bool)
            teacher = np.zeros((len(g), c), np.float32)
            seen = set()
            for r, i in enumerate(g):
                row = rows[i]
                prompt[r, p - len(row["prompt"]):] = row["prompt"]
                comp[r, :len(row["response"])] = row["response"]
                if i not in seen:  # duplicated filler rows stay masked out
                    mask[r, :len(row["response"])] = row["response_mask"]
                    teacher[r, :len(row["response"])] = row["teacher_logprobs"]
                seen.add(i)
            yield {"prompt": prompt, "comp": comp, "mask": mask, "teacher": teacher, "n_real": len(set(g)),
                   "row_ids": g}

    # ------------------------------------------------------------------ step

    def train_step(self, rows: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        cfg = self.cfg
        t0 = time.time()
        micro = list(self._micro_batches(rows, cfg["rows_per_micro_batch"]))
        with self.mesh:
            # _compute_old_log_prob: no-grad forward at the current (pre-update) weights.
            for mb in micro:
                mb["old"] = np.asarray(self._score(self.params, jnp.asarray(mb["prompt"]),
                                                   jnp.asarray(mb["comp"])))
        t_old = time.time() - t0

        # MultiTurnOpdAdvantage: detached per-token advantages, masked.
        row_kl = [0.0] * len(rows)  # per-turn sum of (old - teacher), in input order
        row_adv = [0.0] * len(rows)
        for mb in micro:
            mb["adv"] = cfg["kl_coef"] * (mb["teacher"] - mb["old"]) * mb["mask"]
            kl = ((mb["old"] - mb["teacher"]) * mb["mask"]).sum(-1)
            adv = mb["adv"].sum(-1)
            for r, i in enumerate(dict.fromkeys(mb["row_ids"])):  # first occurrence = real row
                row_kl[i], row_adv[i] = float(kl[r]), float(adv[r])
        kl_sums, adv_sums = row_kl, row_adv

        # update_policy: ppo_epochs x one mini-batch (ppo_mini_batch_size == train_batch_size).
        t1 = time.time()
        n_rows = len(rows)
        acc = {}
        grad_norm = None
        for _ in range(cfg["ppo_epochs"]):
            with self.mesh:
                grads_sum = self._zeros(self.params)
            for mb in micro:
                batch = _MicroBatch(
                    prompt_ids=jnp.asarray(mb["prompt"]), completion_ids=jnp.asarray(mb["comp"]),
                    completion_mask=jnp.asarray(mb["mask"]), old_per_token_logps=jnp.asarray(mb["old"]),
                    advantages=jnp.asarray(mb["adv"], jnp.float32))
                scale = mb["n_real"] / n_rows  # verl: response_mask.shape[0] / ppo_mini_batch_size
                with self.mesh:
                    loss, aux, grads_sum = self._grad_step(self.params, batch, scale, grads_sum)
                acc.setdefault("actor/pg_loss", 0.0)
                acc["actor/pg_loss"] += float(loss)
                for k in ("pg_clipfrac", "pg_clipfrac_lower", "ppo_kl", "entropy"):
                    acc.setdefault(f"actor/{k}", []).append(float(aux[k]))
            with self.mesh:
                self.params, self.opt_state, grad_norm = self._apply_grads(
                    self.params, self.opt_state, grads_sum)
            grad_norm = float(grad_norm)
        self.step += 1
        resp_len = [int(r["response_mask"].sum()) for r in rows]
        prompt_len = [len(r["prompt"]) for r in rows]
        metrics = {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in acc.items()}
        metrics.update({
            "actor/grad_norm": grad_norm,
            "actor/lr": cfg["optimizer"]["lr"],
            "kl/mean": float(np.mean(kl_sums)),
            "kl/std": float(np.std(kl_sums, ddof=1)) if len(kl_sums) > 1 else 0.0,
            "advantages/mean": float(np.mean(adv_sums)),
            "response_length/mean": float(np.mean(resp_len)),
            "response_length/max": float(np.max(resp_len)),
            "prompt_length/mean": float(np.mean(prompt_len)),
            "prompt_length/max": float(np.max(prompt_len)),
            "time/old_log_prob": t_old,
            "time/update_actor": time.time() - t1,
            "tunix/micro_batches": len(micro),
            "_row_kl": row_kl,
        })
        return metrics

    def save_hf(self, output_dir: str) -> str:
        with self.mesh:
            model = nnx.merge(self.graphdef, self.params, self.rest)
            return hf_io.save_hf_checkpoint(model, self.model_dir, output_dir)

    def ready(self) -> bool:
        return True


def default_seq_buckets(max_prompt_tokens: int, max_response_tokens: int, block: int) -> List[int]:
    """Total-length buckets (prompt + response), multiples of the flash block size."""
    top = int(math.ceil((max_prompt_tokens + max_response_tokens) / block) * block)
    buckets, b = [], 2 * block
    while b < top:
        buckets.append(b)
        b *= 2
    return buckets + [top]
