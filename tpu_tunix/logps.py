"""Completion-only log-probs and the OPD policy loss.

Tunix's `common.compute_per_token_logps` runs the LM head over every position
and slices the completion afterwards. At full-history lengths (10240-token
prompt + 512 response) the fp32 [B, 10752, 151936] logits and their gradient
alone exceed a v4 chip's HBM. Here the decoder stack runs over the whole
sequence but the LM head only over the positions that predict completion
tokens.

Works for Tunix's Qwen2/Qwen3 (embedder / layers / final_norm / lm_head).
"""

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.rl import common


def _hidden_states(model, input_tokens, positions, attn_mask, segment_ids):
    x = model.embedder.encode(input_tokens)
    for layer in model.layers:
        _, x = layer(x, positions, None, attn_mask, segment_ids=segment_ids)
    return model.final_norm(x)


def _project(model, h):
    if model.config.use_tied_embedding:
        return model.embedder.decode(h)
    return model.lm_head(h)


def completion_logps(model, prompt_tokens, completion_tokens, pad_id, eos_id, temperature=1.0,
                     return_entropy=False):
    """Per-token log p(completion_t | prefix) for the completion positions: [B, C]."""
    input_tokens, positions, attn_mask, seg_ids = common.process_ids(
        prompt_tokens, completion_tokens, pad_id, eos_id)
    h = _hidden_states(model, input_tokens, positions, attn_mask, seg_ids)
    c = completion_tokens.shape[1]
    logits = _project(model, h[:, -c - 1:-1, :]).astype(jnp.float32)  # positions predicting completion
    if temperature not in (0.0, 1.0):
        logits = logits / temperature
    logps = common.selective_log_softmax(logits, completion_tokens)
    if not return_entropy:
        return logps
    p = jax.nn.softmax(logits, axis=-1)
    entropy = -(p * jax.nn.log_softmax(logits, axis=-1)).sum(-1)
    return logps, entropy


def make_scoring_fn():
    """jit-compiled (graphdef, state, prompt, completion) -> [B, C] log-probs, no gradient."""

    @functools.partial(jax.jit, static_argnums=(0, 4, 5, 6))
    def score(graphdef, state, prompt_tokens, completion_tokens, pad_id, eos_id, temperature):
        model = nnx.merge(graphdef, state)
        return jax.lax.stop_gradient(completion_logps(
            model, prompt_tokens, completion_tokens, pad_id, eos_id, temperature))

    return score


def opd_loss(model, train_example, *, pad_id, eos_id, temperature, clip_low, clip_high, clip_ratio_c,
             loss_agg_mode="token-mean"):
    """Dual-clip PPO loss with per-token advantages (same math as Tunix's grpo_loss_fn
    and Trinity's PPOPolicyLossFn); advantages = kl_coef * (teacher - student_old)."""
    mask = train_example.completion_mask.astype(jnp.float32)
    logps, entropy = completion_logps(model, train_example.prompt_ids, train_example.completion_ids,
                                      pad_id, eos_id, temperature, return_entropy=True)
    old = train_example.old_per_token_logps.astype(jnp.float32)
    adv = train_example.advantages.astype(jnp.float32)

    log_ratio = jnp.clip(logps - old, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clipped = jnp.maximum(pg1, pg2)
    pg3 = -adv * clip_ratio_c
    per_token = jnp.where(adv < 0.0, jnp.minimum(clipped, pg3), clipped)
    loss = common.aggregate_loss(per_token, mask, loss_agg_mode)

    denom = jnp.clip(mask.sum(), min=1.0)
    aux = {
        "pg_clipfrac": ((pg2 > pg1) * mask).sum() / denom,
        "pg_clipfrac_lower": (((clipped > pg3) & (adv < 0.0)) * mask).sum() / denom,
        "ppo_kl": (-log_ratio * mask).sum() / denom,
        "entropy": (jax.lax.stop_gradient(entropy) * mask).sum() / denom,
        "advantage/abs_mean": (jnp.abs(adv) * mask).sum() / denom,
        "is_ratio/max": jnp.max(jnp.where(mask > 0, ratio, 0.0)),
        "is_ratio/min": jnp.min(jnp.where(mask > 0, ratio, jnp.inf)),
    }
    return loss, aux
