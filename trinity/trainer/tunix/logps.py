"""Completion-only log-probs and the OPD policy loss.

Tunix's `common.compute_per_token_logps` runs the LM head over every position
and slices the completion afterwards. At full-history lengths (10240-token
prompt + 512 response) the fp32 [B, 10752, 151936] logits and their gradient
alone exceed a v4 chip's HBM. Here the decoder stack runs over the whole
sequence but the LM head only over the positions that predict completion
tokens.

Works for Tunix's Qwen2/Qwen3 (embedder / layers / final_norm / lm_head) and Gemma 4 (gemma4.py).
"""

import functools

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.rl import common


def _is_gemma4(model):
    return type(model).__name__ == "Gemma4"


def _hidden_states(model, input_tokens, positions, attn_mask, segment_ids):
    if _is_gemma4(model):
        from trinity.trainer.tunix import gemma4

        return gemma4.hidden_states(model, input_tokens, positions, attn_mask)
    x = model.embedder.encode(input_tokens)
    for layer in model.layers:
        _, x = layer(x, positions, None, attn_mask, segment_ids=segment_ids)
    return model.final_norm(x)


def _project(model, h):
    if _is_gemma4(model):
        from trinity.trainer.tunix import gemma4

        return gemma4.project(model, h)
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
    """jit-compiled (model, prompt, completion, pad_id, eos_id, temperature) -> [B, C]
    log-probs, no gradient."""

    @functools.partial(nnx.jit, static_argnames=("pad_id", "eos_id", "temperature"))
    def score(model, prompt_tokens, completion_tokens, pad_id, eos_id, temperature):
        return jax.lax.stop_gradient(completion_logps(
            model, prompt_tokens, completion_tokens, pad_id, eos_id, temperature))

    return score


def opd_loss(model, train_example, *, pad_id, eos_id, temperature, clip_low, clip_high, clip_ratio_c,
             loss_agg_mode="token-mean"):
    """Trinity's PPOPolicyLossFn (algorithm/policy_loss_fn/ppo_policy_loss.py) in JAX:
    dual-clip PPO with per-token advantages, advantages = kl_coef * (teacher - old)."""
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


def guided_opd_loss(model, train_example, *, pad_id, eos_id, temperature, clip_low, clip_high, clip_ratio_c,
                    kl_coef=1.0, loss_agg_mode="token-mean"):
    """DASH-OPD's GuidedOPDPolicyLossFn (algorithm/policy_loss_fn/guided_opd_policy_loss.py) in JAX.

    Student-generated turns get the dual-clip PPO loss of opd_loss; teacher-generated turns (`triggered`, from
    the workflow's is_teacher_turn) get the sampled forward KL kl_coef * -log p on the teacher's tokens. The
    release combines the two token-means weighted by their token counts = one token-mean over both.
    """
    mask = train_example.completion_mask.astype(jnp.float32)
    logps, entropy = completion_logps(model, train_example.prompt_ids, train_example.completion_ids,
                                      pad_id, eos_id, temperature, return_entropy=True)
    old = train_example.old_per_token_logps.astype(jnp.float32)
    adv = train_example.advantages.astype(jnp.float32)
    trig = (train_example.triggered > 0.5)[:, None]  # [B, 1] teacher turn

    log_ratio = jnp.clip(logps - old, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clipped = jnp.maximum(pg1, pg2)
    pg3 = -adv * clip_ratio_c
    pg_tok = jnp.where(adv < 0.0, jnp.minimum(clipped, pg3), clipped)
    sft_tok = kl_coef * (-logps)
    per_token = jnp.where(trig, sft_tok, pg_tok)
    loss = common.aggregate_loss(per_token, mask, loss_agg_mode)

    trig_mask = mask * trig
    pg_mask = mask * (~trig)
    denom = jnp.clip(mask.sum(), min=1.0)
    pg_denom = jnp.clip(pg_mask.sum(), min=1.0)
    aux = {
        "pg_clipfrac": ((pg2 > pg1) * pg_mask).sum() / pg_denom,
        "pg_clipfrac_lower": (((clipped > pg3) & (adv < 0.0)) * pg_mask).sum() / pg_denom,
        "ppo_kl": (-log_ratio * pg_mask).sum() / pg_denom,
        "entropy": (jax.lax.stop_gradient(entropy) * mask).sum() / denom,
        # same keys as uopd_loss so the worker's per-branch bookkeeping applies
        "uopd/sft_loss": jax.lax.stop_gradient((sft_tok * trig_mask).sum() / jnp.clip(trig_mask.sum(), min=1.0)),
        "uopd/pg_loss": jax.lax.stop_gradient((pg_tok * pg_mask).sum() / pg_denom),
    }
    return loss, aux


def uopd_loss(model, train_example, *, pad_id, eos_id, temperature, sft_coef=1.0,
              loss_agg_mode="token-mean"):
    """UOPD's UOPDPolicyLossFn (algorithm/policy_loss_fn/uopd_policy_loss.py) in JAX.

    Per row: student-executed turns get the OPD policy gradient -A * ratio (log-ratio clamped to
    +-20, no PPO clipping, as released); teacher-executed turns (`triggered`) get SFT on the
    teacher's tokens, sft_coef * -log p. Both branches go through one token-mean aggregation.
    """
    mask = train_example.completion_mask.astype(jnp.float32)
    logps, entropy = completion_logps(model, train_example.prompt_ids, train_example.completion_ids,
                                      pad_id, eos_id, temperature, return_entropy=True)
    old = train_example.old_per_token_logps.astype(jnp.float32)
    adv = train_example.advantages.astype(jnp.float32)
    trig = (train_example.triggered > 0.5)[:, None]  # [B, 1]

    log_ratio = jnp.clip(logps - old, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)
    pg_tok = -adv * ratio
    sft_tok = sft_coef * (-logps)
    per_token = jnp.where(trig, sft_tok, pg_tok)
    loss = common.aggregate_loss(per_token, mask, loss_agg_mode)

    trig_mask = mask * trig
    pg_mask = mask * (~trig)
    denom = jnp.clip(mask.sum(), min=1.0)
    pg_denom = jnp.clip(pg_mask.sum(), min=1.0)
    aux = {
        "pg_clipfrac": jnp.zeros(()),
        "pg_clipfrac_lower": jnp.zeros(()),
        "ppo_kl": (-log_ratio * pg_mask).sum() / pg_denom,
        "entropy": (jax.lax.stop_gradient(entropy) * mask).sum() / denom,
        "uopd/sft_loss": jax.lax.stop_gradient((sft_tok * trig_mask).sum() / jnp.clip(trig_mask.sum(), min=1.0)),
        "uopd/pg_loss": jax.lax.stop_gradient((pg_tok * pg_mask).sum() / pg_denom),
    }
    return loss, aux
