"""The tunix trainer's JAX advantage + PPO loss must match Trinity's torch versions.

Compares trinity.trainer.tunix.logps.opd_loss (with the model forward stubbed out)
against PPOPolicyLossFn and _compute_opd_advantage on identical random tensors.
Run: python tests/tpu/test_tunix_loss_parity.py   (CPU is fine: JAX_PLATFORMS=cpu)
"""
import jax.numpy as jnp
import numpy as np
import torch

import trinity.trainer.tunix.logps as L
from trinity.algorithm.advantage_fn.on_policy_distill_advantage import _compute_opd_advantage
from trinity.algorithm.policy_loss_fn.ppo_policy_loss import PPOPolicyLossFn


class Ex:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    for noise in (0.4, 1.5):  # 1.5 pushes ratios past clip_ratio_c, exercising the dual clip
        check(noise)
    print("PARITY OK")


def check(noise):
    rng = np.random.default_rng(0)
    B, C = 6, 40
    mask = rng.random((B, C)) < 0.8
    mask[0] = False  # an all-masked row
    old = -rng.random((B, C)).astype(np.float32) * 3
    teacher = -rng.random((B, C)).astype(np.float32) * 3
    new = (old + rng.normal(0, noise, (B, C))).astype(np.float32)  # ratios well outside the clip range

    # advantage
    adv_t, m_t = _compute_opd_advantage(torch.tensor(old), torch.tensor(teacher), torch.tensor(mask),
                                        torch.tensor(mask), kl_coef=1.0)
    adv_j = 1.0 * (teacher - old) * mask
    assert np.allclose(adv_t.numpy(), adv_j, atol=1e-6), "advantage mismatch"
    kl_sum = ((old - teacher) * mask).sum(-1)
    assert np.isclose(m_t["kl/mean"], kl_sum.mean(), atol=1e-5)

    # policy loss (forward stubbed to return `new` logprobs)
    L.completion_logps = lambda *a, **k: (jnp.asarray(new), jnp.zeros((B, C)))
    ex = Ex(prompt_ids=None, completion_ids=jnp.zeros((B, C), jnp.int32), completion_mask=jnp.asarray(mask),
            old_per_token_logps=jnp.asarray(old), advantages=jnp.asarray(adv_j))
    loss_j, aux = L.opd_loss(None, ex, pad_id=0, eos_id=0, temperature=1.0, clip_low=0.2, clip_high=0.2,
                             clip_ratio_c=3.0, loss_agg_mode="token-mean")
    fn = PPOPolicyLossFn(backend="verl", clip_range=0.2, clip_ratio_c=3.0, loss_agg_mode="token-mean")
    loss_t, m = fn(logprob=torch.tensor(new), old_logprob=torch.tensor(old), action_mask=torch.tensor(mask),
                   advantages=adv_t)
    print("loss  torch", float(loss_t), " jax", float(loss_j))
    for k_t, k_j in [("pg_clipfrac", "pg_clipfrac"), ("ppo_kl", "ppo_kl"), ("pg_clipfrac_lower", "pg_clipfrac_lower")]:
        print(f"{k_j:18s} torch {m[k_t]:.6f}  jax {float(aux[k_j]):.6f}")
        assert abs(m[k_t] - float(aux[k_j])) < 1e-5, k_j
    assert abs(float(loss_t) - float(loss_j)) < 1e-5, "loss mismatch"


if __name__ == "__main__":
    main()
