# -*- coding: utf-8 -*-
"""Role-aware asymmetric KL loss for Guided-OPD."""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.policy_loss_fn.ppo_policy_loss import PPOPolicyLossFn
from trinity.algorithm.policy_loss_fn.sft_loss import SFTLossFn


class GuidedOPDPolicyLossFn(PolicyLossFn):
    """PPO/rKL on student turns and sampled fKL on teacher turns."""

    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = 0.2,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        clip_ratio_c: float = 3.0,
        loss_agg_mode: str = "token-mean",
        enable_sequence_masking: bool = False,
        delta_sequence_masking: float = 0.1,
        fallback_to_policy_gradient: bool = False,
        kl_coef: float = 1.0,
    ) -> None:
        super().__init__(backend=backend)
        if loss_agg_mode != "token-mean":
            raise ValueError(
                "GuidedOPDPolicyLossFn currently supports only token-mean, "
                f"matching the ALFWorld OPD baseline; got {loss_agg_mode!r}"
            )
        self.kl_coef = float(kl_coef)
        self.ppo_loss_fn = PPOPolicyLossFn(
            backend=backend,
            clip_range=clip_range,
            clip_range_low=clip_range_low,
            clip_range_high=clip_range_high,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=loss_agg_mode,
            enable_sequence_masking=enable_sequence_masking,
            delta_sequence_masking=delta_sequence_masking,
            fallback_to_policy_gradient=fallback_to_policy_gradient,
        )
        self.sft_loss_fn = SFTLossFn(backend=backend, loss_agg_mode=loss_agg_mode)

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        teacher_turn_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        if teacher_turn_mask.dtype != torch.bool:
            raise TypeError(
                "teacher_turn_mask must have dtype torch.bool, got " f"{teacher_turn_mask.dtype}"
            )
        if teacher_turn_mask.ndim != 1 or teacher_turn_mask.shape[0] != logprob.shape[0]:
            raise ValueError(
                "teacher_turn_mask must have shape [batch_size], got "
                f"{tuple(teacher_turn_mask.shape)} for batch size {logprob.shape[0]}"
            )
        teacher_turn_mask = teacher_turn_mask.to(device=logprob.device)
        student_turn_mask = ~teacher_turn_mask

        # A differentiable zero keeps empty branches on the correct device and
        # dtype, including the all-teacher first curriculum step.
        zero = logprob.sum() * 0.0
        student_metrics: Dict = {}
        teacher_metrics: Dict = {}

        if student_turn_mask.any():
            student_loss, student_metrics = self.ppo_loss_fn(
                logprob=logprob[student_turn_mask],
                old_logprob=old_logprob[student_turn_mask],
                action_mask=action_mask[student_turn_mask],
                advantages=advantages[student_turn_mask],
            )
        else:
            student_loss = zero

        if teacher_turn_mask.any():
            teacher_nll, teacher_metrics = self.sft_loss_fn(
                logprob=logprob[teacher_turn_mask],
                action_mask=action_mask[teacher_turn_mask],
            )
        else:
            teacher_nll = zero

        student_tokens = action_mask[student_turn_mask].sum().to(logprob.dtype)
        teacher_tokens = action_mask[teacher_turn_mask].sum().to(logprob.dtype)
        total_tokens = student_tokens + teacher_tokens
        loss = (
            student_loss * student_tokens + self.kl_coef * teacher_nll * teacher_tokens
        ) / total_tokens.clamp_min(1.0)

        metrics = {f"student/{key}": value for key, value in student_metrics.items()}
        metrics.update({f"teacher/{key}": value for key, value in teacher_metrics.items()})
        student_token_count = float(student_tokens.detach().item())
        teacher_token_count = float(teacher_tokens.detach().item())
        total_token_count = student_token_count + teacher_token_count
        metrics.update(
            {
                "guided_opd/loss": loss.detach().item(),
                "guided_opd/student_ppo_loss": student_loss.detach().item(),
                "guided_opd/teacher_fkl_loss": (self.kl_coef * teacher_nll).detach().item(),
                "guided_opd/student_tokens": student_token_count,
                "guided_opd/teacher_tokens": teacher_token_count,
                "guided_opd/student_token_fraction": (
                    student_token_count / total_token_count if total_token_count else 0.0
                ),
                "guided_opd/teacher_token_fraction": (
                    teacher_token_count / total_token_count if total_token_count else 0.0
                ),
            }
        )
        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
            "clip_ratio_c": 3.0,
            "loss_agg_mode": "token-mean",
            "enable_sequence_masking": False,
            "delta_sequence_masking": 0.1,
            "fallback_to_policy_gradient": False,
            "kl_coef": 1.0,
        }
