# -*- coding: utf-8 -*-
"""Role-aware reverse-KL advantage for Guided-OPD."""

from collections import defaultdict
from typing import Any, Dict, Tuple

import torch

try:
    from verl import DataProto
except ImportError:  # TPU env (trainer_type: tunix) has no verl; only used in type hints
    DataProto = Any

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn
from trinity.algorithm.advantage_fn.on_policy_distill_advantage import (
    _compute_opd_advantage,
    _parse_run_ids,
)


class GuidedOPDAdvantage(AdvantageFn):
    """Apply vanilla OPD's rKL advantage only to student-generated turns."""

    def __init__(self, kl_coef: float = 1.0) -> None:
        self.kl_coef = kl_coef

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        old_log_probs = exps.batch["old_log_probs"]
        teacher_log_probs = exps.batch["teacher_logprobs"]
        response_mask = exps.batch["response_mask"].bool()
        teacher_turn_mask = exps.batch["teacher_turn_mask"]

        if teacher_turn_mask.dtype != torch.bool:
            raise TypeError(
                "teacher_turn_mask must have dtype torch.bool, got " f"{teacher_turn_mask.dtype}"
            )
        if teacher_turn_mask.ndim != 1 or teacher_turn_mask.shape[0] != old_log_probs.shape[0]:
            raise ValueError(
                "teacher_turn_mask must have shape [batch_size], got "
                f"{tuple(teacher_turn_mask.shape)} for batch size {old_log_probs.shape[0]}"
            )
        teacher_turn_mask = teacher_turn_mask.to(device=response_mask.device)

        teacher_valid_mask = exps.batch.get("teacher_logprobs_valid_mask")
        if teacher_valid_mask is None:
            teacher_valid_mask = response_mask
        else:
            teacher_valid_mask = teacher_valid_mask.to(
                dtype=torch.bool, device=response_mask.device
            )
        student_row_mask = (~teacher_turn_mask).unsqueeze(-1)
        effective_student_mask = teacher_valid_mask & student_row_mask

        # Reuse the exact vanilla OPD computation and metric definitions.  The
        # role mask is supplied as an additional validity mask.
        advantages, metrics = _compute_opd_advantage(
            old_log_probs,
            teacher_log_probs,
            response_mask,
            effective_student_mask,
            self.kl_coef,
        )
        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        effective_student_mask = response_mask & effective_student_mask
        effective_teacher_mask = response_mask & teacher_turn_mask.unsqueeze(-1)
        student_tokens = effective_student_mask.sum()
        teacher_tokens = effective_teacher_mask.sum()
        student_token_count = int(student_tokens.item())
        teacher_token_count = int(teacher_tokens.item())
        if student_token_count:
            student_kl_mean = (
                ((old_log_probs - teacher_log_probs) * effective_student_mask).sum()
                / student_tokens
            ).item()
            student_advantage_mean = (
                (advantages * effective_student_mask).sum() / student_tokens
            ).item()
        else:
            student_kl_mean = 0.0
            student_advantage_mean = 0.0

        metrics.update(
            {
                "guided_opd/student_kl_mean": student_kl_mean,
                "guided_opd/student_advantage_mean": student_advantage_mean,
                "guided_opd/student_token_count": float(student_token_count),
                "guided_opd/teacher_token_count": float(teacher_token_count),
                "guided_opd/teacher_turn_fraction": teacher_turn_mask.float().mean().item(),
            }
        )

        unique_ids = exps.batch.get("unique_ids")
        if unique_ids is not None:
            run_ids = _parse_run_ids(unique_ids)
            kl_per_turn = ((old_log_probs - teacher_log_probs) * effective_student_mask).sum(dim=-1)
            trajectory_kl_sums: Dict[str, float] = defaultdict(float)
            for run_id, kl in zip(run_ids, kl_per_turn.tolist()):
                trajectory_kl_sums[run_id] += kl
            if trajectory_kl_sums:
                values = list(trajectory_kl_sums.values())
                mean = sum(values) / len(values)
                metrics["kl/trajectory_mean"] = mean
                metrics["kl/trajectory_std"] = (
                    (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
                    if len(values) > 1
                    else 0.0
                )

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {"kl_coef": 1.0}
