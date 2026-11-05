# -*- coding: utf-8 -*-
"""Trajectory-local controller for Drift-Aware Switching with Hysteresis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal

Role = Literal["student", "teacher"]


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


@dataclass(frozen=True)
class DASHControllerConfig:
    """Hyperparameters specific to the DASH rollout controller."""

    exact_mean_variance: bool = True
    ema_decay: float = 0.9
    ema_variance_floor: float = 0.01
    z_clip: float = 3.0
    evidence_decay: float = 0.9
    student_drift_slack: float = 0.25
    teacher_recovery_slack: float = 0.25
    takeover_threshold: float = 4.0
    return_threshold: float = 2.0
    min_teacher_span: int = 2
    teacher_budget_ratio: float = 0.30
    # Ablation gates. These take precedence over every threshold, fallback,
    # minimum-span, and budget criterion in their respective direction.
    teacher_takeover_enabled: bool = True
    student_return_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate all controller hyperparameters."""
        for name in (
            "teacher_takeover_enabled",
            "student_return_enabled",
            "exact_mean_variance",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")

        float_fields = (
            "ema_decay",
            "ema_variance_floor",
            "z_clip",
            "evidence_decay",
            "student_drift_slack",
            "teacher_recovery_slack",
            "takeover_threshold",
            "return_threshold",
            "teacher_budget_ratio",
        )
        for name in float_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            _require_finite(name, float(value))

        if not self.exact_mean_variance and not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                "ema_decay must be in [0, 1) when exact_mean_variance is false, "
                f"got {self.ema_decay}"
            )
        if self.ema_variance_floor <= 0.0:
            raise ValueError(f"ema_variance_floor must be positive, got {self.ema_variance_floor}")
        if self.z_clip <= 0.0:
            raise ValueError(f"z_clip must be positive, got {self.z_clip}")
        if not 0.0 <= self.evidence_decay <= 1.0:
            raise ValueError(f"evidence_decay must be in [0, 1], got {self.evidence_decay}")
        if self.student_drift_slack < 0.0:
            raise ValueError(
                f"student_drift_slack must be non-negative, got {self.student_drift_slack}"
            )
        if self.teacher_recovery_slack < 0.0:
            raise ValueError(
                "teacher_recovery_slack must be non-negative, got " f"{self.teacher_recovery_slack}"
            )
        if not self.takeover_threshold >= self.return_threshold > 0.0:
            raise ValueError(
                "DASH requires takeover_threshold >= return_threshold > 0, got "
                f"takeover_threshold={self.takeover_threshold}, "
                f"return_threshold={self.return_threshold}"
            )
        if isinstance(self.min_teacher_span, bool) or not isinstance(self.min_teacher_span, int):
            raise TypeError(
                f"min_teacher_span must be an integer, got {type(self.min_teacher_span).__name__}"
            )
        if self.min_teacher_span <= 0:
            raise ValueError(
                f"min_teacher_span must be a positive integer, got {self.min_teacher_span}"
            )
        if not 0.0 <= self.teacher_budget_ratio <= 1.0:
            raise ValueError(
                f"teacher_budget_ratio must be in [0, 1], got {self.teacher_budget_ratio}"
            )


@dataclass
class RunningStatsState:
    """One direction's trajectory-local normalization state."""

    initialized: bool = False
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0
    m2: float = 0.0

    def observe(self, value: float, config: DASHControllerConfig) -> float:
        """Standardize against prior observations, then incorporate ``value``."""
        if not self.initialized:
            self.initialized = True
            self.mean = value
            self.variance = (
                0.0
                if config.exact_mean_variance
                else max(value * value, config.ema_variance_floor)
            )
            self.count = 1
            return 0.0

        delta = value - self.mean
        z = delta / math.sqrt(max(self.variance, config.ema_variance_floor))
        z = max(-config.z_clip, min(config.z_clip, z))

        if config.exact_mean_variance:
            self.count += 1
            self.mean += delta / self.count
            delta_after_update = value - self.mean
            self.m2 += delta * delta_after_update
            self.variance = max(self.m2 / self.count, 0.0)
        else:
            self.mean = config.ema_decay * self.mean + (1.0 - config.ema_decay) * value
            self.variance = max(
                config.ema_decay * self.variance
                + (1.0 - config.ema_decay) * delta * delta,
                config.ema_variance_floor,
            )
            self.count += 1
        return z


@dataclass(frozen=True)
class DASHDecision:
    """Controller result after observing one completed turn."""

    mode_before: Role
    mode_after: Role
    directional_log_ratio: float
    standardized_residual: float
    support_debt: float
    recovery_credit: float
    switched_student_to_teacher: bool
    switched_teacher_to_student: bool
    forced_return_by_budget: bool
    teacher_span: int
    teacher_budget_used: int
    teacher_budget: int


@dataclass
class DASHController:
    """Path-dependent, bidirectional turn-role controller for one trajectory."""

    config: DASHControllerConfig
    max_turns: int
    is_eval: bool = False
    mode: Role = field(init=False, default="student")
    support_debt: float = field(init=False, default=0.0)
    recovery_credit: float = field(init=False, default=0.0)
    teacher_span: int = field(init=False, default=0)
    used_teacher_turns: int = field(init=False, default=0)
    switch_s_to_t_count: int = field(init=False, default=0)
    switch_t_to_s_count: int = field(init=False, default=0)
    teacher_segment_count: int = field(init=False, default=0)
    student_turns: int = field(init=False, default=0)
    teacher_turns: int = field(init=False, default=0)
    max_support_debt: float = field(init=False, default=0.0)
    max_recovery_credit: float = field(init=False, default=0.0)
    teacher_budget: int = field(init=False)
    student_normalizer: RunningStatsState = field(
        init=False, default_factory=RunningStatsState
    )
    teacher_normalizer: RunningStatsState = field(
        init=False, default_factory=RunningStatsState
    )
    last_observed_role: Role | None = field(init=False, default=None)
    _completed_teacher_segments: List[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Validate trajectory shape and derive the strict teacher budget."""
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int):
            raise TypeError(f"max_turns must be an integer, got {type(self.max_turns).__name__}")
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")
        self.teacher_budget = math.floor(self.config.teacher_budget_ratio * self.max_turns)

    @property
    def remaining_teacher_budget(self) -> int:
        """Return the number of teacher-generated turns still available."""
        return self.teacher_budget - self.used_teacher_turns

    def _validate_observation(self, value: float) -> float:
        value = float(value)
        _require_finite("directional_log_ratio", value)
        return value

    def _decision(
        self,
        *,
        mode_before: Role,
        directional_log_ratio: float,
        standardized_residual: float,
        support_debt: float,
        recovery_credit: float,
        switched_student_to_teacher: bool = False,
        switched_teacher_to_student: bool = False,
        forced_return_by_budget: bool = False,
        teacher_span: int | None = None,
    ) -> DASHDecision:
        return DASHDecision(
            mode_before=mode_before,
            mode_after=self.mode,
            directional_log_ratio=directional_log_ratio,
            standardized_residual=standardized_residual,
            support_debt=support_debt,
            recovery_credit=recovery_credit,
            switched_student_to_teacher=switched_student_to_teacher,
            switched_teacher_to_student=switched_teacher_to_student,
            forced_return_by_budget=forced_return_by_budget,
            teacher_span=self.teacher_span if teacher_span is None else teacher_span,
            teacher_budget_used=self.used_teacher_turns,
            teacher_budget=self.teacher_budget,
        )

    def start_with_teacher(self) -> bool:
        """Start the trajectory in teacher mode when the budget can fund a span."""
        if self.is_eval or self.mode != "student" or self.last_observed_role is not None:
            return False
        if self.remaining_teacher_budget < self.config.min_teacher_span:
            return False
        self.mode = "teacher"
        self.support_debt = 0.0
        self.recovery_credit = 0.0
        self.teacher_span = 0
        return True

    def observe_student(
        self,
        directional_log_ratio: float,
        *,
        takeover_override: bool = False,
    ) -> DASHDecision:
        """Consume student evidence and choose the next role.

        ``takeover_override`` lets an environment-specific, causal fallback
        request teacher takeover without fabricating support debt. The strict
        teacher budget and evaluation's student-only contract still apply.
        """
        if self.mode != "student":
            raise RuntimeError(f"observe_student called while DASH mode is {self.mode}")
        if not isinstance(takeover_override, bool):
            raise TypeError(
                "takeover_override must be a bool, got "
                f"{type(takeover_override).__name__}"
            )
        value = self._validate_observation(directional_log_ratio)
        if self.last_observed_role == "teacher":
            self.switch_t_to_s_count += 1
        self.last_observed_role = "student"
        self.student_turns += 1

        # Evaluation is always student-only and must not accumulate hidden state.
        if self.is_eval:
            return self._decision(
                mode_before="student",
                directional_log_ratio=value,
                standardized_residual=0.0,
                support_debt=self.support_debt,
                recovery_credit=self.recovery_credit,
            )

        z = self.student_normalizer.observe(value, self.config)
        debt = max(
            0.0,
            self.config.evidence_decay * self.support_debt + z - self.config.student_drift_slack,
        )
        self.support_debt = debt
        self.max_support_debt = max(self.max_support_debt, debt)

        switched = False
        if (
            self.config.teacher_takeover_enabled
            and (debt > self.config.takeover_threshold or takeover_override)
            and self.remaining_teacher_budget >= self.config.min_teacher_span
        ):
            self.mode = "teacher"
            self.support_debt = 0.0
            self.recovery_credit = 0.0
            self.teacher_span = 0
            switched = True

        return self._decision(
            mode_before="student",
            directional_log_ratio=value,
            standardized_residual=z,
            support_debt=debt,
            recovery_credit=self.recovery_credit,
            switched_student_to_teacher=switched,
        )

    def observe_teacher(self, directional_log_ratio: float) -> DASHDecision:
        """Consume teacher-sampled forward-KL evidence and choose the next role."""
        if self.mode != "teacher":
            raise RuntimeError(f"observe_teacher called while DASH mode is {self.mode}")
        value = self._validate_observation(directional_log_ratio)
        if self.last_observed_role == "student":
            self.switch_s_to_t_count += 1
        if self.last_observed_role != "teacher":
            self.teacher_segment_count += 1
        self.last_observed_role = "teacher"
        z = self.teacher_normalizer.observe(value, self.config)
        credit = max(
            0.0,
            self.config.evidence_decay * self.recovery_credit
            - z
            - self.config.teacher_recovery_slack,
        )
        self.recovery_credit = credit
        self.max_recovery_credit = max(self.max_recovery_credit, credit)
        self.teacher_span += 1
        self.teacher_turns += 1
        self.used_teacher_turns += 1

        forced = self.used_teacher_turns >= self.teacher_budget
        recovered = (
            self.teacher_span >= self.config.min_teacher_span
            and credit > self.config.return_threshold
        )
        switched = self.config.student_return_enabled and (forced or recovered)
        span = self.teacher_span
        if switched:
            self._completed_teacher_segments.append(span)
            self.mode = "student"
            self.support_debt = 0.0
            self.recovery_credit = 0.0
            self.teacher_span = 0

        return self._decision(
            mode_before="teacher",
            directional_log_ratio=value,
            standardized_residual=z,
            support_debt=self.support_debt,
            recovery_credit=credit,
            switched_teacher_to_student=switched,
            forced_return_by_budget=forced and switched,
            teacher_span=span,
        )

    def summary(self) -> Dict[str, float]:
        """Return numeric trajectory-level metrics without changing controller state."""
        segment_lengths = list(self._completed_teacher_segments)
        if self.mode == "teacher" and self.teacher_span > 0:
            segment_lengths.append(self.teacher_span)
        mean_segment_length = (
            sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0.0
        )
        max_segment_length = max(segment_lengths, default=0)
        total_turns = self.student_turns + self.teacher_turns
        return {
            "student_turns": float(self.student_turns),
            "teacher_turns": float(self.teacher_turns),
            "teacher_turn_fraction": self.teacher_turns / total_turns if total_turns else 0.0,
            "student_to_teacher_switches": float(self.switch_s_to_t_count),
            "teacher_to_student_switches": float(self.switch_t_to_s_count),
            "teacher_segments": float(self.teacher_segment_count),
            "mean_teacher_segment_length": float(mean_segment_length),
            "max_teacher_segment_length": float(max_segment_length),
            "teacher_budget_used": float(self.used_teacher_turns),
            "teacher_budget": float(self.teacher_budget),
            "teacher_budget_fraction": (
                self.used_teacher_turns / self.teacher_budget if self.teacher_budget > 0 else 0.0
            ),
            "teacher_budget_exhausted": float(self.used_teacher_turns >= self.teacher_budget),
            "ended_in_teacher_mode": float(self.last_observed_role == "teacher"),
            "max_support_debt": self.max_support_debt,
            "max_recovery_credit": self.max_recovery_credit,
        }
