# -*- coding: utf-8 -*-
"""DASH-OPD rollout switching for ALFWorld.

Only the turn generator selection differs from Guided-OPD. Student turns keep
the existing reverse-KL/PPO path and teacher turns keep its sampled forward-KL
path through the shared Guided-OPD advantage and policy loss. Paper-facing
trajectory metrics reuse log-probabilities and role decisions already required
by DASH, so they add no model calls and only linear-time scalar bookkeeping.
"""

import hashlib
import json
import random
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import torch

from trinity.common.experience import CustomField, Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.DASH.alfworld.OPD_workflow import (
    OnPolicyDistillVerlAgentAlfworldWorkflow,
)
from trinity.common.workflows.envs.DASH.alfworld.utils import (
    _create_alfworld_env,
    _extract_task,
    _format_history,
    build_alfworld_prompt,
    format_observation,
    parse_action,
    parse_explorer_schedule_step,
    parse_think,
)
from trinity.common.workflows.envs.TCOD.logprob_utils import fill_unscored  # TPU, see _score_turn
from trinity.common.workflows.envs.DASH.dash_opd_controller import (
    DASHController,
    DASHControllerConfig,
    DASHDecision,
)
from trinity.common.workflows.envs.DASH.segment_kl import (
    build_action_token_mask,
    calculate_action_kl_metrics,
    calculate_first_action_kl_metrics,
    calculate_pre_action_think_kl_metrics,
    calculate_think_kl_metrics,
    get_segment_tokenizer,
)

_SUPPORT_DEBT_MONITOR_THRESHOLDS = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
_RECOVERY_CREDIT_MONITOR_THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
_SLACK_MONITOR_THRESHOLDS = (0.0, 0.15, 0.25, 0.35, 0.5)


def linear_teacher_start_probability(schedule_step: int, total_steps: int) -> float:
    """Decay the teacher-start probability from one to zero over training."""
    if isinstance(schedule_step, bool) or not isinstance(schedule_step, int):
        raise TypeError(
            "schedule_step must be an integer, got "
            f"{type(schedule_step).__name__}"
        )
    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TypeError(
            f"total_steps must be an integer, got {type(total_steps).__name__}"
        )
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    progress = min(max(schedule_step, 0), total_steps) / total_steps
    return 1.0 - progress


def _normalize_stagnation_value(value: str) -> str:
    """Normalize harmless text differences while retaining exact semantics."""
    return " ".join(value.split()).casefold()


class _ConsecutiveStagnationDetector:
    """Detect exact action or observation repetition across student turns."""

    def __init__(self, repeat_turns: int):
        if (
            isinstance(repeat_turns, bool)
            or not isinstance(repeat_turns, int)
            or repeat_turns < 2
        ):
            raise ValueError(
                "stagnation_repeat_turns must be an integer >= 2, got "
                f"{repeat_turns!r}"
            )
        self.repeat_turns = repeat_turns
        self._last_action: Optional[str] = None
        self._last_observation: Optional[str] = None
        self.action_streak = 0
        self.observation_streak = 0

    @staticmethod
    def _next_streak(value: str, previous: Optional[str], streak: int) -> int:
        return streak + 1 if value == previous else 1

    def observe(
        self,
        *,
        action: str,
        observation: str,
        is_teacher_turn: bool,
    ) -> Tuple[bool, bool]:
        """Return action/observation stagnation flags for the completed turn."""
        if is_teacher_turn:
            self._last_action = None
            self._last_observation = None
            self.action_streak = 0
            self.observation_streak = 0
            return False, False

        normalized_action = _normalize_stagnation_value(action)
        normalized_observation = _normalize_stagnation_value(observation)
        self.action_streak = self._next_streak(
            normalized_action, self._last_action, self.action_streak
        )
        self.observation_streak = self._next_streak(
            normalized_observation,
            self._last_observation,
            self.observation_streak,
        )
        self._last_action = normalized_action
        self._last_observation = normalized_observation
        return (
            self.action_streak >= self.repeat_turns,
            self.observation_streak >= self.repeat_turns,
        )


def _ols_slope(points: Sequence[Tuple[int, float]]) -> Optional[float]:
    """Return the ordinary-least-squares slope, or ``None`` if undefined."""
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0.0:
        return None
    numerator = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    )
    return numerator / denominator


def _threshold_label(value: float) -> str:
    """Return a stable W&B-safe label such as ``2p5`` for a threshold."""
    return f"{value:g}".replace(".", "p")


def _trajectory_threshold_distribution_metrics(
    *,
    max_support_debt: float,
    max_recovery_credit: float,
    has_teacher_segment: bool,
    takeover_threshold: float,
    return_threshold: float,
) -> dict:
    """Build an empirical tail distribution with constant-time scalar comparisons.

    Support-debt tails use every training trajectory. Recovery-credit tails are
    emitted only for trajectories that actually entered teacher mode, so their
    W&B means are conditional on having recovery evidence instead of being
    diluted by student-only trajectories.
    """
    support_thresholds = sorted(
        {*_SUPPORT_DEBT_MONITOR_THRESHOLDS, float(takeover_threshold)}
    )
    recovery_thresholds = sorted(
        {*_RECOVERY_CREDIT_MONITOR_THRESHOLDS, float(return_threshold)}
    )
    metrics = {
        "support_debt_tail/eligible": 1.0,
        "recovery_credit_tail/eligible": float(has_teacher_segment),
    }
    metrics.update(
        {
            f"support_debt_tail/fraction_gt_{_threshold_label(threshold)}": float(
                max_support_debt > threshold
            )
            for threshold in support_thresholds
        }
    )
    if has_teacher_segment:
        metrics.update(
            {
                f"recovery_credit_tail/fraction_gt_{_threshold_label(threshold)}": float(
                    max_recovery_credit > threshold
                )
                for threshold in recovery_thresholds
            }
        )
    return metrics


def _lag1_autocorrelation(values: Sequence[float]) -> Optional[float]:
    """Return lag-1 correlation, or ``None`` for short/constant sequences."""
    if len(values) < 3:
        return None
    left = values[:-1]
    right = values[1:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    cross = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    return cross / (left_ss * right_ss) ** 0.5


def _signal_diagnostics(
    *,
    decisions: Sequence[DASHDecision],
    teacher_roles: Sequence[bool],
    student_drift_slack: float,
    teacher_recovery_slack: float,
    z_clip: float,
) -> dict:
    """Summarize slack coverage and EMA residual behavior from existing decisions.

    The first observation for each role is excluded because EMA initialization
    structurally returns zero rather than a measured standardized residual.
    Teacher recovery uses ``-z`` so positive values consistently mean evidence
    in favor of the corresponding accumulator.
    """
    role_signals = {
        "student_drift": [
            decision.standardized_residual
            for decision, is_teacher in zip(decisions, teacher_roles)
            if not is_teacher
        ],
        "teacher_recovery": [
            -decision.standardized_residual
            for decision, is_teacher in zip(decisions, teacher_roles)
            if is_teacher
        ],
    }
    role_slacks = {
        "student_drift": float(student_drift_slack),
        "teacher_recovery": float(teacher_recovery_slack),
    }
    metrics = {}
    for role_name, initialized_signals in role_signals.items():
        signals = initialized_signals[1:]
        tail_prefix = f"{role_name}_slack_tail"
        metrics[f"{tail_prefix}/eligible"] = float(bool(signals))
        metrics[f"{tail_prefix}/eligible_observations"] = float(len(signals))
        if not signals:
            continue

        thresholds = sorted(
            {*_SLACK_MONITOR_THRESHOLDS, role_slacks[role_name]}
        )
        metrics.update(
            {
                f"{tail_prefix}/fraction_gt_{_threshold_label(threshold)}": (
                    sum(signal > threshold for signal in signals) / len(signals)
                )
                for threshold in thresholds
            }
        )
        metrics[f"{role_name}_signal_mean"] = sum(signals) / len(signals)
        metrics[f"{role_name}_signal_abs_mean"] = (
            sum(abs(signal) for signal in signals) / len(signals)
        )
        metrics[f"{role_name}_signal_positive_fraction"] = (
            sum(signal > 0.0 for signal in signals) / len(signals)
        )
        metrics[f"{role_name}_signal_clip_fraction"] = (
            sum(abs(signal) >= z_clip for signal in signals) / len(signals)
        )
        lag1 = _lag1_autocorrelation(signals)
        metrics[f"{role_name}_signal_lag1_autocorrelation_valid"] = float(
            lag1 is not None
        )
        if lag1 is not None:
            metrics[f"{role_name}_signal_lag1_autocorrelation"] = lag1
    return metrics


def _handoff_metrics(
    *,
    teacher_roles: Sequence[bool],
    decisions: Sequence[DASHDecision],
    episode_succeeded: bool,
) -> dict:
    """Summarize only handoffs that were followed by an executed student turn."""
    handoff_segment_lengths: List[int] = []
    recovery_segment_lengths: List[int] = []
    post_handoff_student_spans: List[int] = []
    survived_three = 0
    survived_five = 0
    retained_successes = 0
    retakeovers = 0
    budget_forced_handoffs = 0

    turn_count = len(teacher_roles)
    index = 0
    while index < turn_count:
        if not teacher_roles[index]:
            index += 1
            continue

        segment_start = index
        while index + 1 < turn_count and teacher_roles[index + 1]:
            index += 1
        segment_end = index

        # A return decision on the last trajectory turn is not an actual handoff.
        if segment_end + 1 < turn_count and not teacher_roles[segment_end + 1]:
            segment_length = segment_end - segment_start + 1
            handoff_segment_lengths.append(segment_length)
            forced = decisions[segment_end].forced_return_by_budget
            if forced:
                budget_forced_handoffs += 1
            else:
                recovery_segment_lengths.append(segment_length)

            student_span = 0
            cursor = segment_end + 1
            while cursor < turn_count and not teacher_roles[cursor]:
                student_span += 1
                cursor += 1
            post_handoff_student_spans.append(student_span)
            was_retaken = cursor < turn_count
            retakeovers += int(was_retaken)
            completed_successfully = episode_succeeded and not was_retaken
            retained_successes += int(completed_successfully)
            survived_three += int(student_span >= 3 or completed_successfully)
            survived_five += int(student_span >= 5 or completed_successfully)
        index += 1

    handoff_count = len(handoff_segment_lengths)
    metrics = {
        "actual_handoff_count": float(handoff_count),
        "recovery_handoff_count": float(len(recovery_segment_lengths)),
        "budget_forced_handoff_count": float(budget_forced_handoffs),
    }
    if handoff_count:
        metrics.update(
            {
                "mean_completed_teacher_segment_length": (
                    sum(handoff_segment_lengths) / handoff_count
                ),
                "mean_post_handoff_student_span": (
                    sum(post_handoff_student_spans) / handoff_count
                ),
                "handoff_survival_at_3": survived_three / handoff_count,
                "handoff_survival_at_5": survived_five / handoff_count,
                "handoff_retakeover_rate": retakeovers / handoff_count,
                "handoff_success_retention_rate": retained_successes / handoff_count,
                "recovery_handoff_fraction": len(recovery_segment_lengths) / handoff_count,
            }
        )
    if recovery_segment_lengths:
        metrics["mean_teacher_recovery_interval_length"] = (
            sum(recovery_segment_lengths) / len(recovery_segment_lengths)
        )
    return metrics


@WORKFLOWS.register_module("DASH_OPD_alfworld_workflow")
class DASHOPDAlfworldWorkflow(OnPolicyDistillVerlAgentAlfworldWorkflow):
    """ALFWorld OPD with path-dependent bidirectional rollout switching."""

    _GUIDED_ONLY_ARGS = {
        "beta_start",
        "beta_end",
        "curriculum_ratio",
        "guidance_schedule",
        "guidance_seed",
    }

    def __init__(self, *, task, model, auxiliary_models=None):
        """Initialize common OPD behavior and DASH-only controller settings."""
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        args = task.workflow_args
        conflicting_args = sorted(self._GUIDED_ONLY_ARGS.intersection(args))
        if conflicting_args:
            raise ValueError(
                "DASH-OPD does not use Guided-OPD scheduling arguments: "
                + ", ".join(conflicting_args)
            )
        self.controller_config = DASHControllerConfig(
            teacher_takeover_enabled=args.get("teacher_takeover_enabled", True),
            student_return_enabled=args.get("student_return_enabled", True),
            exact_mean_variance=args.get("exact_mean_variance", True),
            ema_decay=args.get("ema_decay", 0.9),
            ema_variance_floor=args.get("ema_variance_floor", 0.01),
            z_clip=args.get("z_clip", 3.0),
            evidence_decay=args.get("evidence_decay", 0.9),
            student_drift_slack=args.get("student_drift_slack", 0.25),
            teacher_recovery_slack=args.get("teacher_recovery_slack", 0.25),
            takeover_threshold=args.get("takeover_threshold", 4.0),
            return_threshold=args.get("return_threshold", 2.0),
            min_teacher_span=args.get("min_teacher_span", 2),
            teacher_budget_ratio=args.get("teacher_budget_ratio", 0.30),
        )
        self.action_only_role_switching = args.get(
            "action_only_role_switching", False
        )
        if not isinstance(self.action_only_role_switching, bool):
            raise TypeError(
                "action_only_role_switching must be a bool, got "
                f"{type(self.action_only_role_switching).__name__}"
            )
        self.stagnation_takeover_enabled = args.get(
            "stagnation_takeover_enabled", True
        )
        if not isinstance(self.stagnation_takeover_enabled, bool):
            raise TypeError(
                "stagnation_takeover_enabled must be a bool, got "
                f"{type(self.stagnation_takeover_enabled).__name__}"
            )
        self.stagnation_repeat_turns = args.get("stagnation_repeat_turns", 3)
        if (
            isinstance(self.stagnation_repeat_turns, bool)
            or not isinstance(self.stagnation_repeat_turns, int)
            or self.stagnation_repeat_turns < 2
        ):
            raise ValueError(
                "stagnation_repeat_turns must be an integer >= 2, got "
                f"{self.stagnation_repeat_turns!r}"
            )
        self.teacher_start_schedule_enabled = args.get(
            "teacher_start_schedule_enabled", True
        )
        if not isinstance(self.teacher_start_schedule_enabled, bool):
            raise TypeError(
                "teacher_start_schedule_enabled must be a bool, got "
                f"{type(self.teacher_start_schedule_enabled).__name__}"
            )
        self.total_steps = args.get("total_steps", 250)
        if (
            isinstance(self.total_steps, bool)
            or not isinstance(self.total_steps, int)
            or self.total_steps <= 0
        ):
            raise ValueError(
                f"total_steps must be a positive integer, got {self.total_steps!r}"
            )
        self.teacher_start_seed = args.get("teacher_start_seed", 42)
        if isinstance(self.teacher_start_seed, bool) or not isinstance(
            self.teacher_start_seed, int
        ):
            raise TypeError(
                "teacher_start_seed must be an integer, got "
                f"{type(self.teacher_start_seed).__name__}"
            )

    def _teacher_start_state(self) -> Tuple[int, float, bool]:
        model_version = getattr(self.task, "rollout_model_version", None)
        parse_failed = (
            isinstance(model_version, bool) or not isinstance(model_version, int)
        )
        if parse_failed:
            schedule_step, _ = parse_explorer_schedule_step(self.task.batch_id)
        else:
            schedule_step = max(model_version, 0)
        probability = (
            0.0
            if self.task.is_eval or not self.teacher_start_schedule_enabled
            else linear_teacher_start_probability(schedule_step, self.total_steps)
        )
        return schedule_step, probability, parse_failed

    def _teacher_start_rng(self) -> random.Random:
        index = json.dumps(
            self.task.index,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        seed_material = "|".join(
            (
                str(self.teacher_start_seed),
                str(getattr(self.task, "rollout_model_version", 0)),
                str(self.task.batch_id),
                index,
                str(self.task.task_id),
                str(getattr(self, "run_id_base", 0)),
            )
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        return random.Random(seed)

    async def run_async(self) -> List[Experience]:
        """Run one ALFWorld episode and always close its environment."""
        env = _create_alfworld_env(self.task_desc)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    @staticmethod
    def _validate_generation_logprobs(response: Experience, *, role: str, turn_index: int) -> int:
        response_token_count = len(response.tokens) - response.prompt_length
        if response.logprobs is None or len(response.logprobs) == 0:
            raise RuntimeError(
                "DASHOPDAlfworldWorkflow requires generation logprobs: "
                f"role={role}, turn={turn_index}, response_tokens={response_token_count}. "
                "Set rollout_args.logprobs (e.g. 0) in task config."
            )
        if len(response.logprobs) != response_token_count or response_token_count <= 0:
            raise RuntimeError(
                "DASH-OPD generation token alignment failed: "
                f"role={role}, turn={turn_index}, logprobs={len(response.logprobs)}, "
                f"response_tokens={response_token_count}, tokens={len(response.tokens)}, "
                f"prompt_length={response.prompt_length}"
            )
        return response_token_count

    async def _score_turn(
        self,
        response: Experience,
        *,
        is_teacher_turn: bool,
        turn_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        role = "teacher" if is_teacher_turn else "student"
        response_token_count = self._validate_generation_logprobs(
            response, role=role, turn_index=turn_index
        )

        if is_teacher_turn:
            teacher_resp_logprobs = response.logprobs.clone()
            student_full_logprobs = await self.model.logprobs_async(
                tokens=response.tokens.tolist(), temperature=self.temperature
            )
            student_resp_logprobs = student_full_logprobs[response.prompt_length - 1 :]
        else:
            student_resp_logprobs = response.logprobs
            teacher_full_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(), temperature=self.temperature
            )
            teacher_resp_logprobs = teacher_full_logprobs[response.prompt_length - 1 :]

        if not (len(student_resp_logprobs) == len(teacher_resp_logprobs) == response_token_count):
            raise RuntimeError(
                "DASH-OPD cross-model token alignment failed: "
                f"role={role}, turn={turn_index}, "
                f"student_logprobs={len(student_resp_logprobs)}, "
                f"teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"response_tokens={response_token_count}, tokens={len(response.tokens)}, "
                f"prompt_length={response.prompt_length}"
            )

        # TPU (not in the release): vLLM-TPU returns NaN for prompt tokens it could not score (chunked-prefill
        # boundaries); take the generating model's logprob there (zero log-ratio) instead of the error below.
        if is_teacher_turn:
            student_resp_logprobs = fill_unscored(student_resp_logprobs, teacher_resp_logprobs)
        else:
            teacher_resp_logprobs = fill_unscored(teacher_resp_logprobs, student_resp_logprobs)
        response.logprobs = student_resp_logprobs
        response.teacher_logprobs = teacher_resp_logprobs
        directional_differences = (
            teacher_resp_logprobs - student_resp_logprobs
            if is_teacher_turn
            else student_resp_logprobs - teacher_resp_logprobs
        )
        if not torch.isfinite(directional_differences).all():
            raise RuntimeError(
                "DASH-OPD computed a non-finite directional log-ratio: "
                f"role={role}, turn={turn_index}"
            )
        valid_response_mask = response.action_mask
        if valid_response_mask is None:
            valid_response_mask = torch.ones_like(directional_differences, dtype=torch.bool)
        else:
            valid_response_mask = valid_response_mask.to(
                device=directional_differences.device, dtype=torch.bool
            )
        if valid_response_mask.ndim != 1 or len(valid_response_mask) != response_token_count:
            raise RuntimeError(
                "DASH-OPD response mask alignment failed: "
                f"role={role}, turn={turn_index}, mask={tuple(valid_response_mask.shape)}, "
                f"response_tokens={response_token_count}"
            )
        if not valid_response_mask.any():
            raise RuntimeError(
                "DASH-OPD cannot update its controller without a valid response token: "
                f"role={role}, turn={turn_index}"
            )
        controller_mask = valid_response_mask
        if self.action_only_role_switching:
            response_token_ids = response.tokens[-response_token_count:].tolist()
            segment_tokenizer = get_segment_tokenizer(
                self.model.config.model_path
            )
            action_token_mask = build_action_token_mask(
                response.response_text or "",
                response_token_ids,
                segment_tokenizer,
            )
            if (
                action_token_mask is None
                or action_token_mask.ndim != 1
                or len(action_token_mask) != response_token_count
            ):
                raise RuntimeError(
                    "DASH-OPD action-only role switching requires at least one "
                    "complete, token-aligned <action>...</action> block: "
                    f"role={role}, turn={turn_index}, "
                    f"response_tokens={response_token_count}"
                )
            controller_mask = controller_mask & action_token_mask.to(
                device=directional_differences.device,
                dtype=torch.bool,
            )
            if not controller_mask.any():
                raise RuntimeError(
                    "DASH-OPD action-only role switching found no action token "
                    "inside the valid response mask: "
                    f"role={role}, turn={turn_index}"
                )
        directional_log_ratio = (
            directional_differences.float()[controller_mask].mean().item()
        )
        return student_resp_logprobs, teacher_resp_logprobs, directional_log_ratio

    @staticmethod
    def _decision_metrics(decision: DASHDecision, *, is_teacher_turn: bool) -> dict:
        budget_fraction = (
            decision.teacher_budget_used / decision.teacher_budget
            if decision.teacher_budget > 0
            else 0.0
        )
        return {
            "dash_opd/is_teacher_turn": float(is_teacher_turn),
            "dash_opd/mode_before_teacher": float(decision.mode_before == "teacher"),
            "dash_opd/mode_after_teacher": float(decision.mode_after == "teacher"),
            "dash_opd/directional_log_ratio": decision.directional_log_ratio,
            "dash_opd/standardized_residual": decision.standardized_residual,
            "dash_opd/support_debt": decision.support_debt,
            "dash_opd/recovery_credit": decision.recovery_credit,
            "dash_opd/switched_student_to_teacher": float(decision.switched_student_to_teacher),
            "dash_opd/switched_teacher_to_student": float(decision.switched_teacher_to_student),
            "dash_opd/forced_return_by_budget": float(decision.forced_return_by_budget),
            "dash_opd/teacher_span": float(decision.teacher_span),
            "dash_opd/teacher_budget_used_so_far": float(decision.teacher_budget_used),
            "dash_opd/teacher_budget": float(decision.teacher_budget),
            "dash_opd/teacher_budget_fraction_so_far": budget_fraction,
        }

    async def _run_episode(self, env) -> List[Experience]:  # noqa: C901
        rollout_start_time = perf_counter()
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []
        controller = DASHController(
            self.controller_config,
            max_turns=self.max_env_steps,
            is_eval=self.task.is_eval,
        )
        teacher_start_step, teacher_start_probability, teacher_start_parse_failed = (
            self._teacher_start_state()
        )
        if teacher_start_parse_failed and not self.task.is_eval:
            self.logger.warning(
                "DASH teacher-start model version is unavailable; "
                "falling back to batch_id=%r.",
                self.task.batch_id,
            )
        teacher_start_requested = (
            teacher_start_probability >= 1.0
            or (
                teacher_start_probability > 0.0
                and self._teacher_start_rng().random() < teacher_start_probability
            )
        )
        teacher_started = teacher_start_requested and controller.start_with_teacher()
        teacher_start_blocked_by_budget = teacher_start_requested and not teacher_started
        stagnation_detector = _ConsecutiveStagnationDetector(
            self.stagnation_repeat_turns
        )

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0
        role_field = CustomField(
            source_field="is_teacher_turn",
            destination_field="teacher_turn_mask",
            data_type=torch.bool,
        )
        per_turn_kl_sums: List[float] = []
        total_response_tokens = 0
        teacher_roles: List[bool] = []
        decisions: List[DASHDecision] = []
        standard_log_ratio_means: List[float] = []
        valid_token_counts: List[int] = []
        student_entropy_points: List[Tuple[int, float]] = []
        student_self_entropy_sum = 0.0
        teacher_cross_entropy_on_student_sum = 0.0
        teacher_self_entropy_sum = 0.0
        student_cross_entropy_on_teacher_sum = 0.0
        student_entropy_at_debt_takeover_decision: List[float] = []
        student_entropy_at_stagnation_takeover_decision: List[float] = []
        teacher_entropy_at_recovery_return_decision: List[float] = []
        teacher_entropy_at_budget_return_decision: List[float] = []
        student_entropy_after_handoff: List[float] = []
        student_generated_tokens = 0
        teacher_generated_tokens = 0
        teacher_scored_sequence_tokens = 0
        teacher_generation_prompt_tokens = 0
        teacher_generation_seconds = 0.0
        teacher_scoring_seconds = 0.0
        stagnation_event_count = 0
        stagnation_takeover_decision_count = 0
        stagnation_takeover_count = 0
        stagnation_budget_block_count = 0
        pending_stagnation_takeover = False

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            reformatted_admissible = "\n ".join(
                f"'{command}'" for command in admissible_commands if command != "help"
            )

            user_content = build_alfworld_prompt(
                task_description=task_description,
                step_count=r,
                history=history,
                history_length=self.history_length,
                initial_no_history_turns=self.initial_no_history_turns,
                current_observation=format_obs,
                admissible_actions=reformatted_admissible,
                use_initial_no_history_template=self.retain_full_chat_history,
            )
            if self.retain_full_chat_history:
                memory = memory + [{"role": "user", "content": user_content}]
                model_messages = memory
            else:
                model_messages = [{"role": "user", "content": user_content}]

            is_teacher_turn = controller.mode == "teacher"
            if is_teacher_turn and pending_stagnation_takeover:
                stagnation_takeover_count += 1
                pending_stagnation_takeover = False
            generator = self.teacher_model if is_teacher_turn else self.model
            generation_start_time = perf_counter()
            responses = await generator.chat_async(model_messages, **kwargs)
            generation_seconds = perf_counter() - generation_start_time
            response = responses[0]
            response_text = response.response_text or ""
            if self.retain_full_chat_history:
                memory.append({"role": "assistant", "content": response_text})

            scoring_start_time = perf_counter()
            student_logprobs, teacher_logprobs, directional_log_ratio = (
                await self._score_turn(
                    response,
                    is_teacher_turn=is_teacher_turn,
                    turn_index=r,
                )
            )
            scoring_seconds = perf_counter() - scoring_start_time
            action = parse_action(response_text)
            action_stagnation, observation_stagnation = stagnation_detector.observe(
                action=action,
                observation=format_obs,
                is_teacher_turn=is_teacher_turn,
            )
            stagnation_detected = (
                self.stagnation_takeover_enabled
                and self.controller_config.teacher_takeover_enabled
                and not self.task.is_eval
                and not is_teacher_turn
                and (action_stagnation or observation_stagnation)
            )
            decision = (
                controller.observe_teacher(directional_log_ratio)
                if is_teacher_turn
                else controller.observe_student(
                    directional_log_ratio,
                    takeover_override=stagnation_detected,
                )
            )
            threshold_takeover_requested = (
                not is_teacher_turn
                and decision.support_debt > self.controller_config.takeover_threshold
            )
            stagnation_fallback_requested = (
                stagnation_detected and not threshold_takeover_requested
            )
            stagnation_takeover_decided = (
                stagnation_fallback_requested
                and decision.switched_student_to_teacher
            )
            stagnation_blocked_by_budget = (
                stagnation_fallback_requested
                and not decision.switched_student_to_teacher
            )
            stagnation_event_count += int(stagnation_detected)
            stagnation_takeover_decision_count += int(stagnation_takeover_decided)
            stagnation_budget_block_count += int(stagnation_blocked_by_budget)
            pending_stagnation_takeover = stagnation_takeover_decided

            response.info["is_teacher_turn"] = bool(is_teacher_turn)
            response.custom_fields = [role_field]
            response.metrics.update(
                self._decision_metrics(decision, is_teacher_turn=is_teacher_turn)
            )
            response.metrics.update(
                {
                    "dash_opd/stagnation_action_repeat_streak": float(
                        stagnation_detector.action_streak
                    ),
                    "dash_opd/stagnation_observation_repeat_streak": float(
                        stagnation_detector.observation_streak
                    ),
                    "dash_opd/stagnation_action_detected": float(
                        self.stagnation_takeover_enabled and action_stagnation
                    ),
                    "dash_opd/stagnation_observation_detected": float(
                        self.stagnation_takeover_enabled and observation_stagnation
                    ),
                    "dash_opd/stagnation_detected": float(stagnation_detected),
                    "dash_opd/stagnation_fallback_requested": float(
                        stagnation_fallback_requested
                    ),
                    "dash_opd/stagnation_takeover_decided": float(
                        stagnation_takeover_decided
                    ),
                    "dash_opd/stagnation_takeover_blocked_by_budget": float(
                        stagnation_blocked_by_budget
                    ),
                    "dash_opd/action_only_role_switching": float(
                        self.action_only_role_switching
                    ),
                }
            )
            turn_responses.append(response)

            response_token_count = len(response.tokens) - response.prompt_length
            valid_response_mask = response.action_mask
            if valid_response_mask is None:
                valid_response_mask = torch.ones_like(student_logprobs, dtype=torch.bool)
            else:
                valid_response_mask = valid_response_mask.to(
                    device=student_logprobs.device, dtype=torch.bool
                )
            standard_log_ratio = (
                (student_logprobs - teacher_logprobs)
                .float()[valid_response_mask]
                .mean()
                .item()
            )
            valid_student_logprobs = student_logprobs.float()[valid_response_mask]
            valid_teacher_logprobs = teacher_logprobs.float()[valid_response_mask]
            student_sampled_entropy = -valid_student_logprobs.mean().item()
            teacher_sampled_entropy = -valid_teacher_logprobs.mean().item()
            if is_teacher_turn:
                response.metrics.update(
                    {
                        "dash_opd/teacher_self_sampled_entropy": teacher_sampled_entropy,
                        "dash_opd/student_cross_entropy_on_teacher_samples": (
                            student_sampled_entropy
                        ),
                        "dash_opd/teacher_sampled_forward_kl": (
                            student_sampled_entropy - teacher_sampled_entropy
                        ),
                    }
                )
                if decision.switched_teacher_to_student:
                    target = (
                        teacher_entropy_at_budget_return_decision
                        if decision.forced_return_by_budget
                        else teacher_entropy_at_recovery_return_decision
                    )
                    target.append(teacher_sampled_entropy)
            else:
                response.metrics.update(
                    {
                        "dash_opd/student_self_sampled_entropy": student_sampled_entropy,
                        "dash_opd/teacher_cross_entropy_on_student_samples": (
                            teacher_sampled_entropy
                        ),
                        "dash_opd/student_sampled_reverse_kl": (
                            teacher_sampled_entropy - student_sampled_entropy
                        ),
                    }
                )
                student_entropy_points.append((r + 1, student_sampled_entropy))
                if decision.switched_student_to_teacher:
                    target = (
                        student_entropy_at_stagnation_takeover_decision
                        if stagnation_fallback_requested
                        else student_entropy_at_debt_takeover_decision
                    )
                    target.append(student_sampled_entropy)
                if r > 0 and teacher_roles[-1]:
                    student_entropy_after_handoff.append(student_sampled_entropy)
            valid_token_count = int(valid_response_mask.sum().item())
            if is_teacher_turn:
                teacher_self_entropy_sum += teacher_sampled_entropy * valid_token_count
                student_cross_entropy_on_teacher_sum += (
                    student_sampled_entropy * valid_token_count
                )
            else:
                student_self_entropy_sum += student_sampled_entropy * valid_token_count
                teacher_cross_entropy_on_student_sum += (
                    teacher_sampled_entropy * valid_token_count
                )
            per_turn_kl_sums.append((student_logprobs - teacher_logprobs).sum().item())
            total_response_tokens += response_token_count
            teacher_roles.append(bool(is_teacher_turn))
            decisions.append(decision)
            standard_log_ratio_means.append(standard_log_ratio)
            valid_token_counts.append(valid_token_count)
            if is_teacher_turn:
                teacher_generated_tokens += response_token_count
                teacher_generation_prompt_tokens += response.prompt_length
                teacher_generation_seconds += generation_seconds
            else:
                student_generated_tokens += response_token_count
                teacher_scored_sequence_tokens += max(len(response.tokens) - 1, 0)
                teacher_scoring_seconds += scoring_seconds

            think = parse_think(response_text) if self.include_think_in_history else None
            history.append(_format_history(format_obs, r + 1, action, think=think))
            observation, reward, done, info = env.step(action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0

        trajectory_kl_divergence = (
            sum(per_turn_kl_sums) / total_response_tokens if total_response_tokens > 0 else 0.0
        )
        rollout_seconds = perf_counter() - rollout_start_time
        for i, response in enumerate(turn_responses):
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

        if turn_responses:
            last_response = turn_responses[-1]
            student_points = [
                (turn_index + 1, log_ratio)
                for turn_index, (is_teacher, log_ratio) in enumerate(
                    zip(teacher_roles, standard_log_ratio_means)
                )
                if not is_teacher
            ]
            late_start_turn = (len(turn_responses) + 1) // 2
            late_student_points = [
                point for point in student_points if point[0] >= late_start_turn
            ]
            late_student_slope = _ols_slope(late_student_points)
            late_student_entropy_points = [
                point for point in student_entropy_points if point[0] >= late_start_turn
            ]
            late_student_entropy_slope = _ols_slope(late_student_entropy_points)

            student_valid_tokens = sum(
                token_count
                for is_teacher, token_count in zip(teacher_roles, valid_token_counts)
                if not is_teacher
            )
            teacher_valid_tokens = sum(
                token_count
                for is_teacher, token_count in zip(teacher_roles, valid_token_counts)
                if is_teacher
            )
            paper_metrics = {
                "mixed_policy_success": 1.0 if self._env_done else 0.0,
                "student_reverse_kl_mean": (
                    sum(
                        value * token_count
                        for is_teacher, value, token_count in zip(
                            teacher_roles,
                            standard_log_ratio_means,
                            valid_token_counts,
                        )
                        if not is_teacher
                    )
                    / student_valid_tokens
                    if student_valid_tokens
                    else 0.0
                ),
                "student_self_sampled_entropy_mean": (
                    student_self_entropy_sum / student_valid_tokens
                    if student_valid_tokens
                    else 0.0
                ),
                "teacher_cross_entropy_on_student_samples_mean": (
                    teacher_cross_entropy_on_student_sum / student_valid_tokens
                    if student_valid_tokens
                    else 0.0
                ),
                "student_sampled_reverse_kl_mean": (
                    (
                        teacher_cross_entropy_on_student_sum
                        - student_self_entropy_sum
                    )
                    / student_valid_tokens
                    if student_valid_tokens
                    else 0.0
                ),
                "late_turn_student_sampled_entropy_point_count": float(
                    len(late_student_entropy_points)
                ),
                "late_turn_student_sampled_entropy_slope_valid": float(
                    late_student_entropy_slope is not None
                ),
                "late_turn_student_kl_point_count": float(len(late_student_points)),
                "late_turn_student_kl_slope_valid": float(late_student_slope is not None),
                "student_generated_tokens": float(student_generated_tokens),
                "teacher_generated_tokens": float(teacher_generated_tokens),
                "teacher_generated_token_fraction": (
                    teacher_generated_tokens
                    / max(student_generated_tokens + teacher_generated_tokens, 1)
                ),
                "teacher_generation_prompt_tokens": float(
                    teacher_generation_prompt_tokens
                ),
                "teacher_scored_sequence_tokens": float(
                    teacher_scored_sequence_tokens
                ),
                "rollout_wall_time_seconds": rollout_seconds,
                "rollout_response_tokens_per_second": (
                    total_response_tokens / rollout_seconds
                    if rollout_seconds > 0.0
                    else 0.0
                ),
                "total_switches": float(
                    controller.switch_s_to_t_count + controller.switch_t_to_s_count
                ),
                "switches_per_10_turns": (
                    10.0
                    * (controller.switch_s_to_t_count + controller.switch_t_to_s_count)
                    / len(turn_responses)
                ),
                "stagnation_event_count": float(stagnation_event_count),
                "stagnation_takeover_decision_count": float(
                    stagnation_takeover_decision_count
                ),
                "stagnation_takeover_count": float(stagnation_takeover_count),
                "stagnation_takeover_blocked_by_budget_count": float(
                    stagnation_budget_block_count
                ),
                "teacher_start_schedule_enabled": float(
                    self.teacher_start_schedule_enabled and not self.task.is_eval
                ),
                "teacher_start_schedule_step": float(teacher_start_step),
                "teacher_start_probability": teacher_start_probability,
                "teacher_start_requested": float(teacher_start_requested),
                "teacher_started": float(teacher_started),
                "teacher_start_blocked_by_budget": float(
                    teacher_start_blocked_by_budget
                ),
                "teacher_start_step_parse_failed": float(
                    teacher_start_parse_failed
                ),
            }
            if teacher_valid_tokens:
                teacher_self_sampled_entropy_mean = (
                    teacher_self_entropy_sum / teacher_valid_tokens
                )
                student_cross_entropy_on_teacher_samples_mean = (
                    student_cross_entropy_on_teacher_sum / teacher_valid_tokens
                )
                paper_metrics.update(
                    {
                        "teacher_forward_kl_mean": sum(
                            -value * token_count
                            for is_teacher, value, token_count in zip(
                                teacher_roles,
                                standard_log_ratio_means,
                                valid_token_counts,
                            )
                            if is_teacher
                        )
                        / teacher_valid_tokens,
                        "teacher_self_sampled_entropy_mean": (
                            teacher_self_sampled_entropy_mean
                        ),
                        "student_cross_entropy_on_teacher_samples_mean": (
                            student_cross_entropy_on_teacher_samples_mean
                        ),
                        "teacher_sampled_forward_kl_mean": (
                            student_cross_entropy_on_teacher_samples_mean
                            - teacher_self_sampled_entropy_mean
                        ),
                        "teacher_minus_student_self_sampled_entropy": (
                            teacher_self_sampled_entropy_mean
                            - paper_metrics["student_self_sampled_entropy_mean"]
                        ),
                    }
                )
            if late_student_points:
                paper_metrics["late_turn_student_kl_mean"] = sum(
                    point[1] for point in late_student_points
                ) / len(late_student_points)
            if late_student_entropy_points:
                paper_metrics["late_turn_student_sampled_entropy_mean"] = sum(
                    point[1] for point in late_student_entropy_points
                ) / len(late_student_entropy_points)
            if late_student_slope is not None:
                paper_metrics["late_turn_student_kl_slope"] = late_student_slope
            if late_student_entropy_slope is not None:
                paper_metrics["late_turn_student_sampled_entropy_slope"] = (
                    late_student_entropy_slope
                )
            conditional_entropy_metrics = {
                "student_sampled_entropy_at_debt_takeover_decision_mean": (
                    student_entropy_at_debt_takeover_decision
                ),
                "student_sampled_entropy_at_stagnation_takeover_decision_mean": (
                    student_entropy_at_stagnation_takeover_decision
                ),
                "teacher_sampled_entropy_at_recovery_return_decision_mean": (
                    teacher_entropy_at_recovery_return_decision
                ),
                "teacher_sampled_entropy_at_budget_return_decision_mean": (
                    teacher_entropy_at_budget_return_decision
                ),
                "student_sampled_entropy_after_handoff_mean": (
                    student_entropy_after_handoff
                ),
            }
            paper_metrics.update(
                {
                    key: sum(values) / len(values)
                    for key, values in conditional_entropy_metrics.items()
                    if values
                }
            )
            if teacher_generated_tokens and teacher_generation_seconds > 0.0:
                paper_metrics["teacher_generation_tokens_per_second"] = (
                    teacher_generated_tokens / teacher_generation_seconds
                )
            if teacher_scored_sequence_tokens and teacher_scoring_seconds > 0.0:
                paper_metrics["teacher_scoring_tokens_per_second"] = (
                    teacher_scored_sequence_tokens / teacher_scoring_seconds
                )
            paper_metrics.update(
                _handoff_metrics(
                    teacher_roles=teacher_roles,
                    decisions=decisions,
                    episode_succeeded=self._env_done,
                )
            )
            last_response.metrics.update(
                {
                    "env_rounds": self._env_rounds,
                    "env_done": 1.0 if self._env_done else 0.0,
                    "kl_divergence": trajectory_kl_divergence,
                }
            )
            controller_summary = controller.summary()
            last_response.metrics.update(
                {f"dash_opd/{key}": value for key, value in controller_summary.items()}
            )
            if not self.task.is_eval:
                distribution_metrics = _trajectory_threshold_distribution_metrics(
                    max_support_debt=controller_summary["max_support_debt"],
                    max_recovery_credit=controller_summary["max_recovery_credit"],
                    has_teacher_segment=controller_summary["teacher_segments"] > 0.0,
                    takeover_threshold=self.controller_config.takeover_threshold,
                    return_threshold=self.controller_config.return_threshold,
                )
                distribution_metrics.update(
                    _signal_diagnostics(
                        decisions=decisions,
                        teacher_roles=teacher_roles,
                        student_drift_slack=self.controller_config.student_drift_slack,
                        teacher_recovery_slack=(
                            self.controller_config.teacher_recovery_slack
                        ),
                        z_clip=self.controller_config.z_clip,
                    )
                )
                last_response.metrics.update(
                    {
                        f"dash_opd/{key}": value
                        for key, value in distribution_metrics.items()
                    }
                )
            last_response.metrics.update(
                {f"dash_opd/{key}": value for key, value in paper_metrics.items()}
            )
            segment_tokenizer = get_segment_tokenizer(self.model.config.model_path)
            last_response.metrics.update(
                calculate_action_kl_metrics(turn_responses, segment_tokenizer)
            )
            last_response.metrics.update(
                calculate_think_kl_metrics(turn_responses, segment_tokenizer)
            )
            last_response.metrics.update(
                calculate_pre_action_think_kl_metrics(turn_responses, segment_tokenizer)
            )
            last_response.metrics.update(
                calculate_first_action_kl_metrics(turn_responses, segment_tokenizer)
            )

        return turn_responses
