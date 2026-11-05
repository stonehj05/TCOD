# TPU port (2026-11-02): copied from github.com/Lucian1115/DASH-OPD (commit dc0aa86)
# trinity/common/workflows/envs/opd/webshop/dash_opd_workflow.py; changed only: import paths (DASH/ here), and the env --
# a game on our shared WebShop servers (TCOD/webshop/env_server.py RemoteWebShopEnv, text_rich observations =
# the release's gym.make("WebAgentTextEnv-v0", observation_mode="text_rich", num_products=None, human_goals=True),
# whose reset / step / close are awaited) instead of an in-process env (one ~20 GB catalog per game).
# -*- coding: utf-8 -*-
"""DASH-OPD rollout switching for WebShop.

The DASH controller and Guided-OPD trainer path are shared with the ALFWorld
implementation.  This module only supplies the WebShop environment,
prompt, reward, and trajectory bookkeeping required around that controller.
"""

import time
from time import perf_counter
from typing import List, Tuple

import torch

from trinity.common.experience import CustomField, Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.webshop.env_server import RemoteWebShopEnv
from trinity.common.workflows.envs.DASH.alfworld.dash_opd_workflow import (
    DASHOPDAlfworldWorkflow,
    _ConsecutiveStagnationDetector,
    _handoff_metrics,
    _ols_slope,
    _signal_diagnostics,
    _trajectory_threshold_distribution_metrics,
)
from trinity.common.workflows.envs.DASH.dash_opd_controller import DASHController, DASHDecision
from trinity.common.workflows.envs.DASH.segment_kl import (
    calculate_action_kl_metrics,
    calculate_first_action_kl_metrics,
    calculate_pre_action_think_kl_metrics,
    calculate_think_kl_metrics,
    get_segment_tokenizer,
)
from trinity.common.workflows.envs.DASH.webshop.OPD_workflow import (
    OnPolicyDistillVerlAgentWebshopWorkflow,
)
from trinity.common.workflows.envs.DASH.webshop.utils import (
    _create_webshop_env,
    _extract_task_description,
    _format_available_actions,
    _format_history,
    build_webshop_prompt,
    format_observation,
    parse_action,
    parse_think,
    validate_action,
)


@WORKFLOWS.register_module("DASH_OPD_webshop_workflow")
class DASHOPDWebshopWorkflow(DASHOPDAlfworldWorkflow):
    """WebShop OPD with trajectory-local DASH role switching.

    Controller configuration, directional log-ratio scoring, action-only
    switching, teacher-start scheduling, and validation are inherited from the
    ALFWorld DASH implementation.  ``reset`` is explicitly routed through the
    WebShop OPD base so its public history defaults remain authoritative.
    """

    def reset(self, task):
        OnPolicyDistillVerlAgentWebshopWorkflow.reset(self, task)

    async def run_async(self) -> List[Experience]:
        env = RemoteWebShopEnv(f"dash_runner/{id(self)}/{time.time()}")
        try:
            return await self._run_episode(env, int(self.task_desc))
        finally:
            await env.close()

    async def _run_episode(
        self,
        env,
        session_id: int,
    ) -> List[Experience]:  # noqa: C901, PLR0915
        rollout_start_time = perf_counter()
        await env.reset(session=session_id)
        observation = env.observation
        task_description = _extract_task_description(observation)
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

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
                "DASH teacher-start model version is unavailable; falling back to " "batch_id=%r.",
                self.task.batch_id,
            )
        teacher_start_requested = teacher_start_probability >= 1.0 or (
            teacher_start_probability > 0.0
            and self._teacher_start_rng().random() < teacher_start_probability
        )
        teacher_started = teacher_start_requested and controller.start_with_teacher()
        teacher_start_blocked_by_budget = teacher_start_requested and not teacher_started
        stagnation_detector = _ConsecutiveStagnationDetector(self.stagnation_repeat_turns)

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

        for turn_index in range(self.max_env_steps):
            formatted_observation = format_observation(observation)
            available_actions = env.get_available_actions()
            formatted_actions = _format_available_actions(available_actions)
            user_content = build_webshop_prompt(
                task_description=task_description,
                step_count=turn_index,
                history=history,
                history_length=self.history_length,
                initial_no_history_turns=self.initial_no_history_turns,
                current_observation=formatted_observation,
                available_actions=formatted_actions,
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
            student_logprobs, teacher_logprobs, directional_log_ratio = await self._score_turn(
                response,
                is_teacher_turn=is_teacher_turn,
                turn_index=turn_index,
            )
            scoring_seconds = perf_counter() - scoring_start_time
            action = parse_action(response_text)
            action_stagnation, observation_stagnation = stagnation_detector.observe(
                action=action,
                observation=formatted_observation,
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
            stagnation_fallback_requested = stagnation_detected and not threshold_takeover_requested
            stagnation_takeover_decided = (
                stagnation_fallback_requested and decision.switched_student_to_teacher
            )
            stagnation_blocked_by_budget = (
                stagnation_fallback_requested and not decision.switched_student_to_teacher
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
                    "dash_opd/stagnation_fallback_requested": float(stagnation_fallback_requested),
                    "dash_opd/stagnation_takeover_decided": float(stagnation_takeover_decided),
                    "dash_opd/stagnation_takeover_blocked_by_budget": float(
                        stagnation_blocked_by_budget
                    ),
                    "dash_opd/action_only_role_switching": float(self.action_only_role_switching),
                }
            )
            turn_responses.append(response)

            response_token_count = len(response.tokens) - response.prompt_length
            valid_response_mask = response.action_mask
            if valid_response_mask is None:
                valid_response_mask = torch.ones_like(student_logprobs, dtype=torch.bool)
            else:
                valid_response_mask = valid_response_mask.to(
                    device=student_logprobs.device,
                    dtype=torch.bool,
                )
            standard_log_ratio = (
                (student_logprobs - teacher_logprobs).float()[valid_response_mask].mean().item()
            )
            valid_student_logprobs = student_logprobs.float()[valid_response_mask]
            valid_teacher_logprobs = teacher_logprobs.float()[valid_response_mask]
            student_sampled_entropy = -valid_student_logprobs.mean().item()
            teacher_sampled_entropy = -valid_teacher_logprobs.mean().item()
            if is_teacher_turn:
                teacher_self_entropy_sum += teacher_sampled_entropy * int(
                    valid_response_mask.sum().item()
                )
                student_cross_entropy_on_teacher_sum += student_sampled_entropy * int(
                    valid_response_mask.sum().item()
                )
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
                student_entropy_points.append((turn_index + 1, student_sampled_entropy))
                student_self_entropy_sum += student_sampled_entropy * int(
                    valid_response_mask.sum().item()
                )
                teacher_cross_entropy_on_student_sum += teacher_sampled_entropy * int(
                    valid_response_mask.sum().item()
                )
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
                if decision.switched_student_to_teacher:
                    target = (
                        student_entropy_at_stagnation_takeover_decision
                        if stagnation_fallback_requested
                        else student_entropy_at_debt_takeover_decision
                    )
                    target.append(student_sampled_entropy)
                if turn_index > 0 and teacher_roles[-1]:
                    student_entropy_after_handoff.append(student_sampled_entropy)

            valid_token_count = int(valid_response_mask.sum().item())
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
            history.append(
                _format_history(
                    formatted_observation,
                    turn_index + 1,
                    action,
                    think=think,
                )
            )
            action_valid, error_message = validate_action(action, available_actions)
            if action_valid:
                observation, reward, done, _ = await env.step(action)
            else:
                observation, reward, done = error_message, 0.0, False
            self._env_rounds = turn_index + 1
            if done:
                self._env_done = True
                self._final_reward = float(reward)
                break
        else:
            self._env_rounds = self.max_env_steps
        rollout_seconds = perf_counter() - rollout_start_time
        for response_index, response in enumerate(turn_responses):
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = response_index

        if turn_responses:
            self._record_dash_metrics(
                turn_responses=turn_responses,
                controller=controller,
                teacher_roles=teacher_roles,
                decisions=decisions,
                standard_log_ratio_means=standard_log_ratio_means,
                valid_token_counts=valid_token_counts,
                student_entropy_points=student_entropy_points,
                student_self_entropy_sum=student_self_entropy_sum,
                teacher_cross_entropy_on_student_sum=teacher_cross_entropy_on_student_sum,
                teacher_self_entropy_sum=teacher_self_entropy_sum,
                student_cross_entropy_on_teacher_sum=student_cross_entropy_on_teacher_sum,
                student_entropy_at_debt_takeover_decision=(
                    student_entropy_at_debt_takeover_decision
                ),
                student_entropy_at_stagnation_takeover_decision=(
                    student_entropy_at_stagnation_takeover_decision
                ),
                teacher_entropy_at_recovery_return_decision=(
                    teacher_entropy_at_recovery_return_decision
                ),
                teacher_entropy_at_budget_return_decision=(
                    teacher_entropy_at_budget_return_decision
                ),
                student_entropy_after_handoff=student_entropy_after_handoff,
                total_response_tokens=total_response_tokens,
                trajectory_kl_divergence=(
                    sum(per_turn_kl_sums) / total_response_tokens if total_response_tokens else 0.0
                ),
                student_generated_tokens=student_generated_tokens,
                teacher_generated_tokens=teacher_generated_tokens,
                teacher_generation_prompt_tokens=teacher_generation_prompt_tokens,
                teacher_scored_sequence_tokens=teacher_scored_sequence_tokens,
                teacher_generation_seconds=teacher_generation_seconds,
                teacher_scoring_seconds=teacher_scoring_seconds,
                rollout_seconds=rollout_seconds,
                teacher_start_step=teacher_start_step,
                teacher_start_probability=teacher_start_probability,
                teacher_start_parse_failed=teacher_start_parse_failed,
                teacher_start_requested=teacher_start_requested,
                teacher_started=teacher_started,
                teacher_start_blocked_by_budget=teacher_start_blocked_by_budget,
                stagnation_event_count=stagnation_event_count,
                stagnation_takeover_decision_count=stagnation_takeover_decision_count,
                stagnation_takeover_count=stagnation_takeover_count,
                stagnation_budget_block_count=stagnation_budget_block_count,
            )
        return turn_responses

    def _record_dash_metrics(self, **state) -> None:  # noqa: C901
        turn_responses = state["turn_responses"]
        teacher_roles = state["teacher_roles"]
        decisions = state["decisions"]
        valid_token_counts = state["valid_token_counts"]
        standard_log_ratio_means = state["standard_log_ratio_means"]
        last_response = turn_responses[-1]
        student_valid_tokens = sum(
            count for is_teacher, count in zip(teacher_roles, valid_token_counts) if not is_teacher
        )
        teacher_valid_tokens = sum(
            count for is_teacher, count in zip(teacher_roles, valid_token_counts) if is_teacher
        )
        late_start_turn = (len(turn_responses) + 1) // 2
        late_student_points = [
            (index + 1, value)
            for index, (is_teacher, value) in enumerate(
                zip(teacher_roles, standard_log_ratio_means)
            )
            if not is_teacher and index + 1 >= late_start_turn
        ]
        late_entropy_points = [
            point for point in state["student_entropy_points"] if point[0] >= late_start_turn
        ]
        late_student_slope = _ols_slope(late_student_points)
        late_entropy_slope = _ols_slope(late_entropy_points)
        controller_summary = state["controller"].summary()

        metrics = {
            "mixed_policy_success": float(self._final_reward >= 1.0),
            "student_reverse_kl_mean": (
                sum(
                    value * count
                    for is_teacher, value, count in zip(
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
                state["student_self_entropy_sum"] / student_valid_tokens
                if student_valid_tokens
                else 0.0
            ),
            "teacher_cross_entropy_on_student_samples_mean": (
                state["teacher_cross_entropy_on_student_sum"] / student_valid_tokens
                if student_valid_tokens
                else 0.0
            ),
            "student_sampled_reverse_kl_mean": (
                (state["teacher_cross_entropy_on_student_sum"] - state["student_self_entropy_sum"])
                / student_valid_tokens
                if student_valid_tokens
                else 0.0
            ),
            "student_generated_tokens": float(state["student_generated_tokens"]),
            "teacher_generated_tokens": float(state["teacher_generated_tokens"]),
            "teacher_generated_token_fraction": (
                state["teacher_generated_tokens"]
                / max(
                    state["student_generated_tokens"] + state["teacher_generated_tokens"],
                    1,
                )
            ),
            "teacher_generation_prompt_tokens": float(state["teacher_generation_prompt_tokens"]),
            "teacher_scored_sequence_tokens": float(state["teacher_scored_sequence_tokens"]),
            "rollout_wall_time_seconds": state["rollout_seconds"],
            "rollout_response_tokens_per_second": (
                state["total_response_tokens"] / state["rollout_seconds"]
                if state["rollout_seconds"] > 0.0
                else 0.0
            ),
            "total_switches": float(
                state["controller"].switch_s_to_t_count + state["controller"].switch_t_to_s_count
            ),
            "switches_per_10_turns": (
                10.0
                * (
                    state["controller"].switch_s_to_t_count
                    + state["controller"].switch_t_to_s_count
                )
                / len(turn_responses)
            ),
            "stagnation_event_count": float(state["stagnation_event_count"]),
            "stagnation_takeover_decision_count": float(
                state["stagnation_takeover_decision_count"]
            ),
            "stagnation_takeover_count": float(state["stagnation_takeover_count"]),
            "stagnation_takeover_blocked_by_budget_count": float(
                state["stagnation_budget_block_count"]
            ),
            "teacher_start_schedule_enabled": float(
                self.teacher_start_schedule_enabled and not self.task.is_eval
            ),
            "teacher_start_schedule_step": float(state["teacher_start_step"]),
            "teacher_start_probability": state["teacher_start_probability"],
            "teacher_start_requested": float(state["teacher_start_requested"]),
            "teacher_started": float(state["teacher_started"]),
            "teacher_start_blocked_by_budget": float(state["teacher_start_blocked_by_budget"]),
            "teacher_start_step_parse_failed": float(state["teacher_start_parse_failed"]),
            "late_turn_student_kl_point_count": float(len(late_student_points)),
            "late_turn_student_kl_slope_valid": float(late_student_slope is not None),
            "late_turn_student_sampled_entropy_point_count": float(len(late_entropy_points)),
            "late_turn_student_sampled_entropy_slope_valid": float(late_entropy_slope is not None),
        }
        if teacher_valid_tokens:
            teacher_entropy = state["teacher_self_entropy_sum"] / teacher_valid_tokens
            student_cross_entropy = (
                state["student_cross_entropy_on_teacher_sum"] / teacher_valid_tokens
            )
            metrics.update(
                {
                    "teacher_forward_kl_mean": sum(
                        -value * count
                        for is_teacher, value, count in zip(
                            teacher_roles,
                            standard_log_ratio_means,
                            valid_token_counts,
                        )
                        if is_teacher
                    )
                    / teacher_valid_tokens,
                    "teacher_self_sampled_entropy_mean": teacher_entropy,
                    "student_cross_entropy_on_teacher_samples_mean": (student_cross_entropy),
                    "teacher_sampled_forward_kl_mean": (student_cross_entropy - teacher_entropy),
                    "teacher_minus_student_self_sampled_entropy": (
                        teacher_entropy - metrics["student_self_sampled_entropy_mean"]
                    ),
                }
            )
        if late_student_points:
            metrics["late_turn_student_kl_mean"] = sum(
                value for _, value in late_student_points
            ) / len(late_student_points)
        if late_entropy_points:
            metrics["late_turn_student_sampled_entropy_mean"] = sum(
                value for _, value in late_entropy_points
            ) / len(late_entropy_points)
        if late_student_slope is not None:
            metrics["late_turn_student_kl_slope"] = late_student_slope
        if late_entropy_slope is not None:
            metrics["late_turn_student_sampled_entropy_slope"] = late_entropy_slope
        if state["teacher_generated_tokens"] and state["teacher_generation_seconds"] > 0:
            metrics["teacher_generation_tokens_per_second"] = (
                state["teacher_generated_tokens"] / state["teacher_generation_seconds"]
            )
        if state["teacher_scored_sequence_tokens"] and state["teacher_scoring_seconds"] > 0:
            metrics["teacher_scoring_tokens_per_second"] = (
                state["teacher_scored_sequence_tokens"] / state["teacher_scoring_seconds"]
            )
        conditional_entropy_metrics = {
            "student_sampled_entropy_at_debt_takeover_decision_mean": state[
                "student_entropy_at_debt_takeover_decision"
            ],
            "student_sampled_entropy_at_stagnation_takeover_decision_mean": state[
                "student_entropy_at_stagnation_takeover_decision"
            ],
            "teacher_sampled_entropy_at_recovery_return_decision_mean": state[
                "teacher_entropy_at_recovery_return_decision"
            ],
            "teacher_sampled_entropy_at_budget_return_decision_mean": state[
                "teacher_entropy_at_budget_return_decision"
            ],
            "student_sampled_entropy_after_handoff_mean": state["student_entropy_after_handoff"],
        }
        metrics.update(
            {
                key: sum(values) / len(values)
                for key, values in conditional_entropy_metrics.items()
                if values
            }
        )
        metrics.update(
            _handoff_metrics(
                teacher_roles=teacher_roles,
                decisions=decisions,
                episode_succeeded=self._final_reward >= 1.0,
            )
        )

        last_response.metrics.update(
            {
                "env_rounds": self._env_rounds,
                "env_done": float(self._env_done),
                "kl_divergence": state["trajectory_kl_divergence"],
                "session_id": float(self.task_desc),
                "webshop_score": self._final_reward,
                "webshop_success": float(self._final_reward >= 1.0),
                **{f"dash_opd/{key}": value for key, value in controller_summary.items()},
                **{f"dash_opd/{key}": value for key, value in metrics.items()},
            }
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
                    teacher_recovery_slack=self.controller_config.teacher_recovery_slack,
                    z_clip=self.controller_config.z_clip,
                )
            )
            last_response.metrics.update(
                {f"dash_opd/{key}": value for key, value in distribution_metrics.items()}
            )

        segment_tokenizer = get_segment_tokenizer(self.model.config.model_path)
        last_response.metrics.update(calculate_action_kl_metrics(turn_responses, segment_tokenizer))
        last_response.metrics.update(calculate_think_kl_metrics(turn_responses, segment_tokenizer))
        last_response.metrics.update(
            calculate_pre_action_think_kl_metrics(turn_responses, segment_tokenizer)
        )
        last_response.metrics.update(
            calculate_first_action_kl_metrics(turn_responses, segment_tokenizer)
        )
