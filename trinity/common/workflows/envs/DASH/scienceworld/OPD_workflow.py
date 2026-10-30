# -*- coding: utf-8 -*-
# Modified for DASH-OPD, 2026.
# Changes to the upstream implementation are maintained in this repository.

"""On-Policy Distillation (OPD) workflow for ScienceWorld.

Copied from DASH-OPD (github.com/Lucian1115/DASH-OPD, commit dc0aa86),
trinity/common/workflows/envs/opd/scienceworld/OPD_workflow.py, registered there as
"OPD_scienceworld_workflow". Changed here: import paths and the registry name only.
"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.common.workflows.envs.DASH.segment_kl import (
    calculate_action_kl_metrics,
    calculate_first_action_kl_metrics,
    calculate_pre_action_think_kl_metrics,
    calculate_think_kl_metrics,
    get_segment_tokenizer,
)
from trinity.common.workflows.envs.DASH.scienceworld.utils import (
    HISTORY_LENGTH,
    INITIAL_NO_HISTORY_TURNS,
    _create_scienceworld_env,
    _format_history,
    _get_compact_action_info,
    _reset_scienceworld_env,
    build_scienceworld_prompt,
    format_observation,
    parse_action,
    parse_think,
)


@WORKFLOWS.register_module("DASH_repo_OPD_scienceworld_workflow")
class OnPolicyDistillVerlAgentScienceworldWorkflow(Workflow):
    """On-policy distillation workflow for ScienceWorld."""

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.reset(task)

        assert (
            self.auxiliary_model_wrappers is not None and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.history_length = task.workflow_args.get("history_length", HISTORY_LENGTH)
        self.initial_no_history_turns = task.workflow_args.get(
            "initial_no_history_turns", INITIAL_NO_HISTORY_TURNS
        )
        self.retain_full_chat_history = task.workflow_args.get("retain_full_chat_history", True)
        self.include_think_in_history = task.workflow_args.get("include_think_in_history", False)
        if self.history_length < -1:
            raise ValueError(
                f"history_length must be -1 or non-negative, got {self.history_length}"
            )
        if (
            isinstance(self.initial_no_history_turns, bool)
            or not isinstance(self.initial_no_history_turns, int)
            or self.initial_no_history_turns < 0
        ):
            raise ValueError(
                "initial_no_history_turns must be a non-negative integer, got "
                f"{self.initial_no_history_turns!r}"
            )
        self.is_eval = task.is_eval

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def compute_reward(self, response: Experience) -> float:
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
        env = _create_scienceworld_env(
            self.task_desc,
            max_env_steps=self.max_env_steps,
            generate_gold_path=False,
        )
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        observation, info, task_description = _reset_scienceworld_env(env)
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []
        best_score = info.get("score", 0)

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            action_templates, objects = _get_compact_action_info(env)
            reformatted_actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
            reformatted_objects = ", ".join(f"'{s}'" for s in objects)

            user_content = build_scienceworld_prompt(
                task_description=task_description,
                step_count=r,
                history=history,
                history_length=self.history_length,
                initial_no_history_turns=self.initial_no_history_turns,
                current_observation=format_obs,
                action_templates=reformatted_actions,
                objects=reformatted_objects,
                use_initial_no_history_template=self.retain_full_chat_history,
            )

            if self.retain_full_chat_history:
                memory = memory + [{"role": "user", "content": user_content}]
                model_messages = memory
            else:
                model_messages = [{"role": "user", "content": user_content}]

            responses = await self.model.chat_async(model_messages, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            if self.retain_full_chat_history:
                memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillScienceworldWorkflow requires student model to return "
                    "logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
            think = parse_think(response_text) if self.include_think_in_history else None
            history.append(_format_history(format_obs, r + 1, action, think=think))
            observation, reward, done, info = env.step(action)
            best_score = max(best_score, info.get("score", best_score + reward))
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                break
        else:
            self._env_rounds = self.max_env_steps

        self._final_reward = best_score / 100.0

        per_turn_kl_sums: List[float] = []
        total_response_tokens = 0
        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )

            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )

            response.teacher_logprobs = teacher_resp_logprobs

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)
            total_response_tokens += len(student_resp_logprobs)

        trajectory_kl_divergence = (
            sum(per_turn_kl_sums) / total_response_tokens if total_response_tokens > 0 else 0.0
        )
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
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
