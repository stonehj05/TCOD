# TPU port (2026-11-02): copied from github.com/Lucian1115/DASH-OPD (commit dc0aa86)
# trinity/common/workflows/envs/opd/webshop/OPD_workflow.py; changed only: import paths (DASH/ here), its registry name (DASH_repo_OPD_webshop_workflow: TCOD registers
# OPD_webshop_workflow) and the env --
# a game on our shared WebShop servers (TCOD/webshop/env_server.py RemoteWebShopEnv, text_rich observations =
# the release's gym.make("WebAgentTextEnv-v0", observation_mode="text_rich", num_products=None, human_goals=True),
# whose reset / step / close are awaited) instead of an in-process env (one ~20 GB catalog per game).
# -*- coding: utf-8 -*-
# Modified for DASH-OPD, 2026.
# Changes to the upstream implementation are maintained in this repository.

"""On-policy distillation workflow for WebShop."""

import time
from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.common.workflows.envs.TCOD.webshop.env_server import RemoteWebShopEnv
from trinity.common.workflows.envs.DASH.segment_kl import (
    calculate_action_kl_metrics,
    calculate_first_action_kl_metrics,
    calculate_pre_action_think_kl_metrics,
    calculate_think_kl_metrics,
    get_segment_tokenizer,
)
from trinity.common.workflows.envs.DASH.webshop.utils import (
    HISTORY_LENGTH,
    INITIAL_NO_HISTORY_TURNS,
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


@WORKFLOWS.register_module("DASH_repo_OPD_webshop_workflow")
class OnPolicyDistillVerlAgentWebshopWorkflow(Workflow):
    """Student rollout with teacher log-prob scoring on every WebShop turn."""

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
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)
        assert (
            self.auxiliary_model_wrappers
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval
        args = task.workflow_args
        self.temperature = float(args.get("temperature", 1.0))
        self.max_env_steps = int(args.get("max_env_steps", 15))
        self.history_length = int(args.get("history_length", HISTORY_LENGTH))
        self.initial_no_history_turns = args.get(
            "initial_no_history_turns", INITIAL_NO_HISTORY_TURNS
        )
        self.retain_full_chat_history = bool(args.get("retain_full_chat_history", True))
        self.include_think_in_history = bool(args.get("include_think_in_history", False))
        if self.max_env_steps <= 0:
            raise ValueError(f"max_env_steps must be positive, got {self.max_env_steps}")
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
        env = RemoteWebShopEnv(f"dash_runner/{id(self)}/{time.time()}")
        try:
            return await self._run_episode(env, int(self.task_desc))
        finally:
            await env.close()

    async def _run_episode(
        self,
        env,
        session_id: int,
        *,
        observation=None,
        history: Optional[List[str]] = None,
        task_description: Optional[str] = None,
        start_step: int = 0,
        step_limit: Optional[int] = None,
    ) -> List[Experience]:
        if observation is None:
            await env.reset(session=session_id)
            observation = env.observation
        if task_description is None:
            task_description = _extract_task_description(observation)
        history = list(history or [])

        self._env_done = False
        self._env_rounds = start_step
        self._final_reward = 0.0
        memory = self.format_messages()
        turn_responses: List[Experience] = []
        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        end_step = self.max_env_steps
        if step_limit is not None:
            end_step = min(end_step, start_step + max(int(step_limit), 0))

        for turn_index in range(start_step, end_step):
            available_actions = env.get_available_actions()
            formatted_observation = format_observation(observation)
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

            response = (await self.model.chat_async(model_messages, **kwargs))[0]
            response_text = response.response_text or ""
            if self.retain_full_chat_history:
                memory.append({"role": "assistant", "content": response_text})
            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillVerlAgentWebshopWorkflow requires student model "
                    "logprobs. Set rollout_args.logprobs (e.g. 0)."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
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

        await self._finalize_turn_responses(
            turn_responses,
            session_id=session_id,
            start_step=start_step,
        )
        return turn_responses

    async def _finalize_turn_responses(
        self,
        turn_responses: List[Experience],
        *,
        session_id: int,
        start_step: int,
    ) -> None:
        per_turn_kl_sums: List[float] = []
        total_response_tokens = 0
        for response_index, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.temperature,
            )
            teacher_response_logprobs = teacher_logprobs[response.prompt_length - 1 :]
            student_response_logprobs = response.logprobs
            if len(teacher_response_logprobs) != len(student_response_logprobs):
                raise RuntimeError(
                    "WebShop OPD token alignment failed: "
                    f"teacher_logprobs={len(teacher_response_logprobs)}, "
                    f"student_logprobs={len(student_response_logprobs)}, "
                    f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
                )
            response.teacher_logprobs = teacher_response_logprobs
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = start_step + response_index
            per_turn_kl_sums.append(
                (student_response_logprobs - teacher_response_logprobs).sum().item()
            )
            total_response_tokens += len(student_response_logprobs)

        if not turn_responses:
            return
        last_response = turn_responses[-1]
        trajectory_kl = (
            sum(per_turn_kl_sums) / total_response_tokens if total_response_tokens else 0.0
        )
        last_response.metrics.update(
            {
                "env_rounds": self._env_rounds,
                "env_done": float(self._env_done),
                "kl_divergence": trajectory_kl,
                "session_id": float(session_id),
                "webshop_score": self._final_reward,
                "webshop_success": float(self._final_reward >= 1.0),
            }
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
