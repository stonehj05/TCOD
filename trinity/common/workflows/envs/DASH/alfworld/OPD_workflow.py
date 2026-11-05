# -*- coding: utf-8 -*-
# Modified for DASH-OPD, 2026.
# Changes to the upstream implementation are maintained in this repository.

"""On-Policy Distillation (OPD) workflow for AlfWorld.

Reference: OnPolicyDistillWorkflow logic; adapted for multi-turn AlfWorld.

Algorithm:
1. Student pre-samples trajectory (runs episode turn-by-turn with logprobs)
2. Split by turns: each turn has fixed prefix [system, obs_1, resp_1, ..., obs_t]
3. Teacher computes logprobs on same (prefix + response) per turn
4. Store teacher_logprobs in experience; advantage_fn uses teacher_logprobs - student_logprobs
5. Return one Experience per turn (like OPD returning one per sample)
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
from trinity.common.workflows.envs.DASH.alfworld.utils import (
    HISTORY_LENGTH,
    INITIAL_NO_HISTORY_TURNS,
    build_alfworld_prompt,
    parse_action,
    parse_think,
    format_observation,
    _extract_task,
    _format_history,
    _create_alfworld_env,
)


@WORKFLOWS.register_module("DASH_repo_OPD_alfworld_workflow")
class OnPolicyDistillVerlAgentAlfworldWorkflow(Workflow):
    """On-policy distillation workflow for AlfWorld.

    Computes and stores teacher_logprobs in each turn's experience.
    The advantage_fn in trainer will compute:
        advantages = teacher_logprobs - student_logprobs

    Use advantage_fn: multi_turn_opd (MultiTurnOpdAdvantage) for this workflow,
    since it returns List[Experience] (one per turn), not a single response.

    Logic aligned with OnPolicyDistillWorkflow:
    - Student samples (with logprobs); teacher computes logprobs on same sequences.
    - Per-turn split: prefix fixed per turn, one Experience per turn.
    - compute_reward() can be overridden by subclasses (default: episode final reward).
    """

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
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 30)
        self.history_length = task.workflow_args.get("history_length", HISTORY_LENGTH)
        self.initial_no_history_turns = task.workflow_args.get(
            "initial_no_history_turns", INITIAL_NO_HISTORY_TURNS
        )
        self.retain_full_chat_history = task.workflow_args.get(
            "retain_full_chat_history", False
        )
        self.include_think_in_history = task.workflow_args.get(
            "include_think_in_history", False
        )
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

    def reset(self, task: Task):
        """Reset the workflow with a new task.

        Unlike BaseSimpleWorkflow, this does NOT require reward_fn.
        """
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def compute_reward(self, response: Experience) -> float:
        """Return episode-level reward (same for all turns in the trajectory).

        Set in _run_episode: env reward when done, 0.0 when max steps exhausted.
        """
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Format initial messages for the episode.

        No system prompt. With retain_full_chat_history=false, each model call
        receives only the current self-contained user prompt.
        """
        return []

    async def run_async(self) -> List[Experience]:
        game_file_path = self.task_desc
        env = _create_alfworld_env(game_file_path)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            reformatted_admissible = "\n ".join(
                f"'{s}'" for s in admissible_commands if s != "help"
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

            # Step 1: Student samples this turn (same pattern as OnPolicyDistillWorkflow)
            responses = await self.model.chat_async(model_messages, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            if self.retain_full_chat_history:
                memory.append({"role": "assistant", "content": response_text})

            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillAlfworldWorkflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            action = parse_action(response_text)
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
            self._final_reward = 0.0  # failure: exhausted max steps

        # Step 2 & 3: Teacher logprobs and fill experience (mirror OnPolicyDistillWorkflow.run_async)
        # response.tokens contains the exact prompt seen by the student followed by
        # its response. The teacher scores that same sequence token-for-token.
        per_turn_kl_sums: List[float] = []
        total_response_tokens = 0
        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),  # full input = prefix + student's response
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

        # Trajectory-level metrics (computed once for the whole trajectory)
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
                calculate_pre_action_think_kl_metrics(
                    turn_responses, segment_tokenizer
                )
            )
            last_response.metrics.update(
                calculate_first_action_kl_metrics(turn_responses, segment_tokenizer)
            )

        return turn_responses
