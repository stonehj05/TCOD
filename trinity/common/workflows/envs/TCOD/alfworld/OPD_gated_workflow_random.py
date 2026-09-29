# -*- coding: utf-8 -*-
"""Gated OPD workflow for AlfWorld -- RANDOM-GATING control, full-memory.

Control condition for OPD_gated_workflow_fullmemory.py (the reasoning-prompt
teacher-yes/no gated variant that reached avg@4 pooled 43.8% at 250 steps /
16.33h -- see that file's and opd_gated_fullmemory.yaml's docstrings). That
run's measured overall teacher apply-rate (teacher says "No" -> OPD
correction applied) was 32.7% across its full 250-step run (n=175 logged
points, std 5.4pp) -- see RANDOM_APPLY_RATE below.

This file answers: does the teacher's actual judgment about WHICH turns to
apply OPD to matter, or does training with OPD applied to roughly the same
FRACTION of turns -- chosen uniformly at random, with no teacher involvement
in the gating decision at all -- produce a similar result? If random gating
at the same rate matches the teacher-gated run's performance, that would
suggest the earlier result is explained by turn-fraction / effective
learning-rate-like dynamics (fewer turns getting a nonzero OPD advantage
each step), not by the teacher correctly identifying which turns most need
correction. If random gating clearly underperforms, that's evidence the
teacher's per-turn judgment carries real signal.

Only real changes from OPD_gated_workflow_fullmemory.py:
  1. No teacher yes/no call at all -- _ask_teacher_yes_no and the
     consistency-prompt machinery (YES_NO_ADDENDUM, parse_yes_no,
     consistency_temperature/consistency_max_tokens config, and the
     consistency_parse_success_rate / consistency_response_length_*
     trajectory metrics, none of which apply here) are all removed.
  2. `apply_opd` is drawn from `random.random() < RANDOM_APPLY_RATE` instead
     of the teacher's verdict. Everything else (growing full-memory
     `messages`, teacher_logprobs computation on every turn regardless of
     the gate decision, teacher_logprobs_valid_mask masking mechanism,
     reward/metrics bookkeeping) is identical.
  3. As a side effect, removing the teacher yes/no call also removes its
     latency cost entirely -- this variant should run close to (or faster
     than) vanilla OPD's per-step time, not the ~80% slower reasoning-prompt
     gated run's, since there is no extra per-turn teacher generation call
     of any kind for gating (the teacher is still called once per turn for
     logprobs scoring, exactly as in ungated vanilla OPD).
"""

import random
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
    _extract_task,
    _format_history,
    _create_alfworld_env,
)

# Measured overall opd_gate_apply_rate from the teacher-gated reasoning-
# prompt full-memory run (alfworld_opd_gated_fullmemory_20260924145239):
# mean 0.3266 (32.7%) across n=175 logged trajectory-level data points
# spanning its complete 250-step run, std 0.0538. Fixed here (not
# recomputed dynamically) so this run's gate-apply rate is a controlled,
# known constant to compare against, rather than something that could drift.
RANDOM_APPLY_RATE = 0.327


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow_random")
class OPDGatedAlfworldWorkflowRandom(Workflow):
    """On-policy distillation workflow for AlfWorld, RANDOM-gating control.

    Identical to OPDGatedAlfworldWorkflowFullMemory (OPD_gated_workflow_
    fullmemory.py) except which turns get the OPD correction is decided by
    an independent random draw at a fixed rate (RANDOM_APPLY_RATE) instead
    of a teacher yes/no consistency judgment. See module docstring.
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
        turn_responses: List[Experience] = []

        # Growing conversation across turns -- same full-memory mechanism as
        # OPD_gated_workflow_fullmemory.py / OPD_workflow_fullmemory.py.
        memory: List[Dict[str, str]] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        n_gated_apply = 0  # turns where the random draw applied OPD
        n_gated_skip = 0  # turns where the random draw skipped OPD

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            reformatted_admissible = "\n ".join(
                f"'{s}'" for s in admissible_commands if s != "help"
            )

            if len(history) < HISTORY_LENGTH:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )
            else:
                action_history_str = "\n".join(
                    history[-HISTORY_LENGTH:]
                    if len(history) >= HISTORY_LENGTH
                    else history
                )
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )

            memory = memory + [{"role": "user", "content": user_content}]

            # Step 1: Student samples this turn (same pattern as OnPolicyDistillWorkflow)
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            if response.logprobs is None:
                raise RuntimeError(
                    "OPDGatedAlfworldWorkflowRandom requires student model to return "
                    "logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )

            action = parse_action(response_text)

            # RANDOM gating: no teacher call at all -- see module docstring.
            apply_opd = random.random() < RANDOM_APPLY_RATE
            response.opd_gate_apply = apply_opd  # consumed in the logprobs loop below
            if apply_opd:
                n_gated_apply += 1
            else:
                n_gated_skip += 1

            turn_responses.append(response)

            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0  # failure: exhausted max steps

        # Step 2 & 3: Teacher logprobs and fill experience -- IDENTICAL to
        # OPD_gated_workflow_fullmemory.py: every turn still gets scored by
        # the teacher regardless of the gate decision (the gate only decides
        # whether that score contributes to the training advantage via
        # teacher_logprobs_valid_mask below).
        per_turn_kl_sums: List[float] = []
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

            apply_opd = getattr(response, "opd_gate_apply", True)
            response.teacher_logprobs_valid_mask = torch.full(
                (len(teacher_resp_logprobs),), apply_opd, dtype=torch.bool
            )

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i
            response.metrics["opd_gate_applied"] = 1.0 if apply_opd else 0.0

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        # Trajectory-level metrics (computed once for the whole trajectory)
        trajectory_kl_divergence = sum(per_turn_kl_sums)
        n_gated_total = n_gated_apply + n_gated_skip
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            last_response.metrics["opd_gate_apply_rate"] = (
                n_gated_apply / n_gated_total if n_gated_total else 0.0
            )

        return turn_responses
