# -*- coding: utf-8 -*-
"""Gated OPD workflow for AlfWorld -- REVERSED gating, full-memory.

Control condition for OPD_gated_workflow_fullmemory.py (the teacher-gated
reasoning-prompt run that reached avg@4 pooled 43.8% at 250 steps / 16.33h).
That file applies the OPD correction on turns where the teacher says "No"
(disagrees with the student's chosen action) and skips it on "Yes" turns.
This file does the exact opposite: apply OPD when the teacher says "Yes"
(agrees with the student), skip it when the teacher says "No".

Purpose: OPD_gated_workflow_random.py already showed that gating on a
random draw at the SAME rate as the teacher's real apply-rate (32.7%)
badly underperforms the real teacher-gated run (31.4% vs. 43.8% avg@4,
even below the untrained zero-shot baseline) -- establishing that the
teacher's judgment carries real signal, not just noise at the right rate.
This file asks a sharper version of that question: is it specifically
"correct the turns the teacher disagrees with" that helps, or would
"reinforce the turns the teacher agrees with" work just as well (or
better)? These are not obviously equivalent -- correcting disagreement
turns pushes the student's policy toward the teacher's preferred action
exactly where the two diverge (an error-correction signal), whereas
reinforcing agreement turns applies the OPD KL-style correction on turns
where teacher and student already coincide (arguably closer to plain
confirmation / a no-op in a well-aligned model, or a mild regularizer
keeping the student close to the teacher's distribution on turns it deems
already correct). If reversed gating underperforms the original, that
supports "correcting disagreement is where the value is"; if reversed
gating matches or beats the original, that would suggest the *fraction and
identity* of gated turns matters less than which polarity gets the
correction, or that reinforcing agreement is at least as valuable.

Only real change from OPD_gated_workflow_fullmemory.py: the gate condition
is flipped --
    apply_opd = teacher_says_yes is True or teacher_says_yes is None
instead of
    apply_opd = teacher_says_yes is False or teacher_says_yes is None
Unparseable teacher responses still default to "apply" in both variants
(kept identical on purpose, so the only variable under test is the Yes/No
polarity, not the parse-failure fallback). Given the original run's
measured Yes-rate was ~67.3%, this variant's gate-apply rate should land
around that figure (the complement of the original's ~32.7% apply-rate),
which is itself an interesting point of comparison against
OPD_gated_workflow_random.py's fixed-32.7%-rate control -- a natural
follow-up would be a random-gating control at ~67% to isolate rate from
polarity from teacher-judgment, though that is not implemented here.

Everything else (full-memory `messages`, reasoning-before-answer yes/no
prompt, teacher logprobs computed on every turn regardless of gate,
consistency metrics) is identical to OPD_gated_workflow_fullmemory.py.
"""

import string
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

# Identical wording to OPD_gated_workflow_fullmemory.py -- only the gate
# polarity differs (see module docstring), not the prompt.
YES_NO_ADDENDUM = """

Now suppose the action chosen for the current step is:
{student_action}

Would you choose this exact action for the current step? First briefly reason \
about whether it is the best available action, then give your final decision \
wrapped in <answer></answer> tags -- either <answer>Yes</answer> or \
<answer>No</answer>. Do not output any other text besides your reasoning and \
the final answer."""


def parse_yes_no(response: str) -> Optional[bool]:
    """Same convention as OPD_gated_workflow_fullmemory.py's parse_yes_no:
    looks at the first word inside <answer></answer> (or of the whole
    response if no tag is present). Returns None if unparseable."""
    try:
        content = (
            response.rsplit("<answer>", 1)[-1].split("</answer>")[0]
            if "<answer>" in response
            else response
        )
        words = content.strip().lower().split()
        if not words:
            return None
        first = words[0].strip(string.punctuation)
        if first == "yes":
            return True
        if first == "no":
            return False
        return None
    except Exception:
        return None


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow_reverse")
class OPDGatedAlfworldWorkflowReverse(Workflow):
    """Gated on-policy distillation workflow for AlfWorld, REVERSED-gating
    control: applies OPD on teacher-agrees ("Yes") turns instead of
    teacher-disagrees ("No") turns. See module docstring.
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

        self.consistency_temperature = task.workflow_args.get("consistency_temperature", 0.0)
        self.consistency_max_tokens = task.workflow_args.get("consistency_max_tokens", 4096)

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

    async def _ask_teacher_yes_no(self, memory: List[dict], action: str) -> tuple:
        """Identical logic to OPD_gated_workflow_fullmemory.py's version --
        see that file's docstring. `memory` here is the full growing
        conversation, ending with the student's just-generated assistant
        response; `memory[:-1]` strips that turn to get the context ending
        at the current turn's user message, onto which the yes/no addendum
        is appended.
        """
        context = memory[:-1]
        yes_no_messages = context[:-1] + [
            {
                "role": "user",
                "content": context[-1]["content"] + YES_NO_ADDENDUM.format(student_action=action),
            }
        ]
        yn_responses = await self.teacher_model.chat_async(
            yes_no_messages,
            temperature=self.consistency_temperature,
            max_tokens=self.consistency_max_tokens,
            n=1,
        )
        response_text = yn_responses[0].response_text or ""
        return parse_yes_no(response_text), response_text

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        turn_responses: List[Experience] = []

        # Growing conversation across turns -- same full-memory mechanism as
        # OPD_gated_workflow_fullmemory.py.
        memory: List[Dict[str, str]] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        n_gated_apply = 0
        n_gated_skip = 0
        n_consistency_calls = 0
        n_consistency_unparseable = 0
        consistency_response_lengths: List[int] = []

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

            # Growing conversation: append this turn's user message onto the
            # full history of every prior turn's (user, assistant) pair,
            # instead of a fresh single-turn `messages` list.
            memory = memory + [{"role": "user", "content": user_content}]

            # Step 1: Student samples this turn (same pattern as OnPolicyDistillWorkflow)
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            if response.logprobs is None:
                raise RuntimeError(
                    "OPDGatedAlfworldWorkflowReverse requires student model to return "
                    "logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )

            action = parse_action(response_text)

            # Gate this turn's eventual OPD correction on the teacher's
            # yes/no verdict, REVERSED relative to
            # OPD_gated_workflow_fullmemory.py: apply on "Yes" (teacher
            # agrees) instead of "No" (teacher disagrees). Unparseable
            # still defaults to "apply", same as the original, so the gate
            # polarity is the only variable under test.
            teacher_says_yes, yes_no_response_text = await self._ask_teacher_yes_no(
                memory, action
            )
            n_consistency_calls += 1
            if teacher_says_yes is None:
                n_consistency_unparseable += 1
            consistency_response_lengths.append(len(yes_no_response_text))
            apply_opd = teacher_says_yes is True or teacher_says_yes is None
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

        # Step 2 & 3: Teacher logprobs and fill experience (mirror OnPolicyDistillWorkflow.run_async)
        # response.tokens is the full sequence for this turn: [prefix | response], where
        # prefix is the FULL accumulated conversation up to and including
        # this turn's user message -- same input the student had.
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
            last_response.metrics["consistency_parse_success_rate"] = (
                1.0 - n_consistency_unparseable / n_consistency_calls
                if n_consistency_calls
                else 0.0
            )
            if consistency_response_lengths:
                last_response.metrics["consistency_response_length_mean"] = sum(
                    consistency_response_lengths
                ) / len(consistency_response_lengths)
                last_response.metrics["consistency_response_length_max"] = max(
                    consistency_response_lengths
                )

        return turn_responses
