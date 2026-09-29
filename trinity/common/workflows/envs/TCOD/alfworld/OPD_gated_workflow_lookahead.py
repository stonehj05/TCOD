# -*- coding: utf-8 -*-
"""Gated OPD workflow for AlfWorld -- LOOK-AHEAD variant.

Sibling of OPD_gated_workflow_fullmemory.py (and its random/reverse
controls), but the gate is no longer "would the teacher have taken the
student's just-generated action" (a backward-looking, single-step question
answerable the instant that one action is produced). Instead, for step t the
gate asks the teacher to judge a forward-looking 5-step WINDOW starting at
and including t (steps [t, t+4]): is the agent making genuine progress
toward the task goal across those steps? OPD is applied at step t only if
the teacher says the window does NOT show progress (or the answer is
unparseable -- same fail-safe default as every other gated variant).

For the last WINDOW_SIZE steps of the episode -- the ones for which a full
forward window cannot be formed without running past the end of the
trajectory -- there is no teacher call at all. Instead the gate falls back
to the trajectory's own ground-truth outcome: apply OPD if the trajectory
eventually failed, skip it if the trajectory succeeded. This is a
deliberate choice, not just a boundary-condition patch: the LAST possible
5-step window (starting at N-WINDOW_SIZE, ending at the final step N-1)
would technically fit, but judging it by asking the teacher would just be a
noisier proxy for information we already have exactly -- whether the
episode succeeded. So the tail window is folded into the outcome-based
rule instead of the windowed teacher-call rule; the windowed rule only
ever fires for t < N - WINDOW_SIZE. For an episode with N <= WINDOW_SIZE
steps, every step falls under the outcome-based rule and no windowed calls
happen at all.

Architecture note: this requires restructuring `_run_episode` into three
sequential passes instead of OPD_gated_workflow_fullmemory.py's single
interleaved loop, because deciding step t's gate needs steps t+1..t+4,
which do not exist yet at the point step t is generated:
  1. Run the full student rollout to completion (no gating decisions).
  2. Walk back over every step 0..N-1 and decide `apply_opd` per step
     (a windowed teacher call, or the outcome-based rule for the tail).
  3. The usual teacher-logprobs-and-mask pass (identical to every other
     gated variant), now consuming the mask pass 2 produced.

Prompt construction for the windowed teacher call follows the SAME
full-memory convention as `_ask_teacher_yes_no` in
OPD_gated_workflow_fullmemory.py: the teacher sees the full growing
`memory` (every turn since the episode began, not a bounded/truncated
history), sliced through the end of the window, with a new trailing user
turn appended asking the progress question. This means, like the existing
yes/no check, each windowed call's prompt grows with how deep into the
episode the window starts -- and this variant makes N-WINDOW_SIZE such
calls per trajectory (vs. one per step for the existing consistency
check), so prompt-token cost from this gate grows faster with episode
length than the existing gated design. Accepted as a known tradeoff per
explicit discussion, not addressed with any truncation here.
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

DEFAULT_WINDOW_SIZE = 5

# Appended as a NEW trailing user turn (not folded into an existing user
# message the way YES_NO_ADDENDUM is) -- this question is retrospective,
# about steps already taken, so there is no "current, not-yet-decided" turn
# to attach it to; the last message in the sliced context is an assistant
# turn (the window's final step's response).
PROGRESS_ADDENDUM = """Based on the {window_len} step(s) shown above (steps \
{start_step}-{end_step}), do you think the agent is making progress toward \
completing the task? First briefly reason about whether those particular \
steps help advance the goal, then give your final decision wrapped in \
<answer></answer> tags -- either <answer>Yes</answer> or <answer>No</answer>. \
Do not output any other text besides your reasoning and the final answer."""


def parse_yes_no(response: str) -> Optional[bool]:
    """Same convention as every other gated variant's parse_yes_no: looks at
    the first word inside <answer></answer> (or of the whole response if no
    tag is present). Returns None if unparseable."""
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


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow_lookahead")
class OPDGatedAlfworldWorkflowLookahead(Workflow):
    """Look-ahead-gated on-policy distillation workflow for AlfWorld.

    OPD is applied at step t based on whether the teacher judges the 5-step
    window [t, t+4] to show progress (apply if not), except for the final
    WINDOW_SIZE steps of the episode, which are gated by the trajectory's
    own success/failure instead. See module docstring for the full
    rationale and the three-pass architecture this requires.
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

        self.window_size = task.workflow_args.get("progress_window_size", DEFAULT_WINDOW_SIZE)
        self.progress_temperature = task.workflow_args.get("progress_temperature", 0.0)
        self.progress_max_tokens = task.workflow_args.get("progress_max_tokens", 512)

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

    async def _ask_teacher_progress(
        self, memory: List[Dict[str, str]], window_start: int, window_end: int
    ) -> tuple:
        """Ask the teacher whether steps [window_start, window_end] (both
        0-indexed, inclusive) show genuine progress toward the task goal.

        `memory` is the FULL final conversation for the whole episode (2
        entries per step: user then assistant). Context is sliced through
        the end of the window (i.e. including the window's last step's
        assistant response) and a new trailing user turn asks the progress
        question -- see module docstring for why this differs from
        `_ask_teacher_yes_no`'s "modify the last user turn" pattern.
        """
        context = memory[: 2 * (window_end + 1)]
        progress_messages = context + [
            {
                "role": "user",
                "content": PROGRESS_ADDENDUM.format(
                    window_len=window_end - window_start + 1,
                    start_step=window_start + 1,
                    end_step=window_end + 1,
                ),
            }
        ]
        pr_responses = await self.teacher_model.chat_async(
            progress_messages,
            temperature=self.progress_temperature,
            max_tokens=self.progress_max_tokens,
            n=1,
        )
        response_text = pr_responses[0].response_text or ""
        return parse_yes_no(response_text), response_text

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        turn_responses: List[Experience] = []

        # Full growing conversation across turns, same as
        # OPD_gated_workflow_fullmemory.py -- kept in full after the episode
        # ends so pass 2 can slice arbitrary windows out of it.
        memory: List[Dict[str, str]] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        # ---- Pass 1: run the full rollout. No gating decisions here -- the
        # look-ahead gate needs future steps that don't exist yet. ----
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

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            if response.logprobs is None:
                raise RuntimeError(
                    "OPDGatedAlfworldWorkflowLookahead requires student model to return "
                    "logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )

            action = parse_action(response_text)
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

        n_total_steps = len(turn_responses)

        # ---- Pass 2: decide apply_opd for every step, now that the full
        # trajectory (and its outcome) is known. ----
        n_gated_apply = 0
        n_gated_skip = 0
        n_windowed_calls = 0
        n_windowed_unparseable = 0
        windowed_response_lengths: List[int] = []

        for t in range(n_total_steps):
            if t + self.window_size < n_total_steps:
                # Full forward window [t, t+window_size-1] fits strictly
                # before the trajectory's final step -- ask the teacher.
                window_end = t + self.window_size - 1
                teacher_says_progress, response_text = await self._ask_teacher_progress(
                    memory, t, window_end
                )
                n_windowed_calls += 1
                if teacher_says_progress is None:
                    n_windowed_unparseable += 1
                windowed_response_lengths.append(len(response_text))
                apply_opd = teacher_says_progress is False or teacher_says_progress is None
            else:
                # Last window_size steps (including the one whose window
                # would otherwise reach exactly the trajectory's end): use
                # the trajectory's own ground-truth outcome instead of a
                # teacher guess.
                apply_opd = not bool(self._final_reward)

            turn_responses[t].opd_gate_apply = apply_opd  # consumed in the logprobs loop below
            if apply_opd:
                n_gated_apply += 1
            else:
                n_gated_skip += 1

        # ---- Pass 3: teacher logprobs and fill experience (mirrors every
        # other gated variant). ----
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
            last_response.metrics["n_windowed_gate_decisions"] = n_windowed_calls
            last_response.metrics["n_outcome_gate_decisions"] = n_gated_total - n_windowed_calls
            if n_windowed_calls:
                last_response.metrics["progress_parse_success_rate"] = (
                    1.0 - n_windowed_unparseable / n_windowed_calls
                )
                last_response.metrics["progress_response_length_mean"] = sum(
                    windowed_response_lengths
                ) / len(windowed_response_lengths)
                last_response.metrics["progress_response_length_max"] = max(
                    windowed_response_lengths
                )

        return turn_responses
