# -*- coding: utf-8 -*-
"""Gated OPD workflow for AlfWorld -- LOOK-AHEAD, SOFT-WEIGHTED variant.

Sibling of OPD_gated_workflow_lookahead.py (see that file's module docstring
for the full window/outcome-partition rationale, which is unchanged here).
The only difference: instead of a hard apply/skip decision on the windowed
steps, positions where the teacher judges the window IS showing progress
are DOWNWEIGHTED (correction strength scaled by `progress_downweight_factor`,
default 0.5) rather than excluded outright. Positions where the teacher says
the window is NOT progress (or the answer is unparseable) keep full-strength
correction, same as the hard variant's "apply" case.

The trajectory's final WINDOW_SIZE steps (where no forward window can be
formed) are UNCHANGED from the hard variant: full correction if the episode
failed, zero correction if it succeeded -- this isn't about "the teacher
thinks progress is happening," it's ground truth, so there's nothing to
soften.

Implementation note -- why this needed NO changes to the shared advantage
function (trinity/algorithm/advantage_fn/on_policy_distill_advantage.py),
unlike what a first look at `teacher_logprobs_valid_mask` suggests:

That mask is consumed as `response_mask & teacher_valid_mask` (a strict
bitwise AND), and `MultiTurnOpdAdvantage.__call__` explicitly casts it to
`response_mask.dtype` (bool) before that -- so a continuous weight like 0.5
stored there would silently round-trip through `.to(dtype=torch.bool)` as
`True`, discarding the downweighting entirely. Rather than touch that
shared code path (used by every OPD-family workflow across ALFWorld/
ScienceWorld/WebShop), the weight is instead baked directly into what gets
stored as `response.teacher_logprobs`:

    advantages = kl_coef * (teacher_logprobs - student_logprobs) * mask

so blending the stored teacher logprob toward the student's own logprob by
a factor `w` -- `blended = student_logprob + w * (teacher_logprob -
student_logprob)` -- scales the resulting advantage by exactly `w` at that
position, for any `w` in [0, 1], through the UNMODIFIED existing machinery:
`w=1.0` reproduces the raw teacher logprob (full correction, bit-for-bit
identical to storing it directly), `w=0.0` collapses to the student's own
logprob (zero advantage, identical to the hard variant's excluded
positions), and `w=0.5` gives exactly half-strength correction. The mask
itself (`teacher_logprobs_valid_mask`) is therefore left `True` everywhere
in this workflow -- there is no longer any position this workflow needs a
boolean valid/invalid distinction for; the weighting alone carries all of
it. Per-turn `kl_divergence` metrics still use the RAW (unblended) teacher
logprobs, so diagnostic reporting of student/teacher divergence stays
honest regardless of what correction strength was actually applied.
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
DEFAULT_DOWNWEIGHT_FACTOR = 0.5

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


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow_lookahead_soft")
class OPDGatedAlfworldWorkflowLookaheadSoft(Workflow):
    """Soft-weighted, look-ahead-gated on-policy distillation workflow for
    AlfWorld.

    Identical to OPDGatedAlfworldWorkflowLookahead except windowed steps the
    teacher judges as showing progress get a downweighted (not excluded)
    correction. See module docstring for the full rationale and why this
    needed no shared advantage-function changes.
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
        self.downweight_factor = task.workflow_args.get(
            "progress_downweight_factor", DEFAULT_DOWNWEIGHT_FACTOR
        )

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
        Identical to OPD_gated_workflow_lookahead.py's version.
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
                    "OPDGatedAlfworldWorkflowLookaheadSoft requires student model to return "
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

        # ---- Pass 2: decide a correction WEIGHT (not a hard apply/skip)
        # for every step, now that the full trajectory (and its outcome) is
        # known. ----
        n_windowed_calls = 0
        n_windowed_unparseable = 0
        windowed_response_lengths: List[int] = []
        gate_weights: List[float] = []

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
                # Downweight (not exclude) when the teacher says the window
                # IS progress; full strength on "No" or unparseable, same
                # fail-safe convention as every other gated variant.
                weight = self.downweight_factor if teacher_says_progress is True else 1.0
            else:
                # Last window_size steps: ground-truth outcome, unchanged
                # from the hard variant -- full correction on failure, none
                # on success. Nothing here is about "the teacher thinks
                # progress is happening," so nothing to soften.
                weight = 1.0 if not bool(self._final_reward) else 0.0

            turn_responses[t].opd_gate_weight = weight  # consumed in the logprobs loop below
            gate_weights.append(weight)

        # ---- Pass 3: teacher logprobs, blend by weight, fill experience. ----
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

            # Raw (unblended) KL for honest diagnostic reporting -- computed
            # BEFORE blending, so this metric reflects actual student/teacher
            # divergence regardless of what correction strength was applied.
            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

            # Blend the stored teacher logprob toward the student's own by
            # this position's weight -- see module docstring for why this
            # scales the resulting advantage by exactly `weight` through the
            # UNMODIFIED shared advantage function, with no mask needed.
            weight = getattr(response, "opd_gate_weight", 1.0)
            blended_teacher_logprobs = student_resp_logprobs + weight * (
                teacher_resp_logprobs - student_resp_logprobs
            )
            response.teacher_logprobs = blended_teacher_logprobs
            response.teacher_logprobs_valid_mask = torch.full(
                (len(teacher_resp_logprobs),), True, dtype=torch.bool
            )

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i
            response.metrics["opd_gate_weight"] = weight

        # Trajectory-level metrics (computed once for the whole trajectory)
        trajectory_kl_divergence = sum(per_turn_kl_sums)
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            # Mean correction weight across all steps -- the continuous
            # generalization of the hard variants' opd_gate_apply_rate
            # (mean of {0,1}), kept under the same metric name for direct
            # comparability across variants.
            last_response.metrics["opd_gate_apply_rate"] = (
                sum(gate_weights) / len(gate_weights) if gate_weights else 0.0
            )
            last_response.metrics["n_windowed_gate_decisions"] = n_windowed_calls
            last_response.metrics["n_outcome_gate_decisions"] = (
                n_total_steps - n_windowed_calls
            )
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
