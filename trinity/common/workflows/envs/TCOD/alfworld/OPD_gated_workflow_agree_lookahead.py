# -*- coding: utf-8 -*-
"""Gated OPD workflow for AlfWorld -- AGREEMENT + LOOK-AHEAD, combined soft weighting.

Combines the two teacher gates used separately elsewhere:

  (A) agreement  (OPD_gated_workflow_fullmemory.py): "would you choose this exact action
      for the current step?"  The criterion is met when the teacher says NO (or the
      answer is unparseable) -- the teacher disagrees with the student's action.
  (B) look-ahead (OPD_gated_workflow_lookahead_soft.py): "do the next `window_size` steps
      make progress toward the task?"  The criterion is met when the teacher says NO (or
      unparseable) -- the student is NOT making progress.

Per step, the OPD correction weight depends on `gate_mode`:

  gate_mode: sum  (default)
      weight = single_criterion_weight * [A met] + single_criterion_weight * [B met]
      i.e. with single_criterion_weight = 0.5: both met -> 1.0 (full OPD), exactly one met
      -> 0.5 (downweighted OPD), neither met -> 0.0 (no OPD on that step).

  gate_mode: disagree_required
      Disagreement is required; progress only sets the strength:
          A met and B met      (disagrees, not making progress) -> disagree_no_progress_weight (1.0)
          A met and B not met  (disagrees, making progress)     -> disagree_progress_weight    (0.5)
          A not met            (teacher agrees)                 -> 0.0, whatever B is
      Since B cannot matter when the teacher agrees, the progress question is only asked for
      the steps where the teacher disagreed (asked after all agreement answers are in).

All teacher prompts of an episode -- both questions for every step, and afterwards the
logprob-scoring calls -- are issued concurrently, at most `teacher_parallel_prompts` at a
time, so the teacher's vLLM engine batches them (continuous batching) instead of serving
one prompt after another. They are asked after the rollout has finished, because (B) needs
the steps that follow; (A) only uses the conversation up to that step, exactly as in the
agreement-gated workflow, so asking it later does not change what the teacher sees. The
answers do not depend on each other, so the order of completion does not matter.

For the trajectory's final `window_size` steps no forward window exists. As in the
look-ahead variants, (B) falls back to the ground-truth outcome there: "not making
progress" iff the episode failed. (A) is still asked for those steps.

The weight is applied the same way as in OPD_gated_workflow_lookahead_soft.py (see that
file's module docstring for why): the stored teacher logprobs are blended toward the
student's own, `student + weight * (teacher - student)`, which scales the advantage by
exactly `weight` through the unmodified shared advantage function. weight = 0 therefore
gives zero advantage on that step. Per-turn `kl_divergence` metrics use the raw teacher
logprobs.

defer_teacher: true  (TPU trainer only, trainer_type: tunix)
    The explorer then only plays the game: no teacher prompt and no teacher scoring here.
    Each turn carries what the gate needs (`info["opd_deferred"]`, see `deferred_payload`):
    the conversation up to the end of its look-ahead window, its action and the episode
    outcome. The trainer asks the teacher the same questions and scores the turn, but only
    for the turns it actually samples into a training batch
    (trinity/trainer/tunix/teacher_gate.py). Prompts, parsing and weights are the functions
    of this module in both places, so a turn gets the same weight either way; most turns an
    explore step produces are never sampled, and their teacher work is simply not done.
"""

import asyncio
import json
import string
import zlib
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
DEFAULT_SINGLE_CRITERION_WEIGHT = 0.5
DEFAULT_TEACHER_PARALLEL_PROMPTS = 16

# Same wording as OPD_gated_workflow_fullmemory.py's agreement check.
YES_NO_ADDENDUM = """

Now suppose the action chosen for the current step is:
{student_action}

Would you choose this exact action for the current step? First briefly reason \
about whether it is the best available action, then give your final decision \
wrapped in <answer></answer> tags -- either <answer>Yes</answer> or \
<answer>No</answer>. Do not output any other text besides your reasoning and \
the final answer."""

# Same wording as OPD_gated_workflow_lookahead_soft.py's progress check.

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


def agree_messages(memory: List[Dict[str, str]], step: int, action: str) -> List[Dict[str, str]]:
    """Agreement prompt for step `step` (0-indexed): the conversation up to and including that
    step's user message, with the yes/no addendum appended to it (the student's own response
    for that step is not shown). Same construction as OPD_gated_workflow_fullmemory.py."""
    context = memory[: 2 * step + 1]
    return context[:-1] + [
        {
            "role": "user",
            "content": context[-1]["content"] + YES_NO_ADDENDUM.format(student_action=action),
        }
    ]


def progress_messages(
    memory: List[Dict[str, str]], window_start: int, window_end: int
) -> List[Dict[str, str]]:
    """Progress prompt for steps [window_start, window_end] (0-indexed, inclusive): the
    conversation through the end of the window, then the question. Identical to
    OPD_gated_workflow_lookahead_soft.py's version."""
    context = memory[: 2 * (window_end + 1)]
    return context + [
        {
            "role": "user",
            "content": PROGRESS_ADDENDUM.format(
                window_len=window_end - window_start + 1,
                start_step=window_start + 1,
                end_step=window_end + 1,
            ),
        }
    ]


def gate_weight(gate: Dict, disagree: bool, not_progress: Optional[bool]) -> float:
    """OPD correction weight of one step. `gate` holds gate_mode and its weights;
    `not_progress` may be None only where it cannot matter (teacher agreed, disagree_required)."""
    if gate["gate_mode"] == "sum":
        return gate["single_criterion_weight"] * (int(disagree) + int(not_progress))
    if not disagree:
        return 0.0
    return gate["disagree_no_progress_weight"] if not_progress else gate["disagree_progress_weight"]


def deferred_payload(
    memory: List[Dict[str, str]], step: int, n_steps: int, action: str, final_reward: float, gate: Dict
) -> Dict:
    """What the trainer needs to gate and score turn `step` by itself (defer_teacher mode).

    The conversation is kept through the end of the step's look-ahead window (the progress
    prompt's context; the agreement prompt's context is a prefix of it), or only up to the
    step's user message for the final `window_size` steps, which use the outcome instead.
    Stored zlib-compressed: the full-memory conversation repeats in every turn of a game.
    """
    has_window = step + gate["window_size"] < n_steps
    keep = 2 * (step + gate["window_size"]) if has_window else 2 * step + 1
    return {
        "step": step,
        "n_steps": n_steps,
        "action": action,
        "final_reward": float(final_reward),
        "has_window": has_window,
        "memory_z": zlib.compress(json.dumps(memory[:keep]).encode()),
        "gate": gate,
    }


def payload_memory(payload: Dict) -> List[Dict[str, str]]:
    return json.loads(zlib.decompress(payload["memory_z"]))


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow_agree_lookahead")
class OPDGatedAlfworldWorkflowAgreeLookahead(Workflow):
    """Agreement + look-ahead gated on-policy distillation workflow for AlfWorld.

    Per step: weight 1.0 if the teacher both disagrees with the student's action and
    judges the following steps as not making progress, 0.5 if exactly one of the two
    holds, 0.0 if neither. See the module docstring.
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
        self.consistency_temperature = task.workflow_args.get("consistency_temperature", 0.0)
        self.consistency_max_tokens = task.workflow_args.get("consistency_max_tokens", 512)
        self.single_criterion_weight = task.workflow_args.get(
            "single_criterion_weight", DEFAULT_SINGLE_CRITERION_WEIGHT
        )
        self.gate_mode = task.workflow_args.get("gate_mode", "sum")
        if self.gate_mode not in ("sum", "disagree_required"):
            raise ValueError(f"unknown gate_mode {self.gate_mode!r} (use 'sum' or 'disagree_required')")
        self.disagree_no_progress_weight = task.workflow_args.get("disagree_no_progress_weight", 1.0)
        self.disagree_progress_weight = task.workflow_args.get("disagree_progress_weight", 0.5)
        # Max teacher requests in flight per episode (1 = strictly sequential).
        self.teacher_parallel_prompts = max(
            1, int(task.workflow_args.get("teacher_parallel_prompts", DEFAULT_TEACHER_PARALLEL_PROMPTS))
        )
        # Leave every teacher call to the trainer (see module docstring).
        self.defer_teacher = bool(task.workflow_args.get("defer_teacher", False))

    def gate_config(self) -> Dict:
        """Everything that determines a step's weight and the teacher calls behind it."""
        return {
            "gate_mode": self.gate_mode,
            "window_size": self.window_size,
            "single_criterion_weight": self.single_criterion_weight,
            "disagree_no_progress_weight": self.disagree_no_progress_weight,
            "disagree_progress_weight": self.disagree_progress_weight,
            "consistency_temperature": self.consistency_temperature,
            "consistency_max_tokens": self.consistency_max_tokens,
            "progress_temperature": self.progress_temperature,
            "progress_max_tokens": self.progress_max_tokens,
            "temperature": self.temperature,
        }

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

    async def _ask_teacher_agree(
        self, memory: List[Dict[str, str]], step: int, action: str
    ) -> tuple:
        """Ask the teacher whether it would choose `action` at step `step` (0-indexed).
        Same prompt construction as OPD_gated_workflow_fullmemory.py: the conversation up
        to and including that step's user message, with the yes/no addendum appended to it
        (the student's own response for that step is not shown).
        """
        yes_no_messages = agree_messages(memory, step, action)
        yn_responses = await self.teacher_model.chat_async(
            yes_no_messages,
            temperature=self.consistency_temperature,
            max_tokens=self.consistency_max_tokens,
            n=1,
        )
        response_text = yn_responses[0].response_text or ""
        return parse_yes_no(response_text), response_text

    async def _ask_teacher_progress(
        self, memory: List[Dict[str, str]], window_start: int, window_end: int
    ) -> tuple:
        """Ask the teacher whether steps [window_start, window_end] (both
        0-indexed, inclusive) show genuine progress toward the task goal.
        Identical to OPD_gated_workflow_lookahead_soft.py's version.
        """
        prompt = progress_messages(memory, window_start, window_end)
        pr_responses = await self.teacher_model.chat_async(
            prompt,
            temperature=self.progress_temperature,
            max_tokens=self.progress_max_tokens,
            n=1,
        )
        response_text = pr_responses[0].response_text or ""
        return parse_yes_no(response_text), response_text

    def _defer(
        self, turn_responses: List[Experience], actions: List[str], memory: List[Dict[str, str]]
    ) -> List[Experience]:
        """defer_teacher: return the rollout with the gate's inputs attached and no teacher
        output; teacher_logprobs is filled by the trainer for the turns it samples."""
        gate, n = self.gate_config(), len(turn_responses)
        for i, response in enumerate(turn_responses):
            if response.info is None:
                response.info = {}
            response.info["opd_deferred"] = deferred_payload(
                memory, i, n, actions[i], self._final_reward, gate
            )
            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i
        if turn_responses:
            turn_responses[-1].metrics["env_rounds"] = self._env_rounds
            turn_responses[-1].metrics["env_done"] = 1.0 if self._env_done else 0.0
        return turn_responses

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        turn_responses: List[Experience] = []
        actions: List[str] = []
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
                    "OPDGatedAlfworldWorkflowAgreeLookahead requires student model to return "
                    "logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )

            action = parse_action(response_text)
            turn_responses.append(response)
            actions.append(action)

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

        if self.defer_teacher:
            return self._defer(turn_responses, actions, memory)

        # ---- Pass 2: ask the teacher both questions for every step, batched (bounded
        # concurrency), then turn the two criteria into a correction weight per step. ----
        n_agree_unparseable = 0
        n_windowed_calls = 0
        n_windowed_unparseable = 0
        agree_response_lengths: List[int] = []
        windowed_response_lengths: List[int] = []
        gate_weights: List[float] = []
        n_disagree = 0
        n_not_progress = 0

        gate = self.gate_config()
        limiter = asyncio.Semaphore(self.teacher_parallel_prompts)

        async def limited(coro):
            async with limiter:
                return await coro

        has_window = [t + self.window_size < n_total_steps for t in range(n_total_steps)]
        ask_agree = [limited(self._ask_teacher_agree(memory, t, actions[t])) for t in range(n_total_steps)]

        def ask_progress(steps):
            return [limited(self._ask_teacher_progress(memory, t, t + self.window_size - 1)) for t in steps]

        if self.gate_mode == "sum":
            # Both questions for every step, all in one batch.
            progress_steps = [t for t in range(n_total_steps) if has_window[t]]
            agree_answers, progress_list = await asyncio.gather(
                asyncio.gather(*ask_agree), asyncio.gather(*ask_progress(progress_steps))
            )
        else:
            # disagree_required: progress only matters where the teacher disagrees, so ask
            # the agreement question for every step first, then progress for those steps only.
            agree_answers = await asyncio.gather(*ask_agree)
            progress_steps = [
                t for t in range(n_total_steps) if has_window[t] and agree_answers[t][0] is not True
            ]
            progress_list = await asyncio.gather(*ask_progress(progress_steps))
        progress_answers = dict(zip(progress_steps, progress_list))
        n_progress_known = 0

        for t in range(n_total_steps):
            teacher_agrees, agree_text = agree_answers[t]
            if teacher_agrees is None:
                n_agree_unparseable += 1
            agree_response_lengths.append(len(agree_text))
            # Criterion A: teacher disagrees with the action ("No" or unparseable -- fail-safe).
            disagree = teacher_agrees is not True

            not_progress = None  # unknown: not asked (teacher agreed, in disagree_required mode)
            if t in progress_answers:
                teacher_says_progress, progress_text = progress_answers[t]
                n_windowed_calls += 1
                if teacher_says_progress is None:
                    n_windowed_unparseable += 1
                windowed_response_lengths.append(len(progress_text))
                # Criterion B: not making progress ("No" or unparseable -- fail-safe).
                not_progress = teacher_says_progress is not True
            elif not has_window[t]:
                # Last window_size steps: no forward window; use the ground-truth outcome
                # (failed episode = not making progress), as in the look-ahead variants.
                not_progress = not bool(self._final_reward)

            weight = gate_weight(gate, disagree, not_progress)
            n_disagree += int(disagree)

            response = turn_responses[t]
            response.opd_gate_weight = weight  # consumed in the logprobs loop below
            if response.metrics is None:
                response.metrics = {}
            response.metrics["opd_gate_disagree"] = float(disagree)
            if not_progress is not None:
                n_progress_known += 1
                n_not_progress += int(not_progress)
                response.metrics["opd_gate_not_progress"] = float(not_progress)
            gate_weights.append(weight)

        # ---- Pass 3: teacher logprobs, blend by weight, fill experience. ----
        per_turn_kl_sums: List[float] = []
        all_teacher_logprobs = await asyncio.gather(
            *[
                limited(
                    self.teacher_model.logprobs_async(
                        tokens=response.tokens.tolist(),  # full input = prefix + student's response
                        temperature=self.temperature,
                    )
                )
                for response in turn_responses
            ]
        )
        for i, response in enumerate(turn_responses):
            teacher_logprobs = all_teacher_logprobs[i]

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
            # Mean correction weight across all steps (same metric name as the other
            # gated variants, for direct comparability).
            last_response.metrics["opd_gate_apply_rate"] = (
                sum(gate_weights) / len(gate_weights) if gate_weights else 0.0
            )
            if n_total_steps:
                # "full" = the weight for both criteria met, "half" = any weight in between.
                full = (
                    self.single_criterion_weight * 2
                    if self.gate_mode == "sum"
                    else self.disagree_no_progress_weight
                )
                last_response.metrics["opd_gate_full_rate"] = (
                    sum(w == full for w in gate_weights) / n_total_steps
                )
                last_response.metrics["opd_gate_half_rate"] = (
                    sum(0.0 < w < full for w in gate_weights) / n_total_steps
                )
                last_response.metrics["opd_gate_none_rate"] = (
                    sum(w == 0.0 for w in gate_weights) / n_total_steps
                )
                last_response.metrics["opd_gate_disagree_rate"] = n_disagree / n_total_steps
                # Over the steps where the progress criterion was evaluated (in
                # disagree_required mode: only the steps the teacher disagreed with).
                if n_progress_known:
                    last_response.metrics["opd_gate_not_progress_rate"] = n_not_progress / n_progress_known
                last_response.metrics["consistency_parse_success_rate"] = (
                    1.0 - n_agree_unparseable / n_total_steps
                )
                last_response.metrics["consistency_response_length_mean"] = sum(
                    agree_response_lengths
                ) / len(agree_response_lengths)
                last_response.metrics["consistency_response_length_max"] = max(agree_response_lengths)
            last_response.metrics["n_windowed_gate_decisions"] = n_windowed_calls
            last_response.metrics["n_outcome_gate_decisions"] = sum(
                1 for t in range(n_total_steps) if not has_window[t]
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
