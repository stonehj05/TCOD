# -*- coding: utf-8 -*-
"""Gated variant of OPD_workflow.py (OPD_alfworld_workflow): instead of
applying the teacher-logprobs OPD correction to EVERY turn uniformly, first
ask the teacher a yes/no consistency question about the student's actual
chosen action at that turn -- the same "would you choose this exact action?"
prompt used throughout the alfworld_ts_probe teacher-consistency experiments
(see e.g. alfworld_ts_probe/05_teacher_consistency.py) -- and only apply the
OPD correction to turns where the teacher says NO. Turns where the teacher
says YES are left unchanged (trained as an ordinary step, no teacher
distillation signal).

Mechanism: the framework already has an (until now unused, outside the
"hint OPD" workflow) per-turn masking hook -- `Experience.
teacher_logprobs_valid_mask` -- consumed by MultiTurnOpdAdvantage /
_compute_opd_advantage: `effective_mask = response_mask & teacher_valid_mask`,
so any turn with an all-False mask gets advantage=0 (no OPD contribution)
while training proceeds normally elsewhere. See
trinity/algorithm/advantage_fn/on_policy_distill_advantage.py and
trinity/trainer/verl/utils.py's gather_response_attrs handling of this
attribute. IMPORTANT: gather_response_attrs only uses this mask if EVERY
experience in the batch has it set -- this workflow therefore sets it on
EVERY turn (all-True when gated "apply", all-False when gated "skip"),
never leaves it unset.

Everything else (student always generates every action in the trajectory,
env stepping, reward computation, teacher_logprobs computation itself) is
IDENTICAL to OPD_workflow.py -- this is deliberately a minimal diff, not a
rewrite. Registered as a separate workflow ("OPD_gated_alfworld_workflow")
so the original OPD_alfworld_workflow is untouched.

Run history: launched against a live 8-GPU Trinity-RFT job
(TCOD_examples/alfworld/opd_gated.yaml, Qwen3-30B-A3B teacher /
Qwen3-4B student). Early runs hit "Prompt was truncated to 10240 tokens"
warnings followed by "Error parsing action: list index out of range" spam.
Root cause: the ORIGINAL per-turn `memory` list (inherited from
OPD_workflow.py) accumulated every past (user, assistant) pair for the
whole episode -- up to max_env_steps=30 turns -- in ADDITION to the
already-capped 2-step textual summary embedded in each turn's own
`user_content` (via ALFWORLD_TEMPLATE / HISTORY_LENGTH=2). That redundant
raw history eventually pushed prompts past model.max_prompt_tokens=10240;
Trinity's truncation keeps only the first N tokens (see
trinity/common/models/model.py's _handle_prompt_truncation), which chops
off the END of the conversation -- i.e. the current turn's own
<action></action> formatting instructions -- causing the model to emit
untagged fragments that TCOD's own (unguarded) parse_action() then fails
on. Confirmed against the actual TCOD paper (arXiv:2604.24005): Eq. 1
formally defines the state as the full uncapped history, but Table 5 /
Appendix E.1 document "History length: 2 steps" as referring only to the
textual action_history field, and Sec. 4.3 claims (inaccurately, per this
codebase) that "encapsulat[ing] the interaction history within the prompt
as a structured context" is what keeps prompts bounded -- it doesn't,
because the raw message list was sent on top of it. Fixed here by
rebuilding `messages` fresh each turn instead of accumulating it, so the
capped 2-step summary is the ONLY history reaching the model, matching
the paper's stated (if not literally implemented) design. See the
`messages = self.format_messages() + [...]` line in _run_episode.
"""

import string
from dataclasses import asdict
from typing import List, Optional

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

# Changed from alfworld_ts_probe/05_teacher_consistency.py's original
# wording (which asked for brief reasoning before the answer): a live
# 8-GPU run pairing the gated workflow with the Qwen3-30B-A3B-Instruct-2507
# teacher crashed from Explorer/Trainer data starvation (Trainer's 30-min
# read timeout hit 0/64 available experiences) in BOTH a plain-student and
# an -Instruct-2507-student config, while the same teacher trained fine
# under the (no-consistency-call) vanilla workflow -- isolating the
# consistency call itself, made every single turn, as the likely site of
# the slowdown. Leading hypothesis: this teacher rambles on the "briefly
# reason" invitation the same way students have been observed to ramble on
# the main action prompt. Removed the reasoning step entirely and demanded
# only the tag -- parse_yes_no() only ever looked for <answer>Yes/No</answer>
# and never depended on reasoning being present, so this is a pure prompt
# tightening, no parsing changes needed. Paired with a much smaller
# consistency_max_tokens in opd_gated.yaml as a hard backstop, since a
# prompt instruction alone doesn't guarantee compliance.
YES_NO_ADDENDUM = """

Now suppose the action chosen for the current step is:
{student_action}

Would you choose this exact action for the current step? Respond with ONLY \
<answer>Yes</answer> or <answer>No</answer> -- no reasoning, no explanation, \
no other text of any kind."""


def parse_yes_no(response: str) -> Optional[bool]:
    """Same convention as alfworld_ts_probe's parse_yes_no: looks at the
    first word inside <answer></answer> (or of the whole response if no tag
    is present). Returns None if unparseable."""
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


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow")
class OPDGatedAlfworldWorkflow(Workflow):
    """On-policy distillation workflow for AlfWorld, gated by a per-turn
    teacher yes/no consistency check.

    Identical to OnPolicyDistillVerlAgentAlfworldWorkflow (OPD_workflow.py)
    except: before computing teacher_logprobs, each turn's student action is
    first checked against a teacher "would you choose this?" yes/no query.
    The OPD correction (teacher_logprobs -> advantage) is only applied
    (teacher_logprobs_valid_mask = True) on turns where the teacher says No;
    turns where it says Yes are masked out (trained as ordinary steps).

    Use the same advantage_fn as the base workflow: multi_turn_opd
    (MultiTurnOpdAdvantage) -- no trainer/advantage_fn changes needed, the
    masking hook it already supports handles this.
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

        # Decoding settings for the yes/no consistency call specifically --
        # deliberately separate from self.temperature (the OPD
        # logprobs-scoring temperature) and from the student's own
        # rollout_args.temperature (typically 1.0 for training exploration).
        # Greedy by default: this is meant to be a single deterministic
        # verdict per turn, not a sampled one -- see the alfworld_ts_probe
        # consistency experiments' rationale for the same choice.
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

    def format_messages(self):
        """Format initial messages for the episode.

        Uses ALFWORLD_TEMPLATE_NO_HIS / ALFWORLD_TEMPLATE from utils.py.
        No system prompt; each user message is self-contained.
        """
        return []

    async def run_async(self) -> List[Experience]:
        game_file_path = self.task_desc
        env = _create_alfworld_env(game_file_path)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _ask_teacher_yes_no(self, memory: List[dict], action: str) -> tuple:
        """Asks the teacher whether it would choose `action` given the exact
        context in `memory` (which at call time ends with the student's own
        just-generated assistant response appended -- see call site). The
        student's response is NOT included as context for this question
        (mirrors alfworld_ts_probe's Call B construction): we present the
        action as a hypothetical to critique, not as something already said
        in-context.

        Returns (parsed, response_text) -- parsed is Optional[bool] (None if
        unparseable); response_text is the raw teacher output, returned so
        the caller can log/aggregate its length. This is temporary
        instrumentation added to verify the tightened YES_NO_ADDENDUM (see
        its own comment) actually keeps responses short in practice, rather
        than assuming prompt compliance.
        """
        # memory[:-1] drops the student's just-appended assistant turn,
        # ending at the current turn's user message.
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

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        n_gated_apply = 0  # turns where teacher said No -> OPD correction applied
        n_gated_skip = 0  # turns where teacher said Yes (or unparseable-default) -> skipped

        # Temporary instrumentation: verify the yes/no consistency prompt is
        # actually working (short, parseable responses) rather than assuming
        # it from the prompt wording alone. See _ask_teacher_yes_no's
        # docstring.
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

            # Single self-contained turn: no growing raw conversation. The
            # capped 2-step summary already embedded in `user_content` (via
            # ALFWORLD_TEMPLATE, HISTORY_LENGTH) is the only history sent to
            # the model -- this replaces the old `memory = memory + [...]`
            # accumulation, which kept every past (user, assistant) pair for
            # the whole episode (up to max_env_steps=30 turns) in ADDITION to
            # the 2-step summary, causing prompt lengths to grow unboundedly
            # and eventually exceed model.max_prompt_tokens (the cause of the
            # "Prompt was truncated" / "Error parsing action: list index out
            # of range" failures seen in early runs). See this file's module
            # docstring discussion of the TCOD paper's Eq. 1 vs. Table 5.
            messages = self.format_messages() + [{"role": "user", "content": user_content}]

            # Step 1: Student samples this turn (same pattern as OnPolicyDistillWorkflow)
            responses = await self.model.chat_async(messages, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""

            if response.logprobs is None:
                raise RuntimeError(
                    "OPDGatedAlfworldWorkflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )

            action = parse_action(response_text)

            # NEW: gate this turn's eventual OPD correction on the teacher's
            # yes/no verdict about the student's OWN chosen action. The
            # student's action is still what gets stepped into the env --
            # this only decides whether teacher_logprobs later contributes
            # to this turn's training advantage. Unparseable verdicts
            # default to "apply" (safer than silently skipping distillation
            # on a parse failure).
            # _ask_teacher_yes_no expects a memory-shaped list ending in the
            # assistant's turn (it strips that turn internally, see its own
            # docstring) -- build that locally; it is NOT carried into the
            # next iteration, so no cross-turn accumulation happens here.
            turn_context = messages + [{"role": "assistant", "content": response_text}]
            teacher_says_yes, yes_no_response_text = await self._ask_teacher_yes_no(
                turn_context, action
            )
            n_consistency_calls += 1
            if teacher_says_yes is None:
                n_consistency_unparseable += 1
            consistency_response_lengths.append(len(yes_no_response_text))
            apply_opd = teacher_says_yes is False or teacher_says_yes is None
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
        # prefix = this turn's single self-contained user message (task
        # description + capped 2-step action_history + current observation +
        # admissible actions -- see `messages` construction above), NOT the
        # whole episode's conversation. Same input the student had.
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

            # NEW: always set teacher_logprobs_valid_mask (never
            # conditionally) -- trainer/verl/utils.py's gather_response_attrs
            # only honors this mask if EVERY experience in the batch has it.
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
            # Temporary instrumentation (see _ask_teacher_yes_no's docstring):
            # confirms whether the tightened YES_NO_ADDENDUM is actually
            # keeping the teacher's consistency-check responses short and
            # parseable, rather than assuming it from the prompt wording.
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
