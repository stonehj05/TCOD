# -*- coding: utf-8 -*-
"""Gated OPD workflow for AlfWorld -- DISAGREEMENT + TEACHER LOOK-AHEAD.

Same agreement gate as OPD_gated_workflow_agree_lookahead.py in `disagree_required` mode, but
the look-ahead is the TEACHER's, not the student's. Per step:

  (A) agreement: "would you choose this exact action for the current step?" (same prompt).
      The teacher agrees -> weight 0, nothing else is done for that step.
  (B) teacher look-ahead, only where the teacher disagrees ("No" or unparseable): the game is
      replayed up to that step, and from there the TEACHER plays `progress_window_size` (5)
      steps itself, starting with its own action for the step it disagreed with. It sees the
      same conversation the student saw up to that step and the same prompt template
      afterwards. Then the teacher is asked the usual progress question about its own
      steps: "do you think the agent is making progress toward completing the task?"

OPD weight of the step:
    teacher agrees                                      -> 0.0
    teacher disagrees, its own steps make progress      -> teacher_progress_weight     (1.0)
    teacher disagrees, its own steps make NO progress   -> teacher_no_progress_weight  (0.5)
i.e. the correction is downweighted where the teacher objects to the student's action but
does not do better itself from that position.

Details:
  - The teacher's steps are played in a fresh copy of the game (ALFWorld is deterministic:
    replaying the student's earlier actions reproduces the state). The student's own game is
    not affected. `teacher_lookahead_replay_ok` reports whether the replayed observation
    equals the one the student saw.
  - If the teacher finishes the task inside its window, that counts as progress and the
    question is not asked. The window is not cut by `max_env_steps`: also for the student's
    last steps the teacher plays its full window.
  - Only an explicit "No" downweights. An unparseable progress answer, or a look-ahead that
    fails (e.g. the game cannot be replayed), leaves the full weight, like the other gated
    variants, which fall back to applying the correction.
  - The weight is applied as in the other soft variants: the stored teacher logprobs are
    blended toward the student's own, `student + weight * (teacher - student)`.

Cost: besides the agreement prompt per step, every disagreed step costs one game replay
(about 3 s to create the game plus 0.13 s per replayed step, CPU) and up to
`progress_window_size` + 1 teacher generations.

defer_teacher: true  (TPU trainer only, trainer_type: tunix)
    As in OPD_gated_workflow_agree_lookahead.py: the explorer only plays the game and attaches
    the gate's inputs to each turn; the trainer does (A), (B) and the scoring for the turns
    it samples (trinity/trainer/tunix/teacher_gate.py). For (B) each turn also carries the
    game file and the student's earlier actions. The trainer runs every teacher look-ahead
    as its own Ray task (one CPU each, on any node: the game files must exist at the same
    path on every node, as scripts/tpu/setup_node.sh arranges), so the replays run in
    parallel. `teacher_lookahead` below is the one implementation used in both places.
"""

import asyncio
import json
import time
import zlib
from typing import Awaitable, Callable, Dict, List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_agree_lookahead import (
    DEFAULT_WINDOW_SIZE,
    OPDGatedAlfworldWorkflowAgreeLookahead,
    parse_yes_no,
    progress_messages,
)
from trinity.common.workflows.envs.TCOD.alfworld.utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _create_alfworld_env_with_checkpoint,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)

GATE_MODE = "teacher_lookahead"
CALL_ATTEMPTS = 3

# ask(messages, temperature, max_tokens) -> the teacher's response text
Ask = Callable[[List[Dict[str, str]], float, int], Awaitable[str]]


def build_user_content(observation: str, info: Dict, history: List[str], task_description: str, r: int):
    """The user message of step `r` (0-indexed) and the formatted observation, exactly as the
    rollout loop of the gated workflows builds them."""
    format_obs = format_observation(observation)
    admissible_commands = info.get("admissible_commands", [])
    if admissible_commands and isinstance(admissible_commands[0], list):
        admissible_commands = admissible_commands[0]
    reformatted_admissible = "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")
    if len(history) < HISTORY_LENGTH:
        user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=format_obs,
            admissible_actions=reformatted_admissible,
        )
    else:
        user_content = ALFWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=r,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history="\n".join(history[-HISTORY_LENGTH:]),
            current_step=r + 1,
            current_observation=format_obs,
            admissible_actions=reformatted_admissible,
        )
    return user_content, format_obs


async def teacher_lookahead(
    ask: Ask,
    game_file: str,
    context: List[Dict[str, str]],
    prior_actions: List[str],
    step: int,
    gate: Dict,
    student_action: str = "",
) -> Dict:
    """Let the teacher play `gate["window_size"]` steps from step `step` (0-indexed) and judge them.

    `context` is the student's conversation through step `step`'s user message
    (2 * step + 1 messages); `prior_actions` are the student's actions of steps 0..step-1.
    Returns `progress` (True / False / None = unparseable) and what happened on the way.
    """
    env, observation, info, history, task_description, n_replayed, done = (
        _create_alfworld_env_with_checkpoint(game_file, list(prior_actions[:step]), step)
    )
    try:
        if done or n_replayed != step:
            raise RuntimeError(f"replay ended after {n_replayed} of {step} steps (done={done})")
        user_content, format_obs = build_user_content(observation, info, history, task_description, step)
        replay_ok = user_content == context[-1]["content"]
        # The teacher continues the conversation the student actually had.
        ctx = list(context)
        teacher_actions: List[str] = []
        finished = False
        for k in range(gate["window_size"]):
            r = step + k
            if k > 0:
                user_content, format_obs = build_user_content(observation, info, history, task_description, r)
                ctx.append({"role": "user", "content": user_content})
            text = await ask(ctx, gate["teacher_rollout_temperature"], gate["teacher_rollout_max_tokens"])
            ctx.append({"role": "assistant", "content": text})
            action = parse_action(text)
            teacher_actions.append(action)
            history.append(_format_history(format_obs, r + 1, action))
            observation, _, done, info = env.step(action)
            if done:
                finished = True
                break
    finally:
        env.close()

    if finished:  # the teacher completed the task: progress by outcome, no question needed
        progress, progress_text = True, ""
    else:
        progress_text = await ask(
            progress_messages(ctx, step, step + len(teacher_actions) - 1),
            gate["progress_temperature"],
            gate["progress_max_tokens"],
        )
        progress = parse_yes_no(progress_text)
    return {
        "progress": progress,
        "asked": not finished,
        "finished": finished,
        "steps": len(teacher_actions),
        "actions": teacher_actions,
        "same_first_action": teacher_actions[0] == student_action,
        "replay_ok": replay_ok,
        "progress_text_len": len(progress_text),
    }


def lookahead_weight(gate: Dict, disagree: bool, teacher_progress: Optional[bool]) -> float:
    """OPD weight of one step. `teacher_progress`: True / False, or None when unknown
    (unparseable answer, failed look-ahead, or not run because the teacher agreed)."""
    if not disagree:
        return 0.0
    if teacher_progress is False:
        return gate["teacher_no_progress_weight"]
    return gate["teacher_progress_weight"]


def deferred_payload(
    memory: List[Dict[str, str]],
    step: int,
    n_steps: int,
    actions: List[str],
    final_reward: float,
    gate: Dict,
    game_file: str,
) -> Dict:
    """What the trainer needs to gate and score turn `step` by itself (defer_teacher mode):
    the conversation through the step's user message, and the game file plus the student's
    earlier actions to replay the game for the teacher's look-ahead. `has_window` is False:
    the student's forward window is not used here."""
    return {
        "step": step,
        "n_steps": n_steps,
        "action": actions[step],
        "final_reward": float(final_reward),
        "has_window": False,
        "memory_z": zlib.compress(json.dumps(memory[: 2 * step + 1]).encode()),
        "gate": gate,
        "game_file": game_file,
        "prior_actions": list(actions[:step]),
    }


def lookahead_task(
    teacher_name: str,
    namespace: str,
    game_file: str,
    memory_z: bytes,
    prior_actions: List[str],
    step: int,
    gate: Dict,
    student_action: str,
) -> Dict:
    """One teacher look-ahead as a Ray task (trainer side): reaches the teacher engine by its
    actor name, so the game replay runs in this task's own process."""
    import ray

    teacher = ray.get_actor(teacher_name, namespace=namespace)

    async def ask(messages, temperature, max_tokens):
        for attempt in range(CALL_ATTEMPTS):
            try:
                responses = ray.get(
                    teacher.chat.remote(
                        messages, lora_request=None, temperature=temperature, max_tokens=max_tokens, n=1
                    )
                )
                return responses[0].response_text or ""
            except Exception:
                if attempt == CALL_ATTEMPTS - 1:
                    raise
                time.sleep(2.0)

    # The teacher starts from the conversation through this step's user message (a payload
    # may hold more: the student's following steps, for the student look-ahead).
    context = json.loads(zlib.decompress(memory_z))[: 2 * step + 1]
    return asyncio.run(
        teacher_lookahead(ask, game_file, context, prior_actions, step, gate, student_action)
    )


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow_teacher_lookahead")
class OPDGatedAlfworldWorkflowTeacherLookahead(OPDGatedAlfworldWorkflowAgreeLookahead):
    """Disagreement-gated OPD where the teacher plays ahead from each step it disagrees with;
    the step is downweighted if the teacher judges its own steps as not making progress.
    See the module docstring."""

    def __init__(self, *, task: Task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        args = task.workflow_args
        self.gate_mode = GATE_MODE
        self.window_size = args.get("progress_window_size", DEFAULT_WINDOW_SIZE)
        self.teacher_progress_weight = args.get("teacher_progress_weight", 1.0)
        self.teacher_no_progress_weight = args.get("teacher_no_progress_weight", 0.5)
        # Decoding of the teacher's own steps.
        self.teacher_rollout_temperature = args.get("teacher_rollout_temperature", 0.0)
        self.teacher_rollout_max_tokens = args.get("teacher_rollout_max_tokens", 512)

    def gate_config(self) -> Dict:
        return {
            "gate_mode": GATE_MODE,
            "window_size": self.window_size,
            "teacher_progress_weight": self.teacher_progress_weight,
            "teacher_no_progress_weight": self.teacher_no_progress_weight,
            "teacher_rollout_temperature": self.teacher_rollout_temperature,
            "teacher_rollout_max_tokens": self.teacher_rollout_max_tokens,
            "consistency_temperature": self.consistency_temperature,
            "consistency_max_tokens": self.consistency_max_tokens,
            "progress_temperature": self.progress_temperature,
            "progress_max_tokens": self.progress_max_tokens,
            "temperature": self.temperature,
        }

    # Hooks for variants that add criteria to the teacher's look-ahead
    # (OPD_gated_workflow_student_teacher_lookahead.py).
    def _deferred_payload(self, memory, step, n_steps, actions, gate) -> Dict:
        return deferred_payload(memory, step, n_steps, actions, self._final_reward, gate, self.task_desc)

    async def _student_not_progress(self, memory, disagreed, n_total_steps, limited) -> Dict[int, bool]:
        """step -> "the student's own next steps make no progress", for the steps where a
        variant needs it. Not used by the plain teacher look-ahead."""
        return {}

    def _step_weight(self, gate, disagree, result, student_not_progress) -> float:
        return lookahead_weight(gate, disagree, result["progress"] if result else None)

    def _full_weight(self) -> float:
        return self.teacher_progress_weight

    def _defer_lookahead(
        self, turn_responses: List[Experience], actions: List[str], memory: List[Dict[str, str]]
    ) -> List[Experience]:
        gate, n = self.gate_config(), len(turn_responses)
        for i, response in enumerate(turn_responses):
            if response.info is None:
                response.info = {}
            response.info["opd_deferred"] = self._deferred_payload(memory, i, n, actions, gate)
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

        # ---- Pass 1: the student's rollout, as in the other gated workflows. ----
        for r in range(self.max_env_steps):
            user_content, format_obs = build_user_content(observation, info, history, task_description, r)
            memory = memory + [{"role": "user", "content": user_content}]

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            if response.logprobs is None:
                raise RuntimeError(
                    "OPDGatedAlfworldWorkflowTeacherLookahead requires student model to return "
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
            return self._defer_lookahead(turn_responses, actions, memory)

        # ---- Pass 2: agreement for every step, then the teacher's look-ahead where it
        # disagrees. Teacher requests are bounded as in the other gated workflows. ----
        gate = self.gate_config()
        limiter = asyncio.Semaphore(self.teacher_parallel_prompts)

        async def limited(coro):
            async with limiter:
                return await coro

        async def ask(messages, temperature, max_tokens):
            responses = await limited(
                self.teacher_model.chat_async(messages, temperature=temperature, max_tokens=max_tokens, n=1)
            )
            return responses[0].response_text or ""

        async def lookahead(t):
            try:
                return await teacher_lookahead(
                    ask, self.task_desc, memory[: 2 * t + 1], actions, t, gate, actions[t]
                )
            except Exception as e:  # keep the episode: the step then gets the full weight
                self.logger.warning(f"teacher look-ahead failed at step {t}: {e!r}")
                return None

        agree_answers = await asyncio.gather(
            *[limited(self._ask_teacher_agree(memory, t, actions[t])) for t in range(n_total_steps)]
        )
        disagreed = [t for t in range(n_total_steps) if agree_answers[t][0] is not True]
        lookahead_results, student_np = await asyncio.gather(
            asyncio.gather(*[lookahead(t) for t in disagreed]),
            self._student_not_progress(memory, disagreed, n_total_steps, limited),
        )
        lookaheads = dict(zip(disagreed, lookahead_results))

        gate_weights: List[float] = []
        for t in range(n_total_steps):
            result = lookaheads.get(t)
            disagree = t in lookaheads
            weight = self._step_weight(gate, disagree, result, student_np.get(t))
            response = turn_responses[t]
            if response.metrics is None:
                response.metrics = {}
            response.metrics["opd_gate_disagree"] = float(disagree)
            if result is not None:
                response.metrics["opd_gate_teacher_no_progress"] = float(result["progress"] is False)
            if t in student_np:
                response.metrics["opd_gate_not_progress"] = float(student_np[t])
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
            teacher_resp_logprobs = all_teacher_logprobs[i][response.prompt_length - 1 :]
            student_resp_logprobs = response.logprobs
            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )
            # Raw (unblended) KL, for reporting.
            per_turn_kl_sums.append((student_resp_logprobs - teacher_resp_logprobs).sum().item())
            weight = gate_weights[i]
            response.teacher_logprobs = student_resp_logprobs + weight * (
                teacher_resp_logprobs - student_resp_logprobs
            )
            response.teacher_logprobs_valid_mask = torch.full(
                (len(teacher_resp_logprobs),), True, dtype=torch.bool
            )
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i
            response.metrics["opd_gate_weight"] = weight

        if turn_responses:
            last = turn_responses[-1].metrics
            last["env_rounds"] = self._env_rounds
            last["env_done"] = 1.0 if self._env_done else 0.0
            last["kl_divergence"] = sum(per_turn_kl_sums)
            last["opd_gate_apply_rate"] = sum(gate_weights) / n_total_steps
            full = self._full_weight()
            last["opd_gate_full_rate"] = sum(w == full for w in gate_weights) / n_total_steps
            last["opd_gate_half_rate"] = sum(0.0 < w < full for w in gate_weights) / n_total_steps
            last["opd_gate_none_rate"] = sum(w == 0.0 for w in gate_weights) / n_total_steps
            last["opd_gate_disagree_rate"] = len(disagreed) / n_total_steps
            last["consistency_parse_success_rate"] = (
                1.0 - sum(a[0] is None for a in agree_answers) / n_total_steps
            )
            last.update(lookahead_metrics(list(lookaheads.values())))
        return turn_responses


def lookahead_metrics(results: List[Optional[Dict]], prefix: str = "teacher_lookahead_") -> Dict[str, float]:
    """Summary of the teacher look-aheads of one episode / one batch (None = failed)."""
    ok = [r for r in results if r is not None]
    metrics = {prefix + "count": len(results), prefix + "errors": len(results) - len(ok)}
    if ok:
        n = len(ok)
        asked = [r for r in ok if r["asked"]]
        metrics.update({
            prefix + "no_progress_rate": sum(r["progress"] is False for r in ok) / n,
            prefix + "finished_rate": sum(r["finished"] for r in ok) / n,
            prefix + "steps_mean": sum(r["steps"] for r in ok) / n,
            prefix + "same_first_action_rate": sum(r["same_first_action"] for r in ok) / n,
            prefix + "replay_ok_rate": sum(r["replay_ok"] for r in ok) / n,
        })
        if asked:
            metrics[prefix + "progress_parse_success_rate"] = (
                1.0 - sum(r["progress"] is None for r in asked) / len(asked)
            )
    return metrics
