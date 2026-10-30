# -*- coding: utf-8 -*-
"""Gated OPD workflow for WebShop -- DISAGREEMENT + STUDENT LOOK-AHEAD + TEACHER LOOK-AHEAD.

The ALFWorld method of OPD_gated_workflow_student_teacher_lookahead.py (agreement question, then
for the steps the teacher disagrees with both the student's and the teacher's own next
`progress_window_size` steps, weighted by both_weight / teacher_only_weight / student_only_weight /
neither_weight), on WebShop, as the ScienceWorld variant (scienceworld/OPD_gated_workflow_student_
teacher_lookahead.py). The questions, their parsing and the weights are the ALFWorld module's;
only the game differs:

  - The rollout is TCOD's OPD_webshop_workflow (webshop/OPD_workflow.py: session id = task, the
    admissible actions listed every turn, invalid actions answered with validate_action's message
    without stepping the game, reward = the shop's score of the purchase, 0 without one) with
    DASH-OPD's WebShop prompts (envs/DASH/webshop/utils.py, copied unchanged from DASH-OPD:
    build_webshop_prompt with history_length 2, initial_no_history_turns 3 and the full
    conversation as memory, as DASH-OPD's WebShop configs). The same prompts in the teacher's
    look-ahead and in eval.
  - Games are played on shared WebShop servers (env_server.py): one catalog per server actor
    instead of one per env. The teacher's look-ahead replays the student's earlier actions on a
    new game of the same session (WebShop is deterministic given the session and the server's
    seeded prices); `teacher_lookahead_replay_ok` checks that the replayed prompt equals the
    student's.
  - Outcome. A WebShop game ends when the agent buys something, scored in [0, 1] by how well
    the product matches the goal; "the task is completed" means a score of 1 (WebShop's success
    rate, as score 100 on ScienceWorld):
      * the student's final `progress_window_size` steps (no forward window) count as not
        making progress unless the student's purchase scored at least `progress_score`;
      * a teacher look-ahead that buys a product scoring at least `progress_score` is progress
        without asking; one that buys a lower-scoring product is no progress, without asking.
    `progress_score` (workflow arg) defaults to 1.0 = completed; most purchases earn partial
    credit (the teacher's: ~24% score 1, ~70% score >= 0.5), so 0.5 counts a reasonable match
    as progress.

defer_teacher: true (TPU trainer only), as on ALFWorld: the explorer only plays the game and
each turn's payload carries `env: "webshop"`, so the trainer's trinity/trainer/tunix/teacher_gate.py
replays it with `lookahead_task` below.

Metrics per episode (logged as batch means): env_score = the purchase's score (also the reward,
0 without a purchase), env_success = 1 if it is 1.0.
"""

import asyncio
import json
import time
import zlib
from typing import Dict, List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.DASH.webshop.utils import (
    _extract_task_description,
    _format_available_actions,
    _format_history,
    build_webshop_prompt,
    format_observation,
    parse_action,
    validate_action,
)
from trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_agree_lookahead import (
    parse_yes_no,
    progress_messages,
)
from trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_student_teacher_lookahead import (
    OPDGatedAlfworldWorkflowStudentTeacherLookahead,
    deferred_payload as _alfworld_deferred_payload,
)
from trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_teacher_lookahead import (
    Ask,
    CALL_ATTEMPTS,
)
from trinity.common.workflows.envs.TCOD.webshop.env_server import RemoteWebShopEnv

ENV = "webshop"
COMPLETED_SCORE = 1.0
# DASH-OPD's WebShop configs (configurations/webshop/*.yaml workflow_args)
HISTORY_LENGTH = 2
INITIAL_NO_HISTORY_TURNS = 3


def build_user_content(observation: str, available_actions: Dict, history: List[str], task_description: str, r: int):
    """The user message of step `r` (0-indexed) and the formatted observation: DASH-OPD's WebShop
    prompt (build_webshop_prompt, as its OPD_webshop_workflow calls it)."""
    format_obs = format_observation(observation)
    user_content = build_webshop_prompt(
        task_description=task_description,
        step_count=r,
        history=history,
        history_length=HISTORY_LENGTH,
        initial_no_history_turns=INITIAL_NO_HISTORY_TURNS,
        current_observation=format_obs,
        available_actions=_format_available_actions(available_actions),
        use_initial_no_history_template=True,  # retain_full_chat_history: true
    )
    return user_content, format_obs


async def play(env: RemoteWebShopEnv, observation: str, action: str):
    """One step as TCOD's OPD_webshop_workflow plays it: an invalid action does not reach the game;
    its observation is validate_action's message. Returns (observation, reward, done)."""
    valid, error_message = validate_action(action, env.get_available_actions())
    if not valid:
        return error_message, 0.0, False
    observation, reward, done, _ = await env.step(action)
    return observation, reward, done


async def replay_game(session: int, actions: List[str], step: int):
    """A new game of the session after the student's first `step` actions.
    Returns (env, observation, history, task_description, n_replayed, done)."""
    env = RemoteWebShopEnv(f"lookahead/{session}/{step}")
    observation = await env.reset(session=session)
    task_description = _extract_task_description(observation)
    history: List[str] = []
    done = False
    for k, action in enumerate(actions[:step]):
        history.append(_format_history(format_observation(observation), k + 1, action))
        observation, _, done = await play(env, observation, action)
        if done:
            break
    return env, observation, history, task_description, len(history), done


async def teacher_lookahead(
    ask: Ask,
    task_desc: str,
    context: List[Dict[str, str]],
    prior_actions: List[str],
    step: int,
    gate: Dict,
    student_action: str = "",
) -> Dict:
    """The ALFWorld `teacher_lookahead` on WebShop: the teacher plays `gate["window_size"]` steps
    from step `step` (0-indexed) in a replayed copy of the game."""
    env, observation, history, task_description, n_replayed, done = await replay_game(
        int(task_desc), prior_actions, step
    )
    try:
        progress_score = gate.get("progress_score", COMPLETED_SCORE)
        if done or n_replayed != step:
            raise RuntimeError(f"replay ended after {n_replayed} of {step} steps (done={done})")
        user_content, format_obs = build_user_content(
            observation, env.get_available_actions(), history, task_description, step
        )
        replay_ok = user_content == context[-1]["content"]
        ctx = list(context)
        teacher_actions: List[str] = []
        ended, completed, progressed = False, False, False
        for k in range(gate["window_size"]):
            r = step + k
            if k > 0:
                user_content, format_obs = build_user_content(
                    observation, env.get_available_actions(), history, task_description, r
                )
                ctx.append({"role": "user", "content": user_content})
            text = await ask(ctx, gate["teacher_rollout_temperature"], gate["teacher_rollout_max_tokens"])
            ctx.append({"role": "assistant", "content": text})
            action = parse_action(text)
            teacher_actions.append(action)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done = await play(env, observation, action)
            if done:
                ended, completed, progressed = True, reward >= COMPLETED_SCORE, reward >= progress_score
                break
    finally:
        await env.close()

    if ended:  # the game is over: progress by outcome (a purchase scoring >= progress_score), no question needed
        progress, progress_text = progressed, ""
    else:
        progress_text = await ask(
            progress_messages(ctx, step, step + len(teacher_actions) - 1),
            gate["progress_temperature"],
            gate["progress_max_tokens"],
        )
        progress = parse_yes_no(progress_text)
    return {
        "progress": progress,
        "asked": not ended,
        "finished": completed,
        "steps": len(teacher_actions),
        "actions": teacher_actions,
        "same_first_action": teacher_actions[0] == student_action,
        "replay_ok": replay_ok,
        "progress_text_len": len(progress_text),
    }


def lookahead_task(
    teacher_name: str,
    namespace: str,
    task_desc: str,
    memory_z: bytes,
    prior_actions: List[str],
    step: int,
    gate: Dict,
    student_action: str,
) -> Dict:
    """One teacher look-ahead as a Ray task (trainer side), as the ALFWorld `lookahead_task`."""
    import ray

    teacher = ray.get_actor(teacher_name, namespace=namespace)

    async def ask(messages, temperature, max_tokens):
        for attempt in range(CALL_ATTEMPTS):
            try:
                responses = await teacher.chat.remote(
                    messages, lora_request=None, temperature=temperature, max_tokens=max_tokens, n=1
                )
                return responses[0].response_text or ""
            except Exception:
                if attempt == CALL_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(2.0)

    context = json.loads(zlib.decompress(memory_z))[: 2 * step + 1]
    return asyncio.run(teacher_lookahead(ask, task_desc, context, prior_actions, step, gate, student_action))


@WORKFLOWS.register_module("OPD_gated_webshop_workflow_student_teacher_lookahead")
class OPDGatedWebshopWorkflowStudentTeacherLookahead(OPDGatedAlfworldWorkflowStudentTeacherLookahead):
    """The ALFWorld student + teacher look-ahead gate on WebShop. See the module docstring."""

    def __init__(self, *, task: Task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 15)  # TCOD's WebShop default
        # a purchase scoring at least this counts as progress in the gate's outcome checks
        self.progress_score = float(task.workflow_args.get("progress_score", COMPLETED_SCORE))
        self._env = None

    def gate_config(self) -> Dict:
        gate = super().gate_config()
        gate["progress_score"] = self.progress_score
        return gate

    def _progressed(self) -> float:
        """1.0 if the student's purchase scored at least `progress_score`, else 0.0: the episode
        outcome the gate uses for the student's final window."""
        return 1.0 if self._final_reward >= self.progress_score else 0.0

    def _completed(self) -> float:
        """1.0 if the student completed the task (bought a product scoring 1), else 0.0: the
        env_success metric (the reward stays the score)."""
        return 1.0 if self._final_reward >= COMPLETED_SCORE else 0.0

    async def run_async(self) -> List[Experience]:
        # one game on a shared server per workflow instance, reused across tasks (TCOD's
        # OPD_webshop_workflow keeps one env per instance too)
        if self._env is None:
            self._env = RemoteWebShopEnv(f"runner/{id(self)}/{time.time()}")
        return await self._run_episode(self._env)

    async def _rollout(self, env):
        """Pass 1: TCOD's OPD_webshop_workflow rollout with DASH-OPD's prompts."""
        session_id = int(self.task_desc)
        observation = await env.reset(session=session_id)
        task_description = _extract_task_description(observation)
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0
        history: List[str] = []
        turn_responses: List[Experience] = []
        actions: List[str] = []
        memory: List[Dict[str, str]] = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            user_content, format_obs = build_user_content(
                observation, env.get_available_actions(), history, task_description, r
            )
            memory = memory + [{"role": "user", "content": user_content}]

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            if response.logprobs is None:
                raise RuntimeError(
                    "OPDGatedWebshopWorkflowStudentTeacherLookahead requires student model to "
                    "return logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )

            action = parse_action(response_text)
            turn_responses.append(response)
            actions.append(action)

            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done = await play(env, observation, action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = float(reward)
                break
        else:
            self._env_rounds = self.max_env_steps
        return turn_responses, actions, memory

    async def _gate_and_score(self, turn_responses, actions, memory):
        result = await super()._gate_and_score(turn_responses, actions, memory)
        if result:
            result[-1].metrics["env_score"] = self._final_reward
            result[-1].metrics["env_success"] = self._completed()
        return result

    async def _teacher_lookahead(self, ask, context, actions, t, gate) -> Dict:
        return await teacher_lookahead(ask, self.task_desc, context, actions, t, gate, actions[t])

    def _deferred_payload(self, memory, step, n_steps, actions, gate) -> Dict:
        # final_reward in the payload is the outcome the trainer's gate uses: progressed or not
        payload = _alfworld_deferred_payload(memory, step, n_steps, actions, self._progressed(), gate, self.task_desc)
        payload["env"] = ENV
        return payload

    async def _student_not_progress(self, memory, disagreed, n_total_steps, limited) -> Dict[int, bool]:
        windowed = [t for t in disagreed if t + self.window_size < n_total_steps]
        answers = await asyncio.gather(
            *[limited(self._ask_teacher_progress(memory, t, t + self.window_size - 1)) for t in windowed]
        )
        result = {t: says_progress is not True for t, (says_progress, _) in zip(windowed, answers)}
        for t in disagreed:
            if t not in result:  # final window: the episode outcome (progressed or not)
                result[t] = not self._progressed()
        return result
