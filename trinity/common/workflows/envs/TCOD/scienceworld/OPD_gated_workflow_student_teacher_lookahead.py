# -*- coding: utf-8 -*-
"""Gated OPD workflow for ScienceWorld -- DISAGREEMENT + STUDENT LOOK-AHEAD + TEACHER LOOK-AHEAD.

The ALFWorld method of OPD_gated_workflow_student_teacher_lookahead.py (see its module
docstring: agreement question, then for the steps the teacher disagrees with both the
student's and the teacher's own next `progress_window_size` steps, weighted by
both_weight / teacher_only_weight / student_only_weight / neither_weight), on ScienceWorld.
The questions, their parsing and the weights are the ALFWorld module's functions; only the
game differs:

  - The rollout is TCOD's OPD_scienceworld_workflow (scienceworld/OPD_workflow.py) with
    DASH-OPD's prompts (arXiv 2607.29078, Appendix L; build_user_content below): the compact
    action list (action templates + objects), the full conversation as memory, the last 2
    steps summarized from turn 4, and reward = best score / 100.
  - The teacher's look-ahead replays the student's earlier actions in a fresh copy of the game
    (replay_game below) and continues the student's conversation with the same prompts.
    ScienceWorld is deterministic when the copy is created exactly as the student's game
    (generateGoldPath off): `teacher_lookahead_replay_ok` checks that the replayed prompt equals
    the student's. Two things make a replay differ otherwise: TCOD's
    _create_scienceworld_env_with_checkpoint loads the game with the gold path ON (changes the
    numbered option lists and object names), and ScienceWorld names objects differently unless the
    object list is requested every turn as the student's game does (~23% of real turns differed in
    the object list). replay_game does both as the student's game: 149/150 real turns matched.
  - Outcome. ScienceWorld scores partial credit and also ends a game that has failed, so
    "the task is completed" means a score of 100 here:
      * the student's final `progress_window_size` steps (no forward window) count as not
        making progress unless the student completed the task (ALFWorld: unless it won);
      * a teacher look-ahead that completes the task is progress without asking; one that
        ends the game without completing it is no progress, without asking (ALFWorld games
        only end by winning).

defer_teacher: true (TPU trainer only), as on ALFWorld: the explorer only plays the game and
each turn's payload carries `env: "scienceworld"`, so the trainer's
trinity/trainer/tunix/teacher_gate.py replays it with `lookahead_task` below.

Metrics per episode (logged as batch means): env_score = best score / 100 (also the reward),
env_success = 1 if the task was completed (score 100). env_done is ScienceWorld's own end of
game, which also happens on failure, so it is not the success rate here (on ALFWorld it is).
"""

import asyncio
import json
import time
import zlib
from typing import Dict, List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS, Task
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
from trinity.common.workflows.envs.TCOD.scienceworld.utils import (
    HISTORY_LENGTH,
    _create_scienceworld_env,
    _format_history,
    _get_compact_action_info,
    _reset_scienceworld_env,
    format_observation,
    parse_action,
)

ENV = "scienceworld"
COMPLETED_SCORE = 100

# The prompts of DASH-OPD (arXiv 2607.29078, Appendix L), used in training, the teacher's look-ahead and eval
# (scripts/tpu/eval/client/13_evaluate_sciworld.py builds its prompts with build_user_content below).
# Laid out line by line as the ALFWorld templates of alfworld/utils.py (the paper's boxes have the same
# wording as those; their line breaks are not recoverable from the paper). The first 3 turns have no history
# summary; from turn 4 the last HISTORY_LENGTH (2) observation-action pairs. TCOD's own ScienceWorld prompts
# (scienceworld/utils.py: "Your ScienceWorld task is", <think> tags, history from turn 3) were used by the
# first run of this method (2026-10-15, deleted).
DASH_SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ScienceWorld text environment.
{task_description}
Your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best valid action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

DASH_SCIWORLD_TEMPLATE = """
You are an expert agent operating in the ScienceWorld text environment.
{task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best valid action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""
NO_HISTORY_TURNS = 3  # DASH-OPD: "The first three turns have no explicit history summary"


def build_user_content(env, observation: str, history: List[str], task_description: str, r: int):
    """The user message of step `r` (0-indexed) and the formatted observation: DASH-OPD's ScienceWorld
    prompts (see above), with TCOD's action list (action templates + objects, from the game)."""
    format_obs = format_observation(observation)
    action_templates, objects = _get_compact_action_info(env)
    reformatted_actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
    reformatted_objects = ", ".join(f"'{s}'" for s in objects)
    if len(history) < NO_HISTORY_TURNS:
        user_content = DASH_SCIWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=format_obs,
            action_templates=reformatted_actions,
            objects=reformatted_objects,
        )
    else:
        user_content = DASH_SCIWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=r,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history="\n".join(history[-HISTORY_LENGTH:]),
            current_step=r + 1,
            current_observation=format_obs,
            action_templates=reformatted_actions,
            objects=reformatted_objects,
        )
    return user_content, format_obs


def replay_game(task_desc: str, actions: List[str], step: int):
    """A fresh copy of the game after the student's first `step` actions, played exactly as the
    student's game: created with generate_gold_path=False (as run_async below), and with the action
    and object lists requested before every step (the student's rollout builds its prompt, which
    asks for them, every turn; ScienceWorld names objects differently later if they are not asked).
    Returns (env, observation, info, history, task_description, n_replayed, done)."""
    env = _create_scienceworld_env(task_desc, generate_gold_path=False)
    observation, info, task_description = _reset_scienceworld_env(env)
    history: List[str] = []
    done = False
    for k, action in enumerate(actions[:step]):
        _get_compact_action_info(env)
        history.append(_format_history(format_observation(observation), k + 1, action))
        observation, _, done, info = env.step(action)
        if done:
            break
    return env, observation, info, history, task_description, len(history), done


async def teacher_lookahead(
    ask: Ask,
    task_desc: str,
    context: List[Dict[str, str]],
    prior_actions: List[str],
    step: int,
    gate: Dict,
    student_action: str = "",
) -> Dict:
    """The ALFWorld `teacher_lookahead` on ScienceWorld: the teacher plays
    `gate["window_size"]` steps from step `step` (0-indexed) in a replayed copy of the game."""
    env, observation, info, history, task_description, n_replayed, done = replay_game(task_desc, prior_actions, step)
    try:
        if done or n_replayed != step:
            raise RuntimeError(f"replay ended after {n_replayed} of {step} steps (done={done})")
        user_content, format_obs = build_user_content(env, observation, history, task_description, step)
        replay_ok = user_content == context[-1]["content"]
        ctx = list(context)
        teacher_actions: List[str] = []
        ended, completed = False, False
        for k in range(gate["window_size"]):
            r = step + k
            if k > 0:
                user_content, format_obs = build_user_content(env, observation, history, task_description, r)
                ctx.append({"role": "user", "content": user_content})
            text = await ask(ctx, gate["teacher_rollout_temperature"], gate["teacher_rollout_max_tokens"])
            ctx.append({"role": "assistant", "content": text})
            action = parse_action(text)
            teacher_actions.append(action)
            history.append(_format_history(format_obs, r + 1, action))
            observation, _, done, info = env.step(action)
            if done:
                ended, completed = True, info.get("score", 0) >= COMPLETED_SCORE
                break
    finally:
        env.close()

    if ended:  # the game is over: progress by outcome, no question needed
        progress, progress_text = completed, ""
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

    context = json.loads(zlib.decompress(memory_z))[: 2 * step + 1]
    return asyncio.run(teacher_lookahead(ask, task_desc, context, prior_actions, step, gate, student_action))


@WORKFLOWS.register_module("OPD_gated_scienceworld_workflow_student_teacher_lookahead")
class OPDGatedScienceworldWorkflowStudentTeacherLookahead(OPDGatedAlfworldWorkflowStudentTeacherLookahead):
    """The ALFWorld student + teacher look-ahead gate on ScienceWorld. See the module docstring."""

    def _completed(self) -> float:
        """1.0 if the student completed the task (score 100), else 0.0: the episode outcome
        used by the gate (the reward stays the score / 100)."""
        return 1.0 if self._best_score >= COMPLETED_SCORE else 0.0

    async def run_async(self) -> List[Experience]:
        env = _create_scienceworld_env(self.task_desc, max_env_steps=self.max_env_steps, generate_gold_path=False)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _rollout(self, env):
        """Pass 1: TCOD's OPD_scienceworld_workflow rollout."""
        observation, info, task_description = _reset_scienceworld_env(env)
        self._env_done = False
        self._env_rounds = 0
        history: List[str] = []
        turn_responses: List[Experience] = []
        actions: List[str] = []
        memory: List[Dict[str, str]] = []
        best_score = info.get("score", 0)

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            user_content, format_obs = build_user_content(env, observation, history, task_description, r)
            memory = memory + [{"role": "user", "content": user_content}]

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            if response.logprobs is None:
                raise RuntimeError(
                    "OPDGatedScienceworldWorkflowStudentTeacherLookahead requires student model to "
                    "return logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )

            action = parse_action(response_text)
            turn_responses.append(response)
            actions.append(action)

            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            best_score = max(best_score, info.get("score", best_score + reward))
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                break
        else:
            self._env_rounds = self.max_env_steps

        self._best_score = best_score
        self._final_reward = best_score / 100.0
        return turn_responses, actions, memory

    async def _gate_and_score(self, turn_responses, actions, memory):
        result = await super()._gate_and_score(turn_responses, actions, memory)
        if result:
            # per episode (the explorer logs their batch means as rollout/env_score/mean and
            # rollout/env_success/mean): the best score / 100, and whether the task was completed
            result[-1].metrics["env_score"] = self._final_reward
            result[-1].metrics["env_success"] = self._completed()
        return result

    async def _teacher_lookahead(self, ask, context, actions, t, gate) -> Dict:
        return await teacher_lookahead(ask, self.task_desc, context, actions, t, gate, actions[t])

    def _deferred_payload(self, memory, step, n_steps, actions, gate) -> Dict:
        # final_reward in the payload is the outcome the trainer's gate uses: completed or not
        payload = _alfworld_deferred_payload(memory, step, n_steps, actions, self._completed(), gate, self.task_desc)
        payload["env"] = ENV
        return payload

    async def _student_not_progress(self, memory, disagreed, n_total_steps, limited) -> Dict[int, bool]:
        windowed = [t for t in disagreed if t + self.window_size < n_total_steps]
        answers = await asyncio.gather(
            *[limited(self._ask_teacher_progress(memory, t, t + self.window_size - 1)) for t in windowed]
        )
        result = {t: says_progress is not True for t, (says_progress, _) in zip(windowed, answers)}
        for t in disagreed:
            if t not in result:  # final window: the episode outcome (completed or not)
                result[t] = not self._completed()
        return result
