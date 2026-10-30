# -*- coding: utf-8 -*-
"""UOPD's OPD workflow on ScienceWorld: the ALFWorld OPD workflow (../alfworld/OPD_workflow.py) with only
its environment layer swapped. Not in the UOPD release (ALFWorld and WebShop only); ported here.

Kept from UOPD's ALFWorld workflow: one user message per turn holding the last HISTORY_LENGTH (2) steps
(no full conversation, no system prompt), the no-history template on the first turn only, the hooks
UOPD uses (_pre_student_gen / _post_student_gen / _select_env_response) and the scoring.
From TCOD's ScienceWorld workflows (TCOD/scienceworld/utils.py, as TCOD-b2f trains): the prompt
templates (<think> tags, action commands + objects), the strict <action> parser, game loading
(simplification "easy") and the reward, best score / 100. Success (env_done) = the task completed
(score 100).
"""

from typing import List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.scienceworld.utils import (
    HISTORY_LENGTH,
    SCIWORLD_TEMPLATE,
    SCIWORLD_TEMPLATE_NO_HIS,
    _create_scienceworld_env,
    _format_history,
    _get_compact_action_info,
    _reset_scienceworld_env,
    format_observation,
    parse_action,
)
from trinity.common.workflows.envs.UOPD.alfworld.OPD_workflow import (
    OnPolicyDistillVerlAgentAlfworldWorkflow,
)

COMPLETED_SCORE = 100


def build_user_content(env, observation: str, history: List[str], task_description: str, r: int):
    """Turn r's single user message (UOPD's 2-step memory, TCOD's ScienceWorld templates)."""
    format_obs = format_observation(observation)
    action_templates, objects = _get_compact_action_info(env)
    actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
    objects = ", ".join(f"'{s}'" for s in objects)
    # No-history template on the first turn only, as UOPD's ALFWorld workflow: every later turn's
    # message must carry the previous steps, since it is the model's only memory.
    if len(history) == 0:
        return SCIWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=format_obs,
            action_templates=actions,
            objects=objects,
        ), format_obs
    return SCIWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=r,
        history_length=min(HISTORY_LENGTH, len(history)),
        action_history="\n".join(history[-HISTORY_LENGTH:]),
        current_step=r + 1,
        current_observation=format_obs,
        action_templates=actions,
        objects=objects,
    ), format_obs


@WORKFLOWS.register_module("UOPD_OPD_scienceworld_workflow")
class OnPolicyDistillScienceWorldWorkflow(OnPolicyDistillVerlAgentAlfworldWorkflow):
    """UOPD's OPD workflow with a ScienceWorld game (task_desc: TCOD's task JSON)."""

    async def run_async(self) -> List[Experience]:
        env = _create_scienceworld_env(self.task_desc, max_env_steps=self.max_env_steps)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        observation, info, task_description = _reset_scienceworld_env(env)
        self._env_done = False
        self._env_rounds = 0
        best_score = info.get("score", 0)

        history: List[str] = []
        turn_responses: List[Experience] = []
        turn_memories: List = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            user_content, format_obs = build_user_content(env, observation, history, task_description, r)
            memory = [{"role": "user", "content": user_content}]
            turn_memories.append(memory)

            await self._pre_student_gen(len(turn_memories) - 1, memory)

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            if response.logprobs is None:
                raise RuntimeError(
                    "UOPD_OPD_scienceworld_workflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            await self._post_student_gen(len(turn_responses) - 1, response, memory)

            response = await self._select_env_response(len(turn_responses) - 1, response, memory)
            turn_responses[-1] = response
            response_text = response.response_text or ""

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            best_score = max(best_score, info.get("score", best_score + reward))
            if done:
                self._env_rounds = r + 1
                break
        else:
            self._env_rounds = self.max_env_steps

        self._env_done = best_score >= COMPLETED_SCORE
        self._final_reward = best_score / 100.0
        return await self._score_and_finalize(turn_responses, turn_memories)
