# -*- coding: utf-8 -*-
"""Vanilla OPD on ScienceWorld with the rollout of our gated ScienceWorld runs.

The student's rollout is OPD_gated_scienceworld_workflow_student_teacher_lookahead's, unchanged
(DASH-OPD's prompts, arXiv 2607.29078 Appendix L, TCOD's compact action list, the full
conversation as memory, reward = best score / 100), so the only difference from those runs is
the gate: none. The teacher scores every turn and every turn gets the full weight, as
OPD_alfworld_workflow_fullmemory on ALFWorld.

defer_teacher: true (TPU trainer): each turn carries gate_mode "always", and
trinity/trainer/tunix/teacher_gate.py only has the teacher score it (no question).
"""

from typing import List

import torch

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.TCOD.scienceworld.OPD_gated_workflow_student_teacher_lookahead import (
    OPDGatedScienceworldWorkflowStudentTeacherLookahead,
)


@WORKFLOWS.register_module("OPD_scienceworld_workflow_dashtpl")
class OnPolicyDistillScienceworldWorkflowDashTpl(OPDGatedScienceworldWorkflowStudentTeacherLookahead):
    """Vanilla OPD (no gate) with the gated ScienceWorld workflow's rollout. See the module docstring."""

    async def _gate_and_score(self, turn_responses, actions, memory) -> List[Experience]:
        per_turn_kl_sums: List[float] = []
        for i, response in enumerate(turn_responses):
            if response.metrics is None:
                response.metrics = {}
            if self.defer_teacher:
                if response.info is None:
                    response.info = {}
                response.info["opd_deferred"] = {
                    "step": i,
                    "gate": {"gate_mode": "always", "temperature": self.temperature},
                }
            else:
                teacher_logprobs = await self.teacher_model.logprobs_async(
                    tokens=response.tokens.tolist(),  # full input = prefix + student's response
                    temperature=self.temperature,
                )
                teacher_resp_logprobs = teacher_logprobs[response.prompt_length - 1:]
                assert len(teacher_resp_logprobs) == len(response.logprobs), (
                    f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                    f"student_logprobs={len(response.logprobs)}"
                )
                response.teacher_logprobs = teacher_resp_logprobs
                response.teacher_logprobs_valid_mask = torch.full(
                    (len(teacher_resp_logprobs),), True, dtype=torch.bool
                )
                per_turn_kl_sums.append((response.logprobs - teacher_resp_logprobs).sum().item())
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i
        if turn_responses:
            last = turn_responses[-1].metrics
            last["env_rounds"] = self._env_rounds
            last["env_done"] = 1.0 if self._env_done else 0.0
            # per episode, as the gated ScienceWorld workflow logs them
            last["env_score"] = self._final_reward
            last["env_success"] = self._completed()
            if per_turn_kl_sums:
                last["kl_divergence"] = sum(per_turn_kl_sums)
        return turn_responses
