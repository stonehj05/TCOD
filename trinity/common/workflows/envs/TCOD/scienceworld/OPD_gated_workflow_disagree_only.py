# -*- coding: utf-8 -*-
"""Ablation of our ScienceWorld method (user, 2026-10-24): the disagreement gate ALONE.

OPD_gated_scienceworld_workflow_student_teacher_lookahead with both look-ahead components removed, as the
WebShop ablation (envs/TCOD/webshop/OPD_gated_workflow_disagree_only.py): no student look-ahead (the
teacher's progress question on the student's next `progress_window_size` steps, and the episode-outcome
check for the final window) and no teacher look-ahead (the teacher playing ahead from a step it disagrees
with). What is left is the agreement question: per student turn the teacher is asked, with the same prompt
and decoding, whether it agrees with the student's action;
    teacher disagrees ("No" or unparseable, fail-safe as in the full method) -> weight 1.0
    teacher agrees                                                           -> weight 0.0
(the full method: 0.0 when the teacher agrees; 1.0 / 0.7 / 0.3 / 0.0 by the two look-aheads when it
disagrees). The weight blends the turn's teacher logprobs toward the student's own, as in the full method.
Everything else is the full method's: the rollout (TCOD's ScienceWorld game with DASH-OPD's prompts, full
conversation, 30 steps), the payload and the deferred scoring in the trainer.

defer_teacher: true only: the gate runs in the trainer (trinity/trainer/tunix/teacher_gate.py, gate_mode
"disagree_only": agreement question only, then gate_weight with both disagree weights 1.0).
"""

from typing import Dict

from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.scienceworld.OPD_gated_workflow_student_teacher_lookahead import (
    OPDGatedScienceworldWorkflowStudentTeacherLookahead,
)

GATE_MODE = "disagree_only"
DISAGREE_WEIGHT = 1.0


@WORKFLOWS.register_module("OPD_gated_scienceworld_workflow_disagree_only")
class OPDGatedScienceworldWorkflowDisagreeOnly(OPDGatedScienceworldWorkflowStudentTeacherLookahead):
    """The disagreement gate alone on ScienceWorld. See the module docstring."""

    def __init__(self, *, task: Task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        if not self.defer_teacher:
            raise ValueError("OPD_gated_scienceworld_workflow_disagree_only needs workflow_args.defer_teacher: true")

    def gate_config(self) -> Dict:
        gate = super().gate_config()
        gate.update(
            gate_mode=GATE_MODE,
            # gate_weight(): 0 when the teacher agrees, else one of these two (no progress signal here)
            disagree_no_progress_weight=DISAGREE_WEIGHT,
            disagree_progress_weight=DISAGREE_WEIGHT,
        )
        return gate
