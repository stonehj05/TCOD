# -*- coding: utf-8 -*-
"""Gated OPD workflow for AlfWorld -- DISAGREEMENT + STUDENT LOOK-AHEAD + TEACHER LOOK-AHEAD.

Combines the two look-aheads used separately in OPD_gated_workflow_agree_lookahead.py
(`disagree_required`: the student's next steps) and OPD_gated_workflow_teacher_lookahead.py
(the teacher's own steps). Per step:

  (A) agreement: "would you choose this exact action for the current step?" The teacher
      agrees -> weight 0, nothing else is done for that step.
  and, only where the teacher disagrees ("No" or unparseable), both of:
  (B) student look-ahead: "do the student's next `progress_window_size` steps make progress
      toward the task?" (same prompt as the agree+look-ahead workflow; the trajectory's
      final `progress_window_size` steps use the episode outcome instead: failed = no progress).
  (C) teacher look-ahead: the teacher plays `progress_window_size` steps itself from that
      position in a replayed copy of the game and is asked whether its own steps make
      progress (exactly `teacher_lookahead` of the teacher look-ahead workflow).

OPD weight of a step the teacher disagrees with, from the two conditions
    S = the student is NOT making progress      T = the teacher IS making progress
    S and T          -> both_weight     (1.0)   the teacher does better from here: full OPD
    exactly one      -> one_weight      (0.5)
    neither          -> neither_weight  (0.0)   the student progresses and the teacher does not
A step the teacher agrees with always gets 0.

Unknown answers follow the two parent workflows: an unparseable answer to (B) counts as "no
progress" (S holds), and an unparseable answer to (C) or a failed look-ahead counts as
"progress" (T holds), so an unknown never lowers the weight. If the teacher completes the task
inside its window, T holds.

(B) and (C) of all disagreed steps are issued together. The weight is applied as in the other
soft variants: the stored teacher logprobs are blended toward the student's own.

defer_teacher: true (TPU trainer only): as in the parent workflows, the explorer only plays
the game; each turn carries the conversation through the end of its student window plus the
game file and the student's earlier actions, and the trainer runs (A), (B), (C) and the
scoring for the turns it samples (trinity/trainer/tunix/teacher_gate.py).
"""

import asyncio
from typing import Dict, List, Optional

from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.TCOD.alfworld import OPD_gated_workflow_agree_lookahead as _agree
from trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_teacher_lookahead import (
    OPDGatedAlfworldWorkflowTeacherLookahead,
)

GATE_MODE = "student_teacher_lookahead"


def combined_weight(
    gate: Dict, disagree: bool, student_not_progress: Optional[bool], teacher_progress: Optional[bool]
) -> float:
    """OPD weight of one step. `student_not_progress`: True / False (None = unknown, counted
    as not making progress); `teacher_progress`: True / False (None = unknown, counted as
    making progress). See the module docstring."""
    if not disagree:
        return 0.0
    met = int(student_not_progress is not False) + int(teacher_progress is not False)
    return (gate["neither_weight"], gate["one_weight"], gate["both_weight"])[met]


def deferred_payload(
    memory: List[Dict[str, str]], step: int, n_steps: int, actions: List[str], final_reward: float,
    gate: Dict, game_file: str,
) -> Dict:
    """The agree+look-ahead payload (conversation through the student's window, `has_window`)
    plus what the teacher's look-ahead needs to replay the game."""
    payload = _agree.deferred_payload(memory, step, n_steps, actions[step], final_reward, gate)
    payload["game_file"] = game_file
    payload["prior_actions"] = list(actions[:step])
    return payload


@WORKFLOWS.register_module("OPD_gated_alfworld_workflow_student_teacher_lookahead")
class OPDGatedAlfworldWorkflowStudentTeacherLookahead(OPDGatedAlfworldWorkflowTeacherLookahead):
    """Disagreement-gated OPD weighted by both look-aheads: full weight where the student's
    next steps make no progress and the teacher's own steps do. See the module docstring."""

    def __init__(self, *, task: Task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        args = task.workflow_args
        self.gate_mode = GATE_MODE
        self.both_weight = args.get("both_weight", 1.0)
        self.one_weight = args.get("one_weight", 0.5)
        self.neither_weight = args.get("neither_weight", 0.0)

    def gate_config(self) -> Dict:
        gate = super().gate_config()
        gate.update(
            gate_mode=GATE_MODE,
            both_weight=self.both_weight,
            one_weight=self.one_weight,
            neither_weight=self.neither_weight,
        )
        return gate

    def _deferred_payload(self, memory, step, n_steps, actions, gate) -> Dict:
        return deferred_payload(memory, step, n_steps, actions, self._final_reward, gate, self.task_desc)

    async def _student_not_progress(self, memory, disagreed, n_total_steps, limited) -> Dict[int, bool]:
        windowed = [t for t in disagreed if t + self.window_size < n_total_steps]
        answers = await asyncio.gather(
            *[limited(self._ask_teacher_progress(memory, t, t + self.window_size - 1)) for t in windowed]
        )
        # "No" or unparseable -> not making progress (fail-safe, as in the agree+look-ahead gate)
        result = {t: says_progress is not True for t, (says_progress, _) in zip(windowed, answers)}
        for t in disagreed:
            if t not in result:  # final window: the episode outcome
                result[t] = not bool(self._final_reward)
        return result

    def _step_weight(self, gate, disagree, result, student_not_progress) -> float:
        return combined_weight(gate, disagree, student_not_progress, result["progress"] if result else None)

    def _full_weight(self) -> float:
        return self.both_weight
