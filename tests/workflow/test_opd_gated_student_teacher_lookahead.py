"""OPD_gated_alfworld_workflow_student_teacher_lookahead with a scripted student and teacher.

Same set-up as test_opd_gated_teacher_lookahead.py (a real ALFWorld game for the replay, no
TPU, no model). Checks the weight table, the explorer-side gate, and that the trainer-side
(defer_teacher) gate asks the same questions and gives the same weights.
Run: python tests/workflow/test_opd_gated_student_teacher_lookahead.py
"""
import asyncio
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_opd_gated_teacher_lookahead as T  # scripted student / teacher / helpers

import trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_agree_lookahead as P
import trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_student_teacher_lookahead as M
from trinity.common.workflows import WORKFLOWS

N_STEPS, WINDOW = T.N_STEPS, T.WINDOW  # 10 steps, window 3: steps 0..6 have a student window
#               step:  0     1      2     3      4     5      6     7     8      9
# teacher agrees     [yes,  NO,   ???,   NO,   yes,  NO,   yes,  yes,   NO,    NO ]   (T.AGREE)
# student progress          yes    no    ???          yes               (tail: failed episode)
# teacher progress          yes    no    ???          no                yes    no     (T.PROGRESS)
# S (student not progressing) F     T     T            F                 T      T
# T (teacher progressing)     T     F     T            F                 T      F
STUDENT_PROGRESS = {1: True, 2: False, 3: None, 5: True}
EXPECTED = [0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]


class Teacher(T.Teacher):
    """Tells the two progress questions apart by whose steps the prompt shows."""

    def __init__(self):
        super().__init__()
        self.student_progress_prompts = []

    def respond(self, messages):
        last = messages[-1]["content"]
        if "making progress toward" in last and not any(
            m["role"] == "assistant" and "my turn" in m["content"] for m in messages
        ):
            self.student_progress_prompts.append(list(messages))
            start = int(re.search(r"\(steps (\d+)-(\d+)\)", last).group(1)) - 1
            return T.answer(STUDENT_PROGRESS[start])
        return super().respond(messages)


def run(defer, path):
    wf = T.make(M.OPDGatedAlfworldWorkflowStudentTeacherLookahead, defer, path)
    wf.teacher_model = Teacher()
    wf.gate_mode, wf.both_weight, wf.one_weight, wf.neither_weight = M.GATE_MODE, 1.0, 0.5, 0.0
    env = T._create_alfworld_env(path)
    try:
        exps = asyncio.run(wf._run_episode(env))
    finally:
        env.close()
    return wf, exps


def check_weight_table():
    g = {"both_weight": 1.0, "one_weight": 0.5, "neither_weight": 0.0}
    w = M.combined_weight
    assert w(g, True, True, True) == 1.0     # student stuck, teacher progresses
    assert w(g, True, True, False) == 0.5    # both stuck
    assert w(g, True, False, True) == 0.5    # both progress
    assert w(g, True, False, False) == 0.0   # student progresses, teacher does not
    assert w(g, True, None, None) == 1.0     # unknowns never lower the weight
    assert w(g, True, False, None) == 0.5 and w(g, True, None, False) == 0.5
    assert all(w(g, False, s, t) == 0.0 for s in (True, False, None) for t in (True, False, None))
    assert WORKFLOWS.get("OPD_gated_alfworld_workflow_student_teacher_lookahead") is M.OPDGatedAlfworldWorkflowStudentTeacherLookahead
    print("weight table OK")


def check_real_game(path):
    wf, exps = run(False, path)
    assert len(exps) == N_STEPS
    teacher = wf.teacher_model
    weights = [e.metrics["opd_gate_weight"] for e in exps]
    assert weights == EXPECTED, weights
    for e, wt in zip(exps, weights):  # student -1, teacher -3 -> blended -1 - 2 w
        assert torch.allclose(e.teacher_logprobs, torch.full((3,), -1.0 - 2.0 * wt))
    # agreement for every step; both look-aheads only where the teacher disagreed
    assert len(teacher.agree_prompts) == N_STEPS
    assert len(teacher.progress_prompts) == 6 and len(teacher.rollout_prompts) == 6 * WINDOW
    asked = sorted(int(re.search(r"\(steps (\d+)-", p[-1]["content"]).group(1)) - 1 for p in teacher.student_progress_prompts)
    assert asked == [1, 2, 3, 5], asked  # disagreed steps with a window; 8 and 9 use the outcome
    for p in teacher.student_progress_prompts:
        t = int(re.search(r"\(steps (\d+)-", p[-1]["content"]).group(1)) - 1
        assert len(p) == 2 * (t + WINDOW) + 1 and f"(steps {t + 1}-{t + WINDOW})" in p[-1]["content"]
    assert exps[1].metrics["opd_gate_not_progress"] == 0.0 and exps[8].metrics["opd_gate_not_progress"] == 1.0
    assert exps[5].metrics["opd_gate_teacher_no_progress"] == 1.0 and "opd_gate_not_progress" not in exps[0].metrics
    last = exps[-1].metrics
    assert abs(last["opd_gate_full_rate"] - 0.2) < 1e-9 and abs(last["opd_gate_half_rate"] - 0.3) < 1e-9
    assert abs(last["opd_gate_none_rate"] - 0.5) < 1e-9 and abs(last["opd_gate_disagree_rate"] - 0.6) < 1e-9
    assert last["teacher_lookahead_count"] == 6 and last["teacher_lookahead_replay_ok_rate"] == 1.0
    print("explorer-side gate OK: weights", weights)

    # defer_teacher: the explorer asks nothing; the trainer-side gate gives the same result
    wf_d, exps_d = run(True, path)
    t_d = wf_d.teacher_model
    assert not (t_d.agree_prompts or t_d.rollout_prompts or t_d.progress_prompts or t_d.student_progress_prompts)
    p5, p8 = exps_d[5].info["opd_deferred"], exps_d[8].info["opd_deferred"]
    assert p5["gate"]["gate_mode"] == M.GATE_MODE and p5["game_file"] == path and len(p5["prior_actions"]) == 5
    assert p5["has_window"] and len(P.payload_memory(p5)) == 2 * (5 + WINDOW)   # through the student's window
    assert not p8["has_window"] and len(P.payload_memory(p8)) == 2 * 8 + 1      # tail: up to its user message

    def canon(prompts):
        return sorted(json.dumps(p) for p in prompts)

    order = [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]
    gate = T.gate_for([T.TeacherActor(t_d)], trainer_lookahead_in_process=True)
    metrics = asyncio.run(gate.annotate([exps_d[i] for i in order]))
    assert [e.metrics["opd_gate_weight"] for e in exps_d] == EXPECTED
    for e, ref in zip(exps_d, exps):
        assert torch.equal(e.teacher_logprobs, ref.teacher_logprobs) and "opd_deferred" not in e.info
        if "opd_gate_not_progress" in ref.metrics:  # (the trainer side also records the outcome for agreed tail steps)
            assert e.metrics["opd_gate_not_progress"] == ref.metrics["opd_gate_not_progress"]
        assert e.metrics.get("opd_gate_teacher_no_progress") == ref.metrics.get("opd_gate_teacher_no_progress")
    for kind in ("agree_prompts", "rollout_prompts", "progress_prompts", "student_progress_prompts"):
        assert canon(getattr(t_d, kind)) == canon(getattr(teacher, kind)), kind
    assert metrics["teacher_gate/teacher_lookahead_count"] == 6 and metrics["teacher_gate/progress_prompts"] == 4
    assert abs(metrics["teacher_gate/opd_gate_full_rate"] - 0.2) < 1e-9
    assert abs(metrics["teacher_gate/opd_gate_half_rate"] - 0.3) < 1e-9
    assert abs(metrics["teacher_gate/opd_gate_apply_rate"] - sum(EXPECTED) / N_STEPS) < 1e-9
    print("deferred (trainer-side) gate == explorer-side gate")


if __name__ == "__main__":
    check_weight_table()
    check_real_game(T.game_file())
    print("ALL OK")
