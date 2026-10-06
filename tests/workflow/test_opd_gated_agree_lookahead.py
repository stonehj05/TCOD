"""Gate logic of OPD_gated_alfworld_workflow_agree_lookahead with a scripted teacher/student/env.

weight = 0.5 * [teacher disagrees with the action] + 0.5 * [not making progress], where the
last `window_size` steps use the episode outcome for the progress criterion.
Run: python tests/workflow/test_opd_gated_agree_lookahead.py
"""
import asyncio
import types

import torch

import trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_agree_lookahead as M
from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS

N_STEPS, WINDOW = 8, 3  # steps 0..4 have a forward window, steps 5..7 use the outcome


class Env:
    def __init__(self, success):
        self.t, self.success = 0, success

    def reset(self):
        return "You are in a room.\n\nYour task is to: put a mug in the sink.", {"admissible_commands": ["look", "go north"]}

    def step(self, action):
        self.t += 1
        done = self.success and self.t == N_STEPS
        return f"obs {self.t}", float(done), done, {"admissible_commands": ["look", "go north"]}

    def close(self):
        pass


class Student:
    def __init__(self):
        self.t = 0

    async def chat_async(self, messages, **kw):
        self.t += 1
        resp = torch.arange(3) + 10 * self.t
        return [Experience(tokens=torch.cat([torch.zeros(4, dtype=torch.long), resp]), prompt_length=4,
                           logprobs=torch.full((3,), -1.0), response_text=f"<action>act{self.t}</action>")]


class Teacher:
    """agree[t] / progress[t]: True = Yes, False = No, None = unparseable."""

    def __init__(self, agree, progress):
        self.agree, self.progress, self.calls, self.max_inflight, self.inflight, self.scored = agree, progress, [], 0, 0, []

    async def chat_async(self, messages, **kw):
        self.inflight += 1; self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        last = messages[-1]["content"]
        if "Would you choose this exact action" in last:
            t = (len(messages) - 1) // 2          # context = [u0,a0,...,u_t + addendum]
            assert f"act{t + 1}" in last, "agreement prompt must quote that step's action"
            assert all(m["role"] != "assistant" or f"act{t + 1}" not in m["content"] for m in messages), \
                "the student's own response for the step must not be in the agreement context"
            ans = self.agree[t]; self.calls.append(("agree", t))
        else:
            start = int(last.split("(steps")[1].split("-")[0])   # 1-indexed window start
            t = start - 1
            assert len(messages) == 2 * (t + WINDOW) + 1, "progress prompt must include the forward window"
            ans = self.progress[t]; self.calls.append(("progress", t))
        text = {True: "ok <answer>Yes</answer>", False: "hmm <answer>No</answer>", None: "no idea"}[ans]
        return [types.SimpleNamespace(response_text=text)]

    async def logprobs_async(self, tokens, temperature=None):
        self.inflight += 1; self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        # teacher logprob -3 on every position; first response token identifies the step
        self.scored.append(int(tokens[4]) // 10)
        return torch.full((len(tokens) - 1,), -3.0)


def run(success, agree, progress, parallel=16, mode="sum", defer=False):
    task = types.SimpleNamespace(workflow_args={"max_env_steps": N_STEPS, "progress_window_size": WINDOW},
                                 rollout_args=types.SimpleNamespace(), format_args=None, raw_task={}, task_desc="x", is_eval=False)
    wf = M.OPDGatedAlfworldWorkflowAgreeLookahead.__new__(M.OPDGatedAlfworldWorkflowAgreeLookahead)
    wf.task, wf.model, wf.teacher_model = task, Student(), Teacher(agree, progress)
    wf.temperature, wf.max_env_steps, wf.window_size = 1.0, N_STEPS, WINDOW
    wf.progress_temperature = wf.consistency_temperature = 0.0
    wf.progress_max_tokens = wf.consistency_max_tokens = 512
    wf.single_criterion_weight = 0.5
    wf.teacher_parallel_prompts = parallel
    wf.gate_mode, wf.disagree_no_progress_weight, wf.disagree_progress_weight = mode, 1.0, 0.5
    wf.defer_teacher, wf._final_reward = defer, 0.0
    type(wf).rollout_args = property(lambda self: {})
    exps = asyncio.run(wf._run_episode(Env(success)))
    return wf, exps


def main():
    assert WORKFLOWS.get("OPD_gated_alfworld_workflow_agree_lookahead") is M.OPDGatedAlfworldWorkflowAgreeLookahead
    #            t:  0      1      2      3      4     | 5      6      7   (no window)
    agree    = [True,  False, True,  False, None,   True,  False, True]
    progress = [True,  True,  False, False, True,   None,  None,  None]   # 5..7 never asked
    # failed episode: steps 5..7 count as "not making progress"
    wf, exps = run(False, agree, progress)
    w = [e.metrics["opd_gate_weight"] for e in exps]
    assert w == [0.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5], w
    for e, wt in zip(exps, w):   # student -1, teacher -3 -> blended = -1 + w * (-2); advantage scales by w
        assert torch.allclose(e.teacher_logprobs, torch.full((3,), -1.0 - 2.0 * wt)), (wt, e.teacher_logprobs)
        assert bool(e.teacher_logprobs_valid_mask.all())
    t = wf.teacher_model
    assert sorted(c for c in t.calls if c[0] == "agree") == [("agree", i) for i in range(8)]
    assert sorted(c for c in t.calls if c[0] == "progress") == [("progress", i) for i in range(5)]
    assert sorted(t.scored) == list(range(1, 9)), "every turn is scored exactly once"
    assert [int(e.tokens[4]) // 10 for e in exps] == list(range(1, 9)) and [e.eid.step for e in exps] == list(range(8))
    # 8 agreement + 5 progress prompts are all in flight together (cap 16 not reached)
    assert t.max_inflight == 13, f"all teacher prompts of the episode should be batched, saw {t.max_inflight}"
    for cap in (1, 4):   # the cap is respected, and batching does not change any weight
        wf_c, exps_c = run(False, agree, progress, parallel=cap)
        assert wf_c.teacher_model.max_inflight == cap, (cap, wf_c.teacher_model.max_inflight)
        assert [e.metrics["opd_gate_weight"] for e in exps_c] == w
    last = exps[-1].metrics
    assert abs(last["opd_gate_apply_rate"] - sum(w) / 8) < 1e-9
    assert (last["opd_gate_full_rate"], last["opd_gate_half_rate"], last["opd_gate_none_rate"]) == (2 / 8, 5 / 8, 1 / 8)
    assert last["n_windowed_gate_decisions"] == 5 and last["n_outcome_gate_decisions"] == 3
    print("failed episode  weights", w)
    # successful episode: steps 5..7 count as "making progress" -> only the agreement criterion can fire
    wf, exps = run(True, agree, progress)
    w = [e.metrics["opd_gate_weight"] for e in exps]
    assert w == [0.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.5, 0.0], w
    print("success episode weights", w)
    # unparseable progress answer counts as "not making progress" (fail-safe)
    wf, exps = run(True, [True] * 8, [None] * 8)
    assert [e.metrics["opd_gate_weight"] for e in exps] == [0.5] * 5 + [0.0] * 3
    check_disagree_required(agree, progress)
    check_deferred(agree, progress)
    print("ALL OK")


class TeacherActor:
    """The Teacher above behind a Ray-actor-like interface (`.chat.remote(...)`), recording prompts."""

    def __init__(self, teacher):
        self.t, self.prompts = teacher, []
        self.chat = types.SimpleNamespace(remote=self._chat)
        self.logprobs = types.SimpleNamespace(remote=self._logprobs)

    async def _chat(self, messages, lora_request=None, **kw):
        self.prompts.append((messages, kw))
        return await self.t.chat_async(messages, **kw)

    async def _logprobs(self, tokens, temperature=None):
        return await self.t.logprobs_async(tokens, temperature=temperature)


def check_deferred(agree, progress):
    """defer_teacher: the explorer makes no teacher call; the trainer-side gate then gives every
    turn the same prompts, weight and teacher_logprobs as the explorer-side gate, also when it
    sees an arbitrary subset of an episode's turns in arbitrary order."""
    import pickle
    import random

    from trinity.trainer.tunix.teacher_gate import DeferredTeacherGate

    def gate(teachers, **args):
        cfg = types.SimpleNamespace(
            buffer=types.SimpleNamespace(explorer_input=types.SimpleNamespace(taskset=None, tasksets=[types.SimpleNamespace(
                workflow_args={"defer_teacher": True, **args})])),  # as after config validation
            explorer=types.SimpleNamespace(name="explorer", auxiliary_models=[types.SimpleNamespace(engine_num=len(teachers))]),
            ray_namespace="x")
        return DeferredTeacherGate(cfg, teachers=teachers)

    for mode in ("sum", "disagree_required"):
        for success in (False, True):
            # reference: explorer-side gate, recording the prompts the teacher saw
            ref_wf, ref = run(success, agree, progress, mode=mode)
            rec = TeacherActor(Teacher(agree, progress))
            wf_d, exps = run(success, agree, progress, mode=mode, defer=True)
            assert not wf_d.teacher_model.calls and not wf_d.teacher_model.scored, "explorer must not call the teacher"
            assert all(e.teacher_logprobs is None and "opd_deferred" in e.info for e in exps)
            assert [e.eid.step for e in exps] == list(range(8)) and exps[-1].metrics["env_rounds"] == 8
            exps = pickle.loads(pickle.dumps(exps))          # as stored in / read from the buffer
            order = list(range(8)); random.Random(mode + str(success)).shuffle(order)
            g = gate([rec, rec])
            assert g.enabled
            m = asyncio.run(g.annotate([exps[i] for i in order]))
            for e_ref, e in zip(ref, exps):
                assert e.metrics["opd_gate_weight"] == e_ref.metrics["opd_gate_weight"], (mode, success, e.eid.step)
                assert torch.allclose(e.teacher_logprobs, e_ref.teacher_logprobs)
                assert e.metrics.get("opd_gate_not_progress") == e_ref.metrics.get("opd_gate_not_progress")
                assert "opd_deferred" not in e.info and bool(e.teacher_logprobs_valid_mask.all())
            # same teacher questions as the explorer-side gate (set of (kind, step) calls)
            assert sorted(rec.t.calls) == sorted(ref_wf.teacher_model.calls), (mode, success)
            assert sorted(rec.t.scored) == list(range(1, 9))
            w = [e.metrics["opd_gate_weight"] for e in exps]
            assert abs(m["teacher_gate/opd_gate_apply_rate"] - sum(w) / 8) < 1e-9
            assert m["teacher_gate/progress_prompts"] == sum(c[0] == "progress" for c in rec.t.calls)
            assert abs(m["teacher_gate/opd_gate_none_rate"] - ref[-1].metrics["opd_gate_none_rate"]) < 1e-9
            assert abs(m["teacher_gate/opd_gate_full_rate"] - ref[-1].metrics["opd_gate_full_rate"]) < 1e-9
    # prompts are identical message for message to what the workflow's own methods send
    wf, _ = run(False, agree, progress, mode="sum")
    sent = []
    async def record(messages, **kw):
        sent.append((messages, kw)); return [types.SimpleNamespace(response_text="<answer>No</answer>")]
    wf.teacher_model = types.SimpleNamespace(chat_async=record)
    wf_d, exps = run(False, agree, progress, mode="sum", defer=True)
    mem = M.payload_memory(exps[0].info["opd_deferred"])   # turn 0 keeps the conversation through its window
    assert len(mem) == 2 * WINDOW
    full = M.payload_memory(exps[2].info["opd_deferred"])  # turn 2: through its window (steps 2..4)
    assert len(full) == 2 * (2 + WINDOW)
    assert len(M.payload_memory(exps[7].info["opd_deferred"])) == 15  # tail turn: up to its user message
    asyncio.run(wf._ask_teacher_agree(full, 2, "act3")); asyncio.run(wf._ask_teacher_progress(full, 2, 4))
    rec = TeacherActor(Teacher(agree, progress))
    asyncio.run(gate([rec]).annotate([exps[2]]))
    assert [p[0] for p in rec.prompts] == [s_[0] for s_ in sent], "trainer-side prompts differ from the workflow's"
    assert [p[1] for p in rec.prompts] == [{"temperature": 0.0, "max_tokens": 512, "n": 1}] * 2
    # skip_zero_weight_scoring: weight-0 turns are not scored and get the student's own logprobs
    wf_d, exps = run(False, agree, progress, mode="disagree_required", defer=True)
    rec = TeacherActor(Teacher(agree, progress))
    m = asyncio.run(gate([rec], skip_zero_weight_scoring=True).annotate(exps))
    w = [e.metrics["opd_gate_weight"] for e in exps]
    assert w == [0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0]
    assert sorted(rec.t.scored) == [2, 4, 5, 7] and m["teacher_gate/scored_turns"] == 4
    for e, wt in zip(exps, w):
        assert torch.allclose(e.teacher_logprobs, torch.full((3,), -1.0 - 2.0 * wt))
    # per-engine request cap
    rec = TeacherActor(Teacher(agree, progress))
    wf_d, exps = run(False, agree, progress, mode="sum", defer=True)
    asyncio.run(gate([rec], trainer_teacher_parallel_prompts=3).annotate(exps))
    assert rec.t.max_inflight == 3, rec.t.max_inflight
    # a batch without deferred turns is left alone
    assert asyncio.run(gate([rec]).annotate(ref)) == {}
    check_vanilla_deferred(gate)
    print("deferred (trainer-side) gate == explorer-side gate")


def check_disagree_required(agree, progress):
    """gate_mode=disagree_required: disagree & not-progress -> 1.0, disagree & progress -> 0.5,
    teacher agrees -> 0 whatever the progress answer; progress is asked only where it matters."""
    #            t:  0      1      2      3      4     | 5      6      7   (no window)
    # agree      = [True,  False, True,  False, None,   True,  False, True]   (None = unparseable = disagree)
    # progress   = [True,  True,  False, False, True,   -      -      -   ]
    wf, exps = run(False, agree, progress, mode="disagree_required")       # failed episode
    w = [e.metrics["opd_gate_weight"] for e in exps]
    #  t0 agrees -> 0 | t1 disagree+progress -> .5 | t2 agrees (not-progress ignored) -> 0 | t3 disagree+not -> 1
    #  t4 unparseable=disagree + progress -> .5 | t5 agrees -> 0 | t6 disagree, failed outcome -> 1 | t7 agrees -> 0
    assert w == [0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0], w
    for e, wt in zip(exps, w):
        assert torch.allclose(e.teacher_logprobs, torch.full((3,), -1.0 - 2.0 * wt))
    t = wf.teacher_model
    assert sorted(c for c in t.calls if c[0] == "agree") == [("agree", i) for i in range(8)]
    # progress asked only for windowed steps the teacher disagreed with: 1, 3, 4
    assert sorted(c for c in t.calls if c[0] == "progress") == [("progress", 1), ("progress", 3), ("progress", 4)], t.calls
    m = exps[-1].metrics
    assert (m["opd_gate_full_rate"], m["opd_gate_half_rate"], m["opd_gate_none_rate"]) == (2 / 8, 2 / 8, 4 / 8)
    assert m["opd_gate_disagree_rate"] == 4 / 8 and m["n_windowed_gate_decisions"] == 3 and m["n_outcome_gate_decisions"] == 3
    assert "opd_gate_not_progress" not in exps[0].metrics and exps[3].metrics["opd_gate_not_progress"] == 1.0
    print("disagree_required, failed  ", w)
    wf, exps = run(True, agree, progress, mode="disagree_required")        # successful episode
    w = [e.metrics["opd_gate_weight"] for e in exps]
    # tail steps count as "making progress": t6 disagree -> 0.5 instead of 1.0
    assert w == [0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.5, 0.0], w
    print("disagree_required, success ", w)
    # teacher agrees everywhere -> no OPD at all, and no progress question is asked
    wf, exps = run(False, [True] * 8, [False] * 8, mode="disagree_required")
    assert [e.metrics["opd_gate_weight"] for e in exps] == [0.0] * 8
    assert not [c for c in wf.teacher_model.calls if c[0] == "progress"]
    # unparseable progress answer for a disagreed step counts as not-progress -> full weight
    wf, exps = run(True, [False] * 8, [None] * 8, mode="disagree_required")
    assert [e.metrics["opd_gate_weight"] for e in exps] == [1.0] * 5 + [0.5] * 3
    for cap in (1, 3):
        wf_c, exps_c = run(False, agree, progress, parallel=cap, mode="disagree_required")
        assert wf_c.teacher_model.max_inflight == cap
        assert [e.metrics["opd_gate_weight"] for e in exps_c] == [0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0]



def check_vanilla_deferred(gate):
    """Vanilla OPD with defer_teacher: trainer-side scoring stores exactly what the workflow stores."""
    import trinity.common.workflows.envs.TCOD.alfworld.OPD_workflow_fullmemory as V

    def run_v(defer):
        wf = V.OnPolicyDistillVerlAgentAlfworldWorkflowFullMemory.__new__(V.OnPolicyDistillVerlAgentAlfworldWorkflowFullMemory)
        wf.task = types.SimpleNamespace(rollout_args=types.SimpleNamespace())
        wf.model, wf.teacher_model = Student(), Teacher([], [])
        wf.temperature, wf.max_env_steps, wf.defer_teacher = 1.0, N_STEPS, defer
        type(wf).rollout_args = property(lambda self: {})
        return wf, asyncio.run(wf._run_episode(Env(False)))

    _, ref = run_v(False)
    wf, exps = run_v(True)
    assert not wf.teacher_model.scored and all(e.teacher_logprobs is None for e in exps)
    rec = TeacherActor(Teacher([], []))
    m = asyncio.run(gate([rec]).annotate(exps[::-1]))
    assert not rec.prompts and sorted(rec.t.scored) == list(range(1, 9)) and m["teacher_gate/scored_turns"] == 8
    assert "teacher_gate/agree_prompts" not in m
    for a, b in zip(ref, exps):
        assert torch.equal(a.teacher_logprobs, b.teacher_logprobs) and a.eid.step == b.eid.step and a.reward == b.reward
    print("vanilla OPD: trainer-side scoring == explorer-side scoring")


if __name__ == "__main__":
    main()
