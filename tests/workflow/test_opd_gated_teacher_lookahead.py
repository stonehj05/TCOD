"""OPD_gated_alfworld_workflow_teacher_lookahead with a scripted student and teacher.

Uses a real ALFWorld game for the replay (needs the task files of scripts/tpu/setup_node.sh,
or ALFWORLD_TEST_GAME=<game file>) and scripted games for the edge cases. No TPU, no model.
Run: python tests/workflow/test_opd_gated_teacher_lookahead.py [--ray]
  --ray  also runs the look-ahead as Ray tasks against a local Ray instance
"""
import asyncio
import json
import os
import re
import sys
import types

RUN_RAY = "--ray" in sys.argv  # read before the imports below, which may rewrite sys.argv

import torch

import trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_agree_lookahead as P
import trinity.common.workflows.envs.TCOD.alfworld.OPD_gated_workflow_teacher_lookahead as M
from trinity.common.experience import Experience
from trinity.common.workflows.envs.TCOD.alfworld.utils import _create_alfworld_env
from trinity.trainer.tunix.teacher_gate import DeferredTeacherGate

os.environ.setdefault("TMPDIR", "/dev/shm/tmp" if os.path.isdir("/dev/shm/tmp") else "/tmp")
N_STEPS, WINDOW = 10, 3
#         step:  0     1      2     3      4     5      6     7     8      9
AGREE = [True, False, None, False, True, False, True, True, False, False]  # None = unparseable
PROGRESS = {1: True, 2: False, 3: None, 5: False, 8: True, 9: False}       # teacher on its own steps
EXPECTED = [0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5]


def game_file():
    if os.environ.get("ALFWORLD_TEST_GAME"):
        return os.environ["ALFWORLD_TEST_GAME"]
    with open(os.path.expanduser("~/alf-data/tcod_tasks/train.jsonl")) as f:
        return json.loads(f.readline())["game_file"]


def admissible(user_content):
    block = user_content.split("admissible actions of the current situation are:\n[", 1)[1].split("].\n", 1)[0]
    return re.findall(r"'([^']*)'", block)


class Student:
    """Picks an admissible action by step; step 4 gives no action tag (an invalid action)."""

    def __init__(self):
        self.t = 0

    async def chat_async(self, messages, **kw):
        cmds, t = admissible(messages[-1]["content"]), self.t
        self.t += 1
        text = "no action here" if t == 4 else f"thinking <action>{cmds[(7 * t + 3) % len(cmds)]}</action>"
        resp = torch.arange(3) + 10 * (t + 1)
        return [Experience(tokens=torch.cat([torch.zeros(4, dtype=torch.long), resp]), prompt_length=4,
                           logprobs=torch.full((3,), -1.0), response_text=text)]


def answer(value):
    return {True: "ok <answer>Yes</answer>", False: "hm <answer>No</answer>", None: "who knows"}[value]


class Teacher:
    def __init__(self):
        self.agree_prompts, self.rollout_prompts, self.progress_prompts = [], [], []

    def respond(self, messages):
        messages = list(messages)  # the caller keeps extending its list
        last = messages[-1]["content"]
        if "Would you choose this exact action" in last:
            self.agree_prompts.append(messages)
            return answer(AGREE[(len(messages) - 1) // 2])
        if "making progress toward" in last:
            self.progress_prompts.append(messages)
            start = int(re.search(r"\(steps (\d+)-(\d+)\)", last).group(1)) - 1
            return answer(PROGRESS[start])
        self.rollout_prompts.append(messages)
        cmds = admissible(last)
        return f"my turn <action>{cmds[(5 * len(messages) + 1) % len(cmds)]}</action>"

    async def chat_async(self, messages, **kw):
        await asyncio.sleep(0)
        return [types.SimpleNamespace(response_text=self.respond(messages))]

    async def logprobs_async(self, tokens, temperature=None):
        return torch.full((len(tokens) - 1,), -3.0)


class TeacherActor:
    """The Teacher behind a Ray-actor-like interface (`.chat.remote(...)`)."""

    def __init__(self, teacher):
        self.t = teacher
        self.chat = types.SimpleNamespace(remote=lambda messages, lora_request=None, **kw: teacher.chat_async(messages, **kw))
        self.logprobs = types.SimpleNamespace(remote=lambda tokens, temperature=None: teacher.logprobs_async(tokens, temperature=temperature))


def make(cls, defer, path):
    task = types.SimpleNamespace(workflow_args={}, rollout_args=types.SimpleNamespace(), format_args=None,
                                 raw_task={}, task_desc=path, is_eval=False)
    wf = cls.__new__(cls)
    wf.task, wf.task_desc, wf.model, wf.teacher_model = task, path, Student(), Teacher()
    wf.temperature, wf.max_env_steps, wf.window_size = 1.0, N_STEPS, WINDOW
    wf.progress_temperature = wf.consistency_temperature = 0.0
    wf.progress_max_tokens = wf.consistency_max_tokens = 512
    wf.single_criterion_weight, wf.teacher_parallel_prompts = 0.5, 16
    wf.gate_mode, wf.disagree_no_progress_weight, wf.disagree_progress_weight = "disagree_required", 1.0, 0.5
    wf.teacher_progress_weight, wf.teacher_no_progress_weight = 1.0, 0.5
    wf.teacher_rollout_temperature, wf.teacher_rollout_max_tokens = 0.0, 512
    wf.defer_teacher, wf._final_reward = defer, 0.0
    wf.logger = types.SimpleNamespace(warning=print)
    type(wf).rollout_args = property(lambda self: {})
    return wf


def run(cls, defer, path):
    wf = make(cls, defer, path)
    env = _create_alfworld_env(path)
    try:
        exps = asyncio.run(wf._run_episode(env))
    finally:
        env.close()
    return wf, exps


def gate_for(teachers, **args):
    cfg = types.SimpleNamespace(
        buffer=types.SimpleNamespace(explorer_input=types.SimpleNamespace(taskset=None, tasksets=[types.SimpleNamespace(
            workflow_args={"defer_teacher": True, **args})])),
        explorer=types.SimpleNamespace(name="explorer", auxiliary_models=[types.SimpleNamespace(engine_num=len(teachers))], env_vars={}),
        ray_namespace="x")
    return DeferredTeacherGate(cfg, teachers=teachers)


def check_real_game(path):
    # the student's rollout is the same conversation as in the existing gated workflow
    _, parent = run(P.OPDGatedAlfworldWorkflowAgreeLookahead, True, path)
    wf, exps = run(M.OPDGatedAlfworldWorkflowTeacherLookahead, False, path)
    assert len(exps) == N_STEPS, "the scripted student should not finish this game"
    memory = P.payload_memory(parent[N_STEPS - 1].info["opd_deferred"])  # through step 9's user message
    assert len(memory) == 2 * N_STEPS - 1
    teacher = wf.teacher_model

    # explorer-side gate: weights, blended teacher logprobs, metrics
    weights = [e.metrics["opd_gate_weight"] for e in exps]
    assert weights == EXPECTED, weights
    for e, w in zip(exps, weights):
        assert torch.allclose(e.teacher_logprobs, torch.full((3,), -1.0 - 2.0 * w))
    last = exps[-1].metrics
    assert last["teacher_lookahead_count"] == 6 and last["teacher_lookahead_errors"] == 0
    assert last["teacher_lookahead_replay_ok_rate"] == 1.0, "replayed observation differs from the student's"
    assert last["teacher_lookahead_steps_mean"] == WINDOW and last["teacher_lookahead_finished_rate"] == 0.0
    assert abs(last["teacher_lookahead_no_progress_rate"] - 3 / 6) < 1e-9
    assert abs(last["teacher_lookahead_progress_parse_success_rate"] - 5 / 6) < 1e-9
    assert abs(last["opd_gate_disagree_rate"] - 0.6) < 1e-9 and abs(last["opd_gate_half_rate"] - 0.3) < 1e-9
    assert exps[2].metrics["opd_gate_teacher_no_progress"] == 1.0 and "opd_gate_teacher_no_progress" not in exps[0].metrics

    # what the teacher saw: agreement for every step; look-ahead only where it disagreed
    assert len(teacher.agree_prompts) == N_STEPS and len(teacher.progress_prompts) == 6
    assert len(teacher.rollout_prompts) == 6 * WINDOW
    for prompt in teacher.progress_prompts:
        t = int(re.search(r"\(steps (\d+)-(\d+)\)", prompt[-1]["content"]).group(1)) - 1
        assert f"(steps {t + 1}-{t + WINDOW})" in prompt[-1]["content"] and len(prompt) == 2 * (t + WINDOW) + 1
        assert prompt[: 2 * t + 1] == memory[: 2 * t + 1], "the teacher starts from the student's conversation"
        teacher_turns = prompt[2 * t + 1 : -1]
        assert all("my turn" in m["content"] for m in teacher_turns[0::2]), "its own steps follow, not the student's"
        if t + 1 < N_STEPS:  # the teacher's game diverges from the student's
            assert teacher_turns[1]["content"] != memory[2 * t + 2]["content"]
    firsts = [p for p in teacher.rollout_prompts if len(p) % 2 == 1 and p in [memory[: 2 * t + 1] for t in PROGRESS]]
    assert len(firsts) == 6, "each look-ahead's first prompt is the student's conversation up to that step"
    print("explorer-side gate OK: weights", weights)

    # defer_teacher: no teacher call in the explorer; the trainer-side gate gives the same result
    wf_d, exps_d = run(M.OPDGatedAlfworldWorkflowTeacherLookahead, True, path)
    t_d = wf_d.teacher_model
    assert not (t_d.agree_prompts or t_d.rollout_prompts or t_d.progress_prompts)
    assert all(e.teacher_logprobs is None and e.info["opd_deferred"]["game_file"] == path for e in exps_d)
    assert exps_d[5].info["opd_deferred"]["prior_actions"][4] == "" and len(exps_d[5].info["opd_deferred"]["prior_actions"]) == 5
    assert len(P.payload_memory(exps_d[5].info["opd_deferred"])) == 11

    def canon(prompts):
        return sorted(json.dumps(p) for p in prompts)

    order = [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]  # a sampled batch: any order
    gate = gate_for([TeacherActor(t_d)], trainer_lookahead_in_process=True)
    metrics = asyncio.run(gate.annotate([exps_d[i] for i in order]))
    assert [e.metrics["opd_gate_weight"] for e in exps_d] == EXPECTED
    for e, ref in zip(exps_d, exps):
        assert torch.equal(e.teacher_logprobs, ref.teacher_logprobs) and "opd_deferred" not in e.info
    for kind in ("agree_prompts", "rollout_prompts", "progress_prompts"):
        assert canon(getattr(t_d, kind)) == canon(getattr(teacher, kind)), kind
    assert metrics["teacher_gate/teacher_lookahead_count"] == 6 and metrics["teacher_gate/teacher_lookahead_replay_ok_rate"] == 1.0
    assert abs(metrics["teacher_gate/opd_gate_half_rate"] - 0.3) < 1e-9 and "time/teacher_gate_lookahead" in metrics
    assert abs(metrics["teacher_gate/opd_gate_full_rate"] - 0.3) < 1e-9
    print("deferred (trainer-side) gate == explorer-side gate")
    return exps, teacher


class FakeEnv:
    """Finishes when it gets the action 'win'."""

    def step(self, action):
        return "next", 0, action == "win", {"admissible_commands": ["look", "win"]}

    def close(self):
        pass


def check_edge_cases():
    gate = {"window_size": 4, "teacher_rollout_temperature": 0.0, "teacher_rollout_max_tokens": 64,
            "progress_temperature": 0.0, "progress_max_tokens": 64,
            "teacher_progress_weight": 1.0, "teacher_no_progress_weight": 0.5}
    context = [{"role": "user", "content": "u0"}]
    real = M._create_alfworld_env_with_checkpoint
    try:
        # the teacher completes the task in its second step: progress, and no question is asked
        M._create_alfworld_env_with_checkpoint = lambda path, actions, step: (
            FakeEnv(), "obs", {"admissible_commands": ["look", "win"]}, [], "task", 0, False)
        asked = []

        async def ask(messages, temperature, max_tokens):
            asked.append(messages[-1]["content"])
            return "<action>win</action>" if len(messages) == 3 else "<action>look</action>"

        result = asyncio.run(M.teacher_lookahead(ask, "g", context, [], 0, gate, "look"))
        assert result["finished"] and result["progress"] is True and not result["asked"]
        assert result["steps"] == 2 and result["same_first_action"] and len(asked) == 2

        # a game that cannot be replayed raises; callers keep the full weight
        M._create_alfworld_env_with_checkpoint = lambda path, actions, step: (
            FakeEnv(), "obs", {}, [], "task", 1, True)
        try:
            asyncio.run(M.teacher_lookahead(ask, "g", context, ["a", "b"], 2, gate))
            raise AssertionError("expected a replay error")
        except RuntimeError as e:
            assert "replay ended" in str(e)
    finally:
        M._create_alfworld_env_with_checkpoint = real
    assert M.lookahead_weight(gate, False, False) == 0.0 and M.lookahead_weight(gate, True, None) == 1.0
    assert M.lookahead_weight(gate, True, True) == 1.0 and M.lookahead_weight(gate, True, False) == 0.5
    assert M.lookahead_metrics([None, None]) == {"teacher_lookahead_count": 2, "teacher_lookahead_errors": 2}
    print("edge cases OK")


def check_ray_tasks(path, exps_ref):
    """The trainer's default path: every look-ahead is a Ray task that reaches the teacher by
    actor name. Local Ray instance, scripted teacher actor, real game."""
    import ray

    ray.init(address="local", namespace="x", include_dashboard=False, num_cpus=8, log_to_driver=False)
    try:
        @ray.remote
        class RemoteTeacher:
            def __init__(self):
                self.teacher = Teacher()

            async def chat(self, messages, lora_request=None, **kw):
                return await self.teacher.chat_async(messages, **kw)

            async def logprobs(self, tokens, temperature=None):
                return await self.teacher.logprobs_async(tokens, temperature=temperature)

        actor = RemoteTeacher.options(name="explorer_auxiliary_model_0_0").remote()
        _, exps_d = run(M.OPDGatedAlfworldWorkflowTeacherLookahead, True, path)
        gate = gate_for([actor])
        gate._task_env = {"TMPDIR": os.environ["TMPDIR"]}
        metrics = asyncio.run(gate.annotate(exps_d))
        assert [e.metrics["opd_gate_weight"] for e in exps_d] == EXPECTED
        for e, ref in zip(exps_d, exps_ref):
            assert torch.equal(e.teacher_logprobs, ref.teacher_logprobs)
        assert metrics["teacher_gate/teacher_lookahead_errors"] == 0
        print(f"Ray-task look-ahead OK ({metrics['time/teacher_gate_lookahead']:.0f}s for 6 look-aheads)")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    check_edge_cases()
    path = game_file()
    exps, _ = check_real_game(path)
    if RUN_RAY:
        check_ray_tasks(path, exps)
    print("ALL OK")
