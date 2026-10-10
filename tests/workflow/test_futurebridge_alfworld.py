"""FutureBridgeAlfworldWorkflow (copied from the FutureBridge-OPD release) in the TPU port:
runs with a scripted student and teacher on a real ALFWorld game. No TPU, no model.

Checks that an episode runs end to end from a curriculum prefix (k > 0) and from the start
(k = 0), that a bridge is attempted on a failed episode, that the validation replay starts
from the Student's actual state (the `_k_star_v4` fix), and that kept bridge turns have the
fields the tunix trainer reads.
Run: python tests/workflow/test_futurebridge_alfworld.py
"""
import asyncio
import json
import os
import re
import types
from dataclasses import dataclass
from typing import Optional

import torch

import trinity.common.workflows.envs.TCOD.alfworld.futurebridge_workflow as M
from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS

os.environ.setdefault("TMPDIR", "/dev/shm/tmp" if os.path.isdir("/dev/shm/tmp") else "/tmp")


@dataclass
class RolloutArgs:
    temperature: float = 1.0
    logprobs: Optional[int] = 0
    n: int = 1


def admissible(user_content):
    block = re.split(r"admissible actions of the current situation are:\s*\[", user_content, 1)[1].split("].\n", 1)[0]
    return re.findall(r"'([^']*)'", block)


def as_exp(messages, text, tag):
    n_prompt = 4 + len(messages)                       # stand-in for the tokenized prompt
    resp = torch.arange(5) + 100 * tag
    return Experience(tokens=torch.cat([torch.zeros(n_prompt, dtype=torch.long), resp]), prompt_length=n_prompt,
                      logprobs=torch.full((5,), -1.0), response_text=text)


class Student:
    """Wanders: picks 'look'-like admissible actions, never the expert's, so the episode fails."""

    def __init__(self):
        self.chats, self.scored = [], 0

    async def chat_async(self, messages, **kw):
        cmds = admissible(messages[-1]["content"])
        self.chats.append(list(messages))
        return [as_exp(messages, f"hmm <action>{cmds[(3 * len(self.chats)) % len(cmds)]}</action>", len(self.chats))]

    async def logprobs_async(self, tokens, temperature=None):
        self.scored += 1
        return torch.full((len(tokens) - 1,), -1.0)


class Teacher:
    """Scores the student's turns below the student (disagreement), with turn 2 the worst;
    prefers the continuation after its own bridge (higher logprob there)."""

    def __init__(self, prefer_bridge=True):
        self.chats, self.scored, self.prefer_bridge = [], [], prefer_bridge

    async def chat_async(self, messages, **kw):
        cmds = admissible(messages[-1]["content"])
        self.chats.append((list(messages), kw))
        return [as_exp(messages, f"I would <action>{cmds[0]}</action>", 9)]

    async def logprobs_async(self, tokens, temperature=None):
        tag = int(tokens[-1]) // 100
        self.scored.append(tag)
        n = len(tokens) - 1
        out = torch.full((n,), -2.0)                    # default: teacher below student (-1)
        if tag == 2:
            out[-5:] = -6.0                             # the most disagreed student turn
        elif tag >= 9 and self.prefer_bridge:
            out[-5:] = -0.5                             # continuation after the bridge: teacher prefers it
        return out


def run(step, prefer_bridge=True):
    with open(os.path.expanduser("~/alf-data/tcod_tasks/train_expert.jsonl")) as f:
        raw = next(d for d in map(json.loads, f) if len(d.get("actions") or []) >= 6)
    cls = M.FutureBridgeAlfworldWorkflow
    assert WORKFLOWS.get("FutureBridgeAlfworldWorkflow") is cls
    wf = cls.__new__(cls)
    wf.task = types.SimpleNamespace(workflow_args={"continuation_steps": 3}, rollout_args=RolloutArgs(), batch_id=step,
                                    raw_task=raw, task_desc=raw["game_file"], is_eval=False, format_args=None)
    wf.raw_task, wf.task_desc, wf.is_eval = raw, raw["game_file"], False
    wf.model, wf.teacher_model = Student(), Teacher(prefer_bridge)
    wf.temperature, wf.max_env_steps = 1.0, 8
    wf.checkpoint_strategy, wf.checkpoint_steps, wf.total_steps = "linear", 5, 250
    # 0.01 -> only the single most-disagreed turn is tried (the paper's bridge_position_top_k = 1);
    # with the code default 0.3 the next turns are tried too whenever a candidate is rejected
    wf.bridge_kl_top_ratio, wf.bridge_kl_lambda, wf.bridge_kl_max_per_ep = 0.01, 0.5, 1
    wf._flex_cache, wf._current_anchor_reliable, wf._current_anchor_type = {}, True, "b2f"
    wf.run_id_base = 0

    # record the state the validation replay reaches right before the bridge action
    seen = {}
    real = M._create_alfworld_env_with_checkpoint

    def spy(path, actions, k):
        seen.setdefault("k", []).append(k)
        return real(path, actions, k)

    M._create_alfworld_env_with_checkpoint = spy
    try:
        exps = asyncio.run(wf.run_async())
    finally:
        M._create_alfworld_env_with_checkpoint = real
    return wf, exps, raw, seen


def main():
    for step, label in ((0, "curriculum prefix"), (10_000, "no prefix")):
        wf, exps, raw, seen = run(step)
        k = max(0, len(raw["actions"]) - 1 - step // 5)
        normal = [e for e in exps if e.eid.step < 5000]
        bridge = [e for e in exps if e.eid.step >= 5000]
        assert wf._k_star_v4 == k, (wf._k_star_v4, k)
        assert [e.eid.step for e in normal] == list(range(k, k + len(normal))), "student turns are numbered from the prefix"
        assert all(e.teacher_logprobs is not None and len(e.teacher_logprobs) == len(e.logprobs) for e in exps)
        assert not wf._env_done, "the scripted student should fail, so a bridge is attempted"
        # the replay for the validation starts from the same prefix as the episode (both k), never from 0 when k > 0
        assert set(seen["k"]) == {k} or k == 0, seen
        assert len(wf.teacher_model.chats) == 1, "one bridge candidate per episode"
        ctx, kw = wf.teacher_model.chats[0]
        assert kw["temperature"] == 0.0 and ctx[-1]["role"] == "user"
        # continuation: up to 3 student turns after the bridge (fewer if the game ends), each
        # scored by student and teacher, starting from the bridged conversation
        cont = [c for c in wf.model.chats if any("I would" in m["content"] for m in c if m["role"] == "assistant")]
        assert 1 <= len(cont) <= 3, len(cont)
        assert cont[0][: len(ctx)] == ctx and "I would" in cont[0][len(ctx)]["content"], "continues from the bridged conversation"
        for e in bridge:  # what the tunix trainer reads
            assert e.metrics["is_bridge"] == 1 and e.prompt_length > 0 and len(e.tokens) - e.prompt_length == len(e.teacher_logprobs)
        assert (len(bridge) == 1) == (k == 0), "kept without a prefix; with the prefix the chosen turn is the last one (nothing to compare)"
        print(f"{label}: k={k}, student turns {len(normal)}, bridge kept {len(bridge)}, "
              f"bridge_token_ratio {normal[-1].metrics['bridge_token_ratio']:.3f}")
    # future validation: the bridge is kept only if the teacher prefers the bridged continuation
    _, kept, _, _ = run(10_000, prefer_bridge=True)
    _, dropped, _, _ = run(10_000, prefer_bridge=False)
    assert sum(e.eid.step >= 5000 for e in kept) == 1 and sum(e.eid.step >= 5000 for e in dropped) == 0
    b = next(e for e in kept if e.eid.step >= 5000)
    assert "I would" in b.response_text and b.reward == 1.0
    print("future validation: kept when the bridged continuation is preferred, dropped otherwise")
    print("ALL OK")


if __name__ == "__main__":
    main()
