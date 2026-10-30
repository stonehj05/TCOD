"""Check WebShop for OPD_gated_webshop_workflow_student_teacher_lookahead (CPU only, ~10 min, ~45 GB RAM).

  1. A WebShopEnvServer (env_server.py) loads the full catalog and has a goal for every session id of
     TCOD's split (0-4195).
  2. WebShop's expert path (DASH-OPD's _get_webshop_gold_action_sequence) buys a product scoring 1 in the
     given sessions: the catalog, search index and goals are consistent.
  3. A second, separately loaded server has the same goals (instruction texts incl. price caps) and
     prices: every server actor builds the same shop.
  4. The workflow's student rollout (mock student playing the expert path, with one invalid action
     inserted) ends with reward 1, and the teacher look-ahead's replay (replay_game) rebuilds the
     student's prompt at every step, on the same server and on the second one.

Usage: python scripts/tpu/checks/check_webshop_server.py [session ...]   (default: 0 1 4096 4195)
Prints RESULT OK / RESULT FAIL.
"""

import asyncio
import resource
import sys
import time
import types

import torch
from transformers import AutoTokenizer

from trinity.common.config import FormatConfig, GenerationConfig
from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.DASH.webshop.utils import _get_webshop_gold_action_sequence
from trinity.common.workflows.envs.TCOD.webshop import OPD_gated_workflow_student_teacher_lookahead as wf
from trinity.common.workflows.envs.TCOD.webshop.env_server import WebShopEnvServer


class LocalEnv:
    """RemoteWebShopEnv on an in-process server."""

    SERVER = None

    def __init__(self, key, *args, **kwargs):
        self.s, self.env_id, self.observation, self._available = LocalEnv.SERVER, None, None, {}

    async def reset(self, session):
        if self.env_id is None:
            self.env_id = self.s.new_env()
        self.observation, self._available = self.s.reset(self.env_id, int(session))
        return self.observation

    def get_available_actions(self):
        return self._available

    async def step(self, action):
        self.observation, reward, done, self._available = self.s.step(self.env_id, action)
        return self.observation, reward, done, {}

    async def close(self):
        if self.env_id is not None:
            self.s.close(self.env_id)
            self.env_id = None


def load(label):
    t0 = time.time()
    server = WebShopEnvServer()
    print(f"{label}: {len(server.server.goals)} goals, loaded in {time.time() - t0:.0f}s, "
          f"process peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    return server


def main():
    sessions = [int(s) for s in sys.argv[1:]] or [0, 1, 4096, 4195]
    ok = True
    a = load("server A")
    ok &= len(a.server.goals) >= 4196

    gold = {}
    for s in sessions:
        env_id = a.new_env()
        try:
            gold[s] = _get_webshop_gold_action_sequence(a.envs[env_id], s)
            print(f"session {s}: expert path scores 1 in {len(gold[s])} steps: {gold[s]}")
        except Exception as e:
            print(f"session {s}: no expert path ({e!r})")
        a.close(env_id)
    ok &= bool(gold)

    b = load("server B")
    same_goals = [g["instruction_text"] for g in a.server.goals] == [g["instruction_text"] for g in b.server.goals]
    same_prices = a.server.product_prices == b.server.product_prices
    print(f"server B: same goals {same_goals}, same prices {same_prices}")
    ok &= same_goals and same_prices

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    script = {}

    class Student:
        config = types.SimpleNamespace(model_path="Qwen/Qwen3-4B")

        def get_openai_async_client(self):
            return None

        async def chat_async(self, messages, **kwargs):
            step = len(messages) // 2
            text = f"Reasoning.<action>{script['actions'][step]}</action>"
            ids = tok(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))["input_ids"]
            rid = tok(text)["input_ids"]
            return [Experience(tokens=torch.tensor(ids + rid), logprobs=-torch.rand(len(rid)),
                               prompt_length=len(ids), response_text=text)]

    class Teacher:
        def get_openai_async_client(self):
            return None

    wf.RemoteWebShopEnv = LocalEnv
    cls = WORKFLOWS.get("OPD_gated_webshop_workflow_student_teacher_lookahead")
    for s, actions in gold.items():
        # one invalid click after the search: answered by validate_action, the game is not stepped
        script["actions"] = actions[:1] + ["click[no such button]"] + actions[1:]
        task = Task(
            workflow=cls, workflow_args={"max_env_steps": 15, "defer_teacher": True},
            rollout_args=GenerationConfig(temperature=1.0, logprobs=0, max_tokens=512),
            format_args=FormatConfig(prompt_key="task_id"), raw_task={"task_id": s}, batch_id=1,
        )
        inst = task.to_workflow(Student(), [Teacher()])
        LocalEnv.SERVER = a
        turns, played, memory = asyncio.run(inst._rollout(LocalEnv("student")))
        result = f"session {s}: {len(turns)} turns, reward {inst._final_reward}, done {inst._env_done}"
        ok &= inst._final_reward == 1.0 and played == script["actions"]
        for label, server in (("A", a), ("B", b)):
            LocalEnv.SERVER = server
            matches = 0
            for k in range(len(played)):
                env, obs, history, task_desc, n, done = asyncio.run(wf.replay_game(s, played, k))
                prompt, _ = wf.build_user_content(obs, env.get_available_actions(), history, task_desc, k)
                asyncio.run(env.close())
                matches += prompt == memory[2 * k]["content"]
            result += f" | replay on {label}: {matches}/{len(played)} prompts match"
            ok &= matches == len(played)
        print(result, flush=True)
    print("history line from turn 4:", ["Prior to this step" in memory[2 * k]["content"] for k in range(len(played))])
    print("---- turn 4 user message ----")
    print(memory[6]["content"][:1500] if len(memory) > 6 else memory[-2]["content"][:1500])
    print("RESULT", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
