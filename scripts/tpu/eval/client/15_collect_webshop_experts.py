#!/usr/bin/env python
"""Collect teacher expert actions for TCOD-B2F on WebShop (user, 2026-10-22: "collect teacher experts").

TCOD's WebShop B2F replays a per-session expert action list (raw_task["actions"]), but TCOD's released
WebShop data has none. This follows UOPD's reproduction of TCOD-B2F on WebShop
(github.com/onepounchman/UOPD scripts/collect_experts.py and data/README.md): ONE greedy rollout of the
teacher per training session with the exact prompts of the training rollout, and only successful games
(score 1) whose actions pass a replay check are kept; every other session gets `actions: []` (B2F then
starts the student at the initial state).

The rollout is TCOD_b2f_webshop_workflow._run_episode (TCOD/webshop/utils.py): TCOD's WebShop template,
the last 2 observation-action pairs in the message, the full conversation as memory, its action parser;
invalid actions do not reach the game. The shop is one in-process WebShopEnvServer (env_server.py, the
training servers' seed: the same goals and prices), `text_rich` observations. Decoding: temperature 0,
max_tokens 512 (TCOD's response limit), thinking off (the trainer's default), 15 game steps.

  python 15_collect_webshop_experts.py --base-url http://localhost:8081/v1 --model teacher \\
      --tasks-jsonl ~/TCOD/TCOD_examples/webshop/webshop_data/train.jsonl --task-start 0 --task-end 1024 \\
      --output part0.jsonl

Run one job per TP4 teacher server with ../collect_webshop_experts.sh.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "../../../..")))  # the TCOD repo

from model_client import ChatModel  # noqa: E402
from trinity.common.workflows.envs.TCOD.webshop.env_server import DEFAULT_SEED, WebShopEnvServer  # noqa: E402
from trinity.common.workflows.envs.TCOD.webshop.utils import (  # noqa: E402
    HISTORY_LENGTH,
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    _extract_task_description,
    _format_available_actions,
    _format_history,
    format_observation,
    parse_action,
    validate_action,
)

SUCCESS_SCORE = 1.0


def _load_tasks(path: str, start: Optional[int], end: Optional[int]) -> List[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [{"row": r, "index": i} for i, r in enumerate(rows)][start:end]


def _user_content(observation, available, history, task_description, r) -> str:
    """The message of turn r in TCOD_b2f_webshop_workflow._run_episode."""
    formatted_observation = format_observation(observation)
    formatted_actions = _format_available_actions(available)
    if len(history) < HISTORY_LENGTH:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=formatted_observation,
            available_actions=formatted_actions,
        )
    return WEBSHOP_TEMPLATE.format(
        task_description=task_description,
        step_count=r,
        history_length=min(HISTORY_LENGTH, len(history)),
        action_history="\n".join(history[-HISTORY_LENGTH:]),
        current_step=r + 1,
        current_observation=formatted_observation,
        available_actions=formatted_actions,
    )


async def teacher_game(server: WebShopEnvServer, client: ChatModel, session: int, max_steps: int) -> dict:
    env_id = server.new_env("text_rich")
    try:
        observation, available = server.reset(env_id, session)
        task_description = _extract_task_description(observation)
        history, memory, executed = [], [], []
        score, done, rounds = 0.0, False, max_steps
        for r in range(max_steps):
            memory = memory + [{"role": "user", "content": _user_content(observation, available, history, task_description, r)}]
            text, _ = await asyncio.to_thread(client.chat, memory)
            memory = memory + [{"role": "assistant", "content": text}]
            action = parse_action(text)
            valid, error_message = validate_action(action, available)
            history.append(_format_history(format_observation(observation), r + 1, action))
            if valid:
                executed.append(action)
                observation, reward, done, available = server.step(env_id, action)
            else:
                observation, reward, done = error_message, 0.0, False
            if done:
                rounds, score = r + 1, float(reward)
                break
        return {"teacher_actions": executed, "turns": rounds, "done": bool(done), "score": score}
    finally:
        server.close(env_id)


def replay(server: WebShopEnvServer, session: int, actions: List[str]) -> float:
    """Score of replaying the executed actions on a fresh game (0 if it does not end on the last one)."""
    env_id = server.new_env("text_rich")
    try:
        server.reset(env_id, session)
        for i, action in enumerate(actions):
            _, reward, done, _ = server.step(env_id, action)
            if done:
                return float(reward) if i == len(actions) - 1 else 0.0
        return 0.0
    finally:
        server.close(env_id)


async def _run_all(tasks: List[dict], args) -> None:
    t0 = time.time()
    server = WebShopEnvServer(seed=args.seed)
    print(f"shop loaded ({len(server.server.goals)} goals) in {time.time() - t0:.0f}s", flush=True)
    client = ChatModel(base_url=args.base_url, api_key=args.api_key, model=args.model,
                       temperature=0.0, max_tokens=args.max_tokens, top_p=1.0, top_k=-1, min_p=0.0,
                       enable_thinking=False)
    done_ids = set()
    if os.path.exists(args.output):  # resume a part that was cut off
        done_ids = {json.loads(l)["index"] for l in open(args.output) if l.strip()}
    todo = [t for t in tasks if t["index"] not in done_ids]
    print(f"{len(tasks)} sessions, {len(done_ids)} already collected, {len(todo)} to go", flush=True)
    limit = asyncio.Semaphore(max(1, args.workers))
    f = open(args.output, "a")
    n_done, n_ok = 0, 0

    async def one(task):
        nonlocal n_done, n_ok
        async with limit:
            row, session = task["row"], int(task["row"]["task_id"])
            rec = {"index": task["index"], "task_id": session}
            try:
                g = await teacher_game(server, client, session, args.max_env_steps)
                ok = g["score"] >= SUCCESS_SCORE and replay(server, session, g["teacher_actions"]) >= SUCCESS_SCORE
                rec.update(g, replay_ok=ok, actions=g["teacher_actions"] if ok else [], error=None)
            except Exception as e:  # noqa: BLE001 - one bad game must not end the job
                print(f"ERROR on session {session}: {type(e).__name__}: {e}", file=sys.stderr)
                rec.update(teacher_actions=[], turns=0, done=False, score=0.0, replay_ok=False, actions=[],
                           error=f"{type(e).__name__}: {e}")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_done += 1
            n_ok += bool(rec["actions"])
            if n_done % 20 == 0 or n_done == len(todo):
                print(f"{n_done}/{len(todo)} done, {n_ok} kept, {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*[one(t) for t in todo])
    f.close()
    print(f"COMPLETE: {len(tasks)} sessions in {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True)
    p.add_argument("--tasks-jsonl", default=os.path.expanduser("~/TCOD/TCOD_examples/webshop/webshop_data/train.jsonl"))
    p.add_argument("--task-start", type=int, default=None)
    p.add_argument("--task-end", type=int, default=None)
    p.add_argument("--workers", type=int, default=32, help="games played at once")
    p.add_argument("--max-env-steps", type=int, default=15)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the shop's seed (training's servers: 0)")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    tasks = _load_tasks(args.tasks_jsonl, args.task_start, args.task_end)
    if not tasks:
        raise SystemExit(f"no tasks in {args.tasks_jsonl}[{args.task_start}:{args.task_end}]")
    asyncio.run(_run_all(tasks, args))


if __name__ == "__main__":
    main()
