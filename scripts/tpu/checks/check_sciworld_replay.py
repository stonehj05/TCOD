"""Pre-flight: do the ScienceWorld teacher look-aheads replay the student's game exactly? Plays training
games as the gated ScienceWorld workflow does (build_user_content every turn, no other game calls), with
actions like a student's (templates filled with listed objects, numbers choosing from ScienceWorld's option
lists, and junk), then runs the workflow's own `teacher_lookahead` (a stand-in teacher that only says
"look around") from several turns of each game and requires replay_ok for every one. CPU only.

  python scripts/tpu/checks/check_sciworld_replay.py [N_GAMES] [SEED]
  python scripts/tpu/checks/check_sciworld_replay.py --db RUN_DIR/buffer/explorer_output.db [N_TURNS]

Exit code 1 if any replay differs (with the first differing prompt line). The look-ahead used to load the
game with generateGoldPath on (TCOD's _create_scienceworld_env_with_checkpoint), which failed this check
on ~7% of replays.
--db: replays real training turns instead (the deferred payloads in a run's experience DB; copy the DB
first if the run is still writing it) and reports the share that match; exit code 1 below 98%. Real games
catch what the simulated student does not: ScienceWorld names objects differently unless the object list
is requested every turn, which ~23% of real turns showed and the simulated games never did.
"""
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))  # the TCOD repo
from trinity.common.workflows.envs.TCOD.scienceworld import utils as u  # noqa: E402
from trinity.common.workflows.envs.TCOD.scienceworld.OPD_gated_workflow_student_teacher_lookahead import (  # noqa: E402
    build_user_content,
    teacher_lookahead,
)

TRAIN = os.path.expanduser("~/sci-data/tcod_tasks/train.jsonl")
JUNK = ["", "look around", "go to kitchen", "1", "2", "0", "pour paint in cup into inventory", "move cup to table"]
GATE = {"window_size": 2, "teacher_rollout_temperature": 0.0, "teacher_rollout_max_tokens": 64,
        "progress_temperature": 0.0, "progress_max_tokens": 8}


async def ask(messages, temperature, max_tokens):  # stand-in teacher
    return "No" if "progress" in messages[-1]["content"].lower() else "<action>look around</action>"


def play(desc, rng, max_steps=20):
    """The student's game as the training rollout plays it; returns its conversation and actions."""
    env = u._create_scienceworld_env(desc, max_env_steps=30, generate_gold_path=False)
    obs, info, task = u._reset_scienceworld_env(env)
    hist, memory, actions = [], [], []
    for r in range(max_steps):
        content, fo = build_user_content(env, obs, hist, task, r)
        memory.append({"role": "user", "content": content})
        tmpl = [s.strip("'") for s in content.split("Available action commands: [")[1].split("]")[0].split("', '")]
        objs = [s.strip("'") for s in content.split("interact with: [")[1].split("]")[0].split("', '")]
        a = rng.choice(JUNK) if rng.random() < 0.3 else rng.choice([x for x in tmpl if not x.startswith("focus")])
        while "OBJ" in a:
            a = a.replace("OBJ", rng.choice(objs), 1)
        memory.append({"role": "assistant", "content": f"<action>{a}</action>"})
        actions.append(a)
        hist.append(u._format_history(fo, r + 1, a))
        obs, _, done, info = env.step(a)
        if done:
            break
    env.close()
    return memory, actions


def main(n_games: int, seed: int) -> None:
    rng = random.Random(seed)
    rows = [json.loads(l)["task_desc"] for l in open(TRAIN) if l.strip()]
    n = bad = 0
    for desc in rng.sample(rows, n_games):
        memory, actions = play(desc, rng)
        for t in range(1, len(actions), 2):
            res = asyncio.run(teacher_lookahead(ask, desc, memory[: 2 * t + 1], actions, t, GATE, actions[t]))
            n += 1
            if not res["replay_ok"]:
                bad += 1
                print(f"MISMATCH {json.loads(desc)['task_name']} turn {t + 1}; actions so far {actions[:t]}")
        print(f"{json.loads(desc)['task_name']:<45} look-aheads so far {n}, replay mismatches {bad}", flush=True)
    print(f"RESULT {n - bad}/{n} look-ahead replays match the student's prompt")
    sys.exit(1 if bad else 0)


def main_db(db: str, n_turns: int) -> None:
    import pickle
    import sqlite3
    import zlib

    from trinity.common.workflows.envs.TCOD.scienceworld.OPD_gated_workflow_student_teacher_lookahead import replay_game

    rows = [pickle.loads(b).info.get("opd_deferred") for (b,) in sqlite3.connect(db).execute(
        "select experience_bytes from pipeline_input")]
    rows = [p for p in rows if p and p.get("env") == "scienceworld" and p["step"] > 0]
    random.Random(0).shuffle(rows)
    n = bad = 0
    for p in rows[:n_turns]:
        t = p["step"]
        student = json.loads(zlib.decompress(p["memory_z"]))[2 * t]["content"]
        env, obs, info, hist, task, n_rep, done = replay_game(p["game_file"], p["prior_actions"], t)
        try:
            replayed = None if done or n_rep != t else build_user_content(env, obs, hist, task, t)[0]
        finally:
            env.close()
        n += 1
        if replayed != student:
            bad += 1
            a, b = student.split("\n"), (replayed or "").split("\n")
            i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"MISMATCH {json.loads(p['game_file'])['task_name']} turn {t + 1}: student {a[i][:120] if i < len(a) else ''!r}"
                  f" / replay {b[i][:120] if i < len(b) else ''!r}", flush=True)
    print(f"RESULT {n - bad}/{n} real training turns replay to the student's prompt")
    sys.exit(1 if n and (n - bad) / n < 0.98 else 0)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--db":
        main_db(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 150)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 12, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
