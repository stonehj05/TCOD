#!/usr/bin/env python
"""Evaluate a trained TCOD checkpoint (e.g. the alfworld_opd_gated run's
global_step_250) on ALFWorld's held-out splits.

Unlike 01_generate_trajectories.py (which uses the DASH-OPD-paper-matching
prompt/context format for the teacher/student consistency probes),
this script uses tcod_alfworld_utils.py / run_episode_tcod.py -- TCOD's
actual trained-on prompt template and single-turn (no growing memory)
context construction -- so the measured success rate reflects what the
checkpoint actually learned, not a differently-formatted eval protocol.

Prereqs: serve the checkpoint's HF export with vLLM, e.g.:

  vllm serve \\
      /data/hs2352/LookAheadOPD/alfworld_ts_probe/checkpoints/ALFWORLD_TCOD/alfworld_opd_gated_20260917183426/global_step_250/actor/huggingface \\
      --port 8010 --served-model-name opd_gated_step250 --tensor-parallel-size 1

Then run, e.g. on the OOD ("unseen") split -- TCOD paper's primary
generalization metric:

  python 12_evaluate_checkpoint.py \\
      --base-url http://localhost:8010/v1 --model opd_gated_step250 \\
      --split unseen \\
      --output data/eval_opd_gated_step250.unseen.jsonl \\
      --summary data/eval_opd_gated_step250.unseen.summary.json

--split also accepts "seen" (test.jsonl), "both" (pooled seen+unseen), and
"hard" (train_hard.jsonl -- the 121-task set the 30B-A3B teacher itself
fails under pass@10 sampling; TCOD paper Sec 5.3's "does the student
surpass the teacher's own capability boundary" metric).

Eval hyperparameters default to the TCOD paper's stated protocol (Appendix
D.4 / Table 5): temperature=0.4, top_p=1.0, top_k=-1, min_p=0.0,
max_tokens=4096, max_env_steps=30, enable_thinking=False -- all already
ChatModel's defaults in model_client.py, so the only required flags are
--base-url/--model/--split/--output/--summary.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from model_client import ChatModel  # noqa: E402
from tcod_alfworld_utils import _create_alfworld_env, _extract_task  # noqa: E402

# Two rollout loops, selected by --full-memory:
#   default (run_episode_tcod.run_episode): single self-contained turn per
#     step, capped HISTORY_LENGTH=2 textual history -- matches what a
#     TCOD-trained checkpoint actually saw during training.
#   --full-memory (run_episode.run_episode): a growing `memory` list of
#     every turn's user+assistant messages passed to the model each step --
#     tests the hypothesis that the paper's own reported numbers (e.g. the
#     teacher's standalone Table 3 SR) were produced with the full
#     conversation accumulating, not the capped per-turn summary.
import run_episode_tcod  # noqa: E402
import run_episode as run_episode_full_memory  # noqa: E402

_ALFWORLD_DATA_DIR = "/data/hs2352/LookAheadOPD/TCOD/TCOD_examples/alfworld/alfworld_data"
DEFAULT_SEEN_JSONL = f"{_ALFWORLD_DATA_DIR}/test.jsonl"
DEFAULT_UNSEEN_JSONL = f"{_ALFWORLD_DATA_DIR}/test_unseen.jsonl"
DEFAULT_HARD_JSONL = f"{_ALFWORLD_DATA_DIR}/train_hard.jsonl"


def _load_jsonl_games(path: str, split_label: str) -> list:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append({"game_file": json.loads(line)["game_file"], "split": split_label})
    return tasks


def _load_tasks(
    seen_jsonl: str,
    unseen_jsonl: str,
    hard_jsonl: str,
    split: str,
    task_start: Optional[int],
    task_end: Optional[int],
) -> list:
    sources = []
    if split in ("seen", "both"):
        sources.append((seen_jsonl, "seen"))
    if split in ("unseen", "both"):
        sources.append((unseen_jsonl, "unseen"))
    if split == "hard":
        sources.append((hard_jsonl, "hard"))

    tasks = []
    for path, label in sources:
        games = _load_jsonl_games(path, label)
        if task_start is not None or task_end is not None:
            games = games[task_start:task_end]
        tasks.extend(games)
    return tasks


def _split_shards(tasks: list, n_workers: int) -> list:
    n_workers = max(1, min(n_workers, len(tasks)))
    shards = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        shards[i % n_workers].append(task)
    return [s for s in shards if s]


def _worker(
    worker_id: int,
    tasks: list,
    base_url: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    min_p: float,
    enable_thinking: bool,
    max_env_steps: int,
    mock: bool,
    shard_output_path: str,
    keep_step_log: bool,
    full_memory: bool,
):
    t0 = time.time()
    print(f"[worker {worker_id}] processing {len(tasks)} games (full_memory={full_memory})")
    run_episode = run_episode_full_memory.run_episode if full_memory else run_episode_tcod.run_episode

    model_client = ChatModel(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        enable_thinking=enable_thinking,
        mock=mock,
    )

    with open(shard_output_path, "w") as f:
        for i, task in enumerate(tasks):
            game_file = task["game_file"]
            env = None
            try:
                env = _create_alfworld_env(game_file)
                observation, info = env.reset()
                task_description = _extract_task(observation)

                result = run_episode(
                    env=env,
                    model_client=model_client,
                    task_description=task_description,
                    observation=observation,
                    info=info,
                    history=None,
                    start_step=0,
                    max_steps=max_env_steps,
                )

                record = {
                    "game_file": game_file,
                    "split": task["split"],
                    "task_description": task_description,
                    "actions": result["actions"],
                    "env_rounds": result["env_rounds"],
                    "final_reward": result["final_reward"],
                    "done": result["done"],
                    "success": result["done"],
                    "error": None,
                }
                if keep_step_log:
                    record["step_log"] = result["step_log"]
            except Exception as e:  # noqa: BLE001 - a single bad game must not
                # take down the rest of this worker's shard.
                print(
                    f"[worker {worker_id}] ERROR on game_file={game_file}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                record = {
                    "game_file": game_file,
                    "split": task["split"],
                    "task_description": None,
                    "actions": [],
                    "env_rounds": 0,
                    "final_reward": 0.0,
                    "done": False,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
                if keep_step_log:
                    record["step_log"] = []
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

            f.write(json.dumps(record) + "\n")
            f.flush()

            if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
                print(
                    f"[worker {worker_id}] {i + 1}/{len(tasks)} games done "
                    f"(last split={record['split']}, success={record['success']}, "
                    f"error={record['error']})"
                )

    print(f"[worker {worker_id}] finished shard in {time.time() - t0:.1f}s")


def _write_summary(records: list, summary_path: str, checkpoint_label: str):
    split_labels = sorted(set(r["split"] for r in records))
    per_split = {}
    for label in split_labels:
        split_records = [r for r in records if r["split"] == label]
        n = len(split_records)
        n_success = sum(1 for r in split_records if r["success"])
        success_rounds = [r["env_rounds"] for r in split_records if r["success"]]
        n_errors = sum(1 for r in split_records if r.get("error"))
        per_split[label] = {
            "n_tasks": n,
            "n_success": n_success,
            "success_rate": n_success / n if n else 0.0,
            "avg_rounds_on_success": (
                sum(success_rounds) / len(success_rounds) if success_rounds else None
            ),
            "n_errors": n_errors,
        }

    n_total = len(records)
    n_success_total = sum(1 for r in records if r["success"])
    summary = {
        "checkpoint": checkpoint_label,
        "n_tasks_total": n_total,
        "n_success_total": n_success_total,
        "success_rate_overall": n_success_total / n_total if n_total else 0.0,
        "per_split": per_split,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="e.g. http://localhost:8010/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True, help="served-model-name on the vLLM server")
    parser.add_argument("--checkpoint-label", default=None,
                         help="Free-text label recorded in the summary JSON "
                              "(default: same as --model)")
    parser.add_argument("--seen-jsonl", default=DEFAULT_SEEN_JSONL,
                         help="140-task 'seen' held-out split (test.jsonl)")
    parser.add_argument("--unseen-jsonl", default=DEFAULT_UNSEEN_JSONL,
                         help="134-task OOD 'unseen' split (test_unseen.jsonl)")
    parser.add_argument("--hard-jsonl", default=DEFAULT_HARD_JSONL,
                         help="121-task 'hard' split (train_hard.jsonl) -- tasks the "
                              "30B-A3B teacher itself fails under pass@10 sampling")
    parser.add_argument("--split", choices=["seen", "unseen", "both", "hard"], default="unseen",
                         help="default 'unseen' -- the paper's primary OOD "
                              "generalization metric. 'both' pools seen+unseen "
                              "(274 tasks)")
    parser.add_argument("--task-start", type=int, default=None,
                         help="0-indexed slice start into each selected split's game "
                              "list. Unset (default) = use every game in the split(s)")
    parser.add_argument("--task-end", type=int, default=None,
                         help="0-indexed slice end (exclusive), paired with --task-start")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-env-steps", type=int, default=30,
                         help="TCOD's ALFWorld max_env_steps (matches training config)")
    parser.add_argument("--temperature", type=float, default=0.4,
                         help="TCOD paper Table 5 eval temperature (training used 1.0)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true",
                         help="Off by default, matching how this checkpoint was trained")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--keep-step-log", action="store_true", default=True)
    parser.add_argument("--no-keep-step-log", dest="keep_step_log", action="store_false")
    parser.add_argument("--full-memory", action="store_true",
                         help="Use run_episode.py's growing-conversation rollout "
                              "loop (every turn's full user+assistant messages "
                              "kept and resent) instead of the default "
                              "single-self-contained-turn loop TCOD checkpoints "
                              "were actually trained on. For testing whether the "
                              "paper's own reported numbers assume full memory.")
    args = parser.parse_args()

    tasks = _load_tasks(
        args.seen_jsonl, args.unseen_jsonl, args.hard_jsonl, args.split,
        args.task_start, args.task_end,
    )
    if not tasks:
        raise SystemExit("No tasks loaded -- check --seen-jsonl/--unseen-jsonl/--hard-jsonl/--split.")

    shards = _split_shards(tasks, args.workers)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    shard_paths = [f"{args.output}.shard{i}.tmp" for i in range(len(shards))]

    split_counts = {}
    for t in tasks:
        split_counts[t["split"]] = split_counts.get(t["split"], 0) + 1
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items()))
    print(
        f"Evaluating model={args.model!r} on {len(tasks)} tasks ({counts_str}) "
        f"across {len(shards)} worker(s), max_env_steps={args.max_env_steps}, "
        f"temperature={args.temperature}, mock={args.mock_model}, "
        f"full_memory={args.full_memory}"
    )

    ctx = mp.get_context("spawn")
    procs = []
    for i, (shard, shard_path) in enumerate(zip(shards, shard_paths)):
        p = ctx.Process(
            target=_worker,
            kwargs=dict(
                worker_id=i,
                tasks=shard,
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                enable_thinking=args.enable_thinking,
                max_env_steps=args.max_env_steps,
                mock=args.mock_model,
                shard_output_path=shard_path,
                keep_step_log=args.keep_step_log,
                full_memory=args.full_memory,
            ),
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"WARNING: worker pid={p.pid} exited with code {p.exitcode}", file=sys.stderr)

    all_records = []
    for shard_path in shard_paths:
        if not os.path.exists(shard_path):
            continue
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
        os.remove(shard_path)

    all_records.sort(key=lambda r: (r["split"], r["game_file"]))
    with open(args.output, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    checkpoint_label = args.checkpoint_label or args.model
    summary = _write_summary(all_records, args.summary, checkpoint_label)

    n_errors = sum(1 for r in all_records if r.get("error"))
    per_split_lines = [
        f"  {label}: {d['n_success']}/{d['n_tasks']} = {d['success_rate']:.1%}"
        + (f" (avg {d['avg_rounds_on_success']:.1f} steps on success)" if d["avg_rounds_on_success"] else "")
        for label, d in sorted(summary["per_split"].items())
    ]
    print(
        f"Done: {len(all_records)}/{len(tasks)} tasks written to {args.output}\n"
        + "\n".join(per_split_lines) + "\n"
        f"  Overall: {summary['n_success_total']}/{summary['n_tasks_total']} = "
        f"{summary['success_rate_overall']:.1%}"
        f"{f', {n_errors} task(s) errored' if n_errors else ''}\n"
        f"Summary written to {args.summary}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
