#!/usr/bin/env python3
"""Rollout (explorer) speed of a Trinity run: one row per explore step + summary.
Usage: rollout_speed.py RUN_DIR [LAST_N]   (RUN_DIR = checkpoints/<project>/<run name>; default last 12 steps)"""
import ast, datetime, re, statistics, sys
run = sys.argv[1]
last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows, starts = [], {}
ts = lambda s: datetime.datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S")
for l in open(run + "/log/explorer.log"):
    m = re.match(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Explore step (\d+) started", l)
    if m: starts[int(m.group(2))] = ts(m.group(1))
    m = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Step (\d+): (\{'rollout/model_version'.*\})", l)
    if m:
        try: rows.append((int(m.group(2)), ts(m.group(1)), ast.literal_eval(m.group(3))))
        except Exception: pass
tr = [ts(m.group(1)) for l in open(run + "/log/trainer.log") if (m := re.match(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Training at step \d+ finished", l))]
print(f"{'step':>4} {'end(UTC)':>8} {'rollout':>8} {'wall':>6} {'turns':>5} {'turns/min':>9} {'done':>5} {'rounds':>6} {'ver':>4} {'w=1':>5} {'w=.5':>5} {'w=0':>5}")
for n, t, d in rows[-last_n:]:
    wait = d.get("time/wait_explore_step", 0) / 60; turns = d.get("experience_pipeline/experience_count", 0)
    wall = (starts[n + 1] - starts[n]).total_seconds() / 60 if n in starts and n + 1 in starts else float("nan")
    g = lambda k: d.get(f"rollout/{k}/mean", float("nan"))
    print(f"{n:>4} {t:%H:%M:%S} {wait:7.1f}m {wall:5.1f}m {turns:5.0f} {turns / wait if wait else 0:9.1f} {g('env_done'):5.2f} {g('env_rounds'):6.1f} "
          f"{d.get('rollout/model_version', 0):4.0f} {g('opd_gate_full_rate'):5.2f} {g('opd_gate_half_rate'):5.2f} {g('opd_gate_none_rate'):5.2f}")
if rows:
    w = [d.get("time/wait_explore_step", 0) / 60 for _, _, d in rows]; x = [d.get("experience_pipeline/experience_count", 0) for _, _, d in rows]
    span = (rows[-1][1] - starts[min(starts)]).total_seconds() / 3600
    print(f"\n{len(rows)} explore steps in {span:.2f} h | rollout time/step: median {statistics.median(w):.1f} min, last-10 mean {statistics.mean(w[-10:]):.1f} min "
          f"| turns/step mean {statistics.mean(x):.0f} | overall {sum(x) / (span * 60):.1f} turns/min | trainer steps done: {len(tr)}")
