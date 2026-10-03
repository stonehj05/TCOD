"""Exact resume state for the Tunix trainer: fp32 master params + optax state + step.

The HF checkpoints written by the trainer are bf16 weights only; resuming from them would
round the fp32 master weights and reset Adam's moments. This module saves/restores the
worker's full training state.

`export_state` / `load_state` take the TunixActorWorker as first argument, so they also work
on a live worker started before this module existed, via Ray's
`worker_handle.__ray_call__.remote(export_state, out_dir, expected_step)`; the actor runs
calls one at a time, so this happens between train steps.

Layout of a state dir: meta.json + params/<i>.npy and opt_state/<i>.npy, one file per
pytree leaf in jax.tree_util.tree_leaves order (verified against the shapes on load).
"""

import json
import os
import time
from typing import Any, Dict

import numpy as np


def _leaves(tree):
    import jax

    return jax.tree_util.tree_leaves(tree)


def export_state(worker: Any, out_dir: str, expected_step: int) -> Dict:
    """Write worker.params / worker.opt_state / worker.step to out_dir (atomically)."""
    import jax

    if worker.step != expected_step:
        raise RuntimeError(f"worker is at step {worker.step}, expected {expected_step}; not exporting")
    t0 = time.time()
    tmp = out_dir.rstrip("/") + ".tmp"
    meta = {"step": worker.step, "params": [], "opt_state": []}
    for name, tree in (("params", worker.params), ("opt_state", worker.opt_state)):
        os.makedirs(os.path.join(tmp, name), exist_ok=True)
        for i, leaf in enumerate(_leaves(tree)):
            arr = np.asarray(jax.device_get(leaf))
            np.save(os.path.join(tmp, name, f"{i}.npy"), arr)
            meta[name].append([list(arr.shape), str(arr.dtype)])
    meta["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f)
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    os.rename(tmp, out_dir)
    return {"step": meta["step"], "params": len(meta["params"]), "opt_state": len(meta["opt_state"]),
            "seconds": meta["seconds"]}


def load_state(worker: Any, in_dir: str) -> int:
    """Replace worker.params / worker.opt_state / worker.step with the state in in_dir."""
    import jax

    with open(os.path.join(in_dir, "meta.json")) as f:
        meta = json.load(f)
    restored = {}
    for name, template in (("params", worker.params), ("opt_state", worker.opt_state)):
        leaves, treedef = jax.tree_util.tree_flatten(template)
        if len(leaves) != len(meta[name]):
            raise ValueError(f"{name}: {len(meta[name])} saved leaves, worker has {len(leaves)}")
        new = []
        for i, leaf in enumerate(leaves):
            arr = np.load(os.path.join(in_dir, name, f"{i}.npy"))
            if tuple(arr.shape) != tuple(leaf.shape) or str(arr.dtype) != str(leaf.dtype):
                raise ValueError(f"{name}[{i}]: saved {arr.shape} {arr.dtype}, worker {leaf.shape} {leaf.dtype}")
            sharding = getattr(leaf, "sharding", None)
            if isinstance(sharding, jax.sharding.NamedSharding):
                new.append(jax.device_put(arr, sharding))  # mesh-sharded params / moments
            else:
                # Scalars such as Adam's step count: leave uncommitted (no fixed device),
                # exactly as a freshly initialized optimizer state holds them; committing
                # them to one device breaks computations with the mesh-sharded leaves.
                new.append(jax.numpy.asarray(arr))
        restored[name] = jax.tree_util.tree_unflatten(treedef, new)
    worker.params, worker.opt_state, worker.step = restored["params"], restored["opt_state"], meta["step"]
    return worker.step


def resume_state_dir(sync_root: str, step: int) -> str:
    """Where a step's resume state lives: next to the weight-sync checkpoints (tmpfs)."""
    return os.path.join(sync_root.rstrip("/") + "_resume", f"global_step_{step}")
