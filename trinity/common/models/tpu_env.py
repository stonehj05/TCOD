"""TPU environment helpers shared by the vLLM-TPU engines and the Tunix trainer."""

import os
from typing import Dict

# TPU_CHIPS_PER_PROCESS_BOUNDS for a process that owns every chip of one host, by chip count.
# 4 -> "2,2,1" is verified on TPU v4 (4 chips per host). The other entries follow the same
# libtpu convention (e.g. 8 chips on one v5e/v6e host = "2,4,1") but are NOT verified here.
_HOST_BOUNDS = {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,4,1"}


def chips_per_host() -> int:
    """TPU chips on one host: $TCOD_TPU_CHIPS_PER_HOST, else the largest per-node `TPU`
    resource Ray reports, else 4 (TPU v4)."""
    if os.environ.get("TCOD_TPU_CHIPS_PER_HOST"):
        return int(os.environ["TCOD_TPU_CHIPS_PER_HOST"])
    try:
        import ray

        counts = [int(n["Resources"].get("TPU", 0)) for n in ray.nodes() if n.get("Alive")]
        if counts and max(counts) > 0:
            return max(counts)
    except Exception:
        pass
    return 4


def moe_env_vars(model_path: str) -> Dict:
    """Extra env for a Gemma 4 MoE engine (26B-A4B) on TPU v4, as tested 2026-10-27: the vLLM model path
    (the auto flax_nnx path keeps the experts unsharded and runs out of HBM) and the gmm_v2 tile patch in
    tpu_moe_site/sitecustomize.py. Empty for every other model."""
    import json
    import os

    cfg = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg):
        return {}
    with open(cfg) as f:
        c = json.load(f)
    if not (c.get("architectures", [""])[0].startswith("Gemma4")
            and c.get("text_config", c).get("enable_moe_block")):
        return {}
    site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpu_moe_site")
    path = os.environ.get("PYTHONPATH")
    return {"MODEL_IMPL_TYPE": "vllm", "PYTHONPATH": site + (":" + path if path else "")}


def whole_host_runtime_env(n_chips: int) -> Dict:
    """Ray runtime_env for an actor that takes every chip of a host (e.g. a 4-chip trainer or
    a tensor_parallel_size=4 engine on TPU v4).

    Ray sets per-host TPU bounds only for sub-host allocations; with all chips of a host
    libtpu would wait for every host of the slice to join (the process hangs at TPU init
    with "Did you run your code on all TPU hosts?"). Declare a single-process slice instead.
    Empty for sub-host allocations (Ray's own settings are right there) and for chip counts
    without a known layout.
    """
    if n_chips != chips_per_host() or n_chips not in _HOST_BOUNDS:
        return {}
    return {"env_vars": {
        "TPU_CHIPS_PER_PROCESS_BOUNDS": _HOST_BOUNDS[n_chips], "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_VISIBLE_CHIPS": ",".join(str(i) for i in range(n_chips)), "TPU_PROCESS_PORT": "8476",
        "TPU_PROCESS_ADDRESSES": "localhost:8476"}}
