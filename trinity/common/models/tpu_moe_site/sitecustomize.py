"""Gemma 4 26B-A4B (MoE) on TPU v4: vLLM-TPU's GMM MoE kernel (tokamax gmm_v2) sizes its tiles from a VMEM
estimate that undercounts on v4 (it planned tm 384 / tk 2816 / tn 256 within the 14.4 MB limit and the compiler
ran out of VMEM). Plan the tiles against GMM_TILE_FRAC (default 0.5) of the limit; the compiler limit itself is
unchanged. Tested 2026-10-27: 9000-token prompts and 32 concurrent 6000-token prompts serve; log-probs vs HF fp32
are as close as HF's own bf16 run.

Python imports this automatically when this directory is on PYTHONPATH, which
trinity.common.models.tpu_env.moe_env_vars sets only for Gemma 4 MoE engines (vLLM's engine core is a
subprocess, so an in-process patch would not reach it).
"""
import os


def _patch():
    from tokamax._src.ops.experimental.gmm_v2 import gmm_v2 as g
    from tpu_inference.layers.common import fused_moe_gmm as fg

    frac = float(os.environ.get("GMM_TILE_FRAC", "0.5"))

    def tiling(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=None):
        return g.calculate_tiling(dims, lhs_cfgs, rhs_cfgs, int(vmem_limit_bytes * frac), fuse_act)

    orig_gmm = fg.gmm_v2

    def gmm(*a, **kw):
        kw.setdefault("tile_info", tiling)
        return orig_gmm(*a, **kw)

    fg.gmm_v2 = gmm


try:
    _patch()
except Exception as e:  # never break interpreter start-up (e.g. a process without tpu_inference)
    print(f"[tpu_moe_site] gmm_v2 tile patch not applied: {e!r}")
