"""Full-history memory check: teacher (Qwen3-30B-A3B, bf16) and student (Qwen3-4B, fp32)
scoring passes on 10240-token prompts + 512-token completions with flash attention.
Run: python tests/check_long_scoring.py [rows]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402
from tunix.rl import common  # noqa: E402
from tunix.utils.mesh import create_mesh  # noqa: E402

from models import load_model  # noqa: E402

P, C = 10240, 512
rows = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def hbm(tag):
    s = [d.memory_stats() for d in jax.local_devices()]
    print(f"[hbm] {tag}: peak {max(x['peak_bytes_in_use'] for x in s)/2**30:.1f} GiB, "
          f"in use {max(x['bytes_in_use'] for x in s)/2**30:.1f} GiB of {s[0]['bytes_limit']/2**30:.1f}", flush=True)


mesh = create_mesh((jax.device_count(), 1), ("fsdp", "tp"))
flash = dict(use_flash_attention=True, flash_attention_block_size=512)
with mesh:
    student, tok, _ = load_model("Qwen/Qwen3-4B", mesh, dtype=jnp.float32, **flash)
    teacher, _, _ = load_model("Qwen/Qwen3-30B-A3B", mesh, dtype=jnp.bfloat16, **flash)
    hbm("models loaded")
    pad, eos = tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")
    rng = np.random.default_rng(0)
    prompt = jnp.asarray(rng.integers(1000, 100000, (rows, P)), jnp.int32)
    comp = jnp.asarray(rng.integers(1000, 100000, (rows, C)), jnp.int32)
    for name, m in [("teacher", teacher), ("student", student)]:
        gd, st = nnx.split(m)
        fn = jax.jit(lambda st, p, c: common.compute_per_token_logps(
            gd, st, prompt_tokens=p, completion_tokens=c, pad_id=pad, eos_id=eos, stop_gradient=True))
        try:
            for i in range(2):
                t = time.time()
                jax.block_until_ready(fn(st, prompt, comp))
                print(f"{name} {rows}x({P}+{C}) scoring pass {i}: {time.time()-t:.1f}s", flush=True)
            hbm(f"after {name} scoring")
        except Exception as e:
            print(f"{name} LONG SCORING FAILED:", str(e).splitlines()[0][:400], flush=True)
