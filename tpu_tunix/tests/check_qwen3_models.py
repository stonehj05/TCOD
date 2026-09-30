"""Check the Qwen3-4B student / Qwen3-30B-A3B teacher pair on the local TPU chips.

- both load as Tunix models (student fp32, teacher bf16) and fit in HBM together
- tokenizers are identical (required for token-level OPD)
- the student generates an ALFWorld turn with thinking disabled (as in Trinity)
- teacher and student score the student's response; teacher logps are sane
- a full-length (10240-token prompt + 512 response) teacher scoring pass fits
Run: python tests/check_qwen3_models.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402
from tunix.generate import sampler as sampler_lib  # noqa: E402
from tunix.rl import common  # noqa: E402
from tunix.utils.mesh import create_mesh  # noqa: E402

from alfworld_env import AlfworldEpisode  # noqa: E402
from models import load_model  # noqa: E402

STUDENT, TEACHER = "Qwen/Qwen3-4B", "Qwen/Qwen3-30B-A3B"


def hbm(tag):
    stats = [d.memory_stats() for d in jax.local_devices()]
    used = [s["bytes_in_use"] / 2**30 for s in stats]
    limit = stats[0]["bytes_limit"] / 2**30
    print(f"[hbm] {tag}: per-chip used {min(used):.1f}-{max(used):.1f} GiB of {limit:.1f} GiB", flush=True)


mesh = create_mesh((jax.device_count(), 1), ("fsdp", "tp"))
with mesh:
    t = time.time()
    student, tok, _ = load_model(STUDENT, mesh, dtype=jnp.float32, use_flash_attention=True)
    print(f"student loaded {time.time()-t:.0f}s", flush=True)
    hbm("student fp32")
    t = time.time()
    teacher, ttok, _ = load_model(TEACHER, mesh, dtype=jnp.bfloat16, use_flash_attention=True)
    print(f"teacher loaded {time.time()-t:.0f}s", flush=True)
    hbm("student fp32 + teacher bf16")

    same_vocab = tok.get_vocab() == ttok.get_vocab()
    print("tokenizers identical:", same_vocab,
          "| student chat template == teacher:", tok.chat_template == ttok.chat_template)
    assert same_vocab

    # One ALFWorld turn, rendered like Trinity (enable_thinking=False).
    game = next(os.path.join(r, "game.tw-pddl") for r, _, f in os.walk(os.path.expanduser("~/alf-data/json_2.1.1/train"))
                if "game.tw-pddl" in f)
    ep = AlfworldEpisode(game)
    prompt = tok.apply_chat_template(ep.messages(), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ep.close()
    print("prompt tail:", repr(prompt[-120:]))
    cfg = student.config
    s = sampler_lib.Sampler(student, tok, sampler_lib.CacheConfig(
        cache_size=2048 + 512, num_layers=cfg.num_layers, num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim))
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id]
    t = time.time()
    out = s(input_strings=[prompt] * 4, max_generation_steps=512, max_prompt_length=2048, temperature=1.0,
            eos_tokens=eos, seed=0, pad_output=True)
    print(f"student generate x4: {time.time()-t:.0f}s")
    for txt in out.text[:2]:
        print("STUDENT:", repr(txt[:300]))

    pad, im_end = tok.pad_token_id, eos[0]
    comps = [np.append(np.asarray(x)[np.asarray(x) != pad], im_end) for x in out.tokens]
    comp = np.full((4, 512), pad, np.int32)
    for i, c in enumerate(comps):
        comp[i, :len(c)] = c[:512]
    prompt_ids, comp_ids = jnp.asarray(out.padded_prompt_tokens), jnp.asarray(comp)
    mask = comp != pad
    for name, m in [("student", student), ("teacher", teacher)]:
        gd, st = nnx.split(m)
        t = time.time()
        lp = np.asarray(common.compute_per_token_logps(gd, st, prompt_tokens=prompt_ids, completion_tokens=comp_ids,
                                                       pad_id=pad, eos_id=im_end, stop_gradient=True))
        print(f"{name} logp/token on student samples: {(lp*mask).sum()/mask.sum():.3f} ({time.time()-t:.0f}s)")
    hbm("after scoring")

    # Full-length scoring pass: 10240-token prompt + 512-token completion, 4 rows (1 per chip).
    rng = np.random.default_rng(0)
    long_prompt = jnp.asarray(rng.integers(1000, 100000, (4, 10240)), jnp.int32)
    long_comp = jnp.asarray(rng.integers(1000, 100000, (4, 512)), jnp.int32)
    gd, st = nnx.split(teacher)
    fn = jax.jit(lambda st, p, c: common.compute_per_token_logps(gd, st, prompt_tokens=p, completion_tokens=c,
                                                                 pad_id=pad, eos_id=im_end, stop_gradient=True))
    try:
        for i in range(2):
            t = time.time()
            jax.block_until_ready(fn(st, long_prompt, long_comp))
            print(f"teacher 4x(10240+512) scoring pass {i}: {time.time()-t:.1f}s")
        hbm("after long scoring")
    except Exception as e:  # report OOM instead of crashing the whole check
        print("LONG SCORING FAILED:", str(e).splitlines()[0][:300])
