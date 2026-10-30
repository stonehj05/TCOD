"""Gemma 4 support for the `tunix` trainer (user, 2026-10-26: Gemma 4 E2B student -> Gemma 4 31B teacher).

The Qwen2/Qwen3 paths in hf_io.py / logps.py / worker.py are unchanged; they dispatch here when the
model directory holds a Gemma 4 checkpoint (`Gemma4ForConditionalGeneration`, text weights only).

What differs from Qwen3, and why:
  * Flash (splash) attention: Tunix's Gemma 4 version has two bugs, fixed in-process by
    `apply_flash_fix()` (the installed package is untouched):
      1. KV-sharing layers (E2B: the last 20 of 35 reuse an earlier layer's k/v) got the k/v AFTER
         the flash path transposed them to [B, N, T, H], so T was read as the head count;
      2. the flash path dropped the attention mask, so real tokens attended to the trainer's left
         padding. Per-position segment ids (as in Tunix's Qwen3) now carry it: token t is real iff it
         may attend to itself (the mask's diagonal).
    Verified on a left-padded batch: flash == the non-flash path == Hugging Face (CPU) to bf16 noise.
  * Flash block 256 (tunix_trainer.py): 512 exceeds TPU v4's 16 MiB VMEM with the 512-wide global heads.
  * The per-layer embedding table (`embed_tokens_per_layer`, 2.35B of E2B's 4.65B parameters) is
    FROZEN (user, 2026-10-26) and kept in bf16: no gradient update, no optimizer state. With it trained
    the step needs ~1 GiB more than a v4 chip has. Everything else (2.30B) is trained as for Qwen.
  * Forward: per-layer inputs, KV sharing between layers, final logit soft-capping (30), tied LM head.
  * HF export: the inverse of tunix/models/gemma4/params_safetensors.py's mapping, written over a copy
    of the base checkpoint's tensors (vision / audio towers, the frozen table and the unused k/v of the
    KV-sharing layers stay as released), so the result loads in vLLM / transformers like the original.
"""

import dataclasses
import inspect
import json
import os
import re
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

ARCHS = ("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")
FROZEN_SUBSTR = "per_layer_input_embedding"  # the per-layer embedding table
FLASH_BLOCK = 256
END_OF_TURN = "<turn|>"
_HF_PREFIX = "model.language_model."


def is_gemma4(model_dir: str) -> bool:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)["architectures"][0] in ARCHS


def _text_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        hf = json.load(f)
    return hf.get("text_config", hf)


def config_name(model_dir: str) -> str:
    """Tunix preset (tunix/models/gemma4/model.py ModelConfig.<name>) from the HF config's shape."""
    t = _text_config(model_dir)
    known = {(35, 1536): "gemma4_e2b", (42, 2560): "gemma4_e4b", (60, 5376): "gemma4_31b",
             (30, 2816): "gemma4_26b_a4b"}
    name = known.get((t["num_hidden_layers"], t["hidden_size"]))
    if name is None:
        raise NotImplementedError(f"no Tunix Gemma 4 preset for (layers, hidden)="
                                  f"{(t['num_hidden_layers'], t['hidden_size'])}")
    return name


# ---------------------------------------------------------------------------------------- flash attention

_FLASH_IF = "    if self.config.use_flash_attention and seq_len > 1:\n"
_OLD_SHARD = """      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block):
        return jax.vmap(kernel)(q_block, k_block, v_block)

      qkv: jaxtyping.Array = sharded_splash_attn(
          splash_attn_kernel, query_proj, key_proj, value_proj
      )
"""
_NEW_SHARD = """      seg = jnp.diagonal(attn_mask[..., :seq_len], axis1=-2, axis2=-1).astype(jnp.int32)

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq, P(shd_b, shd_t), P(shd_b, None)),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block, q_seg, kv_seg):
        return jax.vmap(kernel)(q_block, k_block, v_block,
                                segment_ids=splash.SegmentIds(q=q_seg, kv=kv_seg))

      qkv: jaxtyping.Array = sharded_splash_attn(
          splash_attn_kernel, query_proj, key_proj, value_proj, seg, seg
      )
"""
_OLD_RET = "    return new_cache, attn_output, (key_proj, value_proj)\n"


def apply_flash_fix():
    """Install the fixed Attention.block (see the module docstring) in this process."""
    from tunix.models.gemma4 import model as g4

    if getattr(g4.Attention, "_tcod_flash_fixed", False):
        return
    src = inspect.getsource(g4.Attention.block)
    for old in (_FLASH_IF, _OLD_SHARD, _OLD_RET):
        if src.count(old) != 1:
            raise RuntimeError("tunix gemma4 Attention.block changed; the flash fix no longer applies")
    src = src.replace(_FLASH_IF, "    kv_out = (key_proj, value_proj)  # un-transposed, for KV-sharing layers\n"
                      + _FLASH_IF)
    src = src.replace(_OLD_SHARD, _NEW_SHARD).replace(_OLD_RET, "    return new_cache, attn_output, kv_out\n")
    ns = {}
    exec(compile(textwrap.dedent(src), "<tcod gemma4 flash fix>", "exec"), g4.__dict__, ns)
    g4.Attention.block = ns["block"]
    g4.Attention._tcod_flash_fixed = True


# ---------------------------------------------------------------------------------------- model

def load_model(model_dir: str, mesh, dtype=jnp.float32, *, remat: str = "DECODER",
               flash_block: int = FLASH_BLOCK, use_flash_attention: bool = True):
    from tunix.models.gemma4 import model as g4
    from tunix.models.gemma4 import params_safetensors as g4p

    if use_flash_attention:
        apply_flash_fix()
    config = dataclasses.replace(
        getattr(g4.ModelConfig, config_name(model_dir))(),
        remat_config=getattr(g4.RematConfig, remat),
        use_flash_attention=use_flash_attention,
        use_sliding_window_kv_cache=not use_flash_attention,  # mutually exclusive in Tunix
        flash_attention_block_size=flash_block,
    )
    return g4p.create_model_from_safe_tensors(model_dir, config, mesh, dtype=dtype)


def is_frozen(path) -> bool:
    """A param path (jax tree path) of the frozen per-layer embedding table."""
    return FROZEN_SUBSTR in jax.tree_util.keystr(path)


def freeze(params):
    """Frozen leaves -> bf16 (they are never updated; halves their memory)."""
    return jax.tree_util.tree_map_with_path(
        lambda p, x: x.astype(jnp.bfloat16) if is_frozen(p) else x, params)


def stop_frozen_grad(params):
    """Inside a loss: no backward through the frozen table (its dense gradient alone would be
    ~9 GiB fp32 per step; the optimizer would discard it anyway)."""
    return jax.tree_util.tree_map_with_path(
        lambda p, x: jax.lax.stop_gradient(x) if is_frozen(p) else x, params)


def optimizer(tx, params):
    """`tx` for the trained leaves; frozen leaves get no update and no optimizer state."""
    import optax

    labels = jax.tree_util.tree_map_with_path(lambda p, _: "frozen" if is_frozen(p) else "train", params)
    return optax.multi_transform({"train": tx, "frozen": optax.set_to_zero()}, labels)


def hidden_states(model, input_tokens, positions, attn_mask):
    """Gemma4.__call__ (the prefill path) without the LM head: [B, T, D]."""
    x = model.embedder.encode(input_tokens)
    pli = (model.embedder.encode_per_layer_input(x, input_tokens)
           if model.config.per_layer_input_dim > 0 else None)
    kvs = {}
    for i, layer in enumerate(model.layers):
        src = model.kv_cache_sharing_patterns[i]
        shared = {"k": kvs[src][0], "v": kvs[src][1]} if src != i else None
        _, x, layer_kvs = layer(x, positions, None, attn_mask,
                                per_layer_input=pli[:, :, i, :] if pli is not None else None,
                                kv_shared_cache=shared)
        if i in model.shared_layer_origins:
            kvs[i] = layer_kvs
    return model.final_norm(x)


def project(model, h):
    """Tied LM head + Gemma's final logit soft-capping."""
    logits = model.embedder.decode(h)
    cap = model.config.final_logit_softcap
    if cap is not None:
        logits = jnp.tanh(logits.astype(jnp.float32) / cap) * cap
    return logits


# ---------------------------------------------------------------------------------------- HF export

def _hf_tensors(model, dtype=None) -> dict:
    """NNX params -> {hf_name: np.ndarray} for the text weights Tunix holds, in HF layout (the inverse of
    tunix/models/gemma4/params_safetensors.py). The frozen per-layer table is skipped (never changes)."""
    cfg = model.config
    params = nnx.state(model, nnx.Param)
    flat = {".".join(str(k) for k in path): v for path, v in nnx.to_flat_state(params)}
    out = {}

    def get(key):
        v = flat[key]
        v = getattr(v, "value", v)
        return v.astype(dtype) if dtype is not None else v

    def put(name, x):
        out[_HF_PREFIX + name] = np.asarray(jax.device_get(x))

    put("embed_tokens.weight", get("embedder.input_embedding"))
    put("norm.weight", get("final_norm.scale"))
    if cfg.per_layer_input_dim > 0:
        w = get("embedder.per_layer_model_projection.w")  # [D, L, P] -> [L*P, D]
        put("per_layer_model_projection.weight", w.reshape(w.shape[0], -1).T)
        put("per_layer_projection_norm.weight", get("embedder.per_layer_projection_norm.scale"))
    for i in range(cfg.num_layers):
        p, h = f"layers.{i}.", f"layers.{i}."
        w = get(p + "attn.q_einsum.w")  # [N, D, H] -> [N*H, D]
        put(h + "self_attn.q_proj.weight", jnp.transpose(w, (1, 0, 2)).reshape(w.shape[1], -1).T)
        if p + "attn.kv_einsum.w" in flat:  # [2, K, D, H] -> k, v [K*H, D]
            kv = get(p + "attn.kv_einsum.w")
            for j, proj in enumerate(("k_proj", "v_proj")):
                put(h + f"self_attn.{proj}.weight",
                    jnp.transpose(kv[j], (1, 0, 2)).reshape(kv.shape[2], -1).T)
        elif p + "attn.k_einsum.w" in flat:  # k_eq_v (global layers of some sizes): k only
            k = get(p + "attn.k_einsum.w")
            put(h + "self_attn.k_proj.weight", jnp.transpose(k, (1, 0, 2)).reshape(k.shape[1], -1).T)
        w = get(p + "attn.attn_vec_einsum.w")  # [N, H, D] -> [D, N*H]
        put(h + "self_attn.o_proj.weight", w.reshape(-1, w.shape[-1]).T)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            put(h + f"mlp.{proj}.weight", get(p + f"mlp.{proj}.kernel").T)
        for hf_name, nnx_name in (("input_layernorm", "pre_attention_norm"),
                                  ("post_attention_layernorm", "post_attention_norm"),
                                  ("pre_feedforward_layernorm", "pre_ffw_norm"),
                                  ("post_feedforward_layernorm", "post_ffw_norm"),
                                  ("self_attn.q_norm", "attn._query_norm"),
                                  ("self_attn.k_norm", "attn._key_norm"),
                                  ("post_per_layer_input_norm", "post_per_layer_input_norm")):
            if p + nnx_name + ".scale" in flat:
                put(h + hf_name + ".weight", get(p + nnx_name + ".scale"))
        if p + "skip_scale" in flat:
            put(h + "layer_scalar", get(p + "skip_scale"))
        for proj in ("per_layer_input_gate", "per_layer_projection"):
            if p + proj + ".w" in flat:
                put(h + proj + ".weight", get(p + proj + ".w").T)
    return out


def save_hf_checkpoint(model, base_model_dir: str, output_dir: str, dtype: str = "bfloat16") -> str:
    """The base checkpoint with every text weight Tunix trains replaced by `model`'s, as one
    model.safetensors (plus the base model's config / tokenizer / chat template)."""
    import shutil

    import ml_dtypes
    from safetensors import safe_open
    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)
    jnp_dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[dtype]
    np_dtype = {"bfloat16": ml_dtypes.bfloat16, "float32": np.float32}[dtype]
    new = _hf_tensors(model, jnp_dtype)
    tensors = {}
    for f in sorted(os.listdir(base_model_dir)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(base_model_dir, f), framework="flax") as st:
                for name in st.keys():
                    if name not in new:
                        tensors[name] = np.asarray(st.get_tensor(name))
    missing = [k for k in new if not _in_base(base_model_dir, k)]
    if missing:
        raise KeyError(f"exported names not in the base checkpoint: {missing[:5]}")
    tensors.update(new)
    tensors = {k: np.ascontiguousarray(v, dtype=np_dtype) for k, v in tensors.items()}
    tmp = os.path.join(output_dir, "model.safetensors.tmp")
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, os.path.join(output_dir, "model.safetensors"))
    for name in os.listdir(base_model_dir):
        if name.endswith((".json", ".txt", ".model", ".jinja")) and "index" not in name:
            shutil.copy(os.path.join(base_model_dir, name), os.path.join(output_dir, name))
    return output_dir


_BASE_KEYS = {}


def _in_base(base_model_dir: str, name: str) -> bool:
    if base_model_dir not in _BASE_KEYS:
        from safetensors import safe_open

        keys = set()
        for f in os.listdir(base_model_dir):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(base_model_dir, f), framework="flax") as st:
                    keys.update(st.keys())
        _BASE_KEYS[base_model_dir] = keys
    return name in _BASE_KEYS[base_model_dir]
