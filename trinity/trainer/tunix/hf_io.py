"""Hugging Face checkpoint <-> Tunix NNX model (Qwen2/Qwen3 dense).

Loading uses Tunix's own safetensors loader. Saving inverts its key/layout
mapping (tunix/models/qwen3/params.py): Tunix stores linear weights as [in, out]
(attention projections split per head), HF as [out, in].
"""

import dataclasses
import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def resolve_model_dir(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id_or_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )


def tunix_model_name(model_dir: str) -> str:
    """Tunix config name (e.g. 'qwen3-1.7b') from a HF config.json."""
    with open(os.path.join(model_dir, "config.json")) as f:
        hf = json.load(f)
    arch = hf["architectures"][0]
    if arch not in ("Qwen3ForCausalLM", "Qwen2ForCausalLM"):
        raise NotImplementedError(f"tunix trainer supports Qwen2/Qwen3 dense models, got {arch}")
    family = "qwen3" if arch.startswith("Qwen3") else "qwen2.5"
    size = hf["num_hidden_layers"], hf["hidden_size"]
    # Tunix names models by parameter count; map from the HF layer/width shape.
    known = {
        ("qwen3", 28, 1024): "qwen3-0.6b",
        ("qwen3", 28, 2048): "qwen3-1.7b",
        ("qwen3", 36, 2560): "qwen3-4b",
        ("qwen3", 36, 4096): "qwen3-8b",
        ("qwen3", 40, 5120): "qwen3-14b",
    }
    name = known.get((family, *size))
    if name is None:
        raise NotImplementedError(f"no Tunix config for {family} with (layers, hidden)={size}")
    return name


def load_model(model_dir: str, mesh, dtype=jnp.float32, **config_overrides):
    from tunix.models import automodel

    name = tunix_model_name(model_dir)
    config = automodel.call_model_config(name)
    if config_overrides:
        config = dataclasses.replace(config, **config_overrides)
    return automodel.create_model_from_safe_tensors(name, model_dir, config, mesh, dtype=dtype)


def _hf_tensors(model, dtype=None) -> dict:
    """NNX params -> {hf_name: np.ndarray} in HF layout, gathered to host.

    Casting (`dtype`, e.g. jnp.bfloat16) and the HF re-layout (transposes, head reshapes)
    run on device; each tensor is copied to host right after, so device memory stays flat.
    Doing these on the host single-threaded dominated the per-step sync time.
    """
    cfg = model.config
    out = {}
    params = nnx.state(model, nnx.Param)
    flat = {".".join(str(k) for k in path): v for path, v in nnx.to_flat_state(params)}

    def get(key):  # device array
        v = flat[key]
        v = getattr(v, "value", v)
        return v.astype(dtype) if dtype is not None else v

    def put(name, x):
        out[name] = np.asarray(jax.device_get(x))

    put("model.embed_tokens.weight", get("embedder.input_embedding"))
    put("model.norm.weight", get("final_norm.w"))
    if cfg.use_tied_embedding:
        # Written explicitly so the explorer's separate lm_head copy is updated too.
        out["lm_head.weight"] = out["model.embed_tokens.weight"]
    else:
        put("lm_head.weight", get("lm_head.w").T)
    layer_ids = sorted({int(m.group(1)) for k in flat if (m := re.match(r"layers\.(\d+)\.", k))})
    for i in layer_ids:
        p, h = f"layers.{i}.", f"model.layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj"):  # [D, N, H] -> [N*H, D]
            w = get(p + f"attn.{proj}.w")
            put(h + f"self_attn.{proj}.weight", w.reshape(w.shape[0], -1).T)
        w = get(p + "attn.o_proj.w")  # [N, H, D] -> [D, N*H]
        put(h + "self_attn.o_proj.weight", w.reshape(-1, w.shape[-1]).T)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            put(h + f"mlp.{proj}.weight", get(p + f"mlp.{proj}.kernel").T)
        put(h + "self_attn.q_norm.weight", get(p + "attn.q_norm.w"))
        put(h + "self_attn.k_norm.weight", get(p + "attn.k_norm.w"))
        put(h + "input_layernorm.weight", get(p + "input_layernorm.w"))
        put(h + "post_attention_layernorm.weight", get(p + "post_attention_layernorm.w"))
    return out


def save_hf_checkpoint(model, base_model_dir: str, output_dir: str, dtype: str = "bfloat16") -> str:
    """Write `model` as a HF safetensors checkpoint (plus the base model's config/tokenizer)."""
    import shutil

    import ml_dtypes
    from safetensors.numpy import save_file

    os.makedirs(output_dir, exist_ok=True)
    jnp_dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[dtype]
    np_dtype = {"bfloat16": ml_dtypes.bfloat16, "float32": np.float32}[dtype]
    tensors = {k: np.ascontiguousarray(v, dtype=np_dtype) for k, v in _hf_tensors(model, jnp_dtype).items()}
    tmp = os.path.join(output_dir, "model.safetensors.tmp")
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, os.path.join(output_dir, "model.safetensors"))
    for name in os.listdir(base_model_dir):
        if name.endswith((".json", ".txt", ".model", ".jinja")) and "index" not in name:
            shutil.copy(os.path.join(base_model_dir, name), os.path.join(output_dir, name))
    return output_dir
