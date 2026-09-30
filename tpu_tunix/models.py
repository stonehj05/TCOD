"""Load Hugging Face checkpoints (hub id or local dir) as Tunix NNX models."""

import dataclasses
import os

import jax
import jax.numpy as jnp
import transformers
from huggingface_hub import snapshot_download
from tunix.models import automodel, naming


def resolve_path(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    return snapshot_download(
        model_id_or_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )


def load_model(model_id_or_path: str, mesh: jax.sharding.Mesh, model_name: str | None = None,
               dtype=jnp.bfloat16, **config_overrides):
    """Returns (nnx model, tokenizer, local path).

    `model_name` is Tunix's config name (e.g. "qwen2.5-1.5b-instruct"). It is
    inferred from the hub id when omitted; pass it explicitly for local dirs
    whose basename doesn't match (e.g. a fine-tuned teacher checkpoint).
    `config_overrides` replace ModelConfig fields (e.g. use_flash_attention).
    """
    path = resolve_path(model_id_or_path)
    if model_name is None:
        model_name = naming.ModelNaming(model_id=model_id_or_path.rstrip("/")).model_name
    config = automodel.call_model_config(model_name)
    if config_overrides:
        config = dataclasses.replace(config, **config_overrides)
    model = automodel.create_model_from_safe_tensors(model_name, path, config, mesh, dtype=dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    return model, tokenizer, path
