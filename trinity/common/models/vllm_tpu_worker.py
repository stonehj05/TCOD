"""vLLM-TPU worker extension: load updated weights into a running engine.

TPU counterpart of `vllm_worker.WorkerExtension` (which receives weights over
NCCL). Used with `sync_method: checkpoint`: the trainer saves a Hugging Face
safetensors checkpoint and the explorer calls `update_weight_from_checkpoint`
on every engine worker.

vLLM-TPU (tpu_inference) keeps weights in `model_runner.state`, laid out as the
HF weight transposed to [in, out] and, for attention projections, reshaped to
per-head form (q_proj -> [hidden, heads, head_dim], o_proj -> [heads, head_dim,
hidden]). Embeddings and norms keep the HF layout. The compiled model reads
`model_runner.state_leaves`, so that must be rebuilt after the update.
"""

import glob
import os

# HF weights stored untransposed in vLLM-TPU.
_NO_TRANSPOSE_SUFFIXES = ("embed_tokens.weight",)


def _state_key(hf_name: str) -> tuple:
    """'model.layers.3.mlp.down_proj.weight' -> ('model', 'layers', 3, 'mlp', 'down_proj', 'weight')."""
    return tuple(int(p) if p.isdigit() else p for p in hf_name.split("."))


def _path_entry(p):
    """A jax tree path entry (DictKey / SequenceKey / GetAttrKey) as a plain key."""
    for attr in ("key", "idx", "name"):
        if hasattr(p, attr):
            return getattr(p, attr)
    raise TypeError(f"unexpected tree path entry {p!r}")


def _to_vllm_layout(hf_name, array, target_shape):
    import numpy as np

    if array.ndim >= 2 and not hf_name.endswith(_NO_TRANSPOSE_SUFFIXES):
        array = array.T
    if tuple(array.shape) != tuple(target_shape):
        if int(np.prod(array.shape)) != int(np.prod(target_shape)):
            raise ValueError(f"{hf_name}: cannot map HF shape {array.shape} to {target_shape}")
        array = array.reshape(target_shape)
    return array


class TPUWorkerExtension:
    """Mixed into the vLLM-TPU worker via `worker_extension_cls`."""

    def update_weight_from_checkpoint(self, checkpoint_dir: str) -> int:
        """Replace the engine's weights with the safetensors files in `checkpoint_dir`.

        Returns the number of tensors updated. Tensors missing from the
        checkpoint keep their current values (e.g. a tied lm_head).
        """
        import jax
        from safetensors import safe_open

        runner = self.model_runner
        flat, treedef = jax.tree_util.tree_flatten_with_path(runner.state)
        index = {}
        for i, (path, _) in enumerate(flat):
            keys = tuple(_path_entry(p) for p in path)
            if keys and keys[-1] == "value":
                keys = keys[:-1]
            index[keys] = i

        leaves = [v for _, v in flat]
        updated = 0
        files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors in {checkpoint_dir}")
        for f in files:
            with safe_open(f, framework="flax") as st:  # numpy has no bfloat16
                for name in st.keys():
                    i = index.get(_state_key(name))
                    if i is None:
                        continue
                    old = leaves[i]
                    new = _to_vllm_layout(name, st.get_tensor(name), old.shape)
                    leaves[i] = jax.device_put(new.astype(old.dtype), old.sharding)
                    updated += 1
        runner.state = jax.tree_util.tree_unflatten(treedef, leaves)
        runner.state_leaves = tuple(jax.tree_util.tree_leaves(runner.state))
        return updated
