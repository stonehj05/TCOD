"""NaN handling for prompt-scored logprobs on TPU.

vLLM-TPU sometimes cannot score a prompt token (its entry at a chunked-prefill boundary is <pad>), and
vllm_model.logprobs returns NaN there. Workflows that score in the explorer use the other model's logprob of
the same token instead (zero advantage on that token), as the deferred teacher gate does.
"""

import torch


def fill_unscored(lp, ref) -> torch.Tensor:
    """`lp` with its NaN entries taken from `ref` (same length); entries NaN in both become 0."""
    lp = torch.as_tensor(lp, dtype=torch.float32)
    nan = torch.isnan(lp)
    if not nan.any():
        return lp
    if ref is not None:
        lp = torch.where(nan, torch.as_tensor(ref, dtype=torch.float32), lp)
    return torch.nan_to_num(lp, nan=0.0)
