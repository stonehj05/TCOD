"""Token-aligned tagged-segment KL metrics shared by OPD workflows."""

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from trinity.common.experience import Experience

_ACTION_OPEN_TAG = "<action>"
_ACTION_CLOSE_TAG = "</action>"
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"
_SEGMENT_TAGS = (
    _ACTION_OPEN_TAG,
    _ACTION_CLOSE_TAG,
    _THINK_OPEN_TAG,
    _THINK_CLOSE_TAG,
)


def _contains_segment_tag(text: str, start: int, end: int) -> bool:
    """Return whether a candidate block contains reserved structural markup."""
    return any(text.find(tag, start, end) >= 0 for tag in _SEGMENT_TAGS)


def find_action_content_spans(response_text: str) -> Sequence[Tuple[int, int]]:
    """Return every complete, non-empty action content span in generation order."""
    spans = []
    search_start = 0
    while True:
        open_start = response_text.find(_ACTION_OPEN_TAG, search_start)
        if open_start < 0:
            break

        content_start = open_start + len(_ACTION_OPEN_TAG)
        content_end = response_text.find(_ACTION_CLOSE_TAG, content_start)
        if content_end < 0:
            break

        nested_open = response_text.find(_ACTION_OPEN_TAG, content_start, content_end)
        if nested_open >= 0:
            search_start = nested_open
            continue

        # Reject cross-tag nesting instead of attributing think markup/content
        # to the action. A complete inner think block remains independently
        # eligible for think KL.
        if _contains_segment_tag(response_text, content_start, content_end):
            search_start = content_end + len(_ACTION_CLOSE_TAG)
            continue

        search_start = content_end + len(_ACTION_CLOSE_TAG)
        while content_start < content_end and response_text[content_start].isspace():
            content_start += 1
        while content_end > content_start and response_text[content_end - 1].isspace():
            content_end -= 1
        if content_start < content_end:
            spans.append((content_start, content_end))
    return spans


def find_action_content_span(response_text: str) -> Optional[Tuple[int, int]]:
    """Return content from the first syntactic action block only.

    This deliberately does not fall back to a later action when the first block
    is empty, incomplete, or contains reserved structural markup.
    """
    open_start = response_text.find(_ACTION_OPEN_TAG)
    if open_start < 0:
        return None
    content_start = open_start + len(_ACTION_OPEN_TAG)
    content_end = response_text.find(_ACTION_CLOSE_TAG, content_start)
    if content_end < 0:
        return None
    if _contains_segment_tag(response_text, content_start, content_end):
        return None
    while content_start < content_end and response_text[content_start].isspace():
        content_start += 1
    while content_end > content_start and response_text[content_end - 1].isspace():
        content_end -= 1
    return (content_start, content_end) if content_start < content_end else None


def find_think_content_spans(response_text: str) -> Sequence[Tuple[int, int]]:
    """Return every complete, non-empty think content span in generation order."""
    spans = []
    search_start = 0
    while True:
        open_start = response_text.find(_THINK_OPEN_TAG, search_start)
        if open_start < 0:
            break

        content_start = open_start + len(_THINK_OPEN_TAG)
        content_end = response_text.find(_THINK_CLOSE_TAG, content_start)
        if content_end < 0:
            break

        nested_open = response_text.find(_THINK_OPEN_TAG, content_start, content_end)
        if nested_open >= 0:
            search_start = nested_open
            continue

        # Reject cross-tag nesting so action tokens cannot also be counted as
        # thought tokens. A complete inner action remains independently eligible.
        if _contains_segment_tag(response_text, content_start, content_end):
            search_start = content_end + len(_THINK_CLOSE_TAG)
            continue

        search_start = content_end + len(_THINK_CLOSE_TAG)
        while content_start < content_end and response_text[content_start].isspace():
            content_start += 1
        while content_end > content_start and response_text[content_end - 1].isspace():
            content_end -= 1
        if content_start < content_end:
            spans.append((content_start, content_end))
    return spans


def find_pre_action_think_content_spans(
    response_text: str,
) -> Sequence[Tuple[int, int]]:
    """Return complete think spans that precede the first complete action.

    Think blocks after the selected action cannot have conditioned that action,
    so they are deliberately excluded. If there is no complete non-empty action,
    no think span is eligible for the action-associated think KL metric.
    """
    action_span = find_action_content_span(response_text)
    if action_span is None:
        return []
    action_open = response_text.find(_ACTION_OPEN_TAG)
    return [
        span
        for span in find_think_content_spans(response_text)
        if span[1] <= action_open
    ]


def _build_token_mask_from_spans(
    response_text: str,
    response_token_ids: Sequence[int],
    tokenizer: Any,
    content_spans: Sequence[Tuple[int, int]],
) -> Optional[torch.Tensor]:
    """Map character spans to original tokens after an exact token-ID round trip."""
    if not content_spans:
        return None

    try:
        encoding = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = list(encoding["input_ids"])
        offsets = list(encoding["offset_mapping"])
    except (KeyError, NotImplementedError, TypeError, ValueError):
        return None

    generated_ids = list(response_token_ids)
    if (
        len(encoded_ids) != len(offsets)
        or generated_ids[: len(encoded_ids)] != encoded_ids
    ):
        return None

    trailing_ids = generated_ids[len(encoded_ids) :]
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if any(token_id not in special_ids for token_id in trailing_ids):
        return None

    mask = torch.zeros(len(generated_ids), dtype=torch.bool)
    for token_index, (token_start, token_end) in enumerate(offsets):
        if any(
            token_end > content_start and token_start < content_end
            for content_start, content_end in content_spans
        ):
            mask[token_index] = True
    return mask if mask.any() else None


def build_action_token_mask(
    response_text: str,
    response_token_ids: Sequence[int],
    tokenizer: Any,
) -> Optional[torch.Tensor]:
    """Map all complete action blocks to the original generated response tokens.

    Re-tokenization must reproduce the actual generated token IDs. If it does
    not, the sample is excluded instead of guessing a token alignment. Trailing
    generated special tokens such as EOS are allowed and excluded from the mask.
    A boundary token is included when any part overlaps the action content,
    because token-level KL cannot be divided into smaller character-level parts.
    """
    return _build_token_mask_from_spans(
        response_text,
        response_token_ids,
        tokenizer,
        find_action_content_spans(response_text),
    )


def build_think_token_mask(
    response_text: str,
    response_token_ids: Sequence[int],
    tokenizer: Any,
) -> Optional[torch.Tensor]:
    """Map every complete, non-empty think block to original response tokens."""
    return _build_token_mask_from_spans(
        response_text,
        response_token_ids,
        tokenizer,
        find_think_content_spans(response_text),
    )


@lru_cache(maxsize=None)
def get_segment_tokenizer(model_path: str):
    """Load one cached fast tokenizer per student model path and runner process."""
    if not model_path:
        raise ValueError("Student model_path is required to compute tagged-segment KL.")
    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )


def calculate_action_kl_metrics(
    turn_responses: Sequence[Experience],
    tokenizer: Any,
) -> Dict[str, float]:
    """Calculate token-normalized KL over all action blocks in a trajectory.

    Every complete action block is counted regardless of whether an environment
    parses, validates, or executes it. This measures divergence over all text the
    student explicitly marked as an action, including extra action blocks.
    """
    action_kl_sum = 0.0
    action_token_count = 0
    action_block_count = 0
    action_format_valid_count = 0
    action_token_aligned_count = 0

    for response in turn_responses:
        response_text = response.response_text or ""
        action_spans = find_action_content_spans(response_text)
        if action_spans:
            action_format_valid_count += 1
            action_block_count += len(action_spans)

        student_logprobs = response.logprobs
        teacher_logprobs = response.teacher_logprobs
        if (
            student_logprobs is None
            or teacher_logprobs is None
            or len(student_logprobs) == 0
        ):
            continue
        if len(student_logprobs) != len(teacher_logprobs):
            continue

        response_token_ids = response.tokens[-len(student_logprobs) :].tolist()
        action_mask = build_action_token_mask(
            response_text,
            response_token_ids,
            tokenizer,
        )
        if action_mask is None or len(action_mask) != len(student_logprobs):
            continue

        action_token_aligned_count += 1
        kl_per_token = student_logprobs - teacher_logprobs
        action_mask = action_mask.to(device=kl_per_token.device)
        action_kl_sum += kl_per_token[action_mask].sum().item()
        action_token_count += int(action_mask.sum().item())

    turn_count = len(turn_responses)
    metrics = {
        "action_token_count": float(action_token_count),
        "action_block_count": float(action_block_count),
        "action_format_valid_rate": (
            action_format_valid_count / turn_count if turn_count > 0 else 0.0
        ),
        "action_token_alignment_rate": (
            action_token_aligned_count / action_format_valid_count
            if action_format_valid_count > 0
            else 0.0
        ),
    }
    if action_token_count > 0:
        metrics["action_kl_divergence"] = action_kl_sum / action_token_count
    return metrics


def calculate_think_kl_metrics(
    turn_responses: Sequence[Experience],
    tokenizer: Any,
) -> Dict[str, float]:
    """Calculate token-normalized KL over all complete think blocks.

    Unlike action KL, think KL does not require a complete action. This avoids
    excluding useful reasoning from trajectories whose action was truncated or
    malformed. Whether a think block precedes a valid action remains available
    as a separate diagnostic and does not filter the primary metric.
    """
    think_kl_sum = 0.0
    think_token_count = 0
    think_block_count = 0
    think_format_valid_count = 0
    think_before_action_count = 0
    think_token_aligned_count = 0

    for response in turn_responses:
        response_text = response.response_text or ""
        think_spans = find_think_content_spans(response_text)
        pre_action_think_spans = find_pre_action_think_content_spans(response_text)
        if think_spans:
            think_format_valid_count += 1
            think_block_count += len(think_spans)
        if pre_action_think_spans:
            think_before_action_count += 1

        student_logprobs = response.logprobs
        teacher_logprobs = response.teacher_logprobs
        if (
            student_logprobs is None
            or teacher_logprobs is None
            or len(student_logprobs) == 0
        ):
            continue
        if len(student_logprobs) != len(teacher_logprobs):
            continue

        response_token_ids = response.tokens[-len(student_logprobs) :].tolist()
        think_mask = _build_token_mask_from_spans(
            response_text,
            response_token_ids,
            tokenizer,
            think_spans,
        )
        if think_mask is None or len(think_mask) != len(student_logprobs):
            continue

        think_token_aligned_count += 1
        kl_per_token = student_logprobs - teacher_logprobs
        think_mask = think_mask.to(device=kl_per_token.device)
        think_kl_sum += kl_per_token[think_mask].sum().item()
        think_token_count += int(think_mask.sum().item())

    turn_count = len(turn_responses)
    metrics = {
        "think_token_count": float(think_token_count),
        "think_block_count": float(think_block_count),
        "think_format_valid_rate": (
            think_format_valid_count / turn_count if turn_count > 0 else 0.0
        ),
        "think_before_action_rate": (
            think_before_action_count / think_format_valid_count
            if think_format_valid_count > 0
            else 0.0
        ),
        "think_token_alignment_rate": (
            think_token_aligned_count / think_format_valid_count
            if think_format_valid_count > 0
            else 0.0
        ),
    }
    if think_token_count > 0:
        metrics["think_kl_divergence"] = think_kl_sum / think_token_count
    return metrics


def _calculate_selected_segment_kl_metric(
    turn_responses: Sequence[Experience],
    tokenizer: Any,
    span_selector: Callable[[str], Sequence[Tuple[int, int]]],
    metric_name: str,
) -> Dict[str, float]:
    """Calculate one token-normalized KL metric for selected tagged spans."""
    kl_sum = 0.0
    token_count = 0
    for response in turn_responses:
        student_logprobs = response.logprobs
        teacher_logprobs = response.teacher_logprobs
        if (
            student_logprobs is None
            or teacher_logprobs is None
            or len(student_logprobs) == 0
        ):
            continue
        if len(student_logprobs) != len(teacher_logprobs):
            continue

        response_text = response.response_text or ""
        content_spans = span_selector(response_text)
        response_token_ids = response.tokens[-len(student_logprobs) :].tolist()
        segment_mask = _build_token_mask_from_spans(
            response_text,
            response_token_ids,
            tokenizer,
            content_spans,
        )
        if segment_mask is None or len(segment_mask) != len(student_logprobs):
            continue

        kl_per_token = student_logprobs - teacher_logprobs
        segment_mask = segment_mask.to(device=kl_per_token.device)
        kl_sum += kl_per_token[segment_mask].sum().item()
        token_count += int(segment_mask.sum().item())

    return {metric_name: kl_sum / token_count} if token_count > 0 else {}


def calculate_pre_action_think_kl_metrics(
    turn_responses: Sequence[Experience],
    tokenizer: Any,
) -> Dict[str, float]:
    """Calculate KL over think blocks before the first complete non-empty action."""
    return _calculate_selected_segment_kl_metric(
        turn_responses,
        tokenizer,
        find_pre_action_think_content_spans,
        "pre_action_think_kl_divergence",
    )


def calculate_first_action_kl_metrics(
    turn_responses: Sequence[Experience],
    tokenizer: Any,
) -> Dict[str, float]:
    """Calculate KL over the first syntactic action block in each turn."""

    def select_first_action(response_text: str) -> Sequence[Tuple[int, int]]:
        span = find_action_content_span(response_text)
        return [span] if span is not None else []

    return _calculate_selected_segment_kl_metric(
        turn_responses,
        tokenizer,
        select_first_action,
        "first_action_kl_divergence",
    )
