"""Score normalization utilities shared by eviction methods and analysis."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch


SUPPORTED_NORMALIZATIONS = ("minmax", "zscore", "rank", "softmax", "none")


def normalize_scores(
    scores: Optional[torch.Tensor],
    method: str = "rank",
    dim: int = -1,
) -> Optional[torch.Tensor]:
    """Normalize a score vector while preserving shape and device.

    Non-finite values are ignored for min/max and z-score statistics. They are
    replaced with zeros in the normalized output so downstream top-k selection
    never receives NaNs.
    """
    if scores is None:
        return None
    mode = str(method or "none").lower()
    if mode not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"Unknown score normalization {method!r}; expected one of "
            f"{SUPPORTED_NORMALIZATIONS}"
        )
    values = scores.float()
    if values.numel() == 0 or mode == "none":
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    finite = torch.isfinite(values)
    if not finite.any():
        return torch.zeros_like(values)

    if mode == "minmax":
        finite_values = values[finite]
        lo = finite_values.min()
        hi = finite_values.max()
        denom = (hi - lo).clamp_min(1e-8)
        out = (values - lo) / denom
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    if mode == "zscore":
        finite_values = values[finite]
        mean = finite_values.mean()
        std = finite_values.std(unbiased=False).clamp_min(1e-8)
        out = (values - mean) / std
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "softmax":
        masked = values.masked_fill(~finite, float("-inf"))
        out = torch.softmax(masked, dim=dim)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # rank: highest raw score gets normalized rank 1.0, lowest gets 0.0.
    flat = values.flatten()
    finite_flat = torch.isfinite(flat)
    out = torch.zeros_like(flat)
    finite_idx = finite_flat.nonzero(as_tuple=True)[0]
    if finite_idx.numel() == 1:
        out[finite_idx] = 1.0
    elif finite_idx.numel() > 1:
        finite_scores = flat[finite_idx]
        order = torch.argsort(finite_scores, descending=False)
        ranks = torch.zeros_like(finite_scores)
        ranks[order] = torch.arange(
            finite_scores.numel(),
            device=finite_scores.device,
            dtype=finite_scores.dtype,
        )
        ranks = ranks / max(1, finite_scores.numel() - 1)
        out[finite_idx] = ranks
    return out.reshape_as(values)


def score_stats(scores: Optional[torch.Tensor], topk: int = 5) -> Dict[str, Any]:
    """Return compact numeric diagnostics for a score tensor."""
    if scores is None:
        return {}
    values = scores.detach().float().flatten().cpu()
    if values.numel() == 0:
        return {"numel": 0}
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"numel": int(values.numel()), "all_non_finite": True}
    k = min(int(topk), int(finite.numel()))
    return {
        "numel": int(values.numel()),
        "finite_numel": int(finite.numel()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
        "top_values": [float(x) for x in torch.topk(finite, k).values.tolist()],
    }


def merge_score_stats(
    scores_by_layer: Dict[int, torch.Tensor],
    normalization: str = "none",
    topk: int = 5,
) -> Dict[str, Any]:
    """Summarize raw and normalized score dictionaries."""
    tensors = [s.flatten().float() for s in scores_by_layer.values() if s is not None and s.numel()]
    if not tensors:
        return {}
    raw = torch.cat(tensors)
    norm = normalize_scores(raw, normalization)
    return {
        "score_normalization": normalization,
        "raw_score_stats": score_stats(raw, topk=topk),
        "normalized_score_stats": score_stats(norm, topk=topk),
        "top_score_values": score_stats(raw, topk=topk).get("top_values", []),
    }


def list_stats(values: Iterable[float], topk: int = 5) -> Dict[str, Any]:
    """Score stats for JSON-loaded score vectors."""
    vals = list(values)
    if not vals:
        return {"numel": 0}
    return score_stats(torch.tensor(vals, dtype=torch.float32), topk=topk)
