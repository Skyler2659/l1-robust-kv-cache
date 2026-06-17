"""KV cache format utilities — DynamicCache ↔ legacy conversion, slicing, gathering.

Handles HF transformers 4.45+, 4.51+ DynamicCache and legacy tuple formats.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch


def _is_dynamic_cache(obj: Any) -> bool:
    return obj is not None and not isinstance(obj, (list, tuple))


def to_legacy_cache(
    past_key_values: Any,
) -> Tuple[Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]], Any]:
    """Convert any HF cache format to legacy ``((k, v), ...)`` tuple.

    Returns ``(legacy_cache, original_cache)`` where *original_cache* is used
    to reconstruct the output in ``back_to_original``.
    """
    if past_key_values is None:
        return None, None
    if isinstance(past_key_values, (list, tuple)):
        return tuple(past_key_values), None
    dc = past_key_values
    # transformers >= 4.51: .layers list
    lyrs = getattr(dc, "layers", None)
    if isinstance(lyrs, (list, tuple)) and len(lyrs) > 0:
        return tuple((lyr.keys, lyr.values) for lyr in lyrs), dc
    # transformers 4.45-4.50: .key_cache / .value_cache
    kc = getattr(dc, "key_cache", None)
    vc = getattr(dc, "value_cache", None)
    if isinstance(kc, list) and isinstance(vc, list):
        return tuple((kc[i], vc[i]) for i in range(len(kc))), dc
    # Fallback: to_legacy_cache method
    if hasattr(dc, "to_legacy_cache"):
        try:
            return dc.to_legacy_cache(), dc
        except Exception:
            pass
    return past_key_values, None  # type: ignore[return-value]


def back_to_original(
    original: Any,
    items: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Any:
    """Reconstruct the original cache format from legacy items."""
    if original is None:
        return tuple(items)
    if isinstance(original, tuple):
        return tuple(items)
    dc = original
    # 4.51+: rebuild via .layers
    if hasattr(dc, "layers"):
        for i, (k, v) in enumerate(items):
            if i < len(dc.layers):
                dc.layers[i].keys = k
                dc.layers[i].values = v
            else:
                dc.update(k, v, i)
        return dc
    # 4.45-4.50: key_cache / value_cache
    if hasattr(dc, "key_cache"):
        new_cache = type(dc)()
        new_cache.key_cache = [k for k, v in items]
        new_cache.value_cache = [v for k, v in items]
        return new_cache
    if hasattr(type(dc), "from_legacy_cache"):
        try:
            return type(dc).from_legacy_cache(tuple(items))
        except Exception:
            pass
    return tuple(items)


def get_kv_seq_len(tensor: torch.Tensor, seq_dim: int) -> int:
    """Get sequence length from a K or V tensor."""
    return int(tensor.size(seq_dim))


def slice_by_dim(
    x: torch.Tensor, dim: int, start: int, end: int
) -> torch.Tensor:
    """Slice tensor along *dim* from *start* to *end*."""
    idx = [slice(None)] * x.dim()
    idx[dim] = slice(start, end)
    return x[tuple(idx)]


def gather_by_dim(
    x: torch.Tensor, dim: int, indices: torch.Tensor
) -> torch.Tensor:
    """Gather tensor along *dim* using *indices*.

    KV eviction must index the sequence dimension only. This is especially
    important for GQA/MQA models where the key/value head count can differ from
    query head count. For RoPE models this keeps the already-rotated cached keys
    in their retained order; it does not remap positions or make removed tokens
    appear contiguous again.
    """
    return torch.index_select(x, dim, indices)


def mean_heads(x: torch.Tensor, seq_dim: int = 2) -> torch.Tensor:
    """Average over head dimension for ``[B, H, S, D]`` layout.

    Returns ``[S, D]``.
    """
    if x.dim() == 4 and seq_dim == 2:
        return x[0].mean(dim=0)  # [H, S, D] -> mean over H -> [S, D]
    if x.dim() == 4 and seq_dim == 3:
        return x[0].mean(dim=0).transpose(0, 1)  # [H, D, S] -> [S, D]
    if x.dim() == 3 and seq_dim == 1:
        return x.mean(dim=0)  # [H, S, D] -> [S, D]
    return x


def infer_kv_dims(model_type: str) -> Tuple[int, int]:
    """Infer (k_seq_dim, v_seq_dim) from model type string."""
    mt = (model_type or "").lower()
    if "llama" in mt or "gpt_neox" in mt or "qwen2" in mt or "mistral" in mt:
        return 2, 2
    if "mpt" in mt:
        return 3, 2
    if "falcon" in mt:
        return 1, 1
    return 2, 2
