"""RocketKV KV cache eviction for the benchmark harness (Behnam et al., ICML 2025).

Two-stage compression adapted to post-forward eviction:
  1) SnapKV — coarse permanent eviction using an observation-window attention pattern.
  2) HSA   — fine dynamic top-k selection via head + sequence dimension reductions.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from rocketkv.rocket_core import (
    get_compression_params,
    hsa_keep_indices,
    snapkv_keep_indices,
)


def _is_dynamic_cache(pkv):
    return pkv is not None and not isinstance(pkv, (list, tuple))


def _to_legacy(pkv):
    if pkv is None:
        return None
    if _is_dynamic_cache(pkv):
        lyrs = getattr(pkv, "layers", None)
        if isinstance(lyrs, (list, tuple)) and len(lyrs) > 0:
            return tuple((lyr.keys, lyr.values) for lyr in lyrs)
        kc = getattr(pkv, "key_cache", None)
        vc = getattr(pkv, "value_cache", None)
        if isinstance(kc, list) and isinstance(vc, list):
            return tuple((kc[i], vc[i]) for i in range(len(kc)))
        if hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()
    return pkv


def _back_to_original(original, items):
    if original is None:
        return None
    if _is_dynamic_cache(original):
        if hasattr(original, "layers"):
            for i, (k, v) in enumerate(items):
                if i < len(original.layers):
                    original.layers[i].keys = k
                    original.layers[i].values = v
                else:
                    original.update(k, v, i)
            return original
        if hasattr(original, "key_cache"):
            new_cache = type(original)()
            new_cache.key_cache = [k for k, v in items]
            new_cache.value_cache = [v for k, v in items]
            return new_cache
        if hasattr(type(original), "from_legacy_cache"):
            return type(original).from_legacy_cache(items)
    if isinstance(original, tuple):
        return tuple(items)
    return items


class RocketKVCache:
    """RocketKV two-stage KV cache compression."""

    def __init__(
        self,
        cache_size: int = 256,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
        window_size: int = 32,
        kernel_size: int = 63,
        sink_size: int = 0,
        recent_size: int = 0,
    ):
        self.token_budget = int(cache_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.sink_size = int(sink_size)
        self.recent_size = int(recent_size)
        self._observe_q: Dict[int, List[torch.Tensor]] = {}
        self._snap_applied: Dict[int, bool] = {}
        self._steps = 0

    def _head_dim(self, layer_k: torch.Tensor) -> int:
        return int(layer_k.shape[-1])

    def _record_query(self, layer_idx: int, layer_k: torch.Tensor):
        import shared_q

        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is None:
            return
        buf = self._observe_q.setdefault(layer_idx, [])
        buf.append(q_h.detach())
        cap = max(self.window_size, 1)
        if len(buf) > cap:
            del buf[:-cap]

    def _observe_tensor(self, layer_idx: int, device, dtype) -> Optional[torch.Tensor]:
        buf = self._observe_q.get(layer_idx)
        if not buf:
            return None
        q = torch.stack(buf, dim=0).to(device=device, dtype=dtype)
        if q.dim() == 3:
            q = q.mean(dim=1)
        return q

    def _mean_key_rows(self, layer_k: torch.Tensor) -> Optional[torch.Tensor]:
        if layer_k.dim() == 4 and self.k_seq_dim == 2:
            return layer_k[0].mean(dim=0)
        if layer_k.dim() == 4 and self.k_seq_dim == 3:
            return layer_k[0].mean(dim=0).transpose(0, 1)
        if layer_k.dim() == 3 and self.k_seq_dim == 1:
            return layer_k.mean(dim=0)
        return None

    def _reserved_indices(self, seq_len: int, budget: int, device) -> torch.Tensor:
        budget = min(max(1, int(budget)), int(seq_len))
        sink = min(max(0, self.sink_size), budget)
        recent = min(max(0, self.recent_size), max(0, budget - sink))
        parts = []
        if sink > 0:
            parts.append(torch.arange(0, sink, device=device))
        if recent > 0:
            parts.append(torch.arange(seq_len - recent, seq_len, device=device))
        if not parts:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(parts).unique(sorted=True)

    def _fallback_keep(self, seq_len: int, budget: int, device) -> torch.Tensor:
        budget = min(max(1, int(budget)), int(seq_len))
        reserved = self._reserved_indices(seq_len, budget, device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return reserved[:budget]
        fill_idx = torch.arange(max(0, seq_len - budget), seq_len, device=device)
        if reserved.numel() > 0:
            fill_idx = fill_idx[~torch.isin(fill_idx, reserved)]
        if fill_idx.numel() > fill:
            fill_idx = fill_idx[-fill:]
        return torch.cat([reserved, fill_idx]).unique(sorted=True)

    def _ensure_reserved(self, keep: torch.Tensor, seq_len: int, budget: int) -> torch.Tensor:
        budget = min(max(1, int(budget)), int(seq_len))
        reserved = self._reserved_indices(seq_len, budget, keep.device)
        if reserved.numel() == 0:
            return keep.sort().values[-budget:] if keep.numel() > budget else keep.sort().values

        keep = keep.unique(sorted=True)
        flexible = keep[~torch.isin(keep, reserved)]
        flex_budget = budget - int(reserved.numel())
        if flex_budget <= 0:
            return reserved[:budget]
        if flexible.numel() > flex_budget:
            flexible = flexible[-flex_budget:]
        return torch.cat([reserved, flexible]).unique(sorted=True)

    def _compress_layer(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
    ):
        head_dim = self._head_dim(k)
        capacity, prompt_budget, chunk_size, r, k_hsa = get_compression_params(
            self.token_budget, seq_len, head_dim
        )
        final_budget = min(self.token_budget, seq_len)

        keep = torch.arange(seq_len, device=k.device)

        if seq_len > capacity and not self._snap_applied.get(layer_idx, False):
            q_obs = self._observe_tensor(layer_idx, k.device, k.dtype)
            if q_obs is not None:
                snap_idx = snapkv_keep_indices(
                    k,
                    q_obs,
                    prompt_budget,
                    self.window_size,
                    self.kernel_size,
                    self.k_seq_dim,
                )
                if snap_idx is not None:
                    keep = snap_idx
                    self._snap_applied[layer_idx] = True
            if keep.numel() > capacity:
                keep = self._fallback_keep(seq_len, capacity, k.device)
            keep = self._ensure_reserved(keep, seq_len, capacity)

        if keep.numel() > final_budget:
            import shared_q

            k_rows = self._mean_key_rows(k)
            q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
            if k_rows is not None and q_h is not None:
                rows = k_rows.index_select(0, keep.to(k_rows.device))
                hsa_idx = hsa_keep_indices(
                    q_h.mean(dim=0) if q_h.dim() > 1 else q_h,
                    rows,
                    final_budget,
                    chunk_size,
                    r,
                )
                if hsa_idx is not None:
                    keep = keep.index_select(0, hsa_idx.to(keep.device))
            if keep.numel() > final_budget:
                keep = self._fallback_keep(seq_len, final_budget, k.device)
            keep = self._ensure_reserved(keep, seq_len, final_budget)

        if keep.numel() == seq_len:
            return k, v
        keep_k = keep.to(k.device)
        keep_v = keep.to(v.device)
        return (
            torch.index_select(k, self.k_seq_dim, keep_k),
            torch.index_select(v, self.v_seq_dim, keep_v),
        )

    def _evict(self, past_key_values, record_query: bool):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        self._steps += 1
        items = []
        for layer_idx, (k, v) in enumerate(pkv):
            if record_query:
                self._record_query(layer_idx, k)
            seq_len = k.size(self.k_seq_dim)
            if seq_len <= self.token_budget:
                items.append((k, v))
                continue
            nk, nv = self._compress_layer(layer_idx, k, v, seq_len)
            items.append((nk, nv))
        return _back_to_original(past_key_values, items)

    def __call__(self, past_key_values):
        return self._evict(past_key_values, record_query=True)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        budget = max(1, self.token_budget - int(num_coming))
        if seq_len <= budget:
            return past_key_values
        old = self.token_budget
        try:
            self.token_budget = budget
            return self._evict(past_key_values, record_query=False)
        finally:
            self.token_budget = old

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        keep = torch.cat([
            torch.arange(0, start, device=pkv[0][0].device),
            torch.arange(end, seq_len, device=pkv[0][0].device),
        ])
        items = []
        for k, v in pkv:
            keep_k = keep.to(k.device)
            keep_v = keep.to(v.device)
            items.append((
                torch.index_select(k, self.k_seq_dim, keep_k),
                torch.index_select(v, self.v_seq_dim, keep_v),
            ))
        return _back_to_original(past_key_values, items)
