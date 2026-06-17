"""SnapKV baseline for benchmark KV cache eviction.

Li et al., 2024: "SnapKV: LLM Knows What You are Looking for Before Generation".

This benchmark runs token-by-token rather than as a prompt-prefill + generation
pipeline, so the implementation adapts SnapKV to post-forward eviction: it keeps
an observation window of recent queries from the model patches in ``shared_q``
and selects historical KV rows using the observation-window attention pattern.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from cache_baselines import _restore_cache_type, _to_legacy_cache
from rocketkv.rocket_core import snapkv_keep_indices


class SnapKVCache:
    """SnapKV observation-window attention baseline.

    The selected cache is capped at ``cache_size``. SnapKV itself preserves the
    recent observation window; ``sink_size`` is an optional benchmark stabilizer
    and defaults to zero for the standalone paper-style baseline.
    """

    def __init__(
        self,
        cache_size: int = 256,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
        window_size: int = 32,
        kernel_size: int = 63,
        sink_size: int = 0,
    ):
        self.cache_size = int(cache_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.sink_size = int(sink_size)
        self._observe_q: Dict[int, List[torch.Tensor]] = {}

    def _record_query(self, layer_idx: int):
        import shared_q

        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is None:
            return
        buf = self._observe_q.setdefault(layer_idx, [])
        buf.append(q_h.detach())
        cap = max(1, self.window_size)
        if len(buf) > cap:
            del buf[:-cap]

    def _observe_tensor(self, layer_idx: int, device, dtype) -> Optional[torch.Tensor]:
        buf = self._observe_q.get(layer_idx)
        if not buf:
            return None
        return torch.stack(buf, dim=0).to(device=device, dtype=dtype)

    def _reserved_indices(self, seq_len: int, budget: int, device) -> torch.Tensor:
        sink = min(max(0, self.sink_size), max(0, budget))
        if sink <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.arange(0, min(sink, seq_len), device=device)

    def _fallback_keep(self, seq_len: int, budget: int, device) -> torch.Tensor:
        budget = min(max(1, int(budget)), int(seq_len))
        reserved = self._reserved_indices(seq_len, budget, device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return reserved[:budget]
        recent = torch.arange(max(0, seq_len - fill), seq_len, device=device)
        if reserved.numel() > 0:
            recent = recent[~torch.isin(recent, reserved)]
        if recent.numel() > fill:
            recent = recent[-fill:]
        return torch.cat([reserved, recent]).unique(sorted=True)

    def _ensure_reserved(self, keep: torch.Tensor, seq_len: int, budget: int) -> torch.Tensor:
        budget = min(max(1, int(budget)), int(seq_len))
        reserved = self._reserved_indices(seq_len, budget, keep.device)
        keep = keep.unique(sorted=True)
        if reserved.numel() == 0:
            if keep.numel() <= budget:
                return keep
            return keep[-budget:]

        flexible = keep[~torch.isin(keep, reserved)]
        flex_budget = budget - int(reserved.numel())
        if flex_budget <= 0:
            return reserved[:budget]
        if flexible.numel() > flex_budget:
            flexible = flexible[-flex_budget:]
        return torch.cat([reserved, flexible]).unique(sorted=True)

    def _select_keep(self, layer_idx: int, k: torch.Tensor, seq_len: int, budget: int):
        budget = min(max(1, int(budget)), int(seq_len))
        if seq_len <= budget:
            return None

        q_obs = self._observe_tensor(layer_idx, k.device, k.dtype)
        if q_obs is None:
            return self._fallback_keep(seq_len, budget, k.device)

        obs_window = min(max(1, self.window_size), budget)
        keep = snapkv_keep_indices(
            k,
            q_obs,
            prompt_budget=budget,
            window_size=obs_window,
            kernel_size=self.kernel_size,
            k_seq_dim=self.k_seq_dim,
        )
        if keep is None:
            return self._fallback_keep(seq_len, budget, k.device)
        return self._ensure_reserved(keep.to(k.device), seq_len, budget)

    def _evict(self, past_key_values, budget: int, record_query: bool):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        items = []
        for layer_idx, (k, v) in enumerate(legacy_cache):
            if record_query:
                self._record_query(layer_idx)
            seq_len = k.size(self.k_seq_dim)
            keep = self._select_keep(layer_idx, k, seq_len, budget)
            if keep is None:
                items.append((k, v))
                continue
            items.append((
                torch.index_select(k, self.k_seq_dim, keep.to(k.device)),
                torch.index_select(v, self.v_seq_dim, keep.to(v.device)),
            ))
        return _restore_cache_type(original_cache, tuple(items))

    def __call__(self, past_key_values):
        return self._evict(past_key_values, self.cache_size, record_query=True)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        legacy_cache, _ = _to_legacy_cache(past_key_values)
        seq_len = legacy_cache[0][0].size(self.k_seq_dim)
        budget = max(1, self.cache_size - max(0, int(num_coming)))
        if seq_len <= budget:
            return past_key_values
        return self._evict(past_key_values, budget, record_query=False)

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = legacy_cache[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        keep = torch.cat([
            torch.arange(0, start, device=legacy_cache[0][0].device),
            torch.arange(end, seq_len, device=legacy_cache[0][0].device),
        ])
        items = []
        for k, v in legacy_cache:
            items.append((
                torch.index_select(k, self.k_seq_dim, keep.to(k.device)),
                torch.index_select(v, self.v_seq_dim, keep.to(v.device)),
            ))
        return _restore_cache_type(original_cache, tuple(items))
