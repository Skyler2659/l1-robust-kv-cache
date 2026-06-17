"""Recency / Sliding Window eviction — keep only the most recent tokens."""
from __future__ import annotations
from typing import Optional
import torch
from src.eviction.base import BaseEviction


class FullKVCache(BaseEviction):
    """No eviction — keep full cache (upper bound baseline)."""
    name = "full"
    method_family = "recency"
    supports_backends = ("torch", "mlx")

    def __init__(self, **kwargs):
        kwargs.setdefault("cache_size", 10**9)
        super().__init__(**kwargs)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        return None

    def select_indices(self, scores, seq_len, budget, device):
        return torch.arange(seq_len, device=device)

    def __call__(self, past_key_values):
        return past_key_values

    def evict_for_space(self, past_key_values, num_coming):
        return past_key_values


class RecencyEviction(BaseEviction):
    """Sliding window: keep last ``cache_size`` tokens."""
    name = "recency"
    method_family = "recency"
    supports_backends = ("torch", "mlx")

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        return None

    def select_indices(self, scores, seq_len, budget, device):
        start = max(0, seq_len - budget)
        return self._ensure_budget(
            torch.arange(start, seq_len, device=device),
            seq_len,
            budget,
            device,
        )
