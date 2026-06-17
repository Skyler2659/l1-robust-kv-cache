"""Uniform sampling eviction — keep tokens at uniform intervals."""
from __future__ import annotations
import torch
from src.eviction.base import BaseEviction


class UniformEviction(BaseEviction):
    """Keep tokens at uniform intervals across the sequence."""
    name = "uniform"
    method_family = "geometry"

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        return None

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return self._ensure_budget(reserved, seq_len, budget, device, reserved=reserved)
        all_idx = torch.arange(seq_len, device=device)
        if reserved.numel() > 0:
            mask = ~torch.isin(all_idx, reserved)
            candidates = all_idx[mask]
        else:
            candidates = all_idx
        if candidates.numel() <= fill:
            chosen = candidates
        else:
            step = candidates.numel() / fill
            indices = torch.arange(fill, device=device, dtype=torch.float32) * step
            chosen = candidates[indices.long().clamp(max=candidates.numel() - 1)]
        return self._ensure_budget(
            torch.cat([reserved, chosen]), seq_len, budget, device, reserved=reserved)
