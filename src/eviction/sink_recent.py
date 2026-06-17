"""Sink + Recent eviction (StreamingLLM-style)."""
from __future__ import annotations
import torch
from src.eviction.base import BaseEviction


class SinkRecentEviction(BaseEviction):
    """Keep first ``sink_size`` tokens + last ``recent_size`` tokens."""
    name = "sink_recent"
    method_family = "recency"
    supports_backends = ("torch", "mlx")

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        return None

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        sink = min(self.sink_size, budget)
        recent = min(self.recent_size, max(0, budget - sink))
        parts = []
        if sink > 0:
            parts.append(torch.arange(0, sink, device=device))
        if recent > 0:
            parts.append(torch.arange(seq_len - recent, seq_len, device=device))
        if not parts:
            return self._fill_budget(
                torch.empty(0, dtype=torch.long, device=device),
                seq_len,
                budget,
                device,
            )
        reserved = torch.cat(parts).unique(sorted=True)
        return self._fill_budget(reserved, seq_len, budget, device)
