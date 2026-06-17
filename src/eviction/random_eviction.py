"""Random eviction baseline — seed-controlled random selection."""
from __future__ import annotations
import torch
from src.eviction.base import BaseEviction


class RandomEviction(BaseEviction):
    """Keep ``budget`` random indices (with optional sink+recent reserved)."""
    name = "random"
    method_family = "random"
    supports_backends = ("torch", "mlx")

    def __init__(self, seed=0, **kwargs):
        super().__init__(**kwargs)
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

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
            perm = torch.randperm(candidates.numel(), generator=self._rng)[:fill]
            chosen = candidates[perm]
        return self._ensure_budget(
            torch.cat([reserved, chosen]), seq_len, budget, device, reserved=reserved)

    def reset(self):
        super().reset()
        self._rng.manual_seed(self._rng.initial_seed())
