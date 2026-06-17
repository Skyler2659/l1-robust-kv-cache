"""Outlier and rarity based KV eviction baselines."""
from __future__ import annotations

import torch

from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


class _OutlierBase(BaseEviction):
    method_family = "geometry"
    requires_scores = True
    score_source = "value"
    approximate = True

    def __init__(self, score_source: str = "v", **kwargs):
        super().__init__(**kwargs)
        self.score_source = str(score_source or "v").lower()

    def _rows(self, layer_k, layer_v):
        v_rows = mean_heads(layer_v, self.v_seq_dim)
        if self.score_source in ("v", "value") or v_rows is None:
            return v_rows
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if self.score_source in ("k", "key"):
            return k_rows
        if k_rows is None or k_rows.shape[0] != v_rows.shape[0]:
            return v_rows
        return torch.cat([k_rows.float(), v_rows.float()], dim=-1)

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        if scores is None:
            return self._fill_budget(reserved, seq_len, budget, device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return self._ensure_budget(reserved, seq_len, budget, device, reserved=reserved)
        masked = scores[:seq_len].clone().to(device)
        if reserved.numel() > 0:
            masked[reserved] = -float("inf")
        valid = torch.isfinite(masked)
        if not valid.any():
            return self._fill_budget(reserved, seq_len, budget, device)
        idx = torch.topk(masked, min(fill, int(valid.sum().item()))).indices
        return self._ensure_budget(
            torch.cat([reserved, idx]),
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )


class MahalanobisDistanceEviction(_OutlierBase):
    """Select tokens far from the representation distribution center."""

    name = "mahalanobis_distance"

    def __init__(self, covariance_mode: str = "diagonal", **kwargs):
        super().__init__(**kwargs)
        self.covariance_mode = str(covariance_mode or "diagonal").lower()

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._rows(layer_k, layer_v)
        if rows is None:
            return None
        rows = rows.float()
        centered = rows - rows.mean(dim=0, keepdim=True)
        if self.covariance_mode == "isotropic":
            scale = centered.pow(2).mean().sqrt().clamp_min(1e-6)
            return torch.norm(centered / scale, p=2, dim=1)
        var = centered.var(dim=0, unbiased=False).clamp_min(1e-6)
        return torch.sqrt((centered.pow(2) / var).sum(dim=1))


class ZScoreOutlierEviction(_OutlierBase):
    """Score tokens by aggregate absolute per-dimension z-score."""

    name = "zscore_outlier"

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._rows(layer_k, layer_v)
        if rows is None:
            return None
        rows = rows.float()
        mean = rows.mean(dim=0, keepdim=True)
        std = rows.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return torch.mean(torch.abs((rows - mean) / std), dim=1)


class RandomProjectionOutlierEviction(_OutlierBase):
    """Project rows to a lower dimension, then compute z-score outlierness."""

    name = "random_projection_outlier"

    def __init__(self, projection_dim: int = 64, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._rows(layer_k, layer_v)
        if rows is None:
            return None
        rows = rows.float()
        _, d = rows.shape
        generator = torch.Generator(device="cpu").manual_seed(self.seed + int(layer_idx))
        proj = torch.randn(d, self.projection_dim, generator=generator, dtype=torch.float32)
        proj = proj.to(rows.device) / max(self.projection_dim, 1) ** 0.5
        low = rows @ proj
        mean = low.mean(dim=0, keepdim=True)
        std = low.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return torch.mean(torch.abs((low - mean) / std), dim=1)
