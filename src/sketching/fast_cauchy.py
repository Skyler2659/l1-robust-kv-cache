"""Fast Cauchy transform for L1 subspace embedding."""
from __future__ import annotations
from typing import Optional
import torch


class FastCauchyEstimator:
    """Cauchy random projection for L1 leverage estimation.

    S_{ij} ~ Cauchy(0,1). The Cauchy distribution is 1-stable, so
    ||SAx||_1 ≈ ||Ax||_1 up to distortion.
    """

    def __init__(self, sketch_dim: int = 1024, seed: int = 0):
        self.sketch_dim = sketch_dim
        self.seed = seed
        self._proj: Optional[torch.Tensor] = None
        self._last_scores: Optional[torch.Tensor] = None

    def fit(self, rows: torch.Tensor) -> None:
        n, d = rows.shape
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        # Cauchy samples: ratio of two standard normals
        u = torch.randn(self.sketch_dim, d, generator=gen, dtype=torch.float32)
        v = torch.randn(self.sketch_dim, d, generator=gen, dtype=torch.float32).abs().clamp_min(1e-8)
        self._proj = (u / v).to(rows.device)

    def scores(self, rows: torch.Tensor, force_refit=False, **kw) -> torch.Tensor:
        rows_f = rows.float()
        n, d = rows_f.shape
        if self._proj is None or force_refit or self._proj.shape[1] != d:
            self.fit(rows_f)
        if self._proj is None:
            s = torch.norm(rows_f, p=1, dim=1)
            self._last_scores = s
            return s
        try:
            projected = self._proj @ rows_f.T  # [sketch, n]
            s = projected.abs().sum(dim=0).to(rows.dtype)
            self._last_scores = s
            return s
        except Exception:
            s = torch.norm(rows_f, p=1, dim=1)
            self._last_scores = s
            return s
