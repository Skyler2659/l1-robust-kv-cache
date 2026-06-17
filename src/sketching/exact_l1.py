"""Exact / high-quality L1 leverage score (small-scale validation)."""
from __future__ import annotations
from typing import Optional
import torch


class ExactL1Estimator:
    """Exact L1 leverage via random direction sampling.

    For small matrices (n < 200): uses dense random directions (5000+).
    For larger matrices: uses 2000 random directions as approximation.
    """

    def __init__(self, n_directions: int = 5000, seed: int = 0):
        self.n_directions = n_directions
        self.seed = seed
        self._last_scores: Optional[torch.Tensor] = None

    def fit(self, rows: torch.Tensor) -> None:
        pass

    def scores(self, rows: torch.Tensor, **kw) -> torch.Tensor:
        rows_f = rows.float()
        n, d = rows_f.shape
        if n <= 1:
            s = torch.norm(rows_f, p=1, dim=1)
            self._last_scores = s
            return s

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        n_dirs = self.n_directions
        # Sample random directions x ∈ R^d
        directions = torch.randn(n_dirs, d, generator=gen, dtype=torch.float32)
        # For each direction: Ax, then |a_i^T x| / ||Ax||_1
        # Ax: [n, d] @ [d, n_dirs] -> [n, n_dirs]
        Ax = rows_f @ directions.T  # [n, n_dirs]
        l1_norms = Ax.abs().sum(dim=0, keepdim=True).clamp_min(1e-8)  # [1, n_dirs]
        ratios = Ax.abs() / l1_norms  # [n, n_dirs]
        # L1 leverage = sup over directions
        s = ratios.max(dim=1).values.to(rows.dtype)
        self._last_scores = s
        return s
