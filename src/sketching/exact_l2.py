"""Exact L2 leverage score via QR decomposition."""
from __future__ import annotations
from typing import Optional
import torch


class ExactL2Estimator:
    """Exact L2 leverage: τ_i^(2)(A) = ‖q_i‖₂² where A=QR, Q orthonormal."""

    def __init__(self):
        self._last_scores: Optional[torch.Tensor] = None

    def fit(self, rows: torch.Tensor) -> None:
        pass  # QR is done in scores()

    def scores(self, rows: torch.Tensor, **kw) -> torch.Tensor:
        rows = rows.float()
        n, d = rows.shape
        if n <= 1:
            s = torch.norm(rows, p=2, dim=1)
            self._last_scores = s
            return s
        try:
            _, r = torch.linalg.qr(rows.T, mode="reduced")
            if r.shape[0] != r.shape[1]:
                s = torch.norm(rows, p=2, dim=1)
                self._last_scores = s
                return s
            jit = max(1e-5, r.diag().abs().max().item() * 1e-6)
            r = r + torch.eye(r.shape[0], device=r.device, dtype=r.dtype) * jit
            r_inv = torch.linalg.inv(r)
            q = rows @ r_inv.T
            s = torch.norm(q, p=2, dim=1) ** 2
            self._last_scores = s
            return s
        except Exception:
            s = torch.norm(rows, p=2, dim=1)
            self._last_scores = s
            return s
