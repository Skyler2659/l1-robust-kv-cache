"""L2 leverage score eviction — exact L2 leverage via QR decomposition."""
from __future__ import annotations
from typing import Optional
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


def l2_row_leverage_scores(rows: torch.Tensor) -> torch.Tensor:
    """Compute standard L2 row leverage scores for A with shape [n, d].

    For a full-rank matrix A = QR, the row leverage score is
    ||Q[i, :]||_2^2. Computation is done in float32 for numerical stability
    and the result is returned on the input device.
    """
    rows_f = rows.to(dtype=torch.float32)
    n, d = rows_f.shape
    if n == 0:
        return torch.empty(0, dtype=torch.float32, device=rows.device)
    if n == 1:
        return torch.ones(1, dtype=torch.float32, device=rows.device)
    try:
        q, _ = torch.linalg.qr(rows_f, mode="reduced")
        if q.numel() == 0 or not torch.isfinite(q).all():
            return torch.zeros(n, dtype=torch.float32, device=rows.device)
        return q.pow(2).sum(dim=1).to(device=rows.device)
    except Exception:
        return torch.norm(rows_f, p=2, dim=1).to(device=rows.device)


class L2LeverageEviction(BaseEviction):
    """Select tokens by L2 (statistical) leverage score.

    τ_i^(2)(A) = ||q_i||_2^2 where Q is the orthonormal basis of A's column space.
    Computed via thin QR: A = QR → scores = squared row norms of Q.
    """
    name = "l2_leverage"
    method_family = "geometry"
    supports_backends = ("torch", "mlx")
    requires_scores = True
    score_source = "value"

    def __init__(self, score_source="v", **kwargs):
        super().__init__(**kwargs)
        self.score_source = score_source

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        return l2_row_leverage_scores(rows)

    def _get_rows(self, layer_k, layer_v):
        v_rows = mean_heads(layer_v, self.v_seq_dim)
        if v_rows is None or self.score_source == "v":
            return v_rows
        k_rows = mean_heads(layer_k, self.k_seq_dim)
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
        topk = min(fill, int(valid.sum().item()))
        idx = torch.topk(masked, topk).indices
        return self._ensure_budget(
            torch.cat([reserved, idx]),
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )


def ridge_row_leverage_scores(rows: torch.Tensor, ridge_lambda: float = 1e-3) -> torch.Tensor:
    """Compute ridge leverage scores diag(A(A^T A + λI)^-1 A^T)."""
    rows_f = rows.to(dtype=torch.float32)
    n, d = rows_f.shape
    if n == 0:
        return torch.empty(0, dtype=torch.float32, device=rows.device)
    if n == 1:
        return torch.ones(1, dtype=torch.float32, device=rows.device)
    lam = max(float(ridge_lambda), 1e-8)
    try:
        gram = rows_f.T @ rows_f
        inv = torch.linalg.inv(gram + lam * torch.eye(d, device=rows.device, dtype=rows_f.dtype))
        scores = torch.sum((rows_f @ inv) * rows_f, dim=1)
        return torch.clamp(scores, min=0.0).to(device=rows.device)
    except Exception:
        return l2_row_leverage_scores(rows)


class RidgeLeverageEviction(L2LeverageEviction):
    """Ridge-regularized leverage score baseline."""

    name = "ridge_leverage"
    approximate = False

    def __init__(self, ridge_lambda: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.ridge_lambda = float(ridge_lambda)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        return ridge_row_leverage_scores(rows, self.ridge_lambda)


class ApproximateL2LeverageEviction(L2LeverageEviction):
    """Random-projection approximate L2 leverage score baseline."""

    name = "approximate_l2_leverage"
    approximate = True

    def __init__(self, sketch_dim: int = 256, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.sketch_dim = int(sketch_dim)
        self.seed = int(seed)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        rows_f = rows.float()
        n, d = rows_f.shape
        if d <= self.sketch_dim:
            return l2_row_leverage_scores(rows_f)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + int(layer_idx))
        proj = torch.randn(d, self.sketch_dim, generator=generator, dtype=torch.float32)
        proj = proj.to(rows_f.device) / max(self.sketch_dim, 1) ** 0.5
        return l2_row_leverage_scores(rows_f @ proj)
