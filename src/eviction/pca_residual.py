"""PCA residual score eviction — high reconstruction error = irreplaceable."""
from __future__ import annotations
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


class PCAResidualEviction(BaseEviction):
    """Score = reconstruction error under top-r PCA.

    High residual → hard to reconstruct from principal directions → irreplaceable.
    """
    name = "pca_residual"

    def __init__(self, n_components=16, score_source="v", **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.score_source = score_source

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        if self.score_source == "k":
            rows = mean_heads(layer_k, self.k_seq_dim)
        elif self.score_source == "kv":
            k_r = mean_heads(layer_k, self.k_seq_dim)
            v_r = mean_heads(layer_v, self.v_seq_dim)
            if k_r is not None and v_r is not None and k_r.shape[0] == v_r.shape[0]:
                rows = torch.cat([k_r.float(), v_r.float()], dim=-1)
            else:
                rows = v_r
        else:
            rows = mean_heads(layer_v, self.v_seq_dim)
        if rows is None:
            return None
        rows = rows.float()
        n, d = rows.shape
        r = min(self.n_components, n - 1, d)
        if r <= 0:
            return torch.norm(rows, p=2, dim=1)
        centered = rows - rows.mean(dim=0, keepdim=True)
        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            V_r = Vh[:r].T  # [d, r]
            proj = centered @ V_r  # [n, r]
            reconstructed = proj @ V_r.T  # [n, d]
            residual = torch.norm(centered - reconstructed, p=2, dim=1)
            return residual
        except Exception:
            return torch.norm(rows, p=2, dim=1)

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
