"""L1 leverage score eviction — the project's core method."""
from __future__ import annotations
from typing import Optional
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


class L1LeverageEviction(BaseEviction):
    """L1 leverage score selection with sink + recent + last.

    Adapted from the original ``l1_llm`` implementation.
    Supports ``score_source`` in {"v", "k", "kv"} and configurable
    ``update_interval`` for amortized sketch cost.
    """
    name = "l1_mixed"
    method_family = "geometry"
    supports_backends = ("torch", "mlx")
    requires_scores = True
    score_source = "value"
    approximate = True

    def __init__(
        self,
        score_source="v",
        sketch_dim=1024,
        update_interval=32,
        update_policy="every_n_steps",
        seed=0,
        use_reweight=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.score_source = str(score_source).lower()
        self.sketch_dim = int(sketch_dim)
        self.update_interval = max(0, int(update_interval))
        self.update_policy = str(update_policy or "every_n_steps").lower()
        if self.update_interval == 0 and self.update_policy == "every_n_steps":
            self.update_policy = "prefill_only"
        self.seed = int(seed)
        self.use_reweight = bool(use_reweight)
        self._estimators: dict[int, object] = {}
        self._fit_layers: set[int] = set()
        self.score_update_count = 0

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        seq_len, row_dim = rows.shape
        if seq_len <= self.cache_size:
            return torch.ones(seq_len, device=rows.device)
        est = self._get_estimator(layer_idx)
        force_refit = self._should_refit(layer_idx)
        if force_refit:
            self._fit_layers.add(layer_idx)
            self.score_update_count += 1
        return est.scores(rows, force_refit=force_refit)

    def _should_refit(self, layer_idx: int) -> bool:
        if layer_idx not in self._fit_layers:
            return True
        if self.update_policy in ("prefill_only", "never_after_prefill"):
            return False
        if self.update_policy != "every_n_steps":
            return False
        return self.update_interval > 0 and (self._steps % self.update_interval) == 0

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        if scores is None:
            reserved = self._reserved_indices(seq_len, budget, device)
            return self._fill_budget(reserved, seq_len, budget, device)

        sink = min(self.sink_size, max(0, budget - 1))
        max_recent = max(0, budget - sink - 1)
        recent = min(self.recent_size, max_recent)
        l1_budget = max(0, budget - sink - recent - 1)

        parts = []
        if sink > 0:
            parts.append(torch.arange(sink, device=device, dtype=torch.long))
        if recent > 0:
            rec_start = seq_len - 1 - recent
            parts.append(torch.arange(rec_start, seq_len - 1, device=device))
        if l1_budget > 0:
            cand_start = sink
            cand_end = seq_len - 1 - recent
            if cand_end > cand_start:
                cand_scores = scores[cand_start:cand_end].to(device)
                topk = min(l1_budget, cand_scores.numel())
                l1_idx = torch.topk(cand_scores, k=topk).indices + cand_start
                parts.append(l1_idx)
        parts.append(torch.tensor([seq_len - 1], device=device, dtype=torch.long))
        keep = torch.cat(parts).unique(sorted=True)
        reserved = keep.clone()
        return self._ensure_budget(
            keep,
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved if reserved.numel() <= budget else None,
        )

    def _get_rows(self, layer_k, layer_v):
        v_rows = mean_heads(layer_v, self.v_seq_dim)
        if v_rows is None or self.score_source == "v":
            return v_rows
        if self.score_source == "k":
            return mean_heads(layer_k, self.k_seq_dim)
        # "kv" concat
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if k_rows is None or k_rows.shape[0] != v_rows.shape[0]:
            return v_rows
        return torch.cat([k_rows.float(), v_rows.float()], dim=-1)

    def _get_estimator(self, layer_idx):
        if layer_idx not in self._estimators:
            from src.sketching.woodruff_l1 import WoodruffL1Estimator
            self._estimators[layer_idx] = WoodruffL1Estimator(
                sketch_dim=self.sketch_dim,
                seed=self.seed + layer_idx,
            )
        return self._estimators[layer_idx]

    def reset(self):
        super().reset()
        self._estimators.clear()
        self._fit_layers.clear()
        self.score_update_count = 0
