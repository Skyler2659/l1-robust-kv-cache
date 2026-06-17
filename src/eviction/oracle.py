"""Oracle eviction baselines for upper-bound and sanity-check analysis."""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.eviction.base import BaseEviction


class _OracleBase(BaseEviction):
    method_family = "oracle"
    oracle = True
    supports_backends = ("torch", "mlx")
    requires_attention = False
    requires_scores = False
    score_source = "benchmark_metadata"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.oracle_positions: List[int] = []

    def set_sample_metadata(self, sample: Dict[str, Any]) -> None:
        self.oracle_positions = self._positions_from_sample(sample)

    def _positions_from_sample(self, sample: Dict[str, Any]) -> List[int]:
        raise NotImplementedError

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        seq_len = layer_k.size(self.k_seq_dim)
        scores = torch.zeros(seq_len, device=layer_k.device, dtype=torch.float32)
        if self.oracle_positions:
            idx = torch.tensor(self.oracle_positions, device=layer_k.device, dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < seq_len)]
            if idx.numel() > 0:
                scores[idx.unique()] = 1.0
        return scores

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        oracle = torch.tensor(self.oracle_positions, device=device, dtype=torch.long)
        oracle = oracle[(oracle >= 0) & (oracle < seq_len)].unique(sorted=True)
        keep = torch.cat([reserved, oracle]).unique(sorted=True)
        return self._ensure_budget(
            keep,
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=keep if keep.numel() <= budget else reserved,
        )


class OracleEvidenceEviction(_OracleBase):
    """Keep known evidence token positions when benchmark metadata provides them."""

    name = "oracle_evidence"

    def _positions_from_sample(self, sample: Dict[str, Any]) -> List[int]:
        return [int(x) for x in sample.get("evidence_positions") or []]


class OracleAnswerRegionEviction(_OracleBase):
    """Keep known answer-region token positions for sanity checks only."""

    name = "oracle_answer_region"

    def _positions_from_sample(self, sample: Dict[str, Any]) -> List[int]:
        metadata = sample.get("metadata", {}) or {}
        start = metadata.get("answer_token_start")
        end = metadata.get("answer_token_end")
        if start is None or end is None:
            positions = sample.get("answer_positions") or sample.get("eval_positions") or []
            return [int(x) for x in positions]
        return list(range(int(start), int(end)))
