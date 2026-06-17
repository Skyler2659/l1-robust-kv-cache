"""Rank correlation analysis — Spearman / Kendall τ between scoring methods.

Tests whether attention saliency and L1 leverage produce similar token
rankings. Low rank correlation supports the hypothesis that they measure
fundamentally different aspects of token importance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class CorrelationResult:
    """Result of pairwise rank correlation."""
    method_a: str
    method_b: str
    spearman: float
    kendall_tau: float
    pearson: float
    layer_wise_spearman: Dict[int, float] = field(default_factory=dict)
    n_tokens: int = 0


def _ranks(scores: torch.Tensor) -> torch.Tensor:
    """Convert scores to ranks (1-indexed, ties averaged)."""
    n = scores.numel()
    if n == 0:
        return scores
    sorted_idx = torch.argsort(scores, descending=True)
    ranks = torch.zeros_like(scores, dtype=torch.float64)
    ranks[sorted_idx] = torch.arange(1, n + 1, dtype=torch.float64, device=scores.device)
    return ranks


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    if a.numel() < 3 or b.numel() < 3:
        return 0.0
    min_len = min(a.numel(), b.numel())
    a, b = a[:min_len].double(), b[:min_len].double()
    ra = _ranks(a)
    rb = _ranks(b)
    ra_z = ra - ra.mean()
    rb_z = rb - rb.mean()
    num = (ra_z * rb_z).sum()
    den = (ra_z.pow(2).sum() * rb_z.pow(2).sum()).sqrt().clamp_min(1e-12)
    return (num / den).item()


def _kendall_tau(a: torch.Tensor, b: torch.Tensor, max_n: int = 256) -> float:
    """Kendall tau-b (approximation for large n via subsampling)."""
    min_len = min(a.numel(), b.numel())
    if min_len < 3:
        return 0.0
    a, b = a[:min_len].double(), b[:min_len].double()
    if min_len > max_n:
        idx = torch.randperm(min_len)[:max_n]
        a, b = a[idx], b[idx]
        min_len = max_n
    ra = _ranks(a)
    rb = _ranks(b)
    concordant = 0
    discordant = 0
    n = min_len
    for i in range(n):
        for j in range(i + 1, n):
            a_diff = ra[i] - ra[j]
            b_diff = rb[i] - rb[j]
            prod = a_diff * b_diff
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation."""
    min_len = min(a.numel(), b.numel())
    if min_len < 3:
        return 0.0
    a, b = a[:min_len].double(), b[:min_len].double()
    a_z = a - a.mean()
    b_z = b - b.mean()
    num = (a_z * b_z).sum()
    den = (a_z.pow(2).sum() * b_z.pow(2).sum()).sqrt().clamp_min(1e-12)
    return (num / den).item()


class RankCorrelationAnalyzer:
    """Compute rank correlation between token importance scores.

    Supports attention scores, L1 leverage, L2 leverage, norm-based, etc.
    """

    def pairwise(
        self,
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
        method_a: str = "method_a",
        method_b: str = "method_b",
    ) -> CorrelationResult:
        """Compute rank correlation between two score vectors."""
        sp = _spearman(scores_a, scores_b)
        kt = _kendall_tau(scores_a, scores_b)
        pe = _pearson(scores_a, scores_b)
        return CorrelationResult(
            method_a=method_a, method_b=method_b,
            spearman=sp, kendall_tau=kt, pearson=pe,
            n_tokens=min(scores_a.numel(), scores_b.numel()),
        )

    def layer_wise(
        self,
        scores_a_by_layer: Dict[int, torch.Tensor],
        scores_b_by_layer: Dict[int, torch.Tensor],
        method_a: str = "method_a",
        method_b: str = "method_b",
    ) -> CorrelationResult:
        """Compute per-layer rank correlation and aggregate."""
        layer_sp: Dict[int, float] = {}
        all_a: List[torch.Tensor] = []
        all_b: List[torch.Tensor] = []

        for layer_idx in sorted(scores_a_by_layer.keys()):
            if layer_idx not in scores_b_by_layer:
                continue
            sa = scores_a_by_layer[layer_idx]
            sb = scores_b_by_layer[layer_idx]
            min_len = min(sa.numel(), sb.numel())
            if min_len < 3:
                continue
            layer_sp[layer_idx] = _spearman(sa[:min_len], sb[:min_len])
            all_a.append(sa[:min_len])
            all_b.append(sb[:min_len])

        if all_a:
            global_a = torch.cat(all_a)
            global_b = torch.cat(all_b)
            sp = _spearman(global_a, global_b)
            kt = _kendall_tau(global_a, global_b)
            pe = _pearson(global_a, global_b)
            n = global_a.numel()
        else:
            sp = kt = pe = 0.0
            n = 0

        return CorrelationResult(
            method_a=method_a, method_b=method_b,
            spearman=sp, kendall_tau=kt, pearson=pe,
            layer_wise_spearman=layer_sp,
            n_tokens=n,
        )

    def full_matrix(
        self,
        all_scores: Dict[str, Dict[int, torch.Tensor]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Spearman matrix across all methods."""
        methods = sorted(all_scores.keys())
        matrix: Dict[str, Dict[str, float]] = {}
        for ma in methods:
            matrix[ma] = {}
            for mb in methods:
                if ma == mb:
                    matrix[ma][mb] = 1.0
                elif mb in matrix and ma in matrix[mb]:
                    matrix[ma][mb] = matrix[mb][ma]
                else:
                    r = self.layer_wise(all_scores[ma], all_scores[mb], ma, mb)
                    matrix[ma][mb] = r.spearman
        return matrix
