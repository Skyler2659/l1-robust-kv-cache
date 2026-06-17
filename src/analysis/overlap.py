"""Overlap analysis — Jaccard / set overlap between eviction selections.

Measures how similar the token sets selected by different methods are.
Low overlap between attention-based and L1-based methods supports the
hypothesis that they capture complementary importance signals.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import numpy as np


@dataclass
class OverlapResult:
    """Result of a pairwise overlap analysis."""
    method_a: str
    method_b: str
    jaccard: float
    overlap_count: int
    union_count: int
    a_only_count: int
    b_only_count: int
    layer_wise_jaccard: Dict[int, float] = field(default_factory=dict)
    head_wise_jaccard: Dict[Tuple[int, int], float] = field(default_factory=dict)


class OverlapAnalyzer:
    """Compute token-set overlap between eviction methods.

    Usage:
        analyzer = OverlapAnalyzer()
        # After running eviction methods and collecting last_selected:
        result = analyzer.pairwise_overlap(
            selected_a=eviction_a.last_selected,
            selected_b=eviction_b.last_selected,
            method_a="h2o",
            method_b="l1_leverage",
        )
    """

    def pairwise_overlap(
        self,
        selected_a: Dict[int, torch.Tensor],
        selected_b: Dict[int, torch.Tensor],
        method_a: str = "method_a",
        method_b: str = "method_b",
    ) -> OverlapResult:
        """Compute Jaccard overlap between two methods' selected indices.

        Args:
            selected_a: dict mapping layer_idx -> selected token indices
            selected_b: dict mapping layer_idx -> selected token indices
            method_a: name of method A
            method_b: name of method B

        Returns:
            OverlapResult with global and per-layer Jaccard scores
        """
        layer_jaccards: Dict[int, float] = {}
        all_a: Set[int] = set()
        all_b: Set[int] = set()

        all_layers = sorted(set(selected_a.keys()) | set(selected_b.keys()))
        for layer_idx in all_layers:
            a_set = set(selected_a.get(layer_idx, torch.tensor([])).tolist())
            b_set = set(selected_b.get(layer_idx, torch.tensor([])).tolist())
            all_a |= a_set
            all_b |= b_set

            intersection = a_set & b_set
            union = a_set | b_set
            if union:
                layer_jaccards[layer_idx] = len(intersection) / len(union)
            else:
                layer_jaccards[layer_idx] = 1.0

        global_intersection = all_a & all_b
        global_union = all_a | all_b
        global_jaccard = (len(global_intersection) / len(global_union)
                         if global_union else 1.0)

        return OverlapResult(
            method_a=method_a,
            method_b=method_b,
            jaccard=global_jaccard,
            overlap_count=len(global_intersection),
            union_count=len(global_union),
            a_only_count=len(all_a - all_b),
            b_only_count=len(all_b - all_a),
            layer_wise_jaccard=layer_jaccards,
        )

    def full_matrix(
        self,
        all_selected: Dict[str, Dict[int, torch.Tensor]],
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise Jaccard matrix across all methods.

        Args:
            all_selected: dict mapping method_name -> {layer_idx -> indices}

        Returns:
            Nested dict: matrix[method_a][method_b] = jaccard
        """
        methods = sorted(all_selected.keys())
        matrix: Dict[str, Dict[str, float]] = {}
        for ma in methods:
            matrix[ma] = {}
            for mb in methods:
                if ma == mb:
                    matrix[ma][mb] = 1.0
                elif mb in matrix and ma in matrix[mb]:
                    matrix[ma][mb] = matrix[mb][ma]
                else:
                    r = self.pairwise_overlap(
                        all_selected[ma], all_selected[mb], ma, mb)
                    matrix[ma][mb] = r.jaccard
        return matrix

    @staticmethod
    def to_numpy_matrix(
        matrix: Dict[str, Dict[str, float]],
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert nested dict matrix to numpy array with labels."""
        labels = sorted(matrix.keys())
        n = len(labels)
        arr = np.zeros((n, n))
        for i, la in enumerate(labels):
            for j, lb in enumerate(labels):
                arr[i, j] = matrix[la].get(lb, 0.0)
        return arr, labels
