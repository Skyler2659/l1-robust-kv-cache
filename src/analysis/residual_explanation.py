"""Residual explanation — decompose the PPL gap between full and evicted cache.

The full cache produces some PPL. An eviction method produces higher PPL.
This analysis decomposes the PPL gap: how much of the degradation is due to
losing L1-important tokens vs. attention-important tokens vs. other tokens?

Supports the paper's claim that L1 captures a component of importance
that attention misses.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import torch
from torch.nn import CrossEntropyLoss


@dataclass
class ResidualDecomposition:
    """Decomposition of PPL gap into attributable components."""
    method: str
    full_ppl: float
    evicted_ppl: float
    ppl_gap: float
    # PPL when only L1-selected tokens are removed (attention tokens kept)
    ppl_l1_only_removed: float = 0.0
    # PPL when only attention-selected tokens are removed (L1 tokens kept)
    ppl_attn_only_removed: float = 0.0
    # Fraction of PPL gap explained by L1 tokens
    l1_gap_fraction: float = 0.0
    # Fraction of PPL gap explained by attention tokens
    attn_gap_fraction: float = 0.0
    # Fraction unexplained
    unexplained_fraction: float = 0.0


class ResidualExplanation:
    """Decompose PPL gap between full and evicted cache.

    Experiment:
    1. Full cache PPL (all tokens) → P_full
    2. Evicted cache PPL (method's selection) → P_evict
    3. PPL with only L1 tokens removed (attention tokens kept) → P_l1_removed
    4. PPL with only attention tokens removed (L1 tokens kept) → P_attn_removed

    Gap decomposition:
    - L1 contribution = (P_l1_removed - P_full) / (P_evict - P_full)
    - Attention contribution = (P_attn_removed - P_full) / (P_evict - P_full)
    """

    @torch.no_grad()
    def _ppl(self, model, input_ids: torch.Tensor, device: str) -> float:
        if input_ids.size(1) < 2:
            return float("inf")
        loss_fn = CrossEntropyLoss()
        out = model(input_ids=input_ids.to(device))
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:].to(device)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return math.exp(loss.item())

    def _eval_with_kept_tokens(
        self,
        model,
        input_ids: torch.Tensor,
        keep_indices: Set[int],
        device: str,
    ) -> float:
        """Evaluate PPL keeping only specified token positions."""
        seq_len = input_ids.size(1)
        sorted_idx = sorted(keep_indices)
        if len(sorted_idx) < 2:
            return float("inf")
        idx_tensor = torch.tensor(sorted_idx, dtype=torch.long)
        subset_ids = input_ids[:, idx_tensor]
        return self._ppl(model, subset_ids, device)

    def decompose(
        self,
        model,
        input_ids: torch.Tensor,
        l1_selected: Set[int],
        attn_selected: Set[int],
        evicted_selected: Set[int],
        sink_indices: Optional[Set[int]] = None,
        recent_indices: Optional[Set[int]] = None,
        method_name: str = "decomposition",
        device: str = "cuda",
    ) -> ResidualDecomposition:
        """Decompose PPL gap.

        Args:
            model: HF causal LM
            input_ids: [1, seq_len]
            l1_selected: token positions selected by L1 method
            attn_selected: token positions selected by attention method
            evicted_selected: union of all tokens kept by the eviction method
            sink_indices: always-kept sink token positions
            recent_indices: always-kept recent token positions
            method_name: identifier
            device: compute device

        Returns:
            ResidualDecomposition with gap fractions
        """
        seq_len = input_ids.size(1)
        sink = sink_indices or set()
        recent = recent_indices or set()

        # Base kept tokens (sink + recent)
        base_kept = sink | recent

        # 1. Full cache PPL
        full_set = set(range(seq_len))
        full_ppl = self._eval_with_kept_tokens(model, input_ids, full_set, device)

        # 2. Evicted PPL
        evicted_ppl = self._eval_with_kept_tokens(
            model, input_ids, evicted_selected | base_kept, device)

        # 3. PPL with only L1 tokens removed (keep everything except L1-selected tokens)
        # i.e., keep all tokens minus L1-selected ones that aren't in base
        l1_removable = l1_selected - base_kept
        keep_no_l1 = full_set - l1_removable
        ppl_l1_removed = self._eval_with_kept_tokens(model, input_ids, keep_no_l1, device)

        # 4. PPL with only attention tokens removed
        attn_removable = attn_selected - base_kept
        keep_no_attn = full_set - attn_removable
        ppl_attn_removed = self._eval_with_kept_tokens(
            model, input_ids, keep_no_attn, device)

        # Gap decomposition
        gap = evicted_ppl - full_ppl
        if gap > 1e-6:
            l1_frac = (ppl_l1_removed - full_ppl) / gap
            attn_frac = (ppl_attn_removed - full_ppl) / gap
            unexplained = 1.0 - l1_frac - attn_frac
        else:
            l1_frac = attn_frac = unexplained = 0.0

        return ResidualDecomposition(
            method=method_name,
            full_ppl=full_ppl,
            evicted_ppl=evicted_ppl,
            ppl_gap=gap,
            ppl_l1_only_removed=ppl_l1_removed,
            ppl_attn_only_removed=ppl_attn_removed,
            l1_gap_fraction=l1_frac,
            attn_gap_fraction=attn_frac,
            unexplained_fraction=unexplained,
        )
