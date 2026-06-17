"""Counterfactual deletion — measure PPL impact of removing top-k tokens.

For each method, identify the top-k tokens by importance score, then
delete ONLY those tokens (keeping everything else). If L1-selected tokens
cause larger PPL degradation than attention-selected tokens, this supports
the "geometric irreplaceability" hypothesis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
from torch.nn import CrossEntropyLoss


@dataclass
class DeletionResult:
    """Result of counterfactual deletion for one method."""
    method: str
    original_ppl: float
    deleted_ppl: float
    ppl_ratio: float  # deleted / original (> 1 means deletion hurt)
    n_deleted: int
    n_remaining: int
    seq_len: int
    per_layer_ppl: Dict[int, float] = field(default_factory=dict)


class CounterfactualDeletion:
    """Remove top-k tokens and measure PPL degradation.

    The key insight: if tokens selected by L1 leverage are "geometrically
    irreplaceable," then deleting them should cause larger PPL increases
    than deleting the same number of randomly chosen or attention-selected
    tokens.

    Usage:
        analyzer = CounterfactualDeletion()
        result = analyzer.run(
            model=model,
            input_ids=input_ids,
            scores=scores_tensor,
            n_delete=50,
            method_name="l1_leverage",
        )
    """

    @torch.no_grad()
    def _compute_ppl(
        self, model, input_ids: torch.Tensor, device: str = "cuda",
    ) -> float:
        """Compute PPL on full sequence (no cache eviction)."""
        loss_fn = CrossEntropyLoss()
        outputs = model(input_ids=input_ids.to(device))
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:].to(device)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return math.exp(loss.item())

    @torch.no_grad()
    def _compute_ppl_with_deleted(
        self,
        model,
        input_ids: torch.Tensor,
        delete_mask: torch.Tensor,
        device: str = "cuda",
    ) -> float:
        """Compute PPL after deleting tokens where delete_mask is True.

        Deleted tokens are removed from the sequence before the forward pass.
        """
        keep_mask = ~delete_mask
        kept_ids = input_ids[:, keep_mask.squeeze()]
        if kept_ids.size(1) < 2:
            return float("inf")

        loss_fn = CrossEntropyLoss()
        outputs = model(input_ids=kept_ids.to(device))
        logits = outputs.logits[:, :-1, :]
        targets = kept_ids[:, 1:].to(device)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return math.exp(loss.item())

    def run(
        self,
        model,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        n_delete: int,
        method_name: str = "method",
        device: str = "cuda",
    ) -> DeletionResult:
        """Run counterfactual deletion experiment.

        Args:
            model: HF causal LM
            input_ids: [1, seq_len] input tokens
            scores: [seq_len] importance scores (higher = more important)
            n_delete: number of top-scored tokens to delete
            method_name: identifier for this scoring method
            device: compute device

        Returns:
            DeletionResult with original and deleted PPL
        """
        seq_len = input_ids.size(1)
        n_delete = min(n_delete, seq_len - 2)  # keep at least 2 tokens

        # Original PPL
        original_ppl = self._compute_ppl(model, input_ids, device)

        # Find top-k tokens to delete
        topk_indices = torch.topk(scores, n_delete).indices
        delete_mask = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
        delete_mask[topk_indices] = True

        # PPL after deletion
        deleted_ppl = self._compute_ppl_with_deleted(
            model, input_ids, delete_mask, device)

        ppl_ratio = deleted_ppl / max(original_ppl, 1e-8)

        return DeletionResult(
            method=method_name,
            original_ppl=original_ppl,
            deleted_ppl=deleted_ppl,
            ppl_ratio=ppl_ratio,
            n_deleted=n_delete,
            n_remaining=seq_len - n_delete,
            seq_len=seq_len,
        )

    def compare_methods(
        self,
        model,
        input_ids: torch.Tensor,
        all_scores: Dict[str, torch.Tensor],
        n_delete: int = 50,
        device: str = "cuda",
    ) -> List[DeletionResult]:
        """Compare counterfactual deletion across multiple methods.

        Args:
            model: HF causal LM
            input_ids: input tokens
            all_scores: method_name -> [seq_len] score tensor
            n_delete: tokens to delete per method
            device: compute device

        Returns:
            List of DeletionResult sorted by ppl_ratio descending
        """
        results = []
        for method_name, scores in all_scores.items():
            r = self.run(model, input_ids, scores, n_delete, method_name, device)
            results.append(r)
        results.sort(key=lambda r: r.ppl_ratio, reverse=True)
        return results

    def sweep_deletion_fractions(
        self,
        model,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        fractions: Optional[List[float]] = None,
        method_name: str = "method",
        device: str = "cuda",
    ) -> List[DeletionResult]:
        """Sweep deletion across fractions (e.g., 5%, 10%, 20%, 50%).

        Returns a list of DeletionResult for each fraction.
        """
        if fractions is None:
            fractions = [0.05, 0.1, 0.2, 0.3, 0.5]

        seq_len = input_ids.size(1)
        results = []
        for frac in fractions:
            n_del = max(1, int(seq_len * frac))
            r = self.run(model, input_ids, scores, n_del,
                        f"{method_name}_del{frac:.0%}", device)
            results.append(r)
        return results
