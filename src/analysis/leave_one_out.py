"""Leave-one-out analysis — ground-truth per-token importance.

For each token position, remove that single token and measure PPL change.
This provides a gold-standard importance ranking that can be correlated
with L1 leverage and attention scores.

Expensive (O(n) forward passes) — intended for short subsequences or
representative windows.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
from torch.nn import CrossEntropyLoss


@dataclass
class LOOResult:
    """Leave-one-out importance scores."""
    method: str
    loo_scores: torch.Tensor  # [seq_len] — PPL increase from removing each token
    top_k_indices: torch.Tensor  # indices of top-k LOO tokens
    correlation_with_scores: float = 0.0  # Spearman with provided scores
    seq_len: int = 0


class LeaveOneOutAnalyzer:
    """Compute ground-truth per-token importance via leave-one-out.

    For each position i, compute PPL(sequence without token i) - PPL(full).
    Higher LOO score = more important token.
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

    def compute_loo(
        self,
        model,
        input_ids: torch.Tensor,
        start: int = 0,
        end: Optional[int] = None,
        device: str = "cuda",
        progress_every: int = 50,
    ) -> torch.Tensor:
        """Compute leave-one-out scores for a range of token positions.

        Args:
            model: HF causal LM
            input_ids: [1, seq_len] input tokens
            start: first position to test
            end: last position to test (exclusive), None = seq_len
            device: compute device
            progress_every: print progress interval

        Returns:
            [seq_len] tensor of LOO scores (0 for positions not tested)
        """
        seq_len = input_ids.size(1)
        if end is None:
            end = seq_len
        end = min(end, seq_len)

        # Baseline PPL
        baseline_ppl = self._ppl(model, input_ids, device)
        loo_scores = torch.zeros(seq_len)

        for i in range(start, end):
            # Remove token at position i
            keep_mask = torch.ones(seq_len, dtype=torch.bool)
            keep_mask[i] = False
            reduced_ids = input_ids[:, keep_mask]

            ppl_without = self._ppl(model, reduced_ids, device)
            loo_scores[i] = ppl_without - baseline_ppl

            if progress_every > 0 and (i - start + 1) % progress_every == 0:
                print(f"  [LOO] {i - start + 1}/{end - start} "
                      f"pos={i} ppl_delta={loo_scores[i]:.4f}", flush=True)

        return loo_scores

    def analyze(
        self,
        model,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        start: int = 0,
        end: Optional[int] = None,
        top_k: int = 50,
        method_name: str = "loo",
        device: str = "cuda",
    ) -> LOOResult:
        """Compute LOO scores and correlate with a scoring method.

        Args:
            model: HF causal LM
            input_ids: input tokens
            scores: [seq_len] importance scores from some method
            start: first position
            end: last position (exclusive)
            top_k: number of top LOO tokens to report
            method_name: identifier
            device: compute device

        Returns:
            LOOResult with LOO scores and correlation
        """
        from src.analysis.rank_correlation import _spearman

        seq_len = input_ids.size(1)
        if end is None:
            end = seq_len

        loo_scores = self.compute_loo(model, input_ids, start, end, device)
        top_k = min(top_k, end - start)
        top_k_indices = torch.topk(loo_scores[start:end], top_k).indices + start

        # Correlation with provided scores
        min_len = min(loo_scores[start:end].numel(), scores[start:end].numel())
        corr = _spearman(loo_scores[start:start + min_len],
                         scores[start:start + min_len])

        return LOOResult(
            method=method_name,
            loo_scores=loo_scores,
            top_k_indices=top_k_indices,
            correlation_with_scores=corr,
            seq_len=seq_len,
        )

    def compare_with_methods(
        self,
        model,
        input_ids: torch.Tensor,
        all_scores: Dict[str, torch.Tensor],
        start: int = 0,
        end: Optional[int] = None,
        device: str = "cuda",
    ) -> Dict[str, LOOResult]:
        """Compare LOO ground truth with multiple scoring methods.

        Returns dict mapping method_name -> LOOResult (each with correlation).
        """
        from src.analysis.rank_correlation import _spearman

        seq_len = input_ids.size(1)
        if end is None:
            end = seq_len

        # Compute LOO once
        loo = self.compute_loo(model, input_ids, start, end, device)

        results: Dict[str, LOOResult] = {}
        for method_name, scores in all_scores.items():
            min_len = min(loo[start:end].numel(), scores[start:end].numel())
            corr = _spearman(loo[start:start + min_len],
                             scores[start:start + min_len])
            top_k = min(50, end - start)
            top_k_idx = torch.topk(loo[start:end], top_k).indices + start
            results[method_name] = LOOResult(
                method=method_name,
                loo_scores=loo,
                top_k_indices=top_k_idx,
                correlation_with_scores=corr,
                seq_len=seq_len,
            )
        return results
