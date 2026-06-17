"""Restoration experiment — add tokens back and measure PPL recovery.

Start with a minimal cache (sink + recent only), then add back tokens
selected by each method. If L1-selected tokens restore PPL faster than
attention-selected tokens, this demonstrates their "irreplaceable" nature.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
from torch.nn import CrossEntropyLoss


@dataclass
class RestorationResult:
    """Result of a restoration experiment."""
    method: str
    baseline_ppl: float  # sink+recent only
    restored_ppls: List[float] = field(default_factory=list)
    ppl_recovery: List[float] = field(default_factory=list)  # 0-1 scale
    n_tokens_added: List[int] = field(default_factory=list)
    full_ppl: float = 0.0
    n_total_available: int = 0


class RestorationExperiment:
    """Measure PPL recovery as tokens are added back to a minimal cache.

    The experiment:
    1. Baseline: only sink + recent tokens in cache → high PPL
    2. Add top-k tokens by method score → measure PPL improvement
    3. Plot recovery curve: PPL vs. number of tokens added

    If L1 leverage tokens restore PPL faster than attention tokens,
    it means they capture "irreplaceable" information.
    """

    @torch.no_grad()
    def _eval_ppl_subset(
        self,
        model,
        full_input_ids: torch.Tensor,
        keep_indices: torch.Tensor,
        device: str = "cuda",
    ) -> float:
        """Evaluate PPL on a subset of tokens (selected from full sequence).

        We keep only the tokens at keep_indices and run the model.
        """
        if keep_indices.numel() < 2:
            return float("inf")
        sorted_idx = keep_indices.sort().values
        subset_ids = full_input_ids[:, sorted_idx]
        loss_fn = CrossEntropyLoss()
        outputs = model(input_ids=subset_ids.to(device))
        logits = outputs.logits[:, :-1, :]
        targets = subset_ids[:, 1:].to(device)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return math.exp(loss.item())

    def run(
        self,
        model,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        sink_size: int = 4,
        recent_size: int = 16,
        n_steps: int = 10,
        method_name: str = "method",
        device: str = "cuda",
    ) -> RestorationResult:
        """Run restoration experiment for one scoring method.

        Args:
            model: HF causal LM
            input_ids: [1, seq_len] full input
            scores: [seq_len] importance scores
            sink_size: number of initial tokens always kept
            recent_size: number of final tokens always kept
            n_steps: number of restoration steps
            method_name: identifier
            device: compute device

        Returns:
            RestorationResult with recovery curve
        """
        seq_len = input_ids.size(1)

        # Baseline: sink + recent only
        sink_idx = torch.arange(0, min(sink_size, seq_len))
        recent_start = max(0, seq_len - recent_size)
        recent_idx = torch.arange(recent_start, seq_len)
        baseline_idx = torch.cat([sink_idx, recent_idx]).unique(sorted=True)

        baseline_ppl = self._eval_ppl_subset(model, input_ids, baseline_idx, device)

        # Full PPL (all tokens)
        full_idx = torch.arange(seq_len)
        full_ppl = self._eval_ppl_subset(model, input_ids, full_idx, device)

        # Tokens available for restoration (not in sink/recent)
        baseline_set = set(baseline_idx.tolist())
        available_scores = scores.clone()
        for i in baseline_set:
            if i < available_scores.numel():
                available_scores[i] = -float("inf")

        available_mask = torch.ones(seq_len, dtype=torch.bool)
        for i in baseline_set:
            if i < seq_len:
                available_mask[i] = False
        n_available = int(available_mask.sum().item())

        # Rank available tokens by score
        ranked = torch.argsort(available_scores, descending=True)
        ranked = ranked[available_scores[ranked] > -float("inf")]

        # Restoration steps
        step_sizes = [max(1, n_available // (n_steps + 1)) * (i + 1)
                      for i in range(n_steps)]
        step_sizes = [min(s, n_available) for s in step_sizes]

        restored_ppls = [baseline_ppl]
        n_tokens_added = [0]

        for step_size in step_sizes:
            added_idx = ranked[:step_size]
            current_idx = torch.cat([baseline_idx, added_idx]).unique(sorted=True)
            ppl = self._eval_ppl_subset(model, input_ids, current_idx, device)
            restored_ppls.append(ppl)
            n_tokens_added.append(int(step_size))

        # Compute recovery (0 = baseline PPL, 1 = full PPL)
        ppl_range = baseline_ppl - full_ppl
        if ppl_range > 0:
            ppl_recovery = [(baseline_ppl - p) / ppl_range for p in restored_ppls]
        else:
            ppl_recovery = [0.0] * len(restored_ppls)

        return RestorationResult(
            method=method_name,
            baseline_ppl=baseline_ppl,
            restored_ppls=restored_ppls,
            ppl_recovery=ppl_recovery,
            n_tokens_added=n_tokens_added,
            full_ppl=full_ppl,
            n_total_available=n_available,
        )

    def compare_methods(
        self,
        model,
        input_ids: torch.Tensor,
        all_scores: Dict[str, torch.Tensor],
        sink_size: int = 4,
        recent_size: int = 16,
        n_steps: int = 10,
        device: str = "cuda",
    ) -> List[RestorationResult]:
        """Compare restoration curves across methods.

        Returns list of RestorationResult sorted by final recovery.
        """
        results = []
        for method_name, scores in all_scores.items():
            r = self.run(model, input_ids, scores, sink_size, recent_size,
                        n_steps, method_name, device)
            results.append(r)
        # Sort by final recovery (higher = better)
        results.sort(key=lambda r: r.ppl_recovery[-1] if r.ppl_recovery else 0.0,
                     reverse=True)
        return results
