"""Evidence recall analysis — did the eviction method keep the right tokens?

For needle-in-haystack and multi-hop tasks, we know which tokens contain
the evidence (needle text, supporting facts). This analysis measures what
fraction of those evidence tokens survive eviction under each method.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import torch


@dataclass
class EvidenceRecallResult:
    """Result of evidence recall analysis for one method."""
    method: str
    evidence_recall: float  # fraction of evidence tokens retained
    evidence_precision: float  # fraction of retained tokens that are evidence
    evidence_f1: float
    n_evidence: int
    n_retained: int
    n_total: int
    layer_wise_recall: Dict[int, float] = field(default_factory=dict)


class EvidenceRecallAnalyzer:
    """Measure whether eviction methods retain known evidence tokens.

    Usage:
        analyzer = EvidenceRecallAnalyzer()
        result = analyzer.analyze(
            selected_indices=eviction.last_selected,
            evidence_positions=evidence_token_positions,
            seq_len=total_seq_len,
            method_name="l1_leverage",
        )
    """

    def analyze(
        self,
        selected_indices: Dict[int, torch.Tensor],
        evidence_positions: torch.Tensor,
        seq_len: int,
        method_name: str = "method",
    ) -> EvidenceRecallResult:
        """Analyze evidence recall for a single method.

        Args:
            selected_indices: dict mapping layer_idx -> selected token indices
            evidence_positions: 1-D tensor of ground-truth evidence token positions
            seq_len: total sequence length
            method_name: method identifier

        Returns:
            EvidenceRecallResult with recall, precision, F1
        """
        evidence_set: Set[int] = set(evidence_positions.tolist())
        n_evidence = len(evidence_set)

        if n_evidence == 0:
            return EvidenceRecallResult(
                method=method_name, evidence_recall=0.0,
                evidence_precision=0.0, evidence_f1=0.0,
                n_evidence=0, n_retained=0, n_total=seq_len,
            )

        # Aggregate selected tokens across all layers
        all_selected: Set[int] = set()
        layer_recall: Dict[int, float] = {}

        for layer_idx, indices in selected_indices.items():
            layer_selected = set(indices.tolist())
            all_selected |= layer_selected

            layer_evidence_kept = evidence_set & layer_selected
            layer_recall[layer_idx] = len(layer_evidence_kept) / n_evidence

        # Global metrics
        evidence_kept = evidence_set & all_selected
        recall = len(evidence_kept) / n_evidence
        precision = len(evidence_kept) / len(all_selected) if all_selected else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return EvidenceRecallResult(
            method=method_name,
            evidence_recall=recall,
            evidence_precision=precision,
            evidence_f1=f1,
            n_evidence=n_evidence,
            n_retained=len(all_selected),
            n_total=seq_len,
            layer_wise_recall=layer_recall,
        )

    def compare_methods(
        self,
        all_selected: Dict[str, Dict[int, torch.Tensor]],
        evidence_positions: torch.Tensor,
        seq_len: int,
    ) -> List[EvidenceRecallResult]:
        """Compare evidence recall across multiple methods.

        Args:
            all_selected: method_name -> {layer_idx -> selected indices}
            evidence_positions: ground-truth evidence positions
            seq_len: total sequence length

        Returns:
            List of EvidenceRecallResult, sorted by recall descending
        """
        results = []
        for method_name, selected in all_selected.items():
            r = self.analyze(selected, evidence_positions, seq_len, method_name)
            results.append(r)
        results.sort(key=lambda r: r.evidence_recall, reverse=True)
        return results

    @staticmethod
    def find_evidence_positions(
        tokenizer,
        full_text: str,
        evidence_text: str,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Helper: find token positions of evidence text in the full sequence.

        Uses a simple substring search on decoded tokens.

        Args:
            tokenizer: HF tokenizer
            full_text: complete input text
            evidence_text: the evidence substring to locate
            input_ids: tokenized full text

        Returns:
            1-D tensor of token positions containing the evidence
        """
        # Find character positions
        char_start = full_text.find(evidence_text)
        if char_start == -1:
            return torch.tensor([], dtype=torch.long)

        char_end = char_start + len(evidence_text)

        # Map character positions to token positions
        positions = []
        for i in range(input_ids.size(-1)):
            token_str = tokenizer.decode(input_ids.flatten()[i:i+1])
            # Approximate: check if this token's decoded text overlaps
            # with the evidence region
            running_text = tokenizer.decode(input_ids.flatten()[:i+1])
            if len(running_text) > char_start and len(running_text) - len(token_str) < char_end:
                positions.append(i)

        return torch.tensor(positions, dtype=torch.long)
