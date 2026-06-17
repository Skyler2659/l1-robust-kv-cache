"""Case study export — find and export attention-fail / L1-succeed examples.

Identifies specific samples where:
- Attention-based eviction fails but L1-based succeeds (L1 wins)
- L1-based eviction fails but attention-based succeeds (attention wins)
- Both fail or both succeed

Exports detailed token-level analysis for paper figures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import torch


@dataclass
class CaseStudy:
    """A single case study example."""
    sample_idx: int
    category: str  # "l1_wins", "attn_wins", "both_win", "both_fail"
    attn_ppl: float
    l1_ppl: float
    full_ppl: float
    attn_selected_text: List[str] = field(default_factory=list)
    l1_selected_text: List[str] = field(default_factory=list)
    evidence_text: List[str] = field(default_factory=list)
    attn_kept_evidence: int = 0
    l1_kept_evidence: int = 0
    n_evidence: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CaseStudyExporter:
    """Find and export case studies comparing attention vs L1 eviction.

    Usage:
        exporter = CaseStudyExporter(tokenizer=tokenizer)
        cases = exporter.find_cases(
            results_by_method={"attention": attn_results, "l1": l1_results},
            input_ids=input_ids,
            evidence_positions=evidence_pos,
        )
        exporter.export_json(cases, "output/case_studies.json")
    """

    def __init__(self, tokenizer=None, max_tokens_display: int = 20):
        self.tokenizer = tokenizer
        self.max_tokens_display = max_tokens_display

    def find_cases(
        self,
        results_by_method: Dict[str, List],
        input_ids: torch.Tensor,
        evidence_positions: Optional[torch.Tensor] = None,
        selected_by_method: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
        threshold: float = 0.5,
    ) -> List[CaseStudy]:
        """Find case studies from benchmark results.

        Args:
            results_by_method: method_name -> list of BenchmarkResult
            input_ids: [1, seq_len] or list of input_ids per sample
            evidence_positions: [n_evidence] tensor (if available)
            selected_by_method: method_name -> {layer_idx -> selected indices}
            threshold: PPL ratio threshold for "winning"

        Returns:
            List of CaseStudy instances
        """
        if "attention" not in results_by_method and "h2o" not in results_by_method:
            attn_key = next(iter(results_by_method.keys()), None)
        else:
            attn_key = "attention" if "attention" in results_by_method else "h2o"

        l1_key = None
        for k in results_by_method:
            if "l1" in k.lower():
                l1_key = k
                break
        if l1_key is None:
            l1_key = list(results_by_method.keys())[-1]

        attn_results = results_by_method.get(attn_key, [])
        l1_results = results_by_method.get(l1_key, [])

        cases: List[CaseStudy] = []
        evidence_set = set()
        if evidence_positions is not None:
            evidence_set = set(evidence_positions.tolist())

        for si in range(min(len(attn_results), len(l1_results))):
            attn_ppl = attn_results[si].metrics.get("ppl", float("inf"))
            l1_ppl = l1_results[si].metrics.get("ppl", float("inf"))
            full_ppl = min(attn_ppl, l1_ppl)  # approximate

            # Classify
            attn_ratio = attn_ppl / max(full_ppl, 1e-8)
            l1_ratio = l1_ppl / max(full_ppl, 1e-8)

            if l1_ratio < attn_ratio / threshold and l1_ratio < threshold * 2:
                category = "l1_wins"
            elif attn_ratio < l1_ratio / threshold and attn_ratio < threshold * 2:
                category = "attn_wins"
            elif attn_ratio < threshold * 2 and l1_ratio < threshold * 2:
                category = "both_win"
            else:
                category = "both_fail"

            # Decode selected tokens
            attn_text = []
            l1_text = []
            evidence_text = []
            attn_kept_ev = 0
            l1_kept_ev = 0

            if selected_by_method and self.tokenizer is not None:
                attn_sel = set()
                l1_sel = set()
                for layer_idx, indices in selected_by_method.get(attn_key, {}).items():
                    attn_sel |= set(indices.tolist())
                for layer_idx, indices in selected_by_method.get(l1_key, {}).items():
                    l1_sel |= set(indices.tolist())

                ids_flat = input_ids[si].flatten() if input_ids.dim() > 1 else input_ids.flatten()
                for idx in sorted(attn_sel)[:self.max_tokens_display]:
                    if idx < ids_flat.numel():
                        attn_text.append(self.tokenizer.decode(ids_flat[idx:idx+1]))
                for idx in sorted(l1_sel)[:self.max_tokens_display]:
                    if idx < ids_flat.numel():
                        l1_text.append(self.tokenizer.decode(ids_flat[idx:idx+1]))
                for idx in sorted(evidence_set)[:self.max_tokens_display]:
                    if idx < ids_flat.numel():
                        evidence_text.append(self.tokenizer.decode(ids_flat[idx:idx+1]))

                attn_kept_ev = len(attn_sel & evidence_set)
                l1_kept_ev = len(l1_sel & evidence_set)

            case = CaseStudy(
                sample_idx=si,
                category=category,
                attn_ppl=attn_ppl,
                l1_ppl=l1_ppl,
                full_ppl=full_ppl,
                attn_selected_text=attn_text,
                l1_selected_text=l1_text,
                evidence_text=evidence_text,
                attn_kept_evidence=attn_kept_ev,
                l1_kept_evidence=l1_kept_ev,
                n_evidence=len(evidence_set),
                metadata=attn_results[si].metadata if si < len(attn_results) else {},
            )
            cases.append(case)

        # Sort: l1_wins first, then attn_wins, etc.
        priority = {"l1_wins": 0, "attn_wins": 1, "both_win": 2, "both_fail": 3}
        cases.sort(key=lambda c: (priority.get(c.category, 99), c.sample_idx))
        return cases

    def export_json(self, cases: List[CaseStudy], path: str) -> None:
        """Export case studies to JSON."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = []
        for c in cases:
            data.append({
                "sample_idx": c.sample_idx,
                "category": c.category,
                "attn_ppl": c.attn_ppl,
                "l1_ppl": c.l1_ppl,
                "full_ppl": c.full_ppl,
                "attn_selected_text": c.attn_selected_text,
                "l1_selected_text": c.l1_selected_text,
                "evidence_text": c.evidence_text,
                "attn_kept_evidence": c.attn_kept_evidence,
                "l1_kept_evidence": c.l1_kept_evidence,
                "n_evidence": c.n_evidence,
                "metadata": c.metadata,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[CaseStudy] exported {len(cases)} cases to {path}")

    def summary(self, cases: List[CaseStudy]) -> Dict[str, int]:
        """Count cases by category."""
        counts: Dict[str, int] = {}
        for c in cases:
            counts[c.category] = counts.get(c.category, 0) + 1
        return counts
