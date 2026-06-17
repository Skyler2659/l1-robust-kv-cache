#!/usr/bin/env python3
"""Export case studies — find and visualize attention-fail / L1-succeed cases.

Usage:
    python scripts/export_cases.py --results_dir results/default/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ExperimentConfig
from src.utils.io import load_results, save_results
from src.utils.logging_utils import setup_logging, get_logger

logger = get_logger("export_cases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--baseline", type=str, default="attention")
    parser.add_argument("--improved", type=str, default="l1_mixed")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="improved_ppl < baseline_ppl * threshold")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--format", type=str, default="json", choices=["json", "markdown"])
    args = parser.parse_args()
    setup_logging()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir / "results.json")

    from collections import defaultdict
    by_sample = defaultdict(list)
    for r in results:
        by_sample[r.get("sample_idx", 0)].append(r)

    cases = []
    for sample_idx, sample_results in sorted(by_sample.items()):
        baseline = next(
            (r for r in sample_results if r.get("method") == args.baseline), None
        )
        improved = next(
            (r for r in sample_results if r.get("method") == args.improved), None
        )
        if not baseline or not improved:
            continue
        if "error" in baseline or "error" in improved:
            continue

        b_ppl = baseline.get("ppl", float("inf"))
        i_ppl = improved.get("ppl", float("inf"))

        if i_ppl < b_ppl * args.threshold:
            cases.append({
                "sample_idx": sample_idx,
                "baseline_method": args.baseline,
                "baseline_ppl": b_ppl,
                "improved_method": args.improved,
                "improved_ppl": i_ppl,
                "improvement_ratio": b_ppl / i_ppl if i_ppl > 0 else float("inf"),
                "ground_truth": baseline.get("ground_truth"),
                "evidence_positions": baseline.get("evidence_positions"),
                "baseline_metrics": {k: v for k, v in baseline.items()
                                      if k not in ("evidence_positions", "ground_truth")},
                "improved_metrics": {k: v for k, v in improved.items()
                                      if k not in ("evidence_positions", "ground_truth")},
            })

    # Sort by improvement ratio
    cases.sort(key=lambda c: c["improvement_ratio"], reverse=True)
    cases = cases[: args.max_cases]

    out_dir = results_dir / "case_studies"
    out_dir.mkdir(exist_ok=True)

    if args.format == "json":
        save_results(cases, out_dir / "cases.json")
    else:
        # Markdown export
        md_lines = ["# Case Studies\n"]
        md_lines.append(f"Baseline: {args.baseline} → Improved: {args.improved}\n")
        for i, case in enumerate(cases):
            md_lines.append(f"\n## Case {i+1} (sample {case['sample_idx']})\n")
            md_lines.append(f"- Baseline PPL: {case['baseline_ppl']:.4f}")
            md_lines.append(f"- Improved PPL: {case['improved_ppl']:.4f}")
            md_lines.append(f"- Improvement: {case['improvement_ratio']:.2f}x")
            if case.get("ground_truth"):
                md_lines.append(f"- Ground truth: {case['ground_truth']}")
            md_lines.append("")

        with open(out_dir / "cases.md", "w") as f:
            f.write("\n".join(md_lines))

    logger.info(f"Exported {len(cases)} case studies to {out_dir}")


if __name__ == "__main__":
    main()
