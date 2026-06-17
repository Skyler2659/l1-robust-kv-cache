#!/usr/bin/env python3
"""Post-hoc analysis runner for saved benchmark results."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ExperimentConfig, load_analysis_config
from src.utils.io import load_results, load_scores, save_results
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("run_analysis")


def _load_selected(result: Dict[str, Any]) -> Optional[Dict[int, torch.Tensor]]:
    path = result.get("selected_tokens_path")
    if path:
        p = Path(path)
        if not p.exists():
            return None
        data = load_results(p) if p.suffix == ".json" else load_scores(p)
    else:
        data = result.get("selected_tokens") or result.get("selected_tokens_by_layer")
        if not data:
            return None
    return {int(k): torch.tensor(v, dtype=torch.long) if not isinstance(v, torch.Tensor) else v for k, v in data.items()}


def _load_score_dict(result: Dict[str, Any]) -> Optional[Dict[int, torch.Tensor]]:
    path = result.get("scores_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = load_results(p) if p.suffix == ".json" else load_scores(p)
    return {int(k): torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in data.items()}


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix(path: Path, labels: List[str], matrix: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + labels)
        for left in labels:
            writer.writerow([left] + [matrix.get(left, {}).get(right, 0.0) for right in labels])


def run_analysis(results: List[Dict], cfg: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_results: Dict[str, Any] = {}
    filtered_results = [
        r
        for r in results
        if "error" not in r
        and not r.get("skipped")
        and (cfg.analysis.include_oracle or not r.get("oracle"))
    ]
    skipped = [
        {
            "method": r.get("method"),
            "sample_id": r.get("sample_id", r.get("sample_idx")),
            "budget": r.get("budget"),
            "skipped_reason": r.get("skipped_reason") or r.get("unsupported_reason"),
        }
        for r in results
        if r.get("skipped")
    ]
    if skipped:
        save_results(skipped, analysis_dir / "skipped_methods.json")
        analysis_results["skipped"] = skipped

    analysis_results["tables"] = analyze_metric_tables(filtered_results, analysis_dir)

    if cfg.analysis.overlap:
        logger.info("Running overlap analysis")
        analysis_results["overlap"] = analyze_overlap(filtered_results, analysis_dir)

    if cfg.analysis.rank_correlation:
        logger.info("Running rank correlation analysis")
        analysis_results["rank_correlation"] = analyze_rank_correlation(filtered_results, analysis_dir)

    if cfg.analysis.evidence_recall:
        logger.info("Running evidence recall analysis")
        analysis_results["evidence_recall"] = analyze_evidence_recall(filtered_results, analysis_dir)

    if cfg.analysis.case_study:
        logger.info("Exporting case studies")
        analysis_results["case_study"] = analyze_case_studies(
            filtered_results, analysis_dir, cfg.analysis.case_study_count
        )

    save_results(analysis_results, analysis_dir / "analysis_summary.json")
    return analysis_results


def analyze_overlap(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.overlap import OverlapAnalyzer

    analyzer = OverlapAnalyzer()
    grouped = defaultdict(dict)
    for result in results:
        selected = _load_selected(result)
        if selected is None:
            continue
        key = (result.get("sample_id", result.get("sample_idx")), result.get("budget"))
        grouped[key][result["method"]] = selected

    pair_values = defaultdict(list)
    method_names = set()
    for methods in grouped.values():
        names = sorted(methods)
        method_names.update(names)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = analyzer.pairwise_overlap(methods[left], methods[right], left, right)
                pair_values[f"{left}_vs_{right}"].append(overlap.jaccard)

    summary = {pair: _mean(vals) for pair, vals in pair_values.items()}
    labels = sorted(method_names)
    matrix = {m: {n: (1.0 if m == n else 0.0) for n in labels} for m in labels}
    for pair, value in summary.items():
        left, right = pair.split("_vs_", 1)
        matrix.setdefault(left, {})[right] = value
        matrix.setdefault(right, {})[left] = value
    _write_matrix(analysis_dir / "overlap_matrix.csv", labels, matrix)
    save_results(summary, analysis_dir / "overlap_summary.json")
    if summary:
        return summary
    return {"note": "need at least two methods with selected tokens"} if labels else {"note": "no selected token files"}


def analyze_rank_correlation(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.rank_correlation import RankCorrelationAnalyzer

    analyzer = RankCorrelationAnalyzer()
    grouped = defaultdict(dict)
    for result in results:
        scores = _load_score_dict(result)
        if scores is None:
            continue
        key = (result.get("sample_id", result.get("sample_idx")), result.get("budget"))
        grouped[key][result["method"]] = scores

    pair_values = defaultdict(list)
    method_names = set()
    for methods in grouped.values():
        names = sorted(methods)
        method_names.update(names)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                corr = analyzer.layer_wise(methods[left], methods[right], left, right)
                pair_values[f"{left}_vs_{right}"].append(corr.spearman)

    summary = {pair: _mean(vals) for pair, vals in pair_values.items()}
    labels = sorted(method_names)
    matrix = {m: {n: (1.0 if m == n else 0.0) for n in labels} for m in labels}
    for pair, value in summary.items():
        left, right = pair.split("_vs_", 1)
        matrix.setdefault(left, {})[right] = value
        matrix.setdefault(right, {})[left] = value
    _write_matrix(analysis_dir / "rank_correlation.csv", labels, matrix)
    save_results(summary, analysis_dir / "rank_correlation_summary.json")
    if summary:
        return summary
    return {"note": "need at least two methods with score files"} if labels else {"note": "no score files"}


def analyze_evidence_recall(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.evidence_recall import EvidenceRecallAnalyzer

    analyzer = EvidenceRecallAnalyzer()
    values = defaultdict(list)
    details = []
    for result in results:
        selected = _load_selected(result)
        evidence = result.get("evidence_positions") or []
        if selected is None or not evidence:
            continue
        recall = analyzer.analyze(
            selected_indices=selected,
            evidence_positions=torch.tensor(evidence, dtype=torch.long),
            seq_len=int(result.get("context_length", 0)),
            method_name=result["method"],
        )
        values[result["method"]].append(recall.evidence_recall)
        details.append(asdict(recall))

    summary = {method: _mean(vals) for method, vals in values.items()}
    _write_rows(analysis_dir / "evidence_recall.csv", details)
    save_results(details, analysis_dir / "evidence_recall_details.json")
    save_results(summary, analysis_dir / "evidence_recall_summary.json")
    return summary or {"note": "no evidence positions or selected token files"}


def analyze_metric_tables(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    rows = [r for r in results if "error" not in r and not r.get("skipped")]
    by_method_budget = defaultdict(list)
    by_method_context = defaultdict(list)
    by_method_budget_model = defaultdict(list)
    by_method_model = defaultdict(list)
    for r in rows:
        model = r.get("model_name") or r.get("model")
        by_method_budget[(r.get("method"), r.get("budget"))].append(r)
        by_method_context[(r.get("method"), r.get("context_length"))].append(r)
        by_method_budget_model[(model, r.get("method"), r.get("budget"))].append(r)
        by_method_model[(model, r.get("method"))].append(r)

    def summarize(grouped):
        out = []
        for key, vals in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
            row = {
                    "value": key[-1],
                    "n": len(vals),
                    "accuracy": _mean([1.0 if v.get("correct") else 0.0 for v in vals]),
                    "contains_ground_truth": _mean(
                        [1.0 if v.get("contains_ground_truth") else 0.0 for v in vals]
                    ),
                    "avg_official_score": _mean(
                        [
                            float(v.get("official_score"))
                            for v in vals
                            if v.get("official_score") is not None
                        ]
                    ),
                    "official_accuracy": _mean(
                        [
                            1.0 if v.get("official_correct") else 0.0
                            for v in vals
                            if v.get("official_correct") is not None
                        ]
                    ),
                    "avg_primary_score": _mean(
                        [
                            float(v.get("primary_score"))
                            for v in vals
                            if v.get("primary_score") is not None
                        ]
                    ),
                    "avg_ppl": _mean(
                        [float(v.get("ppl")) for v in vals if v.get("ppl") is not None]
                    ),
                    "avg_evidence_recall": _mean(
                        [
                            float(v.get("evidence_recall"))
                            for v in vals
                            if v.get("evidence_recall") is not None
                        ]
                    ),
                    "avg_tokens_per_second": _mean(
                        [
                            float(v.get("tokens_per_second"))
                            for v in vals
                            if v.get("tokens_per_second") is not None
                        ]
                    ),
            }
            if len(key) == 2:
                row["method"] = key[0]
            elif len(key) == 3:
                row["model_name"] = key[0]
                row["method"] = key[1]
            out.append(row)
        return out

    mb = summarize(by_method_budget)
    mc = summarize(by_method_context)
    mbm = summarize(by_method_budget_model)
    mm = []
    for (model, method), vals in sorted(by_method_model.items(), key=lambda item: tuple(str(x) for x in item[0])):
        mm.append(
            {
                "model_name": model,
                "method": method,
                "n": len(vals),
                "accuracy": _mean([1.0 if v.get("correct") else 0.0 for v in vals]),
                "avg_official_score": _mean(
                    [
                        float(v.get("official_score"))
                        for v in vals
                        if v.get("official_score") is not None
                    ]
                ),
                "official_accuracy": _mean(
                    [
                        1.0 if v.get("official_correct") else 0.0
                        for v in vals
                        if v.get("official_correct") is not None
                    ]
                ),
                "avg_primary_score": _mean(
                    [
                        float(v.get("primary_score"))
                        for v in vals
                        if v.get("primary_score") is not None
                    ]
                ),
                "avg_evidence_recall": _mean(
                    [
                        float(v.get("evidence_recall"))
                        for v in vals
                        if v.get("evidence_recall") is not None
                    ]
                ),
                "avg_tokens_per_second": _mean(
                    [
                        float(v.get("tokens_per_second"))
                        for v in vals
                        if v.get("tokens_per_second") is not None
                    ]
                ),
            }
        )
    for row in mb:
        row["budget"] = row.pop("value")
    for row in mc:
        row["context_length"] = row.pop("value")
    for row in mbm:
        row["budget"] = row.pop("value")
    _write_rows(analysis_dir / "method_budget_accuracy.csv", mb)
    _write_rows(analysis_dir / "method_context_accuracy.csv", mc)
    _write_rows(analysis_dir / "model_method_budget_metrics.csv", mbm)
    _write_rows(analysis_dir / "model_method_metrics.csv", mm)
    return {
        "method_budget_accuracy_csv": str(analysis_dir / "method_budget_accuracy.csv"),
        "method_context_accuracy_csv": str(analysis_dir / "method_context_accuracy.csv"),
        "model_method_budget_metrics_csv": str(analysis_dir / "model_method_budget_metrics.csv"),
        "model_method_metrics_csv": str(analysis_dir / "model_method_metrics.csv"),
    }


def analyze_case_studies(
    results: List[Dict], analysis_dir: Path, count: int = 5
) -> Dict[str, Any]:
    by_sample = defaultdict(list)
    for result in results:
        if "error" not in result and not result.get("skipped"):
            by_sample[result.get("sample_id", result.get("sample_idx"))].append(result)

    cases = []
    for sample_id, sample_results in by_sample.items():
        if len(sample_results) < 2:
            continue
        ranked = sorted(
            sample_results,
            key=lambda r: float(r.get("ppl")) if r.get("ppl") is not None else float("inf"),
        )
        best = ranked[0]
        worst = ranked[-1]
        cases.append(
            {
                "sample_id": sample_id,
                "category": "best_vs_worst",
                "best_method": best.get("method"),
                "best_ppl": best.get("ppl"),
                "worst_method": worst.get("method"),
                "worst_ppl": worst.get("ppl"),
                "ground_truth": best.get("ground_truth"),
                "evidence_positions": best.get("evidence_positions"),
                "best_selected_tokens_path": best.get("selected_tokens_path"),
                "worst_selected_tokens_path": worst.get("selected_tokens_path"),
            }
        )

    cases = cases[:count]
    save_results(cases, analysis_dir / "case_studies.json")
    return {"num_cases": len(cases), "path": str(analysis_dir / "case_studies.json")}


def _apply_analysis_filter(cfg: ExperimentConfig, selected: str) -> None:
    if selected == "all":
        return
    enabled = {name.strip() for name in selected.split(",") if name.strip()}
    cfg.analysis.overlap = "overlap" in enabled
    cfg.analysis.rank_correlation = "rank_correlation" in enabled or "rank_corr" in enabled
    cfg.analysis.evidence_recall = "evidence_recall" in enabled
    cfg.analysis.case_study = "case_study" in enabled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--results_dir", "--run-dir", dest="input", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Analysis fragment or full config")
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        help="Comma-separated: overlap,rank_correlation,evidence_recall,case_study",
    )
    args = parser.parse_args()
    setup_logging()

    results_dir = Path(args.input)
    results = load_results(results_dir / "results.json")
    cfg = ExperimentConfig.from_yaml(results_dir / "config.yaml")
    if args.config:
        cfg.analysis = load_analysis_config(args.config)
    _apply_analysis_filter(cfg, args.analysis)

    summary = run_analysis(results, cfg, results_dir)
    logger.info("Analysis complete: %s", summary)
    logger.info("Analysis outputs saved in %s", results_dir / "analysis")


if __name__ == "__main__":
    main()
