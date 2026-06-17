#!/usr/bin/env python3
"""Generate standard figures from a benchmark run directory."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.io import load_results
from src.visualization.heatmaps import HeatmapGenerator
from src.visualization.plots import PlotGenerator


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "None"):
            return default
        return float(value)
    except Exception:
        return default


def load_matrix(path: Path):
    rows = read_csv_rows(path)
    if not rows:
        return None, []
    labels = [r["method"] for r in rows]
    matrix = np.zeros((len(labels), len(labels)))
    for i, row in enumerate(rows):
        for j, label in enumerate(labels):
            matrix[i, j] = to_float(row.get(label))
    return matrix, labels


def evidence_recall_enabled(run_dir: Path) -> bool:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return True
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return True
    analysis = cfg.get("analysis") or {}
    return bool(analysis.get("evidence_recall", True))


def plot_accuracy_by_budget(run_dir: Path, fig_dir: Path) -> str:
    rows = read_csv_rows(run_dir / "analysis" / "method_budget_accuracy.csv")
    data: Dict[str, Dict[float, float]] = defaultdict(dict)
    for row in rows:
        data[row["method"]][to_float(row.get("budget"))] = to_float(row.get("accuracy"))
    if not data:
        return ""
    return PlotGenerator(str(fig_dir), format="png").budget_sweep(
        data,
        title="Accuracy by Cache Budget",
        xlabel="Cache Budget (tokens)",
        ylabel="Accuracy",
        name="accuracy_by_budget",
    )


def plot_metric_by_method_budget(
    run_dir: Path,
    fig_dir: Path,
    metric: str,
    title: str,
    ylabel: str,
    name: str,
) -> str:
    rows = read_csv_rows(run_dir / "analysis" / "method_budget_accuracy.csv")
    data: Dict[str, Dict[float, float]] = defaultdict(dict)
    for row in rows:
        data[row["method"]][to_float(row.get("budget"))] = to_float(row.get(metric))
    if not data:
        return ""
    return PlotGenerator(str(fig_dir), format="png").budget_sweep(
        data,
        title=title,
        xlabel="Cache Budget (tokens)",
        ylabel=ylabel,
        name=name,
    )


def plot_latency_by_method_budget(run_dir: Path, fig_dir: Path) -> str:
    rows = read_csv_rows(run_dir / "metrics.csv")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("skipped")).lower() == "true":
            continue
        grouped[row.get("method", "unknown")][to_float(row.get("budget"))].append(
            to_float(row.get("total_time_s"))
        )
    data = {
        method: {budget: sum(vals) / len(vals) for budget, vals in budgets.items()}
        for method, budgets in grouped.items()
    }
    if not data:
        return ""
    return PlotGenerator(str(fig_dir), format="png").budget_sweep(
        data,
        title="Latency by Cache Budget",
        xlabel="Cache Budget (tokens)",
        ylabel="Total Time (s)",
        name="latency_by_method_budget",
    )


def plot_method_budget_heatmap(
    run_dir: Path,
    fig_dir: Path,
    metric: str,
    title: str,
    name: str,
) -> str:
    rows = read_csv_rows(run_dir / "analysis" / "method_budget_accuracy.csv")
    if not rows:
        return ""
    methods = sorted({r["method"] for r in rows})
    budgets = sorted({to_float(r.get("budget")) for r in rows})
    matrix = np.full((len(methods), len(budgets)), np.nan)
    for row in rows:
        i = methods.index(row["method"])
        j = budgets.index(to_float(row.get("budget")))
        matrix[i, j] = to_float(row.get(metric), np.nan)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(budgets) * 0.7), max(4, len(methods) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace("_", " ") for m in methods], fontsize=7)
    ax.set_xlabel("Budget")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_model_method_heatmap(
    run_dir: Path,
    fig_dir: Path,
    metric: str,
    title: str,
    name: str,
) -> str:
    rows = read_csv_rows(run_dir / "analysis" / "model_method_metrics.csv")
    if not rows:
        rows = read_csv_rows(run_dir / "metrics.csv")
    if not rows:
        return ""
    grouped = defaultdict(list)
    for row in rows:
        model = row.get("model_name") or row.get("model") or "model"
        method = row.get("method", "method")
        value = row.get(metric)
        if value in (None, "", "None"):
            continue
        grouped[(model, method)].append(to_float(value))
    if not grouped:
        return ""
    models = sorted({k[0] for k in grouped})
    methods = sorted({k[1] for k in grouped})
    matrix = np.full((len(models), len(methods)), np.nan)
    for (model, method), values in grouped.items():
        matrix[models.index(model), methods.index(method)] = sum(values) / len(values)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 0.7), max(3, len(models) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_latency(run_dir: Path, fig_dir: Path) -> str:
    rows = read_csv_rows(run_dir / "metrics.csv")
    if not rows:
        return ""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("skipped")).lower() == "true":
            continue
        method = row.get("method", "unknown")
        for key in ("prefill_time_s", "decode_time_s", "eviction_time_s", "score_time_s", "topk_time_s"):
            grouped[method][key].append(to_float(row.get(key)))
    import matplotlib.pyplot as plt

    methods = sorted(grouped)
    parts = ["prefill_time_s", "decode_time_s", "eviction_time_s", "score_time_s", "topk_time_s"]
    bottoms = np.zeros(len(methods))
    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 0.8), 4))
    for part in parts:
        vals = np.array([sum(grouped[m][part]) / max(1, len(grouped[m][part])) for m in methods])
        ax.bar(methods, vals, bottom=bottoms, label=part.replace("_time_s", ""))
        bottoms += vals
    ax.set_ylabel("Seconds")
    ax.set_title("Latency Breakdown")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = fig_dir / "latency_breakdown.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def plot_evidence_recall_by_depth(run_dir: Path, fig_dir: Path) -> str:
    rows = read_csv_rows(run_dir / "metrics.csv")
    data = defaultdict(dict)
    counts = defaultdict(lambda: defaultdict(list))
    for row in rows:
        depth = row.get("needle_depth")
        if depth in (None, "", "None"):
            continue
        counts[row["method"]][round(to_float(depth), 3)].append(to_float(row.get("evidence_recall")))
    for method, by_depth in counts.items():
        for depth, vals in by_depth.items():
            data[method][depth] = sum(vals) / len(vals)
    if not data:
        return ""
    return HeatmapGenerator(str(fig_dir), format="png").needle_depth_heatmap(
        data,
        title="Evidence Recall by Needle Depth",
        name="evidence_recall_heatmap",
    )


def plot_overlap(run_dir: Path, fig_dir: Path) -> str:
    matrix, labels = load_matrix(run_dir / "analysis" / "overlap_matrix.csv")
    if matrix is None:
        return ""
    return HeatmapGenerator(str(fig_dir), format="png").overlap_heatmap(
        matrix,
        labels,
        title="Method Selected-Token Overlap",
        name="method_overlap_heatmap",
    )


def plot_rank(run_dir: Path, fig_dir: Path) -> str:
    matrix, labels = load_matrix(run_dir / "analysis" / "rank_correlation.csv")
    if matrix is None:
        return ""
    return PlotGenerator(str(fig_dir), format="png").rank_correlation_matrix(
        matrix,
        labels,
        title="Rank Correlation",
        name="rank_correlation_heatmap",
    )


def plot_selected_positions(run_dir: Path, fig_dir: Path) -> str:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return ""
    results = load_results(results_path)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for result in results:
        selected = result.get("selected_tokens") or result.get("selected_tokens_by_layer")
        if not selected:
            path = result.get("selected_tokens_path")
            if path and Path(path).exists() and Path(path).suffix == ".json":
                selected = load_results(path)
        if not selected:
            continue
        union = sorted({int(x) for vals in selected.values() for x in vals})
        if not union:
            continue
        ax.hist(
            union,
            bins=40,
            alpha=0.35,
            label=f"{result.get('method')}_b{result.get('budget')}",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return ""
    ax.set_xlabel("Original Token Position")
    ax.set_ylabel("Selected Count")
    ax.set_title("Selected Token Position Distribution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / "selected_token_position_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def write_case_study_markdown(run_dir: Path) -> str:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return ""
    results = load_results(results_path)
    out = run_dir / "analysis" / "case_study.md"
    lines = ["# Case Studies", ""]
    for result in results[:10]:
        lines.extend(
            [
                f"## sample={result.get('sample_id')} method={result.get('method')} budget={result.get('budget')}",
                "",
                f"- ground_truth: `{result.get('ground_truth')}`",
                f"- official_score: `{result.get('official_score')}`",
                f"- official_metric: `{result.get('official_metric_name')}`",
                f"- contains_ground_truth: `{result.get('contains_ground_truth')}`",
                f"- evidence_recall: `{result.get('evidence_recall')}`",
                f"- prediction: `{str(result.get('prediction') or '')[:300]}`",
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", "--input", dest="run_dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    use_evidence_recall = evidence_recall_enabled(run_dir)

    outputs = {
        "accuracy_by_budget": plot_accuracy_by_budget(run_dir, fig_dir),
        "accuracy_by_method_budget": plot_metric_by_method_budget(
            run_dir,
            fig_dir,
            "accuracy",
            "Accuracy by Cache Budget",
            "Accuracy",
            "accuracy_by_method_budget",
        ),
        "evidence_recall_by_method_budget": (
            plot_metric_by_method_budget(
                run_dir,
                fig_dir,
                "avg_evidence_recall",
                "Evidence Recall by Cache Budget",
                "Evidence Recall",
                "evidence_recall_by_method_budget",
            )
            if use_evidence_recall
            else ""
        ),
        "official_score_by_method_budget": plot_metric_by_method_budget(
            run_dir,
            fig_dir,
            "avg_official_score",
            "Official Score by Cache Budget",
            "Official Score",
            "official_score_by_method_budget",
        ),
        "latency_by_method_budget": plot_latency_by_method_budget(run_dir, fig_dir),
        "method_budget_heatmap": plot_method_budget_heatmap(
            run_dir,
            fig_dir,
            "accuracy",
            "Method x Budget Accuracy",
            "method_budget_heatmap",
        ),
        "official_score_heatmap": plot_method_budget_heatmap(
            run_dir,
            fig_dir,
            "avg_official_score",
            "Method x Budget Official Score",
            "official_score_heatmap",
        ),
        "evidence_recall_heatmap": (
            plot_evidence_recall_by_depth(run_dir, fig_dir)
            if use_evidence_recall
            else ""
        ),
        "method_overlap_heatmap": plot_overlap(run_dir, fig_dir),
        "rank_correlation_heatmap": plot_rank(run_dir, fig_dir),
        "latency_breakdown": plot_latency(run_dir, fig_dir),
        "selected_token_position_distribution": plot_selected_positions(run_dir, fig_dir),
        "model_method_accuracy_heatmap": plot_model_method_heatmap(
            run_dir,
            fig_dir,
            "accuracy",
            "Model x Method Accuracy",
            "model_method_accuracy_heatmap",
        ),
        "model_method_official_score_heatmap": plot_model_method_heatmap(
            run_dir,
            fig_dir,
            "avg_official_score",
            "Model x Method Official Score",
            "model_method_official_score_heatmap",
        ),
        "model_method_evidence_recall_heatmap": (
            plot_model_method_heatmap(
                run_dir,
                fig_dir,
                "avg_evidence_recall",
                "Model x Method Evidence Recall",
                "model_method_evidence_recall_heatmap",
            )
            if use_evidence_recall
            else ""
        ),
        "case_study_markdown": write_case_study_markdown(run_dir),
    }
    (run_dir / "figures" / "figures_summary.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
