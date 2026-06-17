"""Plot generation for paper figures — publication-quality matplotlib plots.

Generates:
- Budget sweep curves (PPL vs. cache budget)
- Overlap / correlation heatmaps
- Restoration curves
- Evidence recall bar charts
- Deletion impact curves
- NIAH depth heatmaps
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# Publication style constants
PAPER_STYLE = {
    "figure.figsize": (6, 4),
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

METHOD_COLORS = {
    "full": "#333333",
    "random": "#AAAAAA",
    "recency": "#888888",
    "sink_recent": "#999999",
    "uniform": "#BBBBBB",
    "attention": "#E41A1C",
    "h2o": "#FF7F00",
    "snapkv": "#984EA3",
    "pyramidkv": "#A65628",
    "rocketkv": "#F781BF",
    "key_norm": "#4DAF4A",
    "value_norm": "#4DAF4A",
    "kv_norm": "#4DAF4A",
    "l2_leverage": "#377EB8",
    "l1_leverage": "#1F78B4",
    "l1_exact": "#08519C",
    "l1_cauchy": "#6BAED6",
    "hybrid_interpolation": "#E7298A",
    "hybrid_budget_split": "#CE1256",
    "clustering": "#66C2A5",
    "pca_residual": "#FC8D62",
}

METHOD_MARKERS = {
    "full": "s",
    "random": "x",
    "recency": "+",
    "uniform": "d",
    "attention": "o",
    "h2o": "^",
    "snapkv": "v",
    "l1_leverage": "D",
    "l2_leverage": "p",
    "hybrid_interpolation": "*",
    "hybrid_budget_split": "P",
}


class PlotGenerator:
    """Generate publication-quality plots for the paper.

    Usage:
        pg = PlotGenerator(output_dir="figures/")
        pg.budget_sweep(results, "budget_sweep.png")
        pg.restoration_curve(results, "restoration.png")
    """

    def __init__(
        self,
        output_dir: str = "figures",
        style: Optional[Dict] = None,
        format: str = "pdf",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self._style = style or PAPER_STYLE

    def _setup_plt(self):
        import matplotlib.pyplot as plt
        plt.rcParams.update(self._style)
        return plt

    def _save(self, fig, name: str, formats: Optional[List[str]] = None):
        fmts = formats or [self.format, "png"]
        paths = []
        for fmt in fmts:
            p = self.output_dir / f"{name}.{fmt}"
            fig.savefig(str(p), format=fmt, bbox_inches="tight")
            paths.append(str(p))
        return paths[0]

    def budget_sweep(
        self,
        results: Dict[str, Dict[float, float]],
        title: str = "PPL vs. Cache Budget",
        xlabel: str = "Cache Budget (%)",
        ylabel: str = "Perplexity",
        name: str = "budget_sweep",
        log_y: bool = False,
    ) -> str:
        """Plot PPL curves as a function of cache budget.

        Args:
            results: method_name -> {budget_fraction: ppl}
            title: plot title
            xlabel, ylabel: axis labels
            name: output file name (without extension)
            log_y: use log scale for y-axis

        Returns:
            Path to saved figure
        """
        plt = self._setup_plt()
        fig, ax = plt.subplots(figsize=(6, 4))

        for method, data in sorted(results.items()):
            budgets = sorted(data.keys())
            ppls = [data[b] for b in budgets]
            color = METHOD_COLORS.get(method, None)
            marker = METHOD_MARKERS.get(method, "o")
            label = method.replace("_", " ").title()
            ax.plot(budgets, ppls, marker=marker, label=label,
                    color=color, markersize=4, linewidth=1.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def restoration_curve(
        self,
        results: List,
        title: str = "PPL Recovery vs. Tokens Added Back",
        name: str = "restoration",
    ) -> str:
        """Plot PPL recovery curves.

        Args:
            results: list of RestorationResult objects
        """
        plt = self._setup_plt()
        fig, ax = plt.subplots(figsize=(6, 4))

        for r in results:
            color = METHOD_COLORS.get(r.method, None)
            marker = METHOD_MARKERS.get(r.method, "o")
            label = r.method.replace("_", " ").title()
            ax.plot(r.n_tokens_added, r.ppl_recovery,
                    marker=marker, label=label, color=color,
                    markersize=4, linewidth=1.5)

        ax.set_xlabel("Tokens Added Back")
        ax.set_ylabel("PPL Recovery (0=baseline, 1=full)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def deletion_impact(
        self,
        results: List,
        title: str = "PPL Impact of Token Deletion",
        name: str = "deletion_impact",
    ) -> str:
        """Plot PPL ratio as a function of deletion fraction."""
        plt = self._setup_plt()
        fig, ax = plt.subplots(figsize=(6, 4))

        for r in results:
            color = METHOD_COLORS.get(r.method.split("_del")[0], None)
            marker = METHOD_MARKERS.get(r.method.split("_del")[0], "o")
            frac = r.n_deleted / max(r.seq_len, 1)
            ax.scatter([frac], [r.ppl_ratio], marker=marker,
                       color=color, s=60, zorder=5)

        ax.set_xlabel("Fraction of Tokens Deleted")
        ax.set_ylabel("PPL Ratio (deleted / original)")
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def evidence_recall_bar(
        self,
        results: List,
        title: str = "Evidence Token Recall by Method",
        name: str = "evidence_recall",
    ) -> str:
        """Bar chart of evidence recall across methods."""
        plt = self._setup_plt()
        fig, ax = plt.subplots(figsize=(8, 4))

        methods = [r.method for r in results]
        recalls = [r.evidence_recall for r in results]
        colors = [METHOD_COLORS.get(m, "#666666") for m in methods]
        labels = [m.replace("_", " ").title() for m in methods]

        bars = ax.barh(labels, recalls, color=colors, edgecolor="white")
        ax.set_xlabel("Evidence Recall")
        ax.set_title(title)
        ax.set_xlim(0, 1)

        for bar, val in zip(bars, recalls):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def rank_correlation_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str = "Rank Correlation (Spearman ρ)",
        name: str = "rank_correlation",
    ) -> str:
        """Heatmap of pairwise rank correlation."""
        plt = self._setup_plt()
        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        ax.set_yticklabels([l.replace("_", " ").title() for l in labels], fontsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(matrix[i, j]) > 0.5 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def context_length_sweep(
        self,
        results: Dict[str, Dict[int, float]],
        title: str = "PPL vs. Context Length",
        name: str = "context_length_sweep",
    ) -> str:
        """Plot PPL vs context length for different methods."""
        plt = self._setup_plt()
        fig, ax = plt.subplots(figsize=(6, 4))

        for method, data in sorted(results.items()):
            lengths = sorted(data.keys())
            ppls = [data[l] for l in lengths]
            color = METHOD_COLORS.get(method, None)
            marker = METHOD_MARKERS.get(method, "o")
            label = method.replace("_", " ").title()
            ax.plot(lengths, ppls, marker=marker, label=label,
                    color=color, markersize=4, linewidth=1.5)

        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Perplexity")
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path
