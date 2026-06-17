"""Heatmap generation for paper figures — depth × accuracy, layer × score.

Generates:
- Needle depth × accuracy heatmap
- Layer × score distribution heatmap
- Overlap heatmap
- Attention weight vs L1 score 2D histogram
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class HeatmapGenerator:
    """Generate heatmap figures for the paper.

    Usage:
        hg = HeatmapGenerator(output_dir="figures/")
        hg.needle_depth_heatmap(depth_results, "niah_depth.png")
    """

    def __init__(
        self,
        output_dir: str = "figures",
        format: str = "pdf",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

    def _save(self, fig, name: str):
        for fmt in [self.format, "png"]:
            p = self.output_dir / f"{name}.{fmt}"
            fig.savefig(str(p), format=fmt, bbox_inches="tight")
        return str(self.output_dir / f"{name}.{self.format}")

    def needle_depth_heatmap(
        self,
        results: Dict[str, Dict[float, float]],
        title: str = "Needle-in-a-Haystack: Depth × Method",
        name: str = "niah_depth_heatmap",
    ) -> str:
        """Heatmap of PPL (or accuracy) by needle depth and method.

        Args:
            results: method_name -> {depth: metric_value}
            title: plot title
            name: output file name

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt

        methods = sorted(results.keys())
        depths = sorted(set(d for data in results.values() for d in data.keys()))

        matrix = np.full((len(methods), len(depths)), np.nan)
        for mi, method in enumerate(methods):
            for di, depth in enumerate(depths):
                if depth in results[method]:
                    matrix[mi, di] = results[method][depth]

        fig, ax = plt.subplots(figsize=(10, max(3, len(methods) * 0.5)))
        im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto",
                        interpolation="nearest")

        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([f"{d:.1f}" for d in depths], fontsize=8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([m.replace("_", " ").title() for m in methods], fontsize=8)
        ax.set_xlabel("Needle Depth (0=beginning, 1=end)")
        ax.set_ylabel("Method")

        for mi in range(len(methods)):
            for di in range(len(depths)):
                val = matrix[mi, di]
                if not np.isnan(val):
                    text_color = "white" if val > np.nanmean(matrix) else "black"
                    ax.text(di, mi, f"{val:.1f}", ha="center", va="center",
                            fontsize=6, color=text_color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="PPL (lower is better)")
        ax.set_title(title)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def layer_score_heatmap(
        self,
        scores_by_layer: Dict[int, np.ndarray],
        title: str = "Importance Score Distribution by Layer",
        name: str = "layer_score_heatmap",
        n_bins: int = 50,
    ) -> str:
        """Heatmap of score distribution across layers.

        Args:
            scores_by_layer: layer_idx -> [n_tokens] score array
            title: plot title
            name: output file name
            n_bins: number of histogram bins

        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt

        layers = sorted(scores_by_layer.keys())
        if not layers:
            return ""

        # Build 2D histogram
        all_scores = np.concatenate([scores_by_layer[l] for l in layers])
        score_min, score_max = np.percentile(all_scores, [1, 99])
        bins = np.linspace(score_min, score_max, n_bins + 1)

        heatmap = np.zeros((len(layers), n_bins))
        for li, layer in enumerate(layers):
            scores = np.clip(scores_by_layer[layer], score_min, score_max)
            hist, _ = np.histogram(scores, bins=bins, density=True)
            heatmap[li] = hist

        fig, ax = plt.subplots(figsize=(8, max(4, len(layers) * 0.3)))
        im = ax.imshow(heatmap, cmap="viridis", aspect="auto",
                        origin="lower", interpolation="nearest")

        ax.set_xticks([0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1])
        ax.set_xticklabels([f"{bins[i]:.3f}" for i in
                           [0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1]],
                           fontsize=7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=7)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Layer")

        fig.colorbar(im, ax=ax, shrink=0.8, label="Density")
        ax.set_title(title)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def overlap_heatmap(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str = "Token Set Overlap (Jaccard)",
        name: str = "overlap_heatmap",
    ) -> str:
        """Heatmap of pairwise Jaccard overlap between methods."""
        import matplotlib.pyplot as plt

        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.45)))
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        ax.set_yticklabels([l.replace("_", " ").title() for l in labels], fontsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard Index")
        ax.set_title(title)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def attention_vs_l1_scatter(
        self,
        attn_scores: np.ndarray,
        l1_scores: np.ndarray,
        title: str = "Attention Saliency vs. L1 Leverage",
        name: str = "attn_vs_l1_scatter",
        n_bins: int = 100,
    ) -> str:
        """2D histogram of attention scores vs L1 leverage scores.

        Shows the joint distribution — if the two signals are independent,
        the histogram should be roughly uniform. Clustering along the
        diagonal would indicate redundancy.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        # Normalize to [0, 1] for visualization
        attn_norm = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-8)
        l1_norm = (l1_scores - l1_scores.min()) / (l1_scores.max() - l1_scores.min() + 1e-8)

        ax.hist2d(attn_norm, l1_norm, bins=n_bins, cmap="viridis",
                  range=[[0, 1], [0, 1]])

        ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y = x")
        ax.set_xlabel("Attention Saliency (normalized)")
        ax.set_ylabel("L1 Leverage Score (normalized)")
        ax.set_title(title)
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path

    def budget_task_heatmap(
        self,
        results: Dict[str, Dict[str, Dict[float, float]]],
        title: str = "PPL by Budget × Task × Method",
        name: str = "budget_task_heatmap",
    ) -> str:
        """Heatmap of PPL across budgets and tasks for a fixed method.

        Args:
            results: task_name -> {budget: ppl}
        """
        import matplotlib.pyplot as plt

        tasks = sorted(results.keys())
        if not tasks:
            return ""

        budgets = sorted(set(b for data in results.values() for b in data.keys()))
        matrix = np.full((len(tasks), len(budgets)), np.nan)
        for ti, task in enumerate(tasks):
            for bi, budget in enumerate(budgets):
                if budget in results[task]:
                    matrix[ti, bi] = results[task][budget]

        fig, ax = plt.subplots(figsize=(max(8, len(budgets) * 0.4), max(4, len(tasks) * 0.4)))
        im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto", interpolation="nearest")

        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{b:.0%}" for b in budgets], fontsize=8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=8)
        ax.set_xlabel("Cache Budget")
        ax.set_ylabel("Task")

        fig.colorbar(im, ax=ax, shrink=0.8, label="PPL")
        ax.set_title(title)
        fig.tight_layout()
        path = self._save(fig, name)
        plt.close(fig)
        return path
