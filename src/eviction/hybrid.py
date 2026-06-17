"""Hybrid eviction — combining attention + geometry signals.

Supports two modes:
1. Score interpolation: score = λ·norm(attn) + (1-λ)·norm(geom)
2. Budget split (set union): S = S_attn ∪ S_geom ∪ S_recent ∪ S_sink
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple
import time
import torch
from src.eviction.base import BaseEviction
from src.eviction.score_normalization import normalize_scores
from src.eviction.kv_utils import (
    to_legacy_cache, back_to_original, get_kv_seq_len,
    gather_by_dim, mean_heads,
)


def _min_max_norm(x: torch.Tensor) -> torch.Tensor:
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)


class HybridEviction(BaseEviction):
    """Hybrid: combine attention-based and geometry-based signals.

    Args:
        hybrid_mode: "interpolation" or "budget_split"
        lambda_attn: interpolation weight for attention (0–1)
        attn_budget_ratio: fraction of budget for attention top-k (budget_split)
        geom_budget_ratio: fraction for geometry top-k (budget_split)
        recent_budget_ratio: fraction for recent window (budget_split)
        sink_budget_ratio: fraction for sink tokens (budget_split)
        geometry_method: "l1", "l2", "norm", "key_norm", "value_norm"
        attention_method: "accumulated"
    """
    name = "hybrid"
    method_family = "hybrid"
    requires_attention = True
    requires_scores = True
    score_source = "hybrid"

    def __init__(
        self,
        hybrid_mode="budget_split",
        lambda_attn=0.5,
        attn_budget_ratio=0.3,
        geom_budget_ratio=0.3,
        recent_budget_ratio=0.3,
        sink_budget_ratio=0.1,
        geometry_method="l1",
        attention_method="accumulated",
        score_normalization="rank",
        components=None,
        score_source="v",
        sketch_dim=1024,
        update_interval=32,
        update_policy="every_n_steps",
        seed=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hybrid_mode = hybrid_mode
        self.lambda_attn = float(lambda_attn)
        self.attn_budget_ratio = float(attn_budget_ratio)
        self.geom_budget_ratio = float(geom_budget_ratio)
        self.recent_budget_ratio = float(recent_budget_ratio)
        self.sink_budget_ratio = float(sink_budget_ratio)
        self.geometry_method = geometry_method
        self.attention_method = attention_method
        self.score_normalization = str(score_normalization or "rank").lower()
        self.components = components or []
        self.score_source = score_source
        self.sketch_dim = sketch_dim
        self.update_interval = max(0, int(update_interval))
        self.update_policy = str(update_policy or "every_n_steps").lower()
        if self.update_interval == 0 and self.update_policy == "every_n_steps":
            self.update_policy = "prefill_only"
        self.seed = seed
        self._acc_scores: dict[int, torch.Tensor] = {}
        self._geom_estimators: dict[int, object] = {}
        self._last_geom_scores: dict[int, torch.Tensor] = {}
        self._fit_layers: set[int] = set()
        self._component_sources_current: dict[int, dict[int, list[str]]] = {}
        self.last_component_sources: dict[int, dict[str, list[str]]] = {}
        self.score_update_count = 0

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        seq_len = layer_k.size(self.k_seq_dim)
        if self.components:
            return self._compute_weighted_components(layer_k, layer_v, layer_idx, seq_len)
        attn_scores = self._compute_attn_scores(layer_k, layer_v, layer_idx, seq_len)
        geom_scores = self._compute_geom_scores(layer_k, layer_v, layer_idx, seq_len)
        if geom_scores is not None:
            self._last_geom_scores[layer_idx] = geom_scores.detach()
        if self.hybrid_mode == "interpolation":
            return self._interpolate(attn_scores, geom_scores, seq_len)
        return geom_scores  # budget_split uses geom scores directly

    def _compute_weighted_components(self, k, v, layer_idx, seq_len):
        merged = None
        total_weight = 0.0
        for component in self.components:
            name = str(component.get("name", "")).lower()
            weight = float(component.get("weight", 1.0))
            score = self._component_score(name, k, v, layer_idx, seq_len)
            if score is None or weight == 0:
                continue
            normed = normalize_scores(score[:seq_len].float(), self.score_normalization)
            merged = normed * weight if merged is None else merged + normed * weight
            total_weight += abs(weight)
        if merged is None:
            return None
        return merged / max(total_weight, 1e-8)

    def _component_score(self, name, k, v, layer_idx, seq_len):
        if name in ("attention", "accumulated_attention", "h2o"):
            return self._compute_attn_scores(k, v, layer_idx, seq_len)
        if name in ("recency", "position"):
            return torch.arange(seq_len, device=v.device, dtype=torch.float32)
        old = self.geometry_method
        try:
            if name in ("l1", "l1_leverage"):
                self.geometry_method = "l1"
                return self._compute_geom_scores(k, v, layer_idx, seq_len)
            if name in ("l2", "l2_leverage"):
                self.geometry_method = "l2"
                return self._compute_geom_scores(k, v, layer_idx, seq_len)
            if name in ("key_l2_norm", "key_norm"):
                self.geometry_method = "key_norm"
                return self._compute_geom_scores(k, v, layer_idx, seq_len)
            if name in ("value_l2_norm", "value_norm", "norm"):
                self.geometry_method = "value_norm"
                return self._compute_geom_scores(k, v, layer_idx, seq_len)
        finally:
            self.geometry_method = old
        return None

    def update_attention(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        if attention_weights is None:
            return
        attn = attention_weights.detach()
        if attn.dim() == 4:
            pooled = attn[:, :, -1, :].mean(dim=(0, 1))
        elif attn.dim() == 3:
            pooled = attn[:, -1, :].mean(dim=0)
        else:
            return
        seq_len = pooled.numel()
        prev = self._acc_scores.get(layer_idx)
        if prev is None or prev.numel() < seq_len:
            new_prev = torch.zeros(seq_len, device=pooled.device, dtype=pooled.dtype)
            if prev is not None:
                new_prev[: prev.numel()] = prev.to(new_prev.device, dtype=new_prev.dtype)
            prev = new_prev
        prev[:seq_len] += pooled.to(prev.device)
        self._acc_scores[layer_idx] = prev

    def select_indices(self, scores, seq_len, budget, device):
        if self.hybrid_mode == "interpolation":
            return self._select_interpolation(scores, seq_len, budget, device)
        return self._select_budget_split(seq_len, budget, device, geom_scores=scores)

    def __call__(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        pkv, original = to_legacy_cache(past_key_values)
        if pkv is None:
            return None
        items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, (k, v) in enumerate(pkv):
            seq_len = get_kv_seq_len(k, self.k_seq_dim)
            self._ensure_position_map(layer_idx, seq_len, k.device)
            if seq_len <= self.cache_size:
                self.last_selected[layer_idx] = self._position_maps[layer_idx].detach().cpu()
                items.append((k, v))
                continue
            t0 = time.perf_counter()
            scores = self.compute_scores(k, v, layer_idx)
            self.profile_times["score_compute"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            if self.hybrid_mode == "interpolation":
                keep = self._select_interpolation(scores, seq_len, self.cache_size, k.device)
            else:
                keep = self._select_budget_split(
                    seq_len,
                    self.cache_size,
                    k.device,
                    layer_idx=layer_idx,
                    geom_scores=scores,
                )
            self.profile_times["topk_select"] += time.perf_counter() - t0
            if self.debug_budget:
                from src.eviction.base import validate_selected_indices
                validate_selected_indices(keep.detach().cpu(), seq_len, self.cache_size)
            selected_positions = self._record_selected_positions(
                layer_idx, keep, k.device
            ).detach().cpu()
            self.last_selected[layer_idx] = selected_positions
            current_sources = self._component_sources_current.get(layer_idx, {})
            if current_sources:
                self.last_component_sources[layer_idx] = {
                    str(int(orig)): current_sources.get(int(cur), ["unknown"])
                    for cur, orig in zip(keep.detach().cpu().tolist(), selected_positions.tolist())
                }
            if scores is not None:
                self.last_scores[layer_idx] = scores.detach().cpu()
            t0 = time.perf_counter()
            new_k = gather_by_dim(k, self.k_seq_dim, keep.to(k.device))
            new_v = gather_by_dim(v, self.v_seq_dim, keep.to(v.device))
            self.profile_times["cache_prune"] += time.perf_counter() - t0
            items.append((new_k, new_v))
        self._steps += 1
        return back_to_original(original, items)

    # ── Attention scores ────────────────────────────────────────────────

    def _compute_attn_scores(self, k, v, layer_idx, seq_len):
        import shared_q
        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is None:
            return None
        k_rows = mean_heads(k, self.k_seq_dim)
        if k_rows is None:
            return None
        head_dim = v.shape[-1]
        q_vec = q_h.mean(dim=0).to(v.device, dtype=torch.float32)
        k_rows = k_rows.to(v.device, dtype=torch.float32)
        if k_rows.shape[-1] != q_vec.numel():
            return None
        logits = q_vec @ k_rows.T / max(head_dim ** 0.5, 1e-6)
        attn = torch.softmax(logits, dim=0)
        prev = self._acc_scores.get(layer_idx)
        if prev is None or prev.numel() < seq_len:
            new_prev = torch.zeros(seq_len, device=k.device, dtype=attn.dtype)
            if prev is not None:
                new_prev[: prev.numel()] = prev
            prev = new_prev
        prev[:seq_len] += attn.to(prev.device)
        self._acc_scores[layer_idx] = prev
        return prev[:seq_len]

    # ── Geometry scores ─────────────────────────────────────────────────

    def _compute_geom_scores(self, k, v, layer_idx, seq_len):
        if self.geometry_method == "l1":
            rows = self._get_score_rows(k, v)
            if rows is None:
                return None
            est = self._get_geom_estimator(layer_idx)
            force = self._should_refit_geom(layer_idx)
            if force:
                self._fit_layers.add(layer_idx)
                self.score_update_count += 1
            return est.scores(rows, force_refit=force)
        elif self.geometry_method == "l2":
            rows = self._get_score_rows(k, v)
            if rows is None:
                return None
            from src.eviction.l2_leverage import l2_row_leverage_scores
            return l2_row_leverage_scores(rows)
        elif self.geometry_method == "recency":
            return torch.arange(seq_len, device=v.device, dtype=torch.float32)
        else:  # "norm", "key_norm", "value_norm"
            if "key" in self.geometry_method:
                rows = mean_heads(k, self.k_seq_dim)
            else:
                rows = mean_heads(v, self.v_seq_dim)
            if rows is None:
                return None
            return torch.norm(rows.float(), p=2, dim=1)

    def _get_score_rows(self, k, v):
        v_rows = mean_heads(v, self.v_seq_dim)
        if v_rows is None or self.score_source == "v":
            return v_rows
        k_rows = mean_heads(k, self.k_seq_dim)
        if k_rows is None or k_rows.shape[0] != v_rows.shape[0]:
            return v_rows
        return torch.cat([k_rows.float(), v_rows.float()], dim=-1)

    def _get_geom_estimator(self, layer_idx):
        if layer_idx not in self._geom_estimators:
            from src.sketching.woodruff_l1 import WoodruffL1Estimator
            self._geom_estimators[layer_idx] = WoodruffL1Estimator(
                sketch_dim=self.sketch_dim, seed=self.seed + layer_idx,
            )
        return self._geom_estimators[layer_idx]

    def _should_refit_geom(self, layer_idx: int) -> bool:
        if layer_idx not in self._fit_layers:
            return True
        if self.update_policy in ("prefill_only", "never_after_prefill"):
            return False
        if self.update_policy != "every_n_steps":
            return False
        return self.update_interval > 0 and (self._steps % self.update_interval) == 0

    # ── Interpolation mode ──────────────────────────────────────────────

    def _interpolate(self, attn, geom, seq_len):
        if attn is None and geom is None:
            return None
        if attn is None:
            return normalize_scores(geom[:seq_len].float(), self.score_normalization)
        if geom is None:
            return normalize_scores(attn[:seq_len].float(), self.score_normalization)
        a = normalize_scores(attn[:seq_len].float(), self.score_normalization)
        g = normalize_scores(geom[:seq_len].float(), self.score_normalization)
        min_len = min(a.numel(), g.numel())
        return self.lambda_attn * a[:min_len] + (1 - self.lambda_attn) * g[:min_len]

    def _select_interpolation(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        if scores is None:
            return self._fill_budget(reserved, seq_len, budget, device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return self._ensure_budget(reserved, seq_len, budget, device, reserved=reserved)
        masked = scores[:seq_len].clone().to(device)
        if reserved.numel() > 0:
            masked[reserved] = -float("inf")
        valid = torch.isfinite(masked)
        if not valid.any():
            return self._fill_budget(reserved, seq_len, budget, device)
        topk = min(fill, int(valid.sum().item()))
        idx = torch.topk(masked, topk).indices
        return self._ensure_budget(
            torch.cat([reserved, idx]),
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )

    # ── Budget split mode ───────────────────────────────────────────────

    def _select_budget_split(self, seq_len, budget, device, layer_idx=None, geom_scores=None):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        sink_b = max(1, int(budget * self.sink_budget_ratio))
        recent_b = max(1, int(budget * self.recent_budget_ratio))
        attn_b = max(1, int(budget * self.attn_budget_ratio))
        geom_b = max(0, int(budget * self.geom_budget_ratio))

        parts = []
        source_map: dict[int, list[str]] = {}
        # Sink
        if sink_b > 0:
            sink_idx = torch.arange(0, min(sink_b, seq_len), device=device)
            parts.append(sink_idx)
            for idx in sink_idx.tolist():
                source_map.setdefault(int(idx), []).append("sink")
        # Recent
        if recent_b > 0:
            start = max(0, seq_len - recent_b)
            recent_idx = torch.arange(start, seq_len, device=device)
            parts.append(recent_idx)
            for idx in recent_idx.tolist():
                source_map.setdefault(int(idx), []).append("recent")

        reserved = torch.cat(parts).unique(sorted=True) if parts else torch.empty(0, dtype=torch.long, device=device)
        selected = reserved

        def take_top(score_vec, take, current, source_name):
            if score_vec is None or take <= 0:
                return current
            if score_vec.numel() < seq_len:
                return current
            masked = score_vec[:seq_len].clone().to(device=device, dtype=torch.float32)
            if current.numel() > 0:
                masked[current] = -float("inf")
            valid = torch.isfinite(masked)
            if not valid.any():
                return current
            topk = min(take, int(valid.sum().item()))
            idx = torch.topk(masked, topk).indices
            for token_idx in idx.detach().cpu().tolist():
                source_map.setdefault(int(token_idx), []).append(source_name)
            return torch.cat([current, idx]).unique(sorted=True)

        remaining = max(0, budget - int(selected.numel()))
        attn_take = min(attn_b, remaining)
        attn_scores = self._acc_scores.get(layer_idx) if layer_idx is not None else None
        selected = take_top(attn_scores, attn_take, selected, "attention")

        remaining = max(0, budget - int(selected.numel()))
        geom_take = min(geom_b, remaining)
        if geom_scores is None and layer_idx is not None:
            geom_scores = self._last_geom_scores.get(layer_idx)
        selected = take_top(geom_scores, geom_take, selected, "geometry")

        combined = None
        if attn_scores is not None and attn_scores.numel() >= seq_len:
            combined = _min_max_norm(attn_scores[:seq_len].float().to(device))
        if geom_scores is not None and geom_scores.numel() >= seq_len:
            geom_norm = _min_max_norm(geom_scores[:seq_len].float().to(device))
            combined = geom_norm if combined is None else combined + geom_norm

        keep = self._ensure_budget(
            selected,
            seq_len,
            budget,
            device,
            scores=combined,
            reserved=reserved,
        )
        if layer_idx is not None:
            for idx in keep.detach().cpu().tolist():
                source_map.setdefault(int(idx), ["fill"])
            self._component_sources_current[int(layer_idx)] = source_map
        return keep

    def reset(self):
        super().reset()
        self._acc_scores.clear()
        self._geom_estimators.clear()
        self._last_geom_scores.clear()
        self._fit_layers.clear()
        self._component_sources_current.clear()
        self.last_component_sources.clear()
        self.score_update_count = 0
