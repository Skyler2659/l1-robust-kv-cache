"""MLX/MLX-LM 4-bit benchmark runner."""
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.benchmarks.factory import load_benchmark
from src.config import ExperimentConfig
from src.eviction.registry import (
    get_method_spec,
    method_requires_attention,
    method_supports_backend,
    unsupported_reason,
    canonicalize_method as registry_canonicalize_method,
)
from src.eviction.score_normalization import list_stats
from src.evaluation.official_metrics import evaluate_official
from src.model_adapters import apply_prompt_format, build_model_adapter
from src.runners.base import BaseRunner, text_hash
from src.utils.io import save_results


SUPPORTED_MLX_METHODS = {
    "full",
    "basic",
    "basic_generate",
    "recency",
    "sink_recent",
    "sink_recency",
    "streamingllm",
    "random",
    "sink_recent_random",
    "attention",
    "accumulated_attention",
    "windowed_attention",
    "attention_decay",
    "h2o",
    "h2o_style",
    "snap",
    "snapkv",
    "snapkv_style",
    "approximate_snapkv",
    "pyramidkv",
    "pyramidkv_style",
    "layer_budget_attention",
    "l1",
    "l1_leverage",
    "l1_prefill_only",
    "l1_decode_only",
    "l2",
    "l2_leverage",
    "l2_prefill_only",
    "l2_key_prefill_only",
    "l2_decode_only",
    "compactor",
    "compactor_style",
    "compactor_l2_attention",
    "key_l2_norm",
    "value_l2_norm",
    "key_l1_norm",
    "value_l1_norm",
    "key_norm",
    "value_norm",
    "sink_recent_l1",
    "sink_recent_l2",
    "attention+l1",
    "attention_l1",
    "attn_l1",
    "hybrid",
    "attention+l2",
    "attention_l2",
    "attn_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "budget_split_hybrid",
    "compactor",
    "sink_recent_attention_l1",
    "oracle_evidence",
    "oracle_answer_region",
}

ATTENTION_SCORE_METHODS = {"attention", "h2o", "windowed_attention", "attention_decay"}
SNAP_METHODS = {"snap", "snapkv"}
PREFILL_COMPRESS_METHODS = {"snapkv", "pyramidkv", "compactor"}
COMPACTORLIKE_HYBRID_METHODS = {"attention_l1_compactor", "attention_l2_compactor"}
HYBRID_METHODS = {
    "attention+l1",
    "attention_l1",
    "attn_l1",
    "hybrid",
    "attention+l2",
    "attention_l2",
    "attn_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_l1",
    "attention_l2",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "budget_split_hybrid",
}
MANUAL_COMPACT_METHODS = {
    "attention",
    "windowed_attention",
    "attention_decay",
    "h2o",
    "snapkv",
    "pyramidkv",
    "l1_leverage",
    "l1_prefill_only",
    "l1_decode_only",
    "l2_leverage",
    "l2_prefill_only",
    "l2_key_prefill_only",
    "l2_decode_only",
    "compactor",
    "key_l2_norm",
    "value_l2_norm",
    "key_l1_norm",
    "value_l1_norm",
    "random",
    "sink_recent_random",
    "sink_recent_l1",
    "sink_recent_l2",
    "attention_l1",
    "attention_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "budget_split_hybrid",
    "oracle_evidence",
    "oracle_answer_region",
}
METHODS_NEED_ATTENTION = ATTENTION_SCORE_METHODS | SNAP_METHODS | HYBRID_METHODS
METHODS_NEED_ATTENTION = METHODS_NEED_ATTENTION | PREFILL_COMPRESS_METHODS


def canonical_method(method: str) -> str:
    try:
        return registry_canonicalize_method(method)[0]
    except Exception:
        method = method.lower().replace("-", "_")
        if method == "snap":
            return "snapkv"
        return method


def normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def answer_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_text(prediction).split()
    gold = normalize_text(ground_truth).split()
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def tensor_to_list(value: Any) -> List[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    if hasattr(value, "tolist"):
        raw = value.tolist()
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return [int(x) for x in raw]
    return [int(x) for x in value]


def token_type(token_text: str) -> str:
    stripped = token_text.strip()
    if not stripped:
        return "whitespace"
    if stripped.isdigit():
        return "number"
    if any(ch.isdigit() for ch in stripped):
        return "alphanumeric"
    if stripped.isalpha():
        return "word"
    if all(not ch.isalnum() for ch in stripped):
        return "punctuation"
    return "mixed"


def _minmax_mx(values):
    import mlx.core as mx

    if values is None:
        return None
    vals = values.astype(mx.float32)
    if vals.shape[0] == 0:
        return vals
    lo = mx.min(vals)
    hi = mx.max(vals)
    return mx.where((hi - lo) > 1e-8, (vals - lo) / (hi - lo), mx.zeros_like(vals))


def _normalize_mx(values, mode: str):
    import mlx.core as mx

    vals = values.astype(mx.float32)
    mode = str(mode or "none").lower()
    if vals.shape[0] == 0 or mode == "none":
        return vals
    if mode == "minmax":
        return _minmax_mx(vals)
    if mode == "zscore":
        mean = mx.mean(vals)
        std = mx.maximum(mx.std(vals), 1e-8)
        return (vals - mean) / std
    if mode == "softmax":
        return mx.softmax(vals, axis=0)
    if mode == "rank":
        order = [int(x) for x in mx.argsort(vals).tolist()]
        denom = max(1, int(vals.shape[0]) - 1)
        ranks = np.zeros(int(vals.shape[0]), dtype=np.float32)
        for rank, idx in enumerate(order):
            ranks[idx] = float(rank) / float(denom)
        return mx.array(ranks)
    return vals


def _merge_score_vectors(attn, geom, lambda_attn: float, normalization: str = "rank"):
    import mlx.core as mx

    if attn is None and geom is None:
        return None
    if attn is None:
        return _normalize_mx(geom, normalization)
    if geom is None:
        return _normalize_mx(attn, normalization)
    n = min(int(attn.shape[0]), int(geom.shape[0]))
    if n <= 0:
        return None
    a = _normalize_mx(attn[:n], normalization)
    g = _normalize_mx(geom[:n], normalization)
    return float(lambda_attn) * a + (1.0 - float(lambda_attn)) * g


def _zscore_mx(values):
    import mlx.core as mx

    vals = values.astype(mx.float32)
    mean = mx.mean(vals)
    std = mx.maximum(mx.std(vals), 1e-8)
    return (vals - mean) / std


def _record_compactor_prefill_tensors(attn_module: Any, q_pre: Any, k_pre: Any, q_post: Any, k_post: Any) -> None:
    """Store full-prompt Q/K chunks needed by faithful Compactor scoring."""
    state = getattr(attn_module, "_l1kv_attention_state", None)
    layer_idx = getattr(attn_module, "_l1kv_layer_idx", None)
    if state is None or layer_idx is None:
        return
    if not state.get("enabled", False) or state.get("current_method") != "compactor":
        return
    if state.get("phase") != "prefill":
        return
    try:
        lid = int(layer_idx)
        state.setdefault("prefill_q_post", {}).setdefault(lid, []).append(q_post.astype(q_post.dtype))
        state.setdefault("prefill_k_post", {}).setdefault(lid, []).append(k_post.astype(k_post.dtype))
        state.setdefault("prefill_k_pre", {}).setdefault(lid, []).append(k_pre.astype(k_pre.dtype))
    except Exception:
        state.setdefault("hook_errors", 0)
        state["hook_errors"] += 1


def _record_attention_from_hook(
    attn_module: Any,
    queries: Any,
    keys: Any,
    query_len: Optional[int] = None,
) -> None:
    """Pool causal current-query attention over heads and store it in runner state.

    SnapKV needs attention from the final observation-window queries to prior
    keys, not just the last query of a prefill chunk. We therefore record up to
    ``max_observe`` recent query rows and apply the within-chunk causal mask
    before pooling over heads.
    """
    import mlx.core as mx

    state = getattr(attn_module, "_l1kv_attention_state", None)
    layer_idx = getattr(attn_module, "_l1kv_layer_idx", None)
    if state is None or layer_idx is None:
        return
    if not state.get("enabled", False):
        return
    if keys is None or queries is None or keys.shape[-2] == 0:
        return
    try:
        q_total = int(queries.shape[-2])
        k_total = int(keys.shape[-2])
        if q_total <= 0 or k_total <= 0:
            return
        max_observe = max(1, int(state.get("max_observe", 32)))
        q_take = min(q_total, max_observe)
        q_recent = queries[:, :, q_total - q_take :, :].astype(mx.float32)
        k = keys.astype(mx.float32)
        n_q_heads = int(q_recent.shape[1])
        n_kv_heads = int(k.shape[1])
        if n_q_heads != n_kv_heads and n_q_heads % max(1, n_kv_heads) == 0:
            repeats = n_q_heads // n_kv_heads
            q = q_recent.reshape(
                q_recent.shape[0],
                n_kv_heads,
                repeats,
                q_take,
                q_recent.shape[-1],
            )
            logits = mx.sum(q[:, :, :, :, None, :] * k[:, :, None, None, :, :], axis=-1)
            logits = logits * float(getattr(attn_module, "scale", 1.0))
            if query_len is not None and int(query_len) > 1:
                q_abs = mx.arange(q_total - q_take, q_total)
                allowed = k_total - int(query_len) + q_abs + 1
                key_pos = mx.arange(k_total)
                causal = key_pos.reshape(1, -1) < allowed.reshape(-1, 1)
                logits = mx.where(causal.reshape(1, 1, 1, q_take, k_total), logits, -mx.inf)
            attn = mx.softmax(logits, axis=-1, precise=True)
            head_rows = mx.mean(attn, axis=(0, 2))
            pooled_rows = mx.mean(head_rows, axis=0)
        else:
            logits = mx.sum(q_recent[:, :, :, None, :] * k[:, :, None, :, :], axis=-1)
            logits = logits * float(getattr(attn_module, "scale", 1.0))
            if query_len is not None and int(query_len) > 1:
                q_abs = mx.arange(q_total - q_take, q_total)
                allowed = k_total - int(query_len) + q_abs + 1
                key_pos = mx.arange(k_total)
                causal = key_pos.reshape(1, -1) < allowed.reshape(-1, 1)
                logits = mx.where(causal.reshape(1, 1, q_take, k_total), logits, -mx.inf)
            attn = mx.softmax(logits, axis=-1, precise=True)
            head_rows = mx.mean(attn, axis=0)
            pooled_rows = mx.mean(head_rows, axis=0)

        pooled_rows = pooled_rows.astype(mx.float32)
        head_rows = head_rows.astype(mx.float32)
        pooled = pooled_rows[-1]
        window_sum = mx.sum(pooled_rows, axis=0)
        seq_len = int(pooled.shape[0])
        state.setdefault("last", {})[int(layer_idx)] = pooled

        accumulated = state.setdefault("accumulated", {})
        prev = accumulated.get(int(layer_idx))
        if prev is None or int(prev.shape[0]) < seq_len:
            new_prev = mx.zeros((seq_len,), dtype=pooled.dtype)
            if prev is not None and int(prev.shape[0]) > 0:
                new_prev = mx.concatenate([prev, new_prev[int(prev.shape[0]) :]], axis=0)
            prev = new_prev
        prev = prev + mx.pad(window_sum, [(0, max(0, int(prev.shape[0]) - seq_len))])[: int(prev.shape[0])]
        accumulated[int(layer_idx)] = prev

        decayed = state.setdefault("decayed", {})
        gamma = float(state.get("decay_gamma", 0.95))
        prev_decay = decayed.get(int(layer_idx))
        if prev_decay is None or int(prev_decay.shape[0]) < seq_len:
            new_prev = mx.zeros((seq_len,), dtype=pooled.dtype)
            if prev_decay is not None and int(prev_decay.shape[0]) > 0:
                new_prev = mx.concatenate(
                    [prev_decay, new_prev[int(prev_decay.shape[0]) :]],
                    axis=0,
                )
            prev_decay = new_prev
        prev_decay = prev_decay * gamma
        prev_decay = prev_decay + mx.pad(window_sum, [(0, max(0, int(prev_decay.shape[0]) - seq_len))])[: int(prev_decay.shape[0])]
        decayed[int(layer_idx)] = prev_decay

        observe = state.setdefault("observe", {}).setdefault(int(layer_idx), [])
        observe.extend([pooled_rows[i] for i in range(int(pooled_rows.shape[0]))])
        if len(observe) > max_observe:
            del observe[:-max_observe]
        observe_heads = state.setdefault("observe_heads", {}).setdefault(int(layer_idx), [])
        observe_heads.extend([head_rows[:, i, :] for i in range(int(head_rows.shape[1]))])
        if len(observe_heads) > max_observe:
            del observe_heads[:-max_observe]
    except Exception:
        state.setdefault("hook_errors", 0)
        state["hook_errors"] += 1


def _cache_head_valid_attention_mask(cache: Any, n_query_heads: int, k_len: int):
    import mlx.core as mx

    valid = getattr(cache, "head_valid_mask", None)
    if valid is None:
        return None
    try:
        h_kv = int(valid.shape[0])
        old_len = int(valid.shape[1])
        if old_len < int(k_len):
            pad = mx.ones((h_kv, int(k_len) - old_len), dtype=mx.bool_)
            valid = mx.concatenate([valid, pad], axis=1)
            cache.head_valid_mask = valid
        elif old_len > int(k_len):
            valid = valid[:, : int(k_len)]
            cache.head_valid_mask = valid
        if h_kv <= 0:
            return None
        if int(n_query_heads) == h_kv:
            valid_q = valid
        elif int(n_query_heads) % h_kv == 0:
            valid_q = mx.repeat(valid, int(n_query_heads) // h_kv, axis=0)
        else:
            return None
        return mx.where(
            valid_q[:, None, :],
            mx.zeros((int(n_query_heads), 1, int(k_len)), dtype=mx.float32),
            mx.full((int(n_query_heads), 1, int(k_len)), -mx.inf, dtype=mx.float32),
        )
    except Exception:
        return None


class MLXL1Estimator:
    """Woodruff-style L1 leverage estimator implemented with MLX arrays.

    The exponential reweighting used by L1 sketches is intentionally heavy
    tailed. For KV eviction, a single near-zero random weight can dominate the
    QR factor and produce unstable token rankings, so the MLX runner uses a
    deterministic sketch with a modest weight floor and pseudo-inverse fallback.
    """

    def __init__(
        self,
        sketch_dim: int = 1024,
        seed: int = 0,
        weight_floor: float = 1e-3,
        condition_limit: float = 1e6,
    ):
        self.sketch_dim = int(sketch_dim)
        self.seed = int(seed)
        self.weight_floor = float(weight_floor)
        self.condition_limit = float(condition_limit)
        self.r_inv = None
        self.last_dim = None
        self.fit_count = 0

    def scores(self, rows):
        import mlx.core as mx

        n, d = rows.shape
        if n <= 1:
            return mx.sum(mx.abs(rows.astype(mx.float32)), axis=1)
        rows_f = rows.astype(mx.float32)
        if self.r_inv is None or self.last_dim != d:
            rng = np.random.default_rng(
                self.seed + self.fit_count * 1_000_003 + int(n) * 9176 + int(d)
            )
            self.fit_count += 1
            if n < self.sketch_dim:
                u = mx.array(rng.uniform(1e-8, 1 - 1e-8, size=(n, 1)).astype(np.float32))
                weights = mx.maximum(-mx.log(1.0 - u), self.weight_floor)
                weighted = rows_f / weights
            else:
                buckets = mx.array(
                    rng.integers(0, self.sketch_dim, size=n, dtype=np.int32)
                )
                signs = mx.array(
                    rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
                )
                u = mx.array(rng.uniform(1e-8, 1 - 1e-8, size=(n, 1)).astype(np.float32))
                weights = mx.maximum(-mx.log(1.0 - u), self.weight_floor)
                weighted_rows = rows_f / weights * signs.reshape(-1, 1)
                idx = mx.arange(self.sketch_dim).reshape(-1, 1)
                mask = (buckets.reshape(1, -1) == idx).astype(mx.float32)
                weighted = mask @ weighted_rows
            try:
                with mx.stream(mx.cpu):
                    _, r = mx.linalg.qr(weighted)
                    if r.shape[0] != r.shape[1]:
                        self.r_inv = None
                        self.last_dim = d
                        return mx.sum(mx.abs(rows_f), axis=1)
                    singular_values = mx.linalg.svd(r, compute_uv=False)
                    s_max = float(mx.max(singular_values).item())
                    s_min = float(mx.min(singular_values).item())
                    if not math.isfinite(s_max) or not math.isfinite(s_min) or s_max <= 0:
                        self.r_inv = None
                        self.last_dim = d
                        return mx.sum(mx.abs(rows_f), axis=1)
                    condition = s_max / max(s_min, 1e-8)
                    if condition > self.condition_limit:
                        return mx.sum(mx.abs(rows_f), axis=1)
                    self.r_inv = mx.linalg.pinv(r)
                self.last_dim = d
            except Exception:
                self.r_inv = None
                self.last_dim = d
                return mx.sum(mx.abs(rows_f), axis=1)
        proj = rows_f @ self.r_inv
        return mx.sum(mx.abs(proj), axis=1)


class MLXL2Estimator:
    """L2 row leverage via thin QR.

    MLX's CPU SVD can abort the process on degenerate matrices in some LAPACK
    paths, bypassing Python exception handling. Thin QR gives the same row-space
    leverage for full-rank inputs and degrades safely to row norm on failure.
    """

    def scores(self, rows):
        import mlx.core as mx

        n, _ = rows.shape
        if n <= 1:
            return mx.sum(rows.astype(mx.float32) ** 2, axis=1)
        rows_f = rows.astype(mx.float32)
        try:
            with mx.stream(mx.cpu):
                q, _ = mx.linalg.qr(rows_f)
                if int(q.shape[0]) != int(n):
                    return mx.sum(rows_f ** 2, axis=1)
                return mx.sum(q ** 2, axis=1)
        except Exception:
            return mx.sum(rows_f ** 2, axis=1)


class MLXCacheEvictor:
    """Manual KVCache evictor for non-rotating MLX cache strategies."""

    def __init__(
        self,
        method: str,
        budget: int,
        cfg: ExperimentConfig,
        num_layers: int,
        attention_state: Optional[Dict[str, Any]] = None,
        oracle_positions: Optional[List[int]] = None,
    ):
        self.method = canonical_method(method)
        self.budget = int(budget)
        self.cfg = cfg
        self.num_layers = int(num_layers)
        self.attention_state = attention_state or {
            "last": {},
            "accumulated": {},
            "decayed": {},
            "observe": {},
            "observe_heads": {},
            "prefill_q_post": {},
            "prefill_k_post": {},
            "prefill_k_pre": {},
            "hook_errors": 0,
        }
        self.oracle_positions = sorted({int(x) for x in (oracle_positions or [])})
        self.position_maps: Dict[int, Any] = {}
        self.next_positions: Dict[int, int] = {}
        self.last_selected: Dict[int, List[int]] = {}
        self.last_scores: Dict[int, List[float]] = {}
        self.last_component_sources: Dict[int, Dict[str, List[str]]] = {}
        self.last_selected_by_head: Dict[int, Dict[str, List[int]]] = {}
        self.head_position_maps: Dict[int, Any] = {}
        self._component_sources_current: Dict[int, Dict[int, List[str]]] = {}
        self._last_attn_scores: Dict[int, Any] = {}
        self._last_geom_scores: Dict[int, Any] = {}
        self._static_score_cache: Dict[int, Any] = {}
        self.l1_estimators = {
            i: MLXL1Estimator(
                cfg.eviction.sketch_dim,
                seed=int(getattr(cfg, "seed", 0)) + i * 1009,
            )
            for i in range(num_layers)
        }
        self.l2_estimators = {i: MLXL2Estimator() for i in range(num_layers)}
        self.profile_times = {
            "score_time_s": 0.0,
            "topk_time_s": 0.0,
            "cache_rebuild_time_s": 0.0,
        }
        self.eviction_count = 0
        self.score_update_count = 0
        self.phase = "prefill"
        self.score_phase_counts = {"prefill": 0, "decode": 0}
        self.score_refit_count = 0
        self.score_refit_phase_counts = {"prefill": 0, "decode": 0}

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase or "decode").lower()

    def _record_score_refit(self) -> None:
        phase = self.phase if self.phase in self.score_refit_phase_counts else "decode"
        self.score_refit_count += 1
        self.score_refit_phase_counts[phase] += 1

    def sync_maps(self, cache: List[Any]) -> None:
        import mlx.core as mx

        for layer_idx, c in enumerate(cache):
            seq_len = int(c.offset)
            head_map = getattr(c, "head_position_map", None)
            if head_map is not None:
                h_count = int(head_map.shape[0])
                old_len = int(head_map.shape[1])
                if old_len < seq_len:
                    start = self.next_positions.get(layer_idx, int(getattr(c, "logical_offset", seq_len)) - (seq_len - old_len))
                    extra = mx.arange(start, start + (seq_len - old_len))
                    extra = mx.broadcast_to(extra.reshape(1, -1), (h_count, int(extra.shape[0])))
                    head_map = mx.concatenate([head_map, extra], axis=1)
                    c.head_position_map = head_map
                    self.next_positions[layer_idx] = start + int(extra.shape[1])
                elif old_len > seq_len:
                    head_map = head_map[:, :seq_len]
                    c.head_position_map = head_map
                valid = getattr(c, "head_valid_mask", None)
                if valid is None:
                    valid = mx.ones((h_count, seq_len), dtype=mx.bool_)
                    c.head_valid_mask = valid
                elif int(valid.shape[1]) < seq_len:
                    pad = mx.ones((h_count, seq_len - int(valid.shape[1])), dtype=mx.bool_)
                    valid = mx.concatenate([valid, pad], axis=1)
                    c.head_valid_mask = valid
                elif int(valid.shape[1]) > seq_len:
                    valid = valid[:, :seq_len]
                    c.head_valid_mask = valid
                self.head_position_maps[layer_idx] = head_map
                by_head: Dict[str, List[int]] = {}
                union = set()
                for h in range(h_count):
                    vals = [
                        int(head_map[h, j].item())
                        for j in range(seq_len)
                        if bool(valid[h, j].item()) and int(head_map[h, j].item()) >= 0
                    ]
                    by_head[str(h)] = vals
                    union.update(vals)
                self.last_selected_by_head[layer_idx] = by_head
                self.last_selected[layer_idx] = sorted(union)
                self.position_maps[layer_idx] = mx.array(sorted(union), dtype=mx.int32)
                continue
            current = self.position_maps.get(layer_idx)
            if current is None or len(current) > seq_len:
                self.position_maps[layer_idx] = mx.arange(seq_len)
                self.next_positions[layer_idx] = seq_len
            elif len(current) < seq_len:
                start = self.next_positions.get(layer_idx, len(current))
                extra = mx.arange(start, start + (seq_len - len(current)))
                self.position_maps[layer_idx] = mx.concatenate([current, extra])
                self.next_positions[layer_idx] = start + len(extra)

    def evict_for_space(self, cache: List[Any], num_coming: int = 1) -> None:
        budget = max(1, self.budget - int(num_coming))
        if cache and int(cache[0].offset) > budget:
            self.evict(cache, budget)

    def evict(self, cache: List[Any], budget: Optional[int] = None) -> None:
        import mlx.core as mx

        budget = int(budget or self.budget)
        self.sync_maps(cache)
        for layer_idx, c in enumerate(cache):
            seq_len = int(c.offset)
            layer_budget = self._layer_budget(layer_idx, len(cache), budget)
            if seq_len <= layer_budget:
                self.last_selected[layer_idx] = self._to_int_list(
                    self.position_maps[layer_idx]
                )
                continue
            score_start = time.perf_counter()
            scores = self._compute_scores(c, layer_idx, seq_len)
            self.profile_times["score_time_s"] += time.perf_counter() - score_start
            if scores is not None:
                self.score_update_count += 1
                phase = self.phase if self.phase in self.score_phase_counts else "decode"
                self.score_phase_counts[phase] += 1
                self.last_scores[layer_idx] = self._to_float_list(scores)

            topk_start = time.perf_counter()
            keep = self._select_indices(scores, seq_len, layer_budget, layer_idx)
            self.profile_times["topk_time_s"] += time.perf_counter() - topk_start

            rebuild_start = time.perf_counter()
            c.keys = mx.take(c.keys[:, :, :seq_len, :], keep, axis=2)
            c.values = mx.take(c.values[:, :, :seq_len, :], keep, axis=2)
            c.offset = int(keep.shape[0])
            self._prune_attention_state(layer_idx, keep, seq_len)
            old_position_map = self.position_maps[layer_idx]
            selected_positions = mx.take(old_position_map, keep, axis=0)
            current_sources = self._component_sources_current.get(layer_idx, {})
            if current_sources:
                self.last_component_sources[layer_idx] = {
                    str(int(orig)): current_sources.get(int(cur), ["unknown"])
                    for cur, orig in zip(keep.tolist(), selected_positions.tolist())
                }
            self.position_maps[layer_idx] = selected_positions
            self.last_selected[layer_idx] = self._to_int_list(
                self.position_maps[layer_idx]
            )
            self.profile_times["cache_rebuild_time_s"] += (
                time.perf_counter() - rebuild_start
            )
            self.eviction_count += 1

    def prefill_compress(self, cache: List[Any], budget: Optional[int] = None) -> None:
        import mlx.core as mx

        budget = int(budget or self.budget)
        if self.method not in PREFILL_COMPRESS_METHODS:
            self.evict(cache, budget)
            return
        self.sync_maps(cache)
        for layer_idx, c in enumerate(cache):
            seq_len = int(c.offset)
            if seq_len <= 0:
                continue
            score_start = time.perf_counter()
            if self.method == "compactor":
                keep_by_head, scores_by_head = self._compactor_headwise_keep(c, layer_idx, seq_len, budget)
                layer_budget = max((len(v) for v in keep_by_head), default=0)
            else:
                layer_budget = self._prefill_layer_budget(layer_idx, len(cache), budget, seq_len)
                if seq_len <= layer_budget:
                    keep_by_head = [list(range(seq_len)) for _ in range(int(c.keys.shape[1]))]
                    scores_by_head = None
                else:
                    keep_by_head, scores_by_head = self._snap_pyramid_headwise_keep(
                        layer_idx,
                        seq_len,
                        layer_budget,
                    )
            self.profile_times["score_time_s"] += time.perf_counter() - score_start
            if scores_by_head is not None:
                self.score_update_count += 1
                phase = self.phase if self.phase in self.score_phase_counts else "prefill"
                self.score_phase_counts[phase] += 1
                self._record_head_scores(layer_idx, scores_by_head)
                self._record_score_refit()

            if seq_len <= layer_budget and all(len(v) == seq_len for v in keep_by_head):
                self.last_selected[layer_idx] = list(range(seq_len))
                self.last_selected_by_head[layer_idx] = {
                    str(h): list(range(seq_len)) for h in range(int(c.keys.shape[1]))
                }
                self.position_maps[layer_idx] = mx.arange(seq_len)
                self.next_positions[layer_idx] = int(getattr(c, "logical_offset", seq_len))
                continue

            topk_start = time.perf_counter()
            keep_by_head = [sorted({int(x) for x in values if 0 <= int(x) < seq_len}) for values in keep_by_head]
            self.profile_times["topk_time_s"] += time.perf_counter() - topk_start

            rebuild_start = time.perf_counter()
            self._apply_headwise_keep(c, layer_idx, keep_by_head, seq_len)
            self.profile_times["cache_rebuild_time_s"] += time.perf_counter() - rebuild_start
            self.eviction_count += 1

    def _record_head_scores(self, layer_idx: int, scores_by_head: Any) -> None:
        import mlx.core as mx

        arr = np.asarray(scores_by_head.tolist(), dtype=np.float32)
        finite = np.isfinite(arr)
        fill = float(arr[finite].max() + 1.0) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
        scores = mx.array(arr)
        if len(scores.shape) == 2:
            agg = mx.mean(scores, axis=0)
        else:
            agg = scores
        self.last_scores[layer_idx] = self._to_float_list(agg)

    def _prefill_layer_budget(self, layer_idx: int, num_layers: int, base_budget: int, seq_len: int) -> int:
        if self.method != "pyramidkv" or num_layers <= 1:
            return int(base_budget)
        window = min(max(1, int(getattr(self.cfg.eviction, "window_size", 64))), int(seq_len))
        if seq_len <= base_budget:
            return int(seq_len)
        base = max(1, int(base_budget) - window)
        if seq_len < base * 2:
            return min(seq_len, base + window)
        beta = max(1, int(getattr(self.cfg.eviction, "pyramid_beta", 20)))
        min_num = base // beta
        max_num = base * 2 - min_num
        hist_len = max(0, seq_len - window)
        if max_num >= hist_len:
            max_num = hist_len
            min_num = base * 2 - max_num
        steps = (max_num - min_num) // max(1, int(num_layers) - 1)
        prefix_keep = max(1, max_num - int(layer_idx) * steps)
        return max(1, min(seq_len, prefix_keep + window))

    def _snap_pyramid_headwise_keep(self, layer_idx: int, seq_len: int, budget: int):
        import mlx.core as mx

        obs = min(max(1, int(getattr(self.cfg.eviction, "window_size", 64))), seq_len)
        hist_len = max(0, seq_len - obs)
        rows = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
        usable = [row[:, :seq_len] for row in rows if int(row.shape[-1]) >= seq_len]
        if usable:
            stacked = mx.stack(usable[-obs:], axis=0)
            prefix_scores = mx.sum(stacked[:, :, :hist_len], axis=0).astype(mx.float32)
        else:
            h = int(self._cache_num_heads(layer_idx))
            prefix_scores = mx.broadcast_to(mx.arange(hist_len).astype(mx.float32).reshape(1, -1), (h, hist_len))
        pooled = self._pool_scores_by_head(prefix_scores)
        h_count = int(pooled.shape[0]) if len(pooled.shape) == 2 else int(self._cache_num_heads(layer_idx))
        hist_budget = max(0, int(budget) - obs)
        keep_by_head: List[List[int]] = []
        for h in range(h_count):
            parts: List[int] = []
            if hist_budget > 0 and hist_len > 0:
                score = pooled[h, :hist_len]
                take = min(hist_budget, hist_len)
                if take >= hist_len:
                    idx = mx.arange(hist_len)
                else:
                    idx = mx.argpartition(-score, max(0, take - 1))[:take]
                parts.extend(int(x) for x in idx.tolist())
            parts.extend(range(max(0, seq_len - obs), seq_len))
            keep_by_head.append(sorted(set(parts)))
        recent_fill = mx.array(1.0, dtype=mx.float32)
        if hist_len > 0:
            recent_fill = mx.max(pooled[:, :hist_len]) + 1.0
            prefix_part = pooled[:, :hist_len]
        else:
            prefix_part = mx.zeros((h_count, 0), dtype=mx.float32)
        if obs > 0:
            recent_part = mx.ones((h_count, seq_len - hist_len), dtype=mx.float32) * recent_fill
            full_scores = mx.concatenate([prefix_part, recent_part], axis=1)
        else:
            full_scores = prefix_part
        return keep_by_head, full_scores

    def _pool_scores_by_head(self, scores: Any):
        import mlx.core as mx

        if scores is None:
            return None
        kernel = int(
            getattr(
                self.cfg.eviction,
                "pooling_kernel",
                getattr(self.cfg.eviction, "kernel_size", 1),
            )
            or 1
        )
        if kernel <= 1 or int(scores.shape[-1]) <= 1:
            return scores.astype(mx.float32)
        method = str(getattr(self.cfg.eviction, "pooling_method", "avgpool") or "avgpool").lower()
        values = np.array(scores.tolist(), dtype=np.float32)
        pad = max(0, kernel // 2)
        padded = np.pad(values, ((0, 0), (pad, pad)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=1)[:, : values.shape[1], :]
        if method in {"avg", "mean", "avgpool"}:
            pooled = windows.mean(axis=-1)
        elif method in {"max", "maxpool"}:
            pooled = windows.max(axis=-1)
        else:
            raise ValueError(f"Unsupported pooling_method={method!r}; expected avgpool or maxpool")
        return mx.array(pooled.astype(np.float32))

    def _compactor_headwise_keep(self, c: Any, layer_idx: int, seq_len: int, budget: int):
        import mlx.core as mx

        q_post, k_post, k_pre = self._compactor_prefill_qk(layer_idx, c, seq_len)
        scores = self._compactor_scores(q_post, k_post, k_pre, seq_len)
        h_count = int(scores.shape[1])
        protected_first, protected_last = self._compactor_protected_counts(seq_len)
        scores_np = np.asarray(scores.tolist(), dtype=np.float32)
        if protected_first > 0:
            scores_np[:protected_first, :] = np.inf
        if protected_last > 0:
            scores_np[max(0, seq_len - protected_last) : seq_len, :] = np.inf
        scores = mx.array(scores_np)
        top_k = min(seq_len * h_count, max(1, int(budget) * h_count))
        flat = scores.reshape(-1)
        if top_k >= int(flat.shape[0]):
            chosen = mx.arange(int(flat.shape[0]))
        else:
            chosen = mx.argpartition(-flat, max(0, top_k - 1))[:top_k]
        keep_by_head: List[List[int]] = [[] for _ in range(h_count)]
        for raw in chosen.tolist():
            token = int(raw) // h_count
            head = int(raw) - token * h_count
            if 0 <= token < seq_len:
                keep_by_head[head].append(token)
        return [sorted(set(v)) for v in keep_by_head], mx.transpose(scores, (1, 0))

    def _compactor_prefill_qk(self, layer_idx: int, c: Any, seq_len: int):
        import mlx.core as mx

        def concat_chunks(name: str):
            chunks = self.attention_state.get(name, {}).get(layer_idx, [])
            if not chunks:
                return None
            return mx.concatenate(chunks, axis=2)[:, :, :seq_len, :].astype(mx.float32)

        q_post = concat_chunks("prefill_q_post")
        k_post = concat_chunks("prefill_k_post")
        k_pre = concat_chunks("prefill_k_pre")
        if q_post is None:
            q_post = c.keys[:, :, :seq_len, :].astype(mx.float32)
        if k_post is None:
            k_post = c.keys[:, :, :seq_len, :].astype(mx.float32)
        if k_pre is None:
            k_pre = c.keys[:, :, :seq_len, :].astype(mx.float32)
        q_post = q_post[0].transpose(1, 0, 2)
        k_post = k_post[0].transpose(1, 0, 2)
        k_pre = k_pre[0].transpose(1, 0, 2)
        return q_post, k_post, k_pre

    def _compactor_scores(self, q_post: Any, k_post: Any, k_pre: Any, seq_len: int):
        import mlx.core as mx

        leverage = self._compactor_leverage_scores(k_pre)
        attn = self._compactor_noncausal_attention_scores(q_post, k_post)
        blend = float(getattr(self.cfg.eviction, "compactor_accum_blending", 0.5))
        return attn + leverage * blend

    def _compactor_leverage_scores(self, k_pre: Any):
        import mlx.core as mx

        n, h_count, dim = [int(x) for x in k_pre.shape]
        sketch_dim = max(1, min(int(getattr(self.cfg.eviction, "compactor_sketch_dim", 48)), dim))
        seed = int(getattr(self.cfg, "seed", 0)) + 1777
        rng = np.random.default_rng(seed)
        phi_np = rng.standard_normal((dim, sketch_dim)).astype(np.float32) / math.sqrt(float(sketch_dim))
        phi = mx.array(phi_np)
        x = mx.matmul(k_pre.transpose(1, 0, 2), phi)
        chunk_size = int(getattr(self.cfg.eviction, "compactor_chunk_size", 512))
        chunk_size = n if chunk_size <= 0 else max(1, chunk_size)
        scores_np = np.zeros((n, h_count), dtype=np.float32)
        reg = 5e-3
        for h in range(h_count):
            for start in range(0, n, chunk_size):
                end = min(n, start + chunk_size)
                chunk = x[h, start:end, :].astype(mx.float32)
                chunk = chunk - mx.mean(chunk, axis=0, keepdims=True)
                gram = mx.matmul(chunk.transpose(1, 0), chunk)
                gram = gram + mx.eye(int(gram.shape[0]), dtype=mx.float32) * reg
                u, s, _ = mx.linalg.svd(gram, stream=mx.cpu)
                s = mx.maximum(s, 1e-8)
                sv = u * (1.0 / mx.sqrt(s)).reshape(1, -1)
                proj = mx.matmul(chunk, sv)
                vals = mx.sum(proj * proj, axis=1)
                scores_np[start:end, h] = np.asarray(vals.tolist(), dtype=np.float32)
        return _zscore_mx(mx.array(scores_np))

    def _compactor_noncausal_attention_scores(self, q_post: Any, k_post: Any):
        import mlx.core as mx

        n, hq, dim = [int(x) for x in q_post.shape]
        hkv = int(k_post.shape[1])
        group = max(1, hq // max(1, hkv))
        chunk_size = max(1, int(getattr(self.cfg.eviction, "compactor_attention_chunk_size", 128)))
        out_np = np.zeros((n, hkv), dtype=np.float32)
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            for h in range(hkv):
                qh = q_post[start:end, h * group : (h + 1) * group, :].reshape(-1, dim).astype(mx.float32)
                kh = k_post[start:end, h, :].astype(mx.float32)
                logits = mx.matmul(qh, kh.transpose(1, 0))
                probs = mx.softmax(logits, axis=-1, precise=True)
                out_np[start:end, h] = np.asarray(mx.sum(probs, axis=0).tolist(), dtype=np.float32)
        return _zscore_mx(mx.array(out_np))

    def _compactor_protected_counts(self, seq_len: int) -> Tuple[int, int]:
        first = getattr(self.cfg.eviction, "compactor_protected_first_tokens", None)
        last = getattr(self.cfg.eviction, "compactor_protected_last_tokens", None)
        first = 16 if first is None else int(first)
        last = 64 if last is None else int(last)
        if first + last >= seq_len:
            return 0, 0
        return max(0, first), max(0, last)

    def _apply_headwise_keep(self, c: Any, layer_idx: int, keep_by_head: List[List[int]], seq_len: int) -> None:
        import mlx.core as mx

        old_position_map = self.position_maps.get(layer_idx)
        if old_position_map is None or int(old_position_map.shape[0]) < seq_len:
            old_position_map = mx.arange(seq_len)
        h_count = int(c.keys.shape[1])
        max_len = max(1, max((len(v) for v in keep_by_head), default=0))
        key_parts = []
        value_parts = []
        map_rows = []
        valid_rows = []
        selected_by_head: Dict[str, List[int]] = {}
        selected_union = set()
        for h in range(h_count):
            values = keep_by_head[h] if h < len(keep_by_head) else []
            idx = mx.array(values, dtype=mx.int32) if values else mx.array([], dtype=mx.int32)
            n = int(idx.shape[0])
            if n > 0:
                kh = mx.take(c.keys[:, h : h + 1, :seq_len, :], idx, axis=2)
                vh = mx.take(c.values[:, h : h + 1, :seq_len, :], idx, axis=2)
                pos = mx.take(old_position_map[:seq_len], idx, axis=0)
                pos_vals = [int(x) for x in pos.tolist()]
            else:
                kh = mx.zeros((c.keys.shape[0], 1, 0, c.keys.shape[-1]), dtype=c.keys.dtype)
                vh = mx.zeros((c.values.shape[0], 1, 0, c.values.shape[-1]), dtype=c.values.dtype)
                pos = mx.array([], dtype=mx.int32)
                pos_vals = []
            if n < max_len:
                pad_n = max_len - n
                kh = mx.concatenate([kh, mx.zeros((c.keys.shape[0], 1, pad_n, c.keys.shape[-1]), dtype=c.keys.dtype)], axis=2)
                vh = mx.concatenate([vh, mx.zeros((c.values.shape[0], 1, pad_n, c.values.shape[-1]), dtype=c.values.dtype)], axis=2)
                pos = mx.concatenate([pos, mx.full((pad_n,), -1, dtype=mx.int32)], axis=0)
            key_parts.append(kh)
            value_parts.append(vh)
            map_rows.append(pos.reshape(1, max_len))
            valid_row = [True] * n + [False] * (max_len - n)
            valid_rows.append(mx.array(valid_row, dtype=mx.bool_).reshape(1, max_len))
            selected_by_head[str(h)] = pos_vals
            selected_union.update(pos_vals)
        c.keys = mx.concatenate(key_parts, axis=1)
        c.values = mx.concatenate(value_parts, axis=1)
        c.offset = int(max_len)
        c.logical_offset = int(getattr(c, "logical_offset", seq_len))
        c.head_position_map = mx.concatenate(map_rows, axis=0)
        c.head_valid_mask = mx.concatenate(valid_rows, axis=0)
        self.head_position_maps[layer_idx] = c.head_position_map
        self.position_maps[layer_idx] = mx.array(sorted(selected_union), dtype=mx.int32)
        self.next_positions[layer_idx] = int(c.logical_offset)
        self.last_selected[layer_idx] = sorted(selected_union)
        self.last_selected_by_head[layer_idx] = selected_by_head

    def _cache_num_heads(self, layer_idx: int) -> int:
        observed = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
        if observed:
            return int(observed[-1].shape[0])
        return int(self.model_info_num_kv_heads())

    def model_info_num_kv_heads(self) -> int:
        return int(getattr(self.cfg.model, "num_key_value_heads", 1) or 1)

    def _score_rows(self, c: Any, seq_len: int, source: Optional[str] = None):
        import mlx.core as mx

        source = (source or self.cfg.eviction.score_source).lower()
        source = {"value": "v", "key": "k", "key_value_concat": "kv"}.get(source, source)
        v_rows = c.values[:, :, :seq_len, :][0].mean(axis=0).astype(mx.float32)
        if source == "v":
            return v_rows
        k_rows = c.keys[:, :, :seq_len, :][0].mean(axis=0).astype(mx.float32)
        if source == "k":
            return k_rows
        return mx.concatenate([k_rows, v_rows], axis=-1)

    def _compute_scores(self, c: Any, layer_idx: int, seq_len: int):
        import mlx.core as mx

        method = self.method
        if method in ("full", "basic", "basic_generate"):
            return None
        if method in ("random", "sink_recent_random", "oracle_evidence", "oracle_answer_region"):
            return None
        if method in ("recency", "sink_recent"):
            return mx.arange(seq_len).astype(mx.float32)
        if method in ATTENTION_SCORE_METHODS:
            mode = "accumulated"
            if method == "windowed_attention":
                mode = "windowed"
            elif method == "attention_decay":
                mode = "decayed"
            scores = self._attention_scores(layer_idx, seq_len, mode=mode)
            self._last_attn_scores[layer_idx] = scores
            self._record_score_refit()
            return scores
        if method in SNAP_METHODS:
            scores = self._attention_scores(layer_idx, seq_len, mode="snapkv")
            self._last_attn_scores[layer_idx] = scores
            self._record_score_refit()
            return scores
        if method in ("key_l2_norm", "value_l2_norm", "key_l1_norm", "value_l1_norm"):
            source = "k" if method.startswith("key") else "v"
            rows = self._score_rows(c, seq_len, source=source)
            p = 1 if "_l1_" in method else 2
            self._record_score_refit()
            return mx.sum(mx.abs(rows), axis=1) if p == 1 else mx.sqrt(mx.sum(rows * rows, axis=1))
        if method in ("l1_decode_only", "l2_decode_only") and self.phase == "prefill":
            return None
        rows = self._score_rows(c, seq_len)
        if method in (
            "l1_prefill_only",
            "l2_prefill_only",
            "l2_key_prefill_only",
            "compactor",
            "attention_l1_compactor",
            "attention_l2_compactor",
        ):
            cached = self._static_score_cache.get(layer_idx)
            if cached is None:
                if method == "l1_prefill_only":
                    cached = self.l1_estimators[layer_idx].scores(rows)
                elif method == "l2_key_prefill_only":
                    key_rows = self._score_rows(c, seq_len, source="k")
                    cached = self.l2_estimators[layer_idx].scores(key_rows)
                elif method in ("compactor", "attention_l1_compactor", "attention_l2_compactor"):
                    key_rows = self._score_rows(c, seq_len, source="k")
                    if method == "attention_l1_compactor":
                        geom = self.l1_estimators[layer_idx].scores(rows)
                    else:
                        geom = self.l2_estimators[layer_idx].scores(key_rows)
                    attn = self._attention_scores(layer_idx, seq_len, mode="accumulated")
                    self._last_attn_scores[layer_idx] = attn
                    self._last_geom_scores[layer_idx] = geom
                    cached = _merge_score_vectors(
                        attn,
                        geom,
                        self.cfg.eviction.lambda_attn,
                        self.cfg.eviction.score_normalization,
                    )
                else:
                    cached = self.l2_estimators[layer_idx].scores(rows)
                self._static_score_cache[layer_idx] = cached
                self._record_score_refit()
            if int(cached.shape[0]) < seq_len:
                pad = mx.zeros((seq_len - int(cached.shape[0]),), dtype=cached.dtype)
                cached = mx.concatenate([cached, pad], axis=0)
                self._static_score_cache[layer_idx] = cached
            return cached[:seq_len]
        if method in ("l1", "l1_leverage", "l1_decode_only", "sink_recent_l1"):
            self._record_score_refit()
            return self.l1_estimators[layer_idx].scores(rows)
        if method in ("l2", "l2_leverage", "l2_decode_only", "sink_recent_l2"):
            self._record_score_refit()
            return self.l2_estimators[layer_idx].scores(rows)
        if method in HYBRID_METHODS:
            attn = self._attention_scores(layer_idx, seq_len, mode="accumulated")
            geom = self._geom_scores(rows, layer_idx)
            self._last_attn_scores[layer_idx] = attn
            self._last_geom_scores[layer_idx] = geom
            self._record_score_refit()
            return _merge_score_vectors(
                attn,
                geom,
                self.cfg.eviction.lambda_attn,
                self.cfg.eviction.score_normalization,
            )
        return None

    def _layer_budget(self, layer_idx: int, num_layers: int, base_budget: int) -> int:
        if self.method != "pyramidkv" or num_layers <= 1:
            return int(base_budget)
        total = int(base_budget) * int(num_layers)
        mode = str(getattr(self.cfg.eviction, "pyramid_mode", "funnel") or "funnel")
        if mode == "funnel":
            weights = np.arange(num_layers, 0, -1, dtype=np.float64)
        elif mode == "inverse_funnel":
            weights = np.arange(1, num_layers + 1, dtype=np.float64)
        else:
            return int(base_budget)
        raw = np.maximum((weights / weights.sum() * total).astype(int), 4)
        return max(1, min(int(base_budget), int(raw[int(layer_idx)])))

    def _select_indices(
        self,
        scores: Any,
        seq_len: int,
        budget: int,
        layer_idx: Optional[int] = None,
    ):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        method = self.method
        if method in ("recency", "basic", "basic_generate"):
            return mx.arange(max(0, seq_len - budget), seq_len)
        if method in ("random", "sink_recent_random"):
            return self._select_random_indices(seq_len, budget, layer_idx)
        if method in ("oracle_evidence", "oracle_answer_region"):
            return self._select_oracle_indices(seq_len, budget, int(layer_idx or 0), scores)
        if method in ("l1_decode_only", "l2_decode_only") and self.phase == "prefill":
            return self._select_sink_recent_indices(seq_len, budget)
        if method in SNAP_METHODS:
            return self._select_snapkv_indices(scores, seq_len, budget)
        if (
            method in HYBRID_METHODS
            and method not in COMPACTORLIKE_HYBRID_METHODS
            and self.cfg.eviction.hybrid_mode == "budget_split"
        ):
            return self._select_hybrid_indices(seq_len, budget, int(layer_idx or 0), scores)

        sink = min(self.cfg.eviction.sink_size, max(0, budget - 1))
        max_recent = max(0, budget - sink - 1)
        recent = min(self.cfg.eviction.recent_size, max_recent)
        mid_budget = max(0, budget - sink - recent - 1)
        parts = []
        if sink > 0:
            parts.append(mx.arange(sink))
        if recent > 0:
            parts.append(mx.arange(seq_len - 1 - recent, seq_len - 1))
        if mid_budget > 0 and scores is not None:
            start = sink
            end = seq_len - 1 - recent
            if end > start:
                cand = scores[start:end]
                take = min(mid_budget, cand.shape[0])
                if take >= cand.shape[0]:
                    idx = mx.arange(cand.shape[0])
                else:
                    idx = mx.argpartition(-cand, max(0, take - 1))[:take]
                parts.append(idx + start)
        parts.append(mx.array([seq_len - 1]))
        keep = mx.sort(mx.concatenate(parts)) if parts else mx.arange(seq_len)
        if keep.shape[0] > budget:
            keep = keep[:budget]
        return keep

    def _select_sink_recent_indices(self, seq_len: int, budget: int):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        sink = min(int(self.cfg.eviction.sink_size), max(0, int(budget)))
        recent_budget = max(0, int(budget) - sink)
        recent_start = max(sink, seq_len - recent_budget)
        keep = sorted(set(range(sink)) | set(range(recent_start, seq_len)))
        return mx.array(keep[:budget], dtype=mx.int32)

    def _select_random_indices(self, seq_len: int, budget: int, layer_idx: Optional[int]):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        rng = np.random.default_rng(
            int(self.cfg.seed) + int(layer_idx or 0) * 1009 + self.eviction_count * 9173 + seq_len
        )
        reserved_parts = []
        if self.method == "sink_recent_random":
            sink = min(self.cfg.eviction.sink_size, budget)
            recent = min(self.cfg.eviction.recent_size, max(0, budget - sink))
            if sink > 0:
                reserved_parts.extend(range(sink))
            if recent > 0:
                reserved_parts.extend(range(seq_len - recent, seq_len))
        reserved = sorted(set(x for x in reserved_parts if 0 <= x < seq_len))
        remaining = max(0, budget - len(reserved))
        candidates = [i for i in range(seq_len) if i not in set(reserved)]
        if remaining > 0 and candidates:
            chosen = rng.choice(candidates, size=min(remaining, len(candidates)), replace=False)
            reserved.extend(int(x) for x in chosen.tolist())
        return mx.array(sorted(set(reserved))[:budget], dtype=mx.int32)

    def _select_oracle_indices(self, seq_len: int, budget: int, layer_idx: int, scores: Any):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        pos_map = self.position_maps.get(layer_idx)
        current = []
        oracle_set = set(self.oracle_positions)
        if pos_map is not None and oracle_set:
            current = [
                i
                for i, original_pos in enumerate(pos_map.tolist())
                if int(original_pos) in oracle_set
            ]
        sink = min(self.cfg.eviction.sink_size, budget)
        recent = min(self.cfg.eviction.recent_size, max(0, budget - sink))
        reserved = list(range(sink))
        if recent > 0:
            reserved.extend(range(seq_len - recent, seq_len))
        keep = mx.array(sorted(set(reserved + current)), dtype=mx.int32)
        return self._ensure_keep_budget(keep, seq_len, budget, scores)

    def _attention_scores(self, layer_idx: int, seq_len: int, mode: str):
        import mlx.core as mx

        if mode == "snapkv":
            observed = self.attention_state.get("observe", {}).get(layer_idx, [])
            usable = [vec[:seq_len] for vec in observed if int(vec.shape[0]) >= seq_len]
            if usable:
                scores = mx.sum(mx.stack(usable, axis=0), axis=0).astype(mx.float32)
                return self._pool_scores_1d(scores)
        if mode == "windowed":
            observed = self.attention_state.get("observe", {}).get(layer_idx, [])
            usable = [vec[:seq_len] for vec in observed if int(vec.shape[0]) >= seq_len]
            if usable:
                return mx.sum(mx.stack(usable, axis=0), axis=0).astype(mx.float32)
        if mode == "decayed":
            decayed = self.attention_state.get("decayed", {}).get(layer_idx)
            if decayed is not None and int(decayed.shape[0]) >= seq_len:
                return decayed[:seq_len].astype(mx.float32)
        accumulated = self.attention_state.get("accumulated", {}).get(layer_idx)
        if accumulated is not None and int(accumulated.shape[0]) >= seq_len:
            return accumulated[:seq_len].astype(mx.float32)
        latest = self.attention_state.get("last", {}).get(layer_idx)
        if latest is not None and int(latest.shape[0]) >= seq_len:
            return latest[:seq_len].astype(mx.float32)
        return mx.arange(seq_len).astype(mx.float32)

    def _pool_scores_1d(self, scores: Any):
        import mlx.core as mx

        kernel = int(
            getattr(
                self.cfg.eviction,
                "pooling_kernel",
                getattr(self.cfg.eviction, "kernel_size", 1),
            )
            or 1
        )
        if kernel <= 1 or int(scores.shape[0]) <= 1:
            return scores.astype(mx.float32)
        method = str(getattr(self.cfg.eviction, "pooling_method", "max") or "max").lower()
        values = np.array(scores.tolist(), dtype=np.float32)
        pad = max(0, kernel // 2)
        padded = np.pad(values, (pad, pad), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)[: values.shape[0]]
        if method == "avg" or method == "mean":
            pooled = windows.mean(axis=-1)
        else:
            pooled = windows.max(axis=-1)
        return mx.array(pooled.astype(np.float32))

    def _geom_scores(self, rows: Any, layer_idx: int):
        import mlx.core as mx

        if self.method in ("attention_l2", "attn_l2", "attention_l2_compactor"):
            return self.l2_estimators[layer_idx].scores(rows)
        if self.method == "attention_norm":
            return mx.sqrt(mx.sum(rows * rows, axis=1))
        if self.method in ("attention_recency", "attention_sink_recency"):
            return mx.arange(int(rows.shape[0])).astype(mx.float32)
        return self.l1_estimators[layer_idx].scores(rows)

    def _select_snapkv_indices(self, scores: Any, seq_len: int, budget: int):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        sink = min(self.cfg.eviction.sink_size, max(0, budget - 1))
        obs = min(max(1, int(getattr(self.cfg.eviction, "window_size", 32))), seq_len)
        hist_len = max(0, seq_len - obs)
        parts = []
        if sink > 0:
            parts.append(mx.arange(sink))
        if hist_len > sink and scores is not None:
            hist_budget = max(0, budget - sink - obs)
            if hist_budget > 0:
                cand = scores[sink:hist_len]
                take = min(hist_budget, int(cand.shape[0]))
                if take >= int(cand.shape[0]):
                    idx = mx.arange(int(cand.shape[0]))
                else:
                    idx = mx.argpartition(-cand, max(0, take - 1))[:take]
                parts.append(idx + sink)
        parts.append(mx.arange(max(0, seq_len - obs), seq_len))
        keep = self._unique_sorted_indices(mx.concatenate(parts)) if parts else mx.arange(seq_len)
        if keep.shape[0] > budget:
            recent = mx.arange(max(0, seq_len - budget), seq_len)
            keep = mx.sort(recent)
        return keep

    def _select_hybrid_indices(self, seq_len: int, budget: int, layer_idx: int, scores: Any):
        import mlx.core as mx

        sink_b = min(seq_len, max(0, int(budget * float(self.cfg.eviction.sink_budget_ratio))))
        configured_sink = min(seq_len, max(0, int(self.cfg.eviction.sink_size)))
        sink_b = min(seq_len, max(sink_b, configured_sink))
        recent_b = min(
            seq_len - sink_b,
            max(0, int(budget * float(self.cfg.eviction.recent_budget_ratio))),
        )
        if sink_b == 0 and self.cfg.eviction.sink_size > 0:
            sink_b = min(self.cfg.eviction.sink_size, max(0, budget - 1))
        if recent_b == 0 and self.cfg.eviction.recent_size > 0:
            recent_b = min(self.cfg.eviction.recent_size, max(0, budget - sink_b - 1))

        parts = []
        source_map: Dict[int, List[str]] = {}
        if sink_b > 0:
            sink_idx = mx.arange(sink_b)
            parts.append(sink_idx)
            for idx in sink_idx.tolist():
                source_map.setdefault(int(idx), []).append("sink")
        if recent_b > 0:
            recent_idx = mx.arange(max(0, seq_len - recent_b), seq_len)
            parts.append(recent_idx)
            for idx in recent_idx.tolist():
                source_map.setdefault(int(idx), []).append("recent")
        selected = self._unique_sorted_indices(mx.concatenate(parts)) if parts else mx.array([], dtype=mx.int32)

        def take_top(score_vec: Any, take: int, current: Any, source_name: str):
            if score_vec is None or take <= 0 or int(current.shape[0]) >= budget:
                return current
            vec = score_vec[:seq_len].astype(mx.float32)
            mask_np = np.ones(seq_len, dtype=bool)
            if int(current.shape[0]) > 0:
                mask_np[[int(x) for x in current.tolist() if 0 <= int(x) < seq_len]] = False
            mask = mx.array(mask_np)
            masked = mx.where(mask, vec, -mx.inf)
            valid_count = int(mx.sum(mask).item())
            if valid_count <= 0:
                return current
            take_n = min(int(take), valid_count, max(0, budget - int(current.shape[0])))
            if take_n <= 0:
                return current
            idx = mx.argpartition(-masked, max(0, take_n - 1))[:take_n]
            for token_idx in idx.tolist():
                source_map.setdefault(int(token_idx), []).append(source_name)
            return self._unique_sorted_indices(mx.concatenate([current, idx]))

        remaining = max(0, budget - int(selected.shape[0]))
        attn_take = min(int(budget * float(self.cfg.eviction.attn_budget_ratio)), remaining)
        selected = take_top(self._last_attn_scores.get(layer_idx), attn_take, selected, "attention")

        remaining = max(0, budget - int(selected.shape[0]))
        geom_ratio = float(getattr(self.cfg.eviction, "l1_budget_ratio", 0.3))
        if self.method in ("attention+l2", "attention_l2", "attn_l2"):
            geom_ratio = float(getattr(self.cfg.eviction, "l1_budget_ratio", 0.3))
        geom_take = min(int(budget * geom_ratio), remaining)
        selected = take_top(self._last_geom_scores.get(layer_idx), geom_take, selected, "geometry")

        if int(selected.shape[0]) < budget:
            selected = take_top(scores, budget - int(selected.shape[0]), selected, "combined")
        if int(selected.shape[0]) < budget:
            fill = mx.arange(max(0, seq_len - budget), seq_len)
            selected = self._unique_sorted_indices(mx.concatenate([selected, fill]))
        keep = mx.sort(selected)
        if int(keep.shape[0]) > budget:
            keep = self._trim_hybrid_keep(keep, budget, source_map, scores)
        for idx in keep.tolist():
            source_map.setdefault(int(idx), ["fill"])
        self._component_sources_current[layer_idx] = source_map
        return keep

    def _trim_hybrid_keep(self, keep: Any, budget: int, source_map: Dict[int, List[str]], scores: Any = None):
        import mlx.core as mx

        values = sorted({int(x) for x in keep.tolist()})
        chosen: List[int] = []

        def add(token_idx: int) -> None:
            if token_idx not in chosen and len(chosen) < budget:
                chosen.append(token_idx)

        sink_values = [x for x in values if "sink" in source_map.get(x, [])]
        recent_values = [x for x in values if "recent" in source_map.get(x, [])]
        for idx in sorted(sink_values):
            add(idx)
        for idx in sorted(recent_values, reverse=True):
            add(idx)

        remaining = [x for x in values if x not in set(chosen)]
        if scores is not None:
            score_vals = scores.tolist()
            remaining.sort(
                key=lambda x: float(score_vals[x]) if 0 <= x < len(score_vals) else float("-inf"),
                reverse=True,
            )
        else:
            remaining.sort(reverse=True)
        for idx in remaining:
            add(idx)
        return mx.array(sorted(chosen[:budget]), dtype=mx.int32)

    @staticmethod
    def _unique_sorted_indices(indices: Any):
        import mlx.core as mx

        values = sorted({int(x) for x in indices.tolist()})
        return mx.array(values, dtype=mx.int32)

    def _ensure_keep_budget(self, keep: Any, seq_len: int, budget: int, scores: Any = None):
        import mlx.core as mx

        values = sorted({int(x) for x in keep.tolist() if 0 <= int(x) < seq_len})
        target = min(seq_len, int(budget))
        if len(values) < target:
            selected = set(values)
            if scores is not None:
                vec = scores[:seq_len].astype(mx.float32)
                candidates = [i for i in range(seq_len) if i not in selected]
                if candidates:
                    cand_scores = [(float(vec[i].item()), i) for i in candidates]
                    cand_scores.sort(reverse=True)
                    values.extend(i for _, i in cand_scores[: target - len(values)])
            if len(values) < target:
                for i in range(max(0, seq_len - target), seq_len):
                    if i not in set(values):
                        values.append(i)
                    if len(set(values)) >= target:
                        break
        if len(values) > target:
            values = sorted(values)[-target:]
        return mx.array(sorted(set(values))[:target], dtype=mx.int32)

    def _prune_attention_state(self, layer_idx: int, keep: Any, seq_len: int) -> None:
        import mlx.core as mx

        for key in ("last", "accumulated", "decayed"):
            vec = self.attention_state.get(key, {}).get(layer_idx)
            if vec is not None and int(vec.shape[0]) >= seq_len:
                self.attention_state[key][layer_idx] = mx.take(vec[:seq_len], keep, axis=0)
        static_scores = self._static_score_cache.get(layer_idx)
        if static_scores is not None and int(static_scores.shape[0]) >= seq_len:
            self._static_score_cache[layer_idx] = mx.take(static_scores[:seq_len], keep, axis=0)
        observe = self.attention_state.get("observe", {}).get(layer_idx)
        if observe:
            self.attention_state["observe"][layer_idx] = [
                mx.take(vec[:seq_len], keep, axis=0)
                for vec in observe
                if int(vec.shape[0]) >= seq_len
            ]
        observe_heads = self.attention_state.get("observe_heads", {}).get(layer_idx)
        if observe_heads:
            self.attention_state["observe_heads"][layer_idx] = [
                mx.take(vec[:, :seq_len], keep, axis=1)
                for vec in observe_heads
                if int(vec.shape[-1]) >= seq_len
            ]

    @staticmethod
    def _to_int_list(arr: Any) -> List[int]:
        return [int(x) for x in arr.tolist()]

    @staticmethod
    def _to_float_list(arr: Any) -> List[float]:
        return [float(x) for x in arr.astype(arr.dtype).tolist()]


class MLXRunner(BaseRunner):
    """Formal MLX-LM backend for KV cache eviction experiments."""

    backend_name = "mlx"

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        self.model = None
        self.tokenizer = None
        self.hf_tokenizer = None
        self.model_info: Dict[str, Any] = {}
        self.attention_state: Dict[str, Any] = {
            "last": {},
            "accumulated": {},
            "decayed": {},
            "observe": {},
            "observe_heads": {},
            "prefill_q_post": {},
            "prefill_k_post": {},
            "prefill_k_pre": {},
            "hook_errors": 0,
            "max_observe": 32,
            "decay_gamma": 0.95,
            "enabled": False,
            "phase": "idle",
            "current_method": None,
        }

    def run(
        self,
        methods: List[str],
        budgets: List[int],
        budget_ratios: Optional[List[float]] = None,
        skip_analysis: bool = False,
    ) -> Path:
        self.load_model()
        _, samples = load_benchmark(self.cfg, self.hf_tokenizer)
        out_dir = self.make_run_dir()
        self.save_run_metadata(out_dir, self.model_info)

        results: List[Dict[str, Any]] = []
        for budget in budgets:
            for method in methods:
                for sample_idx, sample in enumerate(samples):
                    actual_budget = self._actual_budget(sample, budget, budget_ratios or [])
                    try:
                        row = self.run_one(sample, sample_idx, method, actual_budget, out_dir)
                    except Exception as exc:
                        row = self.error_result(sample, sample_idx, method, actual_budget, exc)
                    results.append(row)
                    save_results(
                        row,
                        out_dir
                        / "samples"
                        / f"{method}_b{actual_budget}_s{sample_idx}.json",
                    )

        self.save_result_bundle(results, out_dir)
        if not skip_analysis:
            try:
                from scripts.run_analysis import run_analysis

                run_analysis(results, self.cfg, out_dir)
                from scripts.plot_results import (
                    plot_accuracy_by_budget,
                    plot_metric_by_method_budget,
                    plot_latency_by_method_budget,
                    plot_method_budget_heatmap,
                    plot_model_method_heatmap,
                    plot_evidence_recall_by_depth,
                    plot_latency,
                    plot_overlap,
                    plot_rank,
                    plot_selected_positions,
                    write_case_study_markdown,
                )

                fig_dir = out_dir / "figures"
                figure_outputs = {
                    "accuracy_by_budget": plot_accuracy_by_budget(out_dir, fig_dir),
                    "accuracy_by_method_budget": plot_metric_by_method_budget(
                        out_dir, fig_dir, "accuracy", "Accuracy by Cache Budget", "Accuracy", "accuracy_by_method_budget"
                    ),
                    "evidence_recall_by_method_budget": (
                        plot_metric_by_method_budget(
                            out_dir, fig_dir, "avg_evidence_recall", "Evidence Recall by Cache Budget", "Evidence Recall", "evidence_recall_by_method_budget"
                        )
                        if self.cfg.analysis.evidence_recall
                        else ""
                    ),
                    "official_score_by_method_budget": plot_metric_by_method_budget(
                        out_dir, fig_dir, "avg_official_score", "Official Score by Cache Budget", "Official Score", "official_score_by_method_budget"
                    ),
                    "latency_by_method_budget": plot_latency_by_method_budget(out_dir, fig_dir),
                    "method_budget_heatmap": plot_method_budget_heatmap(
                        out_dir, fig_dir, "accuracy", "Method x Budget Accuracy", "method_budget_heatmap"
                    ),
                    "official_score_heatmap": plot_method_budget_heatmap(
                        out_dir, fig_dir, "avg_official_score", "Method x Budget Official Score", "official_score_heatmap"
                    ),
                    "evidence_recall_heatmap": (
                        plot_evidence_recall_by_depth(out_dir, fig_dir)
                        if self.cfg.analysis.evidence_recall
                        else ""
                    ),
                    "method_overlap_heatmap": plot_overlap(out_dir, fig_dir),
                    "rank_correlation_heatmap": plot_rank(out_dir, fig_dir),
                    "latency_breakdown": plot_latency(out_dir, fig_dir),
                    "selected_token_position_distribution": plot_selected_positions(out_dir, fig_dir),
                    "model_method_accuracy_heatmap": plot_model_method_heatmap(
                        out_dir, fig_dir, "accuracy", "Model x Method Accuracy", "model_method_accuracy_heatmap"
                    ),
                    "model_method_official_score_heatmap": plot_model_method_heatmap(
                        out_dir, fig_dir, "avg_official_score", "Model x Method Official Score", "model_method_official_score_heatmap"
                    ),
                    "model_method_evidence_recall_heatmap": (
                        plot_model_method_heatmap(
                            out_dir, fig_dir, "avg_evidence_recall", "Model x Method Evidence Recall", "model_method_evidence_recall_heatmap"
                        )
                        if self.cfg.analysis.evidence_recall
                        else ""
                    ),
                    "case_study_markdown": write_case_study_markdown(out_dir),
                }
                save_results(figure_outputs, fig_dir / "figures_summary.json")
            except Exception as exc:
                save_results({"error": str(exc)}, out_dir / "analysis" / "analysis_error.json")
        return out_dir

    def load_model(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        cfg = self.cfg.model
        mx.random.seed(self.cfg.seed)
        t0 = time.perf_counter()
        loaded = load(
            cfg.name,
            return_config=True,
            revision=cfg.revision,
            tokenizer_config={"trust_remote_code": cfg.trust_remote_code},
        )
        self.model, self.tokenizer, raw_config = loaded
        load_time = time.perf_counter() - t0
        quant_time = 0.0
        quantized_layers = 0
        if cfg.quant_bits and cfg.mlx_weight_quantize:
            t0 = time.perf_counter()
            nn.quantize(
                self.model,
                group_size=int(cfg.quant_group_size),
                bits=int(cfg.quant_bits),
            )
            mx.eval(self.model.parameters())
            quant_time = time.perf_counter() - t0
            try:
                quantized_layers = sum(
                    1
                    for _, module in self.model.named_modules()
                    if "Quantized" in type(module).__name__
                )
            except Exception:
                quantized_layers = 0
        self.hf_tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.model_info = {
            "model_name": cfg.name,
            "backend": "mlx",
            "quant_bits": cfg.quant_bits,
            "quant_group_size": cfg.quant_group_size,
            "mlx_weight_quantize": cfg.mlx_weight_quantize,
            "load_time_s": load_time,
            "quantize_time_s": quant_time,
            "quantized_layers": quantized_layers,
            "num_layers": len(self.model.model.layers),
            "model_type": raw_config.get("model_type"),
            "vocab_size": raw_config.get("vocab_size"),
            "hidden_size": raw_config.get("hidden_size"),
            "num_attention_heads": raw_config.get("num_attention_heads"),
            "num_key_value_heads": raw_config.get("num_key_value_heads"),
            "max_position_embeddings": raw_config.get("max_position_embeddings"),
            "tokenizer_class": type(self.hf_tokenizer).__name__,
        }
        self.install_attention_hooks()
        self.model_info["attention_hook_installed"] = bool(
            self.attention_state.get("hook_installed", False)
        )
        adapter = build_model_adapter(
            cfg,
            raw_config=raw_config,
            tokenizer=self.hf_tokenizer,
            cache_format="mlx_kv",
            attention_hook_installed=self.model_info["attention_hook_installed"],
        )
        self.model_info.update(adapter.to_dict())

    def reset_attention_state(self) -> None:
        self.attention_state["last"] = {}
        self.attention_state["accumulated"] = {}
        self.attention_state["decayed"] = {}
        self.attention_state["observe"] = {}
        self.attention_state["observe_heads"] = {}
        self.attention_state["prefill_q_post"] = {}
        self.attention_state["prefill_k_post"] = {}
        self.attention_state["prefill_k_pre"] = {}
        self.attention_state["hook_errors"] = 0
        self.attention_state["max_observe"] = max(1, int(getattr(self.cfg.eviction, "window_size", 32)))
        self.attention_state["decay_gamma"] = float(getattr(self.cfg.eviction, "decay_gamma", 0.95))
        self.attention_state["enabled"] = False
        self.attention_state["phase"] = "idle"
        self.attention_state["current_method"] = None

    def install_attention_hooks(self) -> None:
        """Install a minimal Qwen-style MLX attention hook for runtime scores."""
        try:
            from mlx_lm.models.base import scaled_dot_product_attention
        except Exception as exc:
            self.attention_state["hook_error_message"] = str(exc)
            self.attention_state["hook_installed"] = False
            return

        installed = 0
        for layer_idx, layer in enumerate(getattr(self.model.model, "layers", [])):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            attn._l1kv_layer_idx = layer_idx
            attn._l1kv_attention_state = self.attention_state
            cls = type(attn)
            if getattr(cls, "_l1kv_patched", False):
                installed += 1
                continue

            def patched_call(self_attn, x, mask=None, cache=None, _sdpa=scaled_dot_product_attention):
                B, L, _ = x.shape
                queries = self_attn.q_proj(x)
                keys = self_attn.k_proj(x)
                values = self_attn.v_proj(x)

                queries = queries.reshape(B, L, self_attn.n_heads, -1)
                keys = keys.reshape(B, L, self_attn.n_kv_heads, -1)
                if hasattr(self_attn, "q_norm"):
                    queries = self_attn.q_norm(queries)
                if hasattr(self_attn, "k_norm"):
                    keys = self_attn.k_norm(keys)
                queries = queries.transpose(0, 2, 1, 3)
                keys = keys.transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                queries_pre_rope = queries
                keys_pre_rope = keys

                if cache is not None:
                    rope_offset = int(getattr(cache, "logical_offset", cache.offset))
                    queries = self_attn.rope(queries, offset=rope_offset)
                    keys = self_attn.rope(keys, offset=rope_offset)
                    keys_post_rope = keys
                    keys, values = cache.update_and_fetch(keys, values)
                    cache.logical_offset = rope_offset + int(L)
                else:
                    queries = self_attn.rope(queries)
                    keys = self_attn.rope(keys)
                    keys_post_rope = keys

                _record_compactor_prefill_tensors(
                    self_attn,
                    queries_pre_rope,
                    keys_pre_rope,
                    queries,
                    keys_post_rope,
                )
                _record_attention_from_hook(self_attn, queries, keys, query_len=int(L))
                head_mask = _cache_head_valid_attention_mask(
                    cache,
                    int(queries.shape[1]),
                    int(keys.shape[-2]),
                )
                if head_mask is not None:
                    head_mask = head_mask.astype(queries.dtype)
                    mask = head_mask if mask is None else mask + head_mask
                output = _sdpa(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self_attn.scale,
                    mask=mask,
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self_attn.o_proj(output)

            cls._l1kv_patched = True
            cls.__call__ = patched_call
            installed += 1
        self.attention_state["hook_installed"] = installed > 0
        self.attention_state["hooked_layers"] = installed

    def run_one(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        method: str,
        budget: int,
        out_dir: Path,
    ) -> Dict[str, Any]:
        method_key = canonical_method(method)
        try:
            spec = get_method_spec(method)
        except Exception as exc:
            return self.skipped_result(sample, sample_idx, method, budget, str(exc))
        reason = unsupported_reason(method, "mlx")
        if reason or method_key not in SUPPORTED_MLX_METHODS:
            return self.skipped_result(
                sample,
                sample_idx,
                method,
                budget,
                reason or f"MLX backend does not yet support method={method!r}",
                spec=spec,
            )
        if spec.requires_attention and not self.model_info.get("supports_attention_output"):
            return self.skipped_result(
                sample,
                sample_idx,
                method,
                budget,
                "MLX backend does not expose attention weights for this model",
                spec=spec,
            )

        prompt_ids, full_ids, answer_positions, prompt_text = self.sample_tokens(sample)
        oracle_positions = self.oracle_positions_for_sample(sample, method_key)
        generation = self.generate_with_cache(prompt_ids, method_key, budget, oracle_positions)
        ppl_stats = self.teacher_forced_ppl(
            full_ids,
            answer_positions,
            method_key,
            budget,
            oracle_positions,
        )

        selected = generation["selected_tokens_by_layer"]
        evidence = [int(x) for x in sample.get("evidence_positions") or []]
        evidence_stats = self.evidence_stats(selected, evidence)
        selected_artifacts = self.save_selected_and_scores(
            out_dir,
            method,
            budget,
            sample_idx,
            selected,
            generation.get("scores_by_layer") or {},
        )
        gt = sample.get("ground_truth") or sample.get("metadata", {}).get("answer")
        generated_text = generation["generated_text"]
        contains_gt = normalize_text(gt) in normalize_text(generated_text) if gt else False
        exact = normalize_text(generated_text) == normalize_text(gt) if gt else False
        f1 = answer_f1(generated_text, str(gt or ""))
        metadata = sample.get("metadata", {})
        official = evaluate_official(
            self.cfg.benchmark.name,
            metadata.get("task"),
            generated_text,
            sample,
            metadata,
        )
        eval_mode = (self.cfg.benchmark.evaluation or "ppl").lower()
        use_official_primary = (
            eval_mode in {"official", "both"}
            and official.get("official_score") is not None
        )
        if use_official_primary:
            primary_metric = official.get("official_metric_name") or "official_score"
            primary_score = official.get("official_score")
        else:
            primary_metric = "ppl"
            primary_score = ppl_stats.get("ppl")
        row_correct = bool(contains_gt or exact)
        if use_official_primary and official.get("official_correct") is not None:
            row_correct = bool(official.get("official_correct"))

        context_length = len(full_ids)
        effective_update_policy = self.cfg.eviction.update_policy
        effective_update_interval = self.cfg.eviction.update_interval
        effective_score_source = self.cfg.eviction.score_source
        if method_key in {
            "l1_prefill_only",
            "l2_prefill_only",
            "l2_key_prefill_only",
            "compactor",
            "attention_l1_compactor",
            "attention_l2_compactor",
        }:
            effective_update_policy = "prefill_only"
            effective_update_interval = 0
        elif method_key in {"l1_decode_only", "l2_decode_only"}:
            effective_update_policy = "decode_only"
        if method_key == "l2_key_prefill_only":
            effective_score_source = "key"
        elif method_key == "compactor":
            effective_score_source = "pre_rope_key_approximate_leverage+non_causal_attention"
        elif method_key == "attention_l1_compactor":
            effective_score_source = "rank_accumulated_attention+rank_value_l1_leverage"
        elif method_key == "attention_l2_compactor":
            effective_score_source = "rank_accumulated_attention+rank_key_l2_leverage"
        row: Dict[str, Any] = {
            "label": f"{method}_b{budget}_s{sample_idx}",
            "experiment_name": self.cfg.experiment_name,
            "run_id": self.cfg.run_id,
            "sample_id": sample_idx,
            "sample_idx": sample_idx,
            "method": method,
            "canonical_method": method_key,
            "method_family": spec.family,
            "budget": budget,
            "cache_budget": budget,
            "model": self.cfg.model.name,
            "model_name": self.cfg.model.name,
            "model_family": self.model_info.get("model_family"),
            "backend": "mlx",
            "quant_bits": self.cfg.model.quant_bits,
            "benchmark": self.cfg.benchmark.name,
            "dataset_split": self.cfg.benchmark.split,
            "context_length": context_length,
            "max_new_tokens": self.cfg.benchmark.max_new_tokens,
            "tokenizer": self.model_info.get("tokenizer_class"),
            "prompt": prompt_text if self.cfg.save_prompt_text else None,
            "prompt_hash": text_hash(prompt_text),
            "prediction": generated_text.strip(),
            "generated_text": generated_text,
            "generated_token_ids": generation["generated_token_ids"],
            "ground_truth": gt,
            "answers": sample.get("answers") or metadata.get("answers"),
            "all_classes": sample.get("all_classes") or metadata.get("all_classes"),
            "contains_ground_truth": contains_gt,
            "exact_match": exact,
            "answer_f1": f1,
            "correct": row_correct,
            "official_score": official.get("official_score"),
            "official_correct": official.get("official_correct"),
            "official_metric_name": official.get("official_metric_name"),
            "official_metric_implementation": official.get("official_metric_implementation"),
            "official_references": official.get("official_references"),
            "dataset_official": metadata.get("dataset_official"),
            "official_prompt": metadata.get("official_prompt"),
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "metric": {
                "ppl": ppl_stats.get("ppl"),
                "contains_ground_truth": contains_gt,
                "official_score": official.get("official_score"),
                "official_metric_name": official.get("official_metric_name"),
                "primary_metric": primary_metric,
                "primary_score": primary_score,
            },
            "ppl": ppl_stats.get("ppl"),
            "mean_nll": ppl_stats.get("mean_nll"),
            "loss": ppl_stats.get("mean_nll"),
            "latency": generation["total_time_s"] + ppl_stats.get("ppl_time_s", 0.0),
            "tokens_per_second": generation["tokens_per_second"],
            "avg_ms_per_token": generation["avg_ms_per_token"],
            "total_time_s": generation["total_time_s"] + ppl_stats.get("ppl_time_s", 0.0),
            "prefill_time_s": generation["prefill_time_s"],
            "decode_time_s": generation["decode_time_s"],
            "eviction_time_s": generation["eviction_time_s"],
            "score_time_s": generation["score_time_s"],
            "topk_time_s": generation["topk_time_s"],
            "cache_rebuild_time_s": generation["cache_rebuild_time_s"],
            "max_kv_len": generation["max_kv_len"],
            "final_kv_len": generation["final_kv_len"],
            "avg_kv_len": generation["avg_kv_len"],
            "max_kv_len_observed": generation["max_kv_len"],
            "cache_shape_summary": generation["cache_shape_summary"],
            "sink_size": self.cfg.eviction.sink_size,
            "recent_size": self.cfg.eviction.recent_size,
            "score_budget": budget - self.cfg.eviction.sink_size - self.cfg.eviction.recent_size,
            "score_source": effective_score_source,
            "configured_score_source": self.cfg.eviction.score_source,
            "seed": self.cfg.seed,
            "score_normalization": self.cfg.eviction.score_normalization,
            "attention_score_source": self._attention_score_source(method_key),
            "geometry_score_source": self._geometry_score_source(method_key),
            "sketch_dim": self.cfg.eviction.sketch_dim,
            "update_interval": effective_update_interval,
            "update_policy": effective_update_policy,
            "layer_strategy": self.cfg.eviction.layer_strategy,
            "head_strategy": self.cfg.eviction.head_strategy,
            "hybrid_mode": self.cfg.eviction.hybrid_mode,
            "lambda_attn": self.cfg.eviction.lambda_attn,
            "attn_budget_ratio": self.cfg.eviction.attn_budget_ratio,
            "l1_budget_ratio": self.cfg.eviction.l1_budget_ratio,
            "l2_budget_ratio": None,
            "recent_budget_ratio": self.cfg.eviction.recent_budget_ratio,
            "sink_budget_ratio": self.cfg.eviction.sink_budget_ratio,
            "selected_tokens": selected,
            "selected_tokens_by_layer": selected,
            "selected_token_sources": generation.get("selected_token_sources") or {},
            "selected_tokens_by_head": generation.get("selected_tokens_by_head") or {},
            "selected_token_types": self.selected_token_types(
                selected, prompt_ids + generation["generated_token_ids"]
            ),
            "selected_token_texts": self.selected_token_texts(
                selected, prompt_ids + generation["generated_token_ids"]
            ),
            "selected_token_distances_to_query": self.distances_to_target(
                selected, metadata.get("answer_token_start")
            ),
            "selected_token_distances_to_evidence": self.distances_to_evidence(
                selected, evidence
            ),
            "evidence_positions": evidence,
            "evidence_recall": evidence_stats["evidence_recall"],
            "evidence_precision": evidence_stats["evidence_precision"],
            "evidence_overlap_count": evidence_stats["evidence_overlap_count"],
            "needle_depth": metadata.get("needle_depth", metadata.get("depth_bucket")),
            "needle_token_start": metadata.get("needle_token_start"),
            "needle_token_end": metadata.get("needle_token_end"),
            "answer_token_start": metadata.get("answer_token_start"),
            "answer_token_end": metadata.get("answer_token_end"),
            "score_update_count": generation["score_update_count"],
            "score_phase_counts": generation.get("score_phase_counts", {}),
            "score_refit_count": generation.get("score_refit_count", 0),
            "score_refit_phase_counts": generation.get("score_refit_phase_counts", {}),
            "eviction_count": generation["eviction_count"],
            "score_stats": self.score_stats_from_layers(generation.get("scores_by_layer") or {}),
            "raw_score_stats": self.score_stats_from_layers(generation.get("scores_by_layer") or {}),
            "normalized_score_stats": self.normalized_score_stats_from_layers(
                generation.get("scores_by_layer") or {},
                self.cfg.eviction.score_normalization,
            ),
            "top_score_values": self.score_stats_from_layers(generation.get("scores_by_layer") or {}).get("top_values", []),
            "score_update_interval": effective_update_interval,
            "decode_only_prefill_policy": (
                "sink_recent" if method_key in {"l1_decode_only", "l2_decode_only"} else None
            ),
            "cache_budget_scope": generation.get("cache_budget_scope", "total_kv"),
            "prefill_compression": bool(generation.get("prefill_compression", False)),
            "sparse_head_mask": bool(generation.get("sparse_head_mask", False)),
            "faithful_baseline": method_key in PREFILL_COMPRESS_METHODS,
            "protected_first_tokens": (
                (16 if getattr(self.cfg.eviction, "compactor_protected_first_tokens", None) is None else self.cfg.eviction.compactor_protected_first_tokens)
                if method_key == "compactor" else None
            ),
            "protected_last_tokens": (
                (64 if getattr(self.cfg.eviction, "compactor_protected_last_tokens", None) is None else self.cfg.eviction.compactor_protected_last_tokens)
                if method_key == "compactor" else None
            ),
            "vector_shape": self.vector_shape_summary(generation["cache_shape_summary"], method_key),
            "approximate": bool(spec.approximate),
            "experimental": bool(spec.experimental),
            "oracle": bool(spec.oracle),
            "skipped": False,
            "skipped_reason": None,
            "unsupported_reason": None,
            "attention_hook_errors": generation.get("attention_hook_errors", 0),
            "selected_tokens_path": selected_artifacts.get("selected_tokens_path"),
            "scores_path": selected_artifacts.get("scores_path"),
            "metadata": metadata,
        }
        row["sanity_checks"] = self.sanity_checks(row, spec)
        row["sanity_check_failed"] = bool(row["sanity_checks"].get("violations"))
        if method_key in MANUAL_COMPACT_METHODS:
            row["unsupported_warning"] = (
                "MLX manual KVCache compaction keeps already-rotated keys and compacts "
                "the physical cache while preserving a logical RoPE offset for subsequent "
                "tokens. This is a functional research runner for controlled eviction "
                "comparisons; dedicated production cache adapters are still recommended "
                "before claiming kernel-level parity."
            )
        return row

    def sanity_checks(self, row: Dict[str, Any], spec: Any) -> Dict[str, Any]:
        violations: List[str] = []
        budget = int(row.get("budget") or 0)
        context_length = int(row.get("context_length") or 0)
        generated_count = len(row.get("generated_token_ids") or [])
        selected = row.get("selected_tokens_by_layer") or {}
        selected_by_head = row.get("selected_tokens_by_head") or {}
        final_kv_len = row.get("final_kv_len")
        method_key = row.get("canonical_method") or row.get("method")
        scope = row.get("cache_budget_scope") or "total_kv"
        sink_required = method_key not in {"recency", "random", "full"} and method_key not in PREFILL_COMPRESS_METHODS
        if (
            method_key != "full"
            and scope == "total_kv"
            and final_kv_len is not None
            and budget > 0
            and int(final_kv_len) > budget
        ):
            violations.append(f"final_kv_len={final_kv_len} exceeds budget={budget}")
        sink_size = int(row.get("sink_size") or 0)
        recent_size = int(row.get("recent_size") or 0)
        for layer, values in selected.items():
            vals = [int(x) for x in values]
            if len(vals) != len(set(vals)):
                violations.append(f"layer {layer}: duplicate selected tokens")
            if any(x < 0 or (context_length and x >= context_length + row.get("max_new_tokens", 0)) for x in vals):
                violations.append(f"layer {layer}: selected token out of original stream range")
            if method_key != "full" and method_key not in PREFILL_COMPRESS_METHODS and budget > 0 and len(vals) > budget:
                violations.append(f"layer {layer}: selected count {len(vals)} exceeds budget {budget}")
            if sink_required and sink_size > 0 and len(vals) >= sink_size:
                missing_sink = [i for i in range(sink_size) if i not in set(vals)]
                if missing_sink and not getattr(spec, "oracle", False):
                    violations.append(f"layer {layer}: missing sink tokens {missing_sink[:5]}")
            if recent_size > 0 and context_length > 0 and len(vals) >= recent_size:
                recent_start = max(0, int(row.get("final_kv_len") or context_length) - recent_size)
                # Original-position maps can include generated tokens, so only check count/budget here.
                _ = recent_start
        if method_key in PREFILL_COMPRESS_METHODS:
            for layer, head_map in selected_by_head.items():
                total_pairs = 0
                for head, values in (head_map or {}).items():
                    vals = [int(x) for x in values]
                    total_pairs += len(vals)
                    if len(vals) != len(set(vals)):
                        violations.append(f"layer {layer} head {head}: duplicate selected tokens")
                    if any(x < 0 or (context_length and x >= context_length + row.get("max_new_tokens", 0)) for x in vals):
                        violations.append(f"layer {layer} head {head}: selected token out of original stream range")
                    if method_key == "snapkv" and budget > 0 and len(vals) > budget + generated_count:
                        violations.append(f"layer {layer} head {head}: selected count {len(vals)} exceeds budget+generated={budget + generated_count}")
                if method_key == "compactor" and budget > 0 and head_map:
                    h_count = max(1, len(head_map))
                    allowed_pairs = (budget + generated_count) * h_count
                    if total_pairs > allowed_pairs:
                        violations.append(f"layer {layer}: selected pairs {total_pairs} exceeds (budget+generated)*heads={allowed_pairs}")
        return {"passed": not violations, "violations": violations}

    @staticmethod
    def _attention_score_source(method_key: str) -> Optional[str]:
        if method_key == "windowed_attention":
            return "windowed_current_query_attention"
        if method_key == "attention_decay":
            return "decayed_current_query_attention"
        if method_key in ATTENTION_SCORE_METHODS:
            return "accumulated_current_query_attention"
        if method_key in SNAP_METHODS:
            return "observation_window_current_query_attention"
        if method_key == "pyramidkv":
            return "layer_budget_observation_window_attention"
        if method_key == "compactor":
            return "prefill_non_causal_attention"
        if method_key in HYBRID_METHODS:
            return "accumulated_current_query_attention"
        return None

    @staticmethod
    def _geometry_score_source(method_key: str) -> Optional[str]:
        if method_key in ("key_l2_norm", "key_l1_norm"):
            return "key"
        if method_key in ("value_l2_norm", "value_l1_norm"):
            return "value"
        if method_key in ("l1", "l1_leverage", "l1_prefill_only", "l1_decode_only", "sink_recent_l1"):
            return "l1_leverage"
        if method_key in ("l2", "l2_leverage", "l2_prefill_only", "l2_decode_only", "sink_recent_l2"):
            return "l2_leverage"
        if method_key == "l2_key_prefill_only":
            return "key_l2_leverage"
        if method_key == "compactor":
            return "pre_rope_key_approximate_leverage+non_causal_attention"
        if method_key in ("attention_l2", "attn_l2", "attention_l2_compactor"):
            return "l2_leverage"
        if method_key == "attention_l1_compactor":
            return "l1_leverage"
        if method_key == "attention_norm":
            return "value_l2_norm"
        if method_key == "attention_recency":
            return "recency"
        if method_key in HYBRID_METHODS:
            return "l1_leverage"
        return None

    def generate_with_cache(
        self,
        prompt_ids: List[int],
        method: str,
        budget: int,
        oracle_positions: Optional[List[int]] = None,
    ):
        import mlx.core as mx

        self.reset_attention_state()
        self.attention_state["enabled"] = method in METHODS_NEED_ATTENTION
        self.attention_state["current_method"] = method
        cache = self.make_cache(method, budget)
        evictor = None
        if method in MANUAL_COMPACT_METHODS:
            evictor = MLXCacheEvictor(
                method,
                budget,
                self.cfg,
                len(cache),
                attention_state=self.attention_state,
                oracle_positions=oracle_positions,
            )

        t_start = time.perf_counter()
        prefill_time = self.prefill(prompt_ids[:-1], cache, evictor, budget)
        if evictor:
            evictor.set_phase("decode")
        self.attention_state["phase"] = "decode"
        current = int(prompt_ids[-1])
        generated: List[int] = []
        kv_lens: List[int] = []
        decode_time = 0.0
        eviction_time = 0.0
        for _ in range(max(0, self.cfg.benchmark.max_new_tokens)):
            if evictor and method not in PREFILL_COMPRESS_METHODS:
                t0 = time.perf_counter()
                evictor.evict_for_space(cache, 1)
                eviction_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            logits = self.model(mx.array([[current]]), cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            decode_time += time.perf_counter() - t0
            if evictor:
                evictor.sync_maps(cache)
                if method not in PREFILL_COMPRESS_METHODS:
                    t0 = time.perf_counter()
                    evictor.evict(cache, budget)
                    eviction_time += time.perf_counter() - t0
            token = int(next_token.item())
            generated.append(token)
            kv_lens.append(self.cache_len(cache))
            current = token
            if token in getattr(self.tokenizer, "eos_token_ids", set()):
                break

        if evictor:
            evictor.sync_maps(cache)
            selected = evictor.last_selected
            scores = evictor.last_scores
            component_sources = evictor.last_component_sources
            selected_by_head = evictor.last_selected_by_head
            profile = evictor.profile_times
            eviction_count = evictor.eviction_count
            score_update_count = evictor.score_update_count
        else:
            total_consumed = len(prompt_ids) + len(generated)
            if method in ("full", "basic", "basic_generate"):
                keep = list(range(total_consumed))
            elif method == "sink_recent":
                if total_consumed <= budget:
                    keep = list(range(total_consumed))
                else:
                    sink = min(int(self.cfg.eviction.sink_size), max(0, budget))
                    recent_budget = max(0, int(budget) - sink)
                    recent_start = max(sink, total_consumed - recent_budget)
                    keep = sorted(
                        set(range(sink)) | set(range(recent_start, total_consumed))
                    )
            else:
                keep = list(range(max(0, total_consumed - budget), total_consumed))
            selected = {i: keep for i in range(len(cache))}
            scores = {}
            component_sources = {}
            selected_by_head = {}
            profile = {"score_time_s": 0.0, "topk_time_s": 0.0, "cache_rebuild_time_s": 0.0}
            eviction_count = max(0, total_consumed - budget)
            score_update_count = 0

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        total = time.perf_counter() - t_start
        decode_tokens = max(1, len(generated))
        return {
            "generated_token_ids": generated,
            "generated_text": text,
            "selected_tokens_by_layer": {str(k): v for k, v in selected.items()},
            "scores_by_layer": {str(k): v for k, v in scores.items()},
            "selected_token_sources": {str(k): v for k, v in component_sources.items()},
            "selected_tokens_by_head": {str(k): v for k, v in selected_by_head.items()},
            "cache_budget_scope": "prompt_prefill" if method in PREFILL_COMPRESS_METHODS else "total_kv",
            "prefill_compression": bool(method in PREFILL_COMPRESS_METHODS),
            "sparse_head_mask": bool(method == "compactor"),
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "eviction_time_s": eviction_time,
            "score_time_s": profile.get("score_time_s", 0.0),
            "topk_time_s": profile.get("topk_time_s", 0.0),
            "cache_rebuild_time_s": profile.get("cache_rebuild_time_s", 0.0),
            "total_time_s": total,
            "tokens_per_second": len(generated) / decode_time if decode_time > 0 else 0.0,
            "avg_ms_per_token": (decode_time / decode_tokens) * 1000.0,
            "max_kv_len": max(kv_lens) if kv_lens else self.cache_len(cache),
            "final_kv_len": self.cache_len(cache),
            "avg_kv_len": sum(kv_lens) / len(kv_lens) if kv_lens else self.cache_len(cache),
            "cache_shape_summary": self.cache_shape_summary(cache),
            "eviction_count": eviction_count,
            "score_update_count": score_update_count,
            "score_phase_counts": (
                dict(evictor.score_phase_counts) if evictor else {"prefill": 0, "decode": 0}
            ),
            "score_refit_count": (
                int(evictor.score_refit_count) if evictor else 0
            ),
            "score_refit_phase_counts": (
                dict(evictor.score_refit_phase_counts) if evictor else {"prefill": 0, "decode": 0}
            ),
            "attention_hook_errors": int(self.attention_state.get("hook_errors", 0)),
        }

    def teacher_forced_ppl(
        self,
        full_ids: List[int],
        answer_positions: List[int],
        method: str,
        budget: int,
        oracle_positions: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        import mlx.core as mx

        if not answer_positions:
            return {"ppl": None, "mean_nll": None, "ppl_time_s": 0.0}
        self.reset_attention_state()
        self.attention_state["enabled"] = method in METHODS_NEED_ATTENTION
        self.attention_state["current_method"] = method
        answer_positions = answer_positions[: max(1, self.cfg.benchmark.max_new_tokens)]
        answer_start = min(answer_positions)
        if answer_start <= 0:
            return {"ppl": None, "mean_nll": None, "ppl_time_s": 0.0}
        prompt_prefix = full_ids[: answer_start]
        cache = self.make_cache(method, budget)
        evictor = None
        if method in MANUAL_COMPACT_METHODS:
            evictor = MLXCacheEvictor(
                method,
                budget,
                self.cfg,
                len(cache),
                attention_state=self.attention_state,
                oracle_positions=oracle_positions,
            )
        t_start = time.perf_counter()
        self.prefill(prompt_prefix[:-1], cache, evictor, budget)
        if evictor:
            evictor.set_phase("decode")
        self.attention_state["phase"] = "decode"
        current = int(prompt_prefix[-1])
        nlls: List[float] = []
        for pos in answer_positions:
            target = int(full_ids[pos])
            if evictor and method not in PREFILL_COMPRESS_METHODS:
                evictor.evict_for_space(cache, 1)
            logits = self.model(mx.array([[current]]), cache=cache)
            log_probs = logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(log_probs)
            nlls.append(-float(log_probs[0, target].item()))
            if evictor:
                evictor.sync_maps(cache)
                if method not in PREFILL_COMPRESS_METHODS:
                    evictor.evict(cache, budget)
            current = target
        mean_nll = sum(nlls) / len(nlls) if nlls else None
        return {
            "ppl": math.exp(mean_nll) if mean_nll is not None else None,
            "mean_nll": mean_nll,
            "ppl_time_s": time.perf_counter() - t_start,
        }

    def prefill(
        self,
        token_ids: List[int],
        cache: List[Any],
        evictor: Optional[MLXCacheEvictor],
        budget: int,
    ) -> float:
        import mlx.core as mx

        if not token_ids:
            return 0.0
        elapsed = 0.0
        step = max(1, int(self.cfg.model.prefill_step_size))
        if evictor:
            evictor.set_phase("prefill")
        self.attention_state["phase"] = "prefill"
        for start in range(0, len(token_ids), step):
            chunk = token_ids[start : start + step]
            t0 = time.perf_counter()
            logits = self.model(mx.array([chunk]), cache=cache)
            mx.eval(logits)
            elapsed += time.perf_counter() - t0
            if evictor:
                evictor.sync_maps(cache)
                if evictor.method not in PREFILL_COMPRESS_METHODS:
                    evictor.evict(cache, budget)
        if evictor and evictor.method in PREFILL_COMPRESS_METHODS:
            t0 = time.perf_counter()
            evictor.prefill_compress(cache, budget)
            elapsed += time.perf_counter() - t0
        return elapsed

    def make_cache(self, method: str, budget: int) -> List[Any]:
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        num_layers = len(self.model.model.layers)
        if method == "recency":
            return [RotatingKVCache(max_size=int(budget), keep=0) for _ in range(num_layers)]
        if method == "sink_recent":
            keep = min(self.cfg.eviction.sink_size, max(0, int(budget) - 1))
            return [RotatingKVCache(max_size=int(budget), keep=keep) for _ in range(num_layers)]
        return [KVCache() for _ in range(num_layers)]

    @staticmethod
    def cache_len(cache: List[Any]) -> int:
        if not cache:
            return 0
        return int(max(len(c) for c in cache))

    @staticmethod
    def cache_shape_summary(cache: List[Any]) -> Dict[str, Any]:
        layers = []
        for idx, c in enumerate(cache):
            if getattr(c, "keys", None) is None:
                continue
            layers.append(
                {
                    "layer": idx,
                    "offset": int(getattr(c, "offset", 0)),
                    "logical_offset": int(
                        getattr(c, "logical_offset", getattr(c, "offset", 0))
                    ),
                    "len": int(len(c)),
                    "keys_shape": [int(x) for x in c.keys.shape],
                    "values_shape": [int(x) for x in c.values.shape],
                }
            )
        return {"num_layers": len(layers), "layers": layers}

    def sample_tokens(self, sample: Dict[str, Any]) -> Tuple[List[int], List[int], List[int], str]:
        full_ids = tensor_to_list(sample.get("input_ids"))
        answer_positions = [int(x) for x in sample.get("answer_positions") or sample.get("eval_positions") or []]
        prompt_text = sample.get("prompt")
        if prompt_text:
            prompt_text = apply_prompt_format(self.hf_tokenizer, prompt_text, self.cfg.model)
            prompt_ids = [int(x) for x in self.hf_tokenizer.encode(prompt_text)]
        elif answer_positions:
            prompt_ids = full_ids[: min(answer_positions)]
            prompt_text = self.hf_tokenizer.decode(prompt_ids)
        else:
            prompt_ids = full_ids
            prompt_text = self.hf_tokenizer.decode(prompt_ids)
        if not full_ids and "full_text" in sample:
            full_ids = [int(x) for x in self.hf_tokenizer.encode(sample["full_text"])]
        return prompt_ids, full_ids, answer_positions, prompt_text

    def _actual_budget(
        self,
        sample: Dict[str, Any],
        budget: int,
        budget_ratios: List[float],
    ) -> int:
        if not budget_ratios:
            return int(budget)
        full_ids = tensor_to_list(sample.get("input_ids"))
        actual = int(budget)
        for ratio in budget_ratios:
            actual = max(1, int(len(full_ids) * float(ratio)))
        return actual

    def save_selected_and_scores(
        self,
        out_dir: Path,
        method: str,
        budget: int,
        sample_idx: int,
        selected: Dict[str, List[int]],
        scores: Dict[str, List[float]],
    ) -> Dict[str, Optional[str]]:
        paths: Dict[str, Optional[str]] = {"selected_tokens_path": None, "scores_path": None}
        if self.cfg.save_selected_tokens:
            path = out_dir / "selected_tokens" / f"{method}_b{budget}_s{sample_idx}.json"
            save_results(selected, path)
            paths["selected_tokens_path"] = str(path)
        if self.cfg.save_scores and scores:
            path = out_dir / "scores" / f"{method}_b{budget}_s{sample_idx}.json"
            save_results(scores, path)
            paths["scores_path"] = str(path)
        return paths

    @staticmethod
    def evidence_stats(
        selected: Dict[str, List[int]],
        evidence_positions: List[int],
    ) -> Dict[str, Any]:
        evidence = set(int(x) for x in evidence_positions)
        all_selected = set()
        for values in selected.values():
            all_selected.update(int(x) for x in values)
        overlap = evidence & all_selected
        recall = len(overlap) / len(evidence) if evidence else 0.0
        precision = len(overlap) / len(all_selected) if all_selected else 0.0
        return {
            "evidence_recall": recall,
            "evidence_precision": precision,
            "evidence_overlap_count": len(overlap),
        }

    def selected_token_texts(
        self,
        selected: Dict[str, List[int]],
        stream_ids: List[int],
        limit: int = 256,
    ) -> List[Dict[str, Any]]:
        union = sorted({int(x) for vals in selected.values() for x in vals})
        rows = []
        for pos in union[:limit]:
            if 0 <= pos < len(stream_ids):
                text = self.tokenizer.decode([stream_ids[pos]], skip_special_tokens=False)
                rows.append({"position": pos, "text": text, "type": token_type(text)})
        return rows

    def selected_token_types(
        self,
        selected: Dict[str, List[int]],
        stream_ids: List[int],
    ) -> Dict[str, int]:
        counts: Counter = Counter()
        union = sorted({int(x) for vals in selected.values() for x in vals})
        for pos in union:
            if 0 <= pos < len(stream_ids):
                text = self.tokenizer.decode([stream_ids[pos]], skip_special_tokens=False)
                counts[token_type(text)] += 1
        return dict(counts)

    @staticmethod
    def distances_to_target(
        selected: Dict[str, List[int]],
        target_pos: Optional[int],
        limit: int = 512,
    ) -> List[int]:
        if target_pos is None:
            return []
        union = sorted({int(x) for vals in selected.values() for x in vals})
        return [abs(pos - int(target_pos)) for pos in union[:limit]]

    @staticmethod
    def distances_to_evidence(
        selected: Dict[str, List[int]],
        evidence: List[int],
        limit: int = 512,
    ) -> List[int]:
        if not evidence:
            return []
        ev = [int(x) for x in evidence]
        union = sorted({int(x) for vals in selected.values() for x in vals})
        return [min(abs(pos - e) for e in ev) for pos in union[:limit]]

    @staticmethod
    def oracle_positions_for_sample(sample: Dict[str, Any], method_key: str) -> List[int]:
        if method_key == "oracle_evidence":
            return [int(x) for x in sample.get("evidence_positions") or []]
        if method_key == "oracle_answer_region":
            metadata = sample.get("metadata", {}) or {}
            start = metadata.get("answer_token_start")
            end = metadata.get("answer_token_end")
            if start is not None and end is not None:
                return list(range(int(start), int(end)))
            return [int(x) for x in sample.get("answer_positions") or sample.get("eval_positions") or []]
        return []

    @staticmethod
    def score_stats_from_layers(scores: Dict[str, List[float]]) -> Dict[str, Any]:
        values: List[float] = []
        for layer_values in (scores or {}).values():
            values.extend(float(x) for x in layer_values)
        return list_stats(values)

    @staticmethod
    def normalized_score_stats_from_layers(
        scores: Dict[str, List[float]],
        normalization: str,
    ) -> Dict[str, Any]:
        values: List[float] = []
        for layer_values in (scores or {}).values():
            values.extend(float(x) for x in layer_values)
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return {"numel": int(arr.size), "all_non_finite": True}
        mode = str(normalization or "none").lower()
        out = np.zeros_like(arr)
        vals = arr[finite]
        if mode == "minmax":
            denom = max(float(vals.max() - vals.min()), 1e-8)
            out[finite] = (vals - vals.min()) / denom
        elif mode == "zscore":
            out[finite] = (vals - vals.mean()) / max(float(vals.std()), 1e-8)
        elif mode == "softmax":
            shifted = vals - vals.max()
            exp = np.exp(shifted)
            out[finite] = exp / max(float(exp.sum()), 1e-8)
        elif mode == "rank":
            order = np.argsort(vals)
            ranks = np.zeros_like(vals)
            if vals.size > 1:
                ranks[order] = np.arange(vals.size, dtype=np.float32) / float(vals.size - 1)
            else:
                ranks[order] = 1.0
            out[finite] = ranks
        else:
            out[finite] = vals
        return list_stats(out.tolist())

    @staticmethod
    def vector_shape_summary(cache_shape_summary: Dict[str, Any], method_key: str) -> Optional[Dict[str, Any]]:
        if method_key not in {
            "l1_leverage",
            "l1_prefill_only",
            "l1_decode_only",
            "l2_leverage",
            "l2_prefill_only",
            "l2_key_prefill_only",
            "l2_decode_only",
            "compactor",
            "key_l2_norm",
            "value_l2_norm",
            "key_l1_norm",
            "value_l1_norm",
            "sink_recent_l1",
            "sink_recent_l2",
            "attention_l1",
            "attention_l2",
        }:
            return None
        layers = (cache_shape_summary or {}).get("layers") or []
        if not layers:
            return None
        first = layers[0]
        return {
            "keys_shape": first.get("keys_shape"),
            "values_shape": first.get("values_shape"),
            "num_layers": len(layers),
        }

    def skipped_result(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        method: str,
        budget: int,
        reason: str,
        spec: Any = None,
    ) -> Dict[str, Any]:
        metadata = sample.get("metadata", {}) or {}
        prompt = sample.get("prompt")
        if spec is None:
            try:
                spec = get_method_spec(method)
            except Exception:
                spec = None
        return {
            "label": f"{method}_b{budget}_s{sample_idx}",
            "experiment_name": self.cfg.experiment_name,
            "run_id": self.cfg.run_id,
            "sample_id": sample_idx,
            "sample_idx": sample_idx,
            "method": method,
            "canonical_method": canonical_method(method),
            "method_family": getattr(spec, "family", "unknown"),
            "budget": budget,
            "cache_budget": budget,
            "model": self.cfg.model.name,
            "model_name": self.cfg.model.name,
            "model_family": self.model_info.get("model_family"),
            "backend": "mlx",
            "quant_bits": self.cfg.model.quant_bits,
            "benchmark": self.cfg.benchmark.name,
            "context_length": metadata.get("seq_len"),
            "prompt_hash": text_hash(prompt),
            "prediction": None,
            "generated_text": None,
            "ground_truth": sample.get("ground_truth"),
            "correct": None,
            "contains_ground_truth": None,
            "exact_match": None,
            "answer_f1": None,
            "ppl": None,
            "mean_nll": None,
            "official_score": None,
            "official_correct": None,
            "official_metric_name": metadata.get("official_metric_name"),
            "official_metric_implementation": None,
            "dataset_official": metadata.get("dataset_official"),
            "primary_metric": None,
            "primary_score": None,
            "evidence_positions": sample.get("evidence_positions") or [],
            "selected_tokens": {},
            "selected_tokens_by_layer": {},
            "evidence_recall": None,
            "evidence_precision": None,
            "score_stats": {},
            "score_normalization": self.cfg.eviction.score_normalization,
            "seed": self.cfg.seed,
            "score_update_count": 0,
            "max_kv_len": None,
            "final_kv_len": None,
            "cache_shape_summary": {},
            "total_time_s": 0.0,
            "prefill_time_s": 0.0,
            "decode_time_s": 0.0,
            "score_time_s": 0.0,
            "eviction_time_s": 0.0,
            "topk_time_s": 0.0,
            "cache_rebuild_time_s": 0.0,
            "tokens_per_second": None,
            "skipped": True,
            "skipped_reason": reason,
            "unsupported_reason": reason,
            "oracle": bool(getattr(spec, "oracle", False)),
            "metadata": metadata,
        }

    def error_result(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        method: str,
        budget: int,
        exc: Exception,
    ) -> Dict[str, Any]:
        prompt = sample.get("prompt")
        return {
            "label": f"{method}_b{budget}_s{sample_idx}",
            "sample_id": sample_idx,
            "sample_idx": sample_idx,
            "method": method,
            "budget": budget,
            "model": self.cfg.model.name,
            "model_name": self.cfg.model.name,
            "backend": "mlx",
            "quant_bits": self.cfg.model.quant_bits,
            "benchmark": self.cfg.benchmark.name,
            "prompt_hash": text_hash(prompt),
            "ground_truth": sample.get("ground_truth"),
            "evidence_positions": sample.get("evidence_positions"),
            "metadata": sample.get("metadata", {}),
            "error": str(exc),
        }
