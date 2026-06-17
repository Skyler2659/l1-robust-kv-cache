"""Base class for all KV cache eviction strategies.

Every eviction method inherits from BaseEviction and implements at minimum
``compute_scores`` and ``select_indices``.  The ``__call__`` method wires
scoring into actual cache compression.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.eviction.score_normalization import merge_score_stats


def validate_selected_indices(
    selected: torch.Tensor,
    seq_len: int,
    budget: int,
    reserved: Optional[torch.Tensor] = None,
) -> None:
    """Validate a keep-index tensor for KV cache eviction.

    When ``seq_len >= budget`` the selected set must have exactly ``budget``
    unique legal positions. When the sequence is shorter than the budget, all
    positions must be kept.
    """
    if selected.dim() != 1:
        raise ValueError(f"selected must be 1-D, got shape={tuple(selected.shape)}")
    if selected.dtype != torch.long:
        raise ValueError(f"selected must be torch.long, got dtype={selected.dtype}")
    expected = min(int(seq_len), int(budget))
    if selected.numel() != expected:
        raise ValueError(
            f"selected count mismatch: got {selected.numel()}, expected {expected} "
            f"(seq_len={seq_len}, budget={budget})"
        )
    if selected.numel() == 0:
        return
    if selected.min().item() < 0 or selected.max().item() >= seq_len:
        raise ValueError(f"selected indices out of range for seq_len={seq_len}")
    unique = selected.unique(sorted=True)
    if unique.numel() != selected.numel():
        raise ValueError("selected indices contain duplicates")
    if not torch.equal(selected.cpu(), unique.cpu()):
        raise ValueError("selected indices must be sorted ascending")
    if reserved is not None and reserved.numel() > 0 and expected >= reserved.numel():
        reserved = reserved.to(selected.device, dtype=torch.long)
        missing = reserved[~torch.isin(reserved, selected)]
        if missing.numel() > 0:
            raise ValueError(f"reserved indices missing from selected: {missing.tolist()}")


class BaseEviction(ABC):
    """Unified interface for KV cache eviction strategies.

    Subclasses implement ``compute_scores`` (returning a per-token importance
    score) and ``select_indices`` (mapping scores + budget → keep indices).

    Convention:
        * Higher score → more important → more likely to be **kept**.
        * ``k_seq_dim`` / ``v_seq_dim`` indicate which tensor axis is the
          sequence dimension for K / V respectively.
    """

    name = "base"
    method_family = "unknown"
    supports_backends = ("torch",)
    requires_attention = False
    requires_scores = False
    supports_layerwise = True
    supports_headwise = False
    score_source = None
    score_normalization = "none"
    approximate = False
    experimental = False
    oracle = False

    def __init__(
        self,
        cache_size: int,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
        sink_size: int = 0,
        recent_size: int = 0,
        debug_budget: bool = False,
    ):
        self.cache_size = int(cache_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.sink_size = max(0, int(sink_size))
        self.recent_size = max(0, int(recent_size))
        self.debug_budget = bool(debug_budget)
        self._steps = 0
        self._position_maps: Dict[int, torch.Tensor] = {}
        self._next_positions: Dict[int, int] = {}
        self.profile_times: Dict[str, float] = {
            "score_compute": 0.0,
            "topk_select": 0.0,
            "cache_prune": 0.0,
        }
        # Per-layer diagnostic storage (filled by __call__)
        self.last_selected: Dict[int, torch.Tensor] = {}
        self.last_scores: Dict[int, torch.Tensor] = {}

    # ── Abstract interface ──────────────────────────────────────────────

    @abstractmethod
    def compute_scores(
        self,
        layer_k: torch.Tensor,
        layer_v: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Return a 1-D tensor of shape ``[seq_len]`` with importance scores.

        Return ``None`` to skip scoring (e.g. when the cache is small enough).
        Higher score ⇒ more important ⇒ more likely to be kept.
        """

    @abstractmethod
    def select_indices(
        self,
        scores: Optional[torch.Tensor],
        seq_len: int,
        budget: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return sorted 1-D LongTensor of indices to keep."""

    # ── Optional hooks ──────────────────────────────────────────────────

    def update_attention(
        self,
        layer_idx: int,
        attention_weights: torch.Tensor,
    ) -> None:
        """Called after each forward pass for attention-accumulating methods."""

    def on_step_start(self, step: int) -> None:
        """Hook called at the beginning of each decoding step."""

    def reset(self) -> None:
        """Reset any accumulated state between samples."""
        self._steps = 0
        self._position_maps.clear()
        self._next_positions.clear()
        self.last_selected.clear()
        self.last_scores.clear()
        for key in self.profile_times:
            self.profile_times[key] = 0.0

    def set_sample_metadata(self, sample: Dict[str, Any]) -> None:
        """Optional hook for methods that use benchmark metadata.

        Oracle methods override this. Regular baselines intentionally ignore it.
        """

    # ── Main eviction entry point ───────────────────────────────────────

    def __call__(self, past_key_values: Any) -> Any:
        """Post-forward cache compression."""
        if past_key_values is None:
            return None
        from src.eviction.kv_utils import to_legacy_cache, back_to_original, get_kv_seq_len

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
            keep = self.select_indices(scores, seq_len, self.cache_size, k.device)
            self.profile_times["topk_select"] += time.perf_counter() - t0
            if self.debug_budget:
                validate_selected_indices(keep.detach().cpu(), seq_len, self.cache_size)
            self.last_selected[layer_idx] = self._record_selected_positions(
                layer_idx, keep, k.device
            ).detach().cpu()
            if scores is not None:
                self.last_scores[layer_idx] = scores.detach().cpu()

            from src.eviction.kv_utils import gather_by_dim
            t0 = time.perf_counter()
            new_k = gather_by_dim(k, self.k_seq_dim, keep.to(k.device))
            new_v = gather_by_dim(v, self.v_seq_dim, keep.to(v.device))
            self.profile_times["cache_prune"] += time.perf_counter() - t0
            items.append((new_k, new_v))

        self._steps += 1
        return back_to_original(original, items)

    def evict_for_space(self, past_key_values: Any, num_coming: int) -> Any:
        """Pre-allocate eviction: make room for *num_coming* new tokens."""
        if past_key_values is None:
            return None
        from src.eviction.kv_utils import to_legacy_cache, get_kv_seq_len

        pkv, _ = to_legacy_cache(past_key_values)
        if pkv is None:
            return None
        seq_len = get_kv_seq_len(pkv[0][0], self.k_seq_dim)
        budget = max(1, self.cache_size - max(0, int(num_coming)))
        if seq_len <= budget:
            return past_key_values
        # Re-use __call__ with temporary budget
        old = self.cache_size
        try:
            self.cache_size = budget
            return self.__call__(past_key_values)
        finally:
            self.cache_size = old

    # ── Convenience ─────────────────────────────────────────────────────

    def _reserved_indices(self, seq_len: int, budget: int, device) -> torch.Tensor:
        """Sink + recent indices that are always kept."""
        parts: List[torch.Tensor] = []
        sink = min(self.sink_size, max(0, budget))
        if sink > 0:
            parts.append(torch.arange(0, sink, device=device))
        recent = min(self.recent_size, max(0, budget - sink))
        if recent > 0:
            parts.append(torch.arange(seq_len - recent, seq_len, device=device))
        if not parts:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(parts).unique(sorted=True)

    def _ensure_position_map(self, layer_idx: int, seq_len: int, device) -> torch.Tensor:
        """Maintain current-cache-position -> original-token-position mapping."""
        current = self._position_maps.get(layer_idx)
        if current is None or current.numel() > seq_len:
            current = torch.arange(seq_len, dtype=torch.long, device=device)
            self._next_positions[layer_idx] = int(seq_len)
        elif current.numel() < seq_len:
            start = self._next_positions.get(layer_idx, int(current.max().item()) + 1 if current.numel() else 0)
            extra = torch.arange(
                start,
                start + (seq_len - current.numel()),
                dtype=torch.long,
                device=device,
            )
            current = torch.cat([current.to(device), extra])
            self._next_positions[layer_idx] = start + int(extra.numel())
        else:
            current = current.to(device)
        self._position_maps[layer_idx] = current
        return current

    def _record_selected_positions(
        self, layer_idx: int, keep: torch.Tensor, device
    ) -> torch.Tensor:
        pos_map = self._position_maps.get(layer_idx)
        if pos_map is None:
            pos_map = torch.arange(
                int(keep.max().item()) + 1 if keep.numel() else 0,
                dtype=torch.long,
                device=device,
            )
        pos_map = pos_map.to(device)
        selected_positions = pos_map[keep.to(device)]
        self._position_maps[layer_idx] = selected_positions.detach()
        return selected_positions

    def _fill_budget(
        self,
        reserved: torch.Tensor,
        seq_len: int,
        budget: int,
        device,
        prefer_recent: bool = True,
    ) -> torch.Tensor:
        """Fill remaining budget from non-reserved positions."""
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return self._ensure_budget(
                reserved, seq_len, budget, device, reserved=reserved)
        all_idx = torch.arange(seq_len, device=device)
        if reserved.numel() > 0:
            mask = ~torch.isin(all_idx, reserved)
            candidates = all_idx[mask]
        else:
            candidates = all_idx
        if candidates.numel() <= fill:
            chosen = candidates
        elif prefer_recent:
            chosen = candidates[-fill:]
        else:
            chosen = candidates[:fill]
        return self._ensure_budget(
            torch.cat([reserved, chosen]), seq_len, budget, device, reserved=reserved)

    def _ensure_budget(
        self,
        selected: torch.Tensor,
        seq_len: int,
        budget: int,
        device,
        scores: Optional[torch.Tensor] = None,
        reserved: Optional[torch.Tensor] = None,
        prefer_recent: bool = True,
    ) -> torch.Tensor:
        """Return a legal, unique, sorted keep set with exactly the target size."""
        target = min(int(seq_len), int(budget))
        if target <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if seq_len <= budget:
            result = torch.arange(seq_len, dtype=torch.long, device=device)
            if self.debug_budget:
                validate_selected_indices(result.detach().cpu(), seq_len, budget)
            return result

        selected = selected.to(device=device, dtype=torch.long).flatten()
        selected = selected[(selected >= 0) & (selected < seq_len)].unique(sorted=True)
        if reserved is not None:
            reserved = reserved.to(device=device, dtype=torch.long).flatten()
            reserved = reserved[(reserved >= 0) & (reserved < seq_len)].unique(sorted=True)
            selected = torch.cat([selected, reserved]).unique(sorted=True)

        if selected.numel() < target:
            all_idx = torch.arange(seq_len, device=device)
            candidates = all_idx[~torch.isin(all_idx, selected)]
            fill = target - int(selected.numel())
            if candidates.numel() > 0 and fill > 0:
                if scores is not None and scores.numel() >= seq_len:
                    candidate_scores = scores[:seq_len].to(device=device).float()[candidates]
                    finite = torch.isfinite(candidate_scores)
                    if finite.any():
                        valid_candidates = candidates[finite]
                        valid_scores = candidate_scores[finite]
                        topk = min(fill, valid_candidates.numel())
                        chosen = valid_candidates[torch.topk(valid_scores, topk).indices]
                    else:
                        chosen = candidates[-fill:] if prefer_recent else candidates[:fill]
                else:
                    chosen = candidates[-fill:] if prefer_recent else candidates[:fill]
                selected = torch.cat([selected, chosen]).unique(sorted=True)

        if selected.numel() > target:
            if scores is not None and scores.numel() >= seq_len:
                priority = scores[:seq_len].to(device=device).float()[selected].clone()
            else:
                priority = selected.float() if prefer_recent else -selected.float()
            if reserved is not None and reserved.numel() > 0:
                is_reserved = torch.isin(selected, reserved)
                priority[is_reserved] = float("inf")
            top = torch.topk(priority, target).indices
            selected = selected[top].unique(sorted=True)
            if selected.numel() > target:
                selected = selected[:target]

        if selected.numel() < target:
            # Last-resort deterministic fill after crop/unique edge cases.
            all_idx = torch.arange(seq_len, device=device)
            candidates = all_idx[~torch.isin(all_idx, selected)]
            fill = target - int(selected.numel())
            selected = torch.cat([selected, candidates[-fill:]]).unique(sorted=True)

        result = selected[:target].to(dtype=torch.long)
        if self.debug_budget:
            validate_selected_indices(
                result.detach().cpu(),
                seq_len,
                budget,
                reserved=reserved.detach().cpu() if reserved is not None else None,
            )
        return result

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic info for logging."""
        return {
            "method": getattr(self, "name", self.__class__.__name__),
            "class": self.__class__.__name__,
            "method_family": getattr(self, "method_family", "unknown"),
            "cache_size": self.cache_size,
            "sink_size": self.sink_size,
            "recent_size": self.recent_size,
            "steps": self._steps,
            "requires_attention": bool(getattr(self, "requires_attention", False)),
            "requires_scores": bool(getattr(self, "requires_scores", False)),
            "supports_layerwise": bool(getattr(self, "supports_layerwise", True)),
            "supports_headwise": bool(getattr(self, "supports_headwise", False)),
            "score_source": getattr(self, "score_source", None),
            "score_normalization": getattr(self, "score_normalization", "none"),
            "approximate": bool(getattr(self, "approximate", False)),
            "experimental": bool(getattr(self, "experimental", False)),
            "oracle": bool(getattr(self, "oracle", False)),
        }

    def get_debug_info(self) -> Dict[str, Any]:
        """Return method metadata and timing counters."""
        info = self.get_info()
        info["profile_times"] = dict(self.profile_times)
        info["score_update_count"] = getattr(self, "score_update_count", None)
        return info

    def get_score_stats(self) -> Dict[str, Any]:
        """Return raw/normalized score diagnostics for the latest eviction."""
        return merge_score_stats(
            self.last_scores,
            normalization=getattr(self, "score_normalization", "none"),
        )
