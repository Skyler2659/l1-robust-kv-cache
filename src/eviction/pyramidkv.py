"""PyramidKV — layer-wise budget allocation (Cai et al., 2025)."""
from __future__ import annotations
from typing import Any, List, Tuple
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import (
    to_legacy_cache, back_to_original, get_kv_seq_len, gather_by_dim, mean_heads,
)


class PyramidKVEviction(BaseEviction):
    """PyramidKV: different cache budgets per layer.

    Modes:
    - ``funnel``: more budget in lower layers (information funnel down)
    - ``inverse_funnel``: more budget in upper layers
    - ``uniform``: equal budget (degenerates to attention-only)
    """
    name = "pyramidkv"
    method_family = "attention"
    requires_attention = True
    requires_scores = True
    supports_layerwise = True
    score_source = "accumulated_attention"
    experimental = True

    def __init__(self, pyramid_mode="funnel", **kwargs):
        super().__init__(**kwargs)
        self.pyramid_mode = pyramid_mode
        self._acc_scores: dict[int, torch.Tensor] = {}

    def __call__(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        pkv, original = to_legacy_cache(past_key_values)
        if pkv is None:
            return None
        num_layers = len(pkv)
        budgets = self._layer_budgets(num_layers)
        items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, (k, v) in enumerate(pkv):
            seq_len = get_kv_seq_len(k, self.k_seq_dim)
            self._ensure_position_map(layer_idx, seq_len, k.device)
            layer_budget = min(seq_len, budgets[layer_idx])
            if seq_len <= layer_budget:
                self.last_selected[layer_idx] = self._position_maps[layer_idx].detach().cpu()
                items.append((k, v))
                continue
            scores = self._compute_and_accumulate(k, v, layer_idx, seq_len)
            keep = self._select(scores, seq_len, layer_budget, k.device)
            if self.debug_budget:
                from src.eviction.base import validate_selected_indices
                validate_selected_indices(keep.detach().cpu(), seq_len, layer_budget)
            self.last_selected[layer_idx] = self._record_selected_positions(
                layer_idx, keep, k.device
            ).detach().cpu()
            new_k = gather_by_dim(k, self.k_seq_dim, keep.to(k.device))
            new_v = gather_by_dim(v, self.v_seq_dim, keep.to(v.device))
            items.append((new_k, new_v))
        self._steps += 1
        return back_to_original(original, items)

    def _layer_budgets(self, num_layers: int) -> List[int]:
        total = self.cache_size * num_layers
        if self.pyramid_mode == "uniform":
            return [self.cache_size] * num_layers
        elif self.pyramid_mode == "funnel":
            weights = torch.arange(num_layers, 0, -1, dtype=torch.float32)
        elif self.pyramid_mode == "inverse_funnel":
            weights = torch.arange(1, num_layers + 1, dtype=torch.float32)
        else:
            return [self.cache_size] * num_layers
        weights = weights / weights.sum()
        raw = (weights * total).int()
        raw = raw.clamp(min=4, max=self.cache_size)
        return raw.tolist()

    def _compute_and_accumulate(self, k, v, layer_idx, seq_len):
        import shared_q
        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is not None:
            k_rows = mean_heads(k, self.k_seq_dim)
            if k_rows is not None:
                head_dim = v.shape[-1]
                q_vec = q_h.mean(dim=0).to(v.device, dtype=torch.float32)
                k_rows = k_rows.to(v.device, dtype=torch.float32)
                if k_rows.shape[-1] == q_vec.numel():
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
        scores = self._acc_scores.get(layer_idx)
        if scores is not None and scores.numel() >= seq_len:
            return scores[:seq_len]
        return None

    def _select(self, scores, seq_len, budget, device):
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
        heavy = torch.topk(masked, topk).indices
        return self._ensure_budget(
            torch.cat([reserved, heavy]),
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        return None

    def select_indices(self, scores, seq_len, budget, device):
        return self._select(scores, seq_len, budget, device)

    def reset(self):
        super().reset()
        self._acc_scores.clear()
