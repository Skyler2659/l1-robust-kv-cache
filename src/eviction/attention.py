"""Attention-based eviction — accumulate attention scores across steps."""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


class AttentionEviction(BaseEviction):
    """Select tokens by cumulative attention weight.

    Uses ``shared_q.LAST_QUERY_STATES`` to reconstruct current-step attention
    and accumulates it across decoding steps.
    """
    name = "attention"
    method_family = "attention"
    requires_attention = True
    requires_scores = True
    score_source = "attention"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._acc_scores: dict[int, torch.Tensor] = {}

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        seq_len = layer_k.size(self.k_seq_dim)
        scores = self._acc_scores.get(layer_idx)
        if scores is not None and scores.numel() >= seq_len:
            return scores[:seq_len]
        self._accumulate(layer_k, layer_v, layer_idx, seq_len)
        scores = self._acc_scores.get(layer_idx)
        if scores is not None and scores.numel() >= seq_len:
            return scores[:seq_len]
        return None

    def update_attention(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        if attention_weights is None:
            return
        attn = attention_weights.detach()
        if attn.dim() == 4:
            # [batch, heads, query_len, key_len]
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

    def _accumulate(self, layer_k, layer_v, layer_idx, seq_len):
        attn = self._compute_attn(layer_k, layer_v, layer_idx)
        if attn is None:
            return
        prev = self._acc_scores.get(layer_idx)
        if prev is None or prev.numel() < seq_len:
            new = torch.zeros(seq_len, device=layer_k.device, dtype=attn.dtype)
            if prev is not None:
                new[: prev.numel()] = prev
            prev = new
        prev[:seq_len] += attn.to(prev.device)
        self._acc_scores[layer_idx] = prev

    def _compute_attn(self, layer_k, layer_v, layer_idx) -> Optional[torch.Tensor]:
        import shared_q
        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is None:
            return None
        seq_len = layer_k.size(self.k_seq_dim)
        head_dim = layer_v.shape[-1]
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if k_rows is None:
            return None
        q_vec = q_h.mean(dim=0).to(layer_v.device, dtype=torch.float32)
        k_rows = k_rows.to(layer_v.device, dtype=torch.float32)
        if k_rows.shape[-1] != q_vec.numel():
            return None
        logits = torch.matmul(q_vec, k_rows.T) / max(head_dim ** 0.5, 1e-6)
        return torch.softmax(logits, dim=0)

    def reset(self):
        super().reset()
        self._acc_scores.clear()


class AccumulatedAttentionEviction(AttentionEviction):
    """Explicit accumulated-attention alias for H2O-style saliency."""

    name = "accumulated_attention"


class LastTokenAttentionEviction(AttentionEviction):
    """Use only the most recent query-token attention distribution."""

    name = "last_token_attention"

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
        self._acc_scores[layer_idx] = pooled

    def _accumulate(self, layer_k, layer_v, layer_idx, seq_len):
        attn = self._compute_attn(layer_k, layer_v, layer_idx)
        if attn is not None:
            self._acc_scores[layer_idx] = attn[:seq_len]


class WindowedAttentionEviction(AttentionEviction):
    """Accumulate attention over the most recent W decode steps."""

    name = "windowed_attention"

    def __init__(self, attention_window: int = 32, window_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.attention_window = max(1, int(window_size or attention_window))
        self._windows: Dict[int, List[torch.Tensor]] = {}

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
        buf = self._windows.setdefault(layer_idx, [])
        buf.append(pooled)
        if len(buf) > self.attention_window:
            del buf[:-self.attention_window]
        self._acc_scores[layer_idx] = self._sum_window(buf)

    def _accumulate(self, layer_k, layer_v, layer_idx, seq_len):
        attn = self._compute_attn(layer_k, layer_v, layer_idx)
        if attn is None:
            return
        buf = self._windows.setdefault(layer_idx, [])
        buf.append(attn)
        if len(buf) > self.attention_window:
            del buf[:-self.attention_window]
        self._acc_scores[layer_idx] = self._sum_window(buf)

    @staticmethod
    def _sum_window(buf: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(t.numel() for t in buf)
        out = torch.zeros(max_len, device=buf[-1].device, dtype=buf[-1].dtype)
        for tensor in buf:
            out[: tensor.numel()] += tensor.to(out.device, dtype=out.dtype)
        return out

    def reset(self):
        super().reset()
        self._windows.clear()


class AttentionDecayEviction(AttentionEviction):
    """Accumulated attention with exponential time decay."""

    name = "attention_decay"

    def __init__(self, decay_gamma: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.decay_gamma = float(decay_gamma)

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
        self._decayed_update(layer_idx, pooled)

    def _accumulate(self, layer_k, layer_v, layer_idx, seq_len):
        attn = self._compute_attn(layer_k, layer_v, layer_idx)
        if attn is not None:
            self._decayed_update(layer_idx, attn)

    def _decayed_update(self, layer_idx: int, values: torch.Tensor) -> None:
        seq_len = values.numel()
        prev = self._acc_scores.get(layer_idx)
        if prev is None or prev.numel() < seq_len:
            new_prev = torch.zeros(seq_len, device=values.device, dtype=values.dtype)
            if prev is not None:
                new_prev[: prev.numel()] = prev.to(new_prev.device, dtype=new_prev.dtype)
            prev = new_prev
        prev[:seq_len] = prev[:seq_len] * self.decay_gamma + values.to(prev.device)
        self._acc_scores[layer_idx] = prev
