"""SnapKV eviction — observation-window attention selection (Li et al., 2024)."""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


class SnapKVEviction(BaseEviction):
    """SnapKV: observation window attention + max-pool selection."""
    name = "snapkv"
    method_family = "attention"
    requires_attention = True
    requires_scores = True
    score_source = "observation_window_attention"
    approximate = True

    def __init__(self, window_size=32, kernel_size=63, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self._observe_q: Dict[int, List[torch.Tensor]] = {}

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        self._record_query(layer_idx)
        seq_len = layer_k.size(self.k_seq_dim)
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if k_rows is None:
            return None
        q_obs = self._get_observe(layer_idx, layer_k.device, layer_k.dtype)
        if q_obs is None:
            return torch.norm(k_rows, p=1, dim=1)
        obs = min(self.window_size, q_obs.shape[0], seq_len)
        head_dim = k_rows.shape[-1]
        dist = (
            torch.arange(0, obs, device=k_rows.device)[:, None]
            - torch.arange(0, seq_len, device=k_rows.device)[None, :]
            + seq_len - obs
        )
        mask = dist >= 0
        logits = (q_obs[-obs:] @ k_rows.T) / max(head_dim ** 0.5, 1e-6)
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1).masked_fill(~mask, 0.0)
        score = attn[:, :-obs].sum(dim=0) if attn.shape[1] > obs else attn.sum(dim=0)
        if score.numel() == 0:
            return torch.norm(k_rows, p=1, dim=1)
        score = score.unsqueeze(0).unsqueeze(0)
        ks = min(self.kernel_size, score.shape[-1])
        if ks > 1 and ks % 2 == 0:
            ks -= 1
        if ks > 1:
            score = F.max_pool1d(score, kernel_size=ks, padding=ks // 2, stride=1)
        return score.squeeze()

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        if scores is None:
            return self._fill_budget(reserved, seq_len, budget, device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return self._ensure_budget(reserved, seq_len, budget, device, reserved=reserved)
        obs = min(self.window_size, seq_len)
        hist_len = max(0, seq_len - obs)
        hist_scores = scores[:hist_len] if hist_len > 0 else scores.new_zeros(0)
        if hist_scores.numel() > 0:
            masked = hist_scores.clone().to(device)
            if reserved.numel() > 0:
                r_in_hist = reserved[reserved < hist_len]
                if r_in_hist.numel() > 0:
                    masked[r_in_hist] = -float("inf")
            valid = torch.isfinite(masked)
            if valid.any():
                keep_hist = min(fill, int(valid.sum().item()))
                hist_idx = torch.topk(masked, keep_hist).indices.sort().values
            else:
                hist_idx = torch.arange(max(0, hist_len - fill), hist_len, device=device)
        else:
            hist_idx = torch.empty(0, dtype=torch.long, device=device)
        recent_idx = torch.arange(max(0, seq_len - obs), seq_len, device=device)
        keep = torch.cat([reserved, hist_idx, recent_idx]).unique(sorted=True)
        return self._ensure_budget(
            keep,
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )

    def _record_query(self, layer_idx):
        import shared_q
        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is None:
            return
        buf = self._observe_q.setdefault(layer_idx, [])
        buf.append(q_h.detach())
        if len(buf) > max(1, self.window_size):
            del buf[:-self.window_size]

    def _get_observe(self, layer_idx, device, dtype):
        buf = self._observe_q.get(layer_idx)
        if not buf:
            return None
        stacked = torch.stack(buf, dim=0)
        if stacked.dim() == 3:
            stacked = stacked.mean(dim=1)
        return stacked.to(device=device, dtype=dtype)

    def reset(self):
        super().reset()
        self._observe_q.clear()
