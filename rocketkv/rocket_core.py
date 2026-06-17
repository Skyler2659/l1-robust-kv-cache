"""RocketKV core: SnapKV (stage 1) + Hybrid Sparse Attention selection (stage 2).

Adapted from NVlabs/RocketKV (Behnam et al., ICML 2025) for post-forward KV eviction
in the benchmark harness. See https://arxiv.org/abs/2502.14051.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def get_compression_params(
    token_budget: int,
    seq_len: int,
    head_dim: int,
    max_new_tokens: int = 1,
) -> Tuple[int, int, int, int, int]:
    """Adaptive split of compression ratio across RocketKV stages (paper §3.6)."""
    token_budget = max(1, min(int(seq_len), int(token_budget)))
    seq_len = max(1, int(seq_len))
    compression_ratio = max(1.0, float(seq_len) / float(token_budget))
    alpha = min(0.2 + math.log2(compression_ratio) * 0.06, 0.8)
    capacity_budget = max(token_budget, int(float(seq_len) / (compression_ratio ** alpha)))
    prompt_budget = min(
        seq_len - max_new_tokens,
        max(token_budget, capacity_budget - max_new_tokens),
    )
    prompt_budget = max(token_budget, prompt_budget)
    k_hsa = min(max(1, round(token_budget / 2)), capacity_budget)
    stage2_ratio = max(1.0, float(capacity_budget) / float(token_budget))
    chunk_size = min(
        max(1, math.floor(stage2_ratio)),
        max(1, math.ceil(math.sqrt(stage2_ratio))),
    )
    r = min(head_dim, max(1, round(head_dim * chunk_size / stage2_ratio)))
    return capacity_budget, prompt_budget, chunk_size, r, k_hsa


def _mean_key_rows(layer_k: torch.Tensor, k_seq_dim: int) -> Optional[torch.Tensor]:
    if layer_k.dim() == 4 and k_seq_dim == 2:
        return layer_k[0].mean(dim=0)
    if layer_k.dim() == 4 and k_seq_dim == 3:
        return layer_k[0].mean(dim=0).transpose(0, 1)
    if layer_k.dim() == 3 and k_seq_dim == 1:
        return layer_k.mean(dim=0)
    return None


def snapkv_keep_indices(
    layer_k: torch.Tensor,
    observe_queries: torch.Tensor,
    prompt_budget: int,
    window_size: int,
    kernel_size: int,
    k_seq_dim: int = 2,
) -> Optional[torch.Tensor]:
    """Stage-1 SnapKV: permanent eviction indices (observation-window attention)."""
    seq_len = layer_k.size(k_seq_dim)
    obs = min(int(window_size), seq_len)
    budget = int(prompt_budget)
    if seq_len <= budget or obs <= 0 or budget < obs:
        return None

    k_rows = _mean_key_rows(layer_k, k_seq_dim)
    if k_rows is None:
        return None
    head_dim = k_rows.shape[-1]
    device, dtype = k_rows.device, k_rows.dtype

    # observe_queries: [T, D], [T, H, D], or [H, T, D] -> [obs, D].
    if observe_queries.dim() == 3:
        if observe_queries.shape[0] >= obs:
            q_observe = observe_queries[-obs:].mean(dim=1)
        elif observe_queries.shape[1] >= obs:
            q_observe = observe_queries[:, -obs:, :].mean(dim=0)
        else:
            return None
    else:
        if observe_queries.shape[0] < obs:
            return None
        q_observe = observe_queries
    q_observe = q_observe[-obs:].to(device=device, dtype=dtype)

    dist = (
        torch.arange(0, obs, device=device)[:, None]
        - torch.arange(0, seq_len, device=device)[None, :]
        + seq_len
        - obs
    )
    attn_mask = dist >= 0
    logits = (q_observe @ k_rows.T) / max(head_dim ** 0.5, 1e-6)
    logits = logits.masked_fill(~attn_mask, float("-inf"))
    score = torch.softmax(logits, dim=-1)
    score = score.masked_fill(~attn_mask, 0.0)
    score = score[:, :-obs].sum(dim=0)
    if score.numel() == 0:
        return None

    score = score.unsqueeze(0).unsqueeze(0)
    score = F.max_pool1d(
        score, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
    ).squeeze(0).squeeze(0)

    keep_hist = max(0, budget - obs)
    if keep_hist <= 0:
        hist_idx = torch.empty(0, dtype=torch.long, device=device)
    else:
        keep_hist = min(keep_hist, score.numel())
        hist_idx = score.topk(keep_hist, dim=-1).indices.sort().values
    recent_idx = torch.arange(seq_len - obs, seq_len, device=device)
    return torch.cat([hist_idx, recent_idx]).sort().values


def build_chunk_max_keys(
    k_rows: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, int]:
    """Paged K_max along sequence (HSA step 1, Quest-style pages)."""
    seq_len, head_dim = k_rows.shape
    chunk_size = max(1, int(chunk_size))
    pad = chunk_size - ((seq_len - 1) % chunk_size + 1)
    if pad == chunk_size:
        pad = 0
    if pad > 0:
        pad_rows = torch.full(
            (pad, head_dim), torch.finfo(k_rows.dtype).min, device=k_rows.device, dtype=k_rows.dtype
        )
        padded = torch.cat([k_rows, pad_rows], dim=0)
    else:
        padded = k_rows
    n_pages = padded.shape[0] // chunk_size
    pages = padded.view(n_pages, chunk_size, head_dim).amax(dim=1)
    return pages, seq_len


def hsa_keep_indices(
    q_vec: torch.Tensor,
    k_rows: torch.Tensor,
    topk: int,
    chunk_size: int,
    r: int,
) -> Optional[torch.Tensor]:
    """Stage-2 HSA: approximate top-k token indices for the current query."""
    seq_len, head_dim = k_rows.shape
    topk = min(int(topk), seq_len)
    if topk <= 0:
        return None

    chunk_size = max(1, int(chunk_size))
    r = min(int(r), head_dim)
    device, dtype = k_rows.device, k_rows.dtype
    q = q_vec.to(device=device, dtype=dtype).view(-1)
    if q.numel() != head_dim:
        return None

    pages, valid_len = build_chunk_max_keys(k_rows, chunk_size)
    sign = torch.where(q >= 0, 1.0, -1.0)
    page_keys = pages * sign

    abs_q = q.abs()
    i1 = torch.topk(abs_q, r, dim=-1).indices
    q_hat = (q * sign)[i1]
    k_hat = page_keys[:, i1]
    page_scores = (q_hat.unsqueeze(0) * k_hat).sum(dim=-1)
    page_scores = page_scores.repeat_interleave(chunk_size)[:valid_len]

    if page_scores.numel() < topk:
        return torch.arange(seq_len, device=device)
    return page_scores.topk(topk, dim=-1).indices.sort().values


def merge_keep_indices(*parts: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    idx = []
    for p in parts:
        if p is not None and p.numel() > 0:
            idx.append(p)
    if not idx:
        return None
    return torch.cat(idx).unique(sorted=True)
