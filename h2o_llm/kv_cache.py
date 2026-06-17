"""H2O: Heavy-Hitter Oracle — cumulative attention-score based KV cache eviction.

Zhang et al., NeurIPS 2023.
"""
import torch


def _is_dynamic_cache(pkv):
    return pkv is not None and not isinstance(pkv, (list, tuple))


def _to_legacy(pkv):
    if pkv is None:
        return None
    if _is_dynamic_cache(pkv):
        lyrs = getattr(pkv, "layers", None)
        if isinstance(lyrs, (list, tuple)) and len(lyrs) > 0:
            return tuple((lyr.keys, lyr.values) for lyr in lyrs)
        kc = getattr(pkv, "key_cache", None)
        vc = getattr(pkv, "value_cache", None)
        if isinstance(kc, list) and isinstance(vc, list):
            return tuple((kc[i], vc[i]) for i in range(len(kc)))
        if hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()
    return pkv


def _back_to_original(original, items):
    if original is None:
        return None
    if _is_dynamic_cache(original):
        if hasattr(original, "layers"):
            for i, (k, v) in enumerate(items):
                if i < len(original.layers):
                    original.layers[i].keys = k
                    original.layers[i].values = v
                else:
                    original.update(k, v, i)
            return original
        if hasattr(original, "key_cache"):
            new_cache = type(original)()
            new_cache.key_cache = [k for k, v in items]
            new_cache.value_cache = [v for k, v in items]
            return new_cache
        if hasattr(type(original), "from_legacy_cache"):
            return type(original).from_legacy_cache(items)
    if isinstance(original, tuple):
        return tuple(items)
    return items


class H2OKVCache:
    """Heavy-Hitter Oracle: accumulated attention score per token per layer."""

    def __init__(
        self,
        cache_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        recent_size=1,
        sink_size=0,
    ):
        self.cache_size = int(cache_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.recent_size = int(recent_size)
        self.sink_size = int(sink_size)
        # Per-layer accumulated attention scores: layer_idx -> torch.Tensor [numel]
        self._acc_scores = {}
        self._steps = 0

    def _mean_key_rows(self, layer_k):
        if layer_k.dim() == 4 and self.k_seq_dim == 2:
            return layer_k[0].mean(dim=0)
        if layer_k.dim() == 4 and self.k_seq_dim == 3:
            return layer_k[0].mean(dim=0).transpose(0, 1)
        if layer_k.dim() == 3 and self.k_seq_dim == 1:
            return layer_k.mean(dim=0)
        return None

    def _key_heads(self, layer_k, layer_idx, seq_len):
        import shared_q

        key_heads = shared_q.LAST_KEY_STATES.get(layer_idx)
        if (
            key_heads is not None
            and key_heads.dim() == 3
            and key_heads.shape[1] == seq_len
        ):
            return key_heads
        if layer_k.dim() == 4 and self.k_seq_dim == 2:
            return layer_k[0]
        return None

    def _compute_attn_weights(self, layer_k, layer_v, layer_idx):
        """Compute current-step attention weights from Q_last (if available)."""
        import shared_q
        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is None:
            return None
        seq_len = layer_k.size(self.k_seq_dim)
        head_dim = layer_v.shape[-1]

        key_heads = self._key_heads(layer_k, layer_idx, seq_len)
        if q_h.dim() == 2 and key_heads is not None and key_heads.dim() == 3:
            q_heads = q_h.to(layer_v.device, dtype=torch.float32)
            k_heads = key_heads.to(layer_v.device, dtype=torch.float32)
            if k_heads.shape[0] != q_heads.shape[0]:
                if q_heads.shape[0] % k_heads.shape[0] == 0:
                    repeat = q_heads.shape[0] // k_heads.shape[0]
                    k_heads = k_heads.repeat_interleave(repeat, dim=0)
                else:
                    k_heads = None
            if (
                k_heads is not None
                and k_heads.shape[0] == q_heads.shape[0]
                and k_heads.shape[-1] == q_heads.shape[-1]
            ):
                logits = torch.einsum("hd,hsd->hs", q_heads, k_heads)
                logits = logits / max(head_dim ** 0.5, 1e-6)
                return torch.softmax(logits, dim=-1).mean(dim=0)

        q_vec = q_h.mean(dim=0).to(layer_v.device)       # [D]
        k_rows = shared_q.LAST_KEY_ROWS.get(layer_idx)
        if k_rows is None or k_rows.dim() != 2 or k_rows.shape[0] != seq_len:
            k_rows = self._mean_key_rows(layer_k)
        if k_rows is None or k_rows.dim() != 2 or k_rows.shape[-1] != q_vec.numel():
            return None
        k_rows = k_rows.to(layer_v.device, dtype=torch.float32)  # [S, D]
        q_vec = q_vec.to(dtype=torch.float32)
        logits = torch.matmul(q_vec, k_rows.T) / max(head_dim ** 0.5, 1e-6)
        return torch.softmax(logits, dim=0)                # [S]

    def _accumulate_scores(self, layer_idx, k, v, seq_len):
        attn = self._compute_attn_weights(k, v, layer_idx)
        if attn is None:
            return
        prev = self._acc_scores.get(layer_idx)
        if prev is None or prev.numel() < seq_len:
            new_prev = torch.zeros(seq_len, device=k.device, dtype=attn.dtype)
            if prev is not None:
                new_prev[:prev.numel()] = prev
            prev = new_prev
        prev[:seq_len] += attn.to(prev.device)
        self._acc_scores[layer_idx] = prev

    def _select_keep(self, scores, seq_len, budget, device):
        budget = min(max(1, int(budget)), int(seq_len))
        sink_size = min(max(0, self.sink_size), budget)
        recent_size = min(max(0, self.recent_size), max(0, budget - sink_size))
        parts = []
        if sink_size > 0:
            parts.append(torch.arange(0, sink_size, device=device))
        if recent_size > 0:
            parts.append(torch.arange(seq_len - recent_size, seq_len, device=device))

        reserved = (
            torch.cat(parts).unique(sorted=True)
            if parts
            else torch.empty(0, dtype=torch.long, device=device)
        )
        heavy_budget = budget - int(reserved.numel())
        if heavy_budget <= 0:
            return reserved[:budget]
        if scores is None or scores.numel() < seq_len:
            fill = budget - int(reserved.numel())
            if fill <= 0:
                return reserved[:budget]
            fill_idx = torch.arange(max(0, seq_len - budget), seq_len, device=device)
            if reserved.numel() > 0:
                fill_idx = fill_idx[~torch.isin(fill_idx, reserved)]
            if fill_idx.numel() > fill:
                fill_idx = fill_idx[-fill:]
            return torch.cat([reserved, fill_idx]).unique(sorted=True)

        masked = scores[:seq_len].to(device).clone()
        if reserved.numel() > 0:
            masked[reserved] = -float("inf")
        valid = torch.isfinite(masked)
        if not valid.any():
            return reserved[:budget]
        topk = min(heavy_budget, int(valid.sum().item()))
        heavy_keep = torch.topk(masked, topk).indices
        return torch.cat([reserved, heavy_keep]).unique(sorted=True)

    def _evict(self, past_key_values, budget, update_scores):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        self._steps += 1
        items = []
        for layer_idx, (k, v) in enumerate(pkv):
            seq_len = k.size(self.k_seq_dim)
            if update_scores:
                self._accumulate_scores(layer_idx, k, v, seq_len)

            if seq_len <= budget:
                items.append((k, v))
                continue
            scores = self._acc_scores.get(layer_idx)
            keep = self._select_keep(scores, seq_len, budget, k.device)
            keep_k = keep.to(k.device)
            keep_v = keep.to(v.device)
            new_k = torch.index_select(k, self.k_seq_dim, keep_k)
            new_v = torch.index_select(v, self.v_seq_dim, keep_v)
            if scores is not None and scores.numel() >= seq_len:
                self._acc_scores[layer_idx] = scores[:seq_len][keep].clone()
            items.append((new_k, new_v))
        return _back_to_original(past_key_values, items)

    def __call__(self, past_key_values):
        return self._evict(past_key_values, self.cache_size, update_scores=True)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        budget = max(1, self.cache_size - int(num_coming))
        if seq_len <= budget:
            return past_key_values
        return self._evict(past_key_values, budget, update_scores=False)

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        keep = torch.cat([
            torch.arange(0, start, device=pkv[0][0].device),
            torch.arange(end, seq_len, device=pkv[0][0].device),
        ])
        items = []
        for k, v in pkv:
            new_k = torch.index_select(k, self.k_seq_dim, keep.to(k.device))
            new_v = torch.index_select(v, self.v_seq_dim, keep.to(v.device))
            items.append((new_k, new_v))
        return _back_to_original(past_key_values, items)
