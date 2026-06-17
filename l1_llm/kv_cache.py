import torch

from l1_llm.l1_sketch import (
    L1LeverageScoreEstimator,
    compute_reweight,
)


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


def _is_dynamic_cache(past_key_values):
    return past_key_values is not None and not isinstance(past_key_values, (list, tuple))


def _to_legacy(past_key_values):
    if past_key_values is None:
        return None
    if _is_dynamic_cache(past_key_values):
        # 4.51+: .layers list of DynamicLayer
        lyrs = getattr(past_key_values, "layers", None)
        if isinstance(lyrs, (list, tuple)) and len(lyrs) > 0:
            return tuple((lyr.keys, lyr.values) for lyr in lyrs)
        # 4.45-4.50: .key_cache / .value_cache
        kc = getattr(past_key_values, "key_cache", None)
        vc = getattr(past_key_values, "value_cache", None)
        if isinstance(kc, list) and isinstance(vc, list):
            return tuple((kc[i], vc[i]) for i in range(len(kc)))
        # Legacy: to_legacy_cache method
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
    return past_key_values


def _back_to_original(original, items):
    if original is None:
        return None
    if _is_dynamic_cache(original):
        # 4.51+: rebuild via .layers in-place
        if hasattr(original, "layers"):
            for i, (k, v) in enumerate(items):
                if i < len(original.layers):
                    original.layers[i].keys = k
                    original.layers[i].values = v
                else:
                    original.update(k, v, i)
            return original
        # 4.45-4.50: .key_cache / .value_cache
        if hasattr(original, "key_cache"):
            new_cache = type(original)()
            new_cache.key_cache = [k for k, v in items]
            new_cache.value_cache = [v for k, v in items]
            return new_cache
        # Fallback: from_legacy_cache
        if hasattr(type(original), "from_legacy_cache"):
            return type(original).from_legacy_cache(items)
    if isinstance(original, tuple):
        return tuple(items)
    return items


class PlainKVCache:
    def __call__(self, past_key_values):
        return past_key_values

    def evict_for_space(self, past_key_values, num_coming):
        return past_key_values

    def evict_range(self, past_key_values, start, end):
        return past_key_values


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = int(start_size)
        self.recent_size = int(recent_size)
        self.cache_size = self.start_size + self.recent_size
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.k_slice = DIM_TO_SLICE[self.k_seq_dim]
        self.v_slice = DIM_TO_SLICE[self.v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        result = [
            (
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            )
            for k, v in pkv
        ]
        return _back_to_original(past_key_values, result)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        result = [
            (
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            )
            for k, v in pkv
        ]
        return _back_to_original(past_key_values, result)

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        result = [
            (
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            )
            for k, v in pkv
        ]
        return _back_to_original(past_key_values, result)


class L1RobustKVCache:
    def __init__(
        self,
        cache_size=512,
        num_sink_tokens=4,
        k_seq_dim=2,
        v_seq_dim=2,
        sketch_dim=1024,
        recompute_interval=32,
        seed=0,
        per_layer=True,
        use_reweight=False,
        recent_keep=0,
        score_source="v",
    ):
        self.cache_size = max(1, int(cache_size))
        self.num_sink_tokens = max(0, int(num_sink_tokens))
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        self.sketch_dim = int(sketch_dim)
        self.recompute_interval = int(recompute_interval)
        self.seed = int(seed)
        self.per_layer = bool(per_layer)
        self.use_reweight = bool(use_reweight)
        self.recent_keep = int(max(0, recent_keep))
        self.score_source = self._normalize_score_source(score_source)
        self._estimators = {}
        self._steps = 0
        self._shape_warned = False
        fallback_sink = min(self.num_sink_tokens, max(0, self.cache_size - 1))
        self._fallback_cache = StartRecentKVCache(
            start_size=fallback_sink,
            recent_size=max(1, self.cache_size - fallback_sink),
            k_seq_dim=self.k_seq_dim,
            v_seq_dim=self.v_seq_dim,
        )

    def _warn_shape_fallback(self):
        if not self._shape_warned:
            print(
                "Warning: L1RobustKVCache currently requires legacy [B,H,S,D] layout "
                "(k_seq_dim=v_seq_dim=2). Falling back to StartRecentKVCache behavior."
            )
            self._shape_warned = True

    def _normalize_score_source(self, score_source):
        value = str(score_source).lower().strip()
        aliases = {
            "v": "v",
            "value": "v",
            "v_only": "v",
            "value_only": "v",
            "kv": "kv",
            "k_v": "kv",
            "key_value": "kv",
            "kv_concat": "kv",
            "joint": "kv",
        }
        if value not in aliases:
            raise ValueError(
                f"Unknown score_source={score_source!r}; expected 'v' or 'kv'."
            )
        return aliases[value]

    def _get_layer_v_rows(self, layer_v):
        if self.v_seq_dim != 2 or layer_v.dim() != 4:
            return None
        # [B, H, S, D] -> average over heads for B=1 streaming case.
        return layer_v[0].mean(dim=0)

    def _get_layer_k_rows(self, layer_k):
        if self.k_seq_dim != 2 or layer_k is None or layer_k.dim() != 4:
            return None
        # [B, H, S, D] -> average over heads for B=1 streaming case.
        return layer_k[0].mean(dim=0)

    def _get_score_rows(self, layer_k, layer_v):
        v_rows = self._get_layer_v_rows(layer_v)
        if v_rows is None or self.score_source == "v":
            return v_rows
        k_rows = self._get_layer_k_rows(layer_k)
        if k_rows is None or k_rows.shape[0] != v_rows.shape[0]:
            return v_rows
        return torch.cat([k_rows.float(), v_rows.float()], dim=-1)

    def _get_or_create_estimator(self, layer_idx, row_dim):
        est = self._estimators.get(layer_idx)
        target_sketch_dim = max(self.sketch_dim, min(row_dim * row_dim, 4096))
        if est is None:
            est = L1LeverageScoreEstimator(
                sketch_dim=target_sketch_dim,
                seed=self.seed + int(layer_idx),
            )
            self._estimators[layer_idx] = est
        return est

    def _select_indices_for_layer(self, layer_idx, layer_v, layer_k=None):
        score_rows = self._get_score_rows(layer_k, layer_v)
        if score_rows is None:
            return None, None
        seq_len, row_dim = score_rows.shape
        if seq_len <= self.cache_size:
            return None, None
        estimator = self._get_or_create_estimator(layer_idx, row_dim)
        force_refit = (self._steps % self.recompute_interval) == 0
        scores = estimator.scores(score_rows, force_refit=force_refit)

        keep = self._select_with_recency_mix(scores)
        rw = compute_reweight(scores, keep) if self.use_reweight else None
        return keep, rw

    def _select_with_recency_mix(self, scores):
        seq_len = int(scores.numel())
        budget = max(1, int(self.cache_size))
        if seq_len <= budget:
            return torch.arange(seq_len, device=scores.device, dtype=torch.long)

        # Reserve one slot for the latest token.
        sink = min(self.num_sink_tokens, max(0, budget - 1))
        max_recent = max(0, budget - sink - 1)
        recent = min(self.recent_keep, max_recent)
        l1_budget = max(0, budget - sink - recent - 1)

        parts = []
        if sink > 0:
            parts.append(torch.arange(sink, device=scores.device, dtype=torch.long))

        # Keep a recency block right before the latest token.
        if recent > 0:
            rec_start = seq_len - 1 - recent
            rec_idx = torch.arange(rec_start, seq_len - 1, device=scores.device)
            parts.append(rec_idx)

        # Use l1 scores to pick informative older tokens from remaining history.
        if l1_budget > 0:
            cand_start = sink
            cand_end = seq_len - 1 - recent
            if cand_end > cand_start:
                cand_scores = scores[cand_start:cand_end]
                topk = min(l1_budget, cand_scores.numel())
                l1_idx = torch.topk(cand_scores, k=topk).indices + cand_start
                parts.append(l1_idx)

        parts.append(torch.tensor([seq_len - 1], device=scores.device, dtype=torch.long))
        keep = torch.cat(parts).unique(sorted=True)
        if keep.numel() > budget:
            keep = keep[:budget - 1]
            keep = torch.cat(
                [keep, torch.tensor([seq_len - 1], device=scores.device, dtype=torch.long)]
            ).unique(sorted=True)
        return keep

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        if self.k_seq_dim != 2 or self.v_seq_dim != 2:
            self._warn_shape_fallback()
            out = self._fallback_cache(pkv)
            self._steps += 1
            return _back_to_original(past_key_values, out)

        shared_keep = None
        shared_rw = None
        self._steps += 1
        items = []
        for layer_idx, (k, v) in enumerate(pkv):
            if self.per_layer:
                keep, rw = self._select_indices_for_layer(layer_idx, v, layer_k=k)
            else:
                if shared_keep is None:
                    shared_keep, shared_rw = self._select_indices_for_layer(0, v, layer_k=k)
                keep, rw = shared_keep, shared_rw
            if keep is None:
                items.append((k, v))
                continue
            keep_k = keep.to(k.device)
            keep_v = keep.to(v.device)
            new_k = torch.index_select(k, self.k_seq_dim, keep_k)
            new_v = torch.index_select(v, self.v_seq_dim, keep_v)
            if rw is not None:
                scale = rw.to(new_v.device).view(1, 1, -1, 1)
                new_v = new_v * scale
            items.append((new_k, new_v))
        return _back_to_original(past_key_values, items)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        if self.k_seq_dim != 2 or self.v_seq_dim != 2:
            self._warn_shape_fallback()
            return self._fallback_cache.evict_for_space(past_key_values, num_coming)
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        budget = max(1, self.cache_size - max(0, int(num_coming)))
        if seq_len <= budget:
            return past_key_values
        old = self.cache_size
        try:
            self.cache_size = budget
            out = self.__call__(past_key_values)
        finally:
            self.cache_size = old
        return out

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        keep = torch.cat(
            [
                torch.arange(0, start, device=pkv[0][0].device),
                torch.arange(end, seq_len, device=pkv[0][0].device),
            ]
        )
        items = []
        for k, v in pkv:
            new_k = torch.index_select(k, self.k_seq_dim, keep.to(k.device))
            new_v = torch.index_select(v, self.v_seq_dim, keep.to(v.device))
            items.append((new_k, new_v))
        return _back_to_original(past_key_values, items)
