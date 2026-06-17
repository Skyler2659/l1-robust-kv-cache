"""Clustering/diversity-based eviction: farthest point, k-center, k-means medoid."""
from __future__ import annotations
from typing import Optional
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


def _cosine_dist_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance: 1 - cosine_similarity."""
    x_n = torch.nn.functional.normalize(x, dim=-1)
    y_n = torch.nn.functional.normalize(y, dim=-1)
    return 1.0 - x_n @ y_n.T


class FarthestPointEviction(BaseEviction):
    """Farthest-point sampling: greedily pick diverse key vectors."""
    name = "farthest_point"
    method_family = "geometry"
    requires_scores = True
    score_source = "key"
    approximate = True

    def __init__(self, seed=0, **kwargs):
        super().__init__(**kwargs)
        self._seed = seed

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if k_rows is None:
            return None
        seq_len = k_rows.shape[0]
        budget = min(self.cache_size, seq_len)
        reserved = self._reserved_indices(seq_len, budget, k_rows.device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return torch.zeros(seq_len, device=k_rows.device)

        k_norm = torch.nn.functional.normalize(k_rows.float(), dim=-1)
        # Candidates: non-reserved positions
        all_idx = torch.arange(seq_len, device=k_rows.device)
        if reserved.numel() > 0:
            cand_mask = ~torch.isin(all_idx, reserved)
            candidates = all_idx[cand_mask]
        else:
            candidates = all_idx
        if candidates.numel() <= fill:
            scores = torch.zeros(seq_len, device=k_rows.device)
            scores[candidates] = 1.0
            if reserved.numel() > 0:
                scores[reserved] = 1.0
            return scores

        # Greedy farthest-point
        g = torch.Generator().manual_seed(self._seed + layer_idx)
        chosen = []
        first = candidates[torch.randint(0, candidates.numel(), (1,), generator=g)].item()
        chosen.append(first)
        min_dists = torch.full((candidates.numel(),), float("inf"), device=k_rows.device)

        for _ in range(min(fill - 1, candidates.numel() - 1)):
            last_vec = k_norm[chosen[-1]].unsqueeze(0)
            d = 1.0 - (k_norm[candidates] @ last_vec.T).squeeze(-1)
            min_dists = torch.minimum(min_dists, d)
            next_idx = torch.argmax(min_dists).item()
            chosen.append(candidates[next_idx].item())
            if min_dists[next_idx] < 1e-8:
                break

        scores = torch.zeros(seq_len, device=k_rows.device)
        scores[torch.tensor(chosen, dtype=torch.long, device=k_rows.device)] = 1.0
        if reserved.numel() > 0:
            scores[reserved] = 1.0
        return scores

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        if scores is None:
            return self._fill_budget(reserved, seq_len, budget, device)
        s = scores.to(device)
        topk = min(budget, int((s > 0.5).sum().item()))
        if topk == 0:
            return self._fill_budget(reserved, seq_len, budget, device)
        idx = torch.topk(s, topk).indices.sort().values
        return self._ensure_budget(idx, seq_len, budget, device, scores=scores, reserved=reserved)


class KCenterEviction(FarthestPointEviction):
    """K-center greedy — same as farthest point with coverage tracking."""
    name = "k_center"


class KMeansMedoidEviction(BaseEviction):
    """K-means medoid: run k-means, pick closest point to each centroid."""
    name = "kmeans_medoid"
    method_family = "geometry"
    requires_scores = True
    score_source = "key"
    approximate = True

    def __init__(self, n_clusters=64, n_iters=5, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self._seed = seed

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if k_rows is None:
            return None
        n = k_rows.shape[0]
        if n <= self.cache_size:
            return torch.ones(n, device=k_rows.device)
        k_float = torch.nn.functional.normalize(k_rows.float(), dim=-1)
        budget = min(self.cache_size, n)
        k_clus = min(self.n_clusters, budget)
        g = torch.Generator().manual_seed(self._seed + layer_idx)
        perm = torch.randperm(n, generator=g)[:k_clus]
        centroids = k_float[perm].clone()
        for _ in range(self.n_iters):
            sim = k_float @ centroids.T
            assignments = torch.argmax(sim, dim=1)
            for c in range(k_clus):
                members = (assignments == c).nonzero(as_tuple=True)[0]
                if members.numel() > 0:
                    centroids[c] = k_float[members].mean(dim=0)
                    centroids[c] /= centroids[c].norm() + 1e-8
        sim = k_float @ centroids.T
        assignments = torch.argmax(sim, dim=1)
        medoid_scores = torch.zeros(n, device=k_rows.device)
        for c in range(k_clus):
            members = (assignments == c).nonzero(as_tuple=True)[0]
            if members.numel() == 0:
                continue
            member_sim = k_float[members] @ centroids[c]
            best = members[torch.argmax(member_sim)]
            medoid_scores[best] = 1.0
        return medoid_scores

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        if scores is None:
            return self._fill_budget(reserved, seq_len, budget, device)
        s = scores.to(device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return self._ensure_budget(reserved, seq_len, budget, device, reserved=reserved)
        s_copy = s.clone()
        if reserved.numel() > 0:
            s_copy[reserved] = -float("inf")
        valid = torch.isfinite(s_copy) & (s_copy > 0)
        topk = min(fill, int(valid.sum().item()))
        if topk <= 0:
            return self._fill_budget(reserved, seq_len, budget, device)
        idx = torch.topk(s_copy, topk).indices
        return self._ensure_budget(
            torch.cat([reserved, idx]),
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )


class ApproxFacilityLocationEviction(BaseEviction):
    """Approximate facility-location coverage via top average cosine similarity."""

    name = "approximate_facility_location"
    method_family = "geometry"
    requires_scores = True
    score_source = "key"
    approximate = True

    def __init__(self, max_points: int = 512, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.max_points = int(max_points)
        self.seed = int(seed)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = mean_heads(layer_k, self.k_seq_dim)
        if rows is None:
            return None
        rows = torch.nn.functional.normalize(rows.float(), dim=-1)
        n = rows.shape[0]
        if n <= self.max_points:
            refs = rows
        else:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + layer_idx)
            idx = torch.randperm(n, generator=generator)[: self.max_points].to(rows.device)
            refs = rows[idx]
        sim = rows @ refs.T
        return sim.clamp_min(0).mean(dim=1)

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
        idx = torch.topk(masked, min(fill, int(valid.sum().item()))).indices
        return self._ensure_budget(
            torch.cat([reserved, idx]),
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )
