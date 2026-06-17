"""Woodruff-style approximate L1 leverage score via Exp(1)+CountSketch+QR."""
from __future__ import annotations
import torch
from typing import Optional


def _make_cpu_generator(seed):
    if seed is None:
        return None
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _safe_exp_samples(size, device, dtype, generator=None):
    """Generate Exp(1) samples in float32 for numerical safety."""
    sample_device = "cpu" if generator is not None else device
    u = torch.rand(size, device=sample_device, dtype=torch.float32, generator=generator)
    u.clamp_(1e-8, 1 - 1e-8)
    return (-torch.log(1.0 - u)).to(device=device, dtype=dtype)


class CountSketch:
    """Hash-based dimensionality reduction for L1 sketching."""

    def __init__(self, sketch_dim: int, seed=None):
        self.sketch_dim = int(sketch_dim)
        self.seed = seed
        self.hash_buckets = None
        self.signs = None
        self._gen = _make_cpu_generator(seed)

    def _ensure_capacity(self, n, device):
        if self.hash_buckets is not None and self.hash_buckets.device != device:
            self.hash_buckets = self.hash_buckets.to(device)
            self.signs = self.signs.to(device)
        if self.hash_buckets is not None and self.hash_buckets.numel() >= n:
            return
        old_n = 0 if self.hash_buckets is None else self.hash_buckets.numel()
        grow = int(n - old_n)
        new_b = torch.randint(0, self.sketch_dim, (grow,), generator=self._gen).to(device)
        new_s = (torch.randint(0, 2, (grow,), generator=self._gen, dtype=torch.int64) * 2 - 1)
        new_s = new_s.to(device=device, dtype=torch.float32)
        if self.hash_buckets is None:
            self.hash_buckets, self.signs = new_b, new_s
        else:
            self.hash_buckets = torch.cat([self.hash_buckets, new_b])
            self.signs = torch.cat([self.signs, new_s])

    def apply(self, rows: torch.Tensor) -> torch.Tensor:
        n, d = rows.shape
        self._ensure_capacity(n, rows.device)
        b = self.hash_buckets[:n]
        s = self.signs[:n].to(rows.dtype)
        sketch = torch.zeros(self.sketch_dim, d, device=rows.device, dtype=rows.dtype)
        sketch.scatter_add_(0, b.unsqueeze(-1).expand(-1, d), rows * s.unsqueeze(-1))
        return sketch


class L1SubspaceEmbedding:
    """Exp(1) reweighting + CountSketch."""

    def __init__(self, sketch_dim, seed=None, exp_generator=None):
        self.sketch_dim = int(sketch_dim)
        self.count_sketch = CountSketch(self.sketch_dim, seed=seed)
        self._exp_gen = exp_generator or _make_cpu_generator(seed)

    def embed(self, v_rows: torch.Tensor) -> torch.Tensor:
        n = v_rows.shape[0]
        w = _safe_exp_samples((n, 1), v_rows.device, v_rows.dtype, self._exp_gen)
        weighted = v_rows / w
        return self.count_sketch.apply(weighted)


class WoodruffL1Estimator:
    """Approximate L1 leverage score estimator.

    Pipeline: Exp(1) reweight → CountSketch → QR → row L1 norms of A·R⁻¹.
    """

    def __init__(self, sketch_dim=1024, seed=None):
        self.sketch_dim = int(sketch_dim)
        self.seed = seed
        self._exp_gen = _make_cpu_generator(seed)
        self.embedding = L1SubspaceEmbedding(self.sketch_dim, seed=seed, exp_generator=self._exp_gen)
        self.r_inv: Optional[torch.Tensor] = None
        self.last_dim: Optional[int] = None
        self._last_scores: Optional[torch.Tensor] = None

    def fit(self, rows: torch.Tensor) -> None:
        """Build the sketch basis (QR of weighted+sketched matrix)."""
        n, d = rows.shape
        if n <= 1:
            self.r_inv = None
            return
        if n < self.embedding.count_sketch.sketch_dim:
            w = _safe_exp_samples((n, 1), rows.device, torch.float32, self._exp_gen)
            weighted = rows.float() / w
        else:
            weighted = self.embedding.embed(rows).float()
        if torch.isnan(weighted).any():
            self.r_inv = None
            return
        _, r = torch.linalg.qr(weighted, mode="reduced")
        if torch.isnan(r).any():
            self.r_inv = None
            return
        if r.shape[0] != r.shape[1]:
            self.r_inv = None
            self.last_dim = d
            return
        jit = max(1e-4, r.diag().abs().max().item() * 1e-6)
        r = r + torch.eye(r.shape[0], device=r.device, dtype=r.dtype) * jit
        try:
            self.r_inv = torch.linalg.inv(r)
        except Exception:
            r = r + torch.eye(r.shape[0], device=r.device, dtype=r.dtype) * 1e-2
            self.r_inv = torch.linalg.inv(r)
        self.last_dim = d

    def scores(self, rows: torch.Tensor, force_refit: bool = False) -> torch.Tensor:
        """Compute L1 leverage scores for each row."""
        _, d = rows.shape
        if self.r_inv is None or force_refit or self.last_dim != d:
            self.fit(rows)
        if self.r_inv is None:
            fallback = torch.norm(rows, p=1, dim=1)
            self._last_scores = fallback
            return fallback
        proj = rows.float() @ self.r_inv
        s = torch.norm(proj, p=1, dim=1).to(rows.dtype)
        self._last_scores = s
        return s
