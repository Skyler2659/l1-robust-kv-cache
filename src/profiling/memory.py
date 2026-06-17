"""GPU memory tracking utilities."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class MemoryTracker:
    """Track peak and current GPU memory usage."""

    def __init__(self, device: Optional[str] = None):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._peak: float = 0.0
        self._snapshots: list = []

    @property
    def is_cuda(self) -> bool:
        return "cuda" in self._device and torch.cuda.is_available()

    def reset_peak(self) -> None:
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
            self._peak = 0.0

    def record_peak(self) -> float:
        if self.is_cuda:
            peak_bytes = torch.cuda.max_memory_allocated(self._device)
            peak_mb = peak_bytes / (1024 ** 2)
            self._peak = max(self._peak, peak_mb)
            return peak_mb
        return 0.0

    def current_allocated(self) -> float:
        if self.is_cuda:
            return torch.cuda.memory_allocated(self._device) / (1024 ** 2)
        return 0.0

    def current_reserved(self) -> float:
        if self.is_cuda:
            return torch.cuda.memory_reserved(self._device) / (1024 ** 2)
        return 0.0

    def snapshot(self, label: str = "") -> Dict[str, Any]:
        snap = {
            "label": label,
            "peak_mb": self.record_peak(),
            "allocated_mb": self.current_allocated(),
            "reserved_mb": self.current_reserved(),
        }
        self._snapshots.append(snap)
        return snap

    def estimate_kv_cache_memory(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        dtype_bytes: int = 2,
    ) -> float:
        """Estimate KV cache memory in MB."""
        # 2 for K + V, dtype_bytes for precision
        total_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
        return total_bytes / (1024 ** 2)

    def summary(self) -> Dict[str, Any]:
        return {
            "peak_mb": self._peak,
            "snapshots": self._snapshots,
            "device": self._device,
            "cuda_available": self.is_cuda,
        }
