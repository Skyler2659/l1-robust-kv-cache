"""Throughput measurement: tokens/s, latency/token, amortized overhead."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class ThroughputStats:
    """Accumulated throughput statistics for one method/sample."""

    total_steps: int = 0
    total_decode_time: float = 0.0
    total_score_time: float = 0.0
    total_eviction_time: float = 0.0
    total_topk_time: float = 0.0
    total_prefill_time: float = 0.0
    peak_memory_mb: float = 0.0
    avg_kv_len: float = 0.0
    max_kv_len: int = 0
    step_times: List[float] = field(default_factory=list)
    kv_lens: List[int] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.total_decode_time <= 0:
            return float("inf")
        return self.total_steps / self.total_decode_time

    @property
    def avg_ms_per_token(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return (self.total_decode_time / self.total_steps) * 1000.0

    @property
    def score_overhead_ratio(self) -> float:
        if self.total_decode_time <= 0:
            return 0.0
        return self.total_score_time / self.total_decode_time

    @property
    def eviction_overhead_ratio(self) -> float:
        if self.total_decode_time <= 0:
            return 0.0
        return self.total_eviction_time / self.total_decode_time

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_steps": self.total_steps,
            "tokens_per_second": self.tokens_per_second,
            "avg_ms_per_token": self.avg_ms_per_token,
            "score_overhead_ratio": self.score_overhead_ratio,
            "eviction_overhead_ratio": self.eviction_overhead_ratio,
            "total_decode_time_s": self.total_decode_time,
            "total_score_time_s": self.total_score_time,
            "total_eviction_time_s": self.total_eviction_time,
            "total_topk_time_s": self.total_topk_time,
            "total_prefill_time_s": self.total_prefill_time,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_kv_len": self.avg_kv_len,
            "max_kv_len": self.max_kv_len,
        }


class ThroughputTracker:
    """Track per-step timing and throughput during decode loop."""

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._stats = ThroughputStats()
        self._step_start: float = 0.0

    def reset(self) -> None:
        self._stats = ThroughputStats()

    def begin_step(self) -> None:
        self._sync()
        self._step_start = time.perf_counter()

    def end_step(self, kv_len: int = 0) -> float:
        self._sync()
        elapsed = time.perf_counter() - self._step_start
        self._stats.total_steps += 1
        self._stats.total_decode_time += elapsed
        self._stats.step_times.append(elapsed)
        if kv_len > 0:
            self._stats.kv_lens.append(kv_len)
            self._stats.max_kv_len = max(self._stats.max_kv_len, kv_len)
        return elapsed

    def record_phase(self, phase: str, elapsed: float) -> None:
        """Record time for a named phase: 'score', 'eviction', 'topk', 'prefill'."""
        if phase == "score":
            self._stats.total_score_time += elapsed
        elif phase == "eviction":
            self._stats.total_eviction_time += elapsed
        elif phase == "topk":
            self._stats.total_topk_time += elapsed
        elif phase == "prefill":
            self._stats.total_prefill_time += elapsed

    def record_memory(self, peak_mb: float) -> None:
        self._stats.peak_memory_mb = max(self._stats.peak_memory_mb, peak_mb)

    def get_stats(self) -> ThroughputStats:
        if self._stats.kv_lens:
            self._stats.avg_kv_len = sum(self._stats.kv_lens) / len(self._stats.kv_lens)
        return self._stats

    def _sync(self) -> None:
        if self._device and "cuda" in str(self._device) and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)
