"""CUDA-aware timers for profiling."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class CudaTimer:
    """Timer that synchronizes CUDA before measuring."""

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._start: float = 0.0
        self._elapsed: float = 0.0

    def _sync(self) -> None:
        if self._device and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)

    def start(self) -> None:
        self._sync()
        self._start = time.perf_counter()

    def stop(self) -> float:
        self._sync()
        self._elapsed = time.perf_counter() - self._start
        return self._elapsed

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @contextmanager
    def measure(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()


class TimerStack:
    """Named timer collection for multi-phase profiling."""

    def __init__(self, device: Optional[str] = None):
        self._timers: Dict[str, CudaTimer] = {}
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._device = device

    def get(self, name: str) -> CudaTimer:
        if name not in self._timers:
            self._timers[name] = CudaTimer(self._device)
        return self._timers[name]

    @contextmanager
    def measure(self, name: str):
        t = self.get(name)
        t.start()
        try:
            yield t
        finally:
            elapsed = t.stop()
            self._totals[name] = self._totals.get(name, 0.0) + elapsed
            self._counts[name] = self._counts.get(name, 0) + 1

    def record(self, name: str, elapsed: float) -> None:
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for name in self._totals:
            total = self._totals[name]
            count = self._counts[name]
            result[name] = {
                "total_s": total,
                "count": count,
                "avg_ms": (total / count) * 1000.0 if count > 0 else 0.0,
            }
        return result

    def reset(self) -> None:
        self._timers.clear()
        self._totals.clear()
        self._counts.clear()
