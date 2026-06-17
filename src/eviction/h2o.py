"""H2O Heavy-Hitter Oracle eviction (Zhang et al., NeurIPS 2023)."""
from __future__ import annotations
import torch
from src.eviction.attention import AttentionEviction


class H2OEviction(AttentionEviction):
    """H2O: accumulated attention heavy-hitters + mandatory sink + recent."""
    name = "h2o"
    method_family = "attention"
    requires_attention = True
    requires_scores = True
    score_source = "accumulated_attention"

    def __init__(self, h2o_recent_size=None, **kwargs):
        super().__init__(**kwargs)
        if h2o_recent_size is not None:
            self.recent_size = max(self.recent_size, int(h2o_recent_size))
