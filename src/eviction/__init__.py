from src.eviction.base import BaseEviction
from src.eviction.kv_utils import (
    to_legacy_cache,
    back_to_original,
    get_kv_seq_len,
    slice_by_dim,
    gather_by_dim,
    infer_kv_dims,
)
