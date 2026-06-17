"""Eviction method registry, metadata, and safe constructor filtering."""
from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from src.eviction.base import BaseEviction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvictionMethodSpec:
    """Static metadata for an eviction method."""

    name: str
    module_path: Optional[str]
    class_name: Optional[str]
    family: str
    supports_backends: Tuple[str, ...] = ("torch",)
    requires_attention: bool = False
    requires_scores: bool = False
    supports_layerwise: bool = True
    supports_headwise: bool = False
    score_source: Optional[str] = None
    score_normalization: str = "none"
    approximate: bool = False
    experimental: bool = False
    oracle: bool = False
    aliases: Tuple[str, ...] = ()
    unsupported_reason: Optional[str] = None
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def supports_mlx(self) -> bool:
        return "mlx" in self.supports_backends

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["supports_mlx"] = self.supports_mlx
        return data


def _spec(
    name: str,
    module_path: Optional[str],
    class_name: Optional[str],
    family: str,
    *,
    supports: Iterable[str] = ("torch",),
    aliases: Iterable[str] = (),
    default_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> EvictionMethodSpec:
    return EvictionMethodSpec(
        name=name,
        module_path=module_path,
        class_name=class_name,
        family=family,
        supports_backends=tuple(supports),
        aliases=tuple(aliases),
        default_kwargs=dict(default_kwargs or {}),
        **kwargs,
    )


_SPECS: Dict[str, EvictionMethodSpec] = {}
_ALIASES: Dict[str, Tuple[str, Dict[str, Any]]] = {}


def _register(spec: EvictionMethodSpec) -> None:
    _SPECS[spec.name] = spec
    for alias in spec.aliases:
        _ALIASES[_clean(alias)] = (spec.name, dict(spec.default_kwargs))


def _clean(method: str) -> str:
    return str(method).strip().lower().replace("-", "_")


def _init_registry() -> None:
    # Recency/locality
    _register(_spec("full", "src.eviction.recency", "FullKVCache", "recency",
                    supports=("torch", "mlx"), aliases=("full_cache",)))
    _register(_spec("recency", "src.eviction.recency", "RecencyEviction", "recency",
                    supports=("torch", "mlx"), aliases=("sliding_window",)))
    _register(_spec("sink_recent", "src.eviction.sink_recent", "SinkRecentEviction", "recency",
                    supports=("torch", "mlx"), aliases=("sink_recency", "streamingllm", "streaming_llm")))
    _register(_spec("random", "src.eviction.random_eviction", "RandomEviction", "random",
                    supports=("torch", "mlx"), aliases=("random_only",)))
    _register(_spec("sink_recent_random", "src.eviction.random_eviction", "RandomEviction", "random",
                    aliases=("sink_recency_random",), supports=("torch", "mlx")))
    _register(_spec("uniform", "src.eviction.uniform", "UniformEviction", "geometry"))

    # Attention
    _register(_spec("attention", "src.eviction.attention", "AttentionEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="accumulated_attention", aliases=("accumulated_attention",)))
    _register(_spec("last_token_attention", "src.eviction.attention", "LastTokenAttentionEviction", "attention",
                    requires_attention=True, requires_scores=True, score_source="last_token_attention"))
    _register(_spec("windowed_attention", "src.eviction.attention", "WindowedAttentionEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="windowed_attention"))
    _register(_spec("attention_decay", "src.eviction.attention", "AttentionDecayEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="decayed_attention"))
    _register(_spec("h2o", "src.eviction.h2o", "H2OEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="accumulated_attention", aliases=("h2o_style",)))
    _register(_spec("snapkv", "src.eviction.snapkv", "SnapKVEviction", "attention",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True,
                    score_source="observation_window_attention",
                    aliases=("snap", "snapkv_style", "approximate_snapkv")))
    _register(_spec("pyramidkv", "src.eviction.pyramidkv", "PyramidKVEviction", "attention",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_layerwise=True, supports_headwise=True, score_source="layer_budget_observation_attention",
                    aliases=("pyramidkv_style", "layer_budget_attention")))

    # Geometry / norms
    _register(_spec("key_l2_norm", "src.eviction.norm_based", "KeyL2NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="key",
                    aliases=("key_norm",)))
    _register(_spec("value_l2_norm", "src.eviction.norm_based", "ValueL2NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    aliases=("value_norm", "value_l2")))
    _register(_spec("key_l1_norm", "src.eviction.norm_based", "KeyL1NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="key"))
    _register(_spec("value_l1_norm", "src.eviction.norm_based", "ValueL1NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value"))
    _register(_spec("kv_norm", "src.eviction.norm_based", "KVNormEviction", "geometry",
                    requires_scores=True, score_source="key_value_concat", aliases=("norm",)))
    _register(_spec("hidden_l2_norm", None, None, "geometry",
                    requires_scores=True, score_source="hidden",
                    unsupported_reason="Hidden states are not available through the current eviction cache interface."))

    # Leverage / subspace
    _register(_spec("l1_leverage", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    approximate=True, aliases=("l1", "l1_mixed", "l1_only")))
    _register(_spec("l2_leverage", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    aliases=("l2",)))
    _register(_spec("l1_prefill_only", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    approximate=True,
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("l2_prefill_only", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("l2_key_prefill_only", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="key",
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0, "score_source": "k"}))
    _register(_spec("compactor", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True,
                    score_source="prefill_key_leverage+non_causal_attention",
                    unsupported_reason=None,
                    aliases=("compactor_style", "compactor_l2_attention"),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0, "score_source": "k"}))
    _register(_spec("l1_decode_only", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="value",
                    approximate=True,
                    default_kwargs={"update_policy": "decode_only"}))
    _register(_spec("l2_decode_only", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="value",
                    default_kwargs={"update_policy": "decode_only"}))
    _register(_spec("ridge_leverage", "src.eviction.l2_leverage", "RidgeLeverageEviction", "geometry",
                    requires_scores=True, score_source="value"))
    _register(_spec("approximate_l2_leverage", "src.eviction.l2_leverage", "ApproximateL2LeverageEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))
    _register(_spec("approximate_l1_leverage", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True,
                    default_kwargs={"use_reweight": True}))

    # Diversity / coverage
    _register(_spec("farthest_point_sampling", "src.eviction.clustering", "FarthestPointEviction", "geometry",
                    requires_scores=True, score_source="key", approximate=True,
                    aliases=("farthest_point", "cosine_diversity", "k_center", "clustering")))
    _register(_spec("kmeans_centroid", "src.eviction.clustering", "KMeansMedoidEviction", "geometry",
                    requires_scores=True, score_source="key", approximate=True,
                    aliases=("kmeans_medoid",)))
    _register(_spec("facility_location_greedy", "src.eviction.clustering", "ApproxFacilityLocationEviction", "geometry",
                    requires_scores=True, score_source="key", approximate=True,
                    aliases=("approximate_facility_location",)))
    _register(_spec("pca_residual", "src.eviction.pca_residual", "PCAResidualEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))

    # Outlier / rarity
    _register(_spec("mahalanobis_distance", "src.eviction.outlier", "MahalanobisDistanceEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))
    _register(_spec("zscore_outlier", "src.eviction.outlier", "ZScoreOutlierEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))
    _register(_spec("random_projection_outlier", "src.eviction.outlier", "RandomProjectionOutlierEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))

    # Hybrid
    _register(_spec("attention_l1", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+l1_leverage",
                    approximate=True,
                    aliases=("attention+l1", "attn_l1"),
                    default_kwargs={"geometry_method": "l1"}))
    _register(_spec("attention_l2", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+l2_leverage",
                    aliases=("attention+l2", "attn_l2"),
                    default_kwargs={"geometry_method": "l2"}))
    _register(_spec("attention_l1_compactor", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    score_source="rank_attention+rank_l1_leverage_score_fusion",
                    approximate=True,
                    aliases=("attention+l1_compactor", "attn_l1_compactor", "compactorlike_l1_attention"),
                    default_kwargs={"geometry_method": "l1", "hybrid_mode": "interpolation"}))
    _register(_spec("attention_l2_compactor", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    score_source="rank_attention+rank_l2_leverage_score_fusion",
                    aliases=("attention+l2_compactor", "attn_l2_compactor", "compactorlike_l2_attention"),
                    default_kwargs={"geometry_method": "l2", "hybrid_mode": "interpolation"}))
    _register(_spec("attention_recency", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+recency",
                    aliases=("attn_recency",),
                    default_kwargs={"geometry_method": "recency", "hybrid_mode": "interpolation"}))
    _register(_spec("attention_sink_recency", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+sink+recency",
                    default_kwargs={"geometry_method": "recency", "hybrid_mode": "budget_split"}))
    _register(_spec("attention_norm", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+norm",
                    default_kwargs={"geometry_method": "value_norm"}))
    _register(_spec("weighted_score_hybrid", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    requires_attention=True, requires_scores=True,
                    score_source="weighted_components",
                    default_kwargs={"hybrid_mode": "interpolation"}))
    _register(_spec("budget_split_hybrid", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="budget_split_components",
                    aliases=("sink_recent_attention_l1",),
                    default_kwargs={"hybrid_mode": "budget_split", "geometry_method": "l1"}))
    _register(_spec("recency_l1", "src.eviction.l1_leverage", "L1LeverageEviction", "hybrid",
                    requires_scores=True, score_source="recency+l1_leverage"))
    _register(_spec("sink_recent_l1", "src.eviction.l1_leverage", "L1LeverageEviction", "hybrid",
                    supports=("torch", "mlx"), requires_scores=True, score_source="sink+recent+l1_leverage"))
    _register(_spec("sink_recent_l2", "src.eviction.l2_leverage", "L2LeverageEviction", "hybrid",
                    supports=("torch", "mlx"), requires_scores=True, score_source="sink+recent+l2_leverage"))

    # Oracle
    _register(_spec("oracle_evidence", "src.eviction.oracle", "OracleEvidenceEviction", "oracle",
                    supports=("torch", "mlx"), oracle=True, score_source="evidence_positions"))
    _register(_spec("oracle_answer_region", "src.eviction.oracle", "OracleAnswerRegionEviction", "oracle",
                    supports=("torch", "mlx"), oracle=True, score_source="answer_region"))


_init_registry()


def canonicalize_method(method: str, kwargs: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any]]:
    """Resolve aliases and return canonical method plus alias defaults."""
    key = _clean(method)
    kwargs = dict(kwargs or {})
    if key == "clustering":
        key = _clean(kwargs.get("clustering_method", "farthest_point_sampling"))
    if key in _SPECS:
        spec = _SPECS[key]
        return spec.name, dict(spec.default_kwargs)
    canonical, defaults = _ALIASES.get(key, (key, {}))
    if canonical in _SPECS:
        merged = dict(_SPECS[canonical].default_kwargs)
        merged.update(defaults)
        return canonical, merged
    return canonical, dict(defaults)


def list_methods(include_aliases: bool = True) -> List[str]:
    values = set(_SPECS)
    if include_aliases:
        values.update(_ALIASES)
    return sorted(values)


def get_method_spec(method: str) -> EvictionMethodSpec:
    canonical, _ = canonicalize_method(method)
    if canonical not in _SPECS:
        raise ValueError(f"Unknown eviction method: {method!r}. Available: {list_methods()}")
    return _SPECS[canonical]


def method_metadata(method: str) -> Dict[str, Any]:
    return get_method_spec(method).to_dict()


def method_requires_attention(method: str) -> bool:
    return bool(get_method_spec(method).requires_attention)


def method_supports_backend(method: str, backend: str) -> bool:
    spec = get_method_spec(method)
    return str(backend).lower() in spec.supports_backends and spec.unsupported_reason is None


def unsupported_reason(method: str, backend: str) -> Optional[str]:
    spec = get_method_spec(method)
    if spec.unsupported_reason:
        return spec.unsupported_reason
    if str(backend).lower() not in spec.supports_backends:
        return f"{backend} backend does not support method={method}"
    return None


def get_eviction_class(method: str) -> Type[BaseEviction]:
    spec = get_method_spec(method)
    if spec.module_path is None or spec.class_name is None:
        raise NotImplementedError(spec.unsupported_reason or f"Method {method} is not implemented")
    mod = importlib.import_module(spec.module_path)
    return getattr(mod, spec.class_name)


def _accepted_constructor_params(cls: Type) -> set:
    accepted = set()
    for klass in cls.mro():
        if klass is object:
            continue
        try:
            sig = inspect.signature(klass.__init__)
        except (TypeError, ValueError):
            continue
        for name, param in sig.parameters.items():
            if name == "self" or param.kind == param.VAR_KEYWORD:
                continue
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                accepted.add(name)
    return accepted


def create_eviction(
    method: str,
    cache_size: int,
    k_seq_dim: int = 2,
    v_seq_dim: int = 2,
    sink_size: int = 0,
    recent_size: int = 0,
    **kwargs: Any,
) -> BaseEviction:
    """Create an eviction instance and drop irrelevant kwargs with a warning."""
    canonical, alias_defaults = canonicalize_method(method, kwargs)
    spec = get_method_spec(canonical)
    cls = get_eviction_class(canonical)

    if _clean(method) == "l1_only":
        sink_size = 0
        recent_size = 0
    if canonical == "random":
        sink_size = 0
        recent_size = 0

    if "geom_budget_ratio" not in kwargs and "l1_budget_ratio" in kwargs:
        kwargs["geom_budget_ratio"] = kwargs["l1_budget_ratio"]

    base_kwargs = {
        "cache_size": cache_size,
        "k_seq_dim": k_seq_dim,
        "v_seq_dim": v_seq_dim,
        "sink_size": sink_size,
        "recent_size": recent_size,
    }
    merged = {**kwargs, **alias_defaults, **base_kwargs}
    accepted = _accepted_constructor_params(cls)
    filtered = {k: v for k, v in merged.items() if k in accepted and v is not None}
    ignored = sorted(k for k in merged if k not in accepted)
    if ignored:
        logger.debug(
            "Eviction method %s (%s) ignored unsupported parameter(s): %s",
            method,
            cls.__name__,
            ignored,
        )
    instance = cls(**filtered)
    instance.name = canonical
    instance.method_family = spec.family
    instance.supports_backends = spec.supports_backends
    instance.requires_attention = spec.requires_attention
    instance.requires_scores = spec.requires_scores
    instance.supports_layerwise = spec.supports_layerwise
    instance.supports_headwise = spec.supports_headwise
    instance.score_source = getattr(instance, "score_source", None) or spec.score_source
    instance.score_normalization = str(
        filtered.get("score_normalization", spec.score_normalization)
        or spec.score_normalization
    )
    instance.approximate = spec.approximate
    instance.experimental = spec.experimental
    instance.oracle = spec.oracle
    return instance


BASIC_METHODS = ["full", "recency", "sink_recent", "random", "uniform"]
ATTENTION_METHODS = [
    "attention",
    "last_token_attention",
    "windowed_attention",
    "attention_decay",
    "h2o",
    "snapkv",
    "pyramidkv",
]
GEOMETRY_METHODS = [
    "key_l2_norm",
    "value_l2_norm",
    "key_l1_norm",
    "value_l1_norm",
    "l1_leverage",
    "l2_leverage",
    "l1_prefill_only",
    "l2_prefill_only",
    "l2_key_prefill_only",
    "l1_decode_only",
    "l2_decode_only",
    "compactor",
    "ridge_leverage",
    "approximate_l2_leverage",
    "approximate_l1_leverage",
    "farthest_point_sampling",
    "kmeans_centroid",
    "facility_location_greedy",
    "mahalanobis_distance",
    "zscore_outlier",
    "random_projection_outlier",
]
L1_METHODS = ["l1_leverage", "approximate_l1_leverage"]
HYBRID_METHODS = [
    "attention_l1",
    "attention_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "recency_l1",
    "sink_recent_l1",
    "sink_recent_l2",
    "sink_recent_attention_l1",
    "weighted_score_hybrid",
    "budget_split_hybrid",
]
ORACLE_METHODS = ["oracle_evidence", "oracle_answer_region"]

PAPER_BASELINES = [
    "full",
    "recency",
    "sink_recent",
    "attention",
    "h2o",
    "snapkv",
    "key_l2_norm",
    "value_l2_norm",
    "l2_leverage",
    "l1_leverage",
    "attention_l1",
]

AGGRESSIVE_COMPARISON = [
    "recency",
    "sink_recent",
    "attention",
    "h2o",
    "l2_leverage",
    "l1_leverage",
    "attention_l1",
]
