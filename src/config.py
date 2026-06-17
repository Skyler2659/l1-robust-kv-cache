"""Configuration system for full experiment configs and reusable fragments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


_FULL_CONFIG_KEYS = {
    "experiment_name",
    "model",
    "eviction",
    "benchmark",
    "profiling",
    "analysis",
    "cache_budgets",
    "cache_budget_ratios",
    "context_lengths",
    "update_intervals",
    "methods",
    "output_dir",
    "save_selected_tokens",
    "save_scores",
    "save_prompt_text",
    "progress_every",
    "seed",
    "overwrite",
    "run_id",
}


def read_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml_file(data: Dict[str, Any], path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def read_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _field_names(cls) -> set:
    return {f.name for f in fields(cls)}


def _coerce_dataclass(cls, data: Optional[Dict[str, Any]], path: str):
    data = data or {}
    unknown = sorted(set(data) - _field_names(cls))
    if unknown:
        raise ValueError(
            f"Unknown config field(s) at {path}: {unknown}. "
            "If this is a fragment config, load it with the fragment-specific loader."
        )
    return cls(**data)


def detect_config_kind(data: Dict[str, Any]) -> str:
    """Return ``experiment`` or a best-effort fragment kind."""
    if not data:
        return "experiment"
    if any(k in data for k in ("model", "eviction", "benchmark", "profiling", "analysis")):
        return "experiment"
    if any(k in data for k in ("experiment_name", "cache_budgets", "cache_budget_ratios", "seed")):
        return "experiment"
    if "methods" in data:
        return "eviction_fragment"
    if all(k in data for k in ("name",)) and any(
        k in data for k in ("dtype", "device", "trust_remote_code")
    ):
        return "model_fragment"
    if all(k in data for k in ("name",)) and any(
        k in data for k in ("max_steps", "num_samples", "depths", "tasks")
    ):
        return "benchmark_fragment"
    if any(
        isinstance(data.get(k), dict) and "enabled" in data[k]
        for k in (
            "overlap",
            "rank_correlation",
            "evidence_recall",
            "case_study",
        )
    ):
        return "analysis_fragment"
    if set(data).issubset(_field_names(AnalysisConfig)):
        return "analysis_fragment"
    return "fragment"


@dataclass
class ModelConfig:
    name: str = "sshleifer/tiny-gpt2"
    family: Optional[str] = None
    backend: str = "torch"  # torch, mlx
    dtype: str = "float32"  # float32, float16, bfloat16
    device: str = "auto"
    quant_bits: Optional[int] = None
    quant_group_size: int = 64
    mlx_weight_quantize: bool = True
    mlx_cache_type: str = "kv"  # kv, rotating
    prefill_step_size: int = 2048
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    enable_pos_shift: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    local_files_only: bool = False
    revision: Optional[str] = None
    tokenizer_type: Optional[str] = None
    chat_template: Optional[bool] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    max_context_length: Optional[int] = None
    rope_type: Optional[str] = None
    uses_rope: Optional[bool] = None
    cache_format: Optional[str] = None
    supports_attention_output: Optional[bool] = None
    supports_hidden_states: Optional[bool] = None
    supports_mlx_cache_edit: Optional[bool] = None
    default_prompt_format: str = "plain"
    prompt_format: Dict[str, Any] = field(
        default_factory=lambda: {"mode": "plain", "system_prompt": None}
    )
    generation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvictionConfig:
    method: str = "l1_leverage"
    methods: Optional[List[str]] = None
    cache_size: int = 256
    cache_budget_ratio: Optional[float] = None
    sink_size: int = 4
    recent_size: int = 64
    # L1/L2/norm scoring
    score_source: str = "v"  # "v", "k", "kv"
    sketch_dim: int = 1024
    update_interval: int = 32
    update_policy: str = "every_n_steps"  # every_n_steps, prefill_only, never_after_prefill
    seed: int = 0
    use_reweight: bool = False
    score_normalization: str = "rank"
    # Hybrid
    hybrid_mode: str = "budget_split"
    lambda_attn: float = 0.5
    attn_budget_ratio: float = 0.3
    l1_budget_ratio: float = 0.3
    geom_budget_ratio: Optional[float] = None
    recent_budget_ratio: float = 0.3
    sink_budget_ratio: float = 0.1
    geometry_method: str = "l1"
    attention_method: str = "accumulated"
    components: Optional[List[Dict[str, Any]]] = None
    # Layer/head strategy
    layer_strategy: str = "all"
    selected_layers: Optional[List[int]] = None
    head_strategy: str = "shared"
    # PyramidKV
    pyramid_mode: str = "funnel"
    pyramid_beta: int = 20
    # Clustering
    n_clusters: int = 64
    n_iters: int = 5
    clustering_method: str = "farthest_point"
    # Attention/SnapKV-style knobs
    h2o_recent_size: Optional[int] = None
    window_size: int = 32
    attention_window: int = 32
    decay_gamma: float = 0.95
    observation_window: int = 32
    kernel_size: int = 5
    pooling_kernel: int = 5
    pooling_method: str = "avgpool"
    # Compactor faithful prefill compression knobs
    compactor_sketch_dim: int = 48
    compactor_chunk_size: int = 512
    compactor_attention_chunk_size: int = 128
    compactor_accum_blending: float = 0.5
    compactor_protected_first_tokens: Optional[int] = None
    compactor_protected_last_tokens: Optional[int] = None
    ridge_lambda: float = 1e-3
    projection_dim: int = 64
    covariance_mode: str = "diagonal"
    distance_metric: str = "cosine"
    # Debugging
    debug_budget: bool = False


@dataclass
class BenchmarkConfig:
    name: str = "niah"
    # NIAH
    needle_pos: int = 400
    needle_depth: float = 0.5
    prefix_repeat: int = 40
    context_length: int = 4096
    num_needles: int = 1
    depths: Optional[List[float]] = None
    max_words: int = 4000
    needles_per_depth: int = 3
    n_samples: int = 10
    n_depths: int = 20
    use_synthetic_haystack: bool = True
    haystack_repeats: int = 120
    # RULER
    ruler_task: str = "retrieval"
    ruler_num_samples: int = 100
    tasks: Optional[List[str]] = None
    n_samples_per_task: int = 20
    seq_words: int = 2000
    use_official_dataset: bool = False
    require_official_dataset: bool = False
    hf_dataset_name: Optional[str] = None
    hf_dataset_config: Optional[str] = None
    hf_split: Optional[str] = None
    # LongBench / multihop
    longbench_task: str = "narrativeqa"
    longbench_max_length: int = 4096
    dataset: str = "hotpotqa"
    split: str = "validation"
    n_distractors: int = 50
    # General
    max_steps: int = 2048
    max_new_tokens: int = 128
    num_samples: int = 50
    eval_target_only: bool = True
    evaluation: str = "ppl"  # ppl, official, both
    use_official_prompt: bool = False
    prompt_style: Optional[str] = None
    qa_sample_idx: int = 0


@dataclass
class ProfilingConfig:
    enabled: bool = True
    track_memory: bool = True
    track_timing: bool = True
    warmup_steps: int = 5
    log_every: int = 100
    repeats: int = 3


@dataclass
class AnalysisConfig:
    overlap: bool = True
    rank_correlation: bool = True
    evidence_recall: bool = True
    counterfactual_deletion: bool = False
    restoration: bool = False
    leave_one_out: bool = False
    residual_explanation: bool = False
    case_study: bool = True
    case_study_count: int = 5
    include_oracle: bool = False
    output_dir: str = "results/analysis"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    # Sweep dimensions
    cache_budgets: Optional[List[int]] = None
    cache_budget_ratios: Optional[List[float]] = None
    context_lengths: Optional[List[int]] = None
    update_intervals: Optional[List[int]] = None
    methods: Optional[List[str]] = None
    # Output
    output_dir: str = "results"
    save_selected_tokens: bool = True
    save_scores: bool = False
    save_prompt_text: bool = True
    progress_every: int = 100
    seed: int = 42
    overwrite: bool = False
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        kind = detect_config_kind(d)
        if kind != "experiment":
            raise ValueError(
                f"Expected a full experiment config, got {kind}. "
                "Use configs/experiments/**/*.yaml with run_benchmark.py; "
                "configs/models, configs/benchmark, configs/eviction, and configs/analysis "
                "contain model fragments or reusable settings."
            )
        model = _coerce_dataclass(ModelConfig, d.get("model", {}), "model")
        eviction = _coerce_dataclass(EvictionConfig, d.get("eviction", {}), "eviction")
        benchmark = _coerce_dataclass(BenchmarkConfig, d.get("benchmark", {}), "benchmark")
        profiling = _coerce_dataclass(ProfilingConfig, d.get("profiling", {}), "profiling")
        analysis = _coerce_dataclass(AnalysisConfig, d.get("analysis", {}), "analysis")
        top = {
            k: v
            for k, v in d.items()
            if k not in ("model", "eviction", "benchmark", "profiling", "analysis")
        }
        unknown_top = sorted(set(top) - _FULL_CONFIG_KEYS)
        if unknown_top:
            raise ValueError(f"Unknown top-level experiment config field(s): {unknown_top}")
        return cls(
            model=model,
            eviction=eviction,
            benchmark=benchmark,
            profiling=profiling,
            analysis=analysis,
            **top,
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        return cls.from_dict(read_yaml_file(path))

    def to_yaml(self, path: Union[str, Path]) -> None:
        write_yaml_file(self.to_dict(), path)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentConfig":
        return cls.from_dict(read_json_file(path))

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def load_analysis_config(path: Union[str, Path]) -> AnalysisConfig:
    """Load either a flat AnalysisConfig or the nested analysis fragment format."""
    data = read_yaml_file(path)
    kind = detect_config_kind(data)
    if kind == "experiment":
        return ExperimentConfig.from_dict(data).analysis

    flat: Dict[str, Any] = {}
    mapping = {
        "overlap": "overlap",
        "rank_correlation": "rank_correlation",
        "evidence_recall": "evidence_recall",
        "counterfactual_deletion": "counterfactual_deletion",
        "restoration": "restoration",
        "leave_one_out": "leave_one_out",
        "residual_explanation": "residual_explanation",
        "case_study": "case_study",
    }
    for src, dst in mapping.items():
        value = data.get(src)
        if isinstance(value, dict):
            flat[dst] = bool(value.get("enabled", False))
            if src == "case_study" and "count" in value:
                flat["case_study_count"] = int(value["count"])
        elif value is not None:
            flat[dst] = bool(value)
    if "output_dir" in data:
        flat["output_dir"] = data["output_dir"]
    return _coerce_dataclass(AnalysisConfig, flat, "analysis")
