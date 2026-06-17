"""Benchmark factory shared by Torch and MLX runners."""
from __future__ import annotations

import inspect
from typing import Any, Dict, Type

from src.config import ExperimentConfig


def _construct_from_signature(cls: Type, values: Dict[str, Any]):
    sig = inspect.signature(cls.__init__)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    kwargs = {k: v for k, v in values.items() if k in allowed and v is not None}
    return cls(**kwargs)


def instantiate_benchmark(cfg: ExperimentConfig):
    """Instantiate the configured benchmark without loading samples."""
    bench_cfg = cfg.benchmark
    bench_name = bench_cfg.name.lower()
    seed = cfg.seed
    max_samples = bench_cfg.num_samples

    if bench_name == "niah":
        from src.benchmarks.niah import NIAHBenchmark

        values = {
            "depths": bench_cfg.depths or [bench_cfg.needle_depth],
            "max_words": bench_cfg.max_words,
            "needles_per_depth": bench_cfg.needles_per_depth,
            "seed": seed,
            "max_samples": max_samples,
            "use_synthetic_haystack": bench_cfg.use_synthetic_haystack,
            "haystack_repeats": bench_cfg.haystack_repeats,
            "context_length": bench_cfg.context_length,
        }
        return _construct_from_signature(NIAHBenchmark, values)

    if bench_name in ("multi_needle", "multi_niah"):
        from src.benchmarks.niah import MultiNeedleNIAH

        values = {
            "n_needles": bench_cfg.num_needles,
            "max_words": bench_cfg.max_words,
            "n_samples": bench_cfg.n_samples,
            "seed": seed,
            "max_samples": max_samples,
            "use_synthetic_haystack": bench_cfg.use_synthetic_haystack,
            "haystack_repeats": bench_cfg.haystack_repeats,
        }
        return _construct_from_signature(MultiNeedleNIAH, values)

    if bench_name == "variable_depth":
        from src.benchmarks.niah import VariableDepthNIAH

        values = {
            "n_depths": bench_cfg.n_depths,
            "max_words": bench_cfg.max_words,
            "seed": seed,
            "max_samples": max_samples,
            "use_synthetic_haystack": bench_cfg.use_synthetic_haystack,
            "haystack_repeats": bench_cfg.haystack_repeats,
        }
        return _construct_from_signature(VariableDepthNIAH, values)

    if bench_name == "ruler":
        from src.benchmarks.ruler import RULERBenchmark

        values = {
            "tasks": bench_cfg.tasks or [bench_cfg.ruler_task],
            "n_samples_per_task": bench_cfg.n_samples_per_task,
            "seq_words": bench_cfg.seq_words,
            "seed": seed,
            "max_samples": max_samples,
            "use_official_dataset": bench_cfg.use_official_dataset,
            "require_official_dataset": bench_cfg.require_official_dataset,
            "hf_dataset_name": bench_cfg.hf_dataset_name,
            "hf_dataset_config": bench_cfg.hf_dataset_config,
            "hf_split": bench_cfg.hf_split,
        }
        return _construct_from_signature(RULERBenchmark, values)

    if bench_name == "longbench":
        from src.benchmarks.longbench import LongBenchWrapper

        values = {
            "tasks": bench_cfg.tasks or [bench_cfg.longbench_task],
            "max_words": bench_cfg.max_words,
            "n_samples_per_task": bench_cfg.n_samples_per_task,
            "seed": seed,
            "max_samples": max_samples,
            "use_official_prompt": bench_cfg.use_official_prompt,
            "prompt_style": bench_cfg.prompt_style,
            "max_length": bench_cfg.longbench_max_length,
        }
        return _construct_from_signature(LongBenchWrapper, values)

    if bench_name in ("hotpotqa", "multihop"):
        from src.benchmarks.multihop import MultiHopQA

        values = {
            "dataset": bench_cfg.dataset if bench_name == "multihop" else "hotpotqa",
            "split": bench_cfg.split,
            "max_words": bench_cfg.max_words,
            "n_samples": bench_cfg.n_samples,
            "seed": seed,
            "max_samples": max_samples,
        }
        return _construct_from_signature(MultiHopQA, values)

    if bench_name == "reasoning":
        from src.benchmarks.reasoning import ReasoningWithDistractors

        values = {
            "n_samples": bench_cfg.n_samples,
            "n_distractors": bench_cfg.n_distractors,
            "seed": seed,
            "max_samples": max_samples,
        }
        return _construct_from_signature(ReasoningWithDistractors, values)

    raise ValueError(f"Unknown benchmark: {bench_name}")


def load_benchmark(cfg: ExperimentConfig, tokenizer):
    bench = instantiate_benchmark(cfg)
    samples = bench.load_samples(tokenizer, cfg.benchmark.num_samples)
    return bench, samples
