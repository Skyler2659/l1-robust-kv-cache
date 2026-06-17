#!/usr/bin/env python3
"""Main benchmark runner — evaluates eviction methods on long-context tasks.

Usage:
    python scripts/run_benchmark.py --config configs/benchmark/niah.yaml
    python scripts/run_benchmark.py --config configs/benchmark/niah.yaml --method l1_mixed
    python scripts/run_benchmark.py --model Qwen/Qwen2.5-1.5B --benchmark niah --budget 128
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch.nn import CrossEntropyLoss

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ExperimentConfig,
    ModelConfig,
    EvictionConfig,
    BenchmarkConfig,
    read_yaml_file,
    write_yaml_file,
)
from src.eviction.registry import (
    create_eviction,
    PAPER_BASELINES,
    list_methods,
    get_method_spec,
    method_requires_attention,
    unsupported_reason,
)
from src.eviction.score_normalization import merge_score_stats
from src.eviction.kv_utils import get_kv_seq_len, to_legacy_cache
from src.models import load_model_and_tokenizer
from src.profiling.throughput import ThroughputTracker
from src.profiling.memory import MemoryTracker
from src.runners.base import BaseRunner
from src.utils.seed import set_global_seed
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.io import save_jsonl, save_results, save_scores, save_selected_tokens


logger = get_logger("run_benchmark")


def eviction_kwargs_from_config(cfg: EvictionConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    for key in ("method", "methods", "cache_size", "cache_budget_ratio", "seed"):
        data.pop(key, None)
    return data


def make_run_dir(cfg: ExperimentConfig) -> Path:
    base = Path(cfg.output_dir) / cfg.experiment_name
    run_id = cfg.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / run_id
    if out_dir.exists() and not cfg.overwrite:
        suffix = datetime.now().strftime("%f")
        out_dir = base / f"{run_id}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_id = out_dir.name
    return out_dir


def text_hash(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Core decode loop ────────────────────────────────────────────────────

@torch.no_grad()
def run_decode_eval(
    model,
    input_ids: torch.Tensor,
    eviction,
    label: str,
    k_seq_dim: int,
    max_steps: int,
    eval_target_positions=None,
    progress_every: int = 100,
    tracker: ThroughputTracker | None = None,
    memory_tracker: MemoryTracker | None = None,
    output_attentions: bool = False,
) -> dict:
    """Token-by-token decode with eviction, returning PPL + throughput metrics."""
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    nlls: list = []
    step_times: list = []
    kv_lens: list = []

    total_steps = min(input_ids.size(1) - 1, max_steps)
    eval_set = set(eval_target_positions) if eval_target_positions else None
    wall_start = time.perf_counter()

    if eviction is not None:
        eviction.reset()

    if tracker:
        tracker.reset()

    for idx in range(total_steps):
        token = input_ids[:, idx : idx + 1]
        target = input_ids[:, idx + 1 : idx + 2].to(token.device).view(-1)

        # Pre-eviction: make room for incoming token
        if eviction is not None:
            t_evict_start = time.perf_counter()
            past_key_values = eviction.evict_for_space(past_key_values, num_coming=1)
            if tracker:
                tracker.record_phase("eviction", time.perf_counter() - t_evict_start)

        # Forward pass
        if tracker:
            tracker.begin_step()
        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
        )
        if tracker:
            step_elapsed = tracker.end_step()
        else:
            step_elapsed = 0.0

        if output_attentions and eviction is not None:
            attentions = getattr(outputs, "attentions", None)
            if attentions is not None:
                for layer_idx, attention_weights in enumerate(attentions):
                    eviction.update_attention(layer_idx, attention_weights)

        # Compute loss
        nll = loss_fn(
            outputs.logits[:, -1, :].view(-1, model.config.vocab_size), target
        )
        if eval_set is None or (idx + 1) in eval_set:
            nlls.append(nll)

        # Post-forward eviction
        past_key_values = outputs.past_key_values
        if eviction is not None:
            t_evict_start = time.perf_counter()
            past_key_values = eviction(past_key_values)
            if tracker:
                tracker.record_phase("eviction", time.perf_counter() - t_evict_start)

        # Track KV length
        pkv_legacy, _ = to_legacy_cache(past_key_values)
        if pkv_legacy is not None:
            kl = get_kv_seq_len(pkv_legacy[0][0], k_seq_dim)
            kv_lens.append(kl)
            if tracker:
                tracker.record_phase("kv_len", kl)

        # Progress
        step_id = idx + 1
        if progress_every > 0 and (step_id % progress_every == 0 or step_id == total_steps):
            elapsed = time.perf_counter() - wall_start
            tok_s = step_id / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{label}] step={step_id}/{total_steps} "
                f"kv={kv_lens[-1] if kv_lens else 0} "
                f"tok/s={tok_s:.2f} elapsed={elapsed:.1f}s"
            )

    # Memory snapshot
    peak_mb = 0.0
    if memory_tracker:
        peak_mb = memory_tracker.record_peak()
        if tracker:
            tracker.record_memory(peak_mb)

    if not nlls:
        raise ValueError(
            f"No target tokens evaluated. eval_target_positions={eval_target_positions}, "
            f"max_steps={max_steps}"
        )

    mean_nll = torch.stack(nlls).mean().item()
    total_s = time.perf_counter() - wall_start
    stats = tracker.get_stats() if tracker else None

    result = {
        "label": label,
        "steps": total_steps,
        "ppl": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "tok_per_s": total_steps / total_s if total_s > 0 else float("inf"),
        "avg_ms_per_tok": (total_s / total_steps) * 1000.0,
        "max_kv_len": max(kv_lens) if kv_lens else 0,
        "final_kv_len": kv_lens[-1] if kv_lens else 0,
        "peak_memory_mb": peak_mb,
        "total_time_s": total_s,
    }

    if stats:
        result["throughput"] = stats.to_dict()

    # Collect per-layer diagnostics
    if eviction is not None and hasattr(eviction, "last_scores"):
        result["has_scores"] = len(eviction.last_scores) > 0
    if eviction is not None and hasattr(eviction, "last_selected"):
        result["has_selected"] = len(eviction.last_selected) > 0

    return result


def method_needs_attentions(method_name: str) -> bool:
    return method_requires_attention(method_name)


def summarize_score_stats(scores: Dict[int, torch.Tensor], topk: int = 5) -> Dict[str, Any]:
    if not scores:
        return {}
    flat = torch.cat([s.flatten().float() for s in scores.values() if s.numel() > 0])
    if flat.numel() == 0:
        return {}
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {"all_non_finite": True}
    k = min(topk, finite.numel())
    return {
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
        "top_values": [float(x) for x in torch.topk(finite, k).values.tolist()],
    }


def make_skipped_result(
    cfg: ExperimentConfig,
    sample: Dict[str, Any],
    sample_idx: int,
    method_name: str,
    budget: int,
    reason: str,
) -> Dict[str, Any]:
    try:
        spec = get_method_spec(method_name)
        family = spec.family
        oracle = spec.oracle
    except Exception:
        family = "unknown"
        oracle = False
    prompt_text = sample.get("prompt")
    return {
        "label": f"{method_name}_b{budget}_s{sample_idx}",
        "experiment_name": cfg.experiment_name,
        "run_id": cfg.run_id,
        "sample_id": sample_idx,
        "sample_idx": sample_idx,
        "method": method_name,
        "method_family": family,
        "budget": budget,
        "cache_budget": budget,
        "model": cfg.model.name,
        "model_name": cfg.model.name,
        "model_family": cfg.model.family,
        "backend": cfg.model.backend,
        "quant_bits": cfg.model.quant_bits,
        "benchmark": cfg.benchmark.name,
        "context_length": int(sample["input_ids"].size(1)) if "input_ids" in sample else None,
        "prompt_hash": text_hash(prompt_text),
        "prediction": None,
        "generated_text": None,
        "ground_truth": sample.get("ground_truth"),
        "correct": None,
        "contains_ground_truth": None,
        "exact_match": None,
        "ppl": None,
        "mean_nll": None,
        "evidence_positions": sample.get("evidence_positions") or [],
        "selected_tokens": {},
        "selected_tokens_by_layer": {},
        "evidence_recall": None,
        "evidence_precision": None,
        "score_stats": {},
        "score_normalization": cfg.eviction.score_normalization,
        "seed": cfg.seed,
        "score_update_count": 0,
        "max_kv_len": None,
        "final_kv_len": None,
        "cache_shape_summary": {},
        "total_time_s": 0.0,
        "prefill_time_s": 0.0,
        "decode_time_s": 0.0,
        "score_time_s": 0.0,
        "eviction_time_s": 0.0,
        "topk_time_s": 0.0,
        "cache_rebuild_time_s": 0.0,
        "tokens_per_second": None,
        "skipped": True,
        "skipped_reason": reason,
        "unsupported_reason": reason,
        "oracle": oracle,
        "metadata": sample.get("metadata", {}),
    }


def sanity_checks_for_result(result: Dict[str, Any]) -> Dict[str, Any]:
    violations: List[str] = []
    method = result.get("canonical_method") or result.get("method")
    budget = int(result.get("budget") or 0)
    context_length = int(result.get("context_length") or 0)
    max_new = int(result.get("max_new_tokens") or 0)
    final_kv_len = result.get("final_kv_len")
    if method != "full" and final_kv_len is not None and budget > 0 and int(final_kv_len) > budget:
        violations.append(f"final_kv_len={final_kv_len} exceeds budget={budget}")
    selected = result.get("selected_tokens_by_layer") or result.get("selected_tokens") or {}
    for layer, values in selected.items():
        vals = [int(x) for x in values]
        if len(vals) != len(set(vals)):
            violations.append(f"layer {layer}: duplicate selected tokens")
        if any(x < 0 or (context_length and x >= context_length + max_new) for x in vals):
            violations.append(f"layer {layer}: selected token out of original stream range")
        if method != "full" and budget > 0 and len(vals) > budget:
            violations.append(f"layer {layer}: selected count {len(vals)} exceeds budget {budget}")
    return {"passed": not violations, "violations": violations}


# ── Benchmark loading ───────────────────────────────────────────────────

def _construct_from_signature(cls: Type, values: Dict[str, Any]):
    """Construct *cls* using only keyword arguments accepted by its __init__."""
    sig = inspect.signature(cls.__init__)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    kwargs = {k: v for k, v in values.items() if k in allowed and v is not None}
    ignored = sorted(k for k, v in values.items() if k not in allowed and v is not None)
    if ignored:
        logger.debug("%s ignored benchmark fields: %s", cls.__name__, ignored)
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
    """Load benchmark samples based on config."""
    bench = instantiate_benchmark(cfg)
    samples = bench.load_samples(tokenizer, cfg.benchmark.num_samples)
    return bench, samples


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="L1 KV Cache Benchmark Runner")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--backend", type=str, default=None, choices=["torch", "mlx"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--quant_bits", type=int, default=None)
    p.add_argument("--benchmark", type=str, default=None,
                    choices=["niah", "multi_needle", "variable_depth", "ruler",
                             "longbench", "hotpotqa", "reasoning"])
    p.add_argument("--method", type=str, default=None, nargs="+",
                    help="Eviction method(s). Default: paper baselines")
    p.add_argument("--budget", type=int, default=None, nargs="+",
                    help="Cache budget(s)")
    p.add_argument("--budget_ratio", type=float, default=None, nargs="+",
                    help="Cache budget ratio(s) of context length")
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--max_words", type=int, default=None,
                   help="Max haystack words for NIAH benchmark")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--sink_size", type=int, default=None)
    p.add_argument("--recent_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--save_scores", action="store_true")
    p.add_argument("--save_selected", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    return p.parse_args()


def is_matrix_config(data: Dict[str, Any]) -> bool:
    exp = data.get("experiment")
    return isinstance(exp, dict) and bool(exp.get("models")) and bool(exp.get("benchmarks"))


def run_matrix_config(args, data: Dict[str, Any], source_path: str) -> None:
    """Expand model x benchmark x context configs and run this script recursively."""
    matrix = data["experiment"]
    defaults = data.get("defaults", {}) or {}
    experiment_name = matrix.get("name", "matrix_experiment")
    models = matrix.get("models") or []
    benchmarks = matrix.get("benchmarks") or []
    methods = matrix.get("methods") or []
    budgets = matrix.get("budgets") or [256]
    context_lengths = matrix.get("context_lengths") or [None]
    samples = int(matrix.get("samples_per_setting", 1))
    output_dir = matrix.get("output_dir", "results")
    seed = int(matrix.get("seed", 42))

    generated_dir = Path(output_dir) / "_generated_configs" / experiment_name
    generated_dir.mkdir(parents=True, exist_ok=True)
    commands = []

    for model_path in models:
        model_fragment = read_yaml_file(model_path)
        model_cfg = model_fragment.get("model", model_fragment)
        model_slug = Path(model_path).stem
        for benchmark_name in benchmarks:
            for context_length in context_lengths:
                eviction_cfg = dict(defaults.get("eviction", {}))
                eviction_cfg.setdefault("method", methods[0] if methods else "recency")
                eviction_cfg.setdefault("cache_size", budgets[0])
                eviction_cfg["methods"] = methods

                benchmark_cfg = dict(defaults.get("benchmark", {}))
                benchmark_cfg["name"] = benchmark_name
                benchmark_cfg["num_samples"] = samples
                if context_length is not None:
                    benchmark_cfg["context_length"] = int(context_length)
                    benchmark_cfg["max_steps"] = int(context_length)
                    benchmark_cfg.setdefault("max_words", max(80, int(context_length) // 2))
                if benchmark_name == "ruler":
                    benchmark_cfg.setdefault("tasks", ["retrieval"])
                    benchmark_cfg.setdefault("n_samples_per_task", samples)
                if benchmark_name == "niah":
                    benchmark_cfg.setdefault("depths", [0.5])
                    benchmark_cfg.setdefault("needles_per_depth", max(1, samples))

                concrete = {
                    "experiment_name": experiment_name,
                    "model": model_cfg,
                    "eviction": eviction_cfg,
                    "benchmark": benchmark_cfg,
                    "analysis": dict(defaults.get("analysis", {})),
                    "methods": methods,
                    "cache_budgets": budgets,
                    "output_dir": output_dir,
                    "save_selected_tokens": True,
                    "save_scores": True,
                    "save_prompt_text": True,
                    "progress_every": args.progress_every,
                    "seed": seed,
                    "overwrite": False,
                }
                ctx_slug = f"ctx{context_length}" if context_length is not None else "ctxdefault"
                path = generated_dir / f"{model_slug}_{benchmark_name}_{ctx_slug}.yaml"
                write_yaml_file(concrete, path)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(path),
                ]
                if args.skip_analysis:
                    cmd.append("--skip_analysis")
                commands.append(cmd)

    logger.info(
        "Expanded matrix config %s into %d concrete run(s)",
        source_path,
        len(commands),
    )
    for cmd in commands:
        logger.info("Matrix run: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)


def main():
    args = parse_args()
    setup_logging()

    # Load config
    if args.config:
        raw_config = read_yaml_file(args.config)
        if is_matrix_config(raw_config):
            run_matrix_config(args, raw_config, args.config)
            return
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()

    # Apply CLI overrides
    if args.model:
        cfg.model.name = args.model
    if args.backend:
        cfg.model.backend = args.backend
    if args.device:
        cfg.model.device = args.device
    if args.dtype:
        cfg.model.dtype = args.dtype
    if args.quant_bits is not None:
        cfg.model.quant_bits = args.quant_bits
    if args.benchmark:
        cfg.benchmark.name = args.benchmark
    if args.context_length:
        cfg.benchmark.context_length = args.context_length
    if args.max_steps:
        cfg.benchmark.max_steps = args.max_steps
    if args.max_words:
        cfg.benchmark.max_words = args.max_words
    if args.num_samples:
        cfg.benchmark.num_samples = args.num_samples
    if args.sink_size is not None:
        cfg.eviction.sink_size = args.sink_size
    if args.recent_size is not None:
        cfg.eviction.recent_size = args.recent_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir:
        cfg.output_dir = args.output_dir

    # Set seed
    set_global_seed(cfg.seed)

    # Determine methods to run
    methods = args.method or cfg.methods or cfg.eviction.methods or [cfg.eviction.method]
    if methods == ["paper"]:
        methods = PAPER_BASELINES

    # Determine budgets
    budgets = args.budget or cfg.cache_budgets or [cfg.eviction.cache_size]
    budget_ratios = args.budget_ratio or cfg.cache_budget_ratios or []

    if cfg.model.backend.lower() == "mlx":
        logger.info("Dispatching to MLX runner")
        from src.runners.mlx_runner import MLXRunner

        runner = MLXRunner(cfg)
        out_dir = runner.run(
            methods=methods,
            budgets=budgets,
            budget_ratios=budget_ratios,
            skip_analysis=args.skip_analysis,
        )
        logger.info("Results saved to %s", out_dir)
        return

    # Load model
    logger.info(f"Loading model: {cfg.model.name}")
    model, tokenizer, model_info = load_model_and_tokenizer(cfg.model)
    cfg.model.device = model_info["device"]
    k_seq_dim = model_info["k_seq_dim"]
    v_seq_dim = model_info["v_seq_dim"]
    logger.info(f"Model info: {model_info}")

    # Load benchmark
    logger.info(f"Loading benchmark: {cfg.benchmark.name}")
    bench, samples = load_benchmark(cfg, tokenizer)
    logger.info(f"Loaded {len(samples)} samples")

    # Auto-adjust max_steps to cover answer tokens in all samples
    max_answer_pos = 0
    for sample in samples:
        positions = sample.get("answer_positions") or sample.get("eval_positions") or []
        if positions:
            max_answer_pos = max(max_answer_pos, max(positions))
    if max_answer_pos > 0 and cfg.benchmark.max_steps < max_answer_pos:
        seq_cap = max(s["input_ids"].size(1) for s in samples) - 1
        needed = min(max_answer_pos, seq_cap)
        logger.info(f"max_steps: {cfg.benchmark.max_steps} -> {needed} (to reach eval targets)")
        cfg.benchmark.max_steps = needed

    # Prepare output directory
    out_dir = make_run_dir(cfg)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Save config
    cfg.to_yaml(out_dir / "config.yaml")

    # Run experiments
    all_results: list = []
    memory_tracker = MemoryTracker(cfg.model.device)

    for budget in budgets:
        for method_name in methods:
            for sample_idx, sample in enumerate(samples):
                input_ids = sample["input_ids"].to(cfg.model.device)
                answer_positions = sample.get("answer_positions")
                eval_positions = answer_positions if cfg.benchmark.eval_target_only else None

                # Compute actual cache size from ratio if specified
                actual_budget = budget
                for ratio in budget_ratios:
                    actual_budget = max(1, int(input_ids.size(1) * ratio))

                logger.info(
                    f"Running: method={method_name} budget={actual_budget} "
                    f"sample={sample_idx}/{len(samples)}"
                )

                try:
                    method_spec = get_method_spec(method_name)
                    reason = unsupported_reason(method_name, cfg.model.backend)
                except Exception as exc:
                    method_spec = None
                    reason = str(exc)
                if reason:
                    result = make_skipped_result(
                        cfg, sample, sample_idx, method_name, actual_budget, reason
                    )
                    all_results.append(result)
                    save_results(
                        result,
                        out_dir / "samples" / f"{method_name}_b{actual_budget}_s{sample_idx}.json",
                    )
                    continue

                # Create eviction method
                eviction = create_eviction(
                    method=method_name,
                    cache_size=actual_budget,
                    k_seq_dim=k_seq_dim,
                    v_seq_dim=v_seq_dim,
                    seed=cfg.seed,
                    **eviction_kwargs_from_config(cfg.eviction),
                ) if method_name != "full" else None
                if eviction is not None:
                    eviction.set_sample_metadata(sample)

                # Tracker
                tracker = ThroughputTracker(cfg.model.device)
                memory_tracker.reset_peak()

                # Run
                try:
                    result = run_decode_eval(
                        model=model,
                        input_ids=input_ids,
                        eviction=eviction,
                        label=f"{method_name}_b{actual_budget}_s{sample_idx}",
                        k_seq_dim=k_seq_dim,
                        max_steps=cfg.benchmark.max_steps,
                        eval_target_positions=eval_positions,
                        progress_every=args.progress_every,
                        tracker=tracker,
                        memory_tracker=memory_tracker,
                        output_attentions=method_needs_attentions(method_name),
                    )
                except Exception as exc:
                    logger.error(f"Failed: {method_name} budget={actual_budget}: {exc}")
                    result = {
                        "label": f"{method_name}_b{actual_budget}_s{sample_idx}",
                        "error": str(exc),
                    }

                # Add metadata
                prompt_text = sample.get("prompt")
                result.update({
                    "experiment_name": cfg.experiment_name,
                    "run_id": cfg.run_id,
                    "sample_id": sample_idx,
                    "method": method_name,
                    "canonical_method": getattr(eviction, "name", method_name) if eviction else "full",
                    "method_family": getattr(method_spec, "family", "recency") if method_spec else None,
                    "budget": actual_budget,
                    "cache_budget": actual_budget,
                    "sample_idx": sample_idx,
                    "model": cfg.model.name,
                    "model_name": cfg.model.name,
                    "model_family": model_info.get("model_family"),
                    "backend": cfg.model.backend,
                    "quant_bits": cfg.model.quant_bits,
                    "benchmark": cfg.benchmark.name,
                    "max_new_tokens": cfg.benchmark.max_new_tokens,
                    "context_length": input_ids.size(1),
                    "prompt": prompt_text,
                    "prompt_hash": text_hash(prompt_text),
                    "prediction": None,
                    "generated_text": None,
                    "ground_truth": sample.get("ground_truth"),
                    "contains_ground_truth": None,
                    "exact_match": None,
                    "correct": None,
                    "metric": None if "error" in result else {"ppl": result.get("ppl")},
                    "loss": result.get("mean_nll"),
                    "latency": result.get("total_time_s"),
                    "tokens_per_second": result.get("tok_per_s"),
                    "peak_memory": result.get("peak_memory_mb"),
                    "evidence_positions": sample.get("evidence_positions"),
                    "selected_tokens_by_layer": None,
                    "evidence_recall": None,
                    "evidence_precision": None,
                    "score_normalization": cfg.eviction.score_normalization,
                    "seed": cfg.seed,
                    "score_time_s": getattr(eviction, "profile_times", {}).get("score_compute") if eviction else 0.0,
                    "topk_time_s": getattr(eviction, "profile_times", {}).get("topk_select") if eviction else 0.0,
                    "cache_rebuild_time_s": getattr(eviction, "profile_times", {}).get("cache_prune") if eviction else 0.0,
                    "prefill_time_s": 0.0,
                    "decode_time_s": result.get("total_time_s"),
                    "eviction_time_s": result.get("throughput", {}).get("total_eviction_time_s") if result.get("throughput") else None,
                    "cache_shape_summary": {},
                    "metadata": sample.get("metadata", {}),
                    "score_update_count": getattr(eviction, "score_update_count", None) if eviction else None,
                    "approximate": bool(getattr(method_spec, "approximate", False)) if method_spec else False,
                    "experimental": bool(getattr(method_spec, "experimental", False)) if method_spec else False,
                    "oracle": bool(getattr(method_spec, "oracle", False)) if method_spec else False,
                    "skipped": False,
                    "skipped_reason": None,
                    "unsupported_reason": None,
                })

                selected_path = None
                scores_path = None

                save_selected_flag = args.save_selected or cfg.save_selected_tokens
                save_scores_flag = args.save_scores or cfg.save_scores

                if save_scores_flag and eviction and hasattr(eviction, "last_scores"):
                    scores_path = out_dir / "scores" / f"{method_name}_b{actual_budget}_s{sample_idx}.pt"
                    save_scores(eviction.last_scores, scores_path)
                    result["scores_path"] = str(scores_path)

                if save_selected_flag and eviction and hasattr(eviction, "last_selected"):
                    selected_path = out_dir / "selected_tokens" / f"{method_name}_b{actual_budget}_s{sample_idx}.pt"
                    save_selected_tokens(eviction.last_selected, selected_path)
                    result["selected_tokens_path"] = str(selected_path)
                    result["selected_tokens"] = {
                        str(k): v.detach().cpu().tolist()
                        for k, v in eviction.last_selected.items()
                    }
                    result["selected_tokens_by_layer"] = result["selected_tokens"]
                    if hasattr(eviction, "last_component_sources"):
                        result["selected_token_sources"] = getattr(eviction, "last_component_sources", {})

                if eviction is not None and hasattr(eviction, "last_scores"):
                    stats = eviction.get_score_stats() if hasattr(eviction, "get_score_stats") else {}
                    if not stats:
                        stats = summarize_score_stats(eviction.last_scores)
                    if stats:
                        result["score_stats"] = stats
                        if "raw_score_stats" in stats:
                            result["raw_score_stats"] = stats.get("raw_score_stats")
                            result["normalized_score_stats"] = stats.get("normalized_score_stats")
                            result["top_score_values"] = stats.get("top_score_values")
                        if method_needs_attentions(method_name):
                            logger.info(
                                "Score stats for %s sample=%s: min=%.4g max=%.4g "
                                "mean=%.4g std=%.4g top=%s",
                                method_name,
                                sample_idx,
                                stats.get("min", float("nan")),
                                stats.get("max", float("nan")),
                                stats.get("mean", float("nan")),
                                stats.get("std", float("nan")),
                                stats.get("top_values", []),
                            )

                result["sanity_checks"] = sanity_checks_for_result(result)
                result["sanity_check_failed"] = bool(result["sanity_checks"].get("violations"))
                all_results.append(result)
                save_results(
                    result,
                    out_dir / "samples" / f"{method_name}_b{actual_budget}_s{sample_idx}.json",
                )

    # Save all results
    save_results(all_results, out_dir / "results.json")
    save_jsonl(all_results, out_dir / "results.jsonl")
    save_jsonl(all_results, out_dir / "samples.jsonl")
    bundle = BaseRunner(cfg)
    bundle.backend_name = cfg.model.backend
    bundle.save_metrics_csv(all_results, out_dir / "metrics.csv")
    save_results(bundle.summary(all_results), out_dir / "summary.json")
    save_results(bundle.profiling_summary(all_results), out_dir / "profiling_summary.json")

    # Print summary table
    _print_summary_table(all_results)

    # Run analysis if enabled
    if not args.skip_analysis and cfg.analysis.overlap:
        logger.info("Running post-hoc analysis...")
        try:
            from scripts.run_analysis import run_analysis
            run_analysis(all_results, cfg, out_dir)
        except Exception as exc:
            logger.warning(f"Analysis failed: {exc}")

    logger.info(f"Results saved to {out_dir}")


def _print_summary_table(results: list):
    """Print a summary table of results grouped by method."""
    print("\n" + "=" * 100)
    print(f"{'method':<20} {'budget':>8} {'ppl':>10} {'tok/s':>10} {'avg_ms':>10} {'max_kv':>8} {'mem_mb':>8}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['method']:<20} {r.get('budget', '?'):>8} {'ERROR':>10} {r['error'][:40]}")
        elif r.get("skipped"):
            print(f"{r['method']:<20} {r.get('budget', '?'):>8} {'SKIPPED':>10} {str(r.get('skipped_reason'))[:40]}")
        else:
            print(
                f"{r['method']:<20} {r['budget']:>8} {r['ppl']:>10.4f} "
                f"{r['tok_per_s']:>10.2f} {r['avg_ms_per_tok']:>10.3f} "
                f"{r['max_kv_len']:>8} {r.get('peak_memory_mb', 0):>8.1f}"
            )
    print("=" * 100)


if __name__ == "__main__":
    main()
