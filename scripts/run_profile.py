#!/usr/bin/env python3
"""Profile eviction methods for timing, memory, and cache overhead."""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_benchmark import eviction_kwargs_from_config, method_needs_attentions
from src.config import ExperimentConfig
from src.eviction.kv_utils import get_kv_seq_len, to_legacy_cache
from src.eviction.registry import create_eviction, get_method_spec, unsupported_reason
from src.models import load_model_and_tokenizer
from src.profiling.memory import MemoryTracker
from src.utils.io import save_results
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed import set_global_seed

logger = get_logger("run_profile")


def _sync(device: str) -> None:
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


@torch.no_grad()
def _profile_once(
    model,
    input_ids: torch.Tensor,
    eviction,
    method_name: str,
    k_seq_dim: int,
    max_steps: int,
    device: str,
) -> Dict[str, Any]:
    past_key_values = None
    if eviction:
        eviction.reset()
    memory = MemoryTracker(device)
    memory.reset_peak()

    output_attentions = method_needs_attentions(method_name)
    prefill_time = 0.0
    decode_time = 0.0
    eviction_wall = 0.0
    kv_lens: List[int] = []
    total_start = time.perf_counter()

    steps = min(max_steps, input_ids.size(1) - 1)
    if steps <= 0:
        raise ValueError("Not enough tokens to profile")

    # Treat the first token as prefill for timing separation.
    token = input_ids[:, 0:1]
    _sync(device)
    t0 = time.perf_counter()
    outputs = model(input_ids=token, past_key_values=None, use_cache=True, output_attentions=output_attentions)
    _sync(device)
    prefill_time += time.perf_counter() - t0
    past_key_values = outputs.past_key_values
    if output_attentions and eviction and getattr(outputs, "attentions", None) is not None:
        for layer_idx, attn in enumerate(outputs.attentions):
            eviction.update_attention(layer_idx, attn)
    if eviction:
        t0 = time.perf_counter()
        past_key_values = eviction(past_key_values)
        eviction_wall += time.perf_counter() - t0

    for idx in range(1, steps):
        token = input_ids[:, idx : idx + 1]
        if eviction:
            t0 = time.perf_counter()
            past_key_values = eviction.evict_for_space(past_key_values, num_coming=1)
            eviction_wall += time.perf_counter() - t0

        _sync(device)
        t0 = time.perf_counter()
        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
        )
        _sync(device)
        decode_time += time.perf_counter() - t0
        past_key_values = outputs.past_key_values

        if output_attentions and eviction and getattr(outputs, "attentions", None) is not None:
            for layer_idx, attn in enumerate(outputs.attentions):
                eviction.update_attention(layer_idx, attn)

        if eviction:
            t0 = time.perf_counter()
            past_key_values = eviction(past_key_values)
            eviction_wall += time.perf_counter() - t0

        pkv, _ = to_legacy_cache(past_key_values)
        if pkv:
            kv_lens.append(get_kv_seq_len(pkv[0][0], k_seq_dim))

    total_time = time.perf_counter() - total_start
    peak_mb = memory.record_peak() if memory.is_cuda else None
    profile_times = getattr(eviction, "profile_times", {}) if eviction else {}

    return {
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "end_to_end_latency_s": total_time,
        "decode_tokens_per_second": max(0, steps - 1) / decode_time if decode_time > 0 else 0.0,
        "end_to_end_tokens_per_second": steps / total_time if total_time > 0 else 0.0,
        "score_compute_time_s": float(profile_times.get("score_compute", 0.0)),
        "topk_select_time_s": float(profile_times.get("topk_select", 0.0)),
        "cache_eviction_update_time_s": eviction_wall,
        "cache_index_select_time_s": float(profile_times.get("cache_prune", 0.0)),
        "peak_memory_mb": peak_mb,
        "max_kv_len": max(kv_lens) if kv_lens else 0,
        "avg_kv_len": statistics.fmean(kv_lens) if kv_lens else 0.0,
        "score_update_count": getattr(eviction, "score_update_count", None) if eviction else None,
    }


def profile_method(
    model,
    input_ids: torch.Tensor,
    method_name: str,
    cfg: ExperimentConfig,
    model_info: Dict[str, Any],
    budget: int,
    max_steps: int,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    runs = []
    for repeat_idx in range(warmup + repeats):
        eviction = None
        if method_name != "full":
            eviction = create_eviction(
                method=method_name,
                cache_size=budget,
                k_seq_dim=model_info["k_seq_dim"],
                v_seq_dim=model_info["v_seq_dim"],
                seed=cfg.seed,
                **eviction_kwargs_from_config(cfg.eviction),
            )
        result = _profile_once(
            model,
            input_ids,
            eviction,
            method_name,
            model_info["k_seq_dim"],
            max_steps,
            cfg.model.device,
        )
        if repeat_idx >= warmup:
            runs.append(result)

    aggregate: Dict[str, Any] = {"method": method_name, "budget": budget, "runs": runs}
    numeric_keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and v is not None]
    for key in numeric_keys:
        stats = _mean_std([float(r[key]) for r in runs if r.get(key) is not None])
        aggregate[f"{key}_mean"] = stats["mean"]
        aggregate[f"{key}_std"] = stats["std"]
    if runs and runs[0].get("peak_memory_mb") is None:
        aggregate["peak_memory_mb_mean"] = None
        aggregate["peak_memory_mb_std"] = None
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/profile")
    args = parser.parse_args()
    setup_logging()

    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.model.device = args.device

    set_global_seed(cfg.seed)
    model, tokenizer, model_info = load_model_and_tokenizer(cfg.model)
    cfg.model.device = model_info["device"]

    text = ("The quick brown fox jumps over the lazy dog. " * 80).strip()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(cfg.model.device)
    max_steps = min(args.max_steps or cfg.benchmark.max_steps, input_ids.size(1) - 1)
    methods = args.methods or cfg.methods or cfg.eviction.methods or [cfg.eviction.method]
    budget = args.budget or (cfg.cache_budgets[0] if cfg.cache_budgets else cfg.eviction.cache_size)
    warmup = args.warmup if args.warmup is not None else cfg.profiling.warmup_steps
    repeats = args.repeats if args.repeats is not None else cfg.profiling.repeats

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "model": cfg.model.name,
        "device": cfg.model.device,
        "dtype": model_info.get("dtype"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "budget": budget,
        "context_length": int(input_ids.size(1)),
        "generated_tokens": int(max_steps),
        "warmup": warmup,
        "repeats": repeats,
    }

    profile_results = []
    for method_name in methods:
        logger.info("Profiling %s", method_name)
        try:
            reason = unsupported_reason(method_name, cfg.model.backend)
            if reason:
                spec = get_method_spec(method_name)
                profile_results.append(
                    {
                        "method": method_name,
                        "method_family": spec.family,
                        "budget": budget,
                        "skipped": True,
                        "unsupported_reason": reason,
                    }
                )
                continue
            profile_results.append(
                profile_method(
                    model,
                    input_ids,
                    method_name,
                    cfg,
                    model_info,
                    budget,
                    max_steps,
                    warmup,
                    repeats,
                )
            )
        except Exception as exc:
            logger.exception("Profile failed for %s", method_name)
            profile_results.append({"method": method_name, "budget": budget, "error": str(exc)})

    payload = {"environment": env, "results": profile_results}
    save_results(payload, out_dir / "profile_results.json")

    print(f"\n{'method':<18} {'decode tok/s':>12} {'e2e tok/s':>12} {'score ms':>10} {'topk ms':>10} {'peak MB':>10}")
    for row in profile_results:
        if "error" in row:
            print(f"{row['method']:<18} ERROR {row['error'][:60]}")
            continue
        if row.get("skipped"):
            print(f"{row['method']:<18} SKIPPED {row.get('unsupported_reason', '')[:60]}")
            continue
        print(
            f"{row['method']:<18} "
            f"{row.get('decode_tokens_per_second_mean', 0):>12.2f} "
            f"{row.get('end_to_end_tokens_per_second_mean', 0):>12.2f} "
            f"{row.get('score_compute_time_s_mean', 0) * 1000:>10.3f} "
            f"{row.get('topk_select_time_s_mean', 0) * 1000:>10.3f} "
            f"{str(row.get('peak_memory_mb_mean')):>10}"
        )
    logger.info("Profile results saved to %s", out_dir / "profile_results.json")


if __name__ == "__main__":
    main()
