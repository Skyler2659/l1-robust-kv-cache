#!/usr/bin/env python3
"""Ablation runner — systematic sweeps over hyperparameters.

Usage:
    python scripts/run_ablation.py --config configs/benchmark/niah.yaml --sweep budget
    python scripts/run_ablation.py --config configs/benchmark/niah.yaml --sweep update_interval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ExperimentConfig
from src.utils.logging_utils import setup_logging, get_logger

logger = get_logger("run_ablation")

# Default sweep values
SWEEP_DEFAULTS = {
    "budget": [64, 128, 256, 512, 1024],
    "budget_ratio": [0.05, 0.1, 0.2, 0.3, 0.5],
    "update_interval": [1, 16, 32, 64, 128, 0],  # 0 = prefill only
    "sketch_dim": [256, 512, 1024, 2048, 4096],
    "recent_keep": [0, 16, 32, 48, 64, 80, 96, 112],
    "lambda_attn": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "score_source": ["v", "k", "kv"],
    "context_length": [2048, 4096, 8192, 16384],
    "max_new_tokens": [32, 128, 512, 1024],
    "layer_strategy": ["all", "middle", "late"],
}


def run_sweep(
    cfg: ExperimentConfig,
    sweep_dim: str,
    sweep_values: List,
    methods: List[str],
):
    """Run a parameter sweep."""
    from scripts.run_benchmark import run_decode_eval, load_benchmark
    from src.models import load_model_and_tokenizer
    from src.eviction.registry import create_eviction
    from src.utils.seed import set_global_seed
    from src.utils.io import save_results
    from src.profiling.throughput import ThroughputTracker
    from src.profiling.memory import MemoryTracker

    set_global_seed(cfg.seed)

    logger.info(f"Loading model: {cfg.model.name}")
    model, tokenizer, model_info = load_model_and_tokenizer(cfg.model)
    k_seq_dim = model_info["k_seq_dim"]
    v_seq_dim = model_info["v_seq_dim"]

    logger.info(f"Loading benchmark: {cfg.benchmark.name}")
    bench, samples = load_benchmark(cfg, tokenizer)

    out_dir = Path(cfg.output_dir) / f"ablation_{sweep_dim}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for val in sweep_values:
        for method in methods:
            for sample_idx, sample in enumerate(samples):
                input_ids = sample["input_ids"].to(cfg.model.device)
                answer_positions = sample.get("answer_positions")
                eval_positions = answer_positions if cfg.benchmark.eval_target_only else None

                # Apply sweep value to config
                actual_budget = cfg.eviction.cache_size
                actual_update = cfg.eviction.update_interval
                actual_sketch = cfg.eviction.sketch_dim
                actual_recent = cfg.eviction.recent_size

                if sweep_dim == "budget":
                    actual_budget = val
                elif sweep_dim == "recent_keep":
                    actual_recent = val
                elif sweep_dim == "update_interval":
                    actual_update = val
                elif sweep_dim == "sketch_dim":
                    actual_sketch = val

                label = f"{method}_{sweep_dim}{val}_s{sample_idx}"
                logger.info(f"Running: {label}")

                eviction = create_eviction(
                    method=method,
                    cache_size=actual_budget,
                    k_seq_dim=k_seq_dim,
                    v_seq_dim=v_seq_dim,
                    sink_size=cfg.eviction.sink_size,
                    recent_size=actual_recent,
                    score_source=cfg.eviction.score_source,
                    sketch_dim=actual_sketch,
                    update_interval=actual_update,
                    seed=cfg.seed,
                ) if method != "full" else None

                tracker = ThroughputTracker(cfg.model.device)
                memory_tracker = MemoryTracker(cfg.model.device)
                memory_tracker.reset_peak()

                try:
                    result = run_decode_eval(
                        model=model,
                        input_ids=input_ids,
                        eviction=eviction,
                        label=label,
                        k_seq_dim=k_seq_dim,
                        max_steps=cfg.benchmark.max_steps,
                        eval_target_positions=eval_positions,
                        progress_every=200,
                        tracker=tracker,
                        memory_tracker=memory_tracker,
                    )
                except Exception as exc:
                    logger.error(f"Failed: {label}: {exc}")
                    result = {"label": label, "error": str(exc)}

                result.update({
                    "method": method,
                    "sweep_dim": sweep_dim,
                    "sweep_value": val,
                    "sample_idx": sample_idx,
                    "budget": actual_budget,
                })
                all_results.append(result)

    save_results(all_results, out_dir / "ablation_results.json")
    logger.info(f"Ablation results saved to {out_dir}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Ablation: {sweep_dim}")
    print(f"{'='*80}")
    for val in sweep_values:
        for method in methods:
            matching = [r for r in all_results if r.get("sweep_value") == val and r["method"] == method]
            if matching:
                avg_ppl = sum(r.get("ppl", float("inf")) for r in matching) / len(matching)
                avg_tok_s = sum(r.get("tok_per_s", 0) for r in matching) / len(matching)
                print(f"  {sweep_dim}={val:>8}  method={method:<15}  ppl={avg_ppl:>10.4f}  tok/s={avg_tok_s:>8.2f}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sweep", type=str, required=True,
                        choices=list(SWEEP_DEFAULTS.keys()))
    parser.add_argument("--values", type=str, default=None,
                        help="Comma-separated sweep values (overrides defaults)")
    parser.add_argument("--methods", type=str, nargs="+", default=["l1_mixed"])
    args = parser.parse_args()
    setup_logging()

    cfg = ExperimentConfig.from_yaml(args.config)

    if args.values:
        # Parse values based on type
        raw = args.values.split(",")
        if args.sweep in ("score_source", "layer_strategy"):
            sweep_values = [v.strip() for v in raw]
        else:
            sweep_values = [float(v) if "." in v else int(v) for v in raw]
    else:
        sweep_values = SWEEP_DEFAULTS[args.sweep]

    run_sweep(cfg, args.sweep, sweep_values, args.methods)


if __name__ == "__main__":
    main()
