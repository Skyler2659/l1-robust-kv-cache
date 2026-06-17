#!/usr/bin/env python3
"""No-pytest smoke tests for P0 framework repairs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEV_CONFIG = "configs/experiments/dev/tiny_niah_cpu.yaml"
TMP_OUTPUT = Path("/tmp/l1_robust_kv_cache_smoke")


def run(cmd):
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise SystemExit(completed.returncode)
    return completed


def core_assertions():
    from scripts.run_benchmark import instantiate_benchmark
    from src.config import ExperimentConfig
    from src.eviction.base import validate_selected_indices
    from src.eviction.l2_leverage import l2_row_leverage_scores
    from src.eviction.registry import create_eviction

    cfg = ExperimentConfig.from_yaml(ROOT / DEV_CONFIG)
    assert instantiate_benchmark(cfg).name == "niah"

    for method in ["recency", "sink_recent", "attention", "l1_leverage", "l2_leverage", "attention+l1"]:
        eviction = create_eviction(
            method,
            cache_size=8,
            k_seq_dim=2,
            v_seq_dim=2,
            sink_size=2,
            recent_size=3,
            sketch_dim=16,
            debug_budget=True,
        )
        k = torch.randn(1, 2, 20, 4)
        v = torch.randn(1, 2, 20, 4)
        eviction(((k, v),))
        validate_selected_indices(eviction.last_selected[0], 20, 8)

    rows = torch.randn(12, 4)
    scores = l2_row_leverage_scores(rows)
    assert abs(float(scores.sum()) - torch.linalg.matrix_rank(rows.float()).item()) < 1e-4
    print("core assertions ok")


def main():
    core_assertions()
    py = sys.executable
    run([py, "scripts/run_benchmark.py", "--config", DEV_CONFIG, "--num_samples", "1", "--progress_every", "120", "--skip_analysis"])
    latest = sorted((TMP_OUTPUT / "tiny_niah_cpu").glob("20*"))[-1]
    run([py, "scripts/run_analysis.py", "--input", str(latest), "--config", "configs/analysis/basic.yaml"])
    run([
        py,
        "scripts/run_profile.py",
        "--config",
        DEV_CONFIG,
        "--max_steps",
        "32",
        "--warmup",
        "1",
        "--repeats",
        "1",
        "--output_dir",
        str(TMP_OUTPUT / "profile"),
    ])
    print("smoke test ok")


if __name__ == "__main__":
    main()
