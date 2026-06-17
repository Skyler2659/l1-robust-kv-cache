#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="configs/experiments/diagnostics/ruler/qwen25_7b_ruler_niah_single_official.yaml"
RUN_ROOT="results/mlx_qwen25_7b_inst_4bit/ruler"

"$PYTHON_BIN" scripts/run_benchmark.py --config "$CONFIG" "$@"

RUN_DIR="$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
echo "Latest run directory: $RUN_DIR"
"$PYTHON_BIN" scripts/plot_results.py --run-dir "$RUN_DIR"
