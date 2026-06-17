#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-.}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIGS=(
  "configs/experiments/paper/qwen25_7b/01_longbench_narrativeqa_qwen25_7b.yaml"
  "configs/experiments/paper/qwen25_7b/02_longbench_hotpotqa_qwen25_7b.yaml"
  "configs/experiments/paper/qwen25_7b/03_longbench_musique_qwen25_7b.yaml"
  "configs/experiments/paper/qwen25_7b/04_longbench_qmsum_qwen25_7b.yaml"
  "configs/experiments/paper/qwen25_7b/05_longbench_gov_report_qwen25_7b.yaml"
  "configs/experiments/paper/qwen25_7b/06_longbench_multifieldqa_en_qwen25_7b.yaml"
  "configs/experiments/paper/qwen25_7b/07_ruler_niah_single_qwen25_7b.yaml"
  "configs/experiments/paper/qwen25_7b/08_ruler_variable_tracking_qwen25_7b.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "==> Running ${cfg}"
  PYTHONPATH=. "$PYTHON_BIN" scripts/run_benchmark.py --config "$cfg"
done
