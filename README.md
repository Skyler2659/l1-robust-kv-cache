# L1 Robust KV Cache

Research code for KV cache eviction experiments on long-context language-model
inference, with a focus on geometric token importance signals such as L1/L2
leverage scores and their complementarity with attention-based selection.

The current main backend is MLX/MLX-LM 4-bit inference on Apple Silicon. The
project is organized for reproducible paper-style benchmarking rather than one
off demos: methods are registered through a central eviction registry, models
are described through model configs/adapters, and benchmark outputs share a
common result schema.

## Research Question

The working hypothesis is:

> KV cache importance is not fully explained by accumulated attention. Some
> semantically critical or evidence-bearing tokens are better captured by
> geometric structure in the key/value cache matrix, and L1/L2 leverage-style
> scores can be combined with attention to improve tight-budget retention.

The codebase is built to test this through:

- controlled retrieval tasks such as RULER NIAH,
- state-tracking tasks such as RULER variable tracking,
- real long-context QA/summarization tasks from LongBench,
- overlap, rank-correlation, evidence recall, latency, and case-study analyses.

## Main Entry Points

```text
scripts/run_benchmark.py        main benchmark runner
scripts/run_analysis.py         post-hoc tables, overlap, rank correlation, cases
scripts/plot_results.py         standard figures from a completed run directory
scripts/run_profile.py          latency and cache-overhead profiling
scripts/run_paper_qwen25_7b_8subsets.sh
                                official 8-subset experiment plan
scripts/run_ruler_niah_qwen25_7b.sh
                                official RULER NIAH diagnostic
scripts/run_ruler_vt_qwen25_7b.sh
                                official RULER variable-tracking diagnostic
scripts/run_longbench_hotpotqa_qwen25_7b.sh
                                LongBench HotpotQA diagnostic
```

`benchmark.py` and the top-level legacy cache folders are preserved for
historical comparison, but the paper path should use `scripts/run_benchmark.py`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Apple Silicon experiments, verify MLX/MLX-LM is installed in the same
environment. The local scripts default to `.venv/bin/python`.

## Repository Layout

```text
configs/
  experiments/
    paper/qwen25_7b/          final 8-subset Qwen2.5-7B plan
    diagnostics/ruler/        focused official RULER configs
    diagnostics/longbench/    focused LongBench configs
    ablations/niah/           decode/prefill and NIAH ablations
    matrix/                   model x benchmark x method matrix configs
    dev/                      small sanity configs used by tests
    archive/legacy/           old exploratory configs, kept out of main path
  eviction/                   one YAML per single eviction method
  eviction/groups/            method-group templates
  models/                     MLX model adapter/config fragments
  models/hf/                  legacy HuggingFace/Torch model fragments
src/
  eviction/                   registry and method implementations
  runners/mlx_runner.py       MLX 4-bit generation, cache editing, scoring
  benchmarks/                 NIAH, RULER, LongBench adapters
  evaluation/                 official-style RULER and LongBench metrics
  analysis/                   overlap, rank correlation, evidence recall, cases
  visualization/              plots and heatmaps
results/
  mlx_qwen25_7b_inst_4bit/ruler/20260611_225910
  mlx_qwen25_7b_inst_4bit/ruler/20260615_223853
```

Only two completed research runs are currently kept in `results/`:

- `20260611_225910`: official RULER NIAH single-needle, 900 rows.
- `20260615_223853`: official RULER variable tracking, 240 rows.

All temporary smoke, partial LongBench, and exploratory result folders were
removed to keep the workspace clean.

## Eviction Methods

Methods are registered in `src/eviction/registry.py`. Each method carries
metadata for method family, backend support, attention requirements, score
requirements, approximate/experimental status, and oracle status.

Core paper methods:

```text
full
random
recency
sink_recent
attention
snapkv
compactor
pyramidkv
l2_prefill_only
l2_key_prefill_only
l2_leverage
l1_prefill_only
l1_leverage
attention_l2
attention_l1
```

Additional supported or scaffolded baselines include windowed/decayed attention,
H2O-style attention, key/value L1/L2 norm, ridge/approximate leverage, clustering
and outlier baselines, weighted hybrids, budget-split hybrids, and oracle
sanity-check methods. Unsupported methods must skip explicitly with an
`unsupported_reason`; they should not silently fall back to another signal.

## Running Checks

Fast unit tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_p0.py -q
```

Framework smoke test with temporary output under `/tmp`:

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py
```

MLX 15-method sanity test on Qwen2.5-0.5B 4-bit:

```bash
TOKENIZERS_PARALLELISM=false PYTHONPATH=. .venv/bin/python scripts/run_benchmark.py \
  --config configs/experiments/dev/qwen25_05b_mlx_method_sanity.yaml \
  --skip_analysis
```

This sanity config writes to `/tmp/l1_robust_kv_cache_smoke` and does not
pollute `results/`.

## Running Experiments

Official RULER NIAH diagnostic:

```bash
bash scripts/run_ruler_niah_qwen25_7b.sh
```

Official RULER variable tracking diagnostic:

```bash
bash scripts/run_ruler_vt_qwen25_7b.sh
```

LongBench HotpotQA diagnostic:

```bash
bash scripts/run_longbench_hotpotqa_qwen25_7b.sh
```

Full 8-subset Qwen2.5-7B plan:

```bash
bash scripts/run_paper_qwen25_7b_8subsets.sh
```

The 8-subset plan is expensive. Run focused diagnostics before launching the
whole script.

## Results and Analysis

A completed run directory contains:

```text
config.yaml
env.json
model_info.json
results.jsonl
summary.json
metrics.csv
samples/
selected_tokens/
scores/
analysis/
figures/
```

Generate or refresh figures:

```bash
PYTHONPATH=. .venv/bin/python scripts/plot_results.py --run-dir <run_dir>
```

Run post-hoc analysis:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_analysis.py --input <run_dir> --config configs/analysis/basic.yaml
```

The plotting script supports arbitrary method names and produces accuracy,
official-score, latency, overlap, position-distribution, and model-method
heatmaps when the corresponding data exists.

## Current Completed Results

RULER NIAH, `results/mlx_qwen25_7b_inst_4bit/ruler/20260611_225910`:

- 15 methods x 3 budgets x 20 samples = 900 rows.
- Official dataset flag is true for all rows.
- No skipped rows or hook errors.
- L1/L2 leverage and attention hybrids achieve 100 across budgets in this
  diagnostic; attention improves with budget; recency and random are weak.

RULER variable tracking, `results/mlx_qwen25_7b_inst_4bit/ruler/20260615_223853`:

- 8 methods x 3 budgets x 10 samples = 240 rows.
- Official dataset flag is true for all rows.
- No skipped rows or hook errors.
- At budget 256, geometric prefill methods and attention hybrids are strong;
  by 512/1024, attention catches up.

These runs are useful evidence for mechanism analysis and method comparison, but
paper main tables should use fixed-seed random or stratified sampling rather
than always taking the first N official samples.

## Important Semantics

- For standard eviction methods, `budget` is a total live KV budget.
- For `snapkv`, `pyramidkv`, and `compactor`, `cache_budget_scope` is
  `prompt_prefill`: the prompt cache is compressed during prefill, then generated
  decode tokens can append. This is recorded in results and sanity checks.
- LongBench HotpotQA with Qwen Instruct should use `prompt_format.mode:
  chat_template` while keeping `use_official_prompt: true`. This avoids verbose
  answer dilution without changing the official prompt text.
- LongBench tasks generally do not provide gold evidence token spans, so
  evidence recall is meaningful for NIAH/RULER-style evidence-tracked tasks but
  not for most LongBench tasks.

## Known Limitations

- MLX manual cache editing is a research runner path, not a production cache
  kernel.
- Some geometric baselines are scaffolded but marked unsupported on MLX if the
  required signal is unavailable or too expensive.
- `hidden_l2_norm` is unsupported on MLX because hidden states are not exposed
  through the current cache-editing interface.
- Oracle methods are for upper bounds and sanity checks only; exclude them from
  fair comparison tables unless explicitly requested.
- The existing kept RULER runs are strong diagnostics, not the final full paper
  benchmark suite.

## Development Notes

Use `rg` for code search, keep configs in the organized experiment folders, and
write new methods through the registry rather than adding special cases in the
benchmark loop. New methods should expose consistent score stats, selected
tokens, unsupported reasons, and method metadata.
