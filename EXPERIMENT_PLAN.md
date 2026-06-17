# Experiment Plan: L1 Leverage Score for KV Cache Eviction

## Benchmark Tasks (8 subsets)

### LongBench (6 tasks)

| # | Task | Category | Context | Evidence Pattern | Why Selected |
|---|------|----------|---------|-----------------|-------------|
| 1 | narrativeqa | Single-doc QA | ~5K | Concentrated → dispersed | Baseline long-doc QA |
| 2 | hotpotqa | Multi-doc QA | ~3K | Cross-paragraph | **Primary: multi-hop reasoning** |
| 3 | musique | Multi-hop QA | ~3K | Multi-hop | **Primary: most dispersed evidence** |
| 4 | qmsum | Summarization | ~10K | Sparse | Longest context stress test |
| 5 | gov_report | Summarization | ~12K | Sparse | Maximum context length |
| 6 | multifieldqa_en | Multi-domain QA | ~3K | Cross-domain | Cross-domain generalization |

### RULER (2 tasks)

| # | Task | Category | Context | What It Tests |
|---|------|----------|---------|---------------|
| 7 | niah_single | Retrieval | ~4K | Controlled: distance vs recall |
| 8 | variable_tracking | Multi-hop | ~2K | Chain-of-state tracking |

---

## Methods (13 total + RocketKV TBD)

| # | Method | Signal | Update Strategy | Role |
|---|--------|--------|-----------------|------|
| 1 | full | All tokens | No eviction | Upper bound |
| 2 | random | Random | - | Lower bound |
| 3 | recency | Position only | Sliding window | Pure locality baseline |
| 4 | sink_recent | Sink + position | StreamingLLM | StreamingLLM baseline |
| 5 | attention | Causal attention | Per-step accumulate | Pure semantic signal |
| 6 | snapkv | Observation window | Per-step update | Attention variant |
| 7 | pyramidkv | Layer-budget attention | Per-step update | Attention variant |
| 8 | l2_prefill_only | L2 leverage | Prefill once | Compactor-style baseline |
| 9 | l2_leverage | L2 leverage | Every 32 steps | Geometric control |
| 10 | l1_prefill_only | L1 leverage | Prefill once | Verifies dynamic necessity |
| 11 | l1_leverage | L1 leverage | Every 32 steps | Pure geometric signal |
| 12 | attention+l2 | Attn + L2 hybrid | Dynamic mixed | Verifies L1 specificity |
| 13 | **attention+l1** | **Attn + L1 hybrid** | **Dynamic mixed** | **Core method** |

### RocketKV

Whether to add: **Pending decision.** RocketKV (Behnam et al., ICML 2025) is a two-stage method combining SnapKV + Hybrid Sparse Attention (HSA). It exists as a standalone implementation in `rocketkv/` but has not been integrated into the MLX runner's method registry.

- **For**: It is a recent SOTA baseline that combines attention-based eviction with sparse attention, making it a relevant comparison point
- **Against**: It requires significantly more engineering to port to MLX (HSA stage needs chunked key processing), and the 1.5B trial already shows clear separation between methods

---

## Known Issues

### 1. SnapKV MLX Implementation is Simplified (NOT a faithful reproduction)

**File**: `src/runners/mlx_runner.py`, `_attention_scores(mode="snapkv")` (line 689)

**Problem**: The MLX version of SnapKV takes the **mean of observed query vectors** as the score, rather than computing full observation-window attention (Q_obs @ K^T) as described in the original paper (Li et al., 2024).

```python
# Current MLX implementation (simplified):
observed = self.attention_state.get("observe", {}).get(layer_idx, [])
if usable:
    return mx.mean(mx.stack(usable, axis=0), axis=0)

# Should be (original SnapKV):
# 1. Q_obs @ K^T / sqrt(d)  → attention weights
# 2. Causal mask
# 3. Max-pool along sequence dimension
# 4. Sum over observation window queries
```

**Impact**: The PyTorch version (`src/eviction/snapkv.py`) is faithful to the original. The MLX version is an approximation. Results using MLX SnapKV should be labeled as `snapkv (approximate)` or the implementation should be fixed to compute full Q@K^T attention.

### 2. L2 SVD Crashes with SIGABRT on Degenerate Matrices

**File**: `src/runners/mlx_runner.py`, `MLXL2Estimator.scores()` (line 393)

**Problem**: `mx.linalg.svd()` on CPU stream can crash the process (`SIGABRT`, exit code 134) when the input matrix is near-singular. The C++ LAPACK `sgesvdx_` throws an uncaught exception that bypasses Python's `try/except`:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error:
svd_impl: sgesvdx_ failed with code 1
```

**Impact**: `l2_leverage` and `l2_prefill_only` cannot be used reliably until the SVD is replaced with a QR-based alternative or robust preconditioning is added.

### 3. `full` Method Underperforms on Long Contexts

**File**: `configs/mlx/1.5b_trial_real.yaml`

**Problem**: When `prefill_step_size` (1024) is smaller than the actual context length (2500-3100 for LongBench), the `full` method's KV cache may be truncated during chunked prefill, causing it to lose earlier context tokens. This makes `full` an unreliable upper bound.

**Fix**: Set `prefill_step_size` to match or exceed the max expected context length (e.g., 8192).

### 4. evidence_recall Always 0 for LongBench Tasks

**File**: `src/benchmarks/longbench.py`

**Problem**: LongBench benchmark samples do not include `evidence_positions` metadata (unlike NIAH which tracks needle token spans). This causes evidence_recall to be 0 for all methods on all LongBench tasks, making that metric useless for analysis.

**Workaround**: Skip evidence_recall analysis for LongBench; rely on PPL comparison and overlap analysis instead.

---

## Model Plan

| Model | Size | Purpose |
|-------|------|---------|
| Qwen2.5-1.5B | 1.5B | Trend validation (fast sweep, ~2h for 8 tasks) |
| Qwen2.5-7B | 7B | Primary experiments |
| Llama-3.1-8B | 8B | Cross-architecture replication |
| Mistral-7B-v0.3 | 7B | Built-in sliding window → locality bias contrast |
