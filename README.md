# L1/L2 Leverage Scores for KV Cache Eviction

We explored using **leverage scores from randomized linear algebra (a geometric
signal) as a token importance signal for LLM KV caches**, to decide which
tokens to evict during long-context inference. The starting point is how
attention is computed:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

How well a query matches a key ($QK^\top$ large) only decides the weight; the
output is a **weighted sum of the V rows**. A token's actual contribution to
the output also depends on its geometric structure in V space — something pure
attention scoring cannot see. We therefore score the geometric importance of
the KV matrix (leverage scores) directly, as a signal independent of
attention, to retain evidence tokens that attention misses under tight budgets.

## What We Did

1. **Implemented an L1 leverage score estimator**: the concept comes from ℓp
   regression row sampling (Dasgupta et al., SODA 2008); the embedding
   construction borrows Exp(1) reweighting (Woodruff & Zhang, COLT 2013) and
   CountSketch (Charikar et al. 2002). The exact pipeline (Exp(1) reweight →
   CountSketch → QR → row ℓ1 norms) is our own approximate implementation,
   hand-written in PyTorch. We also implemented exact L2 leverage scores
   (squared row norms after QR). Code lives in `src/sketching/`.
2. **Implemented 30+ eviction methods under one framework**: the leverage
   family (L1/L2, prefill-only, key-based scoring, decode-phase updates),
   the attention family (accumulated attention / H2O, windowed), hybrids
   (attention + leverage), and controls such as SnapKV, PyramidKV, Compactor,
   recency, random, norm-based, clustering, PCA, and oracle — all through a
   unified `BaseEviction` abstraction and method registry, with budget-validity
   invariant checks.
3. **Built a complete experiment platform**: YAML experiment configs, a
   unified benchmark runner, official-style scoring (RULER / LongBench),
   overlap / rank-correlation / case-study analysis, plotting scripts, 13 unit
   tests, and a smoke test.
4. **Ported everything to an MLX 4-bit backend**: ran Qwen2.5-7B 4-bit
   quantized inference on Apple Silicon, with prefill compression, decode-phase
   scored eviction, and per-head attention observation; 15 methods pass sanity.
5. **Ran two controlled diagnostics**: RULER NIAH (single-needle retrieval)
   and RULER VT (variable tracking).

## Engineering Implementation

### Two ways of modifying the KV cache

**HF/Torch path** (early, `l1_llm/pos_shift/`): we patched the attention
forward of four architectures (llama / gpt_neox / qwen2 / falcon), replacing
them with our own implementations. After Q/K/V projection, reshape, and RoPE,
the patch stores the results into the `shared_q` global store (last-token
query per layer, head-meaned key rows), from which both leverage scoring and
attention accumulation read. It also handles transformers version differences
(`past_key_value` vs `past_key_values` parameter names, `layer_idx` attribute
polyfill).

**MLX path** (final main path, `src/runners/mlx_runner.py`): we edit the
MLX-LM cache object directly. `keys/values` are the physical storage, `offset`
is the physical length, and `logical_offset` tracks the logical position — so
after compression we still know which real token each slot corresponds to.
`prefill_compress` runs after prefill; during decode, `evict_for_space` frees
room before appending. Each layer is scored and pruned independently. Methods
that keep tokens per head (SnapKV keeps different token sets per head) use a
`head_valid_mask` to mark valid positions per head, which is applied when
computing attention. Attention weights are captured by hooking
`scaled_dot_product_attention`.

### How RoPE works with eviction

This is the core difficulty of the whole system: once keys are rotated with
RoPE, deleting middle tokens makes positions sparse — the cache can no longer
be treated as a plain array.

- **pos_shift (HF path)**: generate cos/sin for the **full cache length**,
  then index by the **original position_ids** — Q is rotated with its current
  real position, K is re-rotated over the full length. Attention scores stay
  correct even with a sparse evicted cache, while keeping the rotated Q needed
  for L1 scoring.
- **MLX path**: `rope_offset` is taken directly from `logical_offset`, so
  compression never changes rotation positions; eviction only does
  `index_select` (taking the kept rows along the sequence dim) and **never
  reorders positions**.
- Numerical safeguards: Q·K^T is computed in float32 to avoid fp16 overflow;
  both old and new `rotary_emb` APIs (passing `position_ids` / passing
  `seq_len`) are supported.

### Position tracking

After each eviction, `BaseEviction` maintains a mapping from each cache slot
to its **original token position** (position map); `gather_by_dim` only gathers
along the sequence dim without reordering. This lets scoring, oracle methods
(using ground-truth evidence positions as an upper bound), and post-hoc
analysis know where each retained token originally was. The MLX side uses
`logical_offset` plus per-layer synchronization.

### Budget semantics and score-update policies

- Standard methods: `budget` is the total live KV; decode evicts first, then
  appends.
- SnapKV / PyramidKV / Compactor: `prompt_prefill` semantics — only the
  prefill prompt cache is compressed, and newly generated decode tokens append
  freely (recorded explicitly as `cache_budget_scope` in results).
- Per-layer budget: the MLX runner supports distributing the budget across
  layers rather than using the same number per layer.
- Score updates: prefill-only (`*_prefill_only`, scored once), every-N-steps
  (`update_interval`, amortizing sketch cost), and continuous decode-phase
  updates.

## Experiment Results

### RULER NIAH (single-needle retrieval)

`results/mlx_qwen25_7b_inst_4bit/ruler/20260611_225910` — 15 methods × 3
budgets (128/256/512) × 20 samples, 900 rows total, all scored with official
metrics.

| method | mean score (0–100) | per budget |
| --- | --- | --- |
| full | 100.0 | 100/100/100 |
| l1_leverage / l2_leverage | 100.0 | 100/100/100 |
| l1/l2_prefill_only | 100.0 | 100/100/100 |
| attention_l1 / attention_l2 | 100.0 | 100/100/100 |
| attention | 60.0 | 0/80/100 |
| snapkv | 43.3 | 5/35/90 |
| pyramidkv | 13.3 | 0/5/35 |
| recency / sink_recent | 1.7 | 0/0/5 |
| compactor / random | 0.0 | 0/0/0 |

Leverage and its hybrids retain the needle at 100% even under the tightest
budget (128), while **attention only catches up at budget 512** — geometric
signals hold evidence tokens under tight budgets where attention fails.

### RULER VT (variable tracking)

`results/mlx_qwen25_7b_inst_4bit/ruler/20260615_223853` — 8 methods × 3
budgets (256/512/1024) × 10 samples, 240 rows total.

| method | mean score (0–100) | per budget |
| --- | --- | --- |
| full | 98.0 | 98/98/98 |
| l2_prefill_only | 97.3 | 100/94/98 |
| attention_l1 | 94.7 | 86/98/100 |
| attention_l2 | 92.0 | 80/96/100 |
| l1_prefill_only | 91.3 | 92/86/96 |
| attention | 85.3 | 56/100/100 |
| snapkv | 73.3 | 38/88/94 |
| compactor | 38.7 | 2/24/90 |

On a state-tracking task, geometric signals remain strong at the tightest
budget (256), while attention catches up once the budget widens; the
attention + leverage hybrids stay stable throughout and are the combination
we favor.

## Conclusion

Leverage scores are a token importance signal independent of attention:
under tight budgets they are clearly better at retaining evidence tokens, and
they complement attention (attention handles semantic selection at wide
budgets; geometric signals guard key evidence). This is the main finding
supported by the codebase and the two experiments above.

## Repository Layout

```text
src/eviction/        30+ eviction methods + registry
src/sketching/       L1/L2 leverage estimators
src/runners/         MLX 4-bit inference + cache editing (2500+ lines)
src/benchmarks/      RULER / LongBench / NIAH adapters
src/evaluation/      official-style scoring
tests/               unit tests (budget invariants, math identities, smoke)
results/             the two complete experiment runs, figures, analysis
configs/             YAML experiment configs
l1_llm/ h2o_llm/ snapkv/ rocketkv/ streaming_llm/   early prototypes (archived)
```

## Quick Start

```bash
pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests/test_p0.py -q   # unit tests
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py           # framework smoke
```

To reproduce a kept run, enter its directory under `results/` and run
`scripts/run_benchmark.py` with the `config.yaml` captured there.
