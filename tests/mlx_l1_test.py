#!/usr/bin/env python3
"""MLX L1 leverage score KV cache eviction 测试。

4-bit 量化模型 + L1 选择 + KVCache 手动裁剪。
对比 PyTorch MPS 的 ~6 tok/s，预期 MLX 能达到 30-50 tok/s。
"""
import time
import mlx.core as mx
import mlx.nn as nn

CACHE_SIZE = 256
SINK_SIZE = 4
RECENT_SIZE = 64
SKETCH_DIM = 1024
UPDATE_INTERVAL = 32
SEED = 42

PROMPT = (
    "This report describes ordinary background facts about libraries, weather, "
    "software releases, meeting notes, travel plans, and unrelated historical "
    "details. The cat sat on the mat. The dog ran in the yard. "
) * 30
MAX_STEPS = 512


# ── L1 Leverage Score Estimator (random direction sampling) ──────────────

class L1Estimator:
    """Random-direction approximate L1 leverage scores.

    对矩阵 A 采样随机高斯方向 x_j，取 sup_j |a_i^T x_j| / ||A x_j||_1。
    避免 CountSketch 复杂性，MLX 原生实现。
    """

    def __init__(self, n_directions=2000, seed=SEED):
        self.n_dirs = n_directions
        mx.random.seed(seed)

    def scores(self, rows):
        """rows: [n, d] — 返回 [n] L1 杠杆分数。"""
        n, d = rows.shape
        if n <= 1:
            return mx.sum(mx.abs(rows), axis=1)

        # 随机方向 [n_dirs, d]
        directions = mx.random.normal(shape=(self.n_dirs, d))
        # 投影 [n, n_dirs]
        proj = rows @ directions.T
        # 每列 L1 范数 [1, n_dirs]
        col_norms = mx.sum(mx.abs(proj), axis=0, keepdims=True)
        col_norms = mx.maximum(col_norms, 1e-8)
        # |投影| / 列范数 → [n, n_dirs]，取每行 max
        ratios = mx.abs(proj) / col_norms
        scores = mx.max(ratios, axis=1)
        return scores


# ── 主逻辑 ────────────────────────────────────────────────────────────────

def main():
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    model_path = "Qwen/Qwen2.5-1.5B"
    print(f"Loading {model_path} ...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    print(f"Loaded {num_layers} layers in {time.time() - t0:.1f}s")

    # ── 4-bit 量化 ──
    print("Applying 4-bit quantization ...")
    t0 = time.time()
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    print(f"Quantized in {time.time() - t0:.1f}s")

    # ── L1 选择器 ──
    estimators = {i: L1Estimator() for i in range(num_layers)}
    fit_flags = {i: False for i in range(num_layers)}
    last_scores = {}

    # ── KVCache ──
    cache = [KVCache() for _ in range(num_layers)]
    step_count = [0]

    def l1_select_and_evict(cache_list):
        """对所有层执行 L1 选择 + 裁剪。"""
        step_count[0] += 1
        for layer_idx in range(num_layers):
            c = cache_list[layer_idx]
            seq_len = c.offset
            if seq_len <= CACHE_SIZE:
                continue

            # 取 V 行（head 维度取均值）
            v = c.values  # [1, n_kv_heads, S, head_dim]
            v_rows = v[0].mean(axis=0)  # [S, head_dim]
            v_rows_flat = v_rows.astype(mx.float32)

            # 定期重算 L1 得分
            force_refit = (step_count[0] % UPDATE_INTERVAL == 0) or not fit_flags[layer_idx]
            if force_refit:
                fit_flags[layer_idx] = True

            est = estimators[layer_idx]
            scores = est.scores(v_rows_flat)

            # 选择: sink + recent + L1 top-k + last
            sink = min(SINK_SIZE, CACHE_SIZE - 1)
            max_recent = max(0, CACHE_SIZE - sink - 1)
            recent = min(RECENT_SIZE, max_recent)
            l1_budget = max(0, CACHE_SIZE - sink - recent - 1)

            keep_parts = []
            # Sink tokens
            if sink > 0:
                keep_parts.append(mx.arange(sink))
            # Recent tokens (before last)
            if recent > 0:
                keep_parts.append(mx.arange(seq_len - 1 - recent, seq_len - 1))
            # L1 top-k from middle region
            if l1_budget > 0:
                cand_start, cand_end = sink, seq_len - 1 - recent
                if cand_end > cand_start:
                    cand_scores = scores[cand_start:cand_end]
                    topk = min(l1_budget, cand_scores.shape[0])
                    # Use argsort + take instead of topk (MLX compat)
                    top_idx = mx.argpartition(-cand_scores, topk)[:topk]
                    l1_idx = top_idx + cand_start
                    keep_parts.append(l1_idx)
            # Last token
            keep_parts.append(mx.array([seq_len - 1]))

            keep = mx.concatenate(keep_parts)
            keep = mx.sort(keep)
            # Enforce budget (MLX boolean indexing not supported, use simple slice)
            if keep.shape[0] > CACHE_SIZE:
                keep = keep[:CACHE_SIZE]

            # 裁剪 cache
            new_k = mx.take(c.keys, keep, axis=2)
            new_v = mx.take(c.values, keep, axis=2)
            c.keys = new_k
            c.values = new_v
            c.offset = keep.shape[0]

            last_scores[layer_idx] = scores

    # ── Prefill ──
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"\nPrompt: {len(prompt_ids)} tokens")
    print(f"Budget: {CACHE_SIZE}, sink={SINK_SIZE}, recent={RECENT_SIZE}")
    print(f"L1: sketch_dim={SKETCH_DIM}, update_every={UPDATE_INTERVAL}")

    t0 = time.time()
    input_array = mx.array([prompt_ids])
    logits = model(input_array, cache=cache)
    mx.eval(logits)
    prefill_time = time.time() - t0
    print(f"Prefill: {prefill_time:.1f}s ({len(prompt_ids)/prefill_time:.0f} tok/s)")

    # 初始 L1 驱逐
    l1_select_and_evict(cache)
    kv_now = cache[0].offset
    print(f"After L1 eviction: {kv_now} KV positions (budget={CACHE_SIZE})")

    # ── Token-by-token decode ──
    next_token = mx.array([[prompt_ids[-1]]])
    decode_times = []
    evict_times = []

    print(f"\nDecoding {MAX_STEPS} tokens ...")
    for step in range(MAX_STEPS):
        # 为即将到来的 token 腾空间
        for c in cache:
            if c.offset >= CACHE_SIZE:
                # 丢掉最旧的 token（临时），L1 选择会在 post-evict 中处理
                c.keys = c.keys[:, :, 1:, :]
                c.values = c.values[:, :, 1:, :]
                c.offset -= 1

        t_step = time.time()
        logits = model(next_token, cache=cache)
        mx.eval(logits)
        dt_model = time.time() - t_step

        # 采样
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

        # L1 驱逐
        t_evict = time.time()
        l1_select_and_evict(cache)
        mx.eval([c.keys for c in cache])  # 确保驱逐完成
        dt_evict = time.time() - t_evict

        decode_times.append(dt_model)
        evict_times.append(dt_evict)

        if (step + 1) % 200 == 0:
            elapsed = sum(decode_times)
            tok_s = (step + 1) / elapsed if elapsed > 0 else 0
            evict_pct = sum(evict_times) / elapsed * 100 if elapsed > 0 else 0
            print(f"  step={step+1}/{MAX_STEPS} tok/s={tok_s:.1f} evict={evict_pct:.0f}%")

    # ── 结果 ──
    total_model = sum(decode_times)
    total_evict = sum(evict_times)
    total_all = total_model + total_evict
    avg_tok_s = MAX_STEPS / total_model if total_model > 0 else 0
    avg_e2e = MAX_STEPS / (total_model + total_evict) if total_all > 0 else 0

    print(f"\n{'='*65}")
    print(f"MLX L1 leverage eviction (4-bit model):")
    print(f"  Budget: {CACHE_SIZE} | sink={SINK_SIZE} recent={RECENT_SIZE}")
    print(f"  Model-only:  {avg_tok_s:.1f} tok/s ({total_model/MAX_STEPS*1000:.1f} ms/tok)")
    print(f"  E2E (w L1):  {avg_e2e:.1f} tok/s")
    print(f"  L1 overhead: {total_evict/total_model*100:.1f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
