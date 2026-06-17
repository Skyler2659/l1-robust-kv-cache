#!/usr/bin/env python3
"""极简 MLX recency 测试：token-by-token 解码 + 滑动窗口 KV cache。

对比 PyTorch MPS 的 ~15 tok/s，预期 MLX 能达到 40-80 tok/s。
"""
import time
import mlx.core as mx

CACHE_SIZE = 256          # KV cache 预算
PROMPT = (
    "This report describes ordinary background facts about libraries, weather, "
    "software releases, meeting notes, travel plans, and unrelated historical "
    "details. The cat sat on the mat. The dog ran in the yard. "
) * 30                    # ~2000 tokens 左右的 prompt
MAX_STEPS = 512           # 解码步数


def load_model():
    """加载 Qwen2.5-1.5B，返回 (model, tokenizer, num_layers)。"""
    from mlx_lm import load

    model_path = "Qwen/Qwen2.5-1.5B"
    print(f"Loading {model_path} ...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    print(f"Loaded in {time.time() - t0:.1f}s, {num_layers} layers")
    return model, tokenizer, num_layers


def make_cache(num_layers, max_size):
    """创建 RotatingKVCache 队列，原生支持滑动窗口。"""
    from mlx_lm.models.cache import RotatingKVCache
    return [RotatingKVCache(max_size=max_size, keep=0) for _ in range(num_layers)]


def main():
    model, tokenizer, num_layers = load_model()

    # Tokenize prompt
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt tokens: {len(prompt_ids)}")
    print(f"Cache size: {CACHE_SIZE}, max decode steps: {MAX_STEPS}")
    print()

    # ── 创建滑动窗口 cache ──
    cache = make_cache(num_layers, CACHE_SIZE)

    # ── Prefill ──
    t0 = time.time()
    input_array = mx.array([prompt_ids])
    logits = model(input_array, cache=cache)
    mx.eval(logits)
    prefill_time = time.time() - t0
    print(f"Prefill: {len(prompt_ids)} tokens in {prefill_time:.1f}s "
          f"({len(prompt_ids) / prefill_time:.0f} tok/s)")
    kv_now = cache[0].offset
    print(f"Initial KV: {kv_now} positions (budget={CACHE_SIZE})")

    # ── Token-by-token 解码 ──
    next_token = mx.array([[prompt_ids[-1]]])
    decode_times = []

    print(f"\nDecoding {MAX_STEPS} tokens (progress every 200)...")
    for step in range(MAX_STEPS):
        t_step = time.time()
        logits = model(next_token, cache=cache)
        mx.eval(logits)
        dt = time.time() - t_step
        decode_times.append(dt)

        # 贪心采样
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)

        if (step + 1) % 200 == 0:
            elapsed = sum(decode_times)
            tok_s = (step + 1) / elapsed if elapsed > 0 else float("inf")
            kv_now = cache[0].offset
            print(f"  step={step+1}/{MAX_STEPS} kv={kv_now} tok/s={tok_s:.1f}")

    # ── 结果 ──
    total_decode_time = sum(decode_times)
    avg_tok_s = MAX_STEPS / total_decode_time if total_decode_time > 0 else 0
    avg_ms = (total_decode_time / MAX_STEPS) * 1000

    print(f"\n{'='*60}")
    print(f"MLX recency results (RotatingKVCache, max_size={CACHE_SIZE}):")
    print(f"  Prefill: {prefill_time:.1f}s")
    print(f"  Decode:  {MAX_STEPS} tokens in {total_decode_time:.1f}s")
    print(f"  Speed:   {avg_tok_s:.1f} tok/s  ({avg_ms:.1f} ms/tok)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
