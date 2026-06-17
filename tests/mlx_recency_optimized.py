#!/usr/bin/env python3
"""MLX 全面加速 recency 测试：4-bit 量化 + RotatingKVCache。

对比 float16 基线 31 tok/s，4-bit 预期 50-80 tok/s。
"""
import time
import mlx.core as mx
import mlx.nn as nn

CACHE_SIZE = 256
PROMPT = (
    "This report describes ordinary background facts about libraries, weather, "
    "software releases, meeting notes, travel plans, and unrelated historical "
    "details. The cat sat on the mat. The dog ran in the yard. "
) * 30
MAX_STEPS = 512


def count_model_bits(model):
    """统计模型总 bit 数（含量化层）。"""
    from mlx.utils import tree_flatten
    flat = tree_flatten(model.parameters())
    total = sum(arr.size * arr.dtype.size * 8 for _, arr in flat)
    return total / 1e9  # GBits


def main():
    from mlx_lm import load
    from mlx_lm.models.cache import RotatingKVCache

    model_path = "Qwen/Qwen2.5-1.5B"
    print(f"Loading {model_path} ...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)
    print(f"Loaded {num_layers} layers in {time.time() - t0:.1f}s")
    print(f"Original dtype: {model.model.embed_tokens.weight.dtype}")
    print(f"Original model size: ~{count_model_bits(model):.1f} GBits = "
          f"~{count_model_bits(model)/8:.1f} GB")

    # ── 4-bit 量化 ──
    print("\n--- 4-bit weight quantization ---")
    t0 = time.time()
    nn.quantize(model, group_size=64, bits=4)
    print(f"Quantize call took {time.time() - t0:.1f}s")
    print(f"After quant size: ~{count_model_bits(model):.1f} GBits = "
          f"~{count_model_bits(model)/8:.1f} GB")

    # 确认量化层存在
    qt_count = sum(1 for _, m in model.named_modules() if 'Quantized' in type(m).__name__)
    qt_count += sum(1 for _, m in model.model.named_modules() if 'Quantized' in type(m).__name__)
    print(f"QuantizedLinear layers: {qt_count}")

    # ── RotatingKVCache（滑动窗口，比手动切快） ──
    cache = [RotatingKVCache(max_size=CACHE_SIZE, keep=0) for _ in range(num_layers)]

    # ── Prefill ──
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"\nPrompt: {len(prompt_ids)} tokens, budget={CACHE_SIZE}, steps={MAX_STEPS}")

    t0 = time.time()
    input_array = mx.array([prompt_ids])
    logits = model(input_array, cache=cache)
    mx.eval(logits)
    prefill_time = time.time() - t0
    print(f"Prefill: {prefill_time:.1f}s ({len(prompt_ids)/prefill_time:.0f} tok/s)")

    # ── Token-by-token decode ──
    next_token = mx.array([[prompt_ids[-1]]])
    decode_times = []

    print(f"\nDecoding ...")
    for step in range(MAX_STEPS):
        t_step = time.time()
        logits = model(next_token, cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_token)
        dt = time.time() - t_step
        decode_times.append(dt)

        if (step + 1) % 200 == 0:
            elapsed = sum(decode_times)
            tok_s_now = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  step={step+1}/{MAX_STEPS} tok/s={tok_s_now:.1f}")

    # ── 结果 ──
    total = sum(decode_times)
    avg_tok_s = MAX_STEPS / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"MLX Optimized recency:")
    print(f"  4-bit weights | RotatingKVCache({CACHE_SIZE})")
    print(f"  Speed:  {avg_tok_s:.1f} tok/s ({total/MAX_STEPS*1000:.1f} ms/tok)")
    print(f"  vs bfloat16 baseline (31.2 tok/s): {avg_tok_s/31.2:.1f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
