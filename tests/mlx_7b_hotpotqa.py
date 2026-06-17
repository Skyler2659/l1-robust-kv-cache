#!/usr/bin/env python3
"""MLX 7B HotpotQA 基准测试：4-bit 量化 + 多种驱逐策略对比。

方法: recency, l1_leverage, l2_leverage
预算: 128, 512
预计: ~15-20 分钟 (M4, 7B, 4-bit)
"""
import time, math, random
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import KVCache

# ── 配置 ──
MODEL = "Qwen/Qwen2.5-7B"
CACHE_BUDGETS = [128, 512]
SINK_SIZE = 4
RECENT_SIZE = 64
SKETCH_DIM = 1024
L1_UPDATE_INTERVAL = 32
N_SAMPLES = 5
SEED = 42

# ── L1 估算器 (随机方向采样) ──
class L1Estimator:
    def __init__(self, n_dirs=2000, seed=SEED):
        self.n_dirs = n_dirs
        self.rng = random.Random(seed)

    def scores(self, rows):
        n, d = rows.shape
        if n <= 1:
            return mx.sum(mx.abs(rows), axis=1)
        # 生成随机方向（每次重新生成以保证随机性）
        dirs = mx.random.normal(shape=(self.n_dirs, d))
        proj = rows @ dirs.T
        col_norms = mx.maximum(mx.sum(mx.abs(proj), axis=0, keepdims=True), 1e-8)
        ratios = mx.abs(proj) / col_norms
        return mx.max(ratios, axis=1)


# ── L2 估算器 (exact via SVD) ──
class L2Estimator:
    def scores(self, rows):
        n, d = rows.shape
        if n <= 1:
            return mx.sum(rows ** 2, axis=1)
        # SVD not supported on GPU, run on CPU
        with mx.stream(mx.cpu):
            u, s, vt = mx.linalg.svd(rows, compute_uv=True)
            # 有效秩
            eps = float(mx.finfo(rows.dtype).eps)
            tol = max(n, d) * eps * mx.max(s)
            rank = int(mx.sum(s > tol).item())
            if rank <= 0:
                return mx.zeros(n)
            scores = mx.sum(u[:, :rank] ** 2, axis=1)
        return scores


# ── KV cache 驱逐逻辑 ──
def evict_cache_l1(layer_kv, scores, seq_len, budget):
    """L1 选择: sink + recent + L1 top-k + last。"""
    if seq_len <= budget:
        return layer_kv
    sink = min(SINK_SIZE, budget - 1)
    max_rec = max(0, budget - sink - 1)
    recent = min(RECENT_SIZE, max_rec)
    l1_budget = max(0, budget - sink - recent - 1)

    parts = [mx.arange(sink)]
    if recent > 0:
        parts.append(mx.arange(seq_len - 1 - recent, seq_len - 1))
    if l1_budget > 0:
        cs, ce = sink, seq_len - 1 - recent
        if ce > cs:
            cand = scores[cs:ce]
            topk = min(l1_budget, cand.shape[0])
            top_idx = mx.argpartition(-cand, topk)[:topk]
            parts.append(top_idx + cs)
    parts.append(mx.array([seq_len - 1]))
    keep = mx.concatenate(parts)
    keep = mx.sort(keep)
    if keep.shape[0] > budget:
        keep = keep[:budget]
    return keep


def evict_cache_l2(layer_kv, scores, seq_len, budget):
    """L2 选择: sink + recent + L2 top-k + last。"""
    if seq_len <= budget:
        return layer_kv
    sink = min(SINK_SIZE, budget - 1)
    max_rec = max(0, budget - sink - 1)
    recent = min(RECENT_SIZE, max_rec)
    l2_budget = max(0, budget - sink - recent - 1)

    parts = [mx.arange(sink)]
    if recent > 0:
        parts.append(mx.arange(seq_len - 1 - recent, seq_len - 1))
    if l2_budget > 0:
        cs, ce = sink, seq_len - 1 - recent
        if ce > cs:
            cand = scores[cs:ce]
            topk = min(l2_budget, cand.shape[0])
            top_idx = mx.argpartition(-cand, topk)[:topk]
            parts.append(top_idx + cs)
    parts.append(mx.array([seq_len - 1]))
    keep = mx.concatenate(parts)
    keep = mx.sort(keep)
    if keep.shape[0] > budget:
        keep = keep[:budget]
    return keep


# ── HotpotQA 数据 ──
def load_hotpotqa(n_samples, tokenizer, max_context_words=2000, min_answer_tokens=5):
    """加载 HotpotQA 样本，筛选答案足够长的样本。"""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    samples = []
    rng = random.Random(SEED)
    # 打乱索引
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for idx in indices:
        row = ds[int(idx)]
        question = row["question"]
        answer = row["answer"]

        # 跳过太短的答案
        answer_tokens = tokenizer.encode(f" {answer}")
        if len(answer_tokens) < min_answer_tokens:
            continue

        # 构建上下文
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        passages = []
        for t, sents in zip(titles, sentences):
            text = " ".join(sents) if isinstance(sents, list) else str(sents)
            passages.append(f"[{t}] {text}")
        context = "\n\n".join(passages)

        # 限制长度
        words = context.split()
        if len(words) > max_context_words:
            context = " ".join(words[:max_context_words])

        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        full = f"{prompt} {answer}"

        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(full)

        if len(full_ids) > len(prompt_ids) and len(prompt_ids) > 500:
            answer_positions = list(range(len(prompt_ids), len(full_ids)))
            samples.append({
                "prompt_ids": prompt_ids,
                "full_ids": full_ids,
                "answer_positions": answer_positions,
                "answer": answer,
                "question": question,
            })
        if len(samples) >= n_samples:
            break
    return samples


# ── 主测试 ──
def run_method(model, tokenizer, samples, method_name, budget, num_layers):
    """运行一种驱逐策略，返回 [(sample_idx, ppl, tok_s, ...)]"""
    results = []

    for si, sample in enumerate(samples):
        prompt_ids = sample["prompt_ids"]
        answer_pos = sample["answer_positions"]

        # 每个样本重建 cache + 重置估算器
        l1_ests = {i: L1Estimator() for i in range(num_layers)}
        l2_ests = {i: L2Estimator() for i in range(num_layers)}
        fit_flags = {i: False for i in range(num_layers)}
        step_count = [0]
        cache = [KVCache() for _ in range(num_layers)]
        step_count[0] = 0

        # ── Prefill ──
        t0 = time.time()
        input_arr = mx.array([prompt_ids])
        logits = model(input_arr, cache=cache)
        mx.eval(logits)
        prefill_t = time.time() - t0

        # ── 初始驱逐（prefill 后） ──
        if method_name != "recency":
            _do_eviction(cache, method_name, budget, l1_ests, l2_ests, fit_flags, step_count)

        # ── Token-by-token decode ──
        next_token = mx.array([[prompt_ids[-1]]])
        losses = []
        decode_times = []

        max_steps = min(len(answer_pos), 200)
        for step in range(max_steps):
            # 目标 token
            target_pos = answer_pos[step]
            target_id = sample["full_ids"][target_pos]

            # Pre-evict: evict down to budget-1
            for c in cache:
                excess = c.offset + 1 - budget
                if excess > 0:
                    c.keys = c.keys[:, :, excess:, :]
                    c.values = c.values[:, :, excess:, :]
                    c.offset -= excess

            # 前向
            t_s = time.time()
            logits = model(next_token, cache=cache)
            mx.eval(logits)
            dt = time.time() - t_s

            # 计算 loss: -log(softmax(target_id))
            logit_vec = logits[:, -1, :] - mx.max(logits[:, -1, :], axis=-1, keepdims=True)
            log_soft = logit_vec - mx.log(mx.sum(mx.exp(logit_vec), axis=-1, keepdims=True))
            loss = -float(log_soft[0, target_id].item())
            losses.append(loss)

            # 采样下一个 token
            next_token = mx.array([[target_id]])

            # Post-evict
            step_count[0] += 1
            if method_name != "recency":
                _do_eviction(cache, method_name, budget, l1_ests, l2_ests, fit_flags, step_count)

            decode_times.append(dt)

            if (step + 1) % 100 == 0:
                elapsed = sum(decode_times)
                tok_s_now = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{method_name}] b={budget} s={si+1} step={step+1}/{max_steps} tok/s={tok_s_now:.1f}")

        if not losses:
            continue

        mean_nll = sum(losses) / len(losses)
        ppl = math.exp(mean_nll)
        total_t = sum(decode_times)
        tok_s = len(losses) / total_t if total_t > 0 else 0

        results.append({
            "method": method_name,
            "budget": budget,
            "sample": si,
            "ppl": ppl,
            "tok_s": tok_s,
            "decode_steps": len(losses),
            "prefill_s": prefill_t,
            "answer": sample["answer"],
            "question": sample["question"][:80],
        })

        print(f"  [{method_name}] b={budget} s={si+1}: ppl={ppl:.2f} tok/s={tok_s:.1f} steps={len(losses)}")

    return results


def _do_eviction(cache, method, budget, l1_ests, l2_ests, fit_flags, step_count):
    for li, c in enumerate(cache):
        seq_len = c.offset
        if seq_len <= budget:
            continue
        v = c.values
        rows = v[0].mean(axis=0).astype(mx.float32)

        force = (step_count[0] % L1_UPDATE_INTERVAL == 0) or not fit_flags[li]
        if force:
            fit_flags[li] = True

        if method == "l1_leverage":
            scores = l1_ests[li].scores(rows)
        elif method == "l2_leverage":
            scores = l2_ests[li].scores(rows)
        else:
            continue

        if method == "l1_leverage":
            keep = evict_cache_l1(c, scores, seq_len, budget)
        else:
            keep = evict_cache_l2(c, scores, seq_len, budget)

        c.keys = mx.take(c.keys, keep, axis=2)
        c.values = mx.take(c.values, keep, axis=2)
        c.offset = keep.shape[0]


# ── Main ──
def main():
    print(f"Loading {MODEL} ...")
    t0 = time.time()
    model, tokenizer = load(MODEL)
    num_layers = len(model.model.layers)
    print(f"Loaded {num_layers} layers in {time.time()-t0:.1f}s")

    # 4-bit 量化
    print("Applying 4-bit quantization ...")
    t0 = time.time()
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    print(f"Quantized in {time.time()-t0:.1f}s")

    # 加载数据
    print(f"\nLoading HotpotQA (up to {N_SAMPLES} samples) ...")
    samples = load_hotpotqa(N_SAMPLES, tokenizer, max_context_words=1200)
    print(f"Loaded {len(samples)} samples")
    for i, s in enumerate(samples):
        print(f"  sample {i+1}: {len(s['prompt_ids'])} prompt tokens, "
              f"{len(s['answer_positions'])} answer tokens | Q: {s['question'][:60]}...")

    methods = ["recency", "l1_leverage", "l2_leverage"]
    all_results = []

    print(f"\n{'='*70}")
    print(f"Running: {len(methods)} methods × {len(CACHE_BUDGETS)} budgets × {len(samples)} samples")
    print(f"Model: {MODEL} (4-bit) on MPS")
    print(f"{'='*70}\n")

    wall_start = time.time()

    for method in methods:
        for budget in CACHE_BUDGETS:
            t0 = time.time()
            res = run_method(model, tokenizer, samples, method, budget, num_layers)
            elapsed = time.time() - t0
            all_results.extend(res)
            print(f"  [{method}] b={budget} done in {elapsed:.0f}s\n")

    total_t = time.time() - wall_start

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"{'method':<15} {'budget':>7} {'ppl':>10} {'tok/s':>10}")
    print("-" * 45)
    for r in sorted(all_results, key=lambda x: (x["method"], x["budget"], x["sample"])):
        print(f"{r['method']:<15} {r['budget']:>7} {r['ppl']:>10.2f} {r['tok_s']:>10.1f}")
    print(f"{'='*70}")

    # 平均 PPL by method+budget
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in all_results:
        by_key[(r["method"], r["budget"])].append(r["ppl"])
    print(f"\n{'method':<15} {'budget':>7} {'avg_ppl':>10}")
    print("-" * 35)
    for k in sorted(by_key):
        avg = sum(by_key[k]) / len(by_key[k])
        print(f"{k[0]:<15} {k[1]:>7} {avg:>10.2f}")

    print(f"\nTotal time: {total_t:.0f}s ({total_t/60:.1f} min)")


if __name__ == "__main__":
    main()
