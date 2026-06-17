#!/usr/bin/env python3
"""MLX NIAH 基准测试：受控针深度 × 多驱逐策略 × 多预算。

4-bit Qwen2.5-7B，预计 ~15 分钟。
"""
import time, math, random
import mlx.core as mx
import mlx.nn as nn
from collections import defaultdict

# ── 配置 ──
MODEL = "Qwen/Qwen2.5-7B"
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]   # needle 深度
NEEDLES_PER_DEPTH = 2                    # 每深度样本数
CACHE_BUDGETS = [64, 128, 256]          # KV cache 预算
SINK_SIZE = 4
RECENT_SIZE = 32
UPDATE_INTERVAL = 32                     # L1/L2 重算间隔
MAX_HAYSTACK_WORDS = 800                # 草堆词数 (~1200 tokens)
MAX_DECODE_STEPS = 50
SEED = 42

# Needle 模板
NEEDLE_TMPL = (
    "The secret passcode is {value}. Remember this passcode."
)
QUESTION = "\nWhat is the secret passcode? The passcode is"
VALUES = ["ZEBRA-8842", "KILO-3391", "DELTA-7720", "FOXTROT-5519",
          "ALPHA-9904", "BRAVO-2217", "GAMMA-6635", "HOTEL-1108"]

# 合成草堆
HAYSTACK = (
    "This report describes ordinary background facts about libraries, weather "
    "patterns, software release cycles, meeting minutes, travel itineraries, "
    "historical summaries, economic indicators, agricultural reports, census "
    "data, infrastructure maintenance logs, and unrelated administrative notes. "
)

random.seed(SEED)
mx.random.seed(SEED)


# ── L1 估算器 (Woodruff-style: Exp(1) reweight + QR) ──
class WoodruffL1:
    def __init__(self, sketch_dim=1024, seed=SEED):
        self.sketch_dim = sketch_dim
        self.rng = random.Random(seed + 7)
        self.r_inv = None
        self.last_dim = None

    def _exp_weights(self, n, device_dtype):
        """Exp(1) 权重，确保 float32 安全。"""
        u = mx.random.uniform(shape=(n, 1))
        u = mx.clip(u, 1e-8, 1 - 1e-8)
        w = -mx.log(1.0 - u)
        return mx.maximum(w, 1e-8)

    def scores(self, rows):
        n, d = rows.shape
        if n <= 1:
            return mx.sum(mx.abs(rows), axis=1)
        rows_f = rows.astype(mx.float32)

        # 是否需要重新拟合 R^{-1}
        force = self.r_inv is None or self.last_dim != d
        if force:
            if n < self.sketch_dim:
                # 小矩阵：直接用 Exp(1) 加权 + QR
                w = self._exp_weights(n, rows_f.dtype)
                weighted = rows_f / w
            else:
                # 大矩阵：CountSketch 降维后再 QR
                # hash: random bucket for each row [0..sketch_dim)
                buckets = mx.random.randint(0, self.sketch_dim, shape=(n,))
                signs = (mx.random.randint(0, 2, shape=(n,)).astype(mx.int32) * 2 - 1).astype(mx.float32)
                # Exp(1) reweight
                w = self._exp_weights(n, rows_f.dtype)
                weighted = rows_f / w * signs.reshape(-1, 1)
                # Scatter via one-hot matrix multiply
                idx = mx.arange(self.sketch_dim).reshape(-1, 1)
                mask = (buckets.reshape(1, -1) == idx).astype(mx.float32)  # [sketch_dim, n]
                weighted = mask @ weighted  # [sketch_dim, d]

            # QR on CPU (not GPU supported)
            with mx.stream(mx.cpu):
                q, r = mx.linalg.qr(weighted)
                if r.shape[0] != r.shape[1]:
                    self.r_inv = None
                    self.last_dim = d
                    return mx.sum(mx.abs(rows_f), axis=1)
                jit = max(1e-4, float(mx.max(mx.abs(mx.diag(r))).item()) * 1e-6)
                r_reg = r + mx.eye(r.shape[0]) * jit
                self.r_inv = mx.linalg.inv(r_reg)
            self.last_dim = d

        # 计算分数: ||rows @ R^{-1}||_1
        proj = rows_f @ self.r_inv
        return mx.sum(mx.abs(proj), axis=1)


# ── L2 估算器 ──
class ExactL2:
    def scores(self, rows):
        n, d = rows.shape
        if n <= 1:
            return mx.sum(rows ** 2, axis=1)
        rows_f = rows.astype(mx.float32)
        with mx.stream(mx.cpu):
            u, s, vt = mx.linalg.svd(rows_f, compute_uv=True)
            eps = float(mx.finfo(rows_f.dtype).eps)
            tol = max(n, d) * eps * mx.max(s)
            rank = int(mx.sum(s > tol).item())
            if rank <= 0:
                return mx.zeros(n)
            scores = mx.sum(u[:, :rank] ** 2, axis=1)
        return scores


# ── NIAH 样本生成 ──
def build_niah_sample(tokenizer, haystack_words, depth, needle_idx):
    value = VALUES[needle_idx % len(VALUES)]
    needle_text = f"\n\n{NEEDLE_TMPL.format(value=value)}\n\n"
    question = QUESTION
    answer = f" {value}"

    n_words = len(haystack_words)
    pos = max(5, min(n_words - 5, int(n_words * depth)))
    prefix = " ".join(haystack_words[:pos])
    suffix = " ".join(haystack_words[pos:])
    prompt_text = prefix + needle_text + suffix + question
    full_text = prompt_text + answer

    prompt_ids = tokenizer.encode(prompt_text)
    full_ids = tokenizer.encode(full_text)
    answer_positions = list(range(len(prompt_ids), len(full_ids)))

    return {
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "answer_positions": answer_positions,
        "depth": depth,
        "value": value,
        "needle_pos_tokens": len(tokenizer.encode(prefix)),
    }


# ── 驱逐逻辑 ──
def select_and_evict(cache, scores, seq_len, budget):
    if seq_len <= budget:
        return
    sink = min(SINK_SIZE, budget - 1)
    max_rec = max(0, budget - sink - 1)
    recent = min(RECENT_SIZE, max_rec)
    mid_budget = max(0, budget - sink - recent - 1)

    parts = [mx.arange(sink)]
    if recent > 0:
        parts.append(mx.arange(seq_len - 1 - recent, seq_len - 1))
    if mid_budget > 0:
        cs, ce = sink, seq_len - 1 - recent
        if ce > cs:
            cand = scores[cs:ce]
            topk = min(mid_budget, cand.shape[0])
            top_idx = mx.argpartition(-cand, topk)[:topk]
            parts.append(top_idx + cs)
    parts.append(mx.array([seq_len - 1]))
    keep = mx.concatenate(parts)
    keep = mx.sort(keep)
    if keep.shape[0] > budget:
        keep = keep[:budget]

    for c in cache:
        c.keys = mx.take(c.keys, keep, axis=2)
        c.values = mx.take(c.values, keep, axis=2)
        c.offset = keep.shape[0]


# ── 主测试循环 ──
def run_benchmark(model, tokenizer, samples, method, budget):
    num_layers = len(model.model.layers)
    results = []

    for si, sample in enumerate(samples):
        from mlx_lm.models.cache import KVCache
        cache = [KVCache() for _ in range(num_layers)]
        prompt_ids = sample["prompt_ids"]
        answer_pos = sample["answer_positions"]
        step_count = [0]
        fit_flags = {i: False for i in range(num_layers)}
        # Reset estimators per sample
        l1_ests = {i: WoodruffL1() for i in range(num_layers)}
        l2_ests = {i: ExactL2() for i in range(num_layers)}

        # Prefill
        logits = model(mx.array([prompt_ids]), cache=cache)
        mx.eval(logits)

        # 初始驱逐
        if method != "recency":
            _do_evict(cache, method, budget, l1_ests, l2_ests, fit_flags, step_count)

        # Decode
        next_token = mx.array([[prompt_ids[-1]]])
        losses = []

        for step in range(min(len(answer_pos), MAX_DECODE_STEPS)):
            target_id = sample["full_ids"][answer_pos[step]]

            # Pre-evict: evict down to budget-1 to make room
            for c in cache:
                excess = c.offset + 1 - budget
                if excess > 0:
                    c.keys = c.keys[:, :, excess:, :]
                    c.values = c.values[:, :, excess:, :]
                    c.offset -= excess

            logits = model(next_token, cache=cache)
            mx.eval(logits)

            # Loss
            logit_vec = logits[:, -1, :] - mx.max(logits[:, -1, :], axis=-1, keepdims=True)
            log_soft = logit_vec - mx.log(mx.sum(mx.exp(logit_vec), axis=-1, keepdims=True))
            losses.append(-float(log_soft[0, target_id].item()))

            next_token = mx.array([[target_id]])
            step_count[0] += 1

            if method != "recency":
                _do_evict(cache, method, budget, l1_ests, l2_ests, fit_flags, step_count)

        if losses:
            ppl = math.exp(sum(losses) / len(losses))
            results.append({
                "method": method, "budget": budget, "sample": si,
                "depth": sample["depth"], "ppl": ppl,
                "steps": len(losses),
                "needle_pos": sample["needle_pos_tokens"],
            })
    return results


def _do_evict(cache, method, budget, l1_ests, l2_ests, fit_flags, step_count):
    force_refit = (step_count[0] % UPDATE_INTERVAL == 0)
    for li, c in enumerate(cache):
        seq_len = c.offset
        if seq_len <= budget:
            continue
        rows = c.values[0].mean(axis=0).astype(mx.float32)

        if method == "l1_leverage":
            if force_refit or not fit_flags.get(li, False):
                l1_ests[li].r_inv = None  # force refit
                fit_flags[li] = True
            scores = l1_ests[li].scores(rows)
        elif method == "l2_leverage":
            scores = l2_ests[li].scores(rows)
        else:
            continue

        select_and_evict([c], scores, seq_len, budget)


# ── Main ──
def main():
    from mlx_lm import load

    print(f"Loading {MODEL} ...")
    t0 = time.time()
    model, tokenizer = load(MODEL)
    num_layers = len(model.model.layers)
    print(f"{num_layers} layers in {time.time()-t0:.0f}s")

    print("4-bit quantization ...")
    t0 = time.time()
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    print(f"Done in {time.time()-t0:.0f}s")

    # 生成样本
    haystack_words = (HAYSTACK * 60).split()[:MAX_HAYSTACK_WORDS + 500]
    samples = []
    for depth in DEPTHS:
        for ni in range(NEEDLES_PER_DEPTH):
            hw = haystack_words.copy()
            random.shuffle(hw)
            samples.append(build_niah_sample(tokenizer, hw[:MAX_HAYSTACK_WORDS],
                                             depth, len(samples)))

    print(f"\n{len(samples)} NIAH samples:")
    for s in samples:
        print(f"  depth={s['depth']:.2f}  needle@token={s['needle_pos_tokens']}  "
              f"prompt={len(s['prompt_ids'])}t  answer={len(s['answer_positions'])}t")

    methods = ["recency", "l1_leverage", "l2_leverage"]
    all_results = []

    print(f"\n{'='*65}")
    print(f"Running: {len(methods)} methods × {len(CACHE_BUDGETS)} budgets × {len(samples)} samples")
    print(f"{'='*65}\n")

    wall_start = time.time()
    for method in methods:
        for budget in CACHE_BUDGETS:
            t0 = time.time()
            res = run_benchmark(model, tokenizer, samples, method, budget)
            all_results.extend(res)
            if res:
                ppls = [r["ppl"] for r in res]
                print(f"  [{method}] b={budget}: ppl={min(ppls):.1f}~{max(ppls):.1f} "
                      f"(avg {sum(ppls)/len(ppls):.1f}) in {time.time()-t0:.0f}s")
            else:
                print(f"  [{method}] b={budget}: no results")

    total_t = time.time() - wall_start

    # ── 汇总表 ──
    print(f"\n{'='*70}")
    print(f"{'method':<15} {'budget':>7} {'depth':>7} {'ppl':>10}")
    print("-" * 42)
    for r in sorted(all_results, key=lambda x: (x["method"], x["budget"], x["depth"])):
        print(f"{r['method']:<15} {r['budget']:>7} {r['depth']:>7.2f} {r['ppl']:>10.2f}")

    # 按 depth 分组平均
    print(f"\n{'='*70}")
    print(f"Avg PPL by depth:")
    print(f"{'depth':>7} {'recency':>12} {'l1_leverage':>12} {'l2_leverage':>12}")
    print("-" * 50)
    for budget in CACHE_BUDGETS:
        print(f"-- budget={budget} --")
        for depth in DEPTHS:
            row = []
            for method in methods:
                vals = [r["ppl"] for r in all_results
                        if r["method"] == method and r["budget"] == budget
                        and abs(r["depth"] - depth) < 0.01]
                avg = sum(vals)/len(vals) if vals else float("nan")
                row.append(avg)
            print(f"  {depth:>5.2f}  {row[0]:>12.2f} {row[1]:>12.2f} {row[2]:>12.2f}")

    print(f"\nTotal: {total_t:.0f}s ({total_t/60:.1f} min)")


if __name__ == "__main__":
    main()
