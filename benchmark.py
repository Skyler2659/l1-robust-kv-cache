"""L1-Robust KV Cache benchmark — multi-strategy, multi-source comparison."""
import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_sources import (build_eval_text, build_hotpotqa_input_ids, build_long_text,
                          build_narrativeqa_input_ids, build_needle_eval_input_ids,
                          build_needle_std_input_ids, build_wikitext_eval_text)
from cache_baselines import SlidingWindowKVCache, get_kv_seq_len


# ── Pos-shift & model helpers ───────────────────────────────────────────

def load_module_from_file(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def maybe_enable_pos_shift(model, sketching_root: Path, enabled: bool = True):
    if not enabled:
        print("pos_shift: disabled by flag")
        return
    model_type = (getattr(model.config, "model_type", "") or "").lower()
    try:
        file_map = {"llama": "modify_llama.py", "gpt_neox": "modify_gpt_neox.py",
                     "qwen2": "modify_qwen2.py", "falcon": "modify_falcon.py"}
        func_map = {"llama": "enable_llama_pos_shift_attention",
                     "gpt_neox": "enable_gpt_neox_pos_shift_attention",
                     "qwen2": "enable_qwen2_pos_shift_attention",
                     "falcon": "enable_falcon_pos_shift_attention"}
        key = next((k for k in file_map if k in model_type), None)
        if key is None:
            print(f"pos_shift: skipped (model_type={model_type})")
            return
        mod_name = file_map[key].replace(".py", "")
        mod = importlib.import_module(f"l1_llm.pos_shift.{mod_name}")
        getattr(mod, func_map[key])(model)
        print(f"pos_shift: enabled for {key}")
    except Exception as exc:
        print(f"pos_shift: failed to enable ({exc}); continuing without patch")


def infer_kv_seq_dims(model_type: str):
    model_type = (model_type or "").lower()
    if "llama" in model_type or "gpt_neox" in model_type or "qwen2" in model_type:
        return 2, 2
    if "mpt" in model_type:
        return 3, 2
    if "falcon" in model_type:
        return 1, 1
    return 2, 2


def parse_recent_keep_grid(raw_value, default_value):
    if raw_value is None or raw_value.strip() == "":
        return [int(default_value)]
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--mixed_recent_keeps did not contain any integers.")
    return values


# ── Core eval loop ──────────────────────────────────────────────────────

@torch.no_grad()
def run_decode_eval(model, input_ids, cache_obj, label, k_seq_dim, max_steps,
                    progress_every=100, eval_target_positions=None):
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values, nlls, step_times, kv_lens = None, [], [], []

    total_steps = min(input_ids.size(1) - 1, max_steps)
    wall_start = time.perf_counter()
    print(f"[{label}] start: total_steps={total_steps}", flush=True)
    eval_set = set(eval_target_positions) if eval_target_positions is not None else None

    for idx in range(total_steps):
        token = input_ids[:, idx:idx + 1]
        target = input_ids[:, idx + 1:idx + 2].to(token.device).view(-1)

        if cache_obj is not None:
            past_key_values = cache_obj.evict_for_space(past_key_values, num_coming=1)

        t0 = time.perf_counter()
        outputs = model(input_ids=token, past_key_values=past_key_values, use_cache=True)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        nll = loss_fn(outputs.logits[:, -1, :].view(-1, model.config.vocab_size), target)
        if eval_set is None or (idx + 1) in eval_set:
            nlls.append(nll)

        past_key_values = outputs.past_key_values
        if cache_obj is not None:
            past_key_values = cache_obj(past_key_values)
        kv_lens.append(get_kv_seq_len(past_key_values, k_seq_dim))

        step_id = idx + 1
        if progress_every > 0 and (step_id % progress_every == 0 or step_id == total_steps):
            elapsed = time.perf_counter() - wall_start
            print(f"[{label}] step={step_id}/{total_steps} kv={kv_lens[-1] if kv_lens else 0}"
                  f" tok/s={step_id / elapsed if elapsed > 0 else float('inf'):.2f}"
                  f" elapsed={elapsed:.1f}s", flush=True)

    if len(nlls) == 0:
        hint = ""
        if eval_set is not None:
            last_target = max(eval_set)
            hint = (
                f" Eval targets at token position(s) {sorted(eval_set)};"
                f" need max_steps >= {last_target} (got total_steps={total_steps})."
            )
        raise ValueError("No target token selected for evaluation." + hint)
    mean_nll = torch.stack(nlls).mean().item()
    total_s = sum(step_times)
    return {"label": label, "steps": total_steps, "ppl": math.exp(mean_nll),
            "tok_per_s": total_steps / total_s if total_s > 0 else float("inf"),
            "avg_ms_per_tok": (total_s / total_steps) * 1000.0,
            "max_kv_len": max(kv_lens) if kv_lens else 0,
            "final_kv_len": kv_lens[-1] if kv_lens else 0}


# ── Output ──────────────────────────────────────────────────────────────

def print_table(results):
    hdr = ("mode", "steps", "ppl", "tok/s", "avg ms/tok", "max kv len", "final kv len")
    mode_width = max(len(hdr[0]), *(len(r["label"]) for r in results), 16)
    print("\n=== Comparison ===")
    print(f"{hdr[0]:<{mode_width}} {hdr[1]:>7} {hdr[2]:>10} {hdr[3]:>10}"
          f" {hdr[4]:>12} {hdr[5]:>12} {hdr[6]:>13}")
    for r in results:
        print(f"{r['label']:<{mode_width}} {r['steps']:>7d} {r['ppl']:>10.4f}"
              f" {r['tok_per_s']:>10.2f} {r['avg_ms_per_tok']:>12.3f}"
              f" {r['max_kv_len']:>12d} {r['final_kv_len']:>13d}")


# ── Strategy dispatch table (eliminates 150+ lines of repetition) ───────

def _build_strategies(args, plain_mod, main_mod, sketch_mod, h2o_mod, snap_mod, rocket_mod,
                      k_seq_dim, v_seq_dim):
    recent_size = max(1, args.cache_size - args.start_size)
    base = dict(k_seq_dim=k_seq_dim, v_seq_dim=v_seq_dim)

    def _l1(recent_keep, score_source="v"):
        return sketch_mod.L1RobustKVCache(
            cache_size=args.cache_size, num_sink_tokens=args.start_size,
            sketch_dim=args.sketch_dim, recompute_interval=args.recompute_interval,
            seed=args.seed, recent_keep=recent_keep, score_source=score_source, **base)

    h2o_recent_size = max(args.h2o_recent_size, args.mixed_recent_keep)
    rocket_recent_size = max(args.rocket_window_size, args.mixed_recent_keep)

    return {
        "plain":           plain_mod.PlainKVCache(),
        "main":            main_mod.StartRecentKVCache(
            start_size=args.start_size, recent_size=recent_size, **base),
        "sliding_window":  SlidingWindowKVCache(cache_size=args.cache_size, **base),
        "recency_only":    SlidingWindowKVCache(cache_size=args.cache_size, **base),
        "sketching":       _l1(args.l1_recent_keep, "v"),
        "kv_sketching":    _l1(args.l1_recent_keep, "kv"),
        "sink_l1_last":    _l1(args.l1_recent_keep, "v"),
        "sink_kv_l1_last": _l1(args.l1_recent_keep, "kv"),
        "l1_mixed":        _l1(args.mixed_recent_keep, "v"),
        "kv_l1_mixed":     _l1(args.mixed_recent_keep, "kv"),
        "sink_recent_l1_last": _l1(args.mixed_recent_keep, "v"),
        "sink_recent_kv_l1_last": _l1(args.mixed_recent_keep, "kv"),
        "h2o":              h2o_mod.H2OKVCache(cache_size=args.cache_size,
                                             recent_size=h2o_recent_size,
                                             sink_size=args.start_size, **base),
        "snapkv":           snap_mod.SnapKVCache(
            cache_size=args.cache_size,
            window_size=args.snapkv_window_size,
            kernel_size=args.snapkv_kernel_size,
            sink_size=args.snapkv_sink_size,
            **base),
        "rocketkv":         rocket_mod.RocketKVCache(
            cache_size=args.cache_size,
            window_size=args.rocket_window_size,
            kernel_size=args.rocket_kernel_size,
            sink_size=args.start_size,
            recent_size=rocket_recent_size,
            **base),
    }


COMPARISON_SPEC = {
    "full":   ["plain", "sketching", "kv_sketching", "main", "sliding_window"],
    "three":  ["recency_only", "sink_l1_last", "sink_kv_l1_last",
               "sink_recent_l1_last", "sink_recent_kv_l1_last"],
    "needle": ["recency_only", "main", "l1_mixed", "kv_l1_mixed", "h2o", "snapkv", "rocketkv"],
}

COMPARISON_HELP = {
    "full":   ["- plain: no eviction baseline; usually best ppl, growing KV.",
               "- main: start+recent heuristic from original StreamingLLM.",
               "- sliding_window: recent-only baseline without sink tokens.",
               "- sketching: V-only L1-robust policy.",
               "- kv_sketching: joint [K||V] L1-robust policy."],
    "three":  ["- recency_only: pure recent window baseline.",
               "- sink_l1_last: sink + V-only L1-selected history + last token.",
               "- sink_kv_l1_last: sink + joint [K||V] L1-selected history + last token.",
               "- sink_recent_l1_last: sink + recent + V-only L1-selected + last.",
               "- sink_recent_kv_l1_last: sink + recent + joint [K||V] L1-selected + last."],
    "needle": ["- recency_only: pure recent window baseline.",
               "- main: start+recent baseline.",
               "- l1_mixed: mixed strategy with V-only L1 scoring.",
               "- kv_l1_mixed: mixed strategy with joint [K||V] L1 scoring.",
               "- h2o: cumulative attention score (Zhang et al., NeurIPS 2023).",
               "- snapkv: observation-window attention selection (Li et al., arXiv 2024).",
               "- rocketkv: SnapKV + hybrid sparse attention (Behnam et al., ICML 2025).",
               "- In needle mode, ppl is computed only on answer tokens."],
    "grid":   ["- recency_only + main (once), then L1 mixed variants for each RK.",
               "- Specify RK list via --mixed_recent_keeps 32,48,64,80,96"],
}


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text_source", type=str, default="wikitext",
                        choices=["repeat", "wikitext", "needle", "needle_std", "long",
                                 "narrativeqa", "hotpotqa"])
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--wikitext_min_chars", type=int, default=12000)
    parser.add_argument("--wikitext_sample_limit", type=int, default=256)
    parser.add_argument("--long_target_words", type=int, default=2000)
    parser.add_argument("--text_repeat", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--sketch_dim", type=int, default=1024)
    parser.add_argument("--recompute_interval", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l1_recent_keep", type=int, default=0)
    parser.add_argument("--mixed_recent_keep", type=int, default=64)
    parser.add_argument("--h2o_recent_size", type=int, default=4,
                        help="Number of recent tokens H2O reserves from eviction")
    parser.add_argument("--snapkv_window_size", type=int, default=32,
                        help="Observation window size for the standalone SnapKV baseline")
    parser.add_argument("--snapkv_kernel_size", type=int, default=63,
                        help="Pooling kernel size for the standalone SnapKV baseline")
    parser.add_argument("--snapkv_sink_size", type=int, default=0,
                        help="Optional sink tokens reserved by the standalone SnapKV baseline")
    parser.add_argument("--rocket_window_size", type=int, default=32,
                        help="SnapKV observation window size for RocketKV stage 1")
    parser.add_argument("--rocket_kernel_size", type=int, default=63,
                        help="SnapKV pooling kernel size for RocketKV stage 1")
    parser.add_argument("--mixed_recent_keeps", type=str, default=None,
                        help="Comma-separated recent_keep values for grid mode")
    parser.add_argument("--grid_score_source", type=str, default="v",
                        choices=["v", "kv", "both"],
                        help="L1 score source for grid mode: v, kv, or both")
    parser.add_argument("--comparison_mode", type=str, default="full",
                        choices=["full", "three", "needle", "grid"])
    parser.add_argument("--needle_pos", type=int, default=400)
    parser.add_argument("--needle_depth", type=float, default=0.5,
                        help="Needle depth fraction for needle_std (0.0-1.0)")
    parser.add_argument("--needle_prefix_repeat", type=int, default=40)
    parser.add_argument("--qa_sample_idx", type=int, default=0)
    parser.add_argument("--qa_max_words", type=int, default=2000)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--enable_pos_shift", action="store_true", default=True)
    parser.add_argument("--disable_pos_shift", action="store_false", dest="enable_pos_shift")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    plain_mod = load_module_from_file("plain_kv", root / "plain_llm" / "kv_cache.py")
    main_mod  = load_module_from_file("main_kv",  root / "streaming_llm" / "kv_cache.py")
    sketch_mod = load_module_from_file("sketch_kv", root / "l1_llm" / "kv_cache.py")
    h2o_mod    = load_module_from_file("h2o_kv",   root / "h2o_llm" / "kv_cache.py")
    snap_mod   = load_module_from_file("snap_kv",  root / "snapkv" / "kv_cache.py")
    rocket_mod = load_module_from_file("rocket_kv", root / "rocketkv" / "kv_cache.py")

    print(f"Loading model: {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = {}
    if args.device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(args.device).eval()
    torch.manual_seed(args.seed)
    maybe_enable_pos_shift(model, sketching_root=root,
                           enabled=args.enable_pos_shift)
    k_seq_dim, v_seq_dim = infer_kv_seq_dims(model.config.model_type)

    # Probe cache format (informational)
    with torch.no_grad():
        probe_out = model(tokenizer("cache probe", return_tensors="pt").input_ids.to(args.device),
                          use_cache=True)
    fmt = "DynamicCache" if not isinstance(probe_out.past_key_values, tuple) else "legacy"
    print(f"KV cache format: {fmt}")

    # Build caches
    caches = _build_strategies(args, plain_mod, main_mod, sketch_mod, h2o_mod, snap_mod, rocket_mod,
                               k_seq_dim, v_seq_dim)

    # Load text
    eval_target_positions = None
    if args.text_source == "repeat":
        text = build_eval_text(args.text_repeat)
        print(f"text_source=repeat repeat={args.text_repeat}")
    elif args.text_source == "long":
        text = build_long_text(split=args.split, sample_idx=0,
                               target_words=args.long_target_words)
        print(f"text_source=long split={args.split} target_words={args.long_target_words}")
    elif args.text_source == "wikitext":
        text = build_wikitext_eval_text(args.dataset_name, args.task, args.split,
                                        args.wikitext_min_chars, args.wikitext_sample_limit)
        print(f"text_source=wikitext dataset={args.dataset_name}/{args.task}"
              f" split={args.split} min_chars={args.wikitext_min_chars}")
    elif args.text_source == "hotpotqa":
        input_ids, eval_target_positions = build_hotpotqa_input_ids(
            tokenizer, split=args.split, sample_idx=args.qa_sample_idx)
        print(f"text_source=hotpotqa split={args.split} sample={args.qa_sample_idx}")
        print(f"qa_eval_target_tokens={len(eval_target_positions)}"
              f" target_start={eval_target_positions[0] if eval_target_positions else 'NA'}")
        text = None
    elif args.text_source == "narrativeqa":
        input_ids, eval_target_positions = build_narrativeqa_input_ids(
            tokenizer, split=args.split, sample_idx=args.qa_sample_idx,
            max_words=args.qa_max_words)
        print(f"text_source=narrativeqa split={args.split} sample={args.qa_sample_idx}")
        print(f"qa_eval_target_tokens={len(eval_target_positions)}"
              f" target_start={eval_target_positions[0] if eval_target_positions else 'NA'}")
        text = None
    elif args.text_source == "needle_std":
        input_ids, eval_target_positions = build_needle_std_input_ids(
            tokenizer, needle_depth_pct=args.needle_depth)
        print(f"text_source=needle_std needle_depth={args.needle_depth}")
        print(f"needle_eval_target_tokens={len(eval_target_positions)}"
              f" target_start={eval_target_positions[0] if eval_target_positions else 'NA'}")
        text = None
    else:
        input_ids, eval_target_positions = build_needle_eval_input_ids(
            tokenizer, needle_pos=args.needle_pos, prefix_repeat=args.needle_prefix_repeat)
        print(f"text_source=needle needle_pos={args.needle_pos}"
              f" prefix_repeat={args.needle_prefix_repeat}")
        print(f"needle_eval_target_tokens={len(eval_target_positions)}"
              f" target_start={eval_target_positions[0] if eval_target_positions else 'NA'}")
        text = None

    if text is not None:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    else:
        input_ids = input_ids.to(args.device)

    if eval_target_positions:
        # Step idx predicts token idx+1; last target at max(eval_target_positions).
        min_steps = max(eval_target_positions)
        seq_cap = input_ids.size(1) - 1
        needed = min(min_steps, seq_cap)
        if args.max_steps < needed:
            print(f"max_steps: {args.max_steps} -> {needed} (to reach eval targets)", flush=True)
            args.max_steps = needed

    print(f"Tokenized length: {input_ids.size(1)} | max_steps: {args.max_steps}"
          f" | cache_size: {args.cache_size} | start_size: {args.start_size}")

    # Run strategies
    if args.comparison_mode == "grid":
        rk_list = parse_recent_keep_grid(args.mixed_recent_keeps, args.mixed_recent_keep)
        print(f"comparison_mode=grid: recency + main + L1 mixed × {rk_list}"
              f" score_source={args.grid_score_source}")
    elif args.comparison_mode == "three":
        print("comparison_mode=three: recency plus V-only and joint [K||V] L1 variants")
    elif args.comparison_mode == "needle":
        print("comparison_mode=needle: recency(sliding_window), main(start+recent),"
              " l1_mixed(V-only), kv_l1_mixed([K||V]), h2o, snapkv, rocketkv")

    results = []
    if args.comparison_mode == "grid":
        # Run recency + main ONCE, then l1_mixed for each RK — all in one table
        results.append(run_decode_eval(model, input_ids, caches["recency_only"],
                                       label="recency_only", k_seq_dim=k_seq_dim,
                                       max_steps=args.max_steps,
                                       progress_every=args.progress_every,
                                       eval_target_positions=eval_target_positions))
        results.append(run_decode_eval(model, input_ids, caches["main"],
                                       label="main", k_seq_dim=k_seq_dim,
                                       max_steps=args.max_steps,
                                       progress_every=args.progress_every,
                                       eval_target_positions=eval_target_positions))
        grid_sources = ["v", "kv"] if args.grid_score_source == "both" else [args.grid_score_source]
        for rk in rk_list:
            for score_source in grid_sources:
                rk_cache = sketch_mod.L1RobustKVCache(
                    cache_size=args.cache_size, num_sink_tokens=args.start_size,
                    sketch_dim=args.sketch_dim, recompute_interval=args.recompute_interval,
                    seed=args.seed, recent_keep=rk, score_source=score_source,
                    k_seq_dim=k_seq_dim, v_seq_dim=v_seq_dim)
                label = f"l1_rk{rk}" if score_source == "v" else f"kv_l1_rk{rk}"
                results.append(run_decode_eval(model, input_ids, rk_cache,
                                               label=label, k_seq_dim=k_seq_dim,
                                               max_steps=args.max_steps,
                                               progress_every=args.progress_every,
                                               eval_target_positions=eval_target_positions))
    else:
        for label in COMPARISON_SPEC[args.comparison_mode]:
            results.append(run_decode_eval(model, input_ids, caches[label], label=label,
                                           k_seq_dim=k_seq_dim, max_steps=args.max_steps,
                                           progress_every=args.progress_every,
                                           eval_target_positions=eval_target_positions))

    print_table(results)
    print("\n=== How to read ===")
    help_lines = COMPARISON_HELP.get(args.comparison_mode,
        [f"- Custom mode: {args.comparison_mode}"])
    for line in help_lines:
        print(line)


if __name__ == "__main__":
    main()
