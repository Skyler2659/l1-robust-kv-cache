"""RULER benchmark wrapper — long-context tasks from NVIDIA.

RULER (Hsieh et al., 2024) tests 4 ability categories:
  1. Retrieval (needle, multi-needle, variable tracking)
  2. Multi-hop (variable tracking with dependencies)
  3. Aggregation (common words, frequent words)
  4. QA (SQuAD, HotpotQA in long context)

This module generates synthetic RULER-style samples when the official
dataset is unavailable, or wraps the official HF dataset when present.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import random
import string
import torch

from src.benchmarks.base import BaseBenchmark, BenchmarkResult
from src.benchmarks.niah import _token_span_from_char_span
from src.evaluation.official_metrics import parse_references, ruler_metric_name, ruler_task_family


def _random_word(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _generate_variable_tracking_sample(
    rng: random.Random, seq_words: int = 2000, n_variables: int = 10,
) -> Dict[str, Any]:
    """Generate variable tracking: 'What is the value of VAR_X?'"""
    variables = {f"VAR_{i}": _random_word(rng, 8) for i in range(n_variables)}
    filler = [_random_word(rng, 5) for _ in range(seq_words)]

    # Scatter variable assignments through the text
    positions = sorted(rng.sample(range(100, seq_words - 100), n_variables))
    for pos, (var, val) in zip(positions, variables.items()):
        filler.insert(pos, f"[{var} = {val}]")

    # Query at the end
    query_var = rng.choice(list(variables.keys()))
    query = f"\nWhat is the value of {query_var}? The value is"
    answer = f" {variables[query_var]}"
    prompt = " ".join(filler) + query
    text = prompt + answer
    return {
        "text": text,
        "prompt": prompt,
        "answer_text": answer,
        "answer": answer.strip(),
        "query_var": query_var,
        "n_variables": n_variables,
    }


def _generate_common_words_sample(
    rng: random.Random, seq_words: int = 2000, n_common: int = 5,
) -> Dict[str, Any]:
    """Generate aggregation task: list words appearing ≥ K times."""
    pool = [_random_word(rng, 5) for _ in range(50)]
    common_words = rng.sample(pool, min(n_common, len(pool)))
    threshold = max(3, seq_words // 100)

    words = []
    for _ in range(seq_words):
        if rng.random() < 0.05 and common_words:
            words.append(rng.choice(common_words))
        else:
            words.append(rng.choice(pool))

    query = (f"\nList all words that appear at least {threshold} times in the text above."
             f" The words are")
    # Compute actual common words
    from collections import Counter
    counts = Counter(words)
    actual_common = sorted(w for w, c in counts.items() if c >= threshold)
    answer = " " + ", ".join(actual_common)

    prompt = " ".join(words) + query
    text = prompt + answer
    return {
        "text": text,
        "prompt": prompt,
        "answer_text": answer,
        "answer": answer.strip(),
        "threshold": threshold,
        "expected_words": actual_common,
    }


def _generate_multi_hop_sample(
    rng: random.Random, seq_words: int = 2000, n_hops: int = 3,
) -> Dict[str, Any]:
    """Generate multi-hop variable tracking: A→B→C chain."""
    entities = [_random_word(rng, 6) for _ in range(n_hops + 1)]
    chain = {}
    for i in range(n_hops):
        chain[entities[i]] = entities[i + 1]

    filler = [_random_word(rng, 5) for _ in range(seq_words)]
    positions = sorted(rng.sample(range(100, seq_words - 100), n_hops))
    clues = []
    for pos, (src, dst) in zip(positions, chain.items()):
        clue = f"[{src} points to {dst}]"
        filler.insert(pos, clue)
        clues.append(clue)

    query = f"\nWhat does {entities[0]} ultimately point to? It points to"
    answer = f" {entities[-1]}"

    prompt = " ".join(filler) + query
    text = prompt + answer
    return {
        "text": text,
        "prompt": prompt,
        "answer_text": answer,
        "answer": answer.strip(),
        "chain": chain,
        "n_hops": n_hops,
        "start": entities[0],
        "end": entities[-1],
    }


def _generate_niah_single_sample(
    rng: random.Random, seq_words: int = 2000,
) -> Dict[str, Any]:
    """Generate a RULER-style single needle retrieval sample."""
    key = f"{_random_word(rng, 6)}-{_random_word(rng, 5)}"
    value = "".join(rng.choice(string.digits) for _ in range(7))
    filler = ["The grass is green. The sky is blue. The sun is yellow."] * max(1, seq_words // 10)
    insert_pos = rng.randrange(0, len(filler))
    needle = f"The special magic number for {key} is {value}."
    filler.insert(insert_pos, needle)
    context = " ".join(filler)
    question = f"What is the special magic number for {key} mentioned in the provided text? "
    answer_prefix = f"The special magic number for {key} mentioned in the provided text is"
    prompt = f"{context}\n\n{question}{answer_prefix}"
    answer_text = f" {value}"
    return {
        "text": prompt + answer_text,
        "prompt": prompt,
        "answer_text": answer_text,
        "answer": value,
        "answers": [value],
        "key": key,
        "needle": needle,
        "dataset_official": False,
    }


RULER_TASK_GENERATORS = {
    "niah_single": _generate_niah_single_sample,
    "niah_single_1": _generate_niah_single_sample,
    "variable_tracking": _generate_variable_tracking_sample,
    "vt": _generate_variable_tracking_sample,
    "common_words": _generate_common_words_sample,
    "multi_hop": _generate_multi_hop_sample,
}

RULER_OFFICIAL_TASK_ALIASES = {
    "niah_single": "niah_single_1",
    "niah": "niah_single_1",
    "variable_tracking": "vt",
}


class RULERBenchmark(BaseBenchmark):
    """RULER-style synthetic long-context benchmark.

    Supports: variable_tracking, common_words, multi_hop.
    Generates samples when official dataset is unavailable.
    """

    name = "ruler"

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        n_samples_per_task: int = 20,
        seq_words: int = 2000,
        seed: int = 0,
        max_samples: Optional[int] = None,
        use_official_dataset: bool = False,
        require_official_dataset: bool = False,
        hf_dataset_name: Optional[str] = None,
        hf_dataset_config: Optional[str] = None,
        hf_split: Optional[str] = None,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.tasks = tasks or ["variable_tracking", "common_words", "multi_hop"]
        self.n_samples_per_task = n_samples_per_task
        self.seq_words = seq_words
        self.use_official_dataset = use_official_dataset
        self.require_official_dataset = require_official_dataset
        self.hf_dataset_name = hf_dataset_name or "xAlg-AI/att-hub-ruler-16k"
        self.hf_dataset_config = hf_dataset_config
        self.hf_split = hf_split

    def _official_task_name(self, task_name: str) -> str:
        return RULER_OFFICIAL_TASK_ALIASES.get(task_name, task_name)

    def _load_official_rows(self, task_name: str) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        hf_task = self.hf_dataset_config or self._official_task_name(task_name)
        split_name = self.hf_split or hf_task
        if self.n_samples_per_task:
            split_name = f"{split_name}[:{self.n_samples_per_task}]"
        ds = load_dataset(
            self.hf_dataset_name,
            hf_task,
            split=split_name,
            trust_remote_code=True,
        )
        return [dict(row) for row in ds]

    def _official_row_to_sample(self, tokenizer, row: Dict[str, Any], task_name: str) -> Dict[str, Any]:
        context = row.get("context", "") or ""
        question = row.get("question", "") or ""
        answer_prefix = row.get("answer_prefix", "") or ""
        answers = parse_references(row.get("answer"))
        prompt = f"{context}\n\n{question}{answer_prefix}"
        if prompt and not prompt.endswith((" ", "\n")):
            prompt += " "
        answer_text = " ".join(answers) if len(answers) > 1 else (answers[0] if answers else "")
        answer_text = f" {answer_text}".rstrip() if answer_text else " unknown"
        text = prompt + answer_text

        ids = tokenizer(text, return_tensors="pt").input_ids
        answer_char_start = len(prompt)
        answer_char_end = answer_char_start + len(answer_text)
        answer_token_start, answer_token_end, eval_positions = _token_span_from_char_span(
            tokenizer, text, answer_char_start, answer_char_end
        )
        official_task = row.get("task") or self._official_task_name(task_name)
        return {
            "input_ids": ids,
            "eval_positions": eval_positions,
            "answer_positions": eval_positions,
            "ground_truth": answers[0] if answers else answer_text.strip(),
            "answers": answers,
            "outputs": answers,
            "prompt": prompt,
            "full_text": text,
            "answer_text": answer_text,
            "metadata": {
                "task": official_task,
                "requested_task": task_name,
                "answer": answers[0] if answers else answer_text.strip(),
                "answers": answers,
                "outputs": answers,
                "answer_char_start": answer_char_start,
                "answer_char_end": answer_char_end,
                "answer_token_start": answer_token_start,
                "answer_token_end": answer_token_end,
                "seq_len": ids.size(1),
                "dataset_official": True,
                "official_dataset_name": self.hf_dataset_name,
                "official_dataset_config": self.hf_dataset_config or self._official_task_name(task_name),
                "official_metric_name": ruler_metric_name(official_task),
                "official_task_family": ruler_task_family(official_task),
                "official_max_new_tokens": row.get("max_new_tokens"),
            },
        }

    def _prepare_official_samples(self, tokenizer) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for task_name in self.tasks:
            rows = self._load_official_rows(task_name)
            for row in rows:
                samples.append(self._official_row_to_sample(tokenizer, row, task_name))
        return samples

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        if self.use_official_dataset:
            try:
                samples = self._prepare_official_samples(tokenizer)
                if samples:
                    return samples
            except Exception as exc:
                if self.require_official_dataset:
                    raise
                print(f"[RULER] failed to load official dataset; using synthetic fallback: {exc}")

        rng = random.Random(self.seed)
        samples: List[Dict[str, Any]] = []

        for task_name in self.tasks:
            gen_fn = RULER_TASK_GENERATORS.get(task_name)
            if gen_fn is None:
                print(f"[RULER] unknown task: {task_name}, skipping")
                continue

            for si in range(self.n_samples_per_task):
                raw = gen_fn(rng, seq_words=self.seq_words)
                text = raw["text"]
                prompt = raw["prompt"]
                answer_text = raw["answer_text"]
                answer = raw["answer"]

                ids = tokenizer(text, return_tensors="pt").input_ids
                answer_char_start = len(prompt)
                answer_char_end = answer_char_start + len(answer_text)
                answer_token_start, answer_token_end, eval_positions = _token_span_from_char_span(
                    tokenizer, text, answer_char_start, answer_char_end
                )

                samples.append({
                    "input_ids": ids,
                    "eval_positions": eval_positions,
                    "answer_positions": eval_positions,
                    "ground_truth": answer,
                    "answers": raw.get("answers", [answer]),
                    "outputs": raw.get("answers", [answer]),
                    "prompt": prompt,
                    "full_text": text,
                    "answer_text": answer_text,
                    "metadata": {
                        "task": task_name,
                        "answer": answer,
                        "answers": raw.get("answers", [answer]),
                        "outputs": raw.get("answers", [answer]),
                        "answer_text": answer_text,
                        "answer_char_start": answer_char_start,
                        "answer_char_end": answer_char_end,
                        "answer_token_start": answer_token_start,
                        "answer_token_end": answer_token_end,
                        "seq_len": ids.size(1),
                        "dataset_official": raw.get("dataset_official", False),
                        "official_metric_name": ruler_metric_name(task_name),
                        "official_task_family": ruler_task_family(task_name),
                        **{
                            k: v
                            for k, v in raw.items()
                            if k not in ("text", "prompt", "answer_text", "answer")
                        },
                    },
                })

        return samples

    def compute_metrics(
        self, nlls: List[float], sample: Dict[str, Any], extra=None,
    ) -> Dict[str, float]:
        if not nlls:
            return {"ppl": float("inf"), "task": sample["metadata"]["task"]}
        mean_nll = sum(nlls) / len(nlls)
        return {
            "ppl": math.exp(mean_nll),
            "answer_ppl": math.exp(mean_nll),
            "task": sample["metadata"]["task"],
            "n_eval_tokens": len(nlls),
        }
