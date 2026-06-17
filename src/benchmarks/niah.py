"""Needle-in-a-Haystack benchmarks with reliable evidence span metadata."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.benchmarks.base import BaseBenchmark


NEEDLE_TEMPLATES = [
    (
        "The secret passcode is {value}. Remember this passcode.",
        "What is the secret passcode? The passcode is",
        " {value}",
    ),
    (
        "The CEO of Acme Corp is {value}.",
        "Who is the CEO of Acme Corp? The CEO is",
        " {value}",
    ),
    (
        "The meeting is scheduled for {value}.",
        "When is the meeting scheduled? It is scheduled for",
        " {value}",
    ),
    (
        "The project deadline is {value}.",
        "What is the project deadline? The deadline is",
        " {value}",
    ),
    (
        "The access code to the vault is {value}.",
        "What is the access code to the vault? The code is",
        " {value}",
    ),
]

VALUES = [
    "ZEBRA-8842",
    "KILO-3391",
    "DELTA-7720",
    "FOXTROT-5519",
    "ALPHA-9904",
    "BRAVO-2217",
    "GAMMA-6635",
    "HOTEL-1108",
]

SYNTHETIC_HAYSTACK = (
    "This report describes ordinary background facts about libraries, weather, "
    "software releases, meeting notes, travel plans, and unrelated historical "
    "details. None of these filler sentences contain the target answer."
)


def _load_haystack_words(
    max_words: int = 8000,
    use_synthetic_haystack: bool = True,
    haystack_repeats: int = 120,
) -> List[str]:
    if use_synthetic_haystack:
        words = (SYNTHETIC_HAYSTACK + " ").split() * max(1, haystack_repeats)
        return words[:max_words]
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        words: List[str] = []
        for row in ds:
            text = (row.get("text", "") or "").strip()
            if not text:
                continue
            words.extend(text.split())
            if len(words) >= max_words + 2000:
                break
        return words[:max_words]
    except Exception:
        words = (SYNTHETIC_HAYSTACK + " ").split() * max(1, haystack_repeats)
        return words[:max_words]


def _token_span_from_char_span(
    tokenizer,
    text: str,
    char_start: int,
    char_end: int,
) -> Tuple[int, int, List[int]]:
    """Map a character span to token start/end using offsets when available."""
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            positions = [
                i
                for i, (start, end) in enumerate(offsets)
                if end > char_start and start < char_end and end > start
            ]
            if positions:
                return positions[0], positions[-1] + 1, positions
    except Exception:
        pass

    prefix_ids = tokenizer(
        text[:char_start], add_special_tokens=False, return_tensors="pt"
    ).input_ids
    span_ids = tokenizer(
        text[char_start:char_end], add_special_tokens=False, return_tensors="pt"
    ).input_ids
    token_start = int(prefix_ids.size(1))
    token_end = token_start + int(span_ids.size(1))
    return token_start, token_end, list(range(token_start, token_end))


def _build_needle_sample(
    tokenizer,
    haystack_words: List[str],
    needle_depth: float,
    needle_idx: int,
    max_words: int,
) -> Dict[str, Any]:
    template_idx = needle_idx % len(NEEDLE_TEMPLATES)
    needle_tmpl, question, answer_tmpl = NEEDLE_TEMPLATES[template_idx]
    value = VALUES[needle_idx % len(VALUES)]

    needle_text = f"\n\n{needle_tmpl.format(value=value)} Remember this.\n\n"
    question_text = f"\n{question}"
    answer_text = answer_tmpl.format(value=value)

    n_words = min(max_words, len(haystack_words))
    hay = haystack_words[:n_words]
    pos = max(5, min(max(5, n_words - 5), int(n_words * needle_depth)))
    prefix = " ".join(hay[:pos])
    suffix = " ".join(hay[pos:])
    prompt_text = prefix + needle_text + suffix + question_text
    full_text = prompt_text + answer_text

    needle_char_start = len(prefix)
    needle_char_end = needle_char_start + len(needle_text)
    answer_char_start = len(prompt_text)
    answer_char_end = answer_char_start + len(answer_text)

    ids = tokenizer(full_text, return_tensors="pt").input_ids
    needle_token_start, needle_token_end, evidence_positions = _token_span_from_char_span(
        tokenizer, full_text, needle_char_start, needle_char_end
    )
    answer_token_start, answer_token_end, answer_positions = _token_span_from_char_span(
        tokenizer, full_text, answer_char_start, answer_char_end
    )

    metadata = {
        "needle_depth": needle_depth,
        "needle_char_start": needle_char_start,
        "needle_char_end": needle_char_end,
        "answer_char_start": answer_char_start,
        "answer_char_end": answer_char_end,
        "needle_token_start": needle_token_start,
        "needle_token_end": needle_token_end,
        "answer_token_start": answer_token_start,
        "answer_token_end": answer_token_end,
        "needle_pos_tokens": needle_token_start,
        "seq_len": int(ids.size(1)),
        "answer": answer_text.strip(),
        "answer_text": answer_text,
        "needle_text": needle_text,
        "value": value,
        "template_idx": template_idx,
    }

    return {
        "input_ids": ids,
        "eval_positions": answer_positions,
        "answer_positions": answer_positions,
        "evidence_positions": evidence_positions,
        "ground_truth": answer_text.strip(),
        "prompt": prompt_text,
        "full_text": full_text,
        "needle_text": needle_text,
        "answer_text": answer_text,
        "metadata": metadata,
    }


class NIAHBenchmark(BaseBenchmark):
    """Single-needle retrieval at controlled depths."""

    name = "niah"

    def __init__(
        self,
        depths: Optional[List[float]] = None,
        max_words: int = 4000,
        needles_per_depth: int = 3,
        seed: int = 0,
        max_samples: Optional[int] = None,
        use_synthetic_haystack: bool = True,
        haystack_repeats: int = 120,
        context_length: Optional[int] = None,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.depths = depths or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.max_words = max_words
        self.needles_per_depth = needles_per_depth
        self.use_synthetic_haystack = use_synthetic_haystack
        self.haystack_repeats = haystack_repeats
        self.context_length = context_length
        self._haystack_words: Optional[List[str]] = None

    def _ensure_haystack(self):
        if self._haystack_words is None:
            self._haystack_words = _load_haystack_words(
                self.max_words + 2000,
                use_synthetic_haystack=self.use_synthetic_haystack,
                haystack_repeats=self.haystack_repeats,
            )

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        self._ensure_haystack()
        rng = random.Random(self.seed)
        samples: List[Dict[str, Any]] = []
        needle_idx = 0
        for depth in self.depths:
            for _ in range(self.needles_per_depth):
                permuted_hay = self._haystack_words.copy()
                rng.shuffle(permuted_hay)
                sample = _build_needle_sample(
                    tokenizer, permuted_hay, depth, needle_idx, self.max_words
                )
                sample["metadata"]["depth_bucket"] = depth
                samples.append(sample)
                needle_idx += 1
        return samples

    def compute_metrics(self, nlls: List[float], sample: Dict[str, Any], extra=None):
        if not nlls:
            return {"ppl": float("inf"), "answer_ppl": float("inf")}
        mean_nll = sum(nlls) / len(nlls)
        return {
            "ppl": math.exp(mean_nll),
            "answer_ppl": math.exp(mean_nll),
            "n_eval_tokens": len(nlls),
        }


class MultiNeedleNIAH(BaseBenchmark):
    """Multi-needle retrieval with scattered facts."""

    name = "multi_niah"

    def __init__(
        self,
        n_needles: int = 5,
        max_words: int = 4000,
        n_samples: int = 10,
        seed: int = 0,
        max_samples: Optional[int] = None,
        use_synthetic_haystack: bool = True,
        haystack_repeats: int = 120,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.n_needles = n_needles
        self.max_words = max_words
        self.n_samples = n_samples
        self.use_synthetic_haystack = use_synthetic_haystack
        self.haystack_repeats = haystack_repeats
        self._haystack_words: Optional[List[str]] = None

    def _ensure_haystack(self):
        if self._haystack_words is None:
            self._haystack_words = _load_haystack_words(
                self.max_words + 2000,
                use_synthetic_haystack=self.use_synthetic_haystack,
                haystack_repeats=self.haystack_repeats,
            )

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        self._ensure_haystack()
        rng = random.Random(self.seed)
        samples: List[Dict[str, Any]] = []

        for sample_idx in range(self.n_samples):
            hay = self._haystack_words.copy()
            rng.shuffle(hay)
            depths = sorted(rng.uniform(0.05, 0.95) for _ in range(self.n_needles))
            used_values = []
            needle_texts = []
            for needle_idx, _depth in enumerate(depths):
                tmpl_idx = (sample_idx * self.n_needles + needle_idx) % len(NEEDLE_TEMPLATES)
                needle_tmpl, _, _ = NEEDLE_TEMPLATES[tmpl_idx]
                value = VALUES[(sample_idx * self.n_needles + needle_idx) % len(VALUES)]
                used_values.append(value)
                needle_texts.append(needle_tmpl.format(value=value))

            n_words = min(self.max_words, len(hay))
            words = hay[:n_words]
            for depth, needle_text in sorted(zip(depths, needle_texts), reverse=True):
                pos = max(5, min(n_words - 5, int(n_words * depth)))
                words.insert(pos, f"\n\n{needle_text} Remember this.\n\n")

            context = " ".join(words)
            questions = []
            answers = []
            for needle_idx in range(self.n_needles):
                tmpl_idx = (sample_idx * self.n_needles + needle_idx) % len(NEEDLE_TEMPLATES)
                _, question, answer_tmpl = NEEDLE_TEMPLATES[tmpl_idx]
                questions.append(question)
                answers.append(answer_tmpl.format(value=used_values[needle_idx]).strip())

            question_block = "\n" + "\n".join(questions)
            answer_block = " " + " ".join(answers)
            full_text = context + question_block + answer_block
            prefix_text = context + question_block

            ids = tokenizer(full_text, return_tensors="pt").input_ids
            prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
            prefix_len = prefix_ids.size(1)
            eval_positions = list(range(prefix_len, ids.size(1)))

            samples.append(
                {
                    "input_ids": ids,
                    "eval_positions": eval_positions,
                    "answer_positions": eval_positions,
                    "ground_truth": answer_block.strip(),
                    "prompt": prefix_text,
                    "full_text": full_text,
                    "metadata": {
                        "n_needles": self.n_needles,
                        "depths": depths,
                        "values": used_values,
                        "answer": answer_block.strip(),
                        "seq_len": int(ids.size(1)),
                    },
                }
            )
        return samples

    def compute_metrics(self, nlls: List[float], sample: Dict[str, Any], extra=None):
        if not nlls:
            return {"ppl": float("inf")}
        mean_nll = sum(nlls) / len(nlls)
        return {"ppl": math.exp(mean_nll), "n_eval_tokens": len(nlls)}


class VariableDepthNIAH(BaseBenchmark):
    """Sweep needle depth continuously."""

    name = "variable_depth_niah"

    def __init__(
        self,
        n_depths: int = 20,
        max_words: int = 4000,
        seed: int = 0,
        max_samples: Optional[int] = None,
        use_synthetic_haystack: bool = True,
        haystack_repeats: int = 120,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.n_depths = n_depths
        self.max_words = max_words
        self.use_synthetic_haystack = use_synthetic_haystack
        self.haystack_repeats = haystack_repeats
        self._haystack_words: Optional[List[str]] = None

    def _ensure_haystack(self):
        if self._haystack_words is None:
            self._haystack_words = _load_haystack_words(
                self.max_words + 2000,
                use_synthetic_haystack=self.use_synthetic_haystack,
                haystack_repeats=self.haystack_repeats,
            )

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        self._ensure_haystack()
        rng = random.Random(self.seed)
        depths = [i / max(1, self.n_depths - 1) for i in range(self.n_depths)]
        samples: List[Dict[str, Any]] = []
        for depth_idx, depth in enumerate(depths):
            hay = self._haystack_words.copy()
            rng.shuffle(hay)
            sample = _build_needle_sample(tokenizer, hay, depth, depth_idx, self.max_words)
            sample["metadata"]["depth_bucket"] = round(depth, 3)
            samples.append(sample)
        return samples

    def compute_metrics(self, nlls: List[float], sample: Dict[str, Any], extra=None):
        if not nlls:
            return {"ppl": float("inf"), "depth": sample["metadata"]["needle_depth"]}
        mean_nll = sum(nlls) / len(nlls)
        return {
            "ppl": math.exp(mean_nll),
            "depth": sample["metadata"]["needle_depth"],
            "n_eval_tokens": len(nlls),
        }
