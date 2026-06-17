"""Reasoning with distractors — math reasoning in long noisy contexts.

Tests whether eviction policies preserve the critical numerical/reasoning
tokens embedded in a sea of irrelevant distractor text. Based on the idea
that L1 leverage scores should identify structurally important tokens
(numbers, operators, key entities) even when attention is diffuse.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import random
import torch

from src.benchmarks.base import BaseBenchmark, BenchmarkResult


# ── Synthetic math problems ────────────────────────────────────────────

def _generate_addition_problem(rng: random.Random) -> Dict[str, Any]:
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    answer = a + b
    return {
        "problem": f"A store has {a} apples. Then {b} more apples arrive. How many apples total?",
        "reasoning": f"{a} + {b} = {answer}",
        "answer": str(answer),
        "numbers": [a, b],
    }


def _generate_multiplication_problem(rng: random.Random) -> Dict[str, Any]:
    a = rng.randint(2, 50)
    b = rng.randint(2, 50)
    answer = a * b
    return {
        "problem": f"A factory produces {a} units per hour. How many units in {b} hours?",
        "reasoning": f"{a} × {b} = {answer}",
        "answer": str(answer),
        "numbers": [a, b],
    }


def _generate_two_step_problem(rng: random.Random) -> Dict[str, Any]:
    a = rng.randint(10, 100)
    b = rng.randint(2, 10)
    c = rng.randint(5, 50)
    answer = a * b + c
    return {
        "problem": (f"John buys {a} boxes. Each box costs ${b}."
                    f" He also pays ${c} for shipping. What is the total cost?"),
        "reasoning": f"{a} × {b} + {c} = {a*b} + {c} = {answer}",
        "answer": str(answer),
        "numbers": [a, b, c],
    }


PROBLEM_GENERATORS = [
    _generate_addition_problem,
    _generate_multiplication_problem,
    _generate_two_step_problem,
]

DISTRACTOR_TEMPLATES = [
    "The weather forecast for today predicts partly cloudy skies with a chance of rain in the afternoon.",
    "Recent studies have shown that regular exercise improves cognitive function and overall health.",
    "The local library announced extended hours for the summer reading program starting next month.",
    "Traffic on the main highway was heavier than usual due to construction near exit forty two.",
    "Scientists discovered a new species of butterfly in the tropical rainforest of South America.",
    "The city council voted to increase funding for public transportation improvements.",
    "A popular restaurant downtown received a five star rating from the national food guide.",
    "The school district plans to build a new elementary school on the east side of town.",
    "Environmental groups are advocating for stricter regulations on industrial water pollution.",
    "The technology conference attracted over three thousand attendees from around the world.",
    "Archaeologists unearthed ancient pottery fragments dating back to the bronze age.",
    "The stock market closed higher today with gains in technology and healthcare sectors.",
    "A new study suggests that drinking green tea may help reduce the risk of heart disease.",
    "The annual music festival will feature performances by both established and emerging artists.",
    "Researchers developed a new algorithm that improves the accuracy of weather predictions.",
]


def _build_reasoning_sample(
    rng: random.Random,
    tokenizer,
    n_distractors: int = 50,
    problem_generator=None,
) -> Dict[str, Any]:
    """Build a reasoning sample with the problem buried in distractors."""
    gen = problem_generator or rng.choice(PROBLEM_GENERATORS)
    problem = gen(rng)

    # Create distractor text
    distractors = []
    for _ in range(n_distractors):
        distractors.append(rng.choice(DISTRACTOR_TEMPLATES))

    # Place the problem at a random position
    pos = rng.randint(n_distractors // 4, 3 * n_distractors // 4)
    all_text_parts = distractors.copy()
    all_text_parts.insert(pos, f"\n\n[Problem] {problem['problem']}\n")

    context = " ".join(all_text_parts)
    question = f"\n\nBased on the problem above, solve: {problem['problem']}\nAnswer: {problem['reasoning']}"
    answer_prompt = f"\n\nBased on the problem above, solve: {problem['problem']}\nAnswer:"

    full_text = context + question
    prefix_text = context + answer_prompt

    ids = tokenizer(full_text, return_tensors="pt").input_ids
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
    prefix_len = prefix_ids.size(1)
    eval_positions = list(range(prefix_len, ids.size(1)))

    return {
        "input_ids": ids,
        "eval_positions": eval_positions,
        "metadata": {
            "problem": problem["problem"],
            "reasoning": problem["reasoning"],
            "answer": problem["answer"],
            "numbers": problem["numbers"],
            "needle_pos_fraction": pos / max(1, len(all_text_parts)),
            "n_distractors": n_distractors,
            "seq_len": ids.size(1),
        },
    }


class ReasoningWithDistractors(BaseBenchmark):
    """Math reasoning embedded in long noisy contexts.

    Tests whether eviction policies can retain the sparse numerical tokens
    needed for computation while discarding irrelevant distractor text.

    Args:
        n_samples: number of evaluation samples
        n_distractors: number of distractor sentences per sample
        seed: random seed
    """

    name = "reasoning"

    def __init__(
        self,
        n_samples: int = 50,
        n_distractors: int = 50,
        seed: int = 0,
        max_samples: Optional[int] = None,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.n_samples = n_samples
        self.n_distractors = n_distractors

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        samples: List[Dict[str, Any]] = []

        for si in range(self.n_samples):
            gen = PROBLEM_GENERATORS[si % len(PROBLEM_GENERATORS)]
            s = _build_reasoning_sample(
                rng, tokenizer,
                n_distractors=self.n_distractors,
                problem_generator=gen,
            )
            samples.append(s)

        return samples

    def compute_metrics(
        self, nlls: List[float], sample: Dict[str, Any], extra=None,
    ) -> Dict[str, float]:
        if not nlls:
            return {"ppl": float("inf")}
        mean_nll = sum(nlls) / len(nlls)
        return {
            "ppl": math.exp(mean_nll),
            "answer_ppl": math.exp(mean_nll),
            "n_eval_tokens": len(nlls),
        }
