"""Multi-hop QA benchmarks — HotpotQA, 2WikiMultihopQA, MuSiQue.

These tasks require reasoning over multiple evidence passages scattered
through a long context. They are the critical test for whether L1 leverage
scores can identify geometrically irreplaceable evidence tokens that pure
attention-based methods might miss.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import random
import torch

from src.benchmarks.base import BaseBenchmark, BenchmarkResult


DATASET_MAP = {
    "hotpotqa": ("hotpot_qa", "distractor"),
    "2wikimultihopqa": ("2wikimultihopqa", "default"),
    "musique": ("musique", "default"),
}


def _load_multihop_dataset(name: str, split: str = "validation", max_samples: int = 100):
    """Load multi-hop QA dataset from HF."""
    try:
        from datasets import load_dataset
        if name == "hotpotqa":
            ds = load_dataset("hotpot_qa", "distractor", split=split,
                              trust_remote_code=True)
        elif name == "2wikimultihopqa":
            ds = load_dataset("Alibaba-NLP/2WikiMultihopQA", split=split,
                              trust_remote_code=True)
        elif name == "musique":
            ds = load_dataset("Alibaba-NLP/musique", split=split,
                              trust_remote_code=True)
        else:
            return []
        return [dict(row) for i, row in enumerate(ds) if i < max_samples]
    except Exception as e:
        print(f"[MultiHop] failed to load {name}: {e}")
        return []


def _format_hotpotqa(row: Dict, max_words: int) -> Dict[str, Any]:
    """Format a HotpotQA sample."""
    title = row.get("context", {}).get("title", [])
    sentences = row.get("context", {}).get("sentences", [])
    question = row.get("question", "")
    answer = row.get("answer", "")

    # Build context from all passages
    passages = []
    for t, sents in zip(title, sentences):
        passage_text = " ".join(sents) if isinstance(sents, list) else str(sents)
        passages.append(f"[{t}] {passage_text}")

    context = "\n\n".join(passages)
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words])

    return {
        "context": context,
        "question": question,
        "answer": answer,
        "supporting_facts": row.get("supporting_facts", {}),
    }


def _format_2wiki(row: Dict, max_words: int) -> Dict[str, Any]:
    """Format a 2WikiMultihopQA sample."""
    context_items = row.get("context", [])
    question = row.get("question", "")
    answer = row.get("answer", "")

    passages = []
    for item in context_items:
        if isinstance(item, dict):
            title = item.get("title", "")
            text = " ".join(item.get("sentences", []))
            passages.append(f"[{title}] {text}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            passages.append(f"[{item[0]}] {' '.join(item[1]) if isinstance(item[1], list) else item[1]}")

    context = "\n\n".join(passages)
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words])

    return {"context": context, "question": question, "answer": answer}


def _format_musique(row: Dict, max_words: int) -> Dict[str, Any]:
    """Format a MuSiQue sample."""
    paragraphs = row.get("paragraphs", [])
    question = row.get("question", "")
    answer = row.get("answer", "")

    passages = []
    for p in paragraphs:
        if isinstance(p, dict):
            title = p.get("title", "")
            text = p.get("paragraph_text", "") or p.get("text", "")
            passages.append(f"[{title}] {text}")

    context = "\n\n".join(passages)
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words])

    return {"context": context, "question": question, "answer": answer}


FORMATTERS = {
    "hotpotqa": _format_hotpotqa,
    "2wikimultihopqa": _format_2wiki,
    "musique": _format_musique,
}


class MultiHopQA(BaseBenchmark):
    """Multi-hop question answering over long contexts.

    Tests whether eviction retains scattered evidence tokens necessary
    for multi-step reasoning, not just locally attended tokens.

    Args:
        dataset: one of "hotpotqa", "2wikimultihopqa", "musique"
        max_words: max context length in words
        n_samples: number of evaluation samples
        seed: random seed
    """

    name = "multihop"

    def __init__(
        self,
        dataset: str = "hotpotqa",
        split: str = "validation",
        max_words: int = 4000,
        n_samples: int = 100,
        seed: int = 0,
        max_samples: Optional[int] = None,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.dataset = dataset
        self.split = split
        self.max_words = max_words
        self.n_samples = n_samples

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        rows = _load_multihop_dataset(self.dataset, self.split, self.n_samples)
        if not rows:
            print(f"[MultiHop] no data for {self.dataset}, generating synthetic")
            return self._generate_synthetic(tokenizer)

        formatter = FORMATTERS.get(self.dataset, _format_hotpotqa)
        samples: List[Dict[str, Any]] = []

        for row in rows:
            formatted = formatter(row, self.max_words)
            context = formatted["context"]
            question = formatted["question"]
            answer = formatted["answer"]

            if not context or not question or not answer:
                continue

            text = f"{context}\n\nQuestion: {question}\nAnswer: {answer}"
            prefix_text = f"{context}\n\nQuestion: {question}\nAnswer:"

            ids = tokenizer(text, return_tensors="pt").input_ids
            prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
            prefix_len = prefix_ids.size(1)
            eval_positions = list(range(prefix_len, ids.size(1)))

            if not eval_positions:
                continue

            samples.append({
                "input_ids": ids,
                "eval_positions": eval_positions,
                "metadata": {
                    "dataset": self.dataset,
                    "answer": answer,
                    "question": question,
                    "seq_len": ids.size(1),
                    "prefix_len": prefix_len,
                    "supporting_facts": formatted.get("supporting_facts", {}),
                },
            })

        return samples

    def _generate_synthetic(self, tokenizer) -> List[Dict[str, Any]]:
        """Generate synthetic multi-hop samples when dataset is unavailable."""
        rng = random.Random(self.seed)
        samples: List[Dict[str, Any]] = []

        templates = [
            # 2-hop: A is related to B, B is related to C → question about A→C
            ("{entity_a} was founded by {entity_b}. {entity_b} was born in {location}.",
             "Where was the founder of {entity_a} born? The founder was born in",
             " {location}"),
            ("{person} works at {company}. {company} is headquartered in {city}.",
             "In which city does {person} work? They work in",
             " {city}"),
        ]

        entities = ["AcmeCorp", "TechVentures", "DataSys Inc", "CloudNine", "NovaStar"]
        people = ["Alice Chen", "Bob Smith", "Carol Davis", "Dan Lee", "Eve Park"]
        locations = ["Seattle", "London", "Tokyo", "Berlin", "Sydney", "Toronto"]

        for si in range(self.n_samples):
            tmpl_idx = si % len(templates)
            context_tmpl, question, answer_tmpl = templates[tmpl_idx]

            entity_a = entities[si % len(entities)]
            entity_b = people[si % len(people)]
            location = locations[si % len(locations)]
            company = entities[(si + 1) % len(entities)]
            city = locations[(si + 2) % len(locations)]
            person = people[(si + 1) % len(people)]

            context_text = context_tmpl.format(
                entity_a=entity_a, entity_b=entity_b, location=location,
                person=person, company=company, city=city)

            # Add filler to make it long
            filler = " ".join(f"Sentence {i}: some unrelated information about topic {i}."
                              for i in range(rng.randint(200, 500)))
            full_context = context_text + "\n\n" + filler

            q_text = question.format(
                entity_a=entity_a, person=person, company=company)
            a_text = answer_tmpl.format(
                location=location, city=city)

            text = f"{full_context}\n\nQuestion: {q_text}\nAnswer: {a_text}"
            prefix_text = f"{full_context}\n\nQuestion: {q_text}\nAnswer:"

            ids = tokenizer(text, return_tensors="pt").input_ids
            prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
            prefix_len = prefix_ids.size(1)
            eval_positions = list(range(prefix_len, ids.size(1)))

            if not eval_positions:
                continue

            samples.append({
                "input_ids": ids,
                "eval_positions": eval_positions,
                "metadata": {
                    "dataset": f"{self.dataset}_synthetic",
                    "answer": a_text.strip(),
                    "question": q_text,
                    "seq_len": ids.size(1),
                },
            })

        return samples

    def compute_metrics(
        self, nlls: List[float], sample: Dict[str, Any], extra=None,
    ) -> Dict[str, float]:
        if not nlls:
            return {"ppl": float("inf"), "dataset": self.dataset}
        mean_nll = sum(nlls) / len(nlls)
        return {
            "ppl": math.exp(mean_nll),
            "answer_ppl": math.exp(mean_nll),
            "dataset": self.dataset,
            "n_eval_tokens": len(nlls),
        }
