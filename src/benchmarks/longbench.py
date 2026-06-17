"""LongBench wrapper — standard long-context QA and summarization tasks.

LongBench (Bai et al., 2023) covers:
- Single-document QA (NarrativeQA, QuALITY, TriviaQA)
- Multi-document QA (HotpotQA, 2WikiMultihopQA, MuSiQue)
- Summarization (GovReport, QMSum, MultiNews)
- Few-shot learning (TREC, SAMSum)
- Code completion (LCC, RepoBench-P)
- Synthetic (PassageCount, PassageRetrieval)

This wrapper loads the HF dataset and converts samples to the common
benchmark interface.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math
import torch

from src.benchmarks.base import BaseBenchmark, BenchmarkResult
from src.evaluation.official_metrics import (
    LONGBENCH_MAX_NEW_TOKENS,
    LONGBENCH_PROMPTS,
    longbench_metric_name,
    parse_references,
)


# LongBench task categories
TASK_CATEGORIES = {
    "single_doc_qa": ["narrativeqa", "quality", "triviaqa"],
    "multi_doc_qa": ["hotpotqa", "2wikimultihopqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "samsum"],
    "code": ["lcc", "repobench-p"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
}

ALL_TASKS = [t for tasks in TASK_CATEGORIES.values() for t in tasks]


STRICT_ANSWER_PROMPTS = {
    "hotpotqa": (
        "Answer the question using the passages below.\n"
        "Return exactly the shortest answer phrase, usually a named entity or title.\n"
        "Do not write a sentence. Do not explain. Do not add citations, punctuation, or extra words.\n\n"
        "Passages:\n{context}\n\n"
        "Question: {input}\n"
        "Answer phrase only:"
    ),
    "musique": (
        "Answer the question using the passages below.\n"
        "Return exactly the shortest answer phrase. Do not explain or add extra words.\n\n"
        "Passages:\n{context}\n\n"
        "Question: {input}\n"
        "Answer phrase only:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer the question.\n"
        "Return exactly the shortest answer phrase. Do not explain or add extra words.\n\n"
        "{context}\n\n"
        "Question: {input}\n"
        "Answer phrase only:"
    ),
}


def _truncate_to_max_words(text: str, max_words: int) -> str:
    if max_words is None or int(max_words) <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _build_qa_sample(
    row: Dict[str, Any],
    max_words: int = 4000,
    use_official_prompt: bool = False,
    prompt_style: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a QA sample from a LongBench row."""
    task = row.get("_task") or row.get("dataset") or "unknown"
    context = row.get("context", "") or ""
    question = row.get("input", "") or row.get("question", "") or ""
    answers = parse_references(row.get("answers", []) or [])

    context = _truncate_to_max_words(context, max_words)
    style = (prompt_style or "").lower()
    if style in {"strict_answer", "answer_only", "strict_qa"} and task in STRICT_ANSWER_PROMPTS:
        text = STRICT_ANSWER_PROMPTS[task].format(context=context, input=question)
        official_prompt = False
    elif use_official_prompt and task in LONGBENCH_PROMPTS:
        text = LONGBENCH_PROMPTS[task].format(context=context, input=question)
        official_prompt = True
    else:
        text = f"{context}\n\nQuestion: {question}\nAnswer:"
        official_prompt = False
    answer_text = f" {answers[0]}" if answers else " unknown"
    full_text = text + answer_text

    return {
        "text": full_text,
        "prefix_text": text,
        "answer": answer_text.strip(),
        "answers": answers,
        "task": task,
        "all_classes": row.get("all_classes"),
        "length": row.get("length"),
        "dataset_official": True,
        "official_prompt": official_prompt,
        "prompt_style": style or ("official" if official_prompt else "default"),
        "official_metric_name": longbench_metric_name(task),
    }


class LongBenchWrapper(BaseBenchmark):
    """Wraps LongBench HF dataset for KV cache eviction evaluation.

    Uses PPL on answer tokens as the primary metric (consistent with
    the rest of the framework).

    Args:
        tasks: list of LongBench task names (e.g. ["narrativeqa", "hotpotqa"])
        max_words: max context words per sample
        n_samples_per_task: max samples per task
        seed: random seed for sample selection
    """

    name = "longbench"

    def __init__(
        self,
        tasks: Optional[List[str]] = None,
        max_words: int = 4000,
        n_samples_per_task: int = 50,
        seed: int = 0,
        max_samples: Optional[int] = None,
        use_official_prompt: bool = False,
        prompt_style: Optional[str] = None,
        max_length: int = 32768,
    ):
        super().__init__(seed=seed, max_samples=max_samples)
        self.tasks = tasks or ["narrativeqa", "hotpotqa", "triviaqa"]
        self.max_words = max_words
        self.n_samples_per_task = n_samples_per_task
        self.use_official_prompt = use_official_prompt
        self.prompt_style = prompt_style
        self.max_length = max_length

    def _load_task_data(self, task_name: str) -> List[Dict[str, Any]]:
        """Load a single LongBench task from HF datasets."""
        try:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", task_name, split="test",
                              trust_remote_code=True)
            rows = []
            for i, row in enumerate(ds):
                if i >= self.n_samples_per_task:
                    break
                row["_task"] = task_name
                rows.append(dict(row))
            return rows
        except Exception as e:
            print(f"[LongBench] failed to load {task_name}: {e}")
            return []

    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for task_name in self.tasks:
            rows = self._load_task_data(task_name)
            if not rows:
                print(f"[LongBench] no data for task: {task_name}")
                continue

            for row in rows:
                raw = _build_qa_sample(
                    row,
                    max_words=self.max_words,
                    use_official_prompt=self.use_official_prompt,
                    prompt_style=self.prompt_style,
                )
                text = raw["text"]
                prefix_text = raw["prefix_text"]

                if self.use_official_prompt and self.max_length and self.max_length > 0:
                    prompt_ids = tokenizer(
                        prefix_text,
                        truncation=False,
                        return_tensors="pt",
                    ).input_ids[0]
                    if prompt_ids.numel() > self.max_length:
                        half = int(self.max_length / 2)
                        prefix_text = (
                            tokenizer.decode(prompt_ids[:half], skip_special_tokens=True)
                            + tokenizer.decode(prompt_ids[-half:], skip_special_tokens=True)
                        )
                        text = prefix_text + f" {raw['answer']}"

                ids = tokenizer(text, return_tensors="pt").input_ids
                prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
                prefix_len = prefix_ids.size(1)
                eval_positions = list(range(prefix_len, ids.size(1)))

                if not eval_positions:
                    continue

                samples.append({
                    "input_ids": ids,
                    "eval_positions": eval_positions,
                    "answer_positions": eval_positions,
                    "prompt": prefix_text,
                    "full_text": text,
                    "ground_truth": raw["answer"],
                    "answers": raw.get("answers", []),
                    "all_classes": raw.get("all_classes"),
                    "metadata": {
                        "task": task_name,
                        "answer": raw["answer"],
                        "answers": raw.get("answers", []),
                        "all_classes": raw.get("all_classes"),
                        "length": raw.get("length"),
                        "dataset_official": raw.get("dataset_official"),
                        "official_prompt": raw.get("official_prompt"),
                        "prompt_style": raw.get("prompt_style"),
                        "official_metric_name": raw.get("official_metric_name"),
                        "official_max_new_tokens": LONGBENCH_MAX_NEW_TOKENS.get(task_name),
                        "seq_len": ids.size(1),
                        "prefix_len": prefix_len,
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
