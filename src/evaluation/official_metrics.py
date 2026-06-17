"""Official-style LongBench and RULER generation metrics.

The implementations mirror the public LongBench/RULER scoring interfaces while
remaining dependency-light for Apple Silicon smoke tests.
"""
from __future__ import annotations

import ast
import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


LONGBENCH_PROMPTS: Dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or "
        "more sentences.\n\nQuery: {input}\nAnswer:"
    ),
}


LONGBENCH_MAX_NEW_TOKENS: Dict[str, int] = {
    "narrativeqa": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
}


LONGBENCH_METRIC_NAMES: Dict[str, str] = {
    "narrativeqa": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "musique": "qa_f1",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
}


def parse_references(value: Any) -> List[str]:
    """Return a clean list of reference strings from HF/list/string variants."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if x is not None]
            except Exception:
                pass
        return [value]
    return [str(value)]


def normalize_answer(text: str) -> str:
    """LongBench English QA normalization."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return " ".join(remove_articles(remove_punc((text or "").lower())).split())


def _f1_tokens(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    return _f1_tokens(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Dependency-free ROUGE-L F score over whitespace tokens."""
    pred_tokens = (prediction or "").split()
    gold_tokens = (ground_truth or "").split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def longbench_metric_name(task: Optional[str]) -> Optional[str]:
    return LONGBENCH_METRIC_NAMES.get((task or "").lower())


def longbench_score(task: str, prediction: str, references: Iterable[str]) -> Optional[float]:
    """Return the official-style LongBench score in 0-100 scale."""
    task_key = (task or "").lower()
    metric_name = longbench_metric_name(task_key)
    refs = [str(x) for x in references if x is not None]
    if not metric_name or not refs:
        return None
    if task_key in {"trec", "triviaqa", "samsum", "lsht"}:
        prediction = prediction.lstrip("\n").split("\n")[0]
    scorer = rouge_l_score if metric_name == "rouge_l" else qa_f1_score
    return round(100.0 * max(scorer(prediction, ref) for ref in refs), 4)


def ruler_task_family(task: Optional[str]) -> str:
    task_key = (task or "").lower()
    if task_key.startswith("niah"):
        return "niah"
    if task_key in {"vt", "variable_tracking"}:
        return "variable_tracking"
    if task_key in {"qa_1", "qa_2", "qa"}:
        return "qa"
    if task_key in {"cwe", "common_words", "common_words_extraction"}:
        return "common_words_extraction"
    if task_key in {"fwe", "freq_words", "freq_words_extraction"}:
        return "freq_words_extraction"
    return task_key


def ruler_metric_name(task: Optional[str]) -> str:
    family = ruler_task_family(task)
    if family == "qa":
        return "string_match_part"
    return "string_match_all"


def ruler_score(task: str, prediction: str, references: Iterable[str]) -> Optional[float]:
    refs = [str(x) for x in references if x is not None]
    if not refs:
        return None
    pred = (prediction or "").lower()
    if ruler_metric_name(task) == "string_match_part":
        score = max(1.0 if ref.lower() in pred else 0.0 for ref in refs)
    else:
        score = sum(1.0 if ref.lower() in pred else 0.0 for ref in refs) / len(refs)
    return round(100.0 * score, 4)


def evaluate_official(
    benchmark: str,
    task: Optional[str],
    prediction: str,
    sample: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a generation with task-specific official-style metrics."""
    metadata = metadata or {}
    refs = (
        sample.get("answers")
        or sample.get("outputs")
        or metadata.get("answers")
        or metadata.get("outputs")
        or [sample.get("ground_truth") or metadata.get("answer")]
    )
    references = parse_references(refs)
    bench = (benchmark or "").lower()
    task_key = (task or metadata.get("task") or metadata.get("dataset") or "").lower()
    score: Optional[float] = None
    metric_name: Optional[str] = None
    implementation: Optional[str] = None
    if bench == "longbench":
        metric_name = longbench_metric_name(task_key)
        score = longbench_score(task_key, prediction, references)
        implementation = "longbench_public_metric_compatible"
        if metric_name == "rouge_l":
            implementation = "longbench_rouge_l_lcs_fallback"
    elif bench == "ruler":
        metric_name = ruler_metric_name(task_key)
        score = ruler_score(task_key, prediction, references)
        implementation = "ruler_public_string_match"
    if score is None:
        return {
            "official_score": None,
            "official_correct": None,
            "official_metric_name": metric_name,
            "official_metric_implementation": implementation,
            "official_references": references,
        }
    correct: Optional[bool] = None
    if bench == "ruler":
        correct = bool(score >= 99.999)
    return {
        "official_score": score,
        "official_correct": correct,
        "official_metric_name": metric_name,
        "official_metric_implementation": implementation,
        "official_references": references,
    }
