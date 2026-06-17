"""Abstract benchmark interface for KV cache eviction evaluation."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass
class BenchmarkResult:
    """Standardized result from a benchmark run."""
    task: str
    method: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "method": self.method,
                "metrics": self.metrics, "metadata": self.metadata}


class BaseBenchmark(ABC):
    """Abstract base class for all benchmark tasks.

    Each benchmark prepares input data, runs autoregressive decoding with
    an eviction policy, and computes task-specific metrics.
    """

    name: str = "base"

    def __init__(self, seed: int = 0, max_samples: Optional[int] = None):
        self.seed = seed
        self.max_samples = max_samples

    @abstractmethod
    def prepare_samples(self, tokenizer) -> List[Dict[str, Any]]:
        """Return list of samples. Each sample has:
        - 'input_ids': torch.LongTensor [1, seq_len]
        - 'eval_positions': list[int] — token positions to evaluate PPL
        - 'metadata': dict — sample-level metadata (e.g. needle_pos, depth)
        """
        ...

    def load_samples(self, tokenizer, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load samples — convenience wrapper for run_benchmark.py.

        Calls prepare_samples() and optionally truncates to num_samples.
        Also renames 'eval_positions' to 'answer_positions' for compatibility
        with the runner script.
        """
        samples = self.prepare_samples(tokenizer)
        if num_samples is not None:
            samples = samples[:num_samples]
        # Add answer_positions alias
        for s in samples:
            if "answer_positions" not in s and "eval_positions" in s:
                s["answer_positions"] = s["eval_positions"]
            if "evidence_positions" not in s and "needle_pos_tokens" in s.get("metadata", {}):
                s["evidence_positions"] = [s["metadata"]["needle_pos_tokens"]]
            if "ground_truth" not in s and "answer" in s.get("metadata", {}):
                s["ground_truth"] = s["metadata"]["answer"]
        return samples

    @abstractmethod
    def compute_metrics(
        self,
        nlls: List[float],
        sample: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Compute task-specific metrics from per-token NLLs."""
        ...

    def evaluate(
        self,
        model,
        tokenizer,
        eviction_fn,
        device: str = "cuda",
        max_steps: int = 4096,
        progress_every: int = 200,
    ) -> List[BenchmarkResult]:
        """Run the full benchmark across all samples and methods.

        Args:
            model: loaded HF causal LM
            tokenizer: HF tokenizer
            eviction_fn: callable(past_key_values) -> past_key_values, or None for full cache
            device: device string
            max_steps: max decode steps per sample
            progress_every: print progress interval

        Returns:
            List of BenchmarkResult, one per sample.
        """
        from torch.nn import CrossEntropyLoss
        import time, math

        samples = self.prepare_samples(tokenizer)
        if self.max_samples is not None:
            samples = samples[:self.max_samples]

        loss_fn = CrossEntropyLoss(reduction="none")
        results: List[BenchmarkResult] = []

        for si, sample in enumerate(samples):
            input_ids = sample["input_ids"].to(device)
            eval_positions = set(sample.get("eval_positions", []))
            past_key_values = None
            nlls: List[float] = []
            kv_lens: List[int] = []
            t0 = time.perf_counter()

            total_steps = min(input_ids.size(1) - 1, max_steps)
            for idx in range(total_steps):
                token = input_ids[:, idx:idx + 1]
                target = input_ids[:, idx + 1:idx + 2].to(device).view(-1)

                if eviction_fn is not None:
                    past_key_values = eviction_fn.evict_for_space(past_key_values, num_coming=1)

                outputs = model(input_ids=token, past_key_values=past_key_values, use_cache=True)
                nll = loss_fn(outputs.logits[:, -1, :].view(-1, model.config.vocab_size), target)

                if not eval_positions or (idx + 1) in eval_positions:
                    nlls.append(nll.item())

                past_key_values = outputs.past_key_values
                if eviction_fn is not None:
                    past_key_values = eviction_fn(past_key_values)

                seq_len = self._get_seq_len(past_key_values)
                kv_lens.append(seq_len)

                if progress_every > 0 and (idx + 1) % progress_every == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  [{self.name}] sample {si}/{len(samples)} "
                          f"step {idx+1}/{total_steps} kv={seq_len} "
                          f"elapsed={elapsed:.1f}s", flush=True)

            if not nlls:
                continue

            metrics = self.compute_metrics(nlls, sample)
            metrics["tok_per_s"] = total_steps / max(time.perf_counter() - t0, 1e-6)
            metrics["max_kv_len"] = max(kv_lens) if kv_lens else 0

            result = BenchmarkResult(
                task=self.name,
                method=eviction_fn.name if hasattr(eviction_fn, "name") else "full",
                metrics=metrics,
                metadata={**sample.get("metadata", {}),
                          "sample_idx": si, "total_steps": total_steps},
            )
            results.append(result)

        return results

    @staticmethod
    def _get_seq_len(past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
        if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            k0 = past_key_values[0][0]
            return k0.shape[2] if k0.dim() == 4 else k0.shape[1]
        return 0
