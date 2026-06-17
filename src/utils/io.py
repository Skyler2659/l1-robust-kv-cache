"""Result I/O — save/load experiment results, scores, and selected tokens."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def save_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    path: Union[str, Path],
) -> None:
    """Save results as JSON. Tensors are converted to lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(results), f, indent=2, ensure_ascii=False)


def _convert(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    return obj


def save_jsonl(rows: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_convert(row), ensure_ascii=False) + "\n")


def load_results(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scores(
    scores: Dict[str, torch.Tensor],
    path: Union[str, Path],
) -> None:
    """Save per-layer scores as .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_scores = {k: v.detach().cpu() for k, v in scores.items()}
    torch.save(cpu_scores, path)


def load_scores(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def save_selected_tokens(
    selected: Dict[int, torch.Tensor],
    path: Union[str, Path],
) -> None:
    """Save per-layer selected indices."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_sel = {str(k): v.detach().cpu() for k, v in selected.items()}
    torch.save(cpu_sel, path)
