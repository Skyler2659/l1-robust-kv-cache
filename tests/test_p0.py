import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.run_benchmark import instantiate_benchmark
from src.benchmarks.niah import NIAHBenchmark
from src.config import ExperimentConfig
from src.eviction.base import validate_selected_indices
from src.eviction.l2_leverage import l2_row_leverage_scores
from src.eviction.registry import create_eviction


class Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharTokenizer:
    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, add_special_tokens=True):
        ids = [ord(ch) for ch in text]
        out = Enc()
        tensor = torch.tensor([ids], dtype=torch.long)
        out["input_ids"] = tensor if return_tensors == "pt" else ids
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        return "".join(chr(int(i)) for i in ids)


def test_config_load_utf8(tmp_path):
    path = tmp_path / "quick_utf8.yaml"
    text = Path("configs/experiments/dev/tiny_niah_cpu.yaml").read_text(encoding="utf-8")
    path.write_text(text.replace("tiny_niah_cpu", "tiny_niah_cpu_中文"), encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.experiment_name == "tiny_niah_cpu_中文"


def test_mlx_config_fields_load():
    cfg = ExperimentConfig.from_yaml("configs/experiments/dev/qwen25_05b_mlx_method_sanity.yaml")
    from src.runners.mlx_runner import SUPPORTED_MLX_METHODS, canonical_method

    assert cfg.model.backend == "mlx"
    assert cfg.model.quant_bits == 4
    assert cfg.model.quant_group_size == 64
    assert cfg.save_selected_tokens is True
    for method in (
        "attention",
        "h2o",
        "snapkv",
        "pyramidkv",
        "attention+l1",
        "attention+l2",
        "l1_decode_only",
        "l2_decode_only",
        "l2_key_prefill_only",
        "compactor",
    ):
        assert canonical_method(method) in SUPPORTED_MLX_METHODS
    for method in ("attention", "snapkv", "pyramidkv", "attention_l1", "attention_l2", "compactor"):
        assert method in cfg.methods
    assert canonical_method("snap") == "snapkv"


def test_registry_metadata_and_aliases():
    from src.eviction.registry import get_method_spec, method_requires_attention, unsupported_reason

    assert get_method_spec("sink_recency").name == "sink_recent"
    assert get_method_spec("h2o_style").family == "attention"
    assert method_requires_attention("accumulated_attention") is True
    assert get_method_spec("oracle_evidence").oracle is True
    assert unsupported_reason("hidden_l2_norm", "mlx")
    assert get_method_spec("l1_decode_only").supports_mlx is True
    assert unsupported_reason("l1_decode_only", "torch")
    assert get_method_spec("compactor_style").name == "compactor"
    assert get_method_spec("l2_key_prefill_only").score_source == "key"


def test_benchmark_instantiation_niah():
    cfg = ExperimentConfig.from_yaml("configs/experiments/dev/tiny_niah_cpu.yaml")
    bench = instantiate_benchmark(cfg)
    assert isinstance(bench, NIAHBenchmark)


def test_ruler_prompt_answer_boundary():
    from src.benchmarks.ruler import RULERBenchmark

    tokenizer = CharTokenizer()
    bench = RULERBenchmark(
        tasks=["variable_tracking"],
        n_samples_per_task=1,
        seq_words=300,
        seed=42,
    )
    sample = bench.load_samples(tokenizer, 1)[0]
    prompt = sample["prompt"]
    answer_text = sample["answer_text"]
    answer_positions = sample["answer_positions"]

    assert sample["full_text"] == prompt + answer_text
    assert prompt.endswith("The value is")
    assert not prompt.endswith(answer_text)
    assert answer_positions[0] == len(prompt)
    assert answer_positions[-1] == len(sample["full_text"]) - 1
    assert sample["metadata"]["answer_token_start"] == answer_positions[0]


def test_eviction_constructor_filter():
    eviction = create_eviction(
        "recency",
        cache_size=8,
        score_source="v",
        sketch_dim=16,
        seed=0,
        k_seq_dim=2,
        v_seq_dim=2,
    )
    assert eviction.cache_size == 8


def test_budget_validity_for_all_methods():
    methods = [
        "recency",
        "sink_recent",
        "sink_recency",
        "random",
        "sink_recent_random",
        "uniform",
        "attention",
        "last_token_attention",
        "windowed_attention",
        "attention_decay",
        "h2o",
        "snapkv",
        "l1_leverage",
        "l2_leverage",
        "ridge_leverage",
        "approximate_l2_leverage",
        "key_norm",
        "value_norm",
        "key_l1_norm",
        "value_l1_norm",
        "kv_norm",
        "farthest_point",
        "kmeans_medoid",
        "facility_location_greedy",
        "pca_residual",
        "mahalanobis_distance",
        "zscore_outlier",
        "random_projection_outlier",
        "attention+l1",
        "attention_l2",
        "attention_recency",
        "sink_recent_l1",
        "sink_recent_l2",
        "oracle_evidence",
    ]
    for method in methods:
        eviction = create_eviction(
            method,
            cache_size=8,
            k_seq_dim=2,
            v_seq_dim=2,
            sink_size=2,
            recent_size=3,
            sketch_dim=16,
            n_clusters=4,
            debug_budget=True,
        )
        eviction.set_sample_metadata(
            {"evidence_positions": [3, 4], "metadata": {"answer_token_start": 10, "answer_token_end": 12}}
        )
        k = torch.randn(1, 2, 20, 4)
        v = torch.randn(1, 2, 20, 4)
        eviction(((k, v),))
        selected = eviction.last_selected[0]
        validate_selected_indices(selected, seq_len=20, budget=8)


def test_l2_leverage_not_equal_norm():
    rows = torch.randn(12, 4)
    scores = l2_row_leverage_scores(rows)
    rank = torch.linalg.matrix_rank(rows.float()).item()
    assert bool((scores >= -1e-6).all())
    assert abs(float(scores.sum()) - rank) < 1e-4
    assert not torch.allclose(scores, torch.norm(rows.float(), dim=1), atol=1e-4)

    q, _ = torch.linalg.qr(torch.randn(10, 3), mode="reduced")
    q_scores = l2_row_leverage_scores(q)
    assert torch.allclose(q_scores, q.pow(2).sum(dim=1), atol=1e-5)

    low_rank = torch.ones(8, 3)
    low_scores = l2_row_leverage_scores(low_rank)
    assert torch.isfinite(low_scores).all()


def test_niah_evidence_span_decode():
    tokenizer = CharTokenizer()
    bench = NIAHBenchmark(depths=[0.2, 0.8], max_words=60, needles_per_depth=1)
    samples = bench.load_samples(tokenizer, 2)
    for sample in samples:
        meta = sample["metadata"]
        ids = sample["input_ids"][0]
        decoded = tokenizer.decode(ids[meta["needle_token_start"] : meta["needle_token_end"]])
        assert meta["value"] in decoded
        assert len(sample["evidence_positions"]) == meta["needle_token_end"] - meta["needle_token_start"]


def test_analysis_loads_json_selected_and_scores(tmp_path):
    from scripts.run_analysis import _load_score_dict, _load_selected

    selected_path = tmp_path / "selected.json"
    scores_path = tmp_path / "scores.json"
    selected_path.write_text(json.dumps({"0": [1, 2, 3]}), encoding="utf-8")
    scores_path.write_text(json.dumps({"0": [0.1, 0.2, 0.3]}), encoding="utf-8")

    selected = _load_selected({"selected_tokens_path": str(selected_path)})
    scores = _load_score_dict({"scores_path": str(scores_path)})

    assert selected[0].dtype == torch.long
    assert selected[0].tolist() == [1, 2, 3]
    assert torch.allclose(scores[0], torch.tensor([0.1, 0.2, 0.3]))


def test_official_metric_helpers():
    from src.evaluation.official_metrics import longbench_score, ruler_score

    assert longbench_score("narrativeqa", "the red door", ["red door"]) == 100.0
    assert longbench_score("hotpotqa", "Barack Obama was born in Hawaii.", ["Hawaii"]) > 0
    assert ruler_score("vt", "AAA BBB", ["AAA", "BBB", "CCC"]) == 66.6667
    assert ruler_score("niah_single_1", "The number is 12345.", ["12345"]) == 100.0


def test_longbench_official_prompt_metadata():
    from src.benchmarks.longbench import _build_qa_sample

    sample = _build_qa_sample(
        {
            "_task": "hotpotqa",
            "context": "Passage A.",
            "input": "Where?",
            "answers": ["Hawaii"],
            "length": 10,
            "all_classes": None,
        },
        max_words=0,
        use_official_prompt=True,
    )
    assert sample["dataset_official"] is True
    assert sample["official_prompt"] is True
    assert sample["official_metric_name"] == "qa_f1"
    assert "Only give me the answer" in sample["prefix_text"]


def test_mlx_faithful_prefill_headwise_smoke():
    mx = pytest.importorskip("mlx.core")
    from src.runners.mlx_runner import MLXCacheEvictor, _cache_head_valid_attention_mask

    class Cache:
        pass

    def make_cache(seq_len=24, heads=2, dim=8):
        c = Cache()
        c.keys = mx.random.normal((1, heads, seq_len, dim))
        c.values = mx.random.normal((1, heads, seq_len, dim))
        c.offset = seq_len
        c.logical_offset = seq_len
        return [c]

    cfg = ExperimentConfig()
    cfg.eviction.window_size = 4
    cfg.eviction.pooling_kernel = 5
    cfg.eviction.pooling_method = "avgpool"
    cfg.eviction.compactor_sketch_dim = 4
    cfg.eviction.compactor_chunk_size = 8
    cfg.eviction.compactor_attention_chunk_size = 8
    cfg.eviction.compactor_protected_first_tokens = 2
    cfg.eviction.compactor_protected_last_tokens = 3
    cfg.seed = 7

    for method in ("snapkv", "pyramidkv"):
        cache = make_cache()
        state = {
            "observe_heads": {0: [mx.random.uniform(shape=(2, 24)) for _ in range(4)]},
            "observe": {},
            "last": {},
            "accumulated": {},
            "decayed": {},
            "hook_errors": 0,
        }
        evictor = MLXCacheEvictor(method, 8, cfg, 1, attention_state=state)
        evictor.set_phase("prefill")
        evictor.prefill_compress(cache, 8)
        c = cache[0]
        mask = _cache_head_valid_attention_mask(c, 4, c.offset)
        assert c.logical_offset == 24
        assert c.head_valid_mask.shape == (2, c.offset)
        assert mask.shape == (4, 1, c.offset)
        assert evictor.last_selected_by_head[0]

    cache = make_cache(seq_len=24, heads=2, dim=8)
    state = {
        "observe_heads": {},
        "observe": {},
        "last": {},
        "accumulated": {},
        "decayed": {},
        "hook_errors": 0,
        "prefill_q_post": {0: [mx.random.normal((1, 4, 24, 8))]},
        "prefill_k_post": {0: [mx.random.normal((1, 2, 24, 8))]},
        "prefill_k_pre": {0: [mx.random.normal((1, 2, 24, 8))]},
    }
    evictor = MLXCacheEvictor("compactor", 8, cfg, 1, attention_state=state)
    evictor.set_phase("prefill")
    evictor.prefill_compress(cache, 8)
    c = cache[0]
    assert c.logical_offset == 24
    assert c.head_valid_mask.shape[1] == c.offset
    assert sum(len(v) for v in evictor.last_selected_by_head[0].values()) <= 16
