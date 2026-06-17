"""Model loading and KV cache probing utilities."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ModelConfig
from src.eviction.kv_utils import infer_kv_dims
from src.model_adapters import build_model_adapter


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(device: str) -> str:
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was requested but CUDA is not available")
        return requested
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("device=mps was requested but MPS is not available")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device: {device!r}; expected cpu/cuda/mps/auto")


def load_model_and_tokenizer(
    cfg: ModelConfig,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a HuggingFace causal LM + tokenizer.

    Returns ``(model, tokenizer, info_dict)``.
    """
    device = resolve_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _DTYPE_MAP.get(cfg.dtype, torch.float32)
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    if device == "mps" and dtype == torch.bfloat16:
        dtype = torch.float16
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if cfg.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if cfg.local_files_only:
        load_kwargs["local_files_only"] = True
    if cfg.attn_implementation:
        load_kwargs["attn_implementation"] = cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(cfg.name, **load_kwargs)
    model = model.to(device).eval()
    model.config.output_attentions = bool(cfg.output_attentions)
    model.config.output_hidden_states = bool(cfg.output_hidden_states)

    info = {
        "model_name": cfg.name,
        "model_type": getattr(model.config, "model_type", "unknown"),
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "requested_device": cfg.device,
        "num_layers": getattr(model.config, "num_hidden_layers", None),
        "num_heads": getattr(model.config, "num_attention_heads", None),
        "num_kv_heads": getattr(
            model.config,
            "num_key_value_heads",
            getattr(model.config, "num_attention_heads", None),
        ),
        "head_dim": getattr(model.config, "head_dim", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
    }

    # Infer KV dims
    k_seq_dim, v_seq_dim = infer_kv_dims(info["model_type"])
    info["k_seq_dim"] = k_seq_dim
    info["v_seq_dim"] = v_seq_dim

    # Probe cache format
    with torch.no_grad():
        probe = model(
            tokenizer("probe", return_tensors="pt").input_ids.to(device),
            use_cache=True,
        )
    info["cache_format"] = (
        "DynamicCache"
        if not isinstance(probe.past_key_values, tuple)
        else "legacy"
    )
    adapter = build_model_adapter(
        cfg,
        raw_config=model.config.to_dict() if hasattr(model.config, "to_dict") else {},
        tokenizer=tokenizer,
        cache_format=info["cache_format"],
    )
    info.update(adapter.to_dict())

    # Enable pos-shift if requested
    if cfg.enable_pos_shift:
        _try_enable_pos_shift(model)

    return model, tokenizer, info


def _try_enable_pos_shift(model: Any) -> None:
    """Attempt to enable RoPE pos-shift monkey patches."""
    model_type = (getattr(model.config, "model_type", "") or "").lower()
    try:
        file_map = {
            "llama": ("modify_llama", "enable_llama_pos_shift_attention"),
            "gpt_neox": ("modify_gpt_neox", "enable_gpt_neox_pos_shift_attention"),
            "qwen2": ("modify_qwen2", "enable_qwen2_pos_shift_attention"),
            "falcon": ("modify_falcon", "enable_falcon_pos_shift_attention"),
        }
        for key, (mod_name, func_name) in file_map.items():
            if key in model_type:
                import importlib
                mod = importlib.import_module(f"l1_llm.pos_shift.{mod_name}")
                getattr(mod, func_name)(model)
                print(f"pos_shift: enabled for {key}")
                return
        print(f"pos_shift: no patch for model_type={model_type}")
    except Exception as exc:
        print(f"pos_shift: failed ({exc}); continuing without patch")
