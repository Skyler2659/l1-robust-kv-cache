"""Model adapter metadata for multi-backend KV-cache experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from src.config import ModelConfig


@dataclass
class ModelAdapter:
    model_family: str
    model_name: str
    backend: str
    quant_bits: Optional[int]
    tokenizer_type: Optional[str]
    chat_template: bool
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]
    max_context_length: Optional[int]
    rope_type: Optional[str]
    uses_rope: bool
    cache_format: str
    supports_attention_output: bool
    supports_hidden_states: bool
    supports_mlx_cache_edit: bool
    default_prompt_format: str
    generation_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def infer_model_family(model_name: str, raw_model_type: Optional[str] = None) -> str:
    text = f"{model_name} {raw_model_type or ''}".lower()
    if "qwen" in text:
        return "qwen"
    if "llama" in text or "meta_llama" in text:
        return "llama"
    if "mistral" in text or "mixtral" in text:
        return "mistral"
    if "yi" in text:
        return "yi"
    if "gemma" in text:
        return "gemma"
    return "unknown"


def _get_token_id(tokenizer: Any, attr: str) -> Optional[int]:
    value = getattr(tokenizer, attr, None)
    if isinstance(value, int):
        return value
    return None


def build_model_adapter(
    cfg: ModelConfig,
    raw_config: Optional[Dict[str, Any]] = None,
    tokenizer: Any = None,
    cache_format: Optional[str] = None,
    attention_hook_installed: Optional[bool] = None,
) -> ModelAdapter:
    raw = raw_config or {}
    family = cfg.family or infer_model_family(cfg.name, raw.get("model_type"))
    tokenizer_type = cfg.tokenizer_type or (type(tokenizer).__name__ if tokenizer is not None else None)
    bos_id = cfg.bos_token_id if cfg.bos_token_id is not None else _get_token_id(tokenizer, "bos_token_id")
    eos_id = cfg.eos_token_id if cfg.eos_token_id is not None else _get_token_id(tokenizer, "eos_token_id")
    pad_id = cfg.pad_token_id if cfg.pad_token_id is not None else _get_token_id(tokenizer, "pad_token_id")
    max_ctx = (
        cfg.max_context_length
        or raw.get("max_position_embeddings")
        or raw.get("max_sequence_length")
        or raw.get("sliding_window")
    )
    rope_scaling = raw.get("rope_scaling") or {}
    rope_type = cfg.rope_type or raw.get("rope_type") or rope_scaling.get("rope_type") or rope_scaling.get("type")
    uses_rope = cfg.uses_rope
    if uses_rope is None:
        uses_rope = family in {"qwen", "llama", "mistral", "yi", "gemma"} or bool(rope_type)

    backend = str(cfg.backend or "torch").lower()
    if cfg.supports_attention_output is not None:
        supports_attention = bool(cfg.supports_attention_output)
    elif backend == "mlx":
        supports_attention = bool(attention_hook_installed)
    else:
        supports_attention = bool(cfg.output_attentions)

    if cfg.supports_hidden_states is not None:
        supports_hidden = bool(cfg.supports_hidden_states)
    else:
        supports_hidden = backend == "torch" and bool(cfg.output_hidden_states)

    if cfg.supports_mlx_cache_edit is not None:
        supports_mlx_cache_edit = bool(cfg.supports_mlx_cache_edit)
    else:
        supports_mlx_cache_edit = backend == "mlx"

    prompt_format = cfg.prompt_format or {"mode": cfg.default_prompt_format, "system_prompt": None}
    chat_template = cfg.chat_template
    if chat_template is None:
        chat_template = bool(prompt_format.get("mode") == "chat_template")

    return ModelAdapter(
        model_family=family,
        model_name=cfg.name,
        backend=backend,
        quant_bits=cfg.quant_bits,
        tokenizer_type=tokenizer_type,
        chat_template=bool(chat_template),
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        max_context_length=int(max_ctx) if max_ctx is not None else None,
        rope_type=rope_type,
        uses_rope=bool(uses_rope),
        cache_format=cfg.cache_format or cache_format or cfg.mlx_cache_type,
        supports_attention_output=supports_attention,
        supports_hidden_states=supports_hidden,
        supports_mlx_cache_edit=supports_mlx_cache_edit,
        default_prompt_format=cfg.default_prompt_format,
        generation_config=dict(cfg.generation or {}),
    )


def apply_prompt_format(tokenizer: Any, prompt: str, cfg: ModelConfig) -> str:
    """Format a benchmark prompt according to model prompt settings."""
    prompt_cfg = cfg.prompt_format or {"mode": "plain", "system_prompt": None}
    mode = str(prompt_cfg.get("mode", "plain") or "plain").lower()
    if mode != "chat_template":
        return prompt
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = []
    system_prompt = prompt_cfg.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt
