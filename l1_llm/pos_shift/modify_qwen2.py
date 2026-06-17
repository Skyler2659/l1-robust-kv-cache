import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import types

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    rotate_half,
    repeat_kv,
)

__all__ = ["enable_qwen2_pos_shift_attention"]

import shared_q


def _resolve_past_kv_layer(layer_past, layer_idx):
    if layer_past is None:
        return None, None, 0
    if hasattr(layer_past, "to_legacy_cache"):
        legacy = layer_past.to_legacy_cache()
        idx = int(layer_idx or 0)
        if idx < len(legacy):
            pk, pv = legacy[idx]
            return pk, pv, int(pk.shape[-2])
        return None, None, 0
    return layer_past[0], layer_past[1], int(layer_past[0].shape[-2])


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def qwen2_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_values")
    _cfg = self.config
    self.num_heads = getattr(self, "num_heads",
        getattr(self, "num_attention_heads", _cfg.num_attention_heads))
    self.head_dim = getattr(self, "head_dim",
        getattr(self, "attention_head_dim",
            getattr(_cfg, "head_dim", None) or _cfg.hidden_size // self.num_heads))
    self.num_key_value_heads = getattr(self, "num_key_value_heads", _cfg.num_key_value_heads)
    self.num_key_value_groups = getattr(self, "num_key_value_groups",
        self.num_heads // self.num_key_value_heads)
    self.hidden_size = getattr(self, "hidden_size", _cfg.hidden_size)
    layer_idx = int(getattr(self, "layer_idx", 0) or 0)
    bsz, q_len, _ = hidden_states.size()

    past_k, past_v, past_kv_len = _resolve_past_kv_layer(past_key_value, layer_idx)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    new_key_states = key_states
    new_value_states = value_states

    kv_seq_len = key_states.shape[-2] + past_kv_len
    if hasattr(self, "rotary_emb"):
        pos_for_rope = torch.arange(kv_seq_len, device=value_states.device).unsqueeze(0)
        try:
            cos, sin = self.rotary_emb(value_states, position_ids=pos_for_rope)
        except TypeError:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    else:
        cos, sin = None, None

    if cos is not None:
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
    shared_q.LAST_QUERY_STATES[layer_idx] = query_states[0, :, -1, :].detach()
    if past_k is not None:
        key_states = torch.cat([past_k, key_states], dim=2)
        value_states = torch.cat([past_v, value_states], dim=2)

    if use_cache:
        if past_key_value is not None and hasattr(past_key_value, "update"):
            past_key_value.update(new_key_states, new_value_states, layer_idx)
            past_key_value_out = past_key_value
        else:
            past_key_value_out = (key_states, value_states)
    else:
        past_key_value_out = None

    if cos is not None:
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
    shared_q.LAST_KEY_ROWS[layer_idx] = key_states[0].mean(dim=0).detach()
    shared_q.LAST_KEY_STATES[layer_idx] = repeat_kv(
        key_states, self.num_key_value_groups
    )[0].detach()

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Compute attention in float32 to avoid fp16 overflow in Q·K^T
    attn_weights = torch.matmul(
        query_states.float(), key_states.transpose(2, 3).float()
    ) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask.float()
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None if not output_attentions else attn_weights, past_key_value_out


def enable_qwen2_pos_shift_attention(model, _counter=None):
    if _counter is None:
        _counter = [0]
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            enable_qwen2_pos_shift_attention(module, _counter)
        if isinstance(module, Qwen2Attention):
            module.layer_idx = _counter[0]
            _counter[0] += 1
            model._modules[name].forward = types.MethodType(
                qwen2_pos_shift_attention_forward, model._modules[name]
            )
