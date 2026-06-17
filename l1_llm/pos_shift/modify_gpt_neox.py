import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.gpt_neox.modeling_gpt_neox import (
    apply_rotary_pos_emb,
    rotate_half,
    GPTNeoXAttention,
)
import types

__all__ = ["enable_gpt_neox_pos_shift_attention"]

import shared_q


def _resolve_past_kv_layer(layer_past, layer_idx):
    """Extract per-layer (past_k, past_v, kv_len) from past_key_value.

    Compatible with legacy per-layer (K,V) tuples (transformers ≤4.48) and the
    full-DynamicCache pattern introduced in 4.50+.
    """
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
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def gpt_neox_pos_shift_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs,
):
    # Polyfill: read everything from config to survive any transformers version
    _cfg = self.config
    self.num_attention_heads = getattr(self, "num_attention_heads",
        _cfg.num_attention_heads)
    self.head_size = getattr(self, "head_size",
        getattr(self, "head_dim",
            getattr(_cfg, "head_dim", None) or _cfg.hidden_size // self.num_attention_heads))
    self.rotary_ndims = getattr(self, "rotary_ndims",
        getattr(self, "rotary_dim", int(self.head_size * 0.25)))
    self.hidden_size = getattr(self, "hidden_size", _cfg.hidden_size)
    layer_idx = int(getattr(self, "layer_idx", 0) or 0)

    # Support both old param name (layer_past) and new (past_key_value / past_key_values)
    past_key_value = layer_past
    if past_key_value is None:
        past_key_value = kwargs.get("past_key_value", kwargs.get("past_key_values"))
    past_k, past_v, past_kv_len = _resolve_past_kv_layer(past_key_value, layer_idx)
    has_layer_past = past_k is not None

    # Compute QKV
    qkv = self.query_key_value(hidden_states)

    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

    # Compute rotary embeddings on rotary_ndims
    query_rot = query[..., : self.rotary_ndims]
    query_pass = query[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    seq_len = key.shape[-2] + past_kv_len
    # rotary_emb API changed across transformers versions
    try:
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
    except TypeError:
        pos_for_rope = torch.arange(seq_len, device=value.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value, pos_for_rope)
    query = apply_rotary_pos_emb_single(query_rot, cos, sin, position_ids)
    query = torch.cat((query, query_pass), dim=-1)
    shared_q.LAST_QUERY_STATES[layer_idx] = query[0, :, -1, :].detach()  # [H, D]

    # Cache QKV values
    if has_layer_past:
        key = torch.cat((past_k, key), dim=-2)
        value = torch.cat((past_v, value), dim=-2)

    present = (key, value) if use_cache else None

    key_rot = key[..., : self.rotary_ndims]
    key_pass = key[..., self.rotary_ndims :]
    key_position_ids = torch.arange(seq_len, device=position_ids.device).unsqueeze(0)
    key = apply_rotary_pos_emb_single(key_rot, cos, sin, key_position_ids)
    key = torch.cat((key, key_pass), dim=-1)
    shared_q.LAST_KEY_ROWS[layer_idx] = key[0].mean(dim=0).detach()
    shared_q.LAST_KEY_STATES[layer_idx] = key[0].detach()

    # Compute attention
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    # Reshape outputs
    attn_output = self._merge_heads(
        attn_output, self.num_attention_heads, self.head_size
    )
    attn_output = self.dense(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def enable_gpt_neox_pos_shift_attention(model, _counter=None):
    if _counter is None:
        _counter = [0]
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            enable_gpt_neox_pos_shift_attention(
                module,
                _counter,
            )

        if isinstance(module, GPTNeoXAttention):
            module.layer_idx = _counter[0]
            _counter[0] += 1
            module.forward = types.MethodType(
                gpt_neox_pos_shift_attention_forward, module
            )
