import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    rotate_half,
)
import types

__all__ = ["enable_falcon_pos_shift_attention"]

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


def falcon_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    **kwargs,
):
    # Polyfill: read from config to survive any transformers version
    _cfg = self.config
    self.num_heads = getattr(self, "num_heads",
        getattr(self, "num_attention_heads", _cfg.num_attention_heads))
    self.head_dim = getattr(self, "head_dim",
        getattr(self, "attention_head_dim",
            getattr(_cfg, "head_dim", _cfg.hidden_size // self.num_heads)))
    num_kv_heads = getattr(self, "num_kv",
        getattr(self, "num_key_value_heads",
            getattr(_cfg, "num_kv", getattr(_cfg, "num_key_value_heads", self.num_heads))))
    layer_idx = int(getattr(self, "layer_idx", 0) or 0)

    past_kv = layer_past
    if past_kv is None:
        past_kv = kwargs.get("past_key_value", kwargs.get("past_key_values"))
    past_k, past_v, past_kv_len = _resolve_past_kv_layer(past_kv, layer_idx)

    fused_qkv = self.query_key_value(
        hidden_states
    )  # [batch_size, seq_length, 3 x hidden_size]

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, q_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(
        batch_size * self.num_heads, q_length, self.head_dim
    )

    # dirty hack to fix the inconsistency between falcon-40b and falcon-7b
    num_kv = self.num_heads if self.num_heads == 128 else num_kv_heads
    key_layer = key_layer.transpose(1, 2).reshape(
        batch_size * num_kv,
        q_length,
        self.head_dim,
    )
    value_layer = value_layer.transpose(1, 2).reshape(
        batch_size * num_kv, q_length, self.head_dim
    )

    past_len = past_kv_len

    query_layer_copy = query_layer.clone()
    query_layer, _ = self.maybe_rotary(query_layer, query_layer_copy, past_len)
    shared_q.LAST_QUERY_STATES[layer_idx] = query_layer.reshape(
        batch_size, self.num_heads, q_length, self.head_dim
    )[0, :, -1, :].detach()
    if past_k is not None:
        key_layer = torch.cat((past_k, key_layer), dim=1)
        value_layer = torch.cat((past_v, value_layer), dim=1)

    if use_cache is True:
        present = (key_layer, value_layer)
    else:
        present = None

    key_layer_copy = key_layer.clone()
    _, key_layer = self.maybe_rotary(key_layer_copy, key_layer, 0)

    _, kv_length, _ = key_layer.shape
    key_heads = key_layer.reshape(batch_size, num_kv, kv_length, self.head_dim)[0]
    shared_q.LAST_KEY_ROWS[layer_idx] = key_heads.mean(dim=0).detach()
    if key_heads.shape[0] != self.num_heads and self.num_heads % key_heads.shape[0] == 0:
        key_heads = key_heads.repeat_interleave(self.num_heads // key_heads.shape[0], dim=0)
    shared_q.LAST_KEY_STATES[layer_idx] = key_heads.detach()

    if alibi is None:
        query_layer_ = query_layer.reshape(
            batch_size, self.num_heads, -1, self.head_dim
        )
        key_layer_ = key_layer.reshape(batch_size, num_kv, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv, -1, self.head_dim)

        if past_k is not None:
            attn_output = F.scaled_dot_product_attention(
                query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True
            )

        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        outputs = (output_tensor, present)
        assert not output_attentions  # not supported.
        return outputs
    else:
        attention_mask_float = (
            (attention_mask * 1.0).masked_fill(attention_mask, -1e9).to(torch.bfloat16)
        )
        matmul_result = query_layer @ key_layer.transpose(-1, -2)

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(
            batch_size, self.num_heads, q_length, kv_length
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        # attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(
            (attention_scores + alibi.view(batch_size, self.num_heads, 1, -1))
            * self.inv_norm_factor
            + attention_mask_float,
            dim=-1,
            dtype=hidden_states.dtype,
        )
        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads, q_length, kv_length
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = attention_probs_reshaped @ value_layer

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


def enable_falcon_pos_shift_attention(model, _counter=None):
    if _counter is None:
        _counter = [0]
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            enable_falcon_pos_shift_attention(
                module,
                _counter,
            )

        if "self_attention" == name[-14:]:
            model._modules[name].layer_idx = _counter[0]
            _counter[0] += 1
            model._modules[name].forward = types.MethodType(
                falcon_pos_shift_attention_forward, model._modules[name]
            )
