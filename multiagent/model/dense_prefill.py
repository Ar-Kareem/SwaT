from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers.modeling_utils import AttentionInterface
from transformers.models.llama.modeling_llama import repeat_kv

from multiagent.model.unaware_comm import _base_model, kv_layer_indices

DENSE_ATTN_NAME = "dense_prefill_two_source"
_RENDERED_ATTR = "_dense_rendered_kv"
_MASK_ATTR = "_dense_layer_mask"


def _dense_two_source_attention(
    module: torch.nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[Tensor, None]:
    kwargs_mask = attention_mask
    rendered = getattr(module, _RENDERED_ATTR, None)
    if rendered is None:
        raise RuntimeError("dense two-source attention ran without rendered KV stashed on the module")
    if attention_mask is None or attention_mask.shape[-1] != 2 * key.shape[2]:
        raise RuntimeError("dense two-source attention requires the prepared [B, 1, R, 2R] mask")
    attention_mask = getattr(module, _MASK_ATTR, None)
    if attention_mask is None:
        attention_mask = kwargs_mask
    rendered_key, rendered_value = rendered
    key = torch.cat([rendered_key, key], dim=2)
    value = torch.cat([rendered_value, value], dim=2)
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=False
    )
    return attn_output.transpose(1, 2).contiguous(), None


AttentionInterface.register(DENSE_ATTN_NAME, _dense_two_source_attention)


def _sliding_window_limit(base) -> int | None:
    config = base.config
    window = getattr(config, "sliding_window", None)
    if window is None or window <= 0:
        return None
    if getattr(config, "use_sliding_window", True) is False:
        return None
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and "sliding_attention" not in layer_types:
        return None
    return int(window)


def _is_sliding(module) -> bool | None:
    if hasattr(module, "is_sliding"):
        return bool(module.is_sliding)
    if hasattr(module, "sliding_window"):
        return module.sliding_window is not None
    return None


def check_dense_sliding_window(base, seq_len: int, attention_modules=()) -> None:
    window = _sliding_window_limit(base)
    types_known = bool(attention_modules) and all(_is_sliding(m) is not None for m in attention_modules)
    if window is not None and seq_len > window and not types_known:
        raise ValueError(
            f"dense prefill on a sliding-window receiver: document length {seq_len} exceeds the "
            f"{window}-token local window; the two-source mask does not implement sliding-window "
            f"attention for local layers. Lower --dense-max-length to <= {window} or implement per-layer masks."
        )


def dense_attention_mask(seq_len: int, window: int, lengths: Tensor | None, dtype: torch.dtype, device) -> Tensor:
    index = torch.arange(seq_len, device=device)
    i, j = index[:, None], index[None, :]
    rendered_ok = j < (i - window)
    real_ok = (j >= (i - window)) & (j <= i)
    allowed = torch.cat([rendered_ok, real_ok], dim=1)[None, None]
    if lengths is not None:
        column_valid = index[None, :] < lengths.to(device)[:, None]
        column_valid = torch.cat([column_valid, column_valid], dim=1)[:, None, None, :]
        allowed = allowed & column_valid
        diagonal = torch.cat([torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device), torch.eye(seq_len, dtype=torch.bool, device=device)], dim=1)
        allowed = allowed | diagonal[None, None]
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(~allowed, torch.finfo(dtype).min)


def dense_prefill_logits(
    receiver_model,
    rendered_layers: tuple[tuple[Tensor, Tensor], ...],
    input_ids: Tensor,
    position_ids: Tensor,
    window: int,
    lengths: Tensor | None = None,
) -> Tensor:
    base = _base_model(receiver_model)
    attention_modules = [base.layers[i].self_attn for i in kv_layer_indices(base.config)]
    if len(attention_modules) != len(rendered_layers):
        raise ValueError(f"rendered {len(rendered_layers)} layers for a {len(attention_modules)}-layer receiver")
    dtype = next(receiver_model.parameters()).dtype
    seq_len = input_ids.shape[1]
    check_dense_sliding_window(base, seq_len, attention_modules)
    mask = dense_attention_mask(seq_len, window, lengths, dtype, input_ids.device)
    local_window = _sliding_window_limit(base)
    local_mask = None
    if local_window is not None and seq_len > local_window:
        index = torch.arange(seq_len, device=input_ids.device)
        outside = (index[None, :] <= index[:, None] - local_window).repeat(1, 2)[None, None]
        local_mask = mask.masked_fill(outside, torch.finfo(dtype).min)
    previous_impl = receiver_model.config._attn_implementation
    for module, (key, value) in zip(attention_modules, rendered_layers):
        setattr(module, _RENDERED_ATTR, (key, value))
        setattr(module, _MASK_ATTR, local_mask if local_mask is not None and _is_sliding(module) else None)
    receiver_model.config._attn_implementation = DENSE_ATTN_NAME
    try:
        output = receiver_model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
        )
    finally:
        receiver_model.config._attn_implementation = previous_impl
        for module in attention_modules:
            setattr(module, _RENDERED_ATTR, None)
            setattr(module, _MASK_ATTR, None)
    return output.logits
