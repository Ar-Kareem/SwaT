from __future__ import annotations

import torch
from torch import Tensor
from transformers.cache_utils import DynamicCache


def full_sender_cache() -> DynamicCache:
    return DynamicCache()


def kv_layer_indices(config) -> list[int]:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    total = int(text.num_hidden_layers)
    cross = set(int(i) for i in (getattr(text, "cross_attention_layers", None) or []))
    return [i for i in range(total) if i not in cross]


def tuple_to_dynamic_cache(kv_tuple, config):
    cache = DynamicCache(config=config)
    indices = kv_layer_indices(config)
    if len(kv_tuple) != len(indices):
        raise ValueError(f"{len(kv_tuple)} rendered layers for a receiver with {len(indices)} kv layers")
    for layer_idx, (k, v) in zip(indices, kv_tuple):
        cache.update(k, v, layer_idx)
    return cache


def _base_model(model):
    for name in ("model", "language_model", "transformer", "gpt_neox"):
        module = getattr(model, name, None)
        if module is None:
            continue
        if hasattr(module, "layers"):
            return module
        for inner_name in ("language_model", "model"):
            inner = getattr(module, inner_name, None)
            if inner is not None and hasattr(inner, "layers"):
                return inner
    if hasattr(model, "layers"):
        return model
    raise AttributeError("could not find transformer layers on model")


def _kv_shape(model) -> tuple[int, int, int]:
    config = getattr(_base_model(model), "config", getattr(model, "config", None))
    if config is None:
        raise AttributeError("model has no config")

    num_layers = len(kv_layer_indices(config))
    num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    return num_layers, num_kv_heads, int(head_dim)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    base_model,
    key_states: Tensor,
    layer_idx: int,
    position_ids: Tensor | None = None,
    inverse: bool = False,
) -> Tensor:
    layers = getattr(base_model, "layers", None)
    if layers is not None:
        attention = getattr(layers[layer_idx], "self_attn", None)
        if getattr(attention, "use_rope", True) is False or getattr(attention, "use_rope", True) == 0:
            return key_states
    rotary_emb = getattr(base_model, "rotary_emb", None)
    if rotary_emb is None:
        return key_states

    batch, heads, seq_len, _ = key_states.shape
    if position_ids is not None and position_ids.ndim == 3:
        if tuple(position_ids.shape) != (batch, heads, seq_len):
            raise ValueError(
                f"head-specific position_ids must have shape {(batch, heads, seq_len)}, "
                f"got {tuple(position_ids.shape)}"
            )
        return torch.cat(
            [
                _apply_rope(
                    base_model,
                    key_states[:, head : head + 1],
                    layer_idx,
                    position_ids[:, head],
                    inverse,
                )
                for head in range(heads)
            ],
            dim=1,
        )

    rotary_device = key_states.device
    for tensor in (*rotary_emb.parameters(), *rotary_emb.buffers()):
        rotary_device = tensor.device
        break
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=rotary_device).unsqueeze(0).expand(batch, -1)
    else:
        position_ids = position_ids.to(device=rotary_device)
        position_ids = position_ids[:, -seq_len:]
    hidden_size = int(getattr(base_model.config, "hidden_size", key_states.shape[-1]))
    rotary_input = torch.empty(
        (batch, seq_len, hidden_size),
        device=rotary_device,
        dtype=key_states.dtype,
    )
    layer_types = getattr(base_model.config, "layer_types", None)
    layer_type = layer_types[layer_idx] if layer_types is not None else None

    try:
        cos, sin = rotary_emb(rotary_input, position_ids, layer_type)
    except TypeError:
        cos, sin = rotary_emb(rotary_input, position_ids)

    cos = cos.unsqueeze(1).to(device=key_states.device, dtype=key_states.dtype)
    sin = sin.unsqueeze(1).to(device=key_states.device, dtype=key_states.dtype)
    if inverse:
        sin = -sin
    return (key_states * cos) + (_rotate_half(key_states) * sin)


def cache_layer(cache, layer_idx: int) -> tuple[Tensor, Tensor]:
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache[layer_idx]
