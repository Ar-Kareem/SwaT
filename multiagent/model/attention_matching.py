from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal
import threading

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from multiagent.model.unaware_comm import _apply_rope, _base_model, cache_layer

ALL_MASK_ATTENTION_FUNCTIONS.register("am_eager", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
_am_head_thread = threading.local()


@dataclass
class AttentionMatchingConfig:
    target_ratio: float
    query_samples: int = 512
    nnls_iters: int = 2
    beta_min: float = -3.0
    beta_max: float = 3.0
    score_method: str = "rms"
    min_keys: int = 0
    head_budget_mode: Literal["uniform", "precomputed", "explicit_ratios"] = "uniform"
    head_budget_shares: list[list[float]] | None = None
    head_budget_ratios: list[list[float]] | None = None
    chunk_size: int = 0
    pack: Literal["original", "fractional"] = "original"
    packing_window: int | None = None
    generation_reserve: int = 0
    query_storage: Literal["cpu", "device"] = "cpu"
    attention_backend: Literal["eager"] = "eager"
    decode_backend: Literal["eager", "torchscript_prefill"] = "eager"
    compaction_workers: int = 1
    oracle_queries: bool = False


def repeat_prefill_prefix_ids(tokenizer) -> list[int]:

    messages = [{"role": "user", "content": "Repeat the previous context verbatim."}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    if prompt.startswith("<bos>"):
        prompt = prompt[len("<bos>") :]
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def _attention_module(layer):
    attention = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attention is None or not hasattr(attention, "q_proj"):
        raise AttributeError("could not find a self-attention module with q_proj")
    return attention


def _num_query_heads(base_model) -> int:
    return int(base_model.config.num_attention_heads)


def _num_kv_heads(base_model) -> int:
    return int(getattr(base_model.config, "num_key_value_heads", base_model.config.num_attention_heads))


def _nnls_projected_gradient(
    matrix: Tensor,
    target: Tensor,
    iters: int,
    lower_bound: float,
    upper_bound: float | None,
) -> Tensor:
    matrix = matrix.float()
    target = target.float()
    try:
        solution = torch.linalg.lstsq(matrix, target.unsqueeze(1)).solution.squeeze(1)
    except RuntimeError:
        solution = torch.linalg.pinv(matrix) @ target
    solution = solution.clamp(min=lower_bound)
    if upper_bound is not None:
        solution = solution.clamp(max=upper_bound)
    if iters <= 0:
        return solution

    vector = getattr(_am_head_thread, "power", None)
    if vector is None:
        vector = torch.randn(matrix.shape[1], device=matrix.device, dtype=torch.float32)
    vector = vector / vector.norm().clamp_min(1e-12)
    for _ in range(5):
        vector = matrix.T @ (matrix @ vector)
        vector = vector / vector.norm().clamp_min(1e-12)
    lipschitz = (matrix @ vector).pow(2).sum().clamp_min(1e-6)
    step = 1.0 / lipschitz
    for _ in range(iters):
        grad = matrix.T @ (matrix @ solution - target)
        solution = (solution - step * grad).clamp(min=lower_bound)
        if upper_bound is not None:
            solution = solution.clamp(max=upper_bound)
    return solution


def _select_key_indices(attention_weights: Tensor, target_size: int, score_method: str) -> Tensor:
    if score_method != "rms":
        raise ValueError(f"unknown AM score method: {score_method}")
    scores = attention_weights.pow(2).mean(dim=0).sqrt()
    return torch.topk(scores, target_size, largest=True).indices.sort().values


def _zero_inactive_entries(key: Tensor, value: Tensor, beta: Tensor) -> Tensor:
    expected = tuple(key.shape[1:3])
    if tuple(value.shape) != tuple(key.shape) or tuple(beta.shape) != expected:
        raise ValueError(
            f"incompatible compact cache shapes: key={tuple(key.shape)}, "
            f"value={tuple(value.shape)}, beta={tuple(beta.shape)}"
        )
    inactive = torch.isneginf(beta)
    inactive_kv = inactive.unsqueeze(0).unsqueeze(-1)
    key.masked_fill_(inactive_kv, 0)
    value.masked_fill_(inactive_kv, 0)
    return inactive


def _zero_and_validate_inactive_entries(key: Tensor, value: Tensor, beta: Tensor) -> None:
    inactive = _zero_inactive_entries(key, value, beta)
    if not inactive.any():
        return
    inactive_kv = inactive.unsqueeze(0).unsqueeze(-1)
    if torch.count_nonzero(key.masked_select(inactive_kv)).item() != 0:
        raise RuntimeError("inactive compacted key entries were not zero")
    if torch.count_nonzero(value.masked_select(inactive_kv)).item() != 0:
        raise RuntimeError("inactive compacted value entries were not zero")


def attention_matching_cache_stats(cache, betas: list[Tensor]) -> dict[str, int]:
    active_entries = 0
    allocated_entries = 0
    active_per_head: list[int] = []
    for layer_idx, beta in enumerate(betas):
        key, value = cache_layer(cache, layer_idx)
        _zero_and_validate_inactive_entries(key, value, beta)
        active_entries += int(torch.isfinite(beta).sum().item())
        allocated_entries += beta.numel()
        active_per_head.extend(int(count) for count in torch.isfinite(beta).sum(dim=1).tolist())
    return {
        "active_entries": active_entries,
        "allocated_entries": allocated_entries,
        "active_kv_vectors": 2 * active_entries,
        "allocated_kv_vectors": 2 * allocated_entries,
        "min_active_entries_per_layer_head": min(active_per_head, default=0),
        "max_active_entries_per_layer_head": max(active_per_head, default=0),
    }


def compact_head_highest_attention(
    key: Tensor,
    value: Tensor,
    queries: Tensor,
    target_size: int,
    scaling: float,
    config: AttentionMatchingConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    seq_len = key.shape[0]
    if target_size <= 0:
        return key[:0], key.new_empty((0,)), value[:0], torch.empty(0, dtype=torch.long, device=key.device)
    if target_size >= seq_len:
        beta = key.new_zeros(seq_len)
        selected = torch.arange(seq_len, dtype=torch.long, device=key.device)
        return key, beta, value, selected

    dtype = key.dtype
    key32 = key.float()
    value32 = value.float()
    queries32 = queries.to(device=key.device, dtype=torch.float32)

    scores = (queries32 @ key32.T) * scaling
    scores = scores - scores.max(dim=1, keepdim=True).values
    exp_scores = scores.exp()
    attention_weights = exp_scores / exp_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

    selected = _select_key_indices(attention_weights, target_size, config.score_method)
    compact_key = key[selected]

    lower = float(torch.exp(torch.tensor(config.beta_min)))
    upper = float(torch.exp(torch.tensor(config.beta_max))) if config.beta_max is not None else None
    weights = _nnls_projected_gradient(
        exp_scores[:, selected],
        exp_scores.sum(dim=1),
        config.nnls_iters,
        lower,
        upper,
    )
    beta32 = weights.clamp_min(1e-12).log()

    target_output = attention_weights @ value32
    compact_scores = (queries32 @ compact_key.float().T) * scaling + beta32.unsqueeze(0)
    compact_weights = F.softmax(compact_scores, dim=-1, dtype=torch.float32)
    try:
        compact_value32 = torch.linalg.lstsq(compact_weights, target_output).solution
    except RuntimeError:
        compact_value32 = torch.linalg.pinv(compact_weights) @ target_output
    return compact_key, beta32.to(dtype=dtype), compact_value32.to(dtype=dtype), selected


def _causal_suffix_mask(
    batch_size: int,
    query_len: int,
    prefix_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    mask = torch.zeros((batch_size, 1, query_len, prefix_len + query_len), device=device, dtype=dtype)
    blocked = torch.triu(torch.ones((query_len, query_len), device=device, dtype=torch.bool), diagonal=1)
    min_value = torch.finfo(dtype).min
    mask[:, :, :, prefix_len:] = mask[:, :, :, prefix_len:].masked_fill(blocked, min_value)
    return mask


def _layer_cache(config, layer_idx: int, key: Tensor, value: Tensor) -> DynamicCache:
    cache = DynamicCache(config=config)
    cache.update(key, value, layer_idx)
    return cache


def _enforce_budget_bounds(budgets: Tensor, total_budget: int, seq_len: int, min_keys: int) -> Tensor:
    budgets = budgets.clamp(min=min_keys, max=seq_len)
    while int(budgets.sum().item()) > total_budget:
        reducible = (budgets > min_keys).nonzero(as_tuple=False).flatten()
        if reducible.numel() == 0:
            break
        index = reducible[torch.argmax(budgets[reducible])]
        budgets[index] -= 1
    while int(budgets.sum().item()) < total_budget:
        expandable = (budgets < seq_len).nonzero(as_tuple=False).flatten()
        if expandable.numel() == 0:
            break
        index = expandable[torch.argmin(budgets[expandable])]
        budgets[index] += 1
    return budgets


def _uniform_head_budgets(num_layers: int, kv_heads: int, seq_len: int, config: AttentionMatchingConfig) -> Tensor:
    head_count = num_layers * kv_heads
    total_budget = max(head_count * config.min_keys, int(round(seq_len * config.target_ratio * head_count)))
    per_head = max(config.min_keys, min(seq_len, int(round(total_budget / head_count))))
    budgets = torch.full((head_count,), per_head, dtype=torch.long)
    return _enforce_budget_bounds(budgets, total_budget, seq_len, config.min_keys).view(num_layers, kv_heads)


def _precomputed_head_budgets(num_layers: int, kv_heads: int, seq_len: int, config: AttentionMatchingConfig) -> Tensor:
    if config.head_budget_shares is None:
        raise ValueError("head_budget_mode='precomputed' requires head_budget_shares")
    shares = torch.tensor(config.head_budget_shares, dtype=torch.float32)
    if tuple(shares.shape) != (num_layers, kv_heads):
        raise ValueError(f"expected head budget shares shape {(num_layers, kv_heads)}, got {tuple(shares.shape)}")
    if (shares < 0).any() or float(shares.sum().item()) <= 0:
        raise ValueError("head budget shares must be nonnegative and have positive sum")

    flat_shares = shares.reshape(-1)
    budgeted_count = int((flat_shares > 0).sum().item())
    total_budget = max(budgeted_count * config.min_keys, int(round(seq_len * config.target_ratio * budgeted_count)))
    total_budget = min(seq_len * budgeted_count, total_budget)
    raw = flat_shares / flat_shares.sum() * total_budget
    budgets = torch.floor(raw).to(dtype=torch.long)
    remaining = total_budget - int(budgets.sum().item())
    if remaining > 0:
        order = torch.argsort(raw - budgets.float(), descending=True)
        budgets[order[:remaining]] += 1
    return _enforce_budget_bounds(budgets, total_budget, seq_len, config.min_keys).view(num_layers, kv_heads)


def _explicit_ratio_head_budgets(num_layers: int, kv_heads: int, seq_len: int, config: AttentionMatchingConfig) -> Tensor:
    if config.head_budget_ratios is None:
        raise ValueError("head_budget_mode='explicit_ratios' requires head_budget_ratios")
    ratios = torch.tensor(config.head_budget_ratios, dtype=torch.float32)
    if tuple(ratios.shape) != (num_layers, kv_heads):
        raise ValueError(f"expected head budget ratios shape {(num_layers, kv_heads)}, got {tuple(ratios.shape)}")
    if (ratios < 0).any() or (ratios > 1).any():
        raise ValueError("head budget ratios must be in [0, 1]")
    budgets = torch.round(ratios * seq_len).to(dtype=torch.long)
    return budgets.clamp(min=config.min_keys, max=seq_len)


def _head_budgets(
    model,
    full_cache,
    repeat_input_ids: Tensor,
    prefix_len: int,
    config: AttentionMatchingConfig,
) -> Tensor:
    base_model = _base_model(model)
    num_layers = len(base_model.layers)
    kv_heads = _num_kv_heads(base_model)
    if config.head_budget_mode == "uniform":
        return _uniform_head_budgets(num_layers, kv_heads, prefix_len, config)
    if config.head_budget_mode == "precomputed":
        return _precomputed_head_budgets(num_layers, kv_heads, prefix_len, config)
    if config.head_budget_mode == "explicit_ratios":
        return _explicit_ratio_head_budgets(num_layers, kv_heads, prefix_len, config)
    raise ValueError(f"unknown AM head budget mode: {config.head_budget_mode}")


def _project_reference_queries(
    base_model,
    layer_idx: int,
    hidden_states: Tensor,
    position_ids: Tensor,
) -> Tensor:
    layer = base_model.layers[layer_idx]
    attention = _attention_module(layer)
    hidden_states = layer.input_layernorm(hidden_states)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    q = attention.q_proj(hidden_states).view(hidden_shape)
    q_norm = getattr(attention, "q_norm", None)
    if q_norm is not None:
        q = q_norm(q)
    q = q.transpose(1, 2)
    q = _apply_rope(base_model, q, layer_idx, position_ids, inverse=False)

    num_query_heads = _num_query_heads(base_model)
    num_kv_heads = _num_kv_heads(base_model)
    query_groups = num_query_heads // num_kv_heads
    batch, _, seq_len, head_dim = q.shape
    q = q.view(batch, num_kv_heads, query_groups, seq_len, head_dim)
    return q.permute(1, 0, 2, 3, 4).reshape(num_kv_heads, batch * query_groups * seq_len, head_dim).detach()


def _reference_query_sample_indices(
    query_count: int,
    max_queries_per_kv_head: int,
    device: torch.device,
) -> Tensor | None:
    if max_queries_per_kv_head <= 0 or query_count <= max_queries_per_kv_head:
        return None
    return torch.randperm(query_count, device=device)[:max_queries_per_kv_head]


def _sample_reference_queries(
    queries: Tensor,
    max_queries_per_kv_head: int,
    sample_indices: Tensor | None = None,
) -> Tensor:
    if max_queries_per_kv_head <= 0 or queries.shape[1] <= max_queries_per_kv_head:
        return queries
    if sample_indices is None:
        sample_indices = _reference_query_sample_indices(
            queries.shape[1], max_queries_per_kv_head, queries.device
        )
    if sample_indices is None:
        return queries
    if int(sample_indices.max().item()) >= queries.shape[1]:
        raise ValueError(
            f"shared query sample index exceeds layer query count {queries.shape[1]}"
        )
    return queries[:, sample_indices.to(device=queries.device), :]


@contextmanager
def _am_eager_implementation(model):
    base_model = _base_model(model)
    previous_impl = getattr(base_model.config, "_attn_implementation", None)
    ALL_ATTENTION_FUNCTIONS.register("am_eager", _am_eager_attention_forward)
    base_model.config._attn_implementation = "am_eager"
    model.config._attn_implementation = "am_eager"
    try:
        yield
    finally:
        base_model.config._attn_implementation = previous_impl
        model.config._attn_implementation = previous_impl


def _advance_through_compacted_layer(
    model,
    hidden_states: Tensor,
    repeat_input_ids: Tensor,
    position_ids: Tensor,
    position_embeddings,
    layer_idx: int,
    compact_layer: tuple[Tensor, Tensor],
    beta: Tensor,
    query_context_cache=None,
    query_context_chunk_start: int = 0,
    query_context_chunk_end: int | None = None,
) -> Tensor:
    base_model = _base_model(model)
    layer = base_model.layers[layer_idx]
    attention = _attention_module(layer)
    key, value = compact_layer
    layer_beta = beta
    if query_context_cache is not None:
        context_key, context_value = cache_layer(query_context_cache, layer_idx)
        chunk_end = query_context_chunk_end
        if chunk_end is None:
            chunk_end = context_key.shape[2]
        key = torch.cat(
            [
                context_key[:, :, :query_context_chunk_start, :],
                key,
                context_key[:, :, chunk_end:, :],
            ],
            dim=2,
        )
        value = torch.cat(
            [
                context_value[:, :, :query_context_chunk_start, :],
                value,
                context_value[:, :, chunk_end:, :],
            ],
            dim=2,
        )
        layer_beta = torch.cat(
            [
                layer_beta.new_zeros((layer_beta.shape[0], query_context_chunk_start)),
                layer_beta,
                layer_beta.new_zeros((layer_beta.shape[0], context_key.shape[2] - chunk_end)),
            ],
            dim=1,
        )
    attention._am_beta = layer_beta
    try:
        layer_cache = _layer_cache(base_model.config, layer_idx, key, value)
        mask = _causal_suffix_mask(
            repeat_input_ids.shape[0],
            repeat_input_ids.shape[1],
            key.shape[2],
            repeat_input_ids.device,
            hidden_states.dtype,
        )
        hidden_states = layer(
            hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=layer_cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
    finally:
        if hasattr(attention, "_am_beta"):
            delattr(attention, "_am_beta")
    return hidden_states


def compact_cache_attention_matching_on_policy(
    model,
    full_cache,
    repeat_input_ids: Tensor,
    prefix_len: int,
    config: AttentionMatchingConfig,
    *,
    logical_prefix_len: int | None = None,
    query_context_cache=None,
    query_context_chunk_start: int = 0,
    query_context_chunk_end: int | None = None,
    source_position_offset: int = 0,
) -> tuple[DynamicCache, list[Tensor], int, int, list[Tensor]]:
    base_model = _base_model(model)
    compact_layers: list[tuple[Tensor, Tensor] | None] = [None] * len(base_model.layers)
    betas: list[Tensor | None] = [None] * len(base_model.layers)
    compact_positions: list[Tensor | None] = [None] * len(base_model.layers)
    budgets = _head_budgets(model, full_cache, repeat_input_ids, prefix_len, config)
    budget_rows: list[list[int]] = budgets.tolist()
    effective_entries = sum(sum(row) for row in budget_rows)
    logical_prefix_len = prefix_len if logical_prefix_len is None else logical_prefix_len
    position_ids = torch.arange(
        logical_prefix_len,
        logical_prefix_len + repeat_input_ids.shape[1],
        dtype=torch.long,
        device=repeat_input_ids.device,
    ).unsqueeze(0)
    shared_query_sample: Tensor | None = None
    hidden_states = base_model.embed_tokens(repeat_input_ids)
    position_embeddings = base_model.rotary_emb(hidden_states, position_ids)

    with _am_eager_implementation(model):
        for layer_idx, layer in enumerate(base_model.layers):
            layer_queries = _project_reference_queries(base_model, layer_idx, hidden_states, position_ids)
            if layer_idx == 0:
                shared_query_sample = _reference_query_sample_indices(
                    layer_queries.shape[1], config.query_samples, layer_queries.device
                )
            layer_queries = _sample_reference_queries(
                layer_queries, config.query_samples, shared_query_sample
            )

            key, value = cache_layer(full_cache, layer_idx)
            if key.shape[0] != 1:
                raise ValueError("attention matching eval currently supports batch size 1")
            _, kv_heads, seq_len, _ = key.shape
            if seq_len != prefix_len:
                raise ValueError(f"full cache length {seq_len} does not match prefix length {prefix_len}")

            attention = _attention_module(layer)
            scaling = float(getattr(attention, "scaling", key.shape[-1] ** -0.5))
            compact_keys = []
            compact_values = []
            compact_betas = []
            head_positions = []
            layer_budgets = budget_rows[layer_idx]
            physical_len = max(layer_budgets)
            head_results = None
            if config.compaction_workers != 1:
                from multiagent.model.am_parallel_heads import compact_layer
                head_results = compact_layer(key[0], value[0], layer_queries.to(device=key.device), layer_budgets, scaling, config)
            for kv_head, target_size in enumerate(layer_budgets):
                if head_results is not None:
                    compact_key, beta, compact_value, selected = head_results[kv_head]
                else:
                    compact_key, beta, compact_value, selected = compact_head_highest_attention(
                        key[0, kv_head],
                        value[0, kv_head],
                        layer_queries[kv_head].to(device=key.device),
                        target_size,
                        scaling,
                        config,
                    )
                selected = selected + source_position_offset
                if compact_key.shape[0] < physical_len:
                    pad_len = physical_len - compact_key.shape[0]
                    compact_key = F.pad(compact_key, (0, 0, 0, pad_len))
                    compact_value = F.pad(compact_value, (0, 0, 0, pad_len))
                    beta = F.pad(beta, (0, pad_len), value=float("-inf"))
                    selected = F.pad(selected, (0, pad_len))
                compact_keys.append(compact_key)
                compact_values.append(compact_value)
                compact_betas.append(beta)
                head_positions.append(selected)

            layer_key = torch.stack(compact_keys, dim=0).unsqueeze(0)
            layer_value = torch.stack(compact_values, dim=0).unsqueeze(0)
            layer_beta = torch.stack(compact_betas, dim=0)
            if any(target_size < physical_len for target_size in layer_budgets):
                _zero_inactive_entries(layer_key, layer_value, layer_beta)
            compact_layer = (layer_key, layer_value)
            compact_layers[layer_idx] = compact_layer
            betas[layer_idx] = layer_beta
            compact_positions[layer_idx] = torch.stack(head_positions, dim=0).unsqueeze(0)

            if layer_idx + 1 < len(base_model.layers):
                hidden_states = _advance_through_compacted_layer(
                    model,
                    hidden_states,
                    repeat_input_ids,
                    position_ids,
                    position_embeddings,
                    layer_idx,
                    compact_layer,
                    layer_beta,
                    query_context_cache=query_context_cache,
                    query_context_chunk_start=query_context_chunk_start,
                    query_context_chunk_end=query_context_chunk_end,
                )

    compact_cache = DynamicCache(config=base_model.config)
    final_betas = []
    final_positions = []
    for layer_idx, compact_layer in enumerate(compact_layers):
        if compact_layer is None or betas[layer_idx] is None or compact_positions[layer_idx] is None:
            raise ValueError(f"layer {layer_idx} was not compacted")
        key, value = compact_layer
        compact_cache.update(key, value, layer_idx)
        final_betas.append(betas[layer_idx])
        final_positions.append(compact_positions[layer_idx])
    return compact_cache, final_betas, logical_prefix_len, effective_entries, final_positions


def compact_cache_attention_matching_from_queries(
    model,
    full_cache,
    layer_queries: list[Tensor],
    prefix_len: int,
    config: AttentionMatchingConfig,
    *,
    logical_prefix_len: int | None = None,
    source_position_offset: int = 0,
) -> tuple[DynamicCache, list[Tensor], int, int, list[Tensor]]:
    base_model = _base_model(model)
    num_layers = len(base_model.layers)
    kv_heads = _num_kv_heads(base_model)
    if len(layer_queries) != num_layers:
        raise ValueError(f"expected {num_layers} query layers, got {len(layer_queries)}")
    reference_device = cache_layer(full_cache, 0)[0].device
    budgets = _head_budgets(
        model,
        full_cache,
        torch.empty((1, 0), dtype=torch.long, device=reference_device),
        prefix_len,
        config,
    )
    budget_rows: list[list[int]] = budgets.tolist()
    effective_entries = sum(sum(row) for row in budget_rows)
    logical_prefix_len = prefix_len if logical_prefix_len is None else logical_prefix_len
    compact_cache = DynamicCache(config=base_model.config)
    final_betas = []
    final_positions = []
    shared_query_sample = _reference_query_sample_indices(
        layer_queries[0].shape[1], config.query_samples, layer_queries[0].device
    )

    parallel_futures = None
    if config.compaction_workers != 1 and reference_device.type == "cuda":
        from multiagent.model.am_parallel_heads import submit_layers
        parallel_futures = submit_layers(model, full_cache, layer_queries, budget_rows,
                                         prefix_len, config, shared_query_sample)

    for layer_idx, layer in enumerate(base_model.layers):
        key, value = cache_layer(full_cache, layer_idx)
        if key.shape[0] != 1 or key.shape[1] != kv_heads:
            raise ValueError(
                f"unexpected full cache shape at layer {layer_idx}: {tuple(key.shape)}"
            )
        if key.shape[2] != prefix_len:
            window_len = key.shape[2]
            compact_cache.update(key, value, layer_idx)
            final_betas.append(key.new_zeros((kv_heads, window_len)))
            final_positions.append(
                (torch.arange(prefix_len - window_len, prefix_len, device=key.device)
                 + source_position_offset).view(1, 1, -1).expand(1, kv_heads, -1)
            )
            continue
        queries = _sample_reference_queries(
            layer_queries[layer_idx].to(device=key.device),
            config.query_samples,
            shared_query_sample,
        )
        if queries.shape[0] != kv_heads:
            raise ValueError(
                f"expected {kv_heads} KV-head query groups, got {queries.shape[0]}"
            )
        attention = _attention_module(layer)
        scaling = float(getattr(attention, "scaling", key.shape[-1] ** -0.5))
        layer_budgets = budget_rows[layer_idx]
        physical_len = max(layer_budgets)
        compact_keys = []
        compact_values = []
        compact_betas = []
        head_positions = []
        head_results = None
        if parallel_futures is not None:
            from multiagent.model.am_parallel_heads import resolve_layer
            head_results = resolve_layer(parallel_futures[layer_idx])
        elif config.compaction_workers != 1:
            from multiagent.model.am_parallel_heads import compact_layer
            head_results = compact_layer(key[0], value[0], queries, layer_budgets, scaling, config)
        for kv_head, target_size in enumerate(layer_budgets):
            if head_results is not None:
                compact_key, beta, compact_value, selected = head_results[kv_head]
            else:
                compact_key, beta, compact_value, selected = compact_head_highest_attention(
                    key[0, kv_head],
                    value[0, kv_head],
                    queries[kv_head],
                    target_size,
                    scaling,
                    config,
                )
            selected = selected + source_position_offset
            if compact_key.shape[0] < physical_len:
                pad_len = physical_len - compact_key.shape[0]
                compact_key = F.pad(compact_key, (0, 0, 0, pad_len))
                compact_value = F.pad(compact_value, (0, 0, 0, pad_len))
                beta = F.pad(beta, (0, pad_len), value=float("-inf"))
                selected = F.pad(selected, (0, pad_len))
            compact_keys.append(compact_key)
            compact_values.append(compact_value)
            compact_betas.append(beta)
            head_positions.append(selected)
        layer_key = torch.stack(compact_keys).unsqueeze(0)
        layer_value = torch.stack(compact_values).unsqueeze(0)
        layer_beta = torch.stack(compact_betas)
        if any(target_size < physical_len for target_size in layer_budgets):
            _zero_inactive_entries(layer_key, layer_value, layer_beta)
        compact_cache.update(layer_key, layer_value, layer_idx)
        final_betas.append(layer_beta)
        final_positions.append(torch.stack(head_positions).unsqueeze(0))
    return compact_cache, final_betas, logical_prefix_len, effective_entries, final_positions


def compact_cache_attention_matching_chunked_from_queries(
    model,
    full_cache,
    layer_queries: list[Tensor],
    article_start: int,
    article_end: int,
    chunk_size: int,
    config: AttentionMatchingConfig,
) -> tuple[DynamicCache, list[Tensor], int, int, list[Tensor]]:
    base_model = _base_model(model)
    full_len = max(
        cache_layer(full_cache, layer_idx)[0].shape[2]
        for layer_idx in range(len(base_model.layers))
    )
    if not 0 <= article_start < article_end <= full_len:
        raise ValueError(
            f"invalid article range [{article_start}, {article_end}) for length {full_len}"
        )
    compacted_chunks = []
    chunk_betas = []
    chunk_positions = []
    effective_entries = 0
    for chunk_start in range(article_start, article_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, article_end)
        source_cache = _cache_slice(model, full_cache, chunk_start, chunk_end)
        compacted, betas, _, entries, positions = compact_cache_attention_matching_from_queries(
            model,
            source_cache,
            layer_queries,
            chunk_end - chunk_start,
            config,
            logical_prefix_len=full_len,
            source_position_offset=chunk_start,
        )
        compacted_chunks.append(compacted)
        chunk_betas.append(betas)
        chunk_positions.append(positions)
        effective_entries += entries

    kept_len = article_start + full_len - article_end
    combined_cache = DynamicCache(config=base_model.config)
    combined_betas = []
    combined_positions = []
    for layer_idx in range(len(base_model.layers)):
        full_key, full_value = cache_layer(full_cache, layer_idx)
        if full_key.shape[2] != full_len:
            window_len = full_key.shape[2]
            combined_cache.update(full_key, full_value, layer_idx)
            combined_betas.append(full_key.new_zeros((full_key.shape[1], window_len)))
            combined_positions.append(
                torch.arange(full_len - window_len, full_len, device=full_key.device)
                .view(1, 1, -1).expand(1, full_key.shape[1], -1)
            )
            effective_entries += window_len * full_key.shape[1]
            continue
        effective_entries += kept_len * _num_kv_heads(base_model)
        key_parts = [full_key[:, :, :article_start]]
        value_parts = [full_value[:, :, :article_start]]
        beta_parts = [full_key.new_zeros((full_key.shape[1], article_start))]
        position_parts = [
            torch.arange(article_start, device=full_key.device)
            .view(1, 1, -1)
            .expand(1, full_key.shape[1], -1)
        ]
        for compacted, betas, positions in zip(
            compacted_chunks,
            chunk_betas,
            chunk_positions,
        ):
            chunk_key, chunk_value = cache_layer(compacted, layer_idx)
            key_parts.append(chunk_key)
            value_parts.append(chunk_value)
            beta_parts.append(betas[layer_idx])
            position_parts.append(positions[layer_idx])
        key_parts.append(full_key[:, :, article_end:])
        value_parts.append(full_value[:, :, article_end:])
        beta_parts.append(full_key.new_zeros((full_key.shape[1], full_len - article_end)))
        position_parts.append(
            torch.arange(article_end, full_len, device=full_key.device)
            .view(1, 1, -1)
            .expand(1, full_key.shape[1], -1)
        )
        combined_key = torch.cat(key_parts, dim=2)
        combined_value = torch.cat(value_parts, dim=2)
        combined_beta = torch.cat(beta_parts, dim=1)
        combined_position = torch.cat(position_parts, dim=2)
        _zero_and_validate_inactive_entries(
            combined_key,
            combined_value,
            combined_beta,
        )
        combined_cache.update(combined_key, combined_value, layer_idx)
        combined_betas.append(combined_beta)
        combined_positions.append(combined_position)
    return combined_cache, combined_betas, full_len, effective_entries, combined_positions


def _cache_slice(model, cache, start: int, end: int) -> DynamicCache:
    base_model = _base_model(model)
    full_len = max(
        cache_layer(cache, layer_idx)[0].shape[2]
        for layer_idx in range(len(base_model.layers))
    )
    sliced = DynamicCache(config=base_model.config)
    for layer_idx in range(len(base_model.layers)):
        key, value = cache_layer(cache, layer_idx)
        if key.shape[2] != full_len:
            sliced.update(key[:, :, 0:0, :], value[:, :, 0:0, :], layer_idx)
            continue
        sliced.update(key[:, :, start:end, :], value[:, :, start:end, :], layer_idx)
    return sliced


def _cache_without_other_article_chunks(
    model,
    cache,
    article_start: int,
    chunk_start: int,
    chunk_end: int,
    article_end: int,
) -> DynamicCache:
    base_model = _base_model(model)
    output = DynamicCache(config=base_model.config)
    for layer_idx in range(len(base_model.layers)):
        key, value = cache_layer(cache, layer_idx)
        query_key = torch.cat(
            [key[:, :, :article_start, :], key[:, :, chunk_start:chunk_end, :], key[:, :, article_end:, :]],
            dim=2,
        )
        query_value = torch.cat(
            [value[:, :, :article_start, :], value[:, :, chunk_start:chunk_end, :], value[:, :, article_end:, :]],
            dim=2,
        )
        output.update(query_key, query_value, layer_idx)
    return output


def compact_cache_attention_matching_chunked(
    model,
    full_cache,
    sender_input_ids: Tensor,
    repeat_instruction_ids: Tensor,
    article_start: int,
    article_end: int,
    chunk_size: int,
    config: AttentionMatchingConfig,
) -> tuple[DynamicCache, list[Tensor], int, int, list[Tensor]]:
    if sender_input_ids.shape[0] != 1:
        raise ValueError("chunked attention matching currently supports batch size 1")
    if chunk_size <= 0:
        raise ValueError("AM chunk size must be positive")

    base_model = _base_model(model)
    full_len = sender_input_ids.shape[1]
    if not 0 <= article_start < article_end <= full_len:
        raise ValueError(
            f"invalid article token range [{article_start}, {article_end}) for prefix length {full_len}"
        )

    compacted_chunks: list[DynamicCache] = []
    chunk_betas: list[list[Tensor]] = []
    chunk_positions: list[list[Tensor]] = []
    effective_entries = 0
    query_chunk_start = article_start

    for chunk_start in range(article_start, article_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, article_end)
        chunk_len = chunk_end - chunk_start
        source_cache = _cache_slice(model, full_cache, chunk_start, chunk_end)
        query_context_cache = _cache_without_other_article_chunks(
            model,
            full_cache,
            article_start,
            chunk_start,
            chunk_end,
            article_end,
        )
        repeat_input_ids = torch.cat(
            [repeat_instruction_ids, sender_input_ids[:, chunk_start:chunk_end]],
            dim=1,
        )
        compacted, betas, _, chunk_entries, positions = compact_cache_attention_matching_on_policy(
            model,
            source_cache,
            repeat_input_ids,
            chunk_len,
            config,
            logical_prefix_len=full_len,
            query_context_cache=query_context_cache,
            query_context_chunk_start=query_chunk_start,
            query_context_chunk_end=query_chunk_start + chunk_len,
            source_position_offset=chunk_start,
        )
        compacted_chunks.append(compacted)
        chunk_betas.append(betas)
        chunk_positions.append(positions)
        effective_entries += chunk_entries

    kept_len = article_start + (full_len - article_end)
    effective_entries += kept_len * len(base_model.layers) * _num_kv_heads(base_model)

    combined_cache = DynamicCache(config=base_model.config)
    combined_betas: list[Tensor] = []
    combined_positions: list[Tensor] = []
    for layer_idx in range(len(base_model.layers)):
        full_key, full_value = cache_layer(full_cache, layer_idx)
        key_parts = [full_key[:, :, :article_start, :]]
        value_parts = [full_value[:, :, :article_start, :]]
        beta_parts = [full_key.new_zeros((full_key.shape[1], article_start))]
        position_parts = [
            torch.arange(article_start, dtype=torch.long, device=full_key.device)
            .view(1, 1, -1)
            .expand(1, full_key.shape[1], -1)
        ]
        for compacted, betas, positions in zip(compacted_chunks, chunk_betas, chunk_positions):
            chunk_key, chunk_value = cache_layer(compacted, layer_idx)
            key_parts.append(chunk_key)
            value_parts.append(chunk_value)
            beta_parts.append(betas[layer_idx])
            position_parts.append(positions[layer_idx])
        key_parts.append(full_key[:, :, article_end:, :])
        value_parts.append(full_value[:, :, article_end:, :])
        beta_parts.append(full_key.new_zeros((full_key.shape[1], full_len - article_end)))
        position_parts.append(
            torch.arange(article_end, full_len, dtype=torch.long, device=full_key.device)
            .view(1, 1, -1)
            .expand(1, full_key.shape[1], -1)
        )

        combined_key = torch.cat(key_parts, dim=2)
        combined_value = torch.cat(value_parts, dim=2)
        combined_beta = torch.cat(beta_parts, dim=1)
        combined_position = torch.cat(position_parts, dim=2)
        _zero_and_validate_inactive_entries(combined_key, combined_value, combined_beta)
        combined_cache.update(combined_key, combined_value, layer_idx)
        combined_betas.append(combined_beta)
        combined_positions.append(combined_position)

    return combined_cache, combined_betas, full_len, effective_entries, combined_positions


def _am_eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
):
    groups = int(getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1]))
    key_states = key.repeat_interleave(groups, dim=1)
    value_states = value.repeat_interleave(groups, dim=1)

    if attention_mask is not None:
        key_len = key_states.shape[-2]
        mask_len = attention_mask.shape[-1]
        if mask_len > key_len:
            attention_mask = attention_mask[..., -key_len:]
        elif mask_len < key_len:
            attention_mask = F.pad(attention_mask, (key_len - mask_len, 0))

    beta = getattr(module, "_am_beta", None)
    if beta is not None:
        beta = beta.to(device=query.device, dtype=query.dtype)
        beta = beta.repeat_interleave(groups, dim=0)
        if key_states.shape[-2] > beta.shape[-1]:
            beta = F.pad(beta, (0, key_states.shape[-2] - beta.shape[-1]))
        else:
            beta = beta[:, : key_states.shape[-2]]

    query_block_size = 256
    outputs = []
    transposed_keys = key_states.transpose(2, 3)
    for start in range(0, query.shape[-2], query_block_size):
        end = min(query.shape[-2], start + query_block_size)
        attn_weights = torch.matmul(query[:, :, start:end], transposed_keys) * scaling
        if attention_mask is not None:
            block_mask = attention_mask
            if block_mask.shape[-2] != 1:
                block_mask = block_mask[..., start:end, :]
            attn_weights = attn_weights + block_mask
        if beta is not None:
            attn_weights = attn_weights + beta.view(1, beta.shape[0], 1, beta.shape[1])
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        outputs.append(torch.matmul(attn_weights, value_states))

    attn_output = torch.cat(outputs, dim=2)
    return attn_output.transpose(1, 2).contiguous(), None


@contextmanager
def beta_attention(model, betas: list[Tensor], backend: str = "eager"):
    base_model = _base_model(model)
    previous_impl = getattr(base_model.config, "_attn_implementation", None)
    if backend != "eager":
        raise ValueError(f"Unknown AM attention backend: {backend}")
    forward = _am_eager_attention_forward
    implementation = "am_" + backend
    ALL_ATTENTION_FUNCTIONS.register(implementation, forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(implementation, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    base_model.config._attn_implementation = implementation
    model.config._attn_implementation = implementation
    attentions = []
    for layer_idx, layer in enumerate(base_model.layers):
        attention = _attention_module(layer)
        attentions.append(attention)
        attention._am_beta = betas[layer_idx]
    try:
        yield
    finally:
        for attention in attentions:
            if hasattr(attention, "_am_beta"):
                delattr(attention, "_am_beta")
            if hasattr(attention, "_am_sdpa_bias"):
                delattr(attention, "_am_sdpa_bias")
        base_model.config._attn_implementation = previous_impl
        model.config._attn_implementation = previous_impl
