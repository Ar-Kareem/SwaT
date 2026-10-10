from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers.cache_utils import DynamicCache

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c import apply_chat_template, chat_prompt_ids, locate_article
from multiagent.dataset.c2c_cross_family import encode_context_pair, tokenize_with_offsets
from multiagent.dataset.c2c_registry import load_c2c_task
from multiagent.dataset.c2c_types import C2CExample
from multiagent.evaluation.native_am import generate_from_beta_cache
from multiagent.evaluation.renderer_am import decode_renderer_cache
from multiagent.evaluation.results import (
    EvalJsonCompactor,
    compact_eval_sample,
    elapsed_metadata,
    eval_json_format_metadata,
    evaluation_config_metadata,
    normalize_grade_result,
    print_eval_sample,
    save_json,
    score_postfix,
    summarize,
    update_metric_stats,
)
from multiagent.evaluation.utils import (
    BoundedContextCache,
    CompactGridAssembler,
    CompressionJournal,
    WINDOW_RESERVE,
    _lookup_rope_positions,
    _resolve_pack_mode,
    add_am_args,
    add_eval_args,
    add_renderer_args,
    am_metadata,
    deterministic_am_context_seed,
    dtype_from_args,
    interleaved_items,
    load_model,
    load_tokenizer,
    make_am_config,
    renderer_am_token_counts,
    seed_attention_matching_context,
    synchronized_time,
    validate_am_args,
    validate_eval_args,
)
from multiagent.model.attention_matching import (
    AttentionMatchingConfig,
    _cache_slice,
    _num_kv_heads,
    attention_matching_cache_stats,
    compact_cache_attention_matching_from_queries,
    compact_cache_attention_matching_on_policy,
    repeat_prefill_prefix_ids,
)
from multiagent.model.cache_renderer import flatten_sender_cache, pred_to_cache
from multiagent.model.query_transfer import collect_normalized_queries, load_query_transfer
from multiagent.model.unaware_comm import _apply_rope, _base_model, cache_layer, full_sender_cache
from multiagent.model.vanilla_loader import resolve_model_path


REPEAT_INSTRUCTION = "\nRepeat the previous context.\n"

def _article_sender_token_span(sender_tokenizer, sender_text: str, article_text: str) -> tuple[int, int] | None:
    if not article_text:
        return None
    article_char_start = sender_text.find(article_text)
    if article_char_start < 0:
        article_text = article_text.strip()
        article_char_start = sender_text.find(article_text) if article_text else -1
    if article_char_start < 0:
        return None
    start = len(sender_tokenizer(sender_text[:article_char_start], add_special_tokens=False)["input_ids"])
    end = len(
        sender_tokenizer(
            sender_text[: article_char_start + len(article_text)],
            add_special_tokens=False,
        )["input_ids"]
    )
    return start, end


def _chunk_article_text(sender_tokenizer, article_text: str, chunk_tokens: int, max_tokens: int) -> list[str]:
    ids, offsets = tokenize_with_offsets(sender_tokenizer, article_text)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
        offsets = offsets[:max_tokens]
    chunks = []
    for start in range(0, len(ids), chunk_tokens):
        end = min(start + chunk_tokens, len(ids))
        chunk = article_text[offsets[start][0] : offsets[end - 1][1]]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _messages_with_chunk(messages, article_text: str, chunk_text: str) -> list[dict[str, str]]:
    replaced = False
    output = []
    for message in messages:
        content = message.get("content", "")
        if not replaced and article_text and article_text in content:
            output.append({**message, "content": content.replace(article_text, chunk_text, 1)})
            replaced = True
        else:
            output.append(dict(message))
    if not replaced:
        raise ValueError("article text was not found in the sender messages")
    return output


@torch.no_grad()
def prepare_beyond_context_cache_native(
    model,
    tokenizer,
    sample: C2CExample,
    device: str,
    max_length: int,
    chunk_tokens: int,
    am_config: AttentionMatchingConfig,
    pack: str = "clamp",
    packing_window: int | None = None,
    position_reserve: int = WINDOW_RESERVE,
) -> tuple[DynamicCache, list[Tensor], int, int, dict[str, int]] | None:
    article_text = str(sample.metadata.get("context", ""))
    if not article_text:
        raise ValueError("beyond-context native mode requires sample.metadata['context']")
    chunk_texts = _chunk_article_text(tokenizer, article_text, chunk_tokens, max_length)
    if not chunk_texts:
        return None
    article_tokens = sum(
        len(tokenizer(chunk, add_special_tokens=False)["input_ids"]) for chunk in chunk_texts
    )
    am_config, span_cap = _resolve_pack_mode(
        pack,
        am_config,
        model,
        article_tokens,
        packing_window=packing_window,
        position_reserve=position_reserve,
    )
    instruction_ids = repeat_prefill_prefix_ids(tokenizer)
    base_model = _base_model(model)
    num_layers = len(base_model.layers)
    kv_heads = _num_kv_heads(base_model)
    window = int(base_model.config.max_position_embeddings)
    assembler = CompactGridAssembler(base_model, kv_heads, span_cap=span_cap)

    compact_entries = 0
    total_article_tokens = 0
    pending_suffix = None
    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk_messages = _messages_with_chunk(sample.sender_messages, article_text, chunk_text)
        chunk_context_text = apply_chat_template(tokenizer, chunk_messages, add_generation_prompt=False)
        context_ids = tokenizer(chunk_context_text, add_special_tokens=False)["input_ids"]
        if not context_ids:
            continue
        if len(context_ids) > window:
            raise ValueError(
                f"chunk {chunk_index} context has {len(context_ids)} tokens, beyond the "
                f"{window}-position window; lower --bc-chunk-tokens"
            )
        span = _article_sender_token_span(tokenizer, chunk_context_text, chunk_text)
        if span is None:
            raise ValueError(f"chunk {chunk_index} text was not found in its chunk context")
        span_start = max(0, min(span[0], len(context_ids)))
        span_end = max(span_start, min(span[1], len(context_ids)))
        if span_end <= span_start:
            continue
        total_article_tokens += span_end - span_start

        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        positions = torch.arange(len(context_ids), dtype=torch.long, device=device).unsqueeze(0)
        chunk_cache = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=positions,
            use_cache=True,
        ).past_key_values

        if chunk_index == 0:
            assembler.add_keep(
                [
                    (key[:, :, :span_start].clone(), value[:, :, :span_start].clone())
                    for key, value in (
                        cache_layer(chunk_cache, layer_idx) for layer_idx in range(num_layers)
                    )
                ],
                positions[0, :span_start].to(torch.float32),
            )
            compact_entries += span_start * num_layers * kv_heads

        source_cache = _cache_slice(model, chunk_cache, span_start, span_end)
        repeat_input_ids = torch.tensor(
            [instruction_ids + context_ids[span_start:span_end]], dtype=torch.long, device=device
        )
        compacted, betas, _, entries, index_positions = compact_cache_attention_matching_on_policy(
            model,
            source_cache,
            repeat_input_ids,
            span_end - span_start,
            am_config,
            logical_prefix_len=len(context_ids),
            query_context_cache=chunk_cache,
            query_context_chunk_start=span_start,
            query_context_chunk_end=span_end,
            source_position_offset=span_start,
        )
        compact_entries += entries
        assembler.add_repack(
            [cache_layer(compacted, layer_idx) for layer_idx in range(num_layers)],
            betas,
            [indices.to(torch.float32) for indices in index_positions],
        )

        suffix_len = len(context_ids) - span_end
        if suffix_len:
            pending_suffix = (
                [
                    (key[:, :, span_end:].clone(), value[:, :, span_end:].clone())
                    for key, value in (
                        cache_layer(chunk_cache, layer_idx) for layer_idx in range(num_layers)
                    )
                ],
                [
                    positions[0, span_end:].to(torch.float32).view(1, 1, -1).expand(1, kv_heads, -1)
                ] * num_layers,
                suffix_len,
            )
        del chunk_cache, source_cache, compacted

    if pending_suffix is not None:
        suffix_caches, suffix_positions, suffix_len = pending_suffix
        suffix_betas = [
            torch.zeros(
                (kv_heads, suffix_len),
                device=device,
                dtype=suffix_caches[0][0].dtype,
            )
        ] * num_layers
        assembler.add_repack(suffix_caches, suffix_betas, suffix_positions)
        compact_entries += suffix_len * num_layers * kv_heads

    cache, final_betas, decode_start = assembler.assemble()
    cache_stats = attention_matching_cache_stats(cache, final_betas)
    print(
        f"beyond-context native: {len(chunk_texts)} chunks / {total_article_tokens} article "
        f"tokens self-prefilled at <={chunk_tokens} local positions -> compact decode start {decode_start}"
    )
    return cache, final_betas, decode_start, compact_entries, cache_stats


@torch.no_grad()
def generate_beyond_context_native(
    model,
    tokenizer,
    sample: C2CExample,
    device: str,
    max_length: int,
    chunk_tokens: int,
    max_new_tokens: int,
    am_config: AttentionMatchingConfig,
    decode_fn: Callable,
    min_new_tokens: int = 0,
    context_cache: dict[str, Any] | None = None,
    pack: str = "clamp",
    packing_window: int | None = None,
) -> tuple[str, str, int]:
    receiver_decode_ids = chat_prompt_ids(
        tokenizer,
        sample.receiver_messages,
        max_length,
        add_generation_prompt=True,
    )
    if not receiver_decode_ids:
        return "", "empty_prompt", 0
    context_id = str(sample.metadata.get("context_id", ""))
    position_reserve = max(WINDOW_RESERVE, len(receiver_decode_ids) + max_new_tokens)
    cache_id = (
        f"{context_id}|position_reserve={position_reserve}"
        if context_id and pack == "fractional"
        else context_id
    )
    prepared = context_cache.get(cache_id) if context_cache is not None and cache_id else None
    if prepared is None:
        prepared = prepare_beyond_context_cache_native(
            model,
            tokenizer,
            sample,
            device,
            max_length,
            chunk_tokens,
            am_config,
            pack=pack,
            packing_window=packing_window,
            position_reserve=position_reserve,
        )
        if prepared is not None and context_cache is not None and cache_id:
            context_cache.clear()
            context_cache[cache_id] = prepared
    if prepared is None:
        return "", "empty_prompt", 0
    base_cache, betas, decode_start, compact_entries, cache_stats = prepared

    if pack == "fractional":
        window = (
            int(packing_window)
            if packing_window is not None
            else int(_base_model(model).config.max_position_embeddings)
        )
        projected_end = decode_start + len(receiver_decode_ids) + max_new_tokens
        if projected_end > window:
            raise RuntimeError(
                f"fractional independent AM exceeds window: decode {decode_start} + "
                f"receiver {len(receiver_decode_ids)} + generation {max_new_tokens} = "
                f"{projected_end} > {window}"
            )

    cache = copy.deepcopy(base_cache) if context_cache is not None and cache_id else base_cache
    text, stop_reason, generated_tokens = decode_fn(
        model,
        tokenizer,
        cache,
        betas,
        receiver_decode_ids,
        decode_start,
        device,
        max_new_tokens,
        min_new_tokens,
    )
    return (
        text,
        f"{stop_reason}:bc-native-{pack}+am{compact_entries}active/"
        f"{cache_stats['allocated_entries']}allocated/decode@{decode_start}"
        f"+recv{len(receiver_decode_ids)}",
        generated_tokens,
    )


def _prompt_article_span(tokenizer, text: str, article: str) -> tuple[int, int, str]:
    char_start, matched, source = locate_article(text, article)
    if char_start < 0:
        raise ValueError(
            f"could not locate the {len(article)}-character article in a "
            f"{len(text)}-character templated prompt"
        )
    start = len(tokenizer(text[:char_start], add_special_tokens=False)["input_ids"])
    end = len(
        tokenizer(text[: char_start + len(matched)], add_special_tokens=False)["input_ids"]
    )
    return start, end, source


def _aligned_renderer_chunks(
    sender_start: int,
    sender_end: int,
    receiver_start: int,
    receiver_end: int,
    cross_causal_limits,
    chunk_tokens: int,
) -> list[tuple[int, int, int, int]]:
    if chunk_tokens <= 0:
        raise ValueError("renderer hybrid chunk_tokens must be positive")
    if sender_end <= sender_start or receiver_end <= receiver_start:
        raise ValueError("renderer hybrid article spans must be non-empty")
    limits = torch.as_tensor(cross_causal_limits, dtype=torch.long)
    chunks: list[tuple[int, int, int, int]] = []
    sender_cursor = sender_start
    receiver_cursor = receiver_start
    while sender_cursor < sender_end:
        sender_next = min(sender_cursor + chunk_tokens, sender_end)
        if sender_next == sender_end:
            receiver_next = receiver_end
        else:
            candidates = torch.nonzero(
                limits[receiver_cursor:receiver_end] >= sender_next,
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                raise ValueError(
                    f"no receiver boundary aligns with sender boundary {sender_next}"
                )
            receiver_next = receiver_cursor + int(candidates[0].item()) + 1
        if receiver_next <= receiver_cursor:
            raise ValueError("aligned renderer chunk made no receiver-token progress")
        chunks.append((sender_cursor, sender_next, receiver_cursor, receiver_next))
        sender_cursor = sender_next
        receiver_cursor = receiver_next
    if receiver_cursor != receiver_end:
        raise ValueError(
            f"aligned renderer chunks ended at receiver {receiver_cursor}, expected {receiver_end}"
        )
    return chunks


def _renderer_transferred_queries(
    receiver_model,
    query_transfer,
    sender_queries: list[torch.Tensor],
    receiver_positions: torch.Tensor,
) -> list[torch.Tensor]:
    receiver_base = _base_model(receiver_model)
    receiver_heads = int(receiver_model.config.get_text_config().num_key_value_heads)
    query_groups = query_transfer.num_heads // receiver_heads
    layer_queries = []
    for layer_idx in range(query_transfer.receiver_layers):
        query = query_transfer.forward_layer_all(layer_idx, sender_queries).permute(0, 2, 1, 3)
        query = _apply_rope(
            receiver_base,
            query,
            layer_idx,
            receiver_positions,
            inverse=False,
        )
        batch, _, query_len, head_dim = query.shape
        query = query.view(batch, receiver_heads, query_groups, query_len, head_dim)
        query = query.permute(1, 0, 2, 3, 4).reshape(
            receiver_heads,
            batch * query_groups * query_len,
            head_dim,
        )
        layer_queries.append(query.detach().cpu())
    return layer_queries


def _collect_hybrid_chunk_queries(
    sender_model,
    receiver_model,
    query_transfer,
    sender_cache,
    sender_len: int,
    local_frame_len: int,
    instruction_ids: list[int],
    chunk_ids: list[int],
    am_config: AttentionMatchingConfig,
    device: str,
) -> list[torch.Tensor]:
    repeat_ids = instruction_ids + chunk_ids
    sender_window = int(
        getattr(_base_model(sender_model).config, "max_position_embeddings", sender_len + len(repeat_ids))
    )
    if sender_len + len(repeat_ids) > sender_window:
        raise ValueError(
            f"AM query pass needs {sender_len + len(repeat_ids)} Qwen positions, "
            f"but its configured window is {sender_window}; lower --bc-chunk-tokens"
        )
    input_ids = torch.tensor([repeat_ids], dtype=torch.long, device=device)
    positions = torch.arange(
        sender_len,
        sender_len + len(repeat_ids),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    receiver_heads = int(receiver_model.config.get_text_config().num_key_value_heads)
    query_groups = query_transfer.num_heads // receiver_heads
    token_limit = max(1, (am_config.query_samples + query_groups - 1) // query_groups)
    query_positions = torch.arange(
        len(instruction_ids), len(repeat_ids), dtype=torch.long, device=device
    )
    if query_positions.numel() > token_limit:
        selected = torch.randperm(query_positions.numel(), device=device)[:token_limit]
        query_positions = query_positions[selected]
    sender_queries = collect_normalized_queries(
        sender_model,
        input_ids,
        positions,
        query_positions,
        past_key_values=sender_cache,
    )
    sender_cache.crop(sender_len)
    receiver_positions = positions[:, query_positions] - sender_len + local_frame_len
    return _renderer_transferred_queries(
        receiver_model,
        query_transfer,
        sender_queries,
        receiver_positions,
    )


@torch.no_grad()
def _prepare_renderer_hybrid_cache(
    renderer,
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    query_transfer,
    sample: C2CExample,
    device: str,
    max_length: int,
    chunk_tokens: int,
    am_config: AttentionMatchingConfig,
    causal_cross_attention: bool,
    pack: str,
    position_mode: str = "interp",
) -> tuple[DynamicCache, list[torch.Tensor], int, int, dict[str, int], dict[str, Any]]:
    context = encode_context_pair(
        sender_tokenizer,
        receiver_tokenizer,
        sample.sender_messages,
        max_length,
        position_mode,
    )
    sender_ids = context["sender_ids"]
    receiver_ids = context["receiver_ids"]
    if not sender_ids or not receiver_ids:
        raise ValueError("renderer hybrid received an empty context prompt")

    article = str(sample.metadata.get("context", ""))
    if not article:
        raise ValueError("renderer hybrid requires sample.metadata['context']")
    sender_text = apply_chat_template(
        sender_tokenizer, sample.sender_messages, add_generation_prompt=False
    )
    receiver_text = apply_chat_template(
        receiver_tokenizer, sample.sender_messages, add_generation_prompt=False
    )
    full_sender_ids = sender_tokenizer(sender_text, add_special_tokens=False)["input_ids"]
    full_receiver_ids = receiver_tokenizer(receiver_text, add_special_tokens=False)["input_ids"]
    if full_sender_ids[: len(sender_ids)] != sender_ids:
        raise ValueError("could not map the truncated sender prompt back to its full prefix")
    if full_receiver_ids[: len(receiver_ids)] != receiver_ids:
        raise ValueError("could not map the truncated receiver prompt back to its full prefix")
    sender_article_start, sender_article_end, sender_span_source = _prompt_article_span(
        sender_tokenizer, sender_text, article
    )
    receiver_article_start, receiver_article_end, receiver_span_source = _prompt_article_span(
        receiver_tokenizer, receiver_text, article
    )
    sender_article_end = min(sender_article_end, len(sender_ids))
    receiver_article_end = min(receiver_article_end, len(receiver_ids))
    if sender_article_end <= sender_article_start or receiver_article_end <= receiver_article_start:
        raise ValueError("the article was fully removed by max-length truncation")

    chunks = _aligned_renderer_chunks(
        sender_article_start,
        sender_article_end,
        receiver_article_start,
        receiver_article_end,
        context["cross_causal_limits"],
        chunk_tokens,
    )
    sender_len = len(sender_ids)
    receiver_len = len(receiver_ids)
    sender_tensor = torch.tensor([sender_ids], dtype=torch.long, device=device)
    sender_positions = torch.arange(sender_len, dtype=torch.long, device=device).unsqueeze(0)
    sender_cache = sender_model(
        input_ids=sender_tensor,
        attention_mask=torch.ones_like(sender_tensor),
        position_ids=sender_positions,
        use_cache=True,
        past_key_values=full_sender_cache(),
    ).past_key_values

    base = _base_model(receiver_model)
    num_layers = len(base.layers)
    sender_layers = len(_base_model(sender_model).layers)
    receiver_heads = int(receiver_model.config.get_text_config().num_key_value_heads)
    receiver_window = int(base.config.max_position_embeddings)
    am_config, span_cap = _resolve_pack_mode(
        pack,
        am_config,
        receiver_model,
        receiver_article_end - receiver_article_start,
    )
    assembler = CompactGridAssembler(base, receiver_heads, span_cap=span_cap)
    virtual = torch.as_tensor(context["virtual_positions"], dtype=torch.float32, device=device)
    causal_limits = torch.as_tensor(
        context["cross_causal_limits"], dtype=torch.long, device=device
    )
    instruction_ids = repeat_prefill_prefix_ids(sender_tokenizer)
    compact_entries = 0
    pending_suffix = None

    def render_region(
        sender_start: int,
        sender_end: int,
        receiver_start: int,
        receiver_end: int,
    ):
        sender_indices = torch.cat(
            [
                torch.arange(sender_article_start, device=device),
                torch.arange(sender_start, sender_end, device=device),
            ]
        )
        true_sender_positions = sender_indices.unsqueeze(0)
        sender_slice = []
        for layer_idx in range(sender_layers):
            key, value = cache_layer(sender_cache, layer_idx)
            sender_slice.append(
                (
                    key.index_select(2, sender_indices),
                    value.index_select(2, sender_indices),
                )
            )
        memory = flatten_sender_cache(
            sender_model,
            sender_slice,
            true_sender_positions,
        )
        local_memory_positions = torch.cat(
            [
                virtual[:sender_article_start],
                virtual[sender_start:sender_end]
                - virtual[sender_start]
                + float(receiver_article_start),
            ]
        ).unsqueeze(0)
        local_receiver_ids = receiver_ids[:receiver_article_start] + receiver_ids[
            receiver_start:receiver_end
        ]
        local_receiver = torch.tensor([local_receiver_ids], dtype=torch.long, device=device)
        local_receiver_positions = torch.arange(
            len(local_receiver_ids), dtype=torch.long, device=device
        ).unsqueeze(0)
        if local_receiver.shape[1] > receiver_window:
            raise ValueError(
                f"renderer local call needs {local_receiver.shape[1]} receiver positions, "
                f"above Llama's {receiver_window}; lower --bc-chunk-tokens"
            )
        limits = None
        if causal_cross_attention:
            prefix_limits = causal_limits[:receiver_article_start].clamp(
                min=0, max=sender_article_start
            )
            region_limits = sender_article_start + (
                causal_limits[receiver_start:receiver_end] - sender_start
            ).clamp(min=0, max=sender_end - sender_start)
            limits = torch.cat([prefix_limits, region_limits]).unsqueeze(0)
        embeds = receiver_model.get_input_embeddings()(local_receiver)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            pred = renderer(
                embeds,
                local_receiver_positions,
                memory,
                local_memory_positions,
                None,
                limits,
            )
        rendered = pred_to_cache(
            receiver_model,
            pred,
            local_receiver_positions,
            next(receiver_model.parameters()).dtype,
        )
        del memory, pred, sender_slice
        return rendered, local_receiver_positions[0]

    for chunk_index, (s_start, s_end, r_start, r_end) in enumerate(chunks):
        last = chunk_index == len(chunks) - 1
        render_sender_end = sender_len if last else s_end
        render_receiver_end = receiver_len if last else r_end
        rendered, local_positions = render_region(
            s_start,
            render_sender_end,
            r_start,
            render_receiver_end,
        )
        prefix_len = receiver_article_start
        article_len = r_end - r_start
        if chunk_index == 0 and prefix_len:
            assembler.add_keep(
                [(key[:, :, :prefix_len], value[:, :, :prefix_len]) for key, value in rendered],
                local_positions[:prefix_len].float(),
            )
            compact_entries += prefix_len * num_layers * receiver_heads

        article_cache = DynamicCache(config=base.config)
        for layer_idx, (key, value) in enumerate(rendered):
            article_cache.update(
                key[:, :, prefix_len : prefix_len + article_len],
                value[:, :, prefix_len : prefix_len + article_len],
                layer_idx,
            )
        local_frame_len = len(local_positions)
        layer_queries = _collect_hybrid_chunk_queries(
            sender_model,
            receiver_model,
            query_transfer,
            sender_cache,
            sender_len,
            local_frame_len,
            instruction_ids,
            sender_ids[s_start:s_end],
            am_config,
            device,
        )
        compacted, betas, _, entries, index_positions = (
            compact_cache_attention_matching_from_queries(
                receiver_model,
                article_cache,
                layer_queries,
                article_len,
                am_config,
            )
        )
        compact_entries += entries
        article_rope_positions = local_positions[
            prefix_len : prefix_len + article_len
        ].float()
        assembler.add_repack(
            [cache_layer(compacted, layer_idx) for layer_idx in range(num_layers)],
            betas,
            _lookup_rope_positions(article_rope_positions, index_positions),
        )
        if last:
            suffix_start = prefix_len + article_len
            suffix_len = len(local_positions) - suffix_start
            if suffix_len:
                suffix_caches = [
                    (key[:, :, suffix_start:], value[:, :, suffix_start:])
                    for key, value in rendered
                ]
                suffix_betas = [
                    torch.zeros(
                        (receiver_heads, suffix_len),
                        dtype=suffix_caches[0][0].dtype,
                        device=device,
                    )
                    for _ in range(num_layers)
                ]
                suffix_positions = [
                    local_positions[suffix_start:].float().view(1, 1, -1).expand(
                        1, receiver_heads, -1
                    )
                    for _ in range(num_layers)
                ]
                pending_suffix = (suffix_caches, suffix_betas, suffix_positions)
                compact_entries += suffix_len * num_layers * receiver_heads
        del rendered, article_cache, compacted, layer_queries

    del sender_cache
    if pending_suffix is not None:
        assembler.add_repack(*pending_suffix)
    cache, betas, decode_start = assembler.assemble()
    cache_stats = attention_matching_cache_stats(cache, betas)
    span_stats = {
        "article_start": receiver_article_start,
        "article_end": receiver_article_end,
        "full_len": receiver_len,
        "source": f"sender:{sender_span_source},receiver:{receiver_span_source}",
    }
    run_stats = {
        "sender_len": sender_len,
        "receiver_len": receiver_len,
        "chunks": len(chunks),
        "chunk_tokens": chunk_tokens,
    }
    print(
        f"renderer hybrid: one {sender_len}-token Qwen prefill, {len(chunks)} local "
        f"renderer+Llama-AM chunks -> compact Llama decode start {decode_start}"
    )
    return cache, betas, decode_start, compact_entries, cache_stats, span_stats | run_stats


@torch.no_grad()
def generate_renderer_hybrid(
    renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer,
    query_transfer, sample, device, max_length, max_new_tokens, am_config, decode_fn,
    min_new_tokens=0, context_cache=None, chunk_tokens=4096, pack="fractional",
    causal_cross_attention=False, decode_prefix="question", position_mode="interp",
    timings=None, on_prepared=None,
):
    if getattr(renderer, "token_alignment", "native") != "native":
        raise ValueError("C2C-longest does not yet support beyond-context evaluation")
    context_id = str(sample.metadata.get("context_id", ""))
    preparation_started = synchronized_time(device) if timings is not None else None
    prepared = context_cache.get(context_id) if context_cache is not None and context_id else None
    context_cache_hit = prepared is not None
    if prepared is None:
        prepared = _prepare_renderer_hybrid_cache(
            renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer,
            query_transfer, sample, device, max_length, chunk_tokens, am_config,
            causal_cross_attention, pack, position_mode,
        )
        if context_cache is not None and context_id:
            context_cache.clear()
            context_cache[context_id] = prepared
    prepared = (copy.deepcopy(prepared[0]),) + prepared[1:]
    if timings is not None:
        timings.update(
            sender_ms=0.0 if context_cache_hit else (synchronized_time(device) - preparation_started) * 1000,
            context_cache_hit=context_cache_hit,
        )
        if not context_cache_hit and on_prepared is not None:
            on_prepared(dict(timings))
        receiver_started = preparation_started if context_cache_hit else synchronized_time(device)
    span = prepared[5]
    return decode_renderer_cache(
        receiver_model, receiver_tokenizer, sample, prepared, device, max_length,
        max_new_tokens, am_config, decode_fn, min_new_tokens=min_new_tokens,
        decode_prefix=decode_prefix, timings=timings,
        receiver_started=receiver_started if timings is not None else None,
        position_limit=int(_base_model(receiver_model).config.max_position_embeddings),
        stop_suffix=f"/fullqwen{span['sender_len']}/chunks{span['chunks']}",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate long-context native-local or rendered-hybrid AM")
    modes = parser.add_subparsers(dest="beyond_context", required=True)
    local = modes.add_parser("local", help="Independently prefill and compact native receiver chunks")
    hybrid = modes.add_parser("hybrid", help="Full sender prefill with local rendered and compacted chunks")
    for entry in (local, hybrid):
        add_eval_args(entry)
        add_am_args(entry)
        entry.add_argument("--am-target-ratio", type=float, required=True)
        entry.add_argument("--bc-chunk-tokens", type=int, default=4096,
                           help="Context tokens per local chunk")
        entry.add_argument("--bc-pack", choices=["clamp", "fractional"], default="clamp")
    local.add_argument("--model", required=True)
    local.add_argument("--adapter", help="PEFT/LoRA adapter merged into the receiver")
    local.add_argument("--sample-ids", help="JSON list of sample ids to evaluate")
    local.add_argument("--bc-native-window", type=int, help="Override the native compact packing window")
    add_renderer_args(hybrid)
    hybrid.add_argument("--am-query-transfer", required=True)
    hybrid.add_argument("--am-query-storage", choices=["cpu", "device"], default="cpu")
    hybrid.add_argument("--am-attention-backend", choices=["eager"], default="eager")
    hybrid.add_argument("--am-decode-backend", choices=["eager", "torchscript_prefill"], default="eager")
    args = parser.parse_args(argv)
    validate_eval_args(parser, args)
    validate_am_args(parser, args)
    if args.am_chunk_size:
        parser.error("long-context runs chunk the article themselves; drop --am-chunk-size")
    if args.bc_chunk_tokens <= 0:
        parser.error("--bc-chunk-tokens must be positive")
    if args.beyond_context == "local" and args.bc_native_window is not None and args.bc_native_window <= WINDOW_RESERVE:
        parser.error(f"--bc-native-window must exceed the {WINDOW_RESERVE}-position packing reserve")
    return args


def main(argv=None):
    args = parse_args(argv)
    command_line = sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
    started_at = time.time()
    dtype = dtype_from_args(args)
    cache = BoundedContextCache(args.context_cache_max)
    common = dict(device=args.device, max_length=args.max_length, max_new_tokens=args.max_new_tokens,
                  min_new_tokens=args.am_min_new_tokens, context_cache=cache,
                  decode_fn=generate_from_beta_cache, chunk_tokens=args.bc_chunk_tokens, pack=args.bc_pack)
    native_window = query_transfer_checkpoint = None
    if args.beyond_context == "local":
        am_config = make_am_config(args)
        model_path = resolve_model_path(args.model)
        tokenizer = load_tokenizer(model_path)
        model = load_model(model_path, dtype, args.device, adapter=args.adapter)
        native_window = args.bc_native_window
        model_payload = {"receiver_model": str(model_path)}
        mode = "beyond_context_native"
        generate = partial(generate_beyond_context_native, model=model, tokenizer=tokenizer,
                           am_config=am_config, packing_window=native_window, **common)
    else:
        am_config = make_am_config(args, query_storage=args.am_query_storage,
                                   attention_backend=args.am_attention_backend, decode_backend=args.am_decode_backend)
        from multiagent.model.cache_renderer import load_renderer_checkpoint

        renderer_file = Path(args.renderer)
        if renderer_file.is_dir():
            renderer_file = renderer_file / "renderer.pt"
        renderer, renderer_payload = load_renderer_checkpoint(renderer_file, args.device)
        if renderer.token_alignment != "native":
            raise ValueError("C2C-longest does not yet support AM or beyond-context evaluation")
        sender_path = resolve_model_path(renderer_payload["sender_model"])
        receiver_path = resolve_model_path(renderer_payload["receiver_model"])
        renderer_no_sender_memory = bool(renderer_payload.get("no_sender_memory"))
        renderer_causal_cross_attention = bool(renderer_payload.get("args", {}).get("causal_cross_attention"))
        renderer_position_mode = str(renderer_payload.get("position_mode") or "interp")
        if renderer_position_mode != "interp":
            print(f"renderer trained with position_mode={renderer_position_mode!r}; using it for the sender memory geometry at eval")
        if renderer_no_sender_memory:
            print("renderer trained WITHOUT sender memory (pure prefill-emulator ablation)")
        if renderer_causal_cross_attention:
            print("renderer trained with CAUSAL cross-attention; applying its saved alignment at eval")
        print(f"renderer_checkpoint: {renderer_file} (step {renderer_payload.get('step')})")
        print(f"sender_model (from checkpoint): {sender_path}")
        print(f"receiver_model (from checkpoint): {receiver_path}")
        sender_tokenizer = load_tokenizer(sender_path)
        receiver_tokenizer = load_tokenizer(receiver_path)
        sender_model = load_model(sender_path, dtype, args.device, adapter=args.sender_adapter)
        receiver_model = load_model(receiver_path, dtype, args.device)
        if renderer_no_sender_memory:
            raise ValueError(
                "renderer AM compaction needs sender self-study queries; "
                "this checkpoint was trained without sender memory"
            )
        query_transfer_checkpoint, query_transfer = load_query_transfer(args.am_query_transfer, args.device)
        print(f"am_query_transfer: {query_transfer_checkpoint}")

        model_payload = {
            "receiver_model": str(receiver_path),
            "sender_model": str(sender_path),
            "renderer": str(args.renderer),
            "sender_device": args.device,
            "receiver_device": args.device,
            "token_alignment": renderer_payload.get("token_alignment", {"mode": "native", "version": 1}),
            "renderer_config": renderer.config_dict(),
        }
        receiver_layer_types = getattr(_base_model(receiver_model).config, "layer_types", None)
        model_payload["renderer_config"]["token_count_layout"] = (
            "full_attention" if not receiver_layer_types or all(
                kind == "full_attention" for kind in receiver_layer_types
            ) else "mixed_attention"
        )
        tokenizer = receiver_tokenizer
        mode = "cache_renderer_attention_matching_hybrid"
        generate = partial(
            generate_renderer_hybrid, renderer=renderer, sender_model=sender_model, receiver_model=receiver_model,
            sender_tokenizer=sender_tokenizer, receiver_tokenizer=receiver_tokenizer, query_transfer=query_transfer,
            am_config=am_config, causal_cross_attention=renderer_causal_cross_attention,
            decode_prefix=args.renderer_decode_prefix, position_mode=renderer_position_mode, **common,
        )
    tasks = {spec: load_c2c_task(spec, root=args.dataset_root) for spec in args.tasks}
    datasets = {spec: task.dataset for spec, task in tasks.items()}
    for spec, dataset in datasets.items():
        if len(dataset) and not str(dataset[0].target_text or "").strip():
            raise ValueError(f"{spec} has no gold target; use a labeled split")
    total = sum(len(d) if args.limit_per_dataset is None else min(len(d), args.limit_per_dataset)
                for d in datasets.values())
    json_compactor = EvalJsonCompactor(tokenizer)
    run_metadata = {
        "mode": mode, **model_payload, "tasks": list(datasets),
        "attention_matching": am_metadata(
            args, am_config, beta_mapping="identity" if args.beyond_context == "hybrid" else None,
            query_transfer=query_transfer_checkpoint, beyond_context=args.beyond_context,
            chunk_tokens=args.bc_chunk_tokens, pack=args.bc_pack, native_window=native_window,
        ),
        "evaluation_config": evaluation_config_metadata(args, dtype=dtype, command_line=command_line, script=__file__),
        "saved_json_format": eval_json_format_metadata(),
    }
    journal = CompressionJournal(args.output, run_metadata) if args.beyond_context == "hybrid" else None
    stats = {spec: {"seen": 0, "correct": 0, "incorrect": 0, "fail": 0} for spec in datasets}
    stop_stats = {spec: {"eos": 0, "cutoff": 0, "empty_prompt": 0} for spec in datasets}
    metric_stats = {}
    rows = []
    resumed_keys = set()
    sample_ids = set(json.loads(Path(args.sample_ids).read_text())) if args.sample_ids else None

    def count_grade(task_key, grade, stop_reason):
        label, metrics, raw_grade = normalize_grade_result(grade)
        stats[task_key]["seen"] += 1
        stats[task_key][label] = stats[task_key].get(label, 0) + 1
        update_metric_stats(metric_stats, task_key, metrics)
        stop_stats[task_key][stop_reason] = stop_stats[task_key].get(stop_reason, 0) + 1
        return label, metrics, raw_grade

    def current_payload():
        return {**run_metadata, **elapsed_metadata(started_at),
                "summary": summarize(stats, stop_stats, metric_stats), "rows": rows}

    if args.resume_from:
        for row in json.loads(Path(args.resume_from).read_text())["rows"]:
            grade = row["grade_result"] if "grade_result" in row else row["grade"]
            count_grade(row["task_key"], grade, row["stop_reason"])
            rows.append(row)
            resumed_keys.add((row["task_key"], int(row["index"])))
        print(f"resumed {len(rows)} graded samples from {args.resume_from}")

    progress = tqdm(interleaved_items(datasets, args.chunk_size, args.limit_per_dataset, args.seed),
                    total=total, desc=f"eval/beyond-context-{args.beyond_context}",
                    bar_format="{desc}: {n_fmt}/{total_fmt}|{percentage:2.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}{postfix}")
    for position, (task_key, index, sample) in enumerate(progress, start=1):
        if (task_key, int(index)) in resumed_keys or (sample_ids is not None and sample.id not in sample_ids):
            continue
        context_seed = seed_attention_matching_context(args.seed, task_key, index, sample, args.device,
                                                      seed_fn=deterministic_am_context_seed)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        timings = None
        options = {}
        if journal is not None:
            timings = {}
            options = dict(timings=timings, on_prepared=lambda measured: journal.write(
                "renderer_context", task_key=task_key, index=index, id=sample.id,
                context_id=sample.metadata.get("context_id"), timings=measured))
        output, stop_reason, generated_tokens = generate(sample=sample, **options)
        compression = renderer_am_token_counts(stop_reason, model_payload.get("renderer_config", {}))
        if journal is not None:
            journal.write("answer", task_key=task_key, index=index, id=sample.id,
                          context_id=sample.metadata.get("context_id"), output=output, stop_reason=stop_reason,
                          generated_tokens=generated_tokens, timings=timings,
                          **({"compression": compression} if compression is not None else {}))
        result, metrics, raw_grade = count_grade(task_key, tasks[task_key].grade(output, sample), stop_reason)
        if args.print_every > 0 and position % args.print_every == 0:
            print_eval_sample(position, total, task_key, sample, output, result, metrics)
        rows.append({
            "task": sample.dataset, "task_key": task_key, "index": index,
            **compact_eval_sample(sample, json_compactor),
            "grade": result, "grade_result": raw_grade, "output": output,
            "stop_reason": stop_reason, "generated_tokens": generated_tokens,
            "am_context_seed": context_seed, "time": time.time(),
            **({"timings": timings} if timings is not None else {}),
            **({"compression": compression} if compression is not None else {}),
        })
        progress.set_postfix(score_postfix(stats, metric_stats))
        if args.output and args.save_every > 0 and len(rows) % args.save_every == 0:
            save_json(args.output, current_payload())
    payload = current_payload()
    print(json.dumps(payload["summary"], indent=2))
    if args.output:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
