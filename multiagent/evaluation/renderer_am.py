from __future__ import annotations

import argparse
import copy
from functools import partial
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c import (
    apply_chat_template,
    chat_prompt_ids,
    generation_prompt_ids,
    locate_article,
    span_tag,
)
from multiagent.dataset.c2c_cross_family import encode_context_pair
from multiagent.dataset.c2c_registry import load_c2c_task
from multiagent.dataset.c2c_types import C2CExample
from multiagent.evaluation.native_am import generate_from_beta_cache
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
    CompressionJournal,
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
    _receiver_header_len,
    validate_am_args,
    validate_eval_args,
)
from multiagent.model.attention_matching import AttentionMatchingConfig, attention_matching_cache_stats, compact_cache_attention_matching_chunked_from_queries, compact_cache_attention_matching_from_queries
from multiagent.model.cache_renderer import flatten_sender_cache, pred_to_cache
from multiagent.model.query_transfer import collect_normalized_queries, load_query_transfer
from multiagent.model.unaware_comm import _apply_rope, _base_model, full_sender_cache, tuple_to_dynamic_cache
from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.train.train_utils import move_kv_cache


@torch.no_grad()
def generate_attention_matching_renderer(
    renderer,
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    query_transfer,
    sample: C2CExample,
    device: str,
    max_length: int,
    max_new_tokens: int,
    am_config: AttentionMatchingConfig,
    decode_fn: Callable,
    min_new_tokens: int = 0,
    context_cache: dict[str, Any] | None = None,
    causal_cross_attention: bool = False,
    decode_prefix: str = "question",
    position_mode: str = "interp",
    timings: dict[str, Any] | None = None,
    on_prepared: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, str, int]:
    if getattr(renderer, "token_alignment", "native") != "native":
        raise ValueError("C2C-longest does not yet support AM or beyond-context evaluation")
    context_id = str(sample.metadata.get("context_id", ""))
    preparation_started = synchronized_time(device) if timings is not None else None
    sender_prefill_ms = 0.0
    prepared = context_cache.get(context_id) if context_cache is not None and context_id else None
    context_cache_hit = prepared is not None
    if prepared is None:
        context = encode_context_pair(
            sender_tokenizer, receiver_tokenizer, sample.sender_messages, max_length, position_mode
        )
        sender_ids = context["sender_ids"]
        receiver_ids_list = context["receiver_ids"]
        if not sender_ids or not receiver_ids_list:
            return "", "empty_prompt", 0
        repeat_instruction_ids = sender_tokenizer(
            "\nRepeat the previous context.\n",
            add_special_tokens=False,
        )["input_ids"]
        if not repeat_instruction_ids:
            return "", "empty_prompt", 0

        sender_len = len(sender_ids)
        sender_input_ids = torch.tensor([sender_ids], dtype=torch.long, device=device)
        sender_positions = torch.arange(sender_len, dtype=torch.long, device=device).unsqueeze(0)
        sender_cache = sender_model(
            input_ids=sender_input_ids,
            attention_mask=torch.ones_like(sender_input_ids),
            position_ids=sender_positions,
            use_cache=True,
            past_key_values=full_sender_cache(),
        ).past_key_values
        if timings is not None:
            sender_prefill_ms = (synchronized_time(device) - preparation_started) * 1000
        memory = flatten_sender_cache(
            sender_model, move_kv_cache(sender_cache, device), sender_positions
        )
        virtual = torch.from_numpy(context["virtual_positions"][None].astype("float32")).to(device)
        receiver_ids = torch.tensor([receiver_ids_list], dtype=torch.long, device=device)
        receiver_positions = torch.arange(receiver_ids.shape[1], device=device).unsqueeze(0)
        embeds = receiver_model.get_input_embeddings()(receiver_ids)
        limits = (torch.from_numpy(context["cross_causal_limits"][None]).to(device)
                  if causal_cross_attention else None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            pred = renderer(embeds, receiver_positions, memory, virtual, None, limits)
        pred_cache = pred_to_cache(
            receiver_model, pred, receiver_positions, next(receiver_model.parameters()).dtype
        )
        del memory, pred
        decode_start = receiver_ids.shape[1]
        rendered_cache = tuple_to_dynamic_cache(pred_cache, receiver_model.config)
        del pred_cache
        if am_config.oracle_queries:
            import re
            from multiagent.evaluation.native_am import compact_attention_matching_prefix
            oracle = compact_attention_matching_prefix(
                receiver_model, receiver_tokenizer, sample, device, max_length, am_config,
                prefix_cache=rendered_cache,
            )
            if oracle is None:
                return "", "empty_prompt", 0
            receiver_cache, betas, _, oracle_len, _, compact_entries, cache_stats, oracle_span = oracle
            if oracle_len != decode_start:
                raise RuntimeError(
                    f"oracle queries: receiver tokenization ({oracle_len}) does not match the rendered "
                    f"cache length ({decode_start})"
                )
            parsed = re.search(r"span(\d+)-(\d+)of(\d+)(\w*)", oracle_span or "")
            am_span_stats = {
                "article_start": int(parsed.group(1)) if parsed else 0,
                "article_end": int(parsed.group(2)) if parsed else int(decode_start),
                "full_len": int(decode_start),
                "source": "oracle-queries:" + (parsed.group(4) if parsed else "full-prefix"),
            }
            del rendered_cache
            print("AM oracle: receiver-model on-policy queries over the rendered cache")
        else:

            receiver_base = _base_model(receiver_model)
            receiver_kv_heads = int(receiver_model.config.get_text_config().num_key_value_heads)
            query_groups = query_transfer.num_heads // receiver_kv_heads
            token_query_limit = max(1, (am_config.query_samples + query_groups - 1) // query_groups)
            query_prefill_started = synchronized_time(device) if timings is not None else None
            repeat_ids = repeat_instruction_ids + sender_ids
            repeat_input_ids = torch.tensor([repeat_ids], dtype=torch.long, device=device)
            repeat_positions = torch.arange(
                sender_len,
                sender_len + len(repeat_ids),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            document_query_positions = torch.arange(
                len(repeat_instruction_ids),
                len(repeat_ids),
                dtype=torch.long,
                device=device,
            )
            if document_query_positions.numel() > token_query_limit:
                selected = torch.randperm(
                    document_query_positions.numel(),
                    device=device,
                )[:token_query_limit]
                document_query_positions = document_query_positions[selected]
            sender_queries = collect_normalized_queries(
                sender_model,
                repeat_input_ids,
                repeat_positions,
                document_query_positions,
                past_key_values=sender_cache,
                **({"output_device": device, "output_dtype": torch.float32}
                   if am_config.query_storage == "device" else {}),
            )
            if timings is not None:
                sender_prefill_ms += (synchronized_time(device) - query_prefill_started) * 1000
            del sender_cache

            layer_queries = []
            selected_positions = repeat_positions[:, document_query_positions] - sender_len + decode_start
            for layer_idx in range(query_transfer.receiver_layers):
                receiver_query = query_transfer.forward_layer_all(layer_idx, sender_queries)
                receiver_query = receiver_query.permute(0, 2, 1, 3)
                receiver_query = _apply_rope(
                    receiver_base,
                    receiver_query,
                    layer_idx,
                    selected_positions,
                    inverse=False,
                )
                batch, _, query_len, head_dim = receiver_query.shape
                receiver_query = receiver_query.view(
                    batch,
                    receiver_kv_heads,
                    query_groups,
                    query_len,
                    head_dim,
                )
                receiver_query = receiver_query.permute(1, 0, 2, 3, 4).reshape(
                    receiver_kv_heads,
                    batch * query_groups * query_len,
                    head_dim,
                )
                query_device = "cpu" if am_config.query_storage == "cpu" else device
                layer_queries.append(receiver_query.detach().to(device=query_device))
            del sender_queries

            am_span_stats = None
            if am_config.chunk_size > 0:
                article_text = str(sample.metadata.get("context", ""))
                receiver_text = apply_chat_template(
                    receiver_tokenizer, sample.sender_messages, add_generation_prompt=False
                )
                full_receiver_ids = receiver_tokenizer(receiver_text, add_special_tokens=False)["input_ids"]
                if full_receiver_ids[len(full_receiver_ids) - decode_start :] == receiver_ids_list:
                    truncation_offset = len(full_receiver_ids) - decode_start
                else:
                    truncation_offset = 0
                article_char_start, article_text, span_source = locate_article(
                    receiver_text, article_text
                )
                if article_char_start < 0:
                    raise ValueError(
                        "could not locate the article inside the rendered receiver prompt "
                        f"(task={sample.metadata.get('task', '?')!r} "
                        f"id={sample.metadata.get('id', '?')!r}); "
                        f"article {len(article_text)} chars, prompt {len(receiver_text)} chars. "
                        "Tried exact/rstrip/normalized matching. Fix the dataset so "
                        "metadata['context'] appears verbatim in the sender message, or pass "
                        "an explicit article span; do not compact the template header."
                    )
                article_start_full = len(
                    receiver_tokenizer(
                        receiver_text[:article_char_start],
                        add_special_tokens=False,
                    )["input_ids"]
                )
                article_end_full = len(
                    receiver_tokenizer(
                        receiver_text[: article_char_start + len(article_text)],
                        add_special_tokens=False,
                    )["input_ids"]
                )
                article_start = max(0, article_start_full - truncation_offset)
                article_end = min(decode_start, article_end_full - truncation_offset)
                if article_end <= article_start:
                    raise ValueError("the article was fully removed by max-length truncation")
                header_len = _receiver_header_len(receiver_tokenizer)
                if article_start < header_len:
                    raise ValueError(
                        f"article_start={article_start} is inside the {header_len}-token chat "
                        "template header, so attention-sink tokens would be evicted by AM. "
                        f"(span_source={span_source}, truncation_offset={truncation_offset}, "
                        f"decode_start={decode_start}). Refusing to produce silently-degraded "
                        "results."
                    )
                am_span_stats = {
                    "article_start": int(article_start),
                    "article_end": int(article_end),
                    "full_len": int(decode_start),
                    "source": span_source,
                }
                chunk_size = am_config.chunk_size
                receiver_cache, betas, _, compact_entries, _ = (
                    compact_cache_attention_matching_chunked_from_queries(
                        receiver_model,
                        rendered_cache,
                        layer_queries,
                        article_start,
                        article_end,
                        chunk_size,
                        am_config,
                    )
                )
            else:
                receiver_cache, betas, _, compact_entries, _ = (
                    compact_cache_attention_matching_from_queries(
                        receiver_model,
                        rendered_cache,
                        layer_queries,
                        decode_start,
                        am_config,
                    )
                )
            del rendered_cache
            cache_stats = attention_matching_cache_stats(receiver_cache, betas)
        if context_cache is not None and context_id:
            context_cache.clear()
            context_cache[context_id] = (
                receiver_cache,
                betas,
                decode_start,
                compact_entries,
                cache_stats,
                am_span_stats,
            )
            receiver_cache = copy.deepcopy(receiver_cache)
    else:
        base_cache, betas, decode_start, compact_entries, cache_stats, am_span_stats = prepared
        receiver_cache = copy.deepcopy(base_cache)

    if timings is not None:
        timings.update(
            sender_ms=0.0 if context_cache_hit else (synchronized_time(device) - preparation_started) * 1000,
            context_cache_hit=context_cache_hit,
        )
        timings.update(
            sender_prefill_ms=sender_prefill_ms,
            sender_am_ms=timings["sender_ms"] - sender_prefill_ms,
            sender_decode_ms=0.0,
        )
        if not context_cache_hit and on_prepared is not None:
            on_prepared(dict(timings))
        receiver_started = preparation_started if context_cache_hit else synchronized_time(device)

    return decode_renderer_cache(
        receiver_model, receiver_tokenizer, sample,
        (receiver_cache, betas, decode_start, compact_entries, cache_stats, am_span_stats),
        device, max_length, max_new_tokens, am_config, decode_fn,
        min_new_tokens=min_new_tokens, decode_prefix=decode_prefix, timings=timings,
        receiver_started=receiver_started if timings is not None else None,
    )


@torch.no_grad()
def decode_renderer_cache(
    receiver_model, receiver_tokenizer, sample, prepared, device, max_length,
    max_new_tokens, am_config, decode_fn, min_new_tokens=0, decode_prefix="question",
    timings=None, receiver_started=None, position_limit=None, stop_suffix="",
):
    receiver_cache, betas, decode_start, compact_entries, cache_stats, am_span_stats = prepared
    if decode_prefix == "generation":
        receiver_decode_ids = generation_prompt_ids(receiver_tokenizer)
    else:
        receiver_decode_ids = chat_prompt_ids(
            receiver_tokenizer,
            sample.receiver_messages,
            max_length,
            add_generation_prompt=True,
        )
    if not receiver_decode_ids:
        return "", "empty_prompt", 0
    if position_limit is not None and decode_start + len(receiver_decode_ids) + max_new_tokens > position_limit:
        raise ValueError(
            f"packed renderer cache plus raw question/generation needs "
            f"{decode_start + len(receiver_decode_ids) + max_new_tokens} positions, "
            f"above its {position_limit}-position window"
        )
    text, stop_reason, generated_tokens = decode_fn(
        receiver_model,
        receiver_tokenizer,
        receiver_cache,
        betas,
        receiver_decode_ids,
        decode_start,
        device,
        max_new_tokens,
        min_new_tokens,
        attention_backend=am_config.attention_backend,
        decode_backend=am_config.decode_backend,
        **({"timings": timings} if timings is not None else {}),
    )
    if timings is not None:
        timings["receiver_ms"] = (synchronized_time(device) - receiver_started) * 1000
    return (
        text,
        f"{stop_reason}:renderer+am{compact_entries}active/"
        f"{cache_stats['allocated_entries']}allocated/{decode_start}recv+{len(receiver_decode_ids)}"
        + stop_suffix
        + (
            span_tag(
                article_start=am_span_stats["article_start"],
                article_end=am_span_stats["article_end"],
                full_len=am_span_stats["full_len"],
                source=am_span_stats["source"],
            )
            if am_span_stats
            else ""
        ),
        generated_tokens,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate rendered receiver caches compressed with Attention Matching")
    add_eval_args(parser)
    add_renderer_args(parser)
    add_am_args(parser)
    parser.add_argument("--am-target-ratio", type=float, required=True, help="Attention Matching cache ratio. Example: 0.1 keeps 10%% of prefix KV entries.")
    parser.add_argument("--am-query-storage", choices=["cpu", "device"], default="cpu",
                        help="Full-context renderer AM query storage. 'device' removes CPU round trips at the cost of extra accelerator memory; fitting and sampling are unchanged.")
    parser.add_argument("--am-attention-backend", choices=["eager"], default="eager",
                        help="Renderer AM answer attention backend (only eager; kept for command compatibility).")
    parser.add_argument("--am-decode-backend", choices=["eager", "torchscript_prefill"], default="eager",
                        help="Renderer AM decoding. torchscript_prefill retains actual cache lengths and eager arithmetic and also traces questions up to 256 tokens.")
    parser.add_argument(
        "--am-oracle-queries",
        action="store_true",
        help=(
            "Diagnostic for --renderer AM: compact the rendered cache with the receiver's own "
            "on-policy reference queries (receiver reads the document in the repeat turn) instead "
            "of transferred sender queries. Isolates query-transfer error from cache-transfer error."
        ),
    )
    parser.add_argument(
        "--am-query-transfer",
        help=(
            "Trained query-transfer checkpoint/directory for receiver-free "
            "transfer-then-compact AM queries."
        ),
    )
    args = parser.parse_args(argv)
    validate_eval_args(parser, args)
    validate_am_args(parser, args)
    if bool(args.am_query_transfer) == args.am_oracle_queries:
        parser.error("choose exactly one of --am-query-transfer or --am-oracle-queries")
    return args


def main(argv=None):
    args = parse_args(argv)
    command_line = sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
    started_at = time.time()
    dtype = dtype_from_args(args)
    am_config = make_am_config(
        args, query_storage=args.am_query_storage, attention_backend=args.am_attention_backend,
        decode_backend=args.am_decode_backend, oracle_queries=args.am_oracle_queries,
    )
    query_transfer_checkpoint = query_transfer = None
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
    if args.am_query_transfer:
        if renderer_no_sender_memory:
            raise ValueError(
                "renderer AM compaction needs sender self-study queries; "
                "this checkpoint was trained without sender memory"
            )
        query_transfer_checkpoint, query_transfer = load_query_transfer(
            args.am_query_transfer,
            args.device,
        )
        print(f"am_query_transfer: {query_transfer_checkpoint}")

    json_compactor = EvalJsonCompactor(receiver_tokenizer)

    tasks = {
        spec: load_c2c_task(
            spec,
            root=args.dataset_root,
        )
        for spec in args.tasks
    }
    datasets = {spec: task.dataset for spec, task in tasks.items()}
    for spec, dataset in datasets.items():
        if len(dataset) == 0:
            continue
        sample = dataset[0]
        if not str(sample.target_text or "").strip():
            raise ValueError(f"{spec} has no gold target; use a labeled split")
    total = sum(
        len(dataset) if args.limit_per_dataset is None else min(len(dataset), args.limit_per_dataset)
        for dataset in datasets.values()
    )

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
    model_payload["attention_matching"] = am_metadata(
        args, am_config, beta_mapping="identity", query_transfer=query_transfer_checkpoint,
    )
    mode = "cache_renderer_attention_matching"
    run_metadata = {
        "mode": mode,
        **model_payload,
        "tasks": list(datasets),
        "evaluation_config": evaluation_config_metadata(args, dtype=dtype, command_line=command_line, script=__file__),
        "saved_json_format": eval_json_format_metadata(),
    }

    context_cache = BoundedContextCache(args.context_cache_max)
    compression_journal = CompressionJournal(args.output, run_metadata)
    generation_options = dict(device=args.device, max_length=args.max_length,
                              max_new_tokens=args.max_new_tokens)
    generation_options.update(
        renderer=renderer, sender_model=sender_model, receiver_model=receiver_model,
        sender_tokenizer=sender_tokenizer, receiver_tokenizer=receiver_tokenizer,
        causal_cross_attention=renderer_causal_cross_attention,
        decode_prefix=args.renderer_decode_prefix, position_mode=renderer_position_mode,
    )
    generation_options.update(am_config=am_config, min_new_tokens=args.am_min_new_tokens,
                              context_cache=context_cache)
    generate = partial(
        generate_attention_matching_renderer, **generation_options,
        query_transfer=query_transfer, decode_fn=generate_from_beta_cache,
    )

    stats = {spec: {"seen": 0, "correct": 0, "incorrect": 0, "fail": 0} for spec in datasets}
    stop_stats = {spec: {"eos": 0, "cutoff": 0, "empty_prompt": 0} for spec in datasets}
    metric_stats: dict[str, dict[str, dict[str, float]]] = {}
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
        saved = json.loads(Path(args.resume_from).read_text())
        for row in saved["rows"]:
            grade = row["grade_result"] if "grade_result" in row else row["grade"]
            count_grade(row["task_key"], grade, row["stop_reason"])
            rows.append(row)
            resumed_keys.add((row["task_key"], int(row["index"])))
        print(f"resumed {len(rows)} graded samples from {args.resume_from}")

    schedule = interleaved_items(datasets, args.chunk_size, args.limit_per_dataset, args.seed)
    progress = tqdm(
        schedule, total=total, desc="eval/renderer-am",
        bar_format="{desc}: {n_fmt}/{total_fmt}|{percentage:2.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}{postfix}",
    )
    remaining = (
        (position, task_key, index, sample)
        for position, (task_key, index, sample) in enumerate(progress, start=1)
        if (task_key, int(index)) not in resumed_keys
        and (sample_ids is None or sample.id in sample_ids)
    )
    for position, task_key, index, sample in remaining:
        am_context_seed = seed_attention_matching_context(
            args.seed, task_key, index, sample, args.device,
            seed_fn=deterministic_am_context_seed,
        )
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        row_timings = {}
        output, stop_reason, generated_tokens = generate(
            sample=sample, timings=row_timings,
            on_prepared=lambda timings: compression_journal.write(
                "renderer_context", task_key=task_key, index=index,
                id=sample.id, context_id=sample.metadata.get("context_id"), timings=timings,
            ),
        )
        row_compression = renderer_am_token_counts(stop_reason, model_payload["renderer_config"])
        compression_journal.write(
            "answer", task_key=task_key, index=index, id=sample.id,
            context_id=sample.metadata.get("context_id"), output=output,
            stop_reason=stop_reason, generated_tokens=generated_tokens, timings=row_timings,
            **({"compression": row_compression} if row_compression is not None else {}),
        )
        result, metrics, raw_grade = count_grade(task_key, tasks[task_key].grade(output, sample), stop_reason)
        if args.print_every > 0 and position % args.print_every == 0:
            print_eval_sample(position, total, task_key, sample, output, result, metrics)
        rows.append({
            "task": sample.dataset, "task_key": task_key, "index": index,
            **compact_eval_sample(sample, json_compactor),
            "grade": result, "grade_result": raw_grade, "output": output,
            "stop_reason": stop_reason, "generated_tokens": generated_tokens,
            "timings": row_timings,
            **({"compression": row_compression} if row_compression is not None else {}),
            "am_context_seed": am_context_seed,
            "time": time.time(),
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
