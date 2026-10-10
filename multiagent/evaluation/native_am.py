from __future__ import annotations

import argparse
import copy
from functools import partial
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c_registry import load_c2c_task
from multiagent.evaluation.results import (
    EvalJsonCompactor, compact_eval_sample, elapsed_metadata, eval_json_format_metadata,
    evaluation_config_metadata, normalize_grade_result, print_eval_sample, save_json,
    score_postfix, summarize, update_metric_stats,
)
from multiagent.evaluation.utils import (
    BoundedContextCache,
    add_am_args,
    add_eval_args,
    am_metadata,
    deterministic_am_context_seed,
    dtype_from_args,
    eos_ids,
    interleaved_items,
    load_model,
    load_tokenizer,
    make_am_config,
    seed_attention_matching_context,
    synchronized_time,
    validate_am_args,
    validate_eval_args,
)
from multiagent.model.vanilla_loader import resolve_model_path

from multiagent.dataset.c2c import apply_chat_template, chat_prompt_ids, locate_article, span_tag
from multiagent.dataset.c2c_types import C2CExample
from multiagent.evaluation.utils import CompactGridAssembler, WINDOW_RESERVE
from multiagent.model.attention_matching import (
    AttentionMatchingConfig,
    attention_matching_cache_stats,
    beta_attention,
    compact_cache_attention_matching_chunked,
    compact_cache_attention_matching_on_policy,
    repeat_prefill_prefix_ids,
)
from multiagent.model.unaware_comm import _base_model, cache_layer


@torch.no_grad()
def compact_attention_matching_prefix(
    model,
    tokenizer,
    sample: C2CExample,
    device: str,
    max_length: int,
    am_config: AttentionMatchingConfig,
    prefix_cache=None,
) -> tuple[Any, list[torch.Tensor], list[torch.Tensor], int, list[int], int, dict[str, int]] | None:
    assert sample.prompt_mode == "split_prompt", "attention matching requires split_prompt examples"
    assert 0.0 < am_config.target_ratio <= 1.0, "AM target ratio must be in (0, 1]"

    sender_text = apply_chat_template(tokenizer, sample.sender_messages, add_generation_prompt=False)
    full_sender_ids = tokenizer(sender_text, add_special_tokens=False)["input_ids"]
    truncation_offset = max(0, len(full_sender_ids) - max_length)
    if truncation_offset:
        print(
            f"Warning: AM input has {len(full_sender_ids)} tokens; "
            f"left-truncating {truncation_offset} tokens to --max-length {max_length}."
        )
    sender_ids = full_sender_ids[truncation_offset:]
    receiver_decode_ids = chat_prompt_ids(
        tokenizer,
        sample.receiver_messages,
        max_length,
        add_generation_prompt=True,
    )
    if not sender_ids or not receiver_decode_ids:
        return None
    repeat_instruction_ids = repeat_prefill_prefix_ids(tokenizer)
    assert repeat_instruction_ids, "repeat-prefill instruction produced no tokens"

    sender_len = len(sender_ids)
    original_sender_len = sender_len
    sender_input_ids = torch.tensor([sender_ids], dtype=torch.long, device=device)
    repeat_input_ids = torch.tensor([repeat_instruction_ids + sender_ids], dtype=torch.long, device=device)
    sender_attention = torch.ones_like(sender_input_ids)
    sender_positions = torch.arange(sender_len, dtype=torch.long, device=device).unsqueeze(0)
    if prefix_cache is None:
        prefix_cache = model(
            input_ids=sender_input_ids,
            attention_mask=sender_attention,
            position_ids=sender_positions,
            use_cache=True,
        ).past_key_values

    article_start, article_end, span_source = 0, sender_len, "full-prefix"
    if am_config.chunk_size > 0:
        article_text = str(sample.metadata.get("context", ""))
        article_char_start, article_text, span_source = locate_article(
            sender_text, article_text
        )
        if article_char_start < 0 and article_text:
            raise ValueError(
                "could not locate the article inside the sender prompt "
                f"(article {len(article_text)} chars, prompt {len(sender_text)} chars). "
                "Fix the dataset so metadata['context'] appears verbatim, or pass an "
                "explicit span; refusing to compact the template header."
            )
        if article_char_start >= 0:
            article_start_full = len(
                tokenizer(sender_text[:article_char_start], add_special_tokens=False)["input_ids"]
            )
            article_end_full = len(
                tokenizer(
                    sender_text[: article_char_start + len(article_text)],
                    add_special_tokens=False,
                )["input_ids"]
            )
            article_start = max(0, article_start_full - truncation_offset)
            article_end = min(sender_len, article_end_full - truncation_offset)
        else:
            article_start, article_end = 0, sender_len
        if article_end <= article_start:
            raise ValueError("the article was fully removed by max-length truncation")
        chunk_size = am_config.chunk_size
        repeat_instruction = torch.tensor(
            [repeat_instruction_ids], dtype=torch.long, device=device
        )
        cache, betas, _, compact_entries, positions = compact_cache_attention_matching_chunked(
            model,
            prefix_cache,
            sender_input_ids,
            repeat_instruction,
            article_start,
            article_end,
            chunk_size,
            am_config,
        )
    else:
        cache, betas, _, compact_entries, positions = compact_cache_attention_matching_on_policy(
            model,
            prefix_cache,
            repeat_input_ids,
            sender_len,
            am_config,
        )

    if am_config.pack == "fractional":
        if am_config.packing_window is None:
            raise ValueError("fractional global AM requires a packing window")
        base_model = _base_model(model)
        kv_heads = int(
            getattr(base_model.config, "num_key_value_heads", base_model.config.num_attention_heads)
        )
        assembler = CompactGridAssembler(
            base_model,
            kv_heads,
            span_cap=float(
                am_config.packing_window
                - max(
                    WINDOW_RESERVE,
                    len(receiver_decode_ids) + am_config.generation_reserve,
                )
            ),
        )
        if article_start:
            assembler.add_keep(
                [
                    tuple(tensor[:, :, :article_start].clone() for tensor in cache_layer(cache, layer_idx))
                    for layer_idx in range(len(base_model.layers))
                ],
                torch.arange(article_start, dtype=torch.float32, device=device),
            )
        assembler.add_repack(
            [
                tuple(tensor[:, :, article_start:].clone() for tensor in cache_layer(cache, layer_idx))
                for layer_idx in range(len(base_model.layers))
            ],
            [beta[:, article_start:].clone() for beta in betas],
            [position[:, :, article_start:].clone() for position in positions],
        )
        cache, betas, sender_len = assembler.assemble()
        print(
            f"global AM fractional packing: full-prefill logical length {len(sender_ids)} "
            f"-> compact decode start {sender_len} under window {am_config.packing_window}"
        )

    cache_stats = attention_matching_cache_stats(cache, betas)
    if cache_stats["active_entries"] != compact_entries:
        raise RuntimeError(
            f"AM active-entry mismatch: allocator reported {compact_entries}, "
            f"cache contains {cache_stats['active_entries']}"
        )
    print(
        "AM cache allocation: "
        f"{cache_stats['active_entries']} active entries / "
        f"{cache_stats['allocated_entries']} dense allocated entries; "
        f"per-layer-head active range "
        f"{cache_stats['min_active_entries_per_layer_head']}.."
        f"{cache_stats['max_active_entries_per_layer_head']}; "
        "all inactive K/V entries verified zero"
    )
    return (
        cache, betas, positions, sender_len, receiver_decode_ids,
        compact_entries, cache_stats,
        span_tag(article_start, article_end, original_sender_len, span_source),
    )


@torch.no_grad()
def _generate_from_beta_cache_eager(
    model,
    tokenizer,
    cache,
    betas: list[torch.Tensor],
    receiver_decode_ids: list[int],
    logical_position: int,
    device: str,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    timings: dict[str, Any] | None = None,
    attention_backend: str = "eager",
) -> tuple[str, str, int]:
    if timings is not None:
        timings.update(receiver_prefill_ms=0.0, receiver_decode_ms=0.0)
    current_id = receiver_decode_ids[-1]
    output_ids = []
    stop_reason = "cutoff"
    stop_ids = eos_ids(tokenizer)
    with beta_attention(model, betas, backend=attention_backend):
        prefill_started = synchronized_time(device) if timings is not None else None
        if len(receiver_decode_ids) > 1:
            receiver_prefix_ids = torch.tensor([receiver_decode_ids[:-1]], dtype=torch.long, device=device)
            receiver_positions = torch.arange(
                logical_position,
                logical_position + receiver_prefix_ids.shape[1],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            cache = model(
                input_ids=receiver_prefix_ids,
                position_ids=receiver_positions,
                past_key_values=cache,
                use_cache=True,
            ).past_key_values
            logical_position += receiver_prefix_ids.shape[1]

        if timings is not None:
            decode_started = synchronized_time(device)
            if len(receiver_decode_ids) > 1:
                timings["receiver_prefill_ms"] = (decode_started - prefill_started) * 1000
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([[current_id]], dtype=torch.long, device=device)
            position_ids = torch.tensor([[logical_position]], dtype=torch.long, device=device)
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            if len(output_ids) < min_new_tokens and stop_ids:
                logits = logits.clone()
                logits[:, list(stop_ids)] = torch.finfo(logits.dtype).min
            next_id = int(logits.argmax(dim=-1).item())
            if next_id in stop_ids:
                stop_reason = "eos"
                break
            output_ids.append(next_id)
            current_id = next_id
            cache = output.past_key_values
            logical_position += 1

        if timings is not None:
            timings["receiver_decode_ms"] = (synchronized_time(device) - decode_started) * 1000
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return text, stop_reason, len(output_ids)


def generate_from_beta_cache(model, tokenizer, cache, betas, receiver_decode_ids,
                             logical_position, device, max_new_tokens, min_new_tokens=0,
                             timings=None, attention_backend="eager", decode_backend="eager"):
    from functools import partial
    decoder = partial(_generate_from_beta_cache_eager, attention_backend=attention_backend)
    if decode_backend == "torchscript_prefill":
        if attention_backend != "eager":
            raise ValueError("TorchScript AM decoding currently requires eager attention")
        from types import SimpleNamespace
        from multiagent.model.am_jit_decode import make_jit_decoder
        interface = SimpleNamespace(beta_attention=partial(beta_attention, backend="eager"),
                                    eos_ids=eos_ids, synchronized_time=synchronized_time)
        decoder = make_jit_decoder(decoder, interface, with_prefill=True)
        return decoder(model, tokenizer, cache, betas, receiver_decode_ids, logical_position,
                       device, max_new_tokens, min_new_tokens, timings)
    if decode_backend != "eager":
        raise ValueError(f"Unknown AM decode backend: {decode_backend}")
    return decoder(model, tokenizer, cache, betas, receiver_decode_ids, logical_position,
                   device, max_new_tokens, min_new_tokens, timings)


@torch.no_grad()
def generate_attention_matching(
    model,
    tokenizer,
    sample: C2CExample,
    device: str,
    max_length: int,
    max_new_tokens: int,
    am_config: AttentionMatchingConfig,
    min_new_tokens: int = 0,
    context_cache: dict[str, Any] | None = None,
) -> tuple[str, str, int]:
    context_id = str(sample.metadata.get("context_id", ""))
    receiver_decode_ids = chat_prompt_ids(
        tokenizer,
        sample.receiver_messages,
        max_length,
        add_generation_prompt=True,
    )
    position_reserve = max(WINDOW_RESERVE, len(receiver_decode_ids) + max_new_tokens)
    cache_id = (
        f"{context_id}|position_reserve={position_reserve}"
        if context_id and am_config.pack == "fractional"
        else context_id
    )
    compacted = context_cache.get(cache_id) if context_cache is not None and cache_id else None
    if compacted is None:
        compacted = compact_attention_matching_prefix(
            model,
            tokenizer,
            sample,
            device,
            max_length,
            am_config,
        )
        if compacted is not None and context_cache is not None and cache_id:
            context_cache.clear()
            context_cache[cache_id] = compacted
    if compacted is None:
        return "", "empty_prompt", 0
    (
        base_cache,
        betas,
        _,
        sender_len,
        _compacted_receiver_decode_ids,
        compact_entries,
        cache_stats,
        am_span,
    ) = compacted
    cache = copy.deepcopy(base_cache) if context_cache is not None and cache_id else base_cache
    if am_config.pack == "fractional":
        assert am_config.packing_window is not None
        projected_end = sender_len + len(receiver_decode_ids) + max_new_tokens
        if projected_end > am_config.packing_window:
            raise RuntimeError(
                f"fractional global AM exceeds window: decode {sender_len} + receiver "
                f"{len(receiver_decode_ids)} + generation {max_new_tokens} = {projected_end} "
                f"> {am_config.packing_window}"
            )
    text, stop_reason, generated_tokens = generate_from_beta_cache(
        model,
        tokenizer,
        cache,
        betas,
        receiver_decode_ids,
        sender_len,
        device,
        max_new_tokens,
        min_new_tokens,
    )
    return (
        text,
        f"{stop_reason}:am{compact_entries}active/"
        f"{cache_stats['allocated_entries']}allocated/{sender_len}+recv{len(receiver_decode_ids)}"
        + am_span,
        generated_tokens,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate native receiver-cache Attention Matching")
    add_eval_args(parser)
    add_am_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--am-target-ratio", type=float, required=True)
    parser.add_argument("--adapter", help="PEFT/LoRA adapter directory merged into the --model weights at load time (vanilla evaluation of a fine-tuned model).")
    parser.add_argument("--sample-ids", help="JSON list of sample ids; every other scheduled sample is skipped.")
    parser.add_argument(
        "--am-pack",
        choices=["original", "fractional"],
        default="original",
        help=(
            "Position geometry after receiver-only global AM. 'original' keeps the full-prefill "
            "RoPE positions; 'fractional' re-packs the selected cache below --am-pack-window."
        ),
    )
    parser.add_argument(
        "--am-pack-window",
        type=int,
        help="Position window for --am-pack fractional (for the assumed-8K Qwen control, use 8192).",
    )
    args = parser.parse_args(argv)
    validate_eval_args(parser, args)
    validate_am_args(parser, args)
    if args.am_pack == "fractional":
        if args.am_pack_window is None or args.am_pack_window <= WINDOW_RESERVE:
            parser.error(f"--am-pack fractional requires --am-pack-window greater than {WINDOW_RESERVE}")
    elif args.am_pack_window is not None:
        parser.error("--am-pack-window requires --am-pack fractional")
    return args


def main(argv=None):
    args = parse_args(argv)
    command_line = sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
    started_at = time.time()
    dtype = dtype_from_args(args)
    am_config = make_am_config(args, pack=args.am_pack, packing_window=args.am_pack_window)
    model_path = resolve_model_path(args.model)
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, dtype, args.device, adapter=args.adapter)
    tasks = {spec: load_c2c_task(spec, root=args.dataset_root) for spec in args.tasks}
    datasets = {spec: task.dataset for spec, task in tasks.items()}
    for spec, dataset in datasets.items():
        if len(dataset) and not str(dataset[0].target_text or "").strip():
            raise ValueError(f"{spec} has no gold target; use a labeled split")
    total = sum(len(d) if args.limit_per_dataset is None else min(len(d), args.limit_per_dataset)
                for d in datasets.values())
    json_compactor = EvalJsonCompactor(tokenizer)
    run_metadata = {
        "mode": "attention_matching",
        "receiver_model": str(model_path), "tasks": list(datasets),
        "attention_matching": am_metadata(args, am_config),
        "evaluation_config": evaluation_config_metadata(args, dtype=dtype, command_line=command_line, script=__file__),
        "saved_json_format": eval_json_format_metadata(),
    }
    options = dict(model=model, tokenizer=tokenizer, device=args.device, max_length=args.max_length,
                   max_new_tokens=args.max_new_tokens, am_config=am_config,
                   min_new_tokens=args.am_min_new_tokens, context_cache=BoundedContextCache(args.context_cache_max))
    generate = partial(generate_attention_matching, **options)

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

    progress = tqdm(
        interleaved_items(datasets, args.chunk_size, args.limit_per_dataset, args.seed),
        total=total, desc="eval/native-am",
        bar_format="{desc}: {n_fmt}/{total_fmt}|{percentage:2.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}{postfix}",
    )
    for position, (task_key, index, sample) in enumerate(progress, start=1):
        if (task_key, int(index)) in resumed_keys or (sample_ids is not None and sample.id not in sample_ids):
            continue
        context_seed = seed_attention_matching_context(args.seed, task_key, index, sample, args.device,
                                                      seed_fn=deterministic_am_context_seed)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        output, stop_reason, generated_tokens = generate(sample=sample)
        result, metrics, raw_grade = count_grade(task_key, tasks[task_key].grade(output, sample), stop_reason)
        if args.print_every > 0 and position % args.print_every == 0:
            print_eval_sample(position, total, task_key, sample, output, result, metrics)
        rows.append({
            "task": sample.dataset, "task_key": task_key, "index": index,
            **compact_eval_sample(sample, json_compactor),
            "grade": result, "grade_result": raw_grade, "output": output,
            "stop_reason": stop_reason, "generated_tokens": generated_tokens,
            "am_context_seed": context_seed, "time": time.time(),
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
