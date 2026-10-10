from __future__ import annotations

import argparse
from functools import partial
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c import chat_prompt_ids, generation_prompt_ids
from multiagent.dataset.c2c_cross_family import encode_context_pair
from multiagent.dataset.c2c_registry import load_c2c_task
from multiagent.dataset.c2c_types import C2CExample
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
    _forward_last_logits,
    _generate_batch,
    _greedy_decode_batch,
    _pad_token_id,
    add_eval_args,
    add_renderer_args,
    dtype_from_args,
    eos_ids,
    interleaved_items,
    load_model,
    load_tokenizer,
    validate_eval_args,
)
from multiagent.model.unaware_comm import _base_model, full_sender_cache, tuple_to_dynamic_cache
from multiagent.model.vanilla_loader import resolve_model_path


@torch.no_grad()
def generate_cache_renderer(
    renderer,
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    sample: C2CExample,
    device: str,
    max_length: int,
    max_new_tokens: int,
    context_cache: dict[str, Any] | None = None,
    no_sender_memory: bool = False,
    causal_cross_attention: bool = False,
    decode_prefix: str = "question",
    position_mode: str = "interp",
    context_encoder=encode_context_pair,
) -> tuple[str, str, int]:
    from multiagent.model.cache_renderer import flatten_sender_cache, pred_to_cache
    from multiagent.train.train_utils import move_kv_cache

    token_alignment = getattr(renderer, "token_alignment", "native")
    if token_alignment == "c2c-longest":
        if no_sender_memory:
            raise ValueError("C2C-longest requires sender memory")
        causal_cross_attention = True
    context_id = str(sample.metadata.get("context_id", ""))
    prepared = context_cache.get(context_id) if context_cache is not None and context_id else None
    if prepared is None:
        context = context_encoder(
            sender_tokenizer, receiver_tokenizer, sample.sender_messages, max_length, position_mode, token_alignment=token_alignment
        )
        if not context["sender_ids"] or not context["receiver_ids"]:
            return "", "empty_prompt", 0
        sender_valid = torch.tensor([context.get("sender_valid", [1] * len(context["sender_ids"]))], dtype=torch.long, device=device)
        receiver_valid = torch.tensor([context.get("receiver_valid", [1] * len(context["receiver_ids"]))], dtype=torch.long, device=device)
        memory = virtual = None
        if not no_sender_memory:
            sender_device = str(next(sender_model.parameters()).device)
            sender_ids = torch.tensor([context["sender_ids"]], dtype=torch.long, device=sender_device)
            sender_positions = torch.arange(sender_ids.shape[1], device=sender_device).unsqueeze(0)
            sender_cache = sender_model(
                input_ids=sender_ids,
                attention_mask=sender_valid.to(sender_device),
                position_ids=sender_positions,
                use_cache=True,
                past_key_values=full_sender_cache(),
            ).past_key_values
            sender_cache = move_kv_cache(sender_cache, sender_device)
            memory = flatten_sender_cache(sender_model, sender_cache, sender_positions).to(device)
            del sender_cache
            virtual = torch.from_numpy(context["virtual_positions"][None].astype("float32")).to(device)
        receiver_ids = torch.tensor([context["receiver_ids"]], dtype=torch.long, device=device)
        receiver_positions = torch.arange(receiver_ids.shape[1], device=device).unsqueeze(0)
        limits = None
        if causal_cross_attention and not no_sender_memory:
            limits = torch.from_numpy(context["cross_causal_limits"][None]).to(device)
        embeds = receiver_model.get_input_embeddings()(receiver_ids)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            pred = renderer(embeds, receiver_positions, memory, virtual,
                            sender_valid if token_alignment != "native" else None, limits,
                            **({"receiver_mask": receiver_valid} if token_alignment != "native" else {}))
        pred_cache = pred_to_cache(
            receiver_model, pred, receiver_positions, next(receiver_model.parameters()).dtype
        )
        prepared = (pred_cache, receiver_ids.shape[1], receiver_valid)
        if context_cache is not None and context_id:
            context_cache.clear()
            context_cache[context_id] = prepared
    pred_cache, receiver_len, receiver_valid = prepared

    if decode_prefix == "generation":
        receiver_decode_ids = generation_prompt_ids(receiver_tokenizer)
    else:
        receiver_decode_ids = chat_prompt_ids(
            receiver_tokenizer, sample.receiver_messages, max_length, add_generation_prompt=True
        )
    if not receiver_decode_ids:
        return "", "empty_prompt", 0
    cache = tuple_to_dynamic_cache(pred_cache, receiver_model.config)

    current_id = receiver_decode_ids[-1]
    attention_len = receiver_len
    if len(receiver_decode_ids) > 1:
        prefix = torch.tensor([receiver_decode_ids[:-1]], dtype=torch.long, device=device)
        attention_len += prefix.shape[1]
        cache = receiver_model(
            input_ids=prefix,
            attention_mask=torch.cat([receiver_valid, receiver_valid.new_ones((1, attention_len - receiver_len))], dim=1),
            position_ids=torch.arange(receiver_len, receiver_len + prefix.shape[1], device=device).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        ).past_key_values

    output_ids = []
    stop_reason = "cutoff"
    stop_ids = eos_ids(receiver_tokenizer)
    next_position = receiver_len + len(receiver_decode_ids) - 1
    for _ in range(max_new_tokens):
        attention_len += 1
        output = receiver_model(
            input_ids=torch.tensor([[current_id]], dtype=torch.long, device=device),
            attention_mask=torch.cat([receiver_valid, receiver_valid.new_ones((1, attention_len - receiver_len))], dim=1),
            position_ids=torch.tensor([[next_position]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )
        next_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
        if next_id in stop_ids:
            stop_reason = "eos"
            break
        output_ids.append(next_id)
        current_id = next_id
        cache = output.past_key_values
        next_position += 1

    return (
        receiver_tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
        f"{stop_reason}:renderer{receiver_len}recv+{len(receiver_decode_ids)}",
        len(output_ids),
    )

@torch.no_grad()
def _prepare_rendered_cache_batch(
    renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, contexts, device,
    no_sender_memory, causal_cross_attention, token_alignment,
):
    from multiagent.model.cache_renderer import bound_to_stats, flatten_sender_cache, pred_to_cache
    from multiagent.train.train_utils import move_kv_cache

    profile = os.environ.get("MULTIKV_PROFILE") == "1"
    marks = []

    def mark(name):
        if profile:
            torch.cuda.synchronize()
            marks.append((name, time.perf_counter()))

    mark("start")
    batch = len(contexts)
    r_lens = [len(c["receiver_ids"]) for c in contexts]
    r_max = max(r_lens)
    receiver_ids = torch.full((batch, r_max), _pad_token_id(receiver_tokenizer), dtype=torch.long, device=device)
    receiver_valid = torch.zeros((batch, r_max), dtype=torch.long, device=device)
    limits = torch.zeros((batch, r_max), dtype=torch.long, device=device) if causal_cross_attention else None
    for row, c in enumerate(contexts):
        n = r_lens[row]
        receiver_ids[row, :n] = torch.tensor(c["receiver_ids"], dtype=torch.long, device=device)
        receiver_valid[row, :n] = torch.tensor(c.get("receiver_valid", [1] * n), dtype=torch.long, device=device)
        if limits is not None:
            limits[row, :n] = torch.from_numpy(c["cross_causal_limits"]).to(device=device, dtype=torch.long)
    receiver_positions = torch.arange(r_max, device=device).unsqueeze(0).expand(batch, r_max).contiguous()
    memory = virtual = memory_mask = None
    if not no_sender_memory:
        sender_device = str(next(sender_model.parameters()).device)
        s_lens = [len(c["sender_ids"]) for c in contexts]
        s_max = max(s_lens)
        sender_ids = torch.full((batch, s_max), _pad_token_id(sender_tokenizer), dtype=torch.long, device=sender_device)
        sender_mask = torch.zeros((batch, s_max), dtype=torch.long, device=sender_device)
        sender_positions = torch.zeros((batch, s_max), dtype=torch.long, device=sender_device)
        virtual = torch.zeros((batch, s_max), dtype=torch.float32, device=device)
        for row, c in enumerate(contexts):
            n = s_lens[row]
            sender_ids[row, :n] = torch.tensor(c["sender_ids"], dtype=torch.long, device=sender_device)
            sender_mask[row, :n] = torch.tensor(c.get("sender_valid", [1] * n), dtype=torch.long, device=sender_device)
            sender_positions[row, :n] = torch.arange(n, device=sender_device)
            vp = torch.from_numpy(c["virtual_positions"].astype("float32")).to(device)
            virtual[row, :vp.numel()] = vp
        mark("pad")
        sender_cache = sender_model(
            input_ids=sender_ids, attention_mask=sender_mask, position_ids=sender_positions,
            use_cache=True, past_key_values=full_sender_cache(),
        ).past_key_values
        mark("sender_prefill")
        sender_cache = move_kv_cache(sender_cache, sender_device)
        memory = flatten_sender_cache(sender_model, sender_cache, sender_positions).to(device)
        del sender_cache
        memory_mask = sender_mask.to(device)
        mark("flatten+move")
    embeds = receiver_model.get_input_embeddings()(receiver_ids)
    bound = bool(getattr(renderer, "bound_stats", False))
    try:
        if bound:
            renderer.bound_stats = False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            pred = renderer(embeds, receiver_positions, memory, virtual, memory_mask, limits,
                            **({"receiver_mask": receiver_valid} if token_alignment != "native" else {}))
    finally:
        renderer.bound_stats = bound
    mark("renderer")
    if bound:
        pred = bound_to_stats(pred, renderer.stat_key_rms, renderer.stat_value_rms,
                              getattr(renderer, "bound_exact_grad", False), receiver_valid)
    pred_cache = pred_to_cache(receiver_model, pred, receiver_positions, next(receiver_model.parameters()).dtype)
    mark("pred_to_cache")
    if profile:
        print("profile prepare B=%d S=%d R=%d: " % (batch, memory.shape[1] if memory is not None else 0, r_max)
              + " ".join(f"{b}={1000*(tb-ta):.0f}ms" for (a, ta), (b, tb) in zip(marks, marks[1:])), flush=True)
    return pred_cache, r_lens, receiver_valid

@torch.no_grad()
def generate_cache_renderer_batch(
    renderer,
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    samples: list,
    device: str,
    max_length: int,
    max_new_tokens: int,
    no_sender_memory: bool = False,
    causal_cross_attention: bool = False,
    decode_prefix: str = "question",
    position_mode: str = "interp",
    context_encoder=encode_context_pair,
) -> list[tuple[str, str, int]]:
    token_alignment = getattr(renderer, "token_alignment", "native")
    if token_alignment == "c2c-longest":
        if no_sender_memory:
            raise ValueError("C2C-longest requires sender memory")
        causal_cross_attention = True
    results: list = [None] * len(samples)
    live, contexts, decode_lists = [], [], []
    for i, sample in enumerate(samples):
        context = context_encoder(
            sender_tokenizer, receiver_tokenizer, sample.sender_messages, max_length, position_mode, token_alignment=token_alignment
        )
        if decode_prefix == "generation":
            decode_ids = generation_prompt_ids(receiver_tokenizer)
        else:
            decode_ids = chat_prompt_ids(receiver_tokenizer, sample.receiver_messages, max_length, add_generation_prompt=True)
        if not context["sender_ids"] or not context["receiver_ids"] or not decode_ids:
            results[i] = ("", "empty_prompt", 0)
            continue
        live.append(i)
        contexts.append(context)
        decode_lists.append(decode_ids)
    if not live:
        return results
    pred_cache, r_lens, receiver_valid = _prepare_rendered_cache_batch(
        renderer, sender_model, receiver_model, sender_tokenizer, receiver_tokenizer, contexts, device,
        no_sender_memory, causal_cross_attention, token_alignment,
    )
    batch, r_max = receiver_valid.shape
    pad_id = _pad_token_id(receiver_tokenizer)
    l_max = max(len(d) for d in decode_lists)
    attention = torch.cat([receiver_valid, torch.zeros((batch, l_max), dtype=torch.long, device=device)], dim=1)
    input_ids = torch.full((batch, l_max), pad_id, dtype=torch.long, device=device)
    positions = torch.zeros((batch, l_max), dtype=torch.long, device=device)
    next_positions = torch.zeros((batch,), dtype=torch.long, device=device)
    for row, decode_ids in enumerate(decode_lists):
        start = l_max - len(decode_ids)
        attention[row, r_max + start:] = 1
        input_ids[row, start:] = torch.tensor(decode_ids, dtype=torch.long, device=device)
        positions[row, start:] = torch.arange(r_lens[row], r_lens[row] + len(decode_ids), device=device)
        next_positions[row] = r_lens[row] + len(decode_ids)
    profile = os.environ.get("MULTIKV_PROFILE") == "1"
    if profile:
        torch.cuda.synchronize(); t0 = time.perf_counter()
    cache = tuple_to_dynamic_cache(pred_cache, receiver_model.config)
    output = _forward_last_logits(
        receiver_model,
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
    )
    next_ids = output.logits[:, -1, :].argmax(dim=-1)
    if profile:
        torch.cuda.synchronize(); t1 = time.perf_counter()
    generated, reasons = _greedy_decode_batch(
        receiver_model, receiver_tokenizer, output.past_key_values, attention, next_ids, next_positions, max_new_tokens, device
    )
    if profile:
        torch.cuda.synchronize()
        print(f"profile decode B={batch}: prefix={1000*(t1-t0):.0f}ms decode={1000*(time.perf_counter()-t1):.0f}ms steps<={max(len(g) for g in generated)+1}", flush=True)
    for row, i in enumerate(live):
        results[i] = (
            receiver_tokenizer.decode(generated[row], skip_special_tokens=True).strip(),
            f"{reasons[row]}:renderer{r_lens[row]}recv+{len(decode_lists[row])}",
            len(generated[row]),
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate uncompressed rendered receiver caches")
    add_eval_args(parser)
    add_renderer_args(parser)
    parser.add_argument("--sender-device", help="Separate sender device; --device holds the receiver and renderer.")
    parser.add_argument("--context-cache-max", type=int, default=0,
                        help="Max prepared context caches held on device (0 = unbounded).")
    args = parser.parse_args(argv)
    validate_eval_args(parser, args)
    return args


def main(argv=None):
    args = parse_args(argv)
    command_line = sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
    started_at = time.time()
    dtype = dtype_from_args(args)
    from multiagent.model.cache_renderer import load_renderer_checkpoint

    renderer_file = Path(args.renderer)
    if renderer_file.is_dir():
        renderer_file = renderer_file / "renderer.pt"
    renderer, renderer_payload = load_renderer_checkpoint(renderer_file, args.device)
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
    sender_model = load_model(sender_path, dtype, args.sender_device or args.device, adapter=args.sender_adapter)
    receiver_model = load_model(receiver_path, dtype, args.device)
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
        "sender_device": args.sender_device or args.device,
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
    mode = "cache_renderer"
    run_metadata = {
        "mode": mode,
        **model_payload,
        "tasks": list(datasets),
        "evaluation_config": evaluation_config_metadata(args, dtype=dtype, command_line=command_line, script=__file__),
        "saved_json_format": eval_json_format_metadata(),
    }

    context_cache = BoundedContextCache(args.context_cache_max)
    generation_options = dict(device=args.device, max_length=args.max_length,
                              max_new_tokens=args.max_new_tokens)
    generation_options.update(
        renderer=renderer, sender_model=sender_model, receiver_model=receiver_model,
        sender_tokenizer=sender_tokenizer, receiver_tokenizer=receiver_tokenizer,
        causal_cross_attention=renderer_causal_cross_attention,
        decode_prefix=args.renderer_decode_prefix, position_mode=renderer_position_mode,
    )
    generation_options.update(no_sender_memory=renderer_no_sender_memory,
                              context_encoder=encode_context_pair)
    generate = partial(generate_cache_renderer, **generation_options, context_cache=context_cache)
    def generate_batch(samples):
        return generate_cache_renderer_batch(**generation_options, samples=samples)

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
        schedule, total=total, desc="eval/renderer",
        bar_format="{desc}: {n_fmt}/{total_fmt}|{percentage:2.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}{postfix}",
    )
    remaining = (
        (position, task_key, index, sample)
        for position, (task_key, index, sample) in enumerate(progress, start=1)
        if (task_key, int(index)) not in resumed_keys
        and (sample_ids is None or sample.id in sample_ids)
    )
    batched = args.batch_size > 1
    while items := list(islice(remaining, args.batch_size)):
        if batched:
            outputs = _generate_batch(generate_batch, [item[3] for item in items])
        else:
            outputs = [generate(sample=items[0][3])]

        if len(outputs) != len(items):
            raise ValueError(f"decoder returned {len(outputs)} answers for {len(items)} samples")
        for (position, task_key, index, sample), (output, stop_reason, generated_tokens) in zip(items, outputs):
            result, metrics, raw_grade = count_grade(task_key, tasks[task_key].grade(output, sample), stop_reason)
            if args.print_every > 0 and position % args.print_every == 0:
                print_eval_sample(position, total, task_key, sample, output, result, metrics)
            rows.append({
                "task": sample.dataset, "task_key": task_key, "index": index,
                **compact_eval_sample(sample, json_compactor),
                "grade": result, "grade_result": raw_grade, "output": output,
                "stop_reason": stop_reason, "generated_tokens": generated_tokens,
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
