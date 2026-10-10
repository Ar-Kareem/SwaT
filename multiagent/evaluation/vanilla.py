from __future__ import annotations

import argparse
import copy
from functools import partial
from itertools import islice
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
    _forward_last_logits,
    _generate_batch,
    _greedy_decode_batch,
    _pad_token_id,
    add_eval_args,
    dtype_from_args,
    eos_ids,
    interleaved_items,
    load_model,
    load_tokenizer,
    synchronized_time,
    validate_eval_args,
)
from multiagent.model.vanilla_loader import resolve_model_path

from multiagent.dataset.c2c import apply_chat_template, chat_prompt_ids
from multiagent.dataset.c2c_cross_family import require_fast, tokenize_with_offsets
from multiagent.dataset.c2c_types import C2CExample


@torch.no_grad()
def generate_vanilla(
    model,
    tokenizer,
    messages,
    device: str,
    max_length: int | None,
    max_new_tokens: int,
    timings: dict[str, Any] | None = None,
) -> tuple[str, str, int]:
    if timings is not None:
        timings.update(receiver_prefill_ms=0.0, receiver_decode_ms=0.0)
    ids = chat_prompt_ids(tokenizer, messages, max_length, add_generation_prompt=True)
    if not ids:
        return "", "empty_prompt", 0

    current_id = ids[-1]
    cache = None
    attention_len = 1
    prefill_started = synchronized_time(device) if timings is not None else None
    if len(ids) > 1:
        prefix_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        prefix_attention = torch.ones_like(prefix_ids)
        backbone = getattr(model, "base_model", model)
        cache = backbone(input_ids=prefix_ids, attention_mask=prefix_attention, use_cache=True).past_key_values
        attention_len = len(ids)

    if timings is not None:
        decode_started = synchronized_time(device)
        if len(ids) > 1:
            timings["receiver_prefill_ms"] = (decode_started - prefill_started) * 1000
    output_ids = []
    stop_reason = "cutoff"
    stop_ids = eos_ids(tokenizer)
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([[current_id]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, attention_len), dtype=torch.long, device=device)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        attention_len += 1

    if timings is not None:
        timings["receiver_decode_ms"] = (synchronized_time(device) - decode_started) * 1000
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip(), stop_reason, len(output_ids)


@torch.no_grad()
def generate_vanilla_batch(
    model,
    tokenizer,
    messages_list: list,
    device: str,
    max_length: int | None,
    max_new_tokens: int,
) -> list[tuple[str, str, int]]:
    prompts = [chat_prompt_ids(tokenizer, messages, max_length, add_generation_prompt=True) for messages in messages_list]
    results: list = [("", "empty_prompt", 0) if not ids else None for ids in prompts]
    live = [i for i, ids in enumerate(prompts) if ids]
    if not live:
        return results
    pad_id = _pad_token_id(tokenizer)
    width = max(len(prompts[i]) for i in live)
    input_ids = torch.full((len(live), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(live), width), dtype=torch.long, device=device)
    for row, i in enumerate(live):
        ids = prompts[i]
        input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, width - len(ids):] = 1
    positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    output = _forward_last_logits(
        model, input_ids=input_ids, attention_mask=attention_mask, position_ids=positions, use_cache=True
    )
    next_ids = output.logits[:, -1, :].argmax(dim=-1)
    generated, reasons = _greedy_decode_batch(
        model, tokenizer, output.past_key_values, attention_mask, next_ids, positions[:, -1] + 1, max_new_tokens, device
    )
    for row, i in enumerate(live):
        results[i] = (tokenizer.decode(generated[row], skip_special_tokens=True).strip(), reasons[row], len(generated[row]))
    return results


@torch.no_grad()
def generate_vanilla_stream(
    model,
    tokenizer,
    messages_list: list,
    device: str,
    max_length: int | None,
    max_new_tokens: int,
    batch_size: int,
) -> list[tuple[str, str, int]]:
    prompts = [chat_prompt_ids(tokenizer, messages, max_length, add_generation_prompt=True) for messages in messages_list]
    results: list = [("", "empty_prompt", 0) if not ids else None for ids in prompts]
    queue = sorted((i for i, ids in enumerate(prompts) if ids), key=lambda i: -len(prompts[i]))
    if not queue:
        return results
    pad_id = _pad_token_id(tokenizer)
    stop_ids = eos_ids(tokenizer)

    def prefill(items):
        w = max(len(prompts[i]) for i in items)
        ids = torch.full((len(items), w), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(items), w), dtype=torch.long, device=device)
        for row, i in enumerate(items):
            n = len(prompts[i])
            ids[row, w - n:] = torch.tensor(prompts[i], dtype=torch.long, device=device)
            mask[row, w - n:] = 1
        positions = (mask.cumsum(dim=1) - 1).clamp(min=0)
        out = _forward_last_logits(model, input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        return out.past_key_values, mask, out.logits[:, -1, :].argmax(dim=-1), positions[:, -1] + 1

    batch = min(batch_size, len(queue))
    first, queue = queue[:batch], queue[batch:]
    cache, mask, pending, positions = prefill(first)
    slots = list(first)
    generated = {i: [] for i in first}

    def consume(rows):
        for row in rows:
            i = slots[row]
            if i is None:
                continue
            token = int(pending[row])
            if token in stop_ids:
                results[i] = (tokenizer.decode(generated[i], skip_special_tokens=True).strip(), "eos", len(generated[i]))
                slots[row] = None
                continue
            generated[i].append(token)
            if len(generated[i]) >= max_new_tokens:
                results[i] = (tokenizer.decode(generated[i], skip_special_tokens=True).strip(), "cutoff", len(generated[i]))
                slots[row] = None

    consume(range(len(slots)))
    while True:
        free = [row for row, i in enumerate(slots) if i is None]
        while free and queue and (len(free) >= max(1, len(slots) // 4) or len(queue) <= len(free)):
            take, queue = queue[:len(free)], queue[len(free):]
            sub_cache, sub_mask, sub_pending, sub_positions = prefill(take)
            total = mask.shape[1]
            sub_total = sub_mask.shape[1]
            for layer, sub_layer in zip(cache.layers, sub_cache.layers):
                for name in ("keys", "values"):
                    main, sub = getattr(layer, name), getattr(sub_layer, name)
                    if main is None or sub is None:
                        continue
                    t, st = main.shape[2], sub.shape[2]
                    for row, j in zip(free, range(len(take))):
                        main[row, :, : t - st, :] = 0
                        main[row, :, t - st:, :] = sub[j]
            filled = free[: len(take)]
            for row, j in zip(filled, range(len(take))):
                mask[row, : total - sub_total] = 0
                mask[row, total - sub_total:] = sub_mask[j]
                pending[row] = sub_pending[j]
                positions[row] = sub_positions[j]
                slots[row] = take[j]
                generated[take[j]] = []
            del sub_cache
            consume(filled)
            free = [row for row, i in enumerate(slots) if i is None]
        active = [i is not None for i in slots]
        if not any(active):
            break
        tokens = torch.tensor([[int(pending[row]) if active[row] else pad_id] for row in range(len(slots))],
                              dtype=torch.long, device=device)
        mask = torch.cat([mask, torch.tensor(active, dtype=mask.dtype, device=device).unsqueeze(1)], dim=1)
        out = _forward_last_logits(model, input_ids=tokens, attention_mask=mask, position_ids=positions.unsqueeze(1),
                                   past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        pending = out.logits[:, -1, :].argmax(dim=-1)
        positions = positions + 1
        consume(range(len(slots)))
    return results


def vanilla_messages(sample: C2CExample, tokenizer, max_length: int) -> list[dict[str, str]]:
    messages = sample.messages or sample.phase1_messages
    context = str(sample.metadata.get("context", ""))
    if not sample.sender_messages or not context:
        if len(chat_prompt_ids(tokenizer, messages, None, add_generation_prompt=True)) > max_length:
            raise ValueError(f"{sample.id} exceeds --max-length and has no separable context to cut")
        return messages
    sender_text = apply_chat_template(tokenizer, sample.sender_messages, add_generation_prompt=False)
    _, offsets = tokenize_with_offsets(require_fast(tokenizer, "vanilla"), sender_text)
    if len(offsets) <= max_length:
        return messages
    stripped = context.strip()
    if not stripped or sender_text.count(stripped) != 1:
        raise ValueError(f"expected context exactly once in the templated context messages for {sample.id}")
    lead = context.index(stripped)
    keep = offsets[max_length - 1][1] - sender_text.index(stripped) + lead
    if keep <= lead:
        raise ValueError(f"the context of {sample.id} was fully removed by the context cut")
    if keep >= lead + len(stripped):
        return messages
    cut = copy.deepcopy(messages)
    replacements = 0
    for message in cut:
        content = str(message.get("content", ""))
        if context in content:
            message["content"] = content.replace(context, context[:keep], 1)
            replacements += 1
    if replacements != 1:
        raise ValueError(f"expected context exactly once in vanilla messages for {sample.id}, found {replacements}")
    return cut


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a plain receiver model on mixed QA datasets")
    add_eval_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="PEFT/LoRA adapter directory merged into the --model weights at load time (vanilla evaluation of a fine-tuned model).")
    parser.add_argument("--sample-ids", help="JSON list of sample ids; every other scheduled sample is skipped.")
    parser.add_argument("--batch-refill", action="store_true",
                        help="Continuous batching for the plain --model path: finished rows are refilled with new "
                             "prompts so a batch is not held back by its slowest answer (requires --batch-size > 1).")
    args = parser.parse_args(argv)
    validate_eval_args(parser, args)
    return args


def main(argv=None):
    args = parse_args(argv)
    command_line = sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
    started_at = time.time()
    dtype = dtype_from_args(args)
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
        "mode": "receiver_only", "receiver_model": str(model_path), "tasks": list(datasets),
        "evaluation_config": evaluation_config_metadata(args, dtype=dtype, command_line=command_line, script=__file__),
        "saved_json_format": eval_json_format_metadata(),
    }
    generation_options = dict(model=model, tokenizer=tokenizer, device=args.device,
                              max_length=None, max_new_tokens=args.max_new_tokens)
    batch_decoder = (partial(generate_vanilla_stream, batch_size=args.batch_size)
                     if args.batch_refill else generate_vanilla_batch)
    def generate_batch(samples):
        return batch_decoder(**generation_options,
                             messages_list=[vanilla_messages(s, tokenizer, args.max_length) for s in samples])

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
        total=total, desc="eval/receiver",
        bar_format="{desc}: {n_fmt}/{total_fmt}|{percentage:2.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}{postfix}",
    )
    remaining = (
        (position, task_key, index, sample)
        for position, (task_key, index, sample) in enumerate(progress, start=1)
        if (task_key, int(index)) not in resumed_keys
        and (sample_ids is None or sample.id in sample_ids)
    )
    window = args.batch_size * (16 if args.batch_refill and args.batch_size > 1 else 1)
    while items := list(islice(remaining, window)):
        if args.batch_size > 1:
            outputs = _generate_batch(generate_batch, [item[3] for item in items])
        else:
            outputs = [generate_vanilla(**generation_options,
                                       messages=vanilla_messages(items[0][3], tokenizer, args.max_length))]
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
                "stop_reason": stop_reason, "generated_tokens": generated_tokens, "time": time.time(),
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
