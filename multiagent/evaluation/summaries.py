from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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

from multiagent.dataset.c2c_registry import load_c2c_task
from multiagent.dataset.c2c import apply_chat_template
from multiagent.dataset.c2c_types import C2CExample
from multiagent.evaluation.utils import (
    CompressionJournal,
    FirstForwardTimer,
    add_eval_args,
    dtype_from_args,
    eos_ids,
    interleaved_items,
    load_model,
    load_tokenizer,
    summarization_context_id,
    synchronized_time,
    validate_eval_args,
)
from multiagent.evaluation.results import (
    EvalJsonCompactor,
    eval_json_format_metadata,
    evaluation_config_metadata,
    compact_eval_sample,
    human_elapsed,
    normalize_grade_result,
    print_eval_sample,
    save_json,
    score_postfix,
    summarize,
    update_metric_stats,
)
from multiagent.evaluation.vanilla import (
    generate_vanilla,
    generate_vanilla_batch,
)
from multiagent.model.vanilla_loader import resolve_model_path


SUMMARY_PROMPTS = {
    "summarize": "Summarize the following text:\n\n{article_text}\n\nSummary:",
    "summarize_indepth": "Summarize the following text in depth:\n\n{article_text}\n\nSummary:",
    "summarize_keypoints_questions": (
        "Summarize the following text, providing all key points that might be necessary "
        "to answer comprehension questions:\n\n{article_text}\n\nSummary:"
    ),
    "summarize_concise": "Summarize the following text concisely:\n\n{article_text}\n\nSummary:",
    "summarize_very_concise": "Summarize the following text very concisely:\n\n{article_text}\n\nSummary:",
}


def load_precomputed_summaries(path: Path, receiver_tokenizer) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload.get("prompts"), dict) or "summarizer_model" not in payload:
        raise ValueError(f"--summaries-from {path} must contain 'summarizer_model' and 'prompts'")
    for prompt_name, contexts in payload["prompts"].items():
        if prompt_name not in SUMMARY_PROMPTS:
            raise ValueError(f"--summaries-from {path}: unknown summary prompt {prompt_name!r}")
        for context_id, record in contexts.items():
            text = record.get("text")
            if not isinstance(text, str) or not text or "context_sha256" not in record:
                raise ValueError(
                    f"--summaries-from {path}: {prompt_name}/{context_id} needs 'text' and 'context_sha256'"
                )
            record["summary_tokens_receiver"] = len(
                receiver_tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            record.pop("original_tokens_receiver", None)
            record.pop("retained_fraction_receiver", None)
    return payload


def messages_with_summarized_context(sample: C2CExample, summary: str) -> list[dict[str, str]]:
    if sample.prompt_mode != "split_prompt":
        raise ValueError("summarization baseline requires split_prompt examples")
    context = str(sample.metadata.get("context", ""))
    if not context:
        raise ValueError(f"sample {sample.id} has no metadata context to summarize")

    sender_messages = copy.deepcopy(sample.sender_messages)
    replacements = 0
    for message in sender_messages:
        content = str(message.get("content", ""))
        if context in content:
            message["content"] = content.replace(context, summary, 1)
            replacements += 1
    if replacements != 1:
        raise ValueError(
            f"expected context exactly once in sender messages for {sample.id}, found {replacements}"
        )
    return [*sender_messages, *copy.deepcopy(sample.receiver_messages)]


@torch.no_grad()
def summarize_context(
    model,
    tokenizer,
    receiver_tokenizer,
    context: str,
    prompt_name: str,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    preparation_started = synchronized_time(device)
    prompt = SUMMARY_PROMPTS[prompt_name].format(article_text=context)
    prompt_text = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] >= max_length:
        raise ValueError(
            f"summarizer prompt is {encoded['input_ids'].shape[1]} tokens, which does not fit "
            f"--summarizer-max-length={max_length}; use a longer-context summarizer or raise the limit"
        )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    original_tokens_summarizer = len(tokenizer(context, add_special_tokens=False)["input_ids"])
    available_tokens = max_length - input_ids.shape[1]
    generation_limit = min(original_tokens_summarizer, available_tokens)
    if generation_limit <= 0:
        raise ValueError(
            f"summarizer prompt occupies {input_ids.shape[1]} tokens, leaving no room under "
            f"--summarizer-max-length={max_length}"
        )

    with FirstForwardTimer(model, device) as prefill:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_limit,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = generated[0, input_ids.shape[1] :]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not summary:
        raise RuntimeError("summarizer generated an empty summary")

    summary_tokens_summarizer = len(tokenizer(summary, add_special_tokens=False)["input_ids"])
    original_tokens_receiver = len(receiver_tokenizer(context, add_special_tokens=False)["input_ids"])
    summary_tokens_receiver = len(receiver_tokenizer(summary, add_special_tokens=False)["input_ids"])
    preparation_ended = synchronized_time(device)
    return {
        "text": summary,
        "prompt": prompt_name,
        "original_tokens_summarizer": original_tokens_summarizer,
        "summary_tokens_summarizer": summary_tokens_summarizer,
        "original_tokens_receiver": original_tokens_receiver,
        "summary_tokens_receiver": summary_tokens_receiver,
        "retained_fraction_receiver": (
            summary_tokens_receiver / original_tokens_receiver
            if original_tokens_receiver
            else 0.0
        ),
        "generation_limit": generation_limit,
        "generated_tokens_including_special": int(generated_ids.numel()),
        "preparation_ms": (preparation_ended - preparation_started) * 1000,
        "sender_prefill_ms": (prefill.end - preparation_started) * 1000,
        "sender_am_ms": 0.0,
        "sender_decode_ms": (preparation_ended - prefill.end) * 1000,
    }


@torch.no_grad()
def summarize_contexts_batched(
    model,
    tokenizer,
    receiver_tokenizer,
    contexts: list[tuple[str, str]],
    prompt_name: str,
    device: str,
    max_length: int,
    batch_size: int,
    on_context: Callable[[str, dict[str, Any]], None] | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, dict[str, Any]]:
    entries = []
    for context_id, context in contexts:
        prompt = SUMMARY_PROMPTS[prompt_name].format(article_text=context)
        prompt_text = apply_chat_template(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(ids) >= max_length:
            raise ValueError(
                f"summarizer prompt is {len(ids)} tokens, which does not fit "
                f"--summarizer-max-length={max_length}; use a longer-context summarizer or raise the limit"
            )
        original_tokens_summarizer = len(tokenizer(context, add_special_tokens=False)["input_ids"])
        generation_limit = min(original_tokens_summarizer, max_length - len(ids))
        if max_new_tokens is not None:
            generation_limit = min(generation_limit, max_new_tokens)
        if generation_limit <= 0:
            raise ValueError(
                f"summarizer prompt occupies {len(ids)} tokens, leaving no room under "
                f"--summarizer-max-length={max_length}"
            )
        entries.append({
            "context_id": context_id,
            "context": context,
            "ids": ids,
            "original_tokens_summarizer": original_tokens_summarizer,
            "generation_limit": generation_limit,
        })

    entries.sort(key=lambda entry: len(entry["ids"]), reverse=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    stop_ids = eos_ids(tokenizer)
    records: dict[str, dict[str, Any]] = {}
    progress = tqdm(
        range(0, len(entries), batch_size),
        desc=f"summarize batched prompt={prompt_name}",
    )
    for start in progress:
        preparation_started = synchronized_time(device)
        batch = entries[start : start + batch_size]
        width = max(len(entry["ids"]) for entry in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, entry in enumerate(batch):
            ids = entry["ids"]
            input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, width - len(ids):] = 1
        with FirstForwardTimer(model, device) as prefill:
            generated = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max(entry["generation_limit"] for entry in batch),
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_part = generated[:, width:]
        preparation_ended = synchronized_time(device)
        preparation_batch_ms = (preparation_ended - preparation_started) * 1000
        for row, entry in enumerate(batch):
            generated_ids = generated_part[row][: entry["generation_limit"]]
            for position, token in enumerate(generated_ids.tolist()):
                if token in stop_ids:
                    generated_ids = generated_ids[: position + 1]
                    break
                if token == pad_id:
                    generated_ids = generated_ids[:position]
                    break
            summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not summary:
                raise RuntimeError(
                    f"summarizer generated an empty summary for context {entry['context_id']}"
                )
            original_tokens_receiver = len(
                receiver_tokenizer(entry["context"], add_special_tokens=False)["input_ids"]
            )
            summary_tokens_receiver = len(
                receiver_tokenizer(summary, add_special_tokens=False)["input_ids"]
            )
            records[entry["context_id"]] = {
                "text": summary,
                "prompt": prompt_name,
                "original_tokens_summarizer": entry["original_tokens_summarizer"],
                "summary_tokens_summarizer": len(
                    tokenizer(summary, add_special_tokens=False)["input_ids"]
                ),
                "original_tokens_receiver": original_tokens_receiver,
                "summary_tokens_receiver": summary_tokens_receiver,
                "retained_fraction_receiver": (
                    summary_tokens_receiver / original_tokens_receiver
                    if original_tokens_receiver
                    else 0.0
                ),
                "generation_limit": entry["generation_limit"],
                "generated_tokens_including_special": int(generated_ids.numel()),
                "preparation_batch_ms": preparation_batch_ms,
                "sender_prefill_batch_ms": (prefill.end - preparation_started) * 1000,
                "sender_am_batch_ms": 0.0,
                "sender_decode_batch_ms": (preparation_ended - prefill.end) * 1000,
                "preparation_batch_context_ids": [item["context_id"] for item in batch],
            }
            if on_context is not None:
                on_context(entry["context_id"], records[entry["context_id"]])
    return records


def evaluate_summarization_prompts(
    *,
    args,
    model,
    tokenizer,
    summarizer_model,
    summarizer_tokenizer,
    summarizer_path: Path,
    tasks: dict,
    datasets: dict,
    run_metadata: dict[str, Any],
    json_compactor: EvalJsonCompactor,
    precomputed_summaries: dict[str, Any] | None = None,
    started_at: float | None = None,
    generate_answer=generate_vanilla,
    save_result=save_json,
) -> None:
    started_at = time.time() if started_at is None else started_at
    journal = CompressionJournal(args.output, run_metadata)
    prompt_names = args.summary_prompts or (
        [args.summary_prompt] if args.summary_prompt is not None else list(SUMMARY_PROMPTS)
    )
    total_per_prompt = sum(
        len(dataset) if args.limit_per_dataset is None else min(len(dataset), args.limit_per_dataset)
        for dataset in datasets.values()
    )
    resume_payload: dict[str, Any] = {}
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"summarization resume JSON not found: {resume_path}")
        with resume_path.open() as f:
            resume_payload = json.load(f)
        if resume_payload.get("mode") != "summarization":
            raise ValueError(
                f"--resume-from must contain a summarization evaluation, got "
                f"mode={resume_payload.get('mode')!r}"
            )
        saved_tasks = resume_payload.get("tasks")
        if saved_tasks is not None and list(saved_tasks) != list(datasets):
            raise ValueError(
                f"resume task mismatch: saved={saved_tasks!r}, requested={list(datasets)!r}"
            )

    saved_results = resume_payload.get("results_by_prompt", {})
    results_by_prompt: dict[str, dict[str, Any]] = {
        prompt_name: saved_results[prompt_name]
        for prompt_name in prompt_names
        if prompt_name in saved_results
    }
    prior_elapsed_seconds = float(resume_payload.get("elapsed_seconds", 0.0) or 0.0)

    def current_payload() -> dict[str, Any]:
        elapsed = time.time() - started_at + prior_elapsed_seconds
        return {
            **run_metadata,
            "elapsed_seconds": round(elapsed, 3),
            "elapsed_human": human_elapsed(elapsed),
            "accuracies_by_prompt": {
                key: value["accuracy"]
                for key, value in results_by_prompt.items()
            },
            "results_by_prompt": results_by_prompt,
        }

    for prompt_index, prompt_name in enumerate(prompt_names, start=1):
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        schedule = list(
            interleaved_items(datasets, args.chunk_size, args.limit_per_dataset, args.seed)
        )
        if len(schedule) != total_per_prompt:
            raise RuntimeError(
                f"summarization schedule has {len(schedule)} rows, expected {total_per_prompt}"
            )
        stats = {spec: {"seen": 0, "correct": 0, "incorrect": 0, "fail": 0} for spec in datasets}
        stop_stats = {spec: {"eos": 0, "cutoff": 0, "empty_prompt": 0} for spec in datasets}
        metric_stats: dict[str, dict[str, dict[str, float]]] = {}
        existing_result = results_by_prompt.get(prompt_name)
        done_keys: set[tuple[Any, Any]] = set()
        if existing_result is None:
            saved_context_cache: dict[str, dict[str, Any]] = {}
            rows: list[dict[str, Any]] = []
            prompt_result = {
                "prompt": prompt_name,
                "accuracy": None,
                "achieved_retained_fraction_weighted": None,
                "achieved_retained_fraction_mean": None,
                "summary": summarize(stats, stop_stats, metric_stats),
                "summarized_contexts": saved_context_cache,
                "rows": rows,
            }
        else:
            prompt_result = existing_result
            rows = list(prompt_result.get("rows", []))
            saved_context_cache = dict(prompt_result.get("summarized_contexts", {}))
            prompt_result["rows"] = rows
            prompt_result["summarized_contexts"] = saved_context_cache
            scheduled_keys = {(task_key, sample.id) for task_key, _, sample in schedule}
            scheduled_shas = {
                (task_key, sample.id): sample.metadata.get("frozen_context_sha256")
                for task_key, _, sample in schedule
            }
            for saved_position, row in enumerate(rows, start=1):
                key = (row.get("task_key"), row.get("id"))
                if key not in scheduled_keys:
                    raise ValueError(
                        f"resume row {saved_position} of prompt {prompt_name!r} is not in this "
                        f"run's schedule: saved=(task_key, id)={key!r}"
                    )
                if key in done_keys:
                    raise ValueError(f"resume rows of prompt {prompt_name!r} contain {key!r} twice")
                saved_sha = row.get("frozen_context_sha256")
                current_sha = scheduled_shas.get(key)
                if saved_sha and current_sha and saved_sha != current_sha:
                    raise ValueError(
                        f"resume row {saved_position} of prompt {prompt_name!r} ({key!r}) was answered on a "
                        f"different context text than the one scheduled now"
                    )
                done_keys.add(key)
                task_key = row["task_key"]
                result = str(row.get("grade", "fail"))
                if result not in stats[task_key]:
                    stats[task_key][result] = 0
                stats[task_key]["seen"] += 1
                stats[task_key][result] += 1
                _, metrics, _ = normalize_grade_result(row.get("grade_result", {"grade": result}))
                update_metric_stats(metric_stats, task_key, metrics)
                stop_reason = str(row.get("stop_reason", "unknown"))
                stop_stats[task_key][stop_reason] = stop_stats[task_key].get(stop_reason, 0) + 1
            prompt_result["summary"] = summarize(stats, stop_stats, metric_stats)
            prompt_result["accuracy"] = prompt_result["summary"]["all"]["accuracy"]
        results_by_prompt[prompt_name] = prompt_result

        remaining = [item for item in schedule if (item[0], item[2].id) not in done_keys]
        completed_rows = len(schedule) - len(remaining)
        if not remaining:
            print(
                f"resume: skipping completed summarization prompt {prompt_name!r} "
                f"({completed_rows}/{total_per_prompt} rows)",
                flush=True,
            )
            continue

        context_cache: dict[str, dict[str, Any]] = {
            context_id: record
            for context_id, record in saved_context_cache.items()
            if isinstance(record, dict)
            and isinstance(record.get("text"), str)
            and "tokens omitted from saved evaluation JSON" not in record["text"]
        }

        def save_context(context_id: str, record: dict[str, Any]) -> None:
            context_cache[context_id] = record
            saved_context_cache[context_id] = record
            journal.write("summary_context", prompt=prompt_name,
                          context_id=context_id, record=record)

        if precomputed_summaries is not None:
            for context_id, record in precomputed_summaries["prompts"].get(prompt_name, {}).items():
                if context_id not in context_cache:
                    save_context(context_id, dict(record))
            missing = sorted({
                summarization_context_id(sample)
                for _, _, sample in remaining
                if summarization_context_id(sample) not in context_cache
            })
            if missing:
                raise ValueError(
                    f"--summaries-from lacks {len(missing)} context(s) for prompt {prompt_name!r}: "
                    f"{missing[:5]}"
                )

        if completed_rows:
            print(
                f"resume: continuing summarization prompt {prompt_name!r} from "
                f"row {completed_rows}/{total_per_prompt}; reusable saved summaries="
                f"{len(context_cache)}/{len(saved_context_cache)}",
                flush=True,
            )
        def record_answer(position, task_key, index, sample, context_id, summary_record, context_cache_hit,
                          output, stop_reason, generated_tokens, timings):
            timings.update({
                "sender_ms": 0.0 if context_cache_hit else summary_record["preparation_ms"],
                "context_cache_hit": context_cache_hit,
            })
            for field in ("sender_prefill_ms", "sender_am_ms", "sender_decode_ms"):
                if context_cache_hit:
                    timings[field] = 0.0
                elif field in summary_record:
                    timings[field] = summary_record[field]
            journal.write("answer", prompt=prompt_name, task_key=task_key,
                          index=index, id=sample.id, context_id=context_id,
                          output=output, stop_reason=stop_reason,
                          generated_tokens=generated_tokens, timings=timings)
            grade_result = tasks[task_key].grade(output, sample)
            result, metrics, raw_grade = normalize_grade_result(grade_result)
            if args.print_every > 0 and position % args.print_every == 0:
                print_eval_sample(position, total_per_prompt, task_key, sample, output, result, metrics)
            stats[task_key]["seen"] += 1
            if result not in stats[task_key]:
                stats[task_key][result] = 0
            stats[task_key][result] += 1
            update_metric_stats(metric_stats, task_key, metrics)
            stop_stats[task_key][stop_reason] = stop_stats[task_key].get(stop_reason, 0) + 1
            row_summary = {key: value for key, value in summary_record.items() if key != "text"}
            row_summary["context_id"] = context_id
            rows.append({
                "task": sample.dataset,
                "task_key": task_key,
                "index": index,
                **compact_eval_sample(sample, json_compactor),
                "grade": result,
                "grade_result": raw_grade,
                "output": output,
                "stop_reason": stop_reason,
                "generated_tokens": generated_tokens,
                "summarization": row_summary,
                "timings": timings,
                "time": time.time(),
            })
            prompt_result["summary"] = summarize(stats, stop_stats, metric_stats)
            prompt_result["accuracy"] = prompt_result["summary"]["all"]["accuracy"]
            progress.set_postfix(score_postfix(stats, metric_stats))
            if args.output and args.save_every > 0 and len(rows) % args.save_every == 0:
                save_result(args.output, current_payload())

        pending_answers: list = []

        def answer_batch(items):
            messages_list = [
                messages_with_summarized_context(sample, summary_record["text"])
                for _, _, _, sample, _, summary_record, _ in items
            ]
            oom = False
            try:
                return generate_vanilla_batch(
                    model=model, tokenizer=tokenizer, messages_list=messages_list, device=args.device,
                    max_length=args.max_length, max_new_tokens=args.max_new_tokens,
                )
            except torch.cuda.OutOfMemoryError:
                if len(items) == 1:
                    raise
                oom = True
            if oom:
                gc.collect()
                torch.cuda.empty_cache()
                half = len(items) // 2
                print(f"receiver batch of {len(items)} hit CUDA OOM; retrying as {half} + {len(items) - half}",
                      flush=True)
                return answer_batch(items[:half]) + answer_batch(items[half:])

        def flush_answers():
            items = list(pending_answers)
            pending_answers.clear()
            started = synchronized_time(args.device)
            results = answer_batch(items)
            batch_ms = (synchronized_time(args.device) - started) * 1000
            for item, (output, stop_reason, generated_tokens) in zip(items, results):
                timings = {
                    "receiver_ms": batch_ms / len(items),
                    "receiver_batch_size": len(items),
                    "receiver_batch_ms": batch_ms,
                }
                record_answer(*item, output, stop_reason, generated_tokens, timings)

        progress = tqdm(
            remaining,
            total=total_per_prompt,
            initial=completed_rows,
            desc=f"summarize/eval prompt={prompt_name} ({prompt_index}/{len(prompt_names)})",
            bar_format=(
                "{desc}: {n_fmt}/{total_fmt}|{percentage:2.0f}%|{bar}| "
                "{elapsed}<{remaining}, {rate_fmt}{postfix}"
            ),
        )
        for position, (task_key, index, sample) in enumerate(
            progress, start=completed_rows + 1
        ):
            context_id = summarization_context_id(sample)
            summary_record = context_cache.get(context_id)
            context_cache_hit = summary_record is not None
            if summary_record is not None and "context_sha256" in summary_record:
                sample_sha = hashlib.sha256(
                    str(sample.metadata.get("context", "")).encode("utf-8")
                ).hexdigest()
                if sample_sha != summary_record["context_sha256"]:
                    raise ValueError(
                        f"precomputed summary for {context_id} was written for a different context text"
                    )
                if "original_tokens_receiver" not in summary_record:
                    original = len(tokenizer(
                        str(sample.metadata.get("context", "")), add_special_tokens=False
                    )["input_ids"])
                    summary_record["original_tokens_receiver"] = original
                    summary_record["retained_fraction_receiver"] = (
                        summary_record["summary_tokens_receiver"] / original if original else 0.0
                    )
            if summary_record is None and summarizer_model is None:
                raise RuntimeError(f"no summarizer loaded and no precomputed summary for {context_id}")
            if summary_record is None:
                context = str(sample.metadata.get("context", ""))
                print(
                    f"\nsummarizing context {context_id} with {summarizer_path} "
                    f"({prompt_name})",
                    flush=True,
                )
                summary_record = summarize_context(
                    model=summarizer_model,
                    tokenizer=summarizer_tokenizer,
                    receiver_tokenizer=tokenizer,
                    context=context,
                    prompt_name=prompt_name,
                    device=args.device,
                    max_length=args.summarizer_max_length,
                )
                save_context(context_id, summary_record)
                total_original = sum(
                    record["original_tokens_receiver"] for record in saved_context_cache.values()
                )
                total_summary = sum(
                    record["summary_tokens_receiver"] for record in saved_context_cache.values()
                )
                prompt_result["achieved_retained_fraction_weighted"] = (
                    total_summary / total_original if total_original else None
                )
                prompt_result["achieved_retained_fraction_mean"] = sum(
                    record["retained_fraction_receiver"]
                    for record in saved_context_cache.values()
                ) / len(saved_context_cache)
                print(
                    "summary size: "
                    f"{summary_record['summary_tokens_receiver']}/"
                    f"{summary_record['original_tokens_receiver']} receiver tokens "
                    f"({summary_record['retained_fraction_receiver']:.3%})",
                    flush=True,
                )

            if args.batch_size > 1:
                pending_answers.append((position, task_key, index, sample, context_id, summary_record, context_cache_hit))
                if len(pending_answers) >= args.batch_size:
                    flush_answers()
                continue
            receiver_started = synchronized_time(args.device)
            answer_messages = messages_with_summarized_context(sample, summary_record["text"])
            timings = {}
            output, stop_reason, generated_tokens = generate_answer(
                model=model,
                tokenizer=tokenizer,
                messages=answer_messages,
                device=args.device,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                timings=timings,
            )
            timings["receiver_ms"] = (synchronized_time(args.device) - receiver_started) * 1000
            record_answer(position, task_key, index, sample, context_id, summary_record, context_cache_hit,
                          output, stop_reason, generated_tokens, timings)
        if pending_answers:
            flush_answers()

    print(json.dumps({
        "results_by_prompt": {
            key: {
                "accuracy": value["accuracy"],
                "achieved_retained_fraction_weighted": value["achieved_retained_fraction_weighted"],
                "achieved_retained_fraction_mean": value["achieved_retained_fraction_mean"],
                "correct": value["summary"]["all"]["correct"],
                "seen": value["summary"]["all"]["seen"],
            }
            for key, value in results_by_prompt.items()
        }
    }, indent=2))
    if args.output:
        save_result(args.output, current_payload())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate text-summary (T2T) QA baselines")
    add_eval_args(parser)
    parser.add_argument("--model", required=True, help="Answering model")
    parser.add_argument("--adapter", help="PEFT/LoRA adapter directory merged into the --model weights at load time (vanilla evaluation of a fine-tuned model).")
    source = parser.add_mutually_exclusive_group(required=True)
    prompts = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--summarizer-model",
        help="Optional model that summarizes each shared context before --model answers its questions.",
    )
    prompts.add_argument(
        "--summary-prompt",
        choices=list(SUMMARY_PROMPTS),
        help="Evaluate one paper summarization prompt variant.",
    )
    prompts.add_argument(
        "--summary-prompts",
        choices=list(SUMMARY_PROMPTS),
        nargs="+",
        help="Evaluate several prompt variants in one run. Default: all five variants.",
    )
    parser.add_argument(
        "--summarizer-max-length",
        type=int,
        help=(
            "Maximum summarizer prompt plus output length. Default: the summarizer model's "
            "max_position_embeddings."
        ),
    )
    source.add_argument(
        "--summaries-from",
        help=(
            "JSON of precomputed summaries written by scripts/exp2_sweep/run.py summarize "
            "({'summarizer_model': path, 'prompts': {prompt: {context_id: record}}}). Runs the "
            "summarization evaluation without loading a summarizer: every scheduled context must "
            "be present for every requested prompt. Mutually exclusive with --summarizer-model."
        ),
    )
    args = parser.parse_args(argv)
    validate_eval_args(parser, args)
    if args.summarizer_max_length is not None and args.summarizer_max_length <= 1:
        parser.error("--summarizer-max-length must be greater than 1")
    if args.summary_prompts is not None and len(set(args.summary_prompts)) != len(args.summary_prompts):
        parser.error("--summary-prompts must not contain duplicates")
    return args


def main(argv=None):
    args = parse_args(argv)
    command_line = sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
    started_at = time.time()
    dtype = dtype_from_args(args)
    model_path = resolve_model_path(args.model)
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, dtype, args.device, adapter=args.adapter)
    summarizer_path = None
    summarizer_tokenizer = None
    summarizer_model = None
    if args.summarizer_model is not None:
        summarizer_path = resolve_model_path(args.summarizer_model)
        print(f"summarizer_model: {summarizer_path}")
        print(f"answering_model: {model_path}")
        summarizer_tokenizer = load_tokenizer(summarizer_path)
        summarizer_model = load_model(summarizer_path, dtype, args.device)
        if args.summarizer_max_length is None:
            args.summarizer_max_length = int(
                getattr(summarizer_model.config, "max_position_embeddings", 32768)
            )
        print(f"summarizer_max_length: {args.summarizer_max_length}")
        torch.manual_seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
    precomputed_summaries = None
    if args.summaries_from is not None:
        precomputed_summaries = load_precomputed_summaries(Path(args.summaries_from), tokenizer)
        summarizer_path = Path(precomputed_summaries["summarizer_model"])
        if args.summarizer_max_length is None:
            args.summarizer_max_length = precomputed_summaries.get("summarizer_max_length")
        print(f"summaries_from: {args.summaries_from} (summarizer {summarizer_path})")
        print(f"answering_model: {model_path}")

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
    selected_summary_prompts = args.summary_prompts or (
        [args.summary_prompt] if args.summary_prompt is not None else list(SUMMARY_PROMPTS)
    )
    summary_metadata = {
        "summarizer_model": str(summarizer_path),
        "summaries_from": args.summaries_from,
        "answering_model": str(model_path),
        "prompts": selected_summary_prompts,
        "summarizer_max_length": args.summarizer_max_length,
        "summary_generation_ceiling": "original context token count",
        "sampling": {
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "thinking": False,
        },
        "reuse_summary_by_context_id": True,
    }
    run_metadata = {
        "mode": "summarization", "receiver_model": str(model_path),
        "summarization": summary_metadata, "tasks": list(datasets),
        "evaluation_config": evaluation_config_metadata(args, dtype=dtype, command_line=command_line, script=__file__),
        "saved_json_format": eval_json_format_metadata(),
    }
    evaluate_summarization_prompts(
        args=args, model=model, tokenizer=tokenizer,
        summarizer_model=summarizer_model, summarizer_tokenizer=summarizer_tokenizer,
        summarizer_path=summarizer_path, tasks=tasks, datasets=datasets,
        run_metadata=run_metadata, json_compactor=EvalJsonCompactor(tokenizer),
        precomputed_summaries=precomputed_summaries, started_at=started_at,
        generate_answer=generate_vanilla, save_result=save_json,
    )


if __name__ == "__main__":
    main()
