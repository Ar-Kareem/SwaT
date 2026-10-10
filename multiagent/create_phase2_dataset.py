from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_module
import re
import sys
import traceback
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c import turn_end_ids
from multiagent.dataset.c2c_registry import MATH_TASKS, MCQ_TASKS, load_c2c_dataset_spec
from multiagent.dataset.c2c_types import C2CExample, ChatMessage
from multiagent.dataset.math_dataset import grade_output as grade_math_output
from multiagent.dataset.mcq_dataset import parse_choice
from multiagent.model.vanilla_loader import load_causal_model, resolve_model_path
from multiagent.paths import DATASETS_ROOT


DEFAULT_TASKS = ("gsm8k:train", "math-algebra:train", "arc-c:train")


def parse_args():
    parser = argparse.ArgumentParser(description="Create receiver-self guided Phase-II trace JSONL")
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", default=str(DATASETS_ROOT))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), help="Tasks to use in task:split form.")
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dtype_from_args(args):
    return dtype_from_name(args.dtype, args.device)


def dtype_from_name(dtype_name: str, device: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def load_model(path: Path, dtype: torch.dtype, device: str):
    model = load_causal_model(path, dtype, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def model_input_device(model) -> torch.device:
    return next(model.parameters()).device


def iter_limited(dataset, start: int, limit_per_dataset: int | None):
    if start < 0:
        raise ValueError("start must be non-negative")
    if start >= len(dataset):
        raise ValueError(f"start={start} is outside dataset with {len(dataset)} rows")
    end = len(dataset) if limit_per_dataset is None else min(len(dataset), start + limit_per_dataset)
    for index in range(start, end):
        yield index, dataset[index]


def phase2_source_sample(example: C2CExample) -> dict:
    sample = dict(example.metadata)
    sample.setdefault("id", example.id)
    sample.setdefault("dataset", example.dataset)
    sample.setdefault("split", example.split)
    if "answer" not in sample and example.target_text is not None:
        sample["answer"] = example.target_text
    return sample


def arc_choices_text(sample: dict) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip(sample["labels"], sample["choices"]))


def trace_prompt_messages(task: str, sample: dict) -> list[ChatMessage]:
    if task == "gsm8k":
        content = (
            "You are given a problem and its ground-truth answer.\n\n"
            f"Problem:\n{sample['question']}\n\n"
            f"Ground-truth answer:\n{sample['answer']}\n\n"
            "Write a step-by-step solution that arrives at the ground-truth answer. "
            f"End with:\nAnswer: {sample['answer']}"
        )
    elif task in MATH_TASKS:
        content = (
            "You are given a math problem and its ground-truth answer.\n\n"
            f"Problem:\n{sample['question']}\n\n"
            f"Ground-truth answer:\n{sample['answer']}\n\n"
            "Write a step-by-step solution that arrives at the ground-truth answer. "
            "End with the final answer in \\boxed{}."
        )
    elif task in MCQ_TASKS:
        content = (
            "You are given a multiple-choice question and its ground-truth answer.\n\n"
            f"Question:\n{sample['question']}\n\n"
            f"Choices:\n{arc_choices_text(sample)}\n\n"
            f"Ground-truth answer:\n{sample['answer']}\n\n"
            "Write a step-by-step explanation that arrives at the ground-truth answer. "
            f"End exactly with:\nAnswer: {sample['answer']}"
        )
    else:
        raise KeyError(task)
    return [{"role": "user", "content": content}]


def apply_chat_template(tokenizer, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def generate_traces(
    model,
    tokenizer,
    messages_batch: list[list[ChatMessage]],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    texts = [apply_chat_template(tokenizer, messages) for messages in messages_batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                       add_special_tokens=False).to(model_input_device(model))
    do_sample = temperature > 0.0
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": sorted(turn_end_ids(tokenizer, getattr(model, "generation_config", None))),
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    output = model.generate(**generate_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    eos_ids = turn_end_ids(tokenizer, getattr(model, "generation_config", None))
    traces = []
    for row in output:
        generated_ids = row[prompt_width:]
        traces.append(
            {
                "text": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
                "stop_reason": "eos" if eos_ids & set(generated_ids.tolist()) else "cutoff",
                "generated_tokens": int(generated_ids.numel()),
            }
        )
    return traces


def ensure_arc_final_answer(trace: str, answer: str, labels: list[str]) -> str:
    trace = trace.rstrip()
    if parse_choice(trace, labels) == answer:
        return trace
    final = f"Answer: {answer}"
    trace = re.sub(r"(?is)\s*(?:final\s+)?answer\s*[:=]\s*[A-Z]\s*\.?\s*$", "", trace).rstrip()
    return f"{trace}\n{final}" if trace else final


def ensure_math_final_answer(task: str, trace: str, answer: str) -> str:
    trace = trace.rstrip()
    if task == "gsm8k" and grade_math_output(trace, {"answer_type": "numeric", "answer": answer}) == "correct":
        return trace
    if task == "math-algebra" and grade_math_output(trace, {"answer_type": "math", "answer": answer}) == "correct":
        return trace
    if task == "gsm8k":
        return f"{trace}\nAnswer: {answer}" if trace else f"Answer: {answer}"
    return f"{trace}\n\\boxed{{{answer}}}" if trace else f"\\boxed{{{answer}}}"


def canonicalize_trace(task: str, sample: dict, trace: str) -> str:
    if task in MCQ_TASKS:
        return ensure_arc_final_answer(trace, sample["answer"], sample["labels"])
    if task in MATH_TASKS:
        return ensure_math_final_answer(task, trace, sample["answer"])
    raise KeyError(task)


def make_record(task: str, sample: dict, receiver_model: str, trace: str, raw_trace: str, generation: dict) -> dict:
    record = {
        "id": sample["id"],
        "dataset": task,
        "split": sample["split"],
        "receiver_model": receiver_model,
        "question": sample["question"],
        "gold_answer": sample["answer"],
        "trace": trace,
        "trace_mode": "receiver_self_guided",
        "stop_reason": generation["stop_reason"],
        "generated_tokens": generation["generated_tokens"],
    }
    if task in MCQ_TASKS:
        record["labels"] = sample["labels"]
        record["choices"] = sample["choices"]
    if trace != raw_trace:
        record["raw_trace"] = raw_trace
    return record


def append_jsonl(path: str | Path, rows: list[dict]):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def existing_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"could not parse existing JSONL {path}:{line_number}") from error
            row_id = row.get("id")
            if row_id is not None:
                ids.add(str(row_id))
    return ids


def batched(items: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_records_for_batch(model, tokenizer, receiver_model_name: str, batch: list[tuple[str, dict]], config: dict):
    prompt_batch = [trace_prompt_messages(task, sample) for task, sample in batch]
    generations = generate_traces(
        model,
        tokenizer,
        prompt_batch,
        config["max_length"],
        config["max_new_tokens"],
        config["temperature"],
        config["top_p"],
    )
    records = []
    for (task, sample), generation in zip(batch, generations):
        raw_trace = generation["text"]
        trace = canonicalize_trace(task, sample, raw_trace)
        records.append(make_record(task, sample, receiver_model_name, trace, raw_trace, generation))
    return records


def worker_main(worker_id: int, device: str, receiver_path: str, receiver_model_name: str, dtype_name: str, items: list, config: dict, queue):
    try:
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        torch.manual_seed(config["seed"] + worker_id)
        dtype = dtype_from_name(dtype_name, device)
        tokenizer = load_tokenizer(Path(receiver_path))
        model = load_model(Path(receiver_path), dtype, device)
        queue.put(("ready", worker_id, device))

        for batch in batched(items, config["batch_size"]):
            records = generate_records_for_batch(model, tokenizer, receiver_model_name, batch, config)
            queue.put(("rows", worker_id, records))
        queue.put(("done", worker_id, None))
    except Exception:
        queue.put(("error", worker_id, traceback.format_exc()))


def run_single_process(args, receiver_path: Path, dtype: torch.dtype, pending: list[tuple[str, dict]], done_ids: set[str]):
    tokenizer = load_tokenizer(receiver_path)
    model = load_model(receiver_path, dtype, args.device)
    print(f"receiver_model: {receiver_path} device={model_input_device(model)}")

    buffer = []
    progress = tqdm(total=len(pending), desc="phase2-traces")
    generated = 0
    config = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    for batch in batched(pending, args.batch_size):
        records = generate_records_for_batch(model, tokenizer, args.receiver_model, batch, config)
        buffer.extend(records)
        for record in records:
            done_ids.add(record["id"])
        generated += len(records)
        if len(buffer) >= args.save_every:
            append_jsonl(args.output, buffer)
            buffer.clear()
        progress.update(len(records))
        progress.set_postfix({"task": batch[-1][0], "new": generated, "total": len(done_ids)})
    progress.close()
    append_jsonl(args.output, buffer)
    print(f"saved {generated} new traces to {args.output} ({len(done_ids)} total)")


def run_multi_gpu(args, receiver_path: Path, pending: list[tuple[str, dict]], done_ids: set[str], num_gpus: int):
    config = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    shards = [pending[index::num_gpus] for index in range(num_gpus)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for worker_id, shard in enumerate(shards):
        device = f"cuda:{worker_id}"
        process = ctx.Process(
            target=worker_main,
            args=(worker_id, device, str(receiver_path), args.receiver_model, args.dtype, shard, config, queue),
        )
        process.start()
        processes.append(process)

    print(f"receiver_model: {receiver_path} replicated on {num_gpus} visible CUDA devices")
    buffer = []
    generated = 0
    finished = 0
    progress = tqdm(total=len(pending), desc="phase2-traces")
    try:
        while finished < len(processes):
            try:
                kind, worker_id, payload = queue.get(timeout=30)
            except queue_module.Empty:
                dead = [index for index, process in enumerate(processes) if process.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"worker process exited unexpectedly: {dead}")
                continue
            if kind == "ready":
                print(f"worker {worker_id} ready on {payload}", flush=True)
            elif kind == "rows":
                records = payload
                buffer.extend(records)
                for record in records:
                    done_ids.add(record["id"])
                generated += len(records)
                if len(buffer) >= args.save_every:
                    append_jsonl(args.output, buffer)
                    buffer.clear()
                progress.update(len(records))
                progress.set_postfix({"worker": worker_id, "new": generated, "total": len(done_ids)})
            elif kind == "done":
                finished += 1
            elif kind == "error":
                raise RuntimeError(f"worker {worker_id} failed:\n{payload}")
    finally:
        progress.close()
        append_jsonl(args.output, buffer)
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
    print(f"saved {generated} new traces to {args.output} ({len(done_ids)} total)")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = dtype_from_args(args)
    receiver_path = resolve_model_path(args.receiver_model)

    datasets = {}
    for spec in args.tasks:
        datasets[spec] = load_c2c_dataset_spec(
            spec,
            root=args.dataset_root,
        )
    done_ids = existing_ids(args.output)
    pending = []
    for spec, dataset in datasets.items():
        for _, example in iter_limited(dataset, args.start, args.limit_per_dataset):
            sample = phase2_source_sample(example)
            if sample["id"] not in done_ids:
                pending.append((sample["dataset"], sample))

    if done_ids:
        print(f"resuming from {args.output}: found {len(done_ids)} existing traces")

    if args.device.startswith("cuda") and torch.cuda.device_count() > 1:
        run_multi_gpu(args, receiver_path, pending, done_ids, torch.cuda.device_count())
    else:
        run_single_process(args, receiver_path, dtype, pending, done_ids)


if __name__ == "__main__":
    main()
