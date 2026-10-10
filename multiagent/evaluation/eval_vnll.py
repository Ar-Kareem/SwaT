#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.dataset.c2c_cross_family import cross_causal_limits, tokenize_with_offsets, virtual_positions
from multiagent.model.unaware_comm import full_sender_cache
from multiagent.model.cache_renderer import flatten_sender_cache, load_renderer_checkpoint, pred_to_cache
from multiagent.model.unaware_comm import tuple_to_dynamic_cache
from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.train.train_utils import load_model, load_tokenizer, move_kv_cache


DEFAULT_DATASET = REPO_ROOT / "multiagent" / "data" / "continuation_nll" / "heldout_2k2k.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", required=True, type=Path, help="Renderer run, checkpoint directory, or renderer.pt")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/eval_vnll.json")
    parser.add_argument("--sender-device", default="cuda:0")
    parser.add_argument("--receiver-device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--corrupt-percent",
        type=float,
        default=0.0,
        help="Replace this percentage of rendered K/V scalar entries with values sampled from the same tensor.",
    )
    parser.add_argument("--corruption-seed", type=int, default=43)
    args = parser.parse_args()
    if not 0.0 <= args.corrupt_percent <= 100.0:
        parser.error("--corrupt-percent must be in [0, 100]")
    return args


def resolve_renderer_file(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.is_file():
        if path.name != "renderer.pt":
            raise ValueError(f"expected renderer.pt, got {path}")
        run_dir = path.parent.parent if path.parent.name.startswith(("checkpoint-", "exit-checkpoint-")) else path.parent
        return path, run_dir
    direct = path / "renderer.pt"
    if direct.exists():
        run_dir = path.parent if path.name.startswith(("checkpoint-", "exit-checkpoint-")) else path
        return direct, run_dir
    latest_path = path / "latest.json"
    if latest_path.exists():
        latest = json.loads(latest_path.read_text())
        candidate = path / str(latest["checkpoint"]) / "renderer.pt"
        if candidate.exists():
            return candidate, path
    checkpoints = sorted(path.glob("*/renderer.pt"), key=lambda item: item.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1], path
    raise FileNotFoundError(f"no renderer checkpoint found under {path}")


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"VNLL dataset is empty: {path}")
    for row in rows:
        if row.get("schema") != "multikv.continuation_nll.v1":
            raise ValueError(f"unsupported VNLL row schema in {path}: {row.get('schema')!r}")
    return rows


def aggregate(stats: dict) -> dict:
    token_count = int(stats["token_count"])
    nll = float(stats["nll_sum"]) / token_count
    return {
        "nll": nll,
        "perplexity": math.exp(nll),
        "token_accuracy": int(stats["token_correct"]) / token_count,
        "scored_tokens": token_count,
        "samples": int(stats["samples"]),
    }


def corrupt_cache_entries(cache, percent: float, seed: int):
    if percent == 0.0:
        return cache
    probability = percent / 100.0
    corrupted_layers = []
    for layer_index, (key, value) in enumerate(cache):
        tensors = []
        for kind_index, tensor in enumerate((key, value)):
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(seed + 2 * layer_index + kind_index)
            original = tensor.flatten()
            corrupted = original.clone()
            chunk_size = 1_048_576
            for start in range(0, original.numel(), chunk_size):
                end = min(start + chunk_size, original.numel())
                length = end - start
                replacement_indices = torch.randint(
                    original.numel(),
                    (length,),
                    generator=generator,
                    device=tensor.device,
                )
                if percent == 100.0:
                    corrupted[start:end] = original[replacement_indices]
                else:
                    mask = torch.rand(length, generator=generator, device=tensor.device) < probability
                    corrupted[start:end] = torch.where(
                        mask,
                        original[replacement_indices],
                        corrupted[start:end],
                    )
            tensors.append(corrupted.view_as(tensor))
        corrupted_layers.append(tuple(tensors))
    return tuple(corrupted_layers)


@torch.no_grad()
def evaluate_row(
    row: dict,
    renderer,
    sender_model,
    receiver_model,
    sender_tokenizer,
    receiver_tokenizer,
    sender_device: str,
    receiver_device: str,
    corrupt_percent: float = 0.0,
    corruption_seed: int = 43,
    causal_cross_attention: bool = False,
    position_mode: str = "interp",
) -> dict:
    prefix_text = str(row["prefix_text"])
    receiver_ids_list, receiver_offsets = tokenize_with_offsets(receiver_tokenizer, prefix_text)
    stored_receiver_ids = [int(value) for value in row["receiver_prefix_token_ids"]]
    if receiver_ids_list != stored_receiver_ids:
        raise ValueError(f"receiver prefix retokenization changed for {row['id']}")
    token_alignment = getattr(renderer, "token_alignment", "native")
    if token_alignment == "c2c-longest":
        from multiagent.dataset.c2c_token_alignment import get_aligner
        sender_ids_list = get_aligner(sender_tokenizer, receiver_tokenizer).convert(stored_receiver_ids)
    else:
        sender_ids_list, sender_offsets = tokenize_with_offsets(sender_tokenizer, prefix_text)
    continuation = [int(value) for value in row["receiver_continuation_token_ids"]]
    if len(continuation) < 2:
        raise ValueError(f"continuation is too short for shifted NLL: {row['id']}")

    sender_ids = torch.tensor([sender_ids_list], dtype=torch.long, device=sender_device)
    sender_positions = torch.arange(sender_ids.shape[1], device=sender_device).unsqueeze(0)
    sender_cache = sender_model(
        input_ids=sender_ids,
        attention_mask=torch.ones_like(sender_ids),
        position_ids=sender_positions,
        use_cache=True, past_key_values=full_sender_cache()
    ).past_key_values
    sender_cache = move_kv_cache(sender_cache, sender_device)
    memory = flatten_sender_cache(sender_model, sender_cache, sender_positions).to(receiver_device)
    del sender_cache

    receiver_ids = torch.tensor([receiver_ids_list], dtype=torch.long, device=receiver_device)
    receiver_positions = torch.arange(receiver_ids.shape[1], device=receiver_device).unsqueeze(0)
    if token_alignment == "c2c-longest":
        memory_positions = receiver_positions.float()
        limits = receiver_positions + 1
    else:
        anchors = np.asarray([0.0, float(len(prefix_text))])
        mapped_positions = virtual_positions(sender_offsets, receiver_offsets, anchors, anchors, position_mode)
        memory_positions = torch.from_numpy(mapped_positions[None].astype("float32")).to(receiver_device)
        limits = None
        if causal_cross_attention:
            limits = torch.from_numpy(
                cross_causal_limits(sender_offsets, receiver_offsets, anchors, anchors)[None]
            ).to(receiver_device)
    embeds = receiver_model.get_input_embeddings()(receiver_ids)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=receiver_device.startswith("cuda"),
    ):
        pred = renderer(embeds, receiver_positions, memory, memory_positions, None, limits)
    pred_cache = pred_to_cache(
        receiver_model,
        pred.float(),
        receiver_positions,
        next(receiver_model.parameters()).dtype,
    )
    pred_cache = corrupt_cache_entries(pred_cache, corrupt_percent, corruption_seed)
    del pred, embeds, memory

    decode_ids = torch.tensor([continuation[:-1]], dtype=torch.long, device=receiver_device)
    labels = torch.tensor([continuation[1:]], dtype=torch.long, device=receiver_device)
    prefix_len = receiver_ids.shape[1]
    output = receiver_model(
        input_ids=decode_ids,
        attention_mask=torch.ones((1, prefix_len + decode_ids.shape[1]), dtype=torch.long, device=receiver_device),
        position_ids=torch.arange(prefix_len, prefix_len + decode_ids.shape[1], device=receiver_device).unsqueeze(0),
        past_key_values=tuple_to_dynamic_cache(pred_cache, receiver_model.config),
        use_cache=False,
    )
    logits = output.logits.float()
    nll_sum = float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum"))
    token_correct = int(logits.argmax(dim=-1).eq(labels).sum())
    token_count = labels.numel()
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "nll": nll_sum / token_count,
        "perplexity": math.exp(nll_sum / token_count),
        "token_accuracy": token_correct / token_count,
        "scored_tokens": token_count,
        "nll_sum": nll_sum,
        "token_correct": token_correct,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    renderer_file, run_dir = resolve_renderer_file(args.renderer)
    percent_slug = format(args.corrupt_percent, "g").replace(".", "p")
    output_path = args.output or (
        run_dir / ("eval_vnll.json" if args.corrupt_percent == 0 else f"eval_vnll_corrupt_p{percent_slug}.json")
    )
    rows = load_rows(args.dataset)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    renderer, payload = load_renderer_checkpoint(renderer_file, args.receiver_device)
    sender_path = resolve_model_path(str(payload["sender_model"]))
    receiver_path = resolve_model_path(str(payload["receiver_model"]))
    sender_tokenizer = load_tokenizer(sender_path)
    receiver_tokenizer = load_tokenizer(receiver_path)
    sender_model = load_model(sender_path, dtype, args.sender_device)
    receiver_model = load_model(receiver_path, dtype, args.receiver_device)

    per_sample = []
    by_dataset = defaultdict(lambda: {"nll_sum": 0.0, "token_correct": 0, "token_count": 0, "samples": 0})
    overall = {"nll_sum": 0.0, "token_correct": 0, "token_count": 0, "samples": 0}
    for sample_index, row in enumerate(tqdm(rows, desc="vnll")):
        result = evaluate_row(
            row,
            renderer,
            sender_model,
            receiver_model,
            sender_tokenizer,
            receiver_tokenizer,
            args.sender_device,
            args.receiver_device,
            args.corrupt_percent,
            args.corruption_seed + sample_index * 100_003,
            causal_cross_attention=bool(payload.get("args", {}).get("causal_cross_attention")),
            position_mode=str(payload.get("position_mode") or "interp"),
        )
        per_sample.append({key: value for key, value in result.items() if key not in ("nll_sum", "token_correct")})
        for bucket in (by_dataset[result["dataset"]], overall):
            bucket["nll_sum"] += result["nll_sum"]
            bucket["token_correct"] += result["token_correct"]
            bucket["token_count"] += result["scored_tokens"]
            bucket["samples"] += 1

    report = {
        "dataset": "vnll",
        "renderer_checkpoint": str(renderer_file),
        "renderer_step": payload.get("step"),
        "token_alignment": payload.get("token_alignment", {"mode": "native", "version": 1}),
        "sender_model": str(payload["sender_model"]),
        "receiver_model": str(payload["receiver_model"]),
        "source_file": str(args.dataset.resolve()),
        "elapsed_seconds": time.time() - started,
        "random_corruption": {
            "percent": args.corrupt_percent,
            "seed": args.corruption_seed,
            "unit": "scalar K/V entries",
            "replacement": "uniform sample from the same original tensor",
            "applied_to": "rendered keys and values before receiver decoding",
        },
        "metrics": {
            "overall": aggregate(overall),
            "by_dataset": {name: aggregate(stats) for name, stats in sorted(by_dataset.items())},
        },
        "samples": per_sample,
        "note": "The first continuation token seeds decoding and is not scored; all 2048 prefix KV entries are rendered.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
