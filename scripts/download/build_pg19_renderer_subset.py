#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.paths import portable


REPO = "emozilla/pg19"
CONFIG = "default"
SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multiagent/data/renderer_text/pg19_train_4k4k_20m.jsonl"),
    )
    parser.add_argument("--sender-model", default="Qwen3-4B")
    parser.add_argument("--receiver-model", default="llama3-8b-instruct")
    parser.add_argument("--target-tokens", type=int, default=20_000_000)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--continuation-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=None,
        help="Explicit PG19 train row to start the cyclic run of books at (default: derived from --seed). "
        "Use this to build a slice disjoint from an earlier one (the 20m slice used rows 631-877).",
    )
    parser.add_argument("--rows-per-request", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if min(args.target_tokens, args.prefix_tokens, args.continuation_tokens) <= 0:
        parser.error("token counts must be positive")
    if args.rows_per_request <= 0 or args.rows_per_request > 100:
        parser.error("--rows-per-request must be in [1, 100]")
    return args


def fetch_rows(offset: int, length: int, timeout: float) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": REPO, "config": CONFIG, "split": SPLIT, "offset": offset, "length": length}
    )
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={"User-Agent": "multikv-pg19-renderer-subset/1.0"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


def token_ids(tokenizer, text: str) -> list[int]:
    return [int(value) for value in tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]]


def stable_decode(tokenizer, ids: list[int]) -> str | None:
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text if token_ids(tokenizer, text) == ids else None


def main() -> None:
    args = parse_args()
    sender_path = resolve_model_path(args.sender_model)
    receiver_path = resolve_model_path(args.receiver_model)
    sender_tokenizer = AutoTokenizer.from_pretrained(sender_path, use_fast=True)
    receiver_tokenizer = AutoTokenizer.from_pretrained(receiver_path, use_fast=True)

    first = fetch_rows(0, 1, args.timeout)
    total_source_rows = int(first["num_rows_total"])
    rng = random.Random(args.seed)
    source_offset = rng.randrange(total_source_rows) if args.start_offset is None else args.start_offset % total_source_rows
    window_tokens = args.prefix_tokens + args.continuation_tokens
    target_samples = (args.target_tokens + window_tokens - 1) // window_tokens
    records: list[dict] = []
    source_rows_seen = 0
    skipped_boundaries = 0

    while len(records) < target_samples and source_rows_seen < total_source_rows:
        request_offset = (source_offset + source_rows_seen) % total_source_rows
        request_length = min(args.rows_per_request, total_source_rows - request_offset)
        payload = fetch_rows(request_offset, request_length, args.timeout)
        rows = payload.get("rows", [])
        if not rows:
            raise RuntimeError(f"PG19 returned no rows at train offset {request_offset}")
        for wrapped in rows:
            row_index = int(wrapped["row_idx"])
            text = str(wrapped["row"].get("text", ""))
            ids = token_ids(receiver_tokenizer, text)
            for window_start in range(0, len(ids) - window_tokens + 1, window_tokens):
                window = ids[window_start : window_start + window_tokens]
                prefix_ids = window[: args.prefix_tokens]
                continuation_ids = window[args.prefix_tokens :]
                prefix_text = stable_decode(receiver_tokenizer, prefix_ids)
                continuation_text = stable_decode(receiver_tokenizer, continuation_ids)
                if prefix_text is None or continuation_text is None:
                    skipped_boundaries += 1
                    continue
                records.append(
                    {
                        "schema": "multikv.renderer_continuation.v1",
                        "id": f"pg19:train:{row_index}:{window_start}",
                        "dataset": "pg19_renderer",
                        "split": "train",
                        "source": {
                            "repo": REPO,
                            "config": CONFIG,
                            "split": SPLIT,
                            "row": row_index,
                            "receiver_token_window_start": window_start,
                        },
                        "prefix_text": prefix_text,
                        "continuation_text": continuation_text,
                        "receiver_prefix_tokens": len(prefix_ids),
                        "receiver_continuation_tokens": len(continuation_ids),
                        "sender_prefix_tokens": len(token_ids(sender_tokenizer, prefix_text)),
                    }
                )
                if len(records) == target_samples:
                    break
            source_rows_seen += 1
            if len(records) == target_samples:
                break
        print(
            f"PG19 train: {len(records):,}/{target_samples:,} windows "
            f"({len(records) * window_tokens:,} receiver tokens; {source_rows_seen} books)",
            flush=True,
        )

    if len(records) < target_samples:
        raise RuntimeError(f"found only {len(records):,} of {target_samples:,} requested windows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_receiver_tokens = len(records) * window_tokens
    manifest = {
        "notes": {"heldout_note": "PG19 test is reserved for multiagent/data/continuation_nll/heldout_2k2k.jsonl"},
        "schema": "multikv.renderer_continuation_manifest.v1",
        "output": str(args.output),
        "source_repo": REPO,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_start_offset": source_offset,
        "source_rows_used": source_rows_seen,
        "sender_model": args.sender_model,
        "sender_tokenizer": portable(sender_path),
        "receiver_model": args.receiver_model,
        "receiver_tokenizer": portable(receiver_path),
        "seed": args.seed,
        "samples": len(records),
        "prefix_tokens_per_sample": args.prefix_tokens,
        "continuation_tokens_per_sample": args.continuation_tokens,
        "total_receiver_tokens": total_receiver_tokens,
        "target_receiver_tokens": args.target_tokens,
        "skipped_unstable_token_boundaries": skipped_boundaries,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records):,} windows / {total_receiver_tokens:,} receiver tokens to {args.output}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
