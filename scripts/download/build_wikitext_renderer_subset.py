#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.paths import portable


PARQUET_URL_TEMPLATE = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    "refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/train/{shard:04d}.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multiagent/data/renderer_text/wikitext103_train_4k4k_10m.jsonl"),
    )
    parser.add_argument("--sender-model", default="Qwen3-4B")
    parser.add_argument("--receiver-model", default="llama3-8b-instruct")
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--continuation-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--shard", type=int, default=0, help="wikitext-103-raw-v1 train parquet shard (0 or 1); the 10m slice used shard 0")
    args = parser.parse_args()
    if min(args.target_tokens, args.prefix_tokens, args.continuation_tokens) <= 0:
        parser.error("token counts must be positive")
    return args


def download_lines(timeout: float, shard: int = 0) -> list[str]:
    with NamedTemporaryFile(suffix=".parquet") as handle:
        request = urllib.request.Request(PARQUET_URL_TEMPLATE.format(shard=shard), headers={"User-Agent": "multikv-wikitext-renderer/1.0"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                break
            except Exception:
                if attempt == 4:
                    raise
                handle.seek(0)
                handle.truncate()
                time.sleep(0.5 * (2**attempt))
        handle.flush()
        return [str(value or "") for value in pq.read_table(handle.name, columns=["text"])["text"].to_pylist()]


def articles(lines: list[str]) -> list[tuple[int, int, str]]:
    output: list[tuple[int, int, str]] = []
    current: list[str] = []
    start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        title = stripped.startswith("= ") and not stripped.startswith("= =") and stripped.endswith(" =")
        if title and current:
            output.append((start, index, "".join(current)))
            current = []
            start = index
        elif not current:
            start = index
        current.append(line)
    if current:
        output.append((start, len(lines), "".join(current)))
    return output


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
    lines = download_lines(args.timeout, args.shard)
    source_articles = articles(lines)
    random.Random(args.seed).shuffle(source_articles)

    window_tokens = args.prefix_tokens + args.continuation_tokens
    target_samples = (args.target_tokens + window_tokens - 1) // window_tokens
    records: list[dict] = []
    articles_used = 0
    skipped_boundaries = 0
    for row_start, row_end, text in source_articles:
        ids = token_ids(receiver_tokenizer, text)
        added = False
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
                    "id": f"wikitext103:train:{row_start}-{row_end}:{window_start}",
                    "dataset": "wikitext103_renderer",
                    "split": "train",
                    "source": {
                        "repo": "Salesforce/wikitext",
                        "config": "wikitext-103-raw-v1",
                        "split": "train",
                        "parquet_shard": args.shard,
                        "rows": [row_start, row_end],
                        "receiver_token_window_start": window_start,
                    },
                    "prefix_text": prefix_text,
                    "continuation_text": continuation_text,
                    "receiver_prefix_tokens": len(prefix_ids),
                    "receiver_continuation_tokens": len(continuation_ids),
                    "sender_prefix_tokens": len(token_ids(sender_tokenizer, prefix_text)),
                }
            )
            added = True
            if len(records) == target_samples:
                break
        articles_used += int(added)
        if len(records) == target_samples:
            break
        if articles_used and articles_used % 100 == 0:
            print(
                f"WikiText-103 train: {len(records):,}/{target_samples:,} windows "
                f"({len(records) * window_tokens:,} receiver tokens; {articles_used} articles)",
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
        "notes": {"heldout_note": "WikiText test/validation are excluded; VNLL uses held-out splits."},
        "schema": "multikv.renderer_continuation_manifest.v1",
        "output": str(args.output),
        "source_repo": "Salesforce/wikitext",
        "source_config": "wikitext-103-raw-v1",
        "source_split": "train",
        "source_parquet_shards": [args.shard],
        "source_articles_used": articles_used,
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
