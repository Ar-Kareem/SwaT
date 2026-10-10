#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from datasets import load_from_disk
from transformers import AutoTokenizer

from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.paths import DATASETS_ROOT, portable


DEFAULT_SOURCE = DATASETS_ROOT / "snapshots/hotpotqa__hotpot_qa/distractor"
DEFAULT_OUTPUT = Path("multiagent/data/extend_context/hotpotqa_interleaved_28k_validation.jsonl")
PLACEMENTS = ((3000, 14000), (5500, 16000), (2500, 18000))


def normalize(text: Any) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).casefold().replace("_", " ")).split())


def contains_normalized(haystack: Any, needle: Any) -> bool:
    normalized_haystack = normalize(haystack)
    normalized_needle = normalize(needle)
    return (
        bool(normalized_needle)
        and f" {normalized_needle} " in f" {normalized_haystack} "
    )


def unique(values: list[Any]) -> list[str]:
    output = []
    for value in values:
        value = str(value)
        if value not in output:
            output.append(value)
    return output


def paragraph_text(sentences: list[Any]) -> str:
    return " ".join(str(sentence) for sentence in sentences).strip()


def document_text(title: str, text: str) -> str:
    return f"<document>\nTitle: {title}\n{text}\n</document>\n\n"


@dataclass(slots=True)
class Candidate:
    source_id: str
    question: str
    answer: str
    level: str
    hop1_title: str
    hop1_text: str
    hop2_title: str
    hop2_text: str
    supporting_facts: dict[str, list[Any]]


@dataclass(slots=True)
class PoolDocument:
    source_id: str
    title: str
    text: str
    normalized: str
    token_ids: array


def supporting_sentences(row: dict[str, Any], title: str) -> str:
    contexts = {
        str(context_title): list(sentences)
        for context_title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
    }
    selected = []
    for fact_title, sentence_id in zip(
        row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
    ):
        if str(fact_title) != title or title not in contexts:
            continue
        sentence_id = int(sentence_id)
        if 0 <= sentence_id < len(contexts[title]):
            selected.append(contexts[title][sentence_id])
    return paragraph_text(selected)


def strict_candidates(dataset) -> list[Candidate]:
    candidates = []
    for raw in dataset:
        row = dict(raw)
        answer = str(row["answer"])
        if row.get("type") != "bridge" or normalize(answer) in {"yes", "no"}:
            continue
        support_titles = unique(list(row["supporting_facts"]["title"]))
        if len(support_titles) != 2:
            continue
        by_title = {
            str(title): paragraph_text(sentences)
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
        }
        if any(title not in by_title for title in support_titles):
            continue
        first, second = support_titles
        first_support = supporting_sentences(row, first)
        second_support = supporting_sentences(row, second)
        direction = None
        if (
            contains_normalized(by_title[first], second)
            and not contains_normalized(by_title[second], first)
            and contains_normalized(second_support, answer)
            and not contains_normalized(by_title[first], answer)
        ):
            direction = (first, second)
        elif (
            contains_normalized(by_title[second], first)
            and not contains_normalized(by_title[first], second)
            and contains_normalized(first_support, answer)
            and not contains_normalized(by_title[second], answer)
        ):
            direction = (second, first)
        if direction is None:
            continue
        hop1_title, hop2_title = direction
        question = str(row["question"])
        if contains_normalized(question, hop2_title) or contains_normalized(question, answer):
            continue
        candidates.append(Candidate(
            source_id=str(row["id"]),
            question=question,
            answer=answer,
            level=str(row.get("level", "hard")),
            hop1_title=hop1_title,
            hop1_text=by_title[hop1_title],
            hop2_title=hop2_title,
            hop2_text=by_title[hop2_title],
            supporting_facts={
                "title": [str(value) for value in row["supporting_facts"]["title"]],
                "sent_id": [int(value) for value in row["supporting_facts"]["sent_id"]],
            },
        ))
    return candidates


def build_pool(dataset, tokenizer) -> list[PoolDocument]:
    unique_documents: dict[tuple[str, str], tuple[str, str, str]] = {}
    for row in dataset:
        source_id = str(row["id"])
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            title = str(title)
            text = paragraph_text(sentences)
            unique_documents.setdefault((title, text), (source_id, title, text))
    records = list(unique_documents.values())
    blocks = [document_text(title, text) for _, title, text in records]
    tokenized = tokenizer(blocks, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    return [
        PoolDocument(
            source_id=source_id,
            title=title,
            text=text,
            normalized=normalize(f"{title}\n{text}"),
            token_ids=array("I", token_ids),
        )
        for (source_id, title, text), token_ids in zip(records, tokenized)
    ]


def traversal(pool_size: int, key: str) -> Iterator[int]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "big") % pool_size
    step = 7919 + 2 * (int.from_bytes(digest[8:12], "big") % 4096)
    while math.gcd(step, pool_size) != 1:
        step += 2
    for offset in range(pool_size):
        yield (start + offset * step) % pool_size


def build_row(
    candidate: Candidate,
    variant: int,
    pool: list[PoolDocument],
    tokenizer,
    target_tokens: int,
    tail_tokens: int,
) -> dict[str, Any]:
    hop1_target, hop2_target = PLACEMENTS[variant % len(PLACEMENTS)]
    if hop2_target >= target_tokens - tail_tokens - 512:
        raise ValueError("hop 2 placement does not leave a clean tail margin")
    hop1_ids = tokenizer(
        document_text(candidate.hop1_title, candidate.hop1_text), add_special_tokens=False
    )["input_ids"]
    hop2_ids = tokenizer(
        document_text(candidate.hop2_title, candidate.hop2_text), add_special_tokens=False
    )["input_ids"]
    forbidden = {
        normalize(candidate.answer),
        normalize(candidate.hop1_title),
        normalize(candidate.hop2_title),
    }
    chosen: set[int] = set()
    chosen_sources = []
    ids: list[int] = []
    document_count = 0
    iterator = traversal(len(pool), f"{candidate.source_id}/v{variant}")

    def allowed(document: PoolDocument, pool_index: int) -> bool:
        if pool_index in chosen or document.source_id == candidate.source_id:
            return False
        padded_document = f" {document.normalized} "
        return not any(value and f" {value} " in padded_document for value in forbidden)

    def fill_to(minimum: int) -> None:
        nonlocal document_count
        while len(ids) < minimum:
            for pool_index in iterator:
                document = pool[pool_index]
                if not allowed(document, pool_index):
                    continue
                chosen.add(pool_index)
                chosen_sources.append(document.source_id)
                ids.extend(document.token_ids)
                document_count += 1
                break
            else:
                raise RuntimeError("distractor pool exhausted before reaching the token target")

    fill_to(hop1_target)
    hop1_start = len(ids)
    ids.extend(hop1_ids)
    document_count += 1
    hop1_end = len(ids)

    fill_to(hop2_target)
    hop2_start = len(ids)
    ids.extend(hop2_ids)
    document_count += 1
    hop2_end = len(ids)
    if hop2_end > target_tokens - tail_tokens - 256:
        raise RuntimeError(f"hop 2 ended too close to the tail boundary: {hop2_end}")

    fill_to(target_tokens)
    ids = ids[:target_tokens]
    context = tokenizer.decode(ids, skip_special_tokens=False)
    roundtrip_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if abs(len(roundtrip_ids) - target_tokens) > 8:
        raise RuntimeError(
            f"serialized context moved too far from its token target: "
            f"{target_tokens} -> {len(roundtrip_ids)}"
        )
    serialization_changed = roundtrip_ids != ids
    ids = roundtrip_ids

    def serialized_span(title: str, text: str) -> tuple[int, int]:
        block = document_text(title, text)
        char_start = context.find(block)
        if char_start < 0:
            raise RuntimeError(f"serialized support document {title!r} was not found verbatim")
        token_start = len(tokenizer(context[:char_start], add_special_tokens=False)["input_ids"])
        token_end = len(
            tokenizer(context[: char_start + len(block)], add_special_tokens=False)["input_ids"]
        )
        return token_start, token_end

    if serialization_changed:
        hop1_start, hop1_end = serialized_span(candidate.hop1_title, candidate.hop1_text)
        hop2_start, hop2_end = serialized_span(candidate.hop2_title, candidate.hop2_text)
    tail = tokenizer.decode(ids[-tail_tokens:], skip_special_tokens=False)
    if contains_normalized(tail, candidate.answer):
        raise RuntimeError("answer leaked into the final tail")
    if contains_normalized(tail, candidate.hop1_title) or contains_normalized(tail, candidate.hop2_title):
        raise RuntimeError("a supporting title leaked into the final tail")
    if not contains_normalized(context, candidate.answer):
        raise RuntimeError("answer disappeared from the constructed context")

    return {
        "id": f"hotpotqa_interleaved/validation/{candidate.source_id}/v{variant}",
        "source_id": candidate.source_id,
        "variant": variant,
        "question": candidate.question,
        "answer": candidate.answer,
        "answers": [candidate.answer],
        "type": "bridge",
        "level": candidate.level,
        "context": context,
        "context_tokens_qwen": len(ids),
        "tail_tokens_qwen": tail_tokens,
        "answer_absent_from_tail": True,
        "support_titles_absent_from_tail": True,
        "supporting_facts": candidate.supporting_facts,
        "hop1": {
            "title": candidate.hop1_title,
            "token_start": hop1_start,
            "token_end": hop1_end,
        },
        "hop2": {
            "title": candidate.hop2_title,
            "token_start": hop2_start,
            "token_end": hop2_end,
        },
        "hop_distance_tokens": hop2_start - hop1_end,
        "document_count": document_count,
        "distractor_document_count": document_count - 2,
        "distractor_source_ids": unique(chosen_sources),
        "construction": {
            "source": "HotpotQA distractor validation only",
            "length_tokenizer": "qwen3-4b",
            "target_context_tokens": target_tokens,
            "placement_targets": [hop1_target, hop2_target],
            "question_in_context": False,
            "strict_directional_bridge": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--length-model", default="qwen3-4b")
    parser.add_argument("--target-context-tokens", type=int, default=28000)
    parser.add_argument("--tail-tokens", type=int, default=8192)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--limit-base", type=int)
    parser.add_argument("--max-output-bytes", type=int, default=950_000_000)
    args = parser.parse_args()
    if not 1 <= args.variants <= len(PLACEMENTS):
        raise ValueError(f"--variants must be between 1 and {len(PLACEMENTS)}")

    dataset = load_from_disk(str(args.source))["validation"]
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(args.length_model), local_files_only=True)
    candidates = strict_candidates(dataset)
    if args.limit_base is not None:
        candidates = candidates[: args.limit_base]
    print(f"strict base questions: {len(candidates)}", flush=True)
    pool = build_pool(dataset, tokenizer)
    print(f"unique natural distractor paragraphs: {len(pool)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    index_rows = []
    counts_by_variant: dict[int, int] = {}
    stopped_at_cap = False
    with temporary.open("wb") as handle:
        for variant in range(args.variants):
            for candidate_index, candidate in enumerate(candidates):
                row = build_row(
                    candidate,
                    variant,
                    pool,
                    tokenizer,
                    args.target_context_tokens,
                    args.tail_tokens,
                )
                payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                if handle.tell() + len(payload) > args.max_output_bytes:
                    stopped_at_cap = True
                    break
                offset = handle.tell()
                handle.write(payload)
                index_rows.append({
                    "offset": offset,
                    "bytes": len(payload),
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "variant": variant,
                })
                counts_by_variant[variant] = counts_by_variant.get(variant, 0) + 1
                if len(index_rows) % 100 == 0:
                    print(
                        f"rows={len(index_rows)} variant={variant} base={candidate_index + 1}/"
                        f"{len(candidates)} size_mb={handle.tell() / 1_000_000:.1f}",
                        flush=True,
                    )
            if stopped_at_cap:
                break
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)

    index_payload = {
        "version": 1,
        "data_file": args.output.name,
        "rows": index_rows,
    }
    index_path = args.output.with_suffix(args.output.suffix + ".index.json")
    index_path.write_text(json.dumps(index_payload, separators=(",", ":")), encoding="utf-8")
    manifest = {
        "version": 1,
        "dataset": "hotpotqa_interleaved",
        "split": "validation",
        "source": portable(args.source),
        "source_rows": len(dataset),
        "strict_base_questions_available": len(strict_candidates(dataset)),
        "strict_base_questions_requested": len(candidates),
        "variants_requested": args.variants,
        "rows_written": len(index_rows),
        "rows_by_variant": counts_by_variant,
        "unique_source_questions_written": len({entry["source_id"] for entry in index_rows}),
        "target_context_tokens_qwen": args.target_context_tokens,
        "tail_tokens_qwen": args.tail_tokens,
        "qwen_native_window": 40960,
        "global_am_chunk_tokens": 4096,
        "global_am_repeat_instruction_tokens": 19,
        "maximum_expected_global_am_position": args.target_context_tokens + 256 + 4096 + 19,
        "max_output_bytes": args.max_output_bytes,
        "stopped_at_size_cap": stopped_at_cap,
        "output_bytes": args.output.stat().st_size,
        "independence_caveat": (
            "Placement variants of the same source_id are not independent evaluation samples; "
            "use variant=0 for the primary unique-question evaluation."
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
