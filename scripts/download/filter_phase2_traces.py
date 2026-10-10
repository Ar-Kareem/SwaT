#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = REPO_ROOT / "multiagent" / "data" / "phase2_traces"
LEAK_PATTERN = re.compile(r"ground[- ]truth", re.IGNORECASE)
LEAD_LEAK = re.compile(
    r"\s*to (?:arrive at|reach|get|obtain|find|determine|match)(?: the)? ground[- ]truth answer(?: of)?[^.\n]*",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=TRACE_DIR / "llama3-8b-instruct_gsm8k_train.jsonl")
    parser.add_argument("--output", type=Path, default=TRACE_DIR / "llama3-8b-instruct_gsm8k_train_clean.jsonl")
    return parser.parse_args()


def strip_lead_leak(row: dict) -> bool:
    trace = LEAD_LEAK.sub("", row["trace"], count=1)
    if trace == row["trace"]:
        return False
    row["trace"] = trace
    return True


def reasons(row: dict) -> list[str]:
    out = []
    if row.get("stop_reason") == "cutoff":
        out.append("cutoff")
    if "raw_trace" in row:
        out.append("patched")
    if LEAK_PATTERN.search(row.get("trace", "")):
        out.append("ground-truth")
    return out


def main() -> None:
    args = parse_args()
    with args.input.open() as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    rows = [json.loads(line) for line in lines]
    counts = {"cutoff": 0, "patched": 0, "ground-truth": 0}
    kept: list[str] = []
    dropped = stripped = 0
    for row in rows:
        stripped += strip_lead_leak(row)
        why = reasons(row)
        for key in why:
            counts[key] += 1
        if why:
            dropped += 1
        else:
            kept.append(json.dumps(row, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n")
    datasets = sorted({row["dataset"] for row in rows})
    print(f"input   : {args.input} ({len(rows)} rows, datasets={datasets})")
    print(f"stripped: {stripped} leaked opening clauses")
    print(f"dropped : {dropped} (cutoff {counts['cutoff']}, patched {counts['patched']}, ground-truth {counts['ground-truth']}; overlapping)")
    print(f"kept    : {len(kept)} ({100 * len(kept) / len(rows):.1f}%) -> {args.output}")


if __name__ == "__main__":
    main()
