#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg19", type=Path, required=True)
    parser.add_argument("--wikitext", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=1221)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    pg19 = read_rows(args.pg19)
    wiki = read_rows(args.wikitext)
    if min(len(pg19), len(wiki)) < args.samples_per_source:
        raise ValueError(
            f"need {args.samples_per_source} rows/source; found pg19={len(pg19)}, wikitext={len(wiki)}"
        )
    rng = random.Random(args.seed)
    pg19 = rng.sample(pg19, args.samples_per_source)
    wiki = rng.sample(wiki, args.samples_per_source)
    rows = [*pg19, *wiki]
    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    receiver_tokens = sum(
        int(row["receiver_prefix_tokens"]) + int(row["receiver_continuation_tokens"]) for row in rows
    )
    manifest = {
        "schema": "multikv.renderer_continuation_mix_manifest.v1",
        "output": str(args.output),
        "seed": args.seed,
        "samples": len(rows),
        "samples_per_source": args.samples_per_source,
        "total_receiver_tokens": receiver_tokens,
        "sources": {"pg19": str(args.pg19), "wikitext103": str(args.wikitext)},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows):,} rows / {receiver_tokens:,} receiver tokens to {args.output}")


if __name__ == "__main__":
    main()
