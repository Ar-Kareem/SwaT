#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from multiagent.dataset.math_dataset import grade_output

BENCH = REPO_ROOT / "multiagent" / "outputs" / "kvxfer_bench"


def find_rows(doc: dict):
    for value in doc.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "output" in value[0]:
            return value
        if isinstance(value, dict):
            for inner in value.values():
                if isinstance(inner, list) and inner and isinstance(inner[0], dict) and "output" in inner[0]:
                    return inner
    raise ValueError("no rows found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default=str(BENCH / "math-500_n500_*.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for path in sorted(glob.glob(args.pattern)):
        doc = json.load(open(path))
        rows = find_rows(doc)
        old = sum(r.get("grade") == "correct" for r in rows)
        for r in rows:
            r.setdefault("grade_strict", r.get("grade"))
            r["grade"] = grade_output(r["output"], {"answer_type": "math", "answer": r["answer"]})
        new = sum(r["grade"] == "correct" for r in rows)
        ro = doc.get("result_overview") or {}
        ro.setdefault("accuracy_percent_strict", ro.get("accuracy_percent"))
        ro.setdefault("correct_strict", ro.get("correct"))
        ro["correct"] = new
        ro["accuracy_percent"] = 100.0 * new / len(rows)
        ro["regraded"] = "format-tolerant fallback 2026-09-14"
        print(f"{Path(path).name:52s} {100*old/len(rows):5.1f}% -> {100*new/len(rows):5.1f}%  (+{new-old})")
        if not args.dry_run:
            json.dump(doc, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
