#!/usr/bin/env python3
""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from multiagent.evaluation import results as E

DEFAULT_OUT = ROOT / "multiagent/outputs/sandbox_agent/exp1_vanilla_matched_truncation_20260930"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    plan = json.loads((args.out / "plan.json").read_text())
    runs = {(p["pair"], p["benchmark"]): p for p in plan if "output" in p}
    for entry in plan:
        run = runs[(entry.get("shares_run_with", entry["pair"]), entry["benchmark"])]
        output = ROOT / run["output"]
        changed = set(json.loads((output.parent / f"{entry['benchmark']}_ids.json").read_text())["changed"])
        new = {row["id"]: row for row in json.loads(output.read_text())["rows"]}
        path = ROOT / entry["old_vanilla"]
        data = json.loads(path.read_text())

        rows = []
        for row in data["rows"]:
            if row["id"] in changed:
                replacement = new[row["id"]]
                if (replacement["task_key"], replacement["index"]) != (row["task_key"], row["index"]):
                    raise ValueError(f"{path}: rerun row {row['id']} does not match the original position")
                row = replacement
            rows.append(row)
        if sum(row["id"] in changed for row in rows) != len(changed):
            raise ValueError(f"{path}: not every changed row was found")

        stats = {spec: {"seen": 0, "correct": 0, "incorrect": 0, "fail": 0} for spec in data["tasks"]}
        stop_stats = {spec: {"eos": 0, "cutoff": 0, "empty_prompt": 0} for spec in data["tasks"]}
        metric_stats: dict = {}
        for row in rows:
            task_key = row["task_key"]
            label, metrics, _ = E.normalize_grade_result(row.get("grade_result", row.get("grade", "fail")))
            stats[task_key]["seen"] += 1
            stats[task_key][label] = stats[task_key].get(label, 0) + 1
            E.update_metric_stats(metric_stats, task_key, metrics)
            stop_stats[task_key][row["stop_reason"]] = stop_stats[task_key].get(row["stop_reason"], 0) + 1

        data["rows"] = rows
        data["summary"] = E.summarize(stats, stop_stats, metric_stats)
        data["evaluation_config"]["input_truncation"] = {
            "max_length": data["evaluation_config"]["input_truncation"]["max_length"],
            "side": "context_prefix",
        }
        E.save_json(path, data)
        print(f"{entry['pair']:5} {entry['benchmark']:15} replaced {len(changed):4}/{len(rows)} rows -> {entry['old_vanilla']}")


if __name__ == "__main__":
    main()
