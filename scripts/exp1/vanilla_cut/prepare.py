#!/usr/bin/env python3
""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "multiagent/images/9_16/exp1"))

import build
from multiagent.dataset.c2c import apply_chat_template, chat_prompt_ids
from multiagent.dataset.c2c_registry import load_c2c_task
from multiagent.evaluation.utils import load_tokenizer
from multiagent.evaluation.vanilla import vanilla_messages
from multiagent.model.vanilla_loader import resolve_model_path

BENCHMARKS = ("longhealth", "quality_dev", "qasper_test100")
DEFAULT_OUT = ROOT / "multiagent/outputs/sandbox_agent/exp1_vanilla_matched_truncation_20260930"


def option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--controls", type=int, default=20, help="Unchanged rows rerun per job as a determinism check.")
    args = parser.parse_args()

    tokenizers: dict[str, object] = {}
    datasets: dict[str, dict] = {}

    def tokenizer(path: str):
        if path not in tokenizers:
            tokenizers[path] = load_tokenizer(Path(path))
        return tokenizers[path]

    plan, queue, seen = [], [], {}
    for pair in build.PAIRS:
        folder = pair.base(ROOT / "multiagent/outputs_mine") / pair.directory
        for benchmark in BENCHMARKS:
            vanilla_path = folder / f"eval_{benchmark}_{build.VANILLA_ARM.get(benchmark, 'vanilla')}_1x.json"
            vanilla = json.loads(vanilla_path.read_text())
            command = vanilla["evaluation_config"]["command_line"]
            max_length = int(option(command, "--max-length"))
            spec = option(command, "--tasks")
            receiver_path = str(resolve_model_path(option(command, "--model")))
            if spec not in datasets:
                datasets[spec] = {sample.id: sample for sample in load_c2c_task(spec).dataset}
            r_tok = tokenizer(receiver_path)

            changed, unchanged, digest = [], [], hashlib.sha256()
            for row in vanilla["rows"]:
                sample = datasets[spec][row["id"]]
                full = sample.messages or sample.phase1_messages
                old = chat_prompt_ids(r_tok, full, max_length, add_generation_prompt=True)
                new = chat_prompt_ids(r_tok, vanilla_messages(sample, r_tok, max_length), None, add_generation_prompt=True)
                digest.update(json.dumps([row["id"], new]).encode())
                if old == new:
                    unchanged.append(row["id"])
                    continue
                full_text = apply_chat_template(r_tok, full, add_generation_prompt=True)
                new_text = apply_chat_template(r_tok, vanilla_messages(sample, r_tok, max_length), add_generation_prompt=True)
                context = sample.metadata["context"]
                head = full_text[: full_text.index(context)]
                tail = full_text[full_text.index(context) + len(context):]
                kept = new_text[len(head): len(new_text) - len(tail)]
                if not (new_text.startswith(head) and new_text.endswith(tail) and context.startswith(kept)):
                    raise ValueError(f"{pair.tag} {benchmark} {row['id']}: new prompt lost the header or the question")
                changed.append(row["id"])
            step = max(1, len(unchanged) // args.controls) if unchanged else 1
            controls = unchanged[::step][: args.controls]

            key = (receiver_path, spec, max_length, digest.hexdigest())
            entry = {
                "pair": pair.tag, "benchmark": benchmark, "old_vanilla": str(vanilla_path.relative_to(ROOT)),
                "receiver": receiver_path, "rows": len(vanilla["rows"]),
                "changed": len(changed), "controls": len(controls),
            }
            if key in seen:
                entry["shares_run_with"] = seen[key]
                plan.append(entry)
                print(f"{pair.tag:5} {benchmark:15} changed {len(changed):4}/{len(vanilla['rows'])}  -> shares run with {seen[key]}")
                continue
            seen[key] = pair.tag
            job_dir = args.out / pair.tag
            job_dir.mkdir(parents=True, exist_ok=True)
            ids_path = job_dir / f"{benchmark}_ids.json"
            ids_path.write_text(json.dumps({"changed": changed, "controls": controls}, indent=1))
            run_ids = job_dir / f"{benchmark}_run_ids.json"
            run_ids.write_text(json.dumps(changed + controls))
            output = job_dir / f"eval_{benchmark}_vanilla_cut_1x.json"
            new_command = list(command)
            new_command[new_command.index("--output") + 1] = str(output.relative_to(ROOT / "multiagent"))
            new_command += ["--sample-ids", str(run_ids.relative_to(ROOT / "multiagent"))]
            new_command = [str(ROOT / "multiagent/evaluation/vanilla.py")
                           if token.endswith("eval_unaware_comm.py") else token for token in new_command]
            if new_command[0].endswith("vanilla.py"):
                new_command = ["../env3.10/bin/python", *new_command]
            queue.append({"cmd": " ".join(new_command), "gpus": 1})
            entry["output"] = str(output.relative_to(ROOT))
            plan.append(entry)
            print(f"{pair.tag:5} {benchmark:15} changed {len(changed):4}/{len(vanilla['rows'])}  controls {len(controls)}")

    queue.sort(key=lambda job: "longhealth" not in job["cmd"])
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1))
    (args.out / "queue.json").write_text(json.dumps(queue, indent=1))
    runs = [p for p in plan if "shares_run_with" not in p]
    print(f"{len(queue)} jobs, {sum(p['changed'] for p in runs)} changed rows + {sum(p['controls'] for p in runs)} control rows")


if __name__ == "__main__":
    main()
