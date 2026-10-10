#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json

from make_renderer15_queue import (
    PY, stage1_cmd, stage2_cmd, renderer_cb, renderer_1x, CLOSED_BOOK, EVALS_1X,
)

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "multiagent"
NAME, SENDER, RECEIVER, MEM = "15I", "qwen3-32b", "llama3-8b-instruct", "bigsender"
BASE = "swat/14E/stage2"
STAGE1 = f"outputs_mine/swat/{NAME}/stage1"
STAGE2 = f"outputs_mine/swat/{NAME}/stage2"
TRACES = "data/phase2_traces/llama3-8b-instruct_gsm8k_train_clean.jsonl"
SLOW_FIRST = ("math-500:test", "gsm8k:test")


def mine(cmd):
    return [p.replace("outputs/kvxfer_bench/", "outputs_mine/kvxfer_bench/") for p in cmd]


def two_gpu(cmd):
    return cmd + ["--sender-device cuda:1"]


def copy_baselines(target: Path) -> dict:
    sources = {}
    for stem, _, _ in EVALS_1X:
        if "boosted" in stem:
            continue
        arms = ["vanilla"] + (["vanilla_terse"] if stem.startswith(("hotpotqa_", "musique_", "squad_")) else [])
        for arm in arms:
            filename = "eval_" + stem.replace("renderer", arm) + ".json"
            src = MULTI / "outputs_mine" / BASE / filename
            data = json.loads(src.read_text())
            assert "Meta-Llama-3-8B-Instruct" in data["receiver_model"], src
            assert data["rows"], src
            dest = target / filename
            payload = src.read_bytes()
            if dest.exists() and dest.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite different baseline: {dest}")
            dest.write_bytes(payload)
            sources[filename] = {"source": str(src.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(),
                                 "rows": len(data["rows"])}
    for spec, _ in CLOSED_BOOK:
        path = MULTI / "outputs_mine/kvxfer_bench" / f"{spec.split(':')[0]}_n500_vanilla.json"
        data = json.loads(path.read_text())
        assert len(data["rows"]) == 500, path
        assert "Meta-Llama-3-8B-Instruct" in data["receiver_model"], path
    assert len((MULTI / TRACES).read_text().splitlines()) >= 6500
    return sources


def main():
    target = MULTI / STAGE2
    target.mkdir(parents=True, exist_ok=True)
    assert not (MULTI / STAGE1 / "exit-checkpoint-5000").exists(), "fresh run: stage 1 checkpoint already exists"
    sources = copy_baselines(target)
    (target / "baseline_provenance.json").write_text(json.dumps(sources, indent=2) + "\n")

    jobs = []

    def add(name, cmd, gpus, after=()):
        jobs.append(dict(name=name, cmd=cmd, gpus=gpus, after=list(after)))

    add("stage1", stage1_cmd(SENDER, RECEIVER, STAGE1, MEM), 2)
    add("stage2", stage2_cmd(SENDER, RECEIVER, STAGE1, STAGE2, TRACES, MEM) + ["--checkpoint-receiver"], 2, ["stage1"])
    eval_names = []
    order = sorted(CLOSED_BOOK, key=lambda x: (x[0] not in SLOW_FIRST, SLOW_FIRST.index(x[0]) if x[0] in SLOW_FIRST else 0))
    for spec, mnt in order:
        name = "eval_" + spec.split(":")[0]
        add(name, two_gpu(mine(renderer_cb(STAGE2, NAME, spec, mnt))), 2, ["stage2"])
        eval_names.append(name)
    for stem, task, limit in EVALS_1X:
        name = "eval_" + stem.replace("_renderer_1x", "")
        add(name, two_gpu(renderer_1x(STAGE2, stem, task, limit)), 2, ["stage2"])
        eval_names.append(name)
    add("exp1_tables", f"{PY} images/9_16/exp1/build.py", 1, eval_names)
    queue = MULTI / "gpu_queue_commands/gpu_queue_commands_renderer15I.json"
    queue.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Prepared {len(jobs)} jobs: {queue}")
    print(f"Reused {len(sources)} context baselines and 8 short-task baselines; provenance: {target}/baseline_provenance.json")


if __name__ == "__main__":
    main()
