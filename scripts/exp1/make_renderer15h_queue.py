#!/usr/bin/env python3
from pathlib import Path
import json

from make_renderer15_queue import (
    PY, stage1_cmd, stage2_cmd, vanilla_cb, renderer_cb, vanilla_1x, renderer_1x, CLOSED_BOOK, EVALS_1X,
)

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "multiagent"
NAME, SENDER, RECEIVER, ALIAS, MEM = "15H", "gemma-3-12b-it", "qwen3-4b", "qwen3-4b", "qwen4b"
STAGE1 = f"outputs_mine/swat/{NAME}/stage1"
STAGE2 = f"outputs_mine/swat/{NAME}/stage2"
DIAG = "outputs/sandbox_agent/exp1_gemma3_12b_qwen3_4b_20260925"
RAW = f"data/phase2_traces/{ALIAS}_gsm8k_train.jsonl"
CLEAN = f"data/phase2_traces/{ALIAS}_gsm8k_train_clean.jsonl"
TERSE = ("hotpotqa_val400", "musique_val400", "squad_validation20pct")
SLOW_FIRST = ("math-500:test", "gsm8k:test")


def mine(cmd):
    return [p.replace("outputs/kvxfer_bench/", "outputs_mine/kvxfer_bench/") for p in cmd]


def main():
    (MULTI / STAGE2).mkdir(parents=True, exist_ok=True)
    assert (MULTI / "data/phase2_traces/qwen3-8b_gsm8k_train_clean.jsonl").exists()
    for f in (f"outputs_mine/kvxfer_bench/gsm8k_n500_vanilla_{ALIAS}.json", RAW, CLEAN):
        assert not (MULTI / f).exists(), f"fresh run: {f} already exists"
    jobs = []

    def add(name, cmd, gpus, after=()):
        jobs.append(dict(name=name, cmd=cmd, gpus=gpus, after=list(after)))

    a = stage1_cmd(SENDER, RECEIVER, STAGE1, MEM)
    b = stage2_cmd(SENDER, RECEIVER, STAGE1, STAGE2, CLEAN, MEM) + ["--checkpoint-receiver"]
    pre_a = [p.replace(STAGE1, DIAG + "/stage1_preflight") for p in a] + ["--preflight-only"]
    pre_b = [p.replace(STAGE2, DIAG + "/stage2_preflight") for p in b if not p.startswith("--init-weights-from")] + ["--preflight-only"]
    add("preflight_kv", pre_a, 2)

    add("traces", [f"{PY} create_phase2_dataset.py --receiver-model {RECEIVER} --device cuda:0 --batch-size 32",
                   f"--max-new-tokens 512 --max-length 2048 --tasks gsm8k:train --output {RAW}"], 1, ["preflight_kv"])
    vanilla_names = []
    for spec, mnt in CLOSED_BOOK:
        name = "vanilla_" + spec.split(":")[0]
        add(name, mine(vanilla_cb(RECEIVER, ALIAS, spec, mnt)), 1, ["preflight_kv"])
        vanilla_names.append(name)
    for stem, task, limit in EVALS_1X:
        name = "vanilla_" + stem.replace("_renderer_1x", "")
        add(name, vanilla_1x(RECEIVER, STAGE2, stem, task, limit), 1, ["preflight_kv"])
        vanilla_names.append(name)
        if stem.replace("_renderer_1x", "") in TERSE:
            ds, split = task.split(":")
            add(name + "_terse", vanilla_1x(RECEIVER, STAGE2, stem.replace("_renderer_", "_vanilla_terse_"), f"{ds}+terse:{split}", limit),
                1, ["preflight_kv"])
            vanilla_names.append(name + "_terse")
    add("filter_traces", f"test $(wc -l < {RAW}) -ge 7400 && {PY} ../scripts/download/filter_phase2_traces.py --input {RAW} --output {CLEAN} "
        f"&& test $(wc -l < {CLEAN}) -ge 6500", 1, ["traces"])
    add("preflight_ce_dense", pre_b, 2, ["filter_traces"])
    add("stage1", a, 2, ["preflight_ce_dense"])
    add("stage2", b, 2, ["stage1"])

    eval_names = []
    order = sorted(CLOSED_BOOK, key=lambda x: (x[0] not in SLOW_FIRST, SLOW_FIRST.index(x[0]) if x[0] in SLOW_FIRST else 0))
    for spec, mnt in order:
        name = "eval_" + spec.split(":")[0]
        add(name, mine(renderer_cb(STAGE2, NAME, spec, mnt)), 1, ["stage2"])
        eval_names.append(name)
    for stem, task, limit in EVALS_1X:
        name = "eval_" + stem.replace("_renderer_1x", "")
        add(name, renderer_1x(STAGE2, stem, task, limit), 1, ["stage2"])
        eval_names.append(name)
    add("exp1_tables", f"{PY} images/9_16/exp1/build.py", 1, eval_names + vanilla_names)
    queue = MULTI / "gpu_queue_commands/gpu_queue_commands_renderer15H.json"
    queue.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Prepared {len(jobs)} jobs: {queue}")


if __name__ == "__main__":
    main()
