#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json

from make_renderer15_queue import (
    stage1_cmd, stage2_cmd, renderer_cb, renderer_1x, CLOSED_BOOK, EVALS_1X,
)

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "multiagent"
BASE = "swat/14E/stage2"
STAGE1 = "outputs_mine/swat/15G/stage1"
STAGE2 = "outputs_mine/swat/15G/stage2"
DIAG = "outputs/sandbox_agent/exp1_gemma3_llama3_20260924"
PYTHON = "../env3.10/bin/python"


def main():
    target = MULTI / STAGE2
    target.mkdir(parents=True, exist_ok=True)
    sources = {}
    for stem, _, _ in EVALS_1X:
        if "boosted" in stem:
            continue
        arm = "vanilla_terse" if stem.startswith(("hotpotqa_", "musique_", "squad_")) else "vanilla"
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
        sources[filename] = {"source": str(src.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(data["rows"])}
    for spec, _ in CLOSED_BOOK:
        path = MULTI / "outputs_mine/kvxfer_bench" / f"{spec.split(':')[0]}_n500_vanilla.json"
        data = json.loads(path.read_text())
        assert len(data["rows"]) == 500, path
        assert "Meta-Llama-3-8B-Instruct" in data["receiver_model"], path
    traces = "data/phase2_traces/llama3-8b-instruct_gsm8k_train_clean.jsonl"
    assert len((MULTI / traces).read_text().splitlines()) >= 6500
    (target / "baseline_provenance.json").write_text(json.dumps(sources, indent=2) + "\n")
    a = stage1_cmd("gemma-3-4b-it", "llama3-8b-instruct", STAGE1, "full")
    b = stage2_cmd("gemma-3-4b-it", "llama3-8b-instruct", STAGE1, STAGE2, traces, "full")
    b.append("--checkpoint-receiver")
    pre_a = [p.replace(STAGE1, DIAG + "/stage1_preflight") for p in a] + ["--preflight-only"]
    pre_b = [p.replace(STAGE2, DIAG + "/stage2_preflight") for p in b if not p.startswith("--init-weights-from")] + ["--preflight-only"]
    jobs = []
    def add(name, cmd, gpus, after=()):
        jobs.append(dict(name=name, cmd=cmd, gpus=gpus, after=list(after)))
    add("preflight_kv", pre_a, 2)
    add("preflight_ce_dense", pre_b, 2, ["preflight_kv"])
    add("stage1", a, 2, ["preflight_ce_dense"])
    add("stage2", b, 2, ["stage1"])
    eval_names = []
    for spec, mnt in CLOSED_BOOK:
        name = "eval_" + spec.split(":")[0]
        cmd = [p.replace("outputs/kvxfer_bench/", "outputs_mine/kvxfer_bench/") for p in renderer_cb(STAGE2, "15G", spec, mnt)]
        add(name, cmd, 1, ["stage2"])
        eval_names.append(name)
    for stem, task, limit in EVALS_1X:
        if "boosted" in stem:
            continue
        name = "eval_" + stem
        add(name, renderer_1x(STAGE2, stem, task, limit), 1, ["stage2"])
        eval_names.append(name)
    add("regrade_math", [f"{PYTHON} ../scripts/exp1/regrade_math500.py",
        "--pattern outputs_mine/kvxfer_bench/math-500_n500_15G.json"], 1, eval_names)
    add("exp1_tables", f"{PYTHON} images/9_16/exp1/build.py", 1, ["regrade_math"])
    queue = MULTI / "gpu_queue_commands/gpu_queue_commands_renderer15G.json"
    queue.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Prepared {len(jobs)} jobs: {queue}")
    print(f"Reused {len(sources)} context baselines and 8 short-task baselines; provenance: {target}")


if __name__ == "__main__":
    main()
