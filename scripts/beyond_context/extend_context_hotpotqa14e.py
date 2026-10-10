from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "multiagent"
sys.path.insert(0, str(BASE.parent))
OUT = BASE / "outputs/extend_context"
TASK = OUT / "extend_context_hotpotqa_renderer14e"
OLD = OUT / "hotpotqa_intsub_n500_renderer8_hybrid_am20_llama31_nonuniform.json"
NEW = OUT / "hotpotqa_intsub_n500_renderer14e_hybrid_am20_llama31_nonuniform.json"
RENDERER = BASE / "outputs_mine/swat/14E/stage2/exit-checkpoint-24000/renderer.pt"
QUERIES = BASE / "outputs_mine/swat_query_transfer/14E/gold-checkpoint-4000/query_transfer.pt"
BUDGET = BASE / "model/head_budget_optimization/head_budgets/Llama-3.1-8B-Instruct/optimized_agnostic.json"
DATA = BASE / "data/extend_context/hotpotqa_interleaved_28k_validation.jsonl"
SOURCES = {
    "Renderer14E → Llama, nonuniform AM20": NEW,
    "Renderer8 → Llama, nonuniform AM20": OLD,
    "Llama independent chunks, nonuniform AM20": OUT / "hotpotqa_intsub_n500_llama3_8b_independent_am20_llama31_nonuniform.json",
    "Qwen full context": OUT / "hotpotqa_interleaved28k_v0_n2182_qwen3_4b_fullctx32k.json",
    "Qwen global AM20": OUT / "hotpotqa_intsub_n500_qwen3_4b_global_am20.json",
    "Qwen independent chunks AM20": OUT / "hotpotqa_intsub_n500_qwen3_4b_independent_am20.json",
    "Llama last 8K": OUT / "hotpotqa_interleaved28k_v0_n2182_llama3_8b_cut8k.json",
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    from transformers import AutoTokenizer
    from multiagent.dataset.c2c import chat_prompt_ids
    from multiagent.dataset.c2c_registry import load_c2c_task
    from multiagent.evaluation.beyond_context import parse_args
    from multiagent.evaluation.utils import interleaved_items

    TASK.mkdir(parents=True, exist_ok=True)
    manifest = TASK / "experiment.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    old = read(OLD)
    args = old["evaluation_config"]["command_line"][1:]
    route = args.index("--beyond-context")
    if args[route + 1] != "hybrid":
        raise ValueError("expected a historical hybrid run")
    del args[route:route + 2]
    args.insert(0, "hybrid")
    overrides = {"--renderer": RENDERER, "--am-query-transfer": QUERIES,
                 "--am-head-budget-path": BUDGET, "--output": NEW}
    for flag, value in overrides.items():
        args[args.index(flag) + 1] = str(value)
    args += ["--resume-from", str(NEW)]
    saved_argv = sys.argv
    try:
        sys.argv = ["evaluation/beyond_context.py", *args]
        parsed = vars(parse_args())
    finally:
        sys.argv = saved_argv
    exempt = {flag[2:].replace("-", "_") for flag in overrides} | {"resume_from"}
    for key, value in old["evaluation_config"]["arguments"].items():
        if key not in exempt and parsed.get(key) != value:
            raise ValueError(f"argument drift: {key}: {value!r} -> {parsed.get(key)!r}")

    spec = old["tasks"][0]
    dataset = load_c2c_task(spec, root=parsed["dataset_root"]).dataset
    schedule = list(interleaved_items({spec: dataset}, 1, 500, 42))
    assert [sample.id for _, _, sample in schedule] == [row["id"] for row in old["rows"]]
    qwen = AutoTokenizer.from_pretrained(old["sender_model"], local_files_only=True)
    llama = AutoTokenizer.from_pretrained(old["receiver_model"], local_files_only=True)
    qwen.truncation_side = llama.truncation_side = "left"
    samples = []
    for (_, index, sample), row in zip(schedule, old["rows"]):
        assert index == row["index"] and sample.target_text == row["target_text"]
        assert sample.metadata["question"] == row["question"]
        sender_len = len(chat_prompt_ids(qwen, sample.sender_messages, 32768, False))
        receiver_len = len(chat_prompt_ids(llama, sample.sender_messages, 32768, False))
        decode_len = len(chat_prompt_ids(llama, sample.receiver_messages, 32768, True))
        expected = re.search(r"/fullqwen(\d+)/chunks\d+/span\d+-\d+of(\d+)", row["stop_reason"])
        expected_decode = re.search(r"/\d+recv\+(\d+)/", row["stop_reason"])
        assert expected and (sender_len, receiver_len) == tuple(map(int, expected.groups())), sample.id
        assert expected_decode and decode_len == int(expected_decode.group(1)), sample.id
        samples.append({"id": sample.id, "index": index, "qwen_context_prompt_tokens": sender_len,
                        "llama_context_prompt_tokens": receiver_len, "llama_question_prompt_tokens": decode_len,
                        "context_sha256": hashlib.sha256(sample.metadata["context"].encode()).hexdigest()})
    watched = [RENDERER, QUERIES, BUDGET, DATA, DATA.with_suffix(".jsonl.index.json"),
               BASE / "evaluation/renderer_am.py", BASE / "evaluation/utils.py", BASE / "evaluation/results.py",
               BASE / "evaluation/beyond_context.py", BASE / "model/attention_matching.py",
               BASE / "dataset/hotpotqa_interleaved_dataset.py", BASE / "dataset/c2c.py",
               BASE / "dataset/c2c_cross_family.py", BASE / "dataset/c2c_registry.py", Path(__file__)]
    watched += list(SOURCES.values())[1:]
    write(manifest, {"notes": {"head_budget_note": "Historical Llama-3.1-8B proxy budget retained for comparability."},
                     "original_result": str(OLD), "output": str(NEW), "argv": args,
                     "sample_count": 500, "samples": samples,
                     "validated": "All 500 ordered IDs, indices, questions, gold answers, sender/receiver context and question prompt token counts match Renderer8; all historical arguments match except checkpoint/path relocation, output, and resume.",
                     "sha256": {str(p): digest(p) for p in watched}})
    write(TASK / "queue_commands.json", [{"name": "hotpotqa_renderer14e_nonuniform_am20", "gpus": 1,
          "cmd": shlex.join([sys.executable, "-u", str(Path(__file__).resolve()), "run"])}])
    print(f"Validated all {len(samples)} original samples; saved {manifest}", flush=True)


def report(require_complete=False):
    import numpy as np

    ids = [row["id"] for row in read(OLD)["rows"]]
    metrics, scores = [], {}
    for label, path in SOURCES.items():
        if not path.exists():
            if require_complete:
                raise FileNotFoundError(path)
            continue
        rows = read(path)["rows"]
        mapping = {row["id"]: row for row in rows}
        assert len(mapping) == len(rows), f"duplicate IDs in {path}"
        if not all(key in mapping for key in ids):
            if require_complete:
                raise ValueError(f"incomplete results: {path}")
            continue
        selected = [mapping[key] for key in ids]
        scores[label] = np.array([[row["grade_result"][metric] for metric in ("em", "f1")] for row in selected])
        em, f1 = 100 * scores[label].mean(axis=0)
        metrics.append({"method": label, "n": len(ids), "em": float(em), "f1": float(f1), "source": str(path)})
    comparisons = []
    reference = next(iter(SOURCES))
    if reference in scores:
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(ids), size=(20000, len(ids)))
        for label, values in scores.items():
            if label == reference:
                continue
            diff = scores[reference] - values
            ci = np.quantile(diff[indices].mean(axis=1) * 100, [0.025, 0.975], axis=0)
            comparisons.append({"reference": reference, "other": label,
                "em_delta": float(diff[:, 0].mean() * 100), "f1_delta": float(diff[:, 1].mean() * 100),
                "em_paired_bootstrap_ci95": ci[:, 0].tolist(), "f1_paired_bootstrap_ci95": ci[:, 1].tolist()})
    write(TASK / "summary.json", {"complete": reference in scores, "metrics": metrics, "paired_comparisons": comparisons})
    with (TASK / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "n", "em", "f1", "source"])
        writer.writeheader()
        writer.writerows(metrics)
    lines = ["# Interleaved HotpotQA, original 500 questions", "",
             "Renderer14E: " + ("complete" if reference in scores else "pending; table contains historical baselines"), "",
             "All methods use the same 500 IDs. Historical nonuniform methods retain the Llama-3.1 proxy head budget.", "",
             "| Method | N | EM | F1 |", "|---|---:|---:|---:|"]
    lines += [f"| {m['method']} | {m['n']} | {m['em']:.2f} | {m['f1']:.2f} |" for m in metrics]
    for c in comparisons:
        low, high = c["f1_paired_bootstrap_ci95"]
        lines += ["", f"14E minus {c['other']}: F1 {c['f1_delta']:+.2f}, paired 95% bootstrap CI [{low:+.2f}, {high:+.2f}]."]
    (TASK / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def run():
    manifest = read(TASK / "experiment.json")
    for path, expected in manifest["sha256"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Input or code changed since preparation: {path}")
    subprocess.run([sys.executable, "-u", str(BASE / "evaluation/beyond_context.py"), *manifest["argv"]], cwd=BASE, check=True)
    rows = read(NEW)["rows"]
    assert [row["id"] for row in rows] == [sample["id"] for sample in manifest["samples"]]
    report(require_complete=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "report", "run"])
    action = parser.parse_args().action
    {"prepare": prepare, "report": report, "run": run}[action]()
