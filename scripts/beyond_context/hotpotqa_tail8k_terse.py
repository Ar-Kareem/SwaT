from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BASE = REPO / "multiagent"
ROOT = BASE / "outputs/extend_context"
OUT = ROOT / "extend_context_hotpotqa_renderer14e"
RESULT = OUT / "llama_tail_8k_terse.json"
PLAN = OUT / "experiment_llama_tail_8k_terse.json"
OLD = ROOT / "hotpotqa_interleaved28k_v0_n2182_llama3_8b_cut8k.json"
RENDERER = ROOT / "hotpotqa_intsub_n500_renderer14e_hybrid_am20_llama31_nonuniform.json"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def terse_messages(context, question):
    from multiagent.dataset.hotpotqa_interleaved_dataset import context_messages
    from multiagent.dataset.qa_dataset import qa_instruction

    return [{"role": "user", "content": context_messages(context)[0]["content"]
             + "\n\n" + qa_instruction("terse")
             + f"\n\nQuestion: {question.strip()}\n\nAnswer:"}]


def prepare():
    from transformers import AutoTokenizer
    from multiagent.dataset.c2c import chat_prompt_ids
    from multiagent.dataset.c2c_registry import load_c2c_task
    from multiagent.dataset.qa_dataset import qa_instruction
    from multiagent.evaluation.utils import interleaved_items

    if PLAN.exists() or RESULT.exists():
        raise FileExistsError("Preserving existing terse baseline; use run to resume")
    old = read(OLD)
    reference = read(RENDERER)["rows"]
    task_key = old["tasks"][0]
    task = load_c2c_task(task_key, root=old["evaluation_config"]["arguments"]["dataset_root"])
    schedule = list(interleaved_items({task_key: task.dataset}, 1, 500, 42))
    assert len(reference) == len(schedule) == 500
    assert [s.id for _, _, s in schedule] == [r["id"] for r in reference]
    previous = {r["id"]: r for r in old["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(old["receiver_model"], local_files_only=True)
    tokenizer.truncation_side = "left"
    samples = []
    for (_, index, sample), ref in zip(schedule, reference):
        assert sample.target_text == ref["target_text"] == previous[sample.id]["target_text"]
        assert sample.metadata["question"] == ref["question"] == previous[sample.id]["question"]
        assert task.grade(previous[sample.id]["output"], sample) == previous[sample.id]["grade_result"]
        context, question = sample.metadata["context"], sample.metadata["question"]
        messages = terse_messages(context, question)
        ids = chat_prompt_ids(tokenizer, messages, 100000, True)
        used = chat_prompt_ids(tokenizer, messages, 8192, True)
        assert used == ids[-8192:] and len(used) == 8192
        tail = tokenizer.decode(used, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert qa_instruction("terse") in tail and question.strip() in tail
        samples.append({"id": sample.id, "index": index, "target_text": sample.target_text,
                        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                        "full_prompt_tokens": len(ids), "input_tokens": len(used),
                        "input_ids_sha256": hashlib.sha256(json.dumps(used).encode()).hexdigest()})
    argv = old["evaluation_config"]["command_line"][1:]
    for flag, value in {"--limit-per-dataset": "500", "--save-every": "1",
                        "--resume-from": str(RESULT), "--output": str(RESULT)}.items():
        argv[argv.index(flag) + 1] = value
    watched = [Path(__file__).resolve(), OLD, RENDERER, task.dataset.path,
               BASE / "dataset/hotpotqa_interleaved_dataset.py",
               BASE / "dataset/qa_dataset.py", BASE / "dataset/c2c.py", BASE / "evaluation/vanilla.py", BASE / "evaluation/results.py", BASE / "evaluation/utils.py",
               BASE / "model/vanilla_loader.py"]
    save(PLAN, {"sample_count": 500, "model": old["receiver_model"], "argv": argv,
                "instruction": qa_instruction("terse"), "samples": samples,
                "protocol": "Same ordered 500 IDs as Renderer14E; same model, BF16, seed 42, greedy decoder, EOS and grader as historical tail baseline. Replace only question-suffix instruction with qa_instruction('terse'); retain Answer: cue. Last 8192 prompt tokens, up to 64 generated tokens, matching historical baseline (no output reserve subtracted). No prediction trimming changes.",
                "sha256": {str(p): digest(p) for p in watched}})
    save(OUT / "queue_llama_tail_8k_terse.json", [{"name": "hotpotqa_tail8k_terse", "gpus": 1,
         "cmd": shlex.join([sys.executable, "-u", str(Path(__file__).resolve()), "run"])}])
    print(f"Validated all 500 matched IDs, gold answers, historical grades, and retained terse prompts. Saved {PLAN}", flush=True)


def report(require_complete=False):
    ids = [r["id"] for r in read(PLAN)["samples"]]
    sources = {"Llama last 8K, original": OLD, "Llama last 8K, terse": RESULT,
               "Renderer14E + AM20": RENDERER}
    maps = {}
    for label, path in sources.items():
        rows = read(path)["rows"] if path.exists() else []
        maps[label] = {r["id"]: r for r in rows}
        assert len(maps[label]) == len(rows)
    matched = [i for i in ids if all(i in m for m in maps.values())]
    complete = len(matched) == 500
    if require_complete and not complete:
        raise ValueError(f"Only {len(matched)}/500 matched results")
    table = []
    for label, mapping in maps.items():
        rows = [mapping[i] for i in matched]
        table.append({"method": label, "n": len(rows),
                      **{metric: 100 * sum(r["grade_result"][metric] for r in rows) / len(rows)
                         if rows else None for metric in ("em", "f1")},
                      "mean_generated_tokens": sum(r["generated_tokens"] for r in rows) / len(rows) if rows else None,
                      "cutoff_count": sum(r["stop_reason"] == "cutoff" for r in rows),
                      "source": str(sources[label])})
    result = {"complete": complete, "matched_n": len(matched), "expected_n": 500, "table": table}
    save(OUT / "summary_llama_tail_8k_terse.json", result)
    print(json.dumps(result, indent=2), flush=True)


def run():
    from multiagent.dataset import hotpotqa_interleaved_dataset
    from multiagent.evaluation import vanilla

    plan = read(PLAN)
    for path, expected in plan["sha256"].items():
        assert digest(Path(path)) == expected, f"Input/code changed: {path}"
    if RESULT.exists():
        rows = read(RESULT)["rows"]
        assert [r["id"] for r in rows] == [s["id"] for s in plan["samples"][:len(rows)]]
    hotpotqa_interleaved_dataset.full_messages = terse_messages
    argv = list(plan["argv"])
    if not RESULT.exists():
        index = argv.index("--resume-from")
        del argv[index:index + 2]
    vanilla.main(argv)
    assert [r["id"] for r in read(RESULT)["rows"]] == [s["id"] for s in plan["samples"]]
    report(require_complete=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "report"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.action]()
