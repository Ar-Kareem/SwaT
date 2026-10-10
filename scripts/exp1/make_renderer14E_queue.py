#!/usr/bin/env python3
import json
from pathlib import Path

PY = "../env3.10/bin/python -u"
REF = "outputs/renderer8_densetext_qa_seed43_4b_to_llama8b_stage2_dense03"
PREV = "outputs/renderer14A_replica8_seed44_4b_to_llama8b_stage2_dense03"
STAGE1 = "outputs/renderer14A_replica8_seed44_4b_to_llama8b_stage1_kv"
OUT = "outputs/renderer14E_closedbook_seed44_4b_to_llama8b_stage2_dense03"
STEPS = 24000
SAVE_EVERY = 4000
LIMIT = 7500
OLD_TEXT = "data/renderer_text/pg19_wikitext103_train_4k4k_20m.jsonl"
OLD_DENSE = ["data/renderer_text/pg19_train_4k4k_20m.jsonl", "data/renderer_text/wikitext103_train_4k4k_10m.jsonl"]
GSM8K_TRACES = "data/phase2_traces/llama3-8b-instruct_gsm8k_train_clean.jsonl"
POOL_E = [OLD_TEXT, "qasper+nox:train", "quality+limit=2500:train", "hotpotqa+limit=2500:train", "musique+limit=2500:train",
          "arc-c:train", "hellaswag+limit=3000:train", "winogrande+limit=3000:train", "mmlu+limit=3000:auxiliary_train",
          GSM8K_TRACES]
QT = "outputs/xfam21_qq_query_transfer_4b_to_llama8b_qpair_w7_mlp/gold-checkpoint-4000/query_transfer.pt"
BUDGET = "model/head_budget_optimization/head_budgets/Llama-3.1-8B-Instruct/optimized_agnostic.json"

EVALS = [
    ("qasper_test100_renderer_am10", "qasper:test", 100, 0.10),
    ("qasper_test100_renderer_am05", "qasper:test", 100, 0.05),
    ("qasper_test100_renderer_am02", "qasper:test", 100, 0.02),
    ("race_test_renderer_1x", "race:test", None, None),
    ("boolq_validation_renderer_1x", "boolq:validation", None, None),
    ("squad_validation20pct_renderer_1x", "squad:validation", 2114, None),
    ("longhealth_renderer_am10", "longhealth+ppc=1:test", None, 0.10),
    ("longhealth_renderer_am05", "longhealth+ppc=1:test", None, 0.05),
    ("longhealth_renderer_am01", "longhealth+ppc=1:test", None, 0.01),
    ("longhealth_renderer_am02", "longhealth+ppc=1:test", None, 0.02),
    ("quality_dev_renderer_1x", "quality:dev", None, None),
    ("musique_val400_renderer_1x", "musique:validation", 400, None),
    ("hotpotqa_val400_renderer_1x", "hotpotqa:validation", 400, None),
    ("longhealth_renderer_1x", "longhealth+ppc=1:test", None, None),
    ("quality_boosted_renderer_1x", "quality:boosted", None, None),
    ("qasper_test100_renderer_1x", "qasper:test", 100, None),
]
CLOSED_BOOK = [("arc-c:test", 64), ("hellaswag:validation", 64), ("winogrande:validation", 64), ("mmlu:test", 64), ("gsm8k:test", 512)]


def stage2_cmd():
    return [f"{PY} train/train_cache_renderer.py", "--sender-model Qwen3-4B --receiver-model llama3-8b-instruct",
            "--datasets " + " ".join(POOL_E), "--exclude-answer-message --max-length 8192 --max-trace-length 4096 --epochs 2",
            "--bound-cache-stats --bound-stats-exact-grad --phase-schedule mixed --ce-batch-fraction 1.0 --ce-loss hard --causal-cross-attention",
            "--downstream-eval-every 500 --downstream-eval-datasets qasper:validation quality:boosted --downstream-eval-strides 1 4",
            f"--save-every {SAVE_EVERY} --seed 44 --device cuda:0 --receiver-device cuda:1",
            f"--limit-per-dataset {LIMIT} --max-steps {STEPS} --learning-rate 3e-6 --lr-scheduler constant --warmup-steps 50",
            f"--init-weights-from {STAGE1}/exit-checkpoint-5000",
            "--dense-batch-fraction 0.3 --dense-w-choices 0 1 4 16 --dense-max-length 4096", "--dense-datasets " + " ".join(OLD_DENSE),
            f"--output-dir {OUT}"]


def eval_cmd(stem, task, limit, ratio):
    entry = "renderer.py" if ratio is None else "renderer_am.py"
    parts = [f"{PY} evaluation/{entry}", f"--renderer {OUT}/exit-checkpoint-{STEPS}/renderer.pt", f"--tasks {task}"]
    if limit:
        parts.append(f"--limit-per-dataset {limit}")
    parts.append("--max-length 8192 --max-new-tokens 64 --seed 42 --device cuda:0")
    if ratio is None:
        parts.append("--context-cache-max 4")
    else:
        parts += [f"--am-target-ratio {ratio:.2f}", f"--am-query-transfer {QT}", "--am-head-budget-mode precomputed",
                  f"--am-head-budget-path {BUDGET}", "--am-query-samples 10000 --am-nnls-iters 2 --am-score-method rms --am-chunk-size 8096 --context-cache-max 1"]
    parts.append(f"--output {OUT}/eval_{stem}.json")
    return parts


def closed_book_cmd(spec, max_new_tokens):
    task = spec.split(":")[0]
    return [f"{PY} evaluation/renderer.py", f"--renderer {OUT}/exit-checkpoint-{STEPS} --renderer-decode-prefix generation",
            f"--tasks {spec} --limit-per-dataset 500 --max-new-tokens {max_new_tokens} --max-length 2048 --seed 42 --device cuda:0",
            f"--output outputs/kvxfer_bench/{task}_n500_14E.json"]


def main() -> None:
    jobs = [f"# renderer14 run E: run-A recipe + cache-only MCQ letters + Llama GSM8K traces, {STEPS} steps -> evals -> closed-book -> gate (scripts/exp1/make_renderer14E_queue.py)"]
    jobs.append({"cmd": "test -s " + GSM8K_TRACES + " || (echo 'clean GSM8K trace file missing' && exit 1)", "gpus": 1})
    jobs.append({"cmd": stage2_cmd(), "gpus": 2})
    jobs.append({"cmd": closed_book_cmd("gsm8k:test", 512), "gpus": 1})
    for stem, task, limit, ratio in EVALS:
        jobs.append({"cmd": eval_cmd(stem, task, limit, ratio), "gpus": 1})
    for spec, mnt in CLOSED_BOOK:
        if spec != "gsm8k:test":
            jobs.append({"cmd": closed_book_cmd(spec, mnt), "gpus": 1})
    jobs.append({"cmd": [f"{PY} ../scripts/exp1/renderer_chain_gate.py", f"--run {OUT}", f"--reference {REF}", f"--expected-steps {STEPS}", f"--previous {PREV}"], "gpus": 1})
    path = Path(__file__).resolve().parents[2] / "multiagent" / "gpu_queue_commands" / "gpu_queue_commands_renderer14E.json"
    path.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"wrote {path} with {sum(1 for j in jobs if not isinstance(j, str))} jobs")


if __name__ == "__main__":
    main()
