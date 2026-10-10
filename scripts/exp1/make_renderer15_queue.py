#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

PY = "../env3.10/bin/python -u"
OLD_TEXT = "data/renderer_text/pg19_wikitext103_train_4k4k_20m.jsonl"
OLD_DENSE = ["data/renderer_text/pg19_train_4k4k_20m.jsonl", "data/renderer_text/wikitext103_train_4k4k_10m.jsonl"]
POOL_A = [OLD_TEXT, "qasper+nox:train", "quality:train", "hotpotqa:train", "musique:train"]


def pool_e(traces):
    return [OLD_TEXT, "qasper+nox:train", "quality+limit=2500:train", "hotpotqa+limit=2500:train", "musique+limit=2500:train",
            "arc-c:train", "hellaswag+limit=3000:train", "winogrande+limit=3000:train", "mmlu+limit=3000:auxiliary_train", traces]


RUNS = [
    ("15A", "llama3-8b-instruct", "qwen3-14b", "qwen3-14b", "full"),
    ("15B", "Qwen3-4B", "gemma-2-9b-it", "gemma-2-9b-it", "gemma"),
    ("15C", "gemma-2-9b-it", "ministral-3-14b-instruct", "ministral-3-14b-instruct", "full"),
    ("15D", "ministral-3-8b-instruct", "llama3.2-11b-vision-instruct", "llama3.2-11b-vision-instruct", "full"),
    ("15E", "gemma-3-4b-it", "granite-3.3-8b-instruct", "granite-3.3-8b-instruct", "full"),
    ("15F", "Qwen3-4B", "qwen3-8b", "qwen3-8b", "qwen8b"),
    ("15H", "gemma-3-12b-it", "qwen3-4b", "qwen3-4b", "qwen4b"),
    ("15I", "qwen3-32b", "llama3-8b-instruct", "llama3-8b-instruct", "bigsender"),
]
MEM = {"full": "--max-trace-length 4096 --dense-max-length 4096",
       "qwen8b": "--max-trace-length 4096 --dense-max-length 4096",
       "qwen4b": "--max-trace-length 4096 --dense-max-length 4096",
       "bigsender": "--max-trace-length 4096 --dense-max-length 4096",
       "gemma": "--max-trace-length 2048 --dense-max-length 2560 --empty-cache-every 25"}
LAYERS = {"full": 8, "gemma": 7, "qwen8b": 9, "qwen4b": 9, "bigsender": 8}
RENDERER_DEV = {m: "cuda:0" for m in LAYERS} | {"bigsender": "cuda:1"}
STEPS, SAVE_EVERY, LIMIT = 24000, 4000, 7500

EVALS_1X = [
    ("race_test_renderer_1x", "race:test", None), ("boolq_validation_renderer_1x", "boolq:validation", None),
    ("squad_validation20pct_renderer_1x", "squad:validation", 2114), ("quality_dev_renderer_1x", "quality:dev", None),
    ("musique_val400_renderer_1x", "musique:validation", 400), ("hotpotqa_val400_renderer_1x", "hotpotqa:validation", 400),
    ("longhealth_renderer_1x", "longhealth+ppc=1:test", None), ("quality_boosted_renderer_1x", "quality:boosted", None),
    ("qasper_test100_renderer_1x", "qasper:test", 100),
]
CLOSED_BOOK = [("gsm8k:test", 512), ("arc-c:test", 64), ("hellaswag:validation", 64), ("winogrande:validation", 64), ("mmlu:test", 64),
               ("medqa:test", 64), ("openbookqa:test", 64), ("math-500:test", 1024)]


def out_dirs(name, sender, receiver):
    short = {"llama3-8b-instruct": "llama8b", "Qwen3-4B": "4b", "gemma-2-9b-it": "gemma2_9b", "qwen3-14b": "qwen14b", "ministral-3-14b-instruct": "ministral3_14b",
             "ministral-3-8b-instruct": "ministral3_8b", "llama3.2-11b-vision-instruct": "llama32_11bv",
             "gemma-3-4b-it": "gemma3_4b", "granite-3.3-8b-instruct": "granite33_8b", "qwen3-8b": "qwen8b",
             "gemma-3-12b-it": "gemma3_12b", "qwen3-4b": "qwen3_4b", "qwen3-32b": "qwen3_32b"}
    tag = f"{short[sender]}_to_{short[receiver]}"
    return (f"outputs/renderer{name}_closedbook_seed44_{tag}_stage1_kv", f"outputs/renderer{name}_closedbook_seed44_{tag}_stage2_dense03")


def stage1_cmd(sender, receiver, out, mem):
    return [f"{PY} train/train_cache_renderer.py", f"--sender-model {sender} --receiver-model {receiver}",
            "--datasets " + " ".join(POOL_A), "--exclude-answer-message", f"--output-dir {out}",
            "--limit-per-dataset 2500 --max-length 8192 --max-trace-length 4096 --batch-size 1 --epochs 1 --max-steps 5000",
            "--learning-rate 3e-4 --lr-scheduler cosine --warmup-steps 300 --weight-decay 0.001 --max-grad-norm 1.0 --grad-skip-threshold 5.0",
            f"--arch v2 --d-model 1024 --n-layers {LAYERS[mem]} --n-heads 16 --ffn-dim 4096 --bound-cache-stats --stat-calibration-batches 8 --bound-stats-exact-grad",
            "--causal-cross-attention --phase-schedule kv-only --position-mode interp",
            "--downstream-eval-every 1000 --downstream-eval-datasets qasper:validation quality:boosted --downstream-eval-samples 64 --downstream-eval-strides 1 4 --downstream-generate-samples 64 --downstream-max-new-tokens 32",
            f"--device cuda:0 --receiver-device cuda:1 --renderer-device {RENDERER_DEV[mem]} --report-every 30 --save-every 5000 --seed 44"]


def stage2_cmd(sender, receiver, stage1, out, traces, mem):
    return [f"{PY} train/train_cache_renderer.py", f"--sender-model {sender} --receiver-model {receiver}",
            "--datasets " + " ".join(pool_e(traces)), f"--exclude-answer-message --max-length 8192 {MEM[mem]} --epochs 2",
            "--bound-cache-stats --bound-stats-exact-grad --phase-schedule mixed --ce-batch-fraction 1.0 --ce-loss hard --causal-cross-attention",
            "--downstream-eval-every 500 --downstream-eval-datasets qasper:validation quality:boosted --downstream-eval-strides 1 4",
            f"--save-every {SAVE_EVERY} --seed 44 --device cuda:0 --receiver-device cuda:1 --renderer-device {RENDERER_DEV[mem]}",
            f"--limit-per-dataset {LIMIT} --max-steps {STEPS} --learning-rate 3e-6 --lr-scheduler constant --warmup-steps 50",
            f"--init-weights-from {stage1}/exit-checkpoint-5000",
            "--dense-batch-fraction 0.3 --dense-w-choices 0 1 4 16", "--dense-datasets " + " ".join(OLD_DENSE),
            f"--arch v2 --d-model 1024 --n-layers {LAYERS[mem]} --n-heads 16 --ffn-dim 4096",
            f"--output-dir {out}"]


def vanilla_cb(receiver, alias, spec, mnt):
    task = spec.split(":")[0]
    return [f"{PY} evaluation/vanilla.py --model {receiver}", f"--tasks {spec} --limit-per-dataset 500 --max-new-tokens {mnt} --max-length 2048 --seed 42 --device cuda:0",
            f"--output outputs/kvxfer_bench/{task}_n500_vanilla_{alias}.json"]


def renderer_cb(out, name, spec, mnt):
    task = spec.split(":")[0]
    return [f"{PY} evaluation/renderer.py --renderer {out}/exit-checkpoint-{STEPS} --renderer-decode-prefix generation",
            f"--tasks {spec} --limit-per-dataset 500 --max-new-tokens {mnt} --max-length 2048 --seed 42 --device cuda:0",
            f"--output outputs/kvxfer_bench/{task}_n500_{name}.json"]


def vanilla_1x(receiver, out, stem, task, limit):
    parts = [f"{PY} evaluation/vanilla.py --model {receiver}", f"--tasks {task}"]
    if limit:
        parts.append(f"--limit-per-dataset {limit}")
    parts += ["--max-length 8192 --max-new-tokens 64 --seed 42 --device cuda:0", f"--output {out}/eval_{stem.replace('_renderer_', '_vanilla_')}.json"]
    return parts


def renderer_1x(out, stem, task, limit):
    parts = [f"{PY} evaluation/renderer.py", f"--renderer {out}/exit-checkpoint-{STEPS}/renderer.pt", f"--tasks {task}"]
    if limit:
        parts.append(f"--limit-per-dataset {limit}")
    parts += ["--max-length 8192 --max-new-tokens 64 --seed 42 --device cuda:0 --context-cache-max 4", f"--output {out}/eval_{stem}.json"]
    return parts


def job_minutes_from_logs(logs):
    import re
    root = Path(__file__).resolve().parents[2] / "multiagent"
    seen = {}
    for log in logs:
        text = (root / log).read_text(errors="ignore")
        starts = {m.group(1): m.group(2) for m in re.finditer(r"(job\d+) START gpus=\[[0-9, ]+\] log=\S+ :: (.*)", text)}
        for m in re.finditer(r"(job\d+) DONE after ([0-9.]+) min", text):
            cmd = starts.get(m.group(1), "")
            task = re.search(r"--tasks (\S+)", cmd)
            if task:
                key = (task.group(1), "vanilla" if "--model " in cmd else "renderer")
                seen[key] = max(seen.get(key, 0.0), float(m.group(2)))
    return seen


def sort_eval_jobs(jobs, minutes):
    import re
    def est(job):
        cmd = " ".join(job["cmd"]) if isinstance(job["cmd"], list) else job["cmd"]
        task = re.search(r"--tasks (\S+)", cmd)
        if not task or not any(entry in cmd for entry in ("eval_unaware_comm", "evaluation/renderer.py", "evaluation/vanilla.py")):
            return -1.0
        return minutes.get((task.group(1), "vanilla" if "--model " in cmd else "renderer"), 60.0)
    out, block = [], []
    def flush():
        block.sort(key=est); out.extend(block); block.clear()
    for j in jobs:
        if isinstance(j, str):
            flush(); out.append(j)
        else:
            block.append(j)
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="15A,15B", help="comma list of run names to emit")
    ap.add_argument("--no-evals", action="store_true", help="emit only trace filter + stage 1 + stage 2")
    ap.add_argument("--no-train", action="store_true", help="emit only the eval block (stage 2 exit checkpoint must exist)")
    ap.add_argument("--skip-stage1", action="store_true", help="emit stage 2 without the trace filter and stage 1 (stage 1 exit checkpoint must exist)")
    ap.add_argument("--wait-for-log", default=None, help="nohup log of the trace queue; the filter job waits until it says 'queue finished' (max 4h)")
    ap.add_argument("--out", default=None, help="queue file name under multiagent/gpu_queue_commands/ (default gpu_queue_commands_renderer15.json)")
    ap.add_argument("--sort-by-logs", nargs="*", default=None, metavar="NOHUP_LOG",
                    help="reorder each run's eval jobs fastest-first using 'DONE after N min' times of matching (task, vanilla/renderer) jobs "
                         "in these earlier queue logs (relative to multiagent/); unknown tasks sort as 60 min")
    args = ap.parse_args()
    runs = [r for r in RUNS if r[0] in args.runs.split(",")]
    jobs = [f"# renderer15 ({args.runs}{', train only' if args.no_evals else ''}): 14E recipe on new cross-family pairs (scripts/exp1/make_renderer15_queue.py). Per run: filter traces -> stage 1 -> stage 2" + ("" if args.no_evals else " -> vanilla baselines + renderer evals")]
    for name, sender, receiver, alias, mem in runs:
        raw = f"data/phase2_traces/{alias}_gsm8k_train.jsonl"
        clean = f"data/phase2_traces/{alias}_gsm8k_train_clean.jsonl"
        stage1, stage2 = out_dirs(name, sender, receiver)
        jobs.append(f"# ---- run {name}: {sender} -> {receiver}")
        if not args.no_train:
            if not args.skip_stage1:
                wait = f"for i in $(seq 1 240); do grep -q 'queue finished' {args.wait_for_log} && break; sleep 60; done && " if args.wait_for_log else ""
                filter_job = {"cmd": wait + f"test $(wc -l < {raw}) -ge 7400 && {PY} ../scripts/download/filter_phase2_traces.py --input {raw} --output {clean} && test $(wc -l < {clean}) -ge 6500", "gpus": 1}
                if args.wait_for_log:
                    jobs.append({"cmd": stage1_cmd(sender, receiver, stage1, mem), "gpus": 2})
                    jobs.append(filter_job)
                else:
                    jobs.append(filter_job)
                    jobs.append({"cmd": stage1_cmd(sender, receiver, stage1, mem), "gpus": 2})
            else:
                jobs.append({"cmd": f"test -f {stage1}/exit-checkpoint-5000/renderer.pt && test -s {clean}", "gpus": 1})
            jobs.append({"cmd": stage2_cmd(sender, receiver, stage1, stage2, clean, mem), "gpus": 2})
        if args.no_evals:
            continue
        jobs.append({"cmd": f"test -f {stage2}/exit-checkpoint-{STEPS}/renderer.pt", "gpus": 1})
        jobs.append({"cmd": renderer_cb(stage2, name, "gsm8k:test", 512), "gpus": 1})
        jobs.append({"cmd": vanilla_cb(receiver, alias, "gsm8k:test", 512), "gpus": 1})
        for stem, task, limit in EVALS_1X:
            jobs.append({"cmd": renderer_1x(stage2, stem, task, limit), "gpus": 1})
            jobs.append({"cmd": vanilla_1x(receiver, stage2, stem, task, limit), "gpus": 1})
        for spec, mnt in CLOSED_BOOK[1:]:
            jobs.append({"cmd": renderer_cb(stage2, name, spec, mnt), "gpus": 1})
            jobs.append({"cmd": vanilla_cb(receiver, alias, spec, mnt), "gpus": 1})
    if args.sort_by_logs:
        jobs = sort_eval_jobs(jobs, job_minutes_from_logs(args.sort_by_logs))
    path = Path(__file__).resolve().parents[2] / "multiagent" / "gpu_queue_commands" / (args.out or "gpu_queue_commands_renderer15.json")
    path.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"wrote {path} with {sum(1 for j in jobs if not isinstance(j, str))} jobs")


if __name__ == "__main__":
    main()
