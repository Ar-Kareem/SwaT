#!/usr/bin/env python3
import json
from pathlib import Path

PY = "../env3.10/bin/python -u"
SENDER = "Qwen3-4B"
PAIRS = [
    ("qwen3_8b", "qwen3-8b", "qwen3-8b", 9),
    ("qwen2_7b", "qwen2-7b-instruct", "qwen2-7b-instruct", 7),
    ("llama3_8b", "llama3-8b-instruct", "llama3-8b-instruct", 8),
]
TEXT = "data/renderer_text/pg19_wikitext103_train_4k4k_20m.jsonl"
STAGE1_POOL = [TEXT, "qasper+nox:train", "quality:train", "hotpotqa:train", "musique:train"]
POOL = [TEXT, "qasper+nox:train", "quality+limit=2500:train", "hotpotqa+limit=2500:train", "musique+limit=2500:train",
        "arc-c:train", "hellaswag+limit=3000:train", "winogrande+limit=3000:train", "mmlu+limit=3000:auxiliary_train"]
RENDERER_PREFLIGHT_TOKENS = 9600
ARMS = {
    "renderer": "--arch v2 --d-model 1024 --n-layers {blocks} --n-heads 16 --ffn-dim 4096 --bound-cache-stats --bound-stats-exact-grad "
                f"--learning-rate 3e-6 --weight-decay 0.001 --preflight-max-receiver-tokens {RENDERER_PREFLIGHT_TOKENS}",
    "dense": "--arch dense-latent --headwise-hidden 2048 --learning-rate 1e-5",
}
ADAPTERS = {
    "renderer": "--arch v2 --d-model 1024 --n-layers {blocks} --n-heads 16 --ffn-dim 4096 --bound-cache-stats "
                f"--stat-calibration-batches 8 --bound-stats-exact-grad --preflight-max-receiver-tokens {RENDERER_PREFLIGHT_TOKENS}",
    "dense": "--arch dense-latent --headwise-hidden 2048",
}
STAGE1_GRAD_SKIP = {"renderer": 5.0, "dense": 0.0}
SEEDS = (44, 42, 43)
STAGE1_STEPS, STAGE2_STEPS, SAVE_EVERY = 5000, 16000, 8000
TRAIN_ROOT = "outputs/abl_many_seeds"
DEVICES = "--device cuda:0 --receiver-device cuda:0 --renderer-device cuda:0"
PREFLIGHT_ROOT = "outputs/sandbox_agent/abl_dense_redo_preflight_20261002"
ROOT = Path(__file__).resolve().parents[2] / "multiagent"


def traces_path(alias, clean=False):
    return f"data/phase2_traces/{alias}_gsm8k_train{'_clean' if clean else ''}.jsonl"


def stage2_preflight_cmd(pair, receiver, trace_alias, blocks, arm):
    pool = list(POOL)
    if (ROOT / traces_path(trace_alias, clean=True)).exists():
        pool.append(traces_path(trace_alias, clean=True))
    return [f"{PY} train/train_cache_renderer.py", f"--sender-model {SENDER} --receiver-model {receiver}",
            "--datasets " + " ".join(pool), "--exclude-answer-message --max-length 8192 --max-trace-length 4096",
            "--phase-schedule mixed --ce-batch-fraction 1.0 --ce-loss hard --causal-cross-attention --dense-batch-fraction 0",
            ARMS[arm].format(blocks=blocks), "--epochs 2 --limit-per-dataset 7500 --max-steps 24000 --lr-scheduler constant --warmup-steps 50 --seed 44",
            "--device cuda:0 --receiver-device cuda:0 --renderer-device cuda:0 --oom-preflight --preflight-only",
            f"--output-dir {PREFLIGHT_ROOT}/{pair}_{arm}"]


def run_dir(pair, arm, seed, stage):
    return f"{TRAIN_ROOT}/{pair}/{arm}/seed{seed}/stage{stage}"


def stage1_cmd(pair, receiver, blocks, arm, seed):
    return [f"{PY} train/train_cache_renderer.py", f"--sender-model {SENDER} --receiver-model {receiver}",
            "--datasets " + " ".join(STAGE1_POOL), "--exclude-answer-message",
            f"--limit-per-dataset 2500 --max-length 8192 --max-trace-length 4096 --batch-size 1 --epochs 1 --max-steps {STAGE1_STEPS}",
            "--learning-rate 3e-4 --lr-scheduler cosine --warmup-steps 300 --weight-decay 0.001 --max-grad-norm 1.0",
            f"--grad-skip-threshold {STAGE1_GRAD_SKIP[arm]}",
            ADAPTERS[arm].format(blocks=blocks), "--causal-cross-attention --phase-schedule kv-only --position-mode interp",
            f"{DEVICES} --report-every 30 --save-every -1 --seed {seed}", f"--output-dir {run_dir(pair, arm, seed, 1)}"]


def stage2_cmd(pair, receiver, trace_alias, blocks, arm, seed):
    init = f"{run_dir(pair, arm, seed, 1)}/exit-checkpoint-{STAGE1_STEPS}"
    wait = (f"for i in $(seq 1 720); do test -f {init}/renderer.pt && break; sleep 60; done && test -f {init}/renderer.pt "
            f"&& test -s {traces_path(trace_alias, clean=True)} && sleep 30 && ")
    return [wait + f"{PY} train/train_cache_renderer.py", f"--sender-model {SENDER} --receiver-model {receiver}",
            "--datasets " + " ".join(POOL + [traces_path(trace_alias, clean=True)]),
            "--exclude-answer-message --max-length 8192 --max-trace-length 4096 --batch-size 1 --epochs 2",
            "--phase-schedule mixed --ce-batch-fraction 1.0 --ce-loss hard --causal-cross-attention --position-mode interp --dense-batch-fraction 0",
            f"--limit-per-dataset 7500 --max-steps {STAGE2_STEPS} --learning-rate 3e-6 --lr-scheduler constant --warmup-steps 50 --weight-decay 0.001",
            ADAPTERS[arm].format(blocks=blocks), f"--init-weights-from {init}",
            f"{DEVICES} --report-every 30 --save-every {SAVE_EVERY} --seed {seed}", f"--output-dir {run_dir(pair, arm, seed, 2)}"]


GEN_EVALS = [("math-500:test", "math-500_n500", 1024), ("gsm8k:test", "gsm8k_n500", 512)]
EVAL_BATCH = 64


def gen_eval_cmd(pair, arm, seed, spec, stem, max_new_tokens):
    ckpt = f"{run_dir(pair, arm, seed, 2)}/checkpoint-16000"
    return [f"{PY} evaluation/renderer.py", f"--renderer {ckpt} --renderer-decode-prefix generation",
            f"--tasks {spec} --limit-per-dataset 500 --max-new-tokens {max_new_tokens} --max-length 2048 --seed 42",
            f"--device cuda:0 --batch-size {EVAL_BATCH}", f"--output {ckpt}/eval_{stem}.json"]


def race_eval_cmd(pair, arm, seed):
    ckpt = f"{run_dir(pair, arm, seed, 2)}/checkpoint-16000"
    return [f"{PY} evaluation/renderer.py", f"--renderer {ckpt}/renderer.pt --tasks race:test",
            "--max-length 8192 --max-new-tokens 64 --seed 42 --device cuda:0 --context-cache-max 4",
            f"--output {ckpt}/eval_race_test_renderer_1x.json"]


def main() -> None:
    raw, clean = traces_path("qwen2-7b-instruct"), traces_path("qwen2-7b-instruct", clean=True)
    traces = ["# abl_dense redo: Qwen2-7B-Instruct GSM8K self-guided traces (scripts/abl_closed_form/make_abl_dense_redo_queue.py)",
              {"cmd": f"{PY} create_phase2_dataset.py --receiver-model qwen2-7b-instruct --device cuda:0 --batch-size 32 "
                      f"--max-new-tokens 512 --max-length 2048 --tasks gsm8k:train --output {raw}", "gpus": 2},
              {"cmd": f"test $(wc -l < {raw}) -ge 7400 && {PY} ../scripts/download/filter_phase2_traces.py --input {raw} --output {clean} "
                      f"&& test $(wc -l < {clean}) -ge 6500", "gpus": 1}]
    preflight = ["# abl_dense redo: one-GPU stage-2 OOM preflights, 3 pairs x 2 arms (scripts/abl_closed_form/make_abl_dense_redo_queue.py)"]
    for arm in ARMS:
        for pair, receiver, trace_alias, blocks in PAIRS:
            preflight.append({"cmd": stage2_preflight_cmd(pair, receiver, trace_alias, blocks, arm), "gpus": 1})
    train = [f"# abl_many_seeds: dense latent vs renderer, 3 pairs x 2 arms x seeds {SEEDS}, one GPU per job (scripts/abl_closed_form/make_abl_dense_redo_queue.py)"]
    for seed in SEEDS:
        train.append(f"# ---- seed {seed}: stage 1")
        for pair, receiver, trace_alias, blocks in PAIRS:
            for arm in ARMS:
                train.append({"cmd": stage1_cmd(pair, receiver, blocks, arm, seed), "gpus": 1})
        train.append(f"# ---- seed {seed}: stage 2")
        for pair, receiver, trace_alias, blocks in PAIRS:
            for arm in ARMS:
                train.append({"cmd": stage2_cmd(pair, receiver, trace_alias, blocks, arm, seed), "gpus": 1})
    gen43 = ["# abl_many_seeds seed 43: GSM8K + MATH-500 at checkpoint-16000, batch 64 (scripts/abl_closed_form/make_abl_dense_redo_queue.py)"]
    for spec, stem, mnt in GEN_EVALS:
        for pair, receiver, trace_alias, blocks in PAIRS:
            for arm in ARMS:
                gen43.append({"cmd": gen_eval_cmd(pair, arm, 43, spec, stem, mnt), "gpus": 1})
    race43 = ["# abl_many_seeds seed 43: RACE at checkpoint-16000, serial like seeds 42/44 (scripts/abl_closed_form/make_abl_dense_redo_queue.py)"]
    for pair, receiver, trace_alias, blocks in PAIRS:
        for arm in ARMS:
            race43.append({"cmd": race_eval_cmd(pair, arm, 43), "gpus": 1})
    for name, jobs in (("abl_many_seeds_eval_seed43_race", race43), ("abl_many_seeds_eval_seed43_gen", gen43), ("abl_dense_redo_traces", traces), ("abl_dense_redo_preflight", preflight), ("abl_many_seeds_train", train)):
        path = ROOT / "gpu_queue_commands" / f"gpu_queue_commands_{name}.json"
        path.write_text(json.dumps(jobs, indent=1) + "\n")
        print(f"wrote {path} with {sum(1 for j in jobs if not isinstance(j, str))} jobs")


if __name__ == "__main__":
    main()
