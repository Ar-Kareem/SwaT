#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagent.paths import DATASETS_ROOT

ROOT = Path(__file__).resolve().parents[2]
PY = "../env3.10/bin/python"
STAGE1_INIT = "outputs/renderer14A_replica8_seed44_4b_to_llama8b_stage1_kv/exit-checkpoint-5000"
STEPS_S2 = 12000
STEPS_S1 = 5000

COMMON = [
    "--sender-model Qwen3-4B --receiver-model llama3-8b-instruct",
    "--exclude-answer-message --max-length 8192 --max-trace-length 2048 --batch-size 1",
    "--arch v2 --d-model 1024 --n-layers 8 --n-heads 16 --ffn-dim 4096",
    "--bound-cache-stats --bound-stats-exact-grad --causal-cross-attention",
    "--device cuda:0 --receiver-device cuda:1 --renderer-device cuda:0",
    "--weight-decay 0.001 --max-grad-norm 1.0 --report-every 30 --loss-window 100",
]

S1_DATA = "--datasets data/renderer_text/pg19_wikitext103_train_4k4k_20m.jsonl qasper+nox:train quality:train hotpotqa:train musique:train --limit-per-dataset 2500"
S1_RECIPE = (
    f"--epochs 1 --max-steps {STEPS_S1} --learning-rate 3e-4 --lr-scheduler cosine --warmup-steps 300 "
    "--grad-skip-threshold 5.0 --phase-schedule kv-only --dense-batch-fraction 0 "
    f"--save-every {STEPS_S1} --downstream-eval-every 1000 --downstream-eval-datasets qasper:validation quality:boosted "
    "--downstream-eval-samples 64 --downstream-eval-strides 1 4 --downstream-generate-samples 64 --downstream-max-new-tokens 32"
)

S2_DATA = (
    "--datasets data/renderer_text/pg19_wikitext103_train_4k4k_20m.jsonl qasper+nox:train quality+limit=2500:train "
    "hotpotqa+limit=2500:train musique+limit=2500:train arc-c:train hellaswag+limit=3000:train winogrande+limit=3000:train "
    "mmlu+limit=3000:auxiliary_train data/phase2_traces/llama3-8b-instruct_gsm8k_train_clean.jsonl --limit-per-dataset 7500"
)
S2_RECIPE = (
    f"--epochs 2 --max-steps {STEPS_S2} --learning-rate 3e-6 --lr-scheduler constant --warmup-steps 50 "
    "--phase-schedule mixed --ce-batch-fraction 1.0 --ce-loss hard "
    "--dense-datasets data/renderer_text/pg19_train_4k4k_20m.jsonl data/renderer_text/wikitext103_train_4k4k_10m.jsonl "
    "--dense-max-length 3072 --dense-w-choices 0 1 4 16 --dense-loss hard "
    "--save-every 6000 --downstream-eval-every 1000 --downstream-eval-datasets qasper:validation quality:boosted "
    "--downstream-eval-samples 64 --downstream-eval-strides 1 4 --downstream-generate-samples 64 --downstream-max-new-tokens 32"
)

LONG_1X = [
    ("race:test", "eval_race_test_renderer_1x.json", None),
    ("boolq:validation", "eval_boolq_validation_renderer_1x.json", None),
    ("squad:validation", "eval_squad_validation20pct_renderer_1x.json", 2114),
    ("quality:dev", "eval_quality_dev_renderer_1x.json", None),
    ("longhealth+ppc=1:test", "eval_longhealth_renderer_1x.json", None),
    ("hotpotqa:validation", "eval_hotpotqa_val400_renderer_1x.json", 400),
    ("musique:validation", "eval_musique_val400_renderer_1x.json", 400),
    ("quality:boosted", "eval_quality_boosted_renderer_1x.json", None),
    ("qasper:test", "eval_qasper_test100_renderer_1x.json", 100),
]
CLOSED_BOOK = [
    ("gsm8k:test", "gsm8k", 512),
    ("mmlu:test", "mmlu", 64),
    ("medqa:test", "medqa", 64),
    ("arc-c:test", "arc-c", 64),
    ("hellaswag:validation", "hellaswag", 64),
    ("winogrande:validation", "winogrande", 64),
    ("openbookqa:test", "openbookqa", 64),
]


def train_job(name: str, stage: str, extra: str, init: str | None, seed: int, after: list[str] | None = None) -> dict:
    parts = [f"{PY} train/train_cache_renderer.py", *COMMON]
    parts.append(S1_DATA if stage == "s1" else S2_DATA)
    parts.append(S1_RECIPE if stage == "s1" else S2_RECIPE)
    if init:
        parts.append(f"--init-weights-from {init}")
    parts.append(f"--seed {seed} --output-dir outputs/{name}")
    if extra:
        parts.append(extra)
    job = {"cmd": parts, "gpus": 2, "name": name}
    if after:
        job["after"] = list(after)
    return job


def eval_jobs(name: str, steps: int) -> list:
    ck = f"outputs/{name}/exit-checkpoint-{steps}"
    wait = ""
    jobs = []
    for task, stem, limit in LONG_1X:
        lim = f" --limit-per-dataset {limit}" if limit else ""
        jobs.append(
            f"{wait}{PY} evaluation/renderer.py --renderer {ck}/renderer.pt --tasks {task}{lim} --max-length 8192 "
            f"--max-new-tokens 64 --seed 42 --device cuda:0 --context-cache-max 4 --output outputs/{name}/{stem}"
        )
    for task, stem, new_tokens in CLOSED_BOOK:
        jobs.append(
            f"{wait}{PY} evaluation/renderer.py --renderer {ck} --renderer-decode-prefix generation --tasks {task} "
            f"--limit-per-dataset 500 --max-new-tokens {new_tokens} --max-length 2048 --seed 42 --device cuda:0 "
            f"--output outputs/{name}/eval_closedbook_{stem}_n500.json"
        )
    jobs.append(
        f"{wait}{PY} evaluation/eval_vnll.py --renderer {ck} --sender-device cuda:0 --receiver-device cuda:0 "
        f"--output outputs/{name}/eval_vnll.json"
    )
    return [{"cmd": cmd, "gpus": 1, "after": [name]} for cmd in jobs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-optional", action="store_true", help="Drop denseonly and scaledpos.")
    parser.add_argument("--headwise-only", action="store_true", help="Emit only the headwise chain (preflight, stage-1, stage-2, evals) to --output (default ..._ablation16b.json).")
    parser.add_argument("--no-musique", action="store_true", help="MuSiQue snapshot absent on this host: drop it from the train pools and the eval suite (all runs alike).")
    parser.add_argument("--output", default=str(ROOT / "multiagent" / "gpu_queue_commands_ablation16.json"))
    args = parser.parse_args()
    if args.headwise_only and args.output.endswith("gpu_queue_commands_ablation16.json"):
        args.output = args.output.replace("ablation16.json", "ablation16b.json")
    global S1_DATA, S2_DATA, LONG_1X
    if args.no_musique:
        S1_DATA = S1_DATA.replace(" musique:train", "")
        S2_DATA = S2_DATA.replace("musique+limit=2500:train ", "")
        LONG_1X = [row for row in LONG_1X if not row[0].startswith("musique")]

    P = "abl16"
    s1_pointwise = f"{P}P_pointwise_seed44_4b_to_llama8b_stage1_kv"
    s1_naive = f"{P}N_naivepos_seed44_4b_to_llama8b_stage1_kv"
    s1_scaled = f"{P}S_scaledpos_seed44_4b_to_llama8b_stage1_kv"
    control = f"{P}C_control_seed44_4b_to_llama8b_stage2_dense03"
    control45 = f"{P}C_control_seed45_4b_to_llama8b_stage2_dense03"
    pointwise = f"{P}P_pointwise_seed44_4b_to_llama8b_stage2_dense03"
    naive = f"{P}N_naivepos_seed44_4b_to_llama8b_stage2_dense03"
    nodense = f"{P}D0_nodense_seed44_4b_to_llama8b_stage2"
    denseonly = f"{P}D1_denseonly_seed44_4b_to_llama8b_stage2_dense10"
    scaled = f"{P}S_scaledpos_seed44_4b_to_llama8b_stage2_dense03"

    pw_extra = "--arch pointwise --ffn-dim 8192"
    if args.headwise_only:
        s1_headwise = f"{P}H_headwise_seed44_4b_to_llama8b_stage1_kv"
        headwise = f"{P}H_headwise_seed44_4b_to_llama8b_stage2_dense03"
        hw_extra = "--arch headwise --headwise-hidden 2048"
        preflight = train_job("ZTEMP_preflight/headwise", "s2", f"--dense-batch-fraction 0.3 {hw_extra} --preflight-only", None, 44)
        jobs = [
            "# ablation16b: headwise (C2C-style per-head MLP) arm. Generated by scripts/ablations/make_ablation_queue.py --headwise-only",
            preflight,
            train_job(s1_headwise, "s1", hw_extra, None, 44),
            train_job(headwise, "s2", f"--dense-batch-fraction 0.3 {hw_extra}", f"outputs/{s1_headwise}/exit-checkpoint-{STEPS_S1}", 44, after=[s1_headwise]),
            *eval_jobs(headwise, STEPS_S2),
        ]
        Path(args.output).write_text(json.dumps(jobs, indent=1) + "\n")
        print(f"wrote {args.output}: {len(jobs)} entries (headwise only)")
        return
    jobs: list = [
        "# ablation16: Qwen3-4B -> Llama-3-8B, one delta per run. Generated by scripts/ablations/make_ablation_queue.py",
        *([] if args.no_musique else [f"test -s {DATASETS_ROOT}/snapshots/voidful__MuSiQue/default/train/dataset_info.json"]),
        f"test -s {STAGE1_INIT}/renderer.pt",
        train_job(s1_pointwise, "s1", pw_extra, None, 44),
        train_job(s1_naive, "s1", "--position-mode naive", None, 44),
        train_job(control, "s2", "--dense-batch-fraction 0.3", STAGE1_INIT, 44),
        train_job(control45, "s2", "--dense-batch-fraction 0.3", STAGE1_INIT, 45),
        *eval_jobs(control, STEPS_S2),
        *eval_jobs(control45, STEPS_S2),
        train_job(pointwise, "s2", f"--dense-batch-fraction 0.3 {pw_extra}", f"outputs/{s1_pointwise}/exit-checkpoint-{STEPS_S1}", 44, after=[s1_pointwise]),
        train_job(naive, "s2", "--dense-batch-fraction 0.3 --position-mode naive", f"outputs/{s1_naive}/exit-checkpoint-{STEPS_S1}", 44, after=[s1_naive]),
        *eval_jobs(pointwise, STEPS_S2),
        *eval_jobs(naive, STEPS_S2),
        train_job(nodense, "s2", "--dense-batch-fraction 0", STAGE1_INIT, 44),
    ]
    if args.no_optional:
        jobs += eval_jobs(nodense, STEPS_S2)
    else:
        jobs += [
            train_job(denseonly, "s2", "--dense-batch-fraction 1.0", STAGE1_INIT, 44),
            *eval_jobs(nodense, STEPS_S2),
            *eval_jobs(denseonly, STEPS_S2),
            train_job(s1_scaled, "s1", "--position-mode scaled", None, 44),
            train_job(scaled, "s2", "--dense-batch-fraction 0.3 --position-mode scaled", f"outputs/{s1_scaled}/exit-checkpoint-{STEPS_S1}", 44, after=[s1_scaled]),
            *eval_jobs(scaled, STEPS_S2),
        ]
    Path(args.output).write_text(json.dumps(jobs, indent=1) + "\n")
    n_train = sum(1 for j in jobs if isinstance(j, dict) and j["gpus"] == 2)
    n_eval = sum(1 for j in jobs if isinstance(j, dict) and j["gpus"] == 1)
    print(f"wrote {args.output}: {n_train} training jobs (2 GPUs), {n_eval} eval jobs (1 GPU), {len(jobs)} entries")


if __name__ == "__main__":
    main()
