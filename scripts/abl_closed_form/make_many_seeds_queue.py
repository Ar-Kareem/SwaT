import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/exp1"))
import make_abl_many_seeds_eval_queue as m
import make_renderer15_queue as r15
from multiagent.paths import MODELS_ROOT, SCRATCH_ROOT

ROOT = REPO / "multiagent"
BASE = "outputs/abl_many_seeds"
FEATURES = "/tmp/multikv_abl_many_seeds_closed_form"
MAPPERS = SCRATCH_ROOT / "abl_many_seeds_closed_form"
SEEDS = [42, 43, 44]
SENDER = MODELS_ROOT / "qwen/Qwen3-4B"
PAIRS = {
    "qwen3_8b": (MODELS_ROOT / "qwen/Qwen3-8B", 8),
    "qwen2_7b": (MODELS_ROOT / "qwen/Qwen2-7B-Instruct", 12),
    "llama3_8b": (MODELS_ROOT / "llama3/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/"
                  "c4a54320a52ed5f88b7a2f84496903ea4ff07b45", 12),
}
SLOW = {"gsm8k:test", "math-500:test"}
EVAL = "../env3.10/bin/python -u ../scripts/abl_closed_form/evaluate.py"


def fit_job(pair, seed):
    receiver, k = PAIRS[pair]
    run, features, mapper = (f"{BASE}/{pair}/closed_form/seed{seed}", f"{FEATURES}/{pair}/seed{seed}",
                             f"{MAPPERS}/{pair}/seed{seed}")
    return [f"mkdir -p {run} && cd .. && ./env3.10/bin/python -u scripts/abl_closed_form/fit.py",
            f"--calibration-dir multiagent/{run}/calibration --features-dir {features} --output-dir {mapper}",
            f"--k {k} --sender {SENDER} --receiver {receiver}",
            f"&& cd multiagent && cp {mapper}/*.json {run}/ && ln -sfn {mapper}/renderer.pt {run}/renderer.pt",
            f"&& rm -rf {features}"]


def eval_jobs(pair, seed, slow):
    run = f"{BASE}/{pair}/closed_form/seed{seed}"
    wait = f"while [ ! -f {run}/complete.json ]; do sleep 60; done;"
    tasks = [(task, r15.renderer_1x("X", stem, task, limit)) for stem, task, limit in m.EVALS_1X if task not in m.SKIP]
    tasks += [(spec, m.closed_book("X", spec, mnt, batch_size=16)) for spec, mnt in r15.CLOSED_BOOK]
    jobs = []
    for spec, cmd in sorted(tasks, key=lambda t: -m.MINUTES[t[0]]):
        if (spec in SLOW) != slow:
            continue
        cmd = [part.replace(f"X/{m.EXIT}/renderer.pt", f"{run}/renderer.pt").replace(f"X/{m.EXIT}", f"{run}/renderer.pt")
               .replace("--output X/", f"--output {run}/") for part in cmd]
        cmd[0] = wait + " " + cmd[0].replace("../env3.10/bin/python -u evaluation/renderer.py", EVAL)
        assert "X/" not in " ".join(cmd) and EVAL in cmd[0], cmd
        jobs.append({"cmd": cmd, "gpus": 1})
    return jobs


def main():
    order = [(pair, seed) for seed in SEEDS for pair in PAIRS]
    jobs = [f"# abl_many_seeds closed-form: fit + evals for {len(order)} mappers (scripts/abl_closed_form/make_many_seeds_queue.py)"]
    jobs += [{"cmd": fit_job(pair, seed), "gpus": 1} for pair, seed in order]
    for slow in [False, True]:
        for pair, seed in order:
            jobs += eval_jobs(pair, seed, slow)
    path = ROOT / "gpu_queue_commands" / "gpu_queue_commands_abl_many_seeds_closed_form.json"
    path.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"wrote {path} with {len(jobs) - 1} jobs")


if __name__ == "__main__":
    main()
