import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_renderer15_queue as r15

ROOT = Path(__file__).resolve().parents[2] / "multiagent"
BASE = "outputs/abl_many_seeds"
SKIP = {"gsm8k", "math-500", "quality:boosted"}
BATCHED = {"gsm8k", "math-500"}
EVALS_1X = r15.EVALS_1X + [("narrativeqa_test400_renderer_1x", "narrativeqa:test", 400)]
EXIT = f"exit-checkpoint-{r15.STEPS}"
MINUTES = {"longhealth+ppc=1:test": 1.4, "arc-c:test": 1.7, "winogrande:validation": 1.7, "hellaswag:validation": 1.8,
           "mmlu:test": 1.8, "openbookqa:test": 1.8, "medqa:test": 1.9, "qasper:test": 2.2, "quality:boosted": 2.5,
           "hotpotqa:validation": 3.1, "musique:validation": 4.4, "quality:dev": 5.6, "race:test": 11.6,
           "squad:validation": 12.6, "boolq:validation": 15.2, "narrativeqa:test": 2.3,
           "gsm8k:test": 85.2, "math-500:test": 128.5}


def closed_book(stage2, spec, mnt, batch_size=1):
    cmd = r15.renderer_cb(stage2, "x", spec, mnt)
    if batch_size > 1 and spec.split(":")[0] in BATCHED:
        cmd[1] += f" --batch-size {batch_size}"
    cmd[-1] = f"--output {stage2}/eval_{spec.split(':')[0]}_n500.json"
    return cmd


def at_checkpoint(cmd, stage2, checkpoint):
    if checkpoint == EXIT:
        return cmd
    cmd = [part.replace(f"{stage2}/{EXIT}", f"{stage2}/{checkpoint}") for part in cmd]
    cmd[-1] = cmd[-1].replace(f"--output {stage2}/", f"--output {stage2}/{checkpoint}/")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", help="emit only these task specs (overrides SKIP), longest first")
    ap.add_argument("--batch-size", type=int, default=1, help="renderer --batch-size for gsm8k/math-500 jobs")
    ap.add_argument("--checkpoint", default=EXIT, help="checkpoint directory under each stage2 to evaluate")
    ap.add_argument("--out", default="gpu_queue_commands_abl_many_seeds_evals.json")
    args = ap.parse_args()
    stage2s = sorted(str(p.parent.parent.relative_to(ROOT)) for p in ROOT.glob(f"{BASE}/*/*/seed*/stage2/{args.checkpoint}/renderer.pt"))
    tasks = [(task, at_checkpoint(r15.renderer_1x(s, stem, task, limit), s, args.checkpoint))
             for s in stage2s for stem, task, limit in EVALS_1X if task not in SKIP or args.tasks]
    tasks += [(spec, at_checkpoint(closed_book(s, spec, mnt, args.batch_size), s, args.checkpoint))
              for s in stage2s for spec, mnt in r15.CLOSED_BOOK if spec.split(":")[0] not in SKIP or args.tasks]
    tasks = [t for t in tasks if not args.tasks or t[0] in args.tasks]
    tasks.sort(key=lambda t: MINUTES[t[0]], reverse=bool(args.tasks))
    jobs = [f"# abl_many_seeds evals at {args.checkpoint}: {', '.join(stage2s)} (scripts/exp1/make_abl_many_seeds_eval_queue.py)"]
    jobs += [{"cmd": cmd, "gpus": 1} for _, cmd in tasks]
    path = ROOT / "gpu_queue_commands" / args.out
    path.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"wrote {path} with {len(jobs) - 1} jobs")


if __name__ == "__main__":
    main()
