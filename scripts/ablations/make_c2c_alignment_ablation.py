#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from make_ablation_queue import train_job, eval_jobs

ROOT = Path(__file__).resolve().parents[2]
NAME = "abl16A_c2clongest_seed44_4b_to_llama8b"


def make_jobs():
    stage1, stage2 = NAME + "_stage1_kv", NAME + "_stage2_dense03"
    jobs = [
        train_job(stage1, "s1", "--token-alignment c2c-longest", None, 44),
        train_job(stage2, "s2", "--token-alignment c2c-longest --dense-batch-fraction 0.3",
                  f"outputs/{stage1}/exit-checkpoint-5000", 44, after=[stage1]),
    ]
    for job in jobs:
        job["cmd"] = [part.replace(" musique:train", "").replace("musique+limit=2500:train ", "")
                      for part in job["cmd"]]
    jobs.extend(job for job in eval_jobs(stage2, 12000)
                if "--tasks musique:" not in job["cmd"] and "--tasks quality:boosted" not in job["cmd"])
    return jobs


def main():
    path = ROOT / "multiagent/gpu_queue_commands_c2c_alignment.json"
    jobs = make_jobs()
    path.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Prepared 2 training jobs and {len(jobs) - 2} evaluation jobs: {path}")
    print("No jobs launched. Stage 1: 5k; stage 2: 12k; all dependencies explicit.")


if __name__ == "__main__":
    main()
