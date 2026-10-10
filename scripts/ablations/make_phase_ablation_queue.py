#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from make_ablation_queue import eval_jobs, train_job

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "multiagent"
PHASE1 = "abl16S1_phase1only_seed44_4b_to_llama8b"
PHASE2 = "abl16S2_phase2only_seed44_4b_to_llama8b_stage2_dense03"
CONTROL = "abl16C_control_seed44_4b_to_llama8b_stage2_dense03"


def main():
    control = json.loads((MULTI / "outputs" / CONTROL / "args.json").read_text())
    source = MULTI / control["init_weights_from"]
    if not (source / "renderer.pt").is_file():
        raise FileNotFoundError(source / "renderer.pt")
    phase1_dir = MULTI / "outputs" / PHASE1
    phase1_dir.mkdir(exist_ok=True)
    alias = phase1_dir / "exit-checkpoint-5000"
    if alias.is_symlink():
        if alias.resolve() != source.resolve():
            raise ValueError(f"existing checkpoint alias targets a different source: {alias}")
    elif alias.exists():
        raise FileExistsError(alias)
    else:
        alias.symlink_to(os.path.relpath(source, phase1_dir), target_is_directory=True)
    (phase1_dir / "ablation.json").write_text(json.dumps({
        "ablation": "no phase 2", "source_checkpoint": str(source.relative_to(MULTI)),
        "training_required": False, "control": CONTROL,
    }, indent=2) + "\n")

    train = train_job(PHASE2, "s2", "--dense-batch-fraction 0.3 --token-alignment native", None, 44)
    train["cmd"] = [part.replace("musique+limit=2500:train ", "") for part in train["cmd"]]
    jobs = [train]
    for name, steps in ((PHASE1, 5000), (PHASE2, 12000)):
        for job in eval_jobs(name, steps):
            if "--tasks musique:" in job["cmd"] or "--tasks quality:boosted" in job["cmd"]:
                continue
            job["after"] = [PHASE2]
            jobs.append(job)
    output = MULTI / "gpu_queue_commands_phase_ablations.json"
    output.write_text(json.dumps(jobs, indent=2) + "\n")
    (MULTI / "phase_ablation_manifest.json").write_text(json.dumps({
        "phase1_only": {"output": f"outputs/{PHASE1}", "source_checkpoint": str(source.relative_to(MULTI))},
        "phase2_only": {"output": f"outputs/{PHASE2}", "steps": 12000,
                        "initialization": "random renderer; fresh Adam; fresh cache-stat calibration",
                        "control": CONTROL, "learning_rate": 3e-6, "dense_fraction": 0.3},
        "physical_gpus": [2, 3], "jobs": str(output.relative_to(MULTI)),
    }, indent=2) + "\n")
    print(f"Prepared 1 training job and {len(jobs) - 1} evaluations: {output}")


if __name__ == "__main__":
    main()
