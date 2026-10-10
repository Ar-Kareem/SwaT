#!/usr/bin/env python3
"""Experiment 1 for one sender -> receiver pair: train swat with the recipe in a config JSON, evaluate it, write results.

    ./env3.10/bin/python scripts/exp1/run.py run --config scripts/exp1/config.json --run multiagent/outputs/exp1_my_pair --gpus 1,2

`run` writes <run>/queue.json and launches it detached through multiagent/gpu_queue.py (log: <run>/queue.log).
`prepare` writes the queue without launching. `summarize` turns whatever evals exist into <run>/results.json.
Re-running `run` on the same directory skips every step whose output already exists, so it resumes.

The config holds the whole recipe; this script only turns it into commands (scripts/exp1/config.json is the exp1
recipe for Granite-3.3-8B -> SmolLM3-3B). Nothing is defaulted: every key below is required.
  sender, receiver   model alias from multiagent/model/vanilla_loader.py, a path under MULTIKV_MODELS_ROOT, or an absolute path
  traces             raw, clean: trace JSONL paths; min_raw_rows, min_clean_rows: filter guards; gpus;
                     generate_args: multiagent/create_phase2_dataset.py flags. An existing `clean` file skips generation.
  train_common.args  multiagent/train/train_cache_renderer.py flags shared by both stages
  stage1, stage2     gpus; args: that stage's trainer flags (must not repeat a train_common flag). The finished stage is
                     <output-dir>/exit-checkpoint-<max-steps>.
  eval               enabled; output_dir; common_args; vanilla/renderer: gpus and args for that arm; benchmarks: list of
                     {key, domain, metric, args, vanilla_args, renderer_args} (vanilla_args optional), each run through
                     evaluation/vanilla.py or evaluation/renderer.py once per arm into <output_dir>/<key>_<arm>.json. metric is
                     "accuracy" (row grade) or a grade_result field ("f1", "answer_f1").
Flag dicts map "flag-name" -> value: true adds --flag-name, false/null omit it, a list passes several values.
String values may use {repo} {run} {sender} {receiver} {traces} {stage1_checkpoint} {stage2_checkpoint}.

Settings past pairs changed from scripts/exp1/config.json: n-layers divides the receiver's kv layers (36 -> 9, 40/32 -> 8,
Gemma-2's 42 -> 7); a Gemma-2 receiver (256k vocab) used stage-2 max-trace-length 2048, dense-max-length 2560,
empty-cache-every 25; a sender that fills its GPU alone (Qwen3-32B) used renderer-device cuda:1 in training and
sender-device cuda:1 with 2 GPUs for renderer evals. Stage 2 sets checkpoint-receiver (as 15G/H/I did): on the 2-GPU
layout the receiver GPU fails the OOM preflight's 8192+12288 CE shape without it (77+ GiB; 22 GiB with it), and recomputing
the frozen receiver's activations leaves the math unchanged. The vanilla arm of the four
span-QA benchmarks uses the +terse prompt because the default prompt was bad due to answering in sentences and 
collapsing the vanilla token F1 (unfairly giving SwaT high retention).

Vanilla eval jobs do not depend on training, so they fill idle GPUs while it runs. `summarize` reports per-benchmark
receiver (vanilla) score, swat (renderer) score, retention = 100 * renderer / vanilla, and mean retention overall
and per domain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "multiagent"
sys.path.insert(0, str(ROOT))
from multiagent.paths import PYTHON

ARMS = ("vanilla", "renderer")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def require(mapping: dict, keys: str, where: str) -> None:
    missing = [k for k in keys.split() if k not in mapping]
    if missing:
        raise SystemExit(f"config {where} is missing {missing}")


def validate(c: dict) -> None:
    require(c, "sender receiver traces train_common stage1 stage2 eval", "top level")
    require(c["traces"], "raw clean min_raw_rows min_clean_rows gpus generate_args", "traces")
    require(c["train_common"], "args", "train_common")
    for stage in ("stage1", "stage2"):
        require(c[stage], "gpus args", stage)
        require(c[stage]["args"], "output-dir max-steps", f"{stage}.args")
        overlap = set(c[stage]["args"]) & set(c["train_common"]["args"])
        if overlap:
            raise SystemExit(f"{stage}.args repeats train_common flags {sorted(overlap)}")
    require(c["eval"], "enabled output_dir common_args vanilla renderer benchmarks", "eval")
    for arm in ARMS:
        require(c["eval"][arm], "gpus args", f"eval.{arm}")
    keys = [b.get("key") for b in c["eval"]["benchmarks"]]
    if len(set(keys)) != len(keys):
        raise SystemExit("eval benchmark keys must be unique")
    for b in c["eval"]["benchmarks"]:
        require(b, "key domain metric args renderer_args", f"benchmark {b.get('key')}")


def substitute(value, names: dict):
    if isinstance(value, str):
        return value.format(**names)
    if isinstance(value, dict):
        return {k: substitute(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, names) for v in value]
    return value


def resolve(raw: dict, run: Path) -> dict:
    """Config with every placeholder filled in."""
    c = raw
    validate(c)
    names = {"repo": str(ROOT), "run": str(run), "sender": c["sender"], "receiver": c["receiver"]}
    names["traces"] = substitute(c["traces"]["clean"], names)
    for stage in ("stage1", "stage2"):
        args = c[stage]["args"]
        names[f"{stage}_checkpoint"] = f"{substitute(args['output-dir'], names)}/exit-checkpoint-{args['max-steps']}"
    return substitute(c, names)


def flags(args: dict) -> list[str]:
    out = []
    for name, value in args.items():
        if value is None or value is False:
            continue
        out.append(f"--{name}")
        if isinstance(value, list):
            out.extend(str(v) for v in value)
        elif value is not True:
            out.append(str(value))
    return out


def check_blocks(c: dict) -> None:
    from transformers import AutoConfig
    from multiagent.model.unaware_comm import kv_layer_indices
    from multiagent.model.vanilla_loader import resolve_model_path

    for stage in ("stage1", "stage2"):
        blocks = {**c["train_common"]["args"], **c[stage]["args"]}.get("n-layers")
        if blocks is None:
            raise SystemExit(f"{stage}: n-layers must be set (train_common or {stage}.args); the trainer default is not part of the recipe")
    kv_layers = len(kv_layer_indices(AutoConfig.from_pretrained(resolve_model_path(c["receiver"]))))
    if kv_layers % int(blocks):
        print(f"NOTE: n-layers {blocks} does not divide the receiver's {kv_layers} kv layers; receiver layers are split proportionally (paper pairs all used a divisor)")


def done_marker(output: Path) -> Path:
    return output.with_suffix(".done")


def build_queue(c: dict, run: Path) -> list:
    py = [str(PYTHON), "-u"]
    queue = []

    def add(name, cmd, gpus, after=(), output=None):
        if output is not None and Path(output).exists():
            return  # resume: finished steps drop out, and so do dependencies on them
        queued = {job["name"] for job in queue}
        queue.append({"name": name, "cmd": shlex.join(cmd) if isinstance(cmd, list) else cmd, "gpus": gpus,
                      "after": [a for a in after if a in queued]})

    t = c["traces"]
    add("traces", [*py, str(MULTI / "create_phase2_dataset.py"), *flags(t["generate_args"]), "--output", t["raw"]],
        t["gpus"], output=t["clean"] if Path(t["clean"]).exists() else t["raw"])
    add("filter_traces", f"test $(wc -l < {shlex.quote(t['raw'])}) -ge {int(t['min_raw_rows'])} && "
        + shlex.join([*py, str(ROOT / "scripts/download/filter_phase2_traces.py"), "--input", t["raw"], "--output", t["clean"]])
        + f" && test $(wc -l < {shlex.quote(t['clean'])}) -ge {int(t['min_clean_rows'])}", 1, ["traces"], output=t["clean"])
    train = [*py, str(MULTI / "train/train_cache_renderer.py"), *flags(c["train_common"]["args"])]
    for stage, after in (("stage1", []), ("stage2", ["stage1", "filter_traces"])):
        args = c[stage]["args"]
        checkpoint = Path(args["output-dir"]) / f"exit-checkpoint-{args['max-steps']}" / "renderer.pt"
        add(stage, [*train, *flags(args)], c[stage]["gpus"], after, output=checkpoint)
    e = c["eval"]
    if not e["enabled"]:
        return queue
    for arm in ("renderer", "vanilla"):  # renderer arms first so they start as soon as stage 2 finishes
        for b in e["benchmarks"]:
            name = f"{b['key']}_{arm}"
            output = Path(e["output_dir"]) / f"{name}.json"
            args = {**e["common_args"], **e[arm]["args"], **b["args"], **b.get(f"{arm}_args", {})}
            # The evaluator checkpoints partial rows to --output every 25 rows, so the file alone does not mean done:
            # <name>.done is touched only after a clean exit, and resume and summarize go by it.
            entry = MULTI / ("evaluation/vanilla.py" if arm == "vanilla" else "evaluation/renderer.py")
            add(name, shlex.join([*py, str(entry), *flags(args), "--output", str(output)])
                + " && touch " + shlex.quote(str(done_marker(output))),
                e[arm]["gpus"], ["stage2"] if arm == "renderer" else [], output=done_marker(output))
    # summarize always reruns (cheap) once the evals still in the queue are done
    add("summarize", [*py, str(Path(__file__).resolve()), "summarize", "--run", str(run)], 1,
        [j["name"] for j in queue if j["name"].endswith(ARMS)])
    return queue


def read_scores(path: Path, metric: str) -> dict:
    scores = {}
    for row in read(path)["rows"]:
        value = float(row["grade"] == "correct") if metric == "accuracy" else float(row["grade_result"][metric])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {metric} in {path}")
        scores[(row["id"], row["index"])] = value
    return scores


def summarize(run: Path) -> dict:
    """Per-benchmark vanilla score, renderer score, and retention (percent); means over the benchmarks present."""
    c = read(run / "config.json")
    benchmarks, missing = {}, []
    for b in c["eval"]["benchmarks"]:
        paths = {arm: Path(c["eval"]["output_dir"]) / f"{b['key']}_{arm}.json" for arm in ARMS}
        if not all(done_marker(p).exists() for p in paths.values()):
            missing.append(b["key"])
            continue
        vanilla, renderer = (read_scores(paths[arm], b["metric"]) for arm in ARMS)
        if renderer.keys() != vanilla.keys():
            raise ValueError(f"renderer and vanilla rows differ for {b['key']}")
        vanilla_score, renderer_score = 100 * mean(vanilla.values()), 100 * mean(renderer.values())
        benchmarks[b["key"]] = {"domain": b["domain"], "metric": b["metric"], "n": len(renderer),
                                "vanilla": vanilla_score, "renderer": renderer_score,
                                "retention": 100 * renderer_score / vanilla_score if vanilla_score else None}

    def retention(keys):
        values = [benchmarks[k]["retention"] for k in keys if benchmarks[k]["retention"] is not None]
        return mean(values) if values else None

    domains = sorted({b["domain"] for b in benchmarks.values()})
    results = {"sender": c["sender"], "receiver": c["receiver"], "benchmarks": benchmarks,
               "mean_retention": {"all": retention(list(benchmarks)),
                                  **{d: retention([k for k, b in benchmarks.items() if b["domain"] == d]) for d in domains}},
               "missing": missing}
    write(run / "results.json", results)
    for key, b in benchmarks.items():
        print(f"{key:24s} vanilla {b['vanilla']:6.1f}  renderer {b['renderer']:6.1f}  retention {b['retention'] or float('nan'):6.1f}%")
    print(f"mean retention: {results['mean_retention']}; missing: {missing or 'none'}")
    return results


def launch(run: Path, gpus: str | None) -> None:
    args = [str(PYTHON), str(MULTI / "gpu_queue.py"), "--commands", str(run / "queue.json"), "--name", "exp1"]
    if gpus:
        args += ["--gpus", gpus]
    with (run / "queue.log").open("a") as log:
        process = subprocess.Popen(args, cwd=MULTI, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write(run / "launch.json", {"pid": process.pid, "command": args})
    print(f"Detached exp1 queue PID {process.pid}; log: {run / 'queue.log'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    parser.add_argument("--run", type=Path, required=True, help="Run directory; holds the resolved config, traces, both stages, evals, results")
    parser.add_argument("--config", type=Path, help="Recipe config JSON (required for prepare/run)")
    parser.add_argument("--gpus", help="Physical GPU IDs the detached queue may use, e.g. 1,2 (training needs two)")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == "summarize":
        summarize(run)
        return
    if not args.config:
        parser.error(f"{args.action} needs --config")
    import math_verify  # noqa: F401  without it MATH-500 silently falls back to exact-string grading (worse grading)
    config = resolve(read(args.config), run)
    if (run / "config.json").exists() and read(run / "config.json") != config:
        raise SystemExit(f"{run / 'config.json'} holds a different recipe; use a new --run directory")
    check_blocks(config)
    write(run / "config.json", config)
    queue = build_queue(config, run)
    write(run / "queue.json", queue)
    print(f"wrote {run / 'queue.json'}: {len(queue)} job(s) ({', '.join(j['name'] for j in queue[:4])}{', ...' if len(queue) > 4 else ''})")
    if args.action == "run":
        if not queue:
            print("nothing to do; every output exists")
            return
        launch(run, args.gpus)


if __name__ == "__main__":
    main()
