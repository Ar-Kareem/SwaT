#!/usr/bin/env python3
"""Experiment 2 for one sender -> receiver pair: train the query transfer, run T2T vs SwaT+AM over the sweep datasets,
time both arms, write results.

    ./env3.10/bin/python scripts/exp2/run.py run --config scripts/exp2/config.json --run multiagent/outputs/exp2_my_pair --gpus 1,2

`run` writes <run>/queue.json and launches it detached through multiagent/gpu_queue.py (log: <run>/queue.log).
`prepare` writes the queue without launching. `summarize` turns whatever cells exist into <run>/results.json.
Re-running `run` on the same directory skips every step whose output already exists, so it resumes.

The pair's renderer comes from exp1 (scripts/exp1/run.py, <exp1 run>/stage2/exit-checkpoint-<max-steps>/renderer.pt).
Everything else is trained or generated here. The steps:
  tq_features, tq_pairing   multiagent/train/head_pairings/{collect_query_features,score_pairings}.py: which sender
                            (layer, q-head) feeds each receiver unit (train/head_pairings/README.md)
  train_tq                  multiagent/train/train_query_transfer.py: the query-transfer MLP AM uses to turn sender
                            queries into receiver queries; its exit checkpoint is linked as <output-dir>/final
  prepare, summaries, t2t, am   scripts/exp2_sweep/run.py {prepare + fit, summarize, eval-pair, am} on a one-pair sweep
                            config this script writes to <run>/sweep_config.json (results under <run>/sweep, evaluator
                            intermediates under <run>/sweep_scratch); am is one job per dataset (am_<dataset>) so
                            they run in parallel on the free GPUs
  speed_subset, speed_sender, speed_receiver   scripts/exp2_sweep/speed.py on the same config (results under
                            <run>/timings); they run one after another so the timings never overlap
  results                   `summarize`

The config holds the whole recipe (scripts/exp2/config.json: the Granite-3.3-8B -> SmolLM3-3B pair of
scripts/exp1/config.json). Nothing is defaulted: every key below is required.
  name               pair name in the sweep tree (<run>/sweep/<dataset>/<name>/{t2t,am}.json)
  sender, receiver   model alias from multiagent/model/vanilla_loader.py, a path under MULTIKV_MODELS_ROOT, or an absolute path
  renderer           the pair's stage-2 renderer.pt
  budget             AM head-budget JSON (multiagent/model/head_budget_optimization/head_budgets/<receiver>/optimized_agnostic.json)
                     or "uniform" for receivers without one (the optimizer that made those files is not in the repo)
  train_tq           features, pairing, train: gpus and args (that script's flags) for each of the three steps
  pair               sweep pair options (variant, fit_filter, fit_filter_exempt, am_overrides, ...)
  sweep              sweep protocol keys, copied into the sweep config verbatim (see scripts/exp2_sweep/config.json)
  gpus               prepare, summaries, t2t, am: GPUs per sweep step
  speed              enabled; contexts: timed contexts per dataset (plus one warmup); gpus
Flag dicts map "flag-name" -> value: true adds --flag-name, false/null omit it, a list passes several values.
String values may use {repo} {run} {sender} {receiver} {features} {pairing} {query_transfer}.

Recipe notes: the train_tq recipe is the one every AM pair used (args.json of the 14E/15A/15H qq_query_transfer runs).
this script uses the exit checkpoint and stops it at max-steps 2000 (the cosine horizon stays the full 6 epochs, so that 
checkpoint is the 2000-step one of the full run): a from-scratch 14E rerun at 2000 steps reproduced the paper's exp2 scores. 
Contexts over 7,500 receiver tokens are left out of both arms (fit filter, as 14E/15A), 
and LongHealth, whose documents are all longer, is plainly truncated to its first 7,500 receiver tokens for both arms.

`summarize` reports per dataset and summary level: T2T and SwaT score (token F1 where the grader reports it, accuracy
otherwise), tokens communicated, S GPU and R prefill per request (once timed), and their mean over the datasets by budget
rank plus an overall mean, using the shared calculations in multiagent/evaluation/exp2_metrics.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MULTI = ROOT / "multiagent"
SWEEP = ROOT / "scripts/exp2_sweep"
sys.path.insert(0, str(ROOT))
from multiagent.paths import PYTHON
from multiagent.evaluation import exp2_metrics

SWEEP_STEPS = ("prepare", "summaries", "t2t", "am")
METRICS = ("score", "tokens", "s_gpu", "r_prefill")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load("exp2_sweep_run", SWEEP / "run.py")


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
    require(c, "name sender receiver renderer budget train_tq pair sweep gpus speed", "top level")
    for step in ("features", "pairing", "train"):
        require(c["train_tq"], step, "train_tq")
        require(c["train_tq"][step], "gpus args", f"train_tq.{step}")
    require(c["train_tq"]["features"]["args"], "output", "train_tq.features.args")
    require(c["train_tq"]["train"]["args"], "output-dir pairing-file", "train_tq.train.args")
    require(c["gpus"], " ".join(SWEEP_STEPS), "gpus")
    require(c["speed"], "enabled contexts gpus", "speed")
    reserved = {"output", "scratch", "pairs", "am", "active_pairs", "timings_output", "speed_pairs", "speed_truncated"}
    if reserved & set(c["sweep"]):
        raise SystemExit(f"sweep must not set {sorted(reserved & set(c['sweep']))}; run.py writes them")
    if {"sender", "receiver"} & set(c["pair"]):
        raise SystemExit("pair must not set sender/receiver; they are the top-level keys")


def substitute(value, names: dict):
    if isinstance(value, str):
        return value.format(**names)
    if isinstance(value, dict):
        return {k: substitute(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, names) for v in value]
    return value


def pairing_path(features: str, args: dict) -> str:
    """The pairing file score_pairings.py writes for these flags (its tag rule)."""
    tag = (("untied" if args.get("no-tie-kv") else "tied") + ("_rms" if args.get("rms-scales") else "")
           + (f"_w{args['allowed-width']}" if args.get("allowed-width") else ""))
    return str(Path(args.get("output-dir") or Path(features).parent) / f"pairing_{tag}.json")


def resolve(raw: dict, run: Path) -> dict:
    """Config with every placeholder filled in, plus the derived step outputs under "paths"."""
    c = raw
    validate(c)
    names = {"repo": str(ROOT), "run": str(run), "sender": c["sender"], "receiver": c["receiver"]}
    tq = c["train_tq"]
    names["features"] = substitute(tq["features"]["args"]["output"], names)
    names["pairing"] = pairing_path(names["features"], substitute(tq["pairing"]["args"], names))
    names["query_transfer"] = f"{substitute(tq['train']['args']['output-dir'], names)}/final/query_transfer.pt"
    return {**substitute(c, names), "paths": {k: names[k] for k in ("features", "pairing", "query_transfer")}}


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


def sweep_config(c: dict, run: Path) -> dict:
    """The one-pair exp2_sweep config: the protocol keys of `sweep` plus this run's pair, AM inputs and directories."""
    rel = lambda p: str(p.relative_to(ROOT))
    return {**c["sweep"], "output": rel(run / "sweep"), "scratch": rel(run / "sweep_scratch"),
            "pairs": {c["name"]: {"sender": c["sender"], "receiver": c["receiver"], **c["pair"]}},
            "am": {c["name"]: {"renderer": c["renderer"], "query_transfer": c["paths"]["query_transfer"], "budget": c["budget"]}},
            "timings_output": rel(run / "timings"), "speed_pairs": [c["name"]], "speed_truncated": []}


def finalize_tq(run: Path) -> None:
    """Link the trainer's exit checkpoint as <output-dir>/final (the path the AM config names)."""
    out = Path(read(run / "config.json")["train_tq"]["train"]["args"]["output-dir"])
    checkpoint = read(out / "latest.json")["checkpoint"]
    if not checkpoint.startswith("exit-checkpoint-") or not (out / checkpoint / "query_transfer.pt").exists():
        raise SystemExit(f"{out}: latest checkpoint {checkpoint} is not a finished exit checkpoint")
    final = out / "final"
    final.unlink(missing_ok=True)
    final.symlink_to(checkpoint)
    print(f"{final} -> {checkpoint}")


# ----------------------------------------------------------------------------- done checks (the sweep's own rules)

def sweep_state(config_path: Path):
    """(config, run, manifest or None) of the sweep; manifest only when prepare finished with this exact config."""
    config = sweep.load_config(config_path)
    run = sweep.out_dir(config)
    manifest_path = run / "manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else None
    if manifest is not None and manifest["config"] != read(config_path):
        manifest = None
    return config, run, manifest


def prepared(config_path: Path) -> bool:
    config, run, manifest = sweep_state(config_path)
    if manifest is None:
        return False
    if "fit_filter" not in config:
        return True
    return all(sweep.fit_sidecar_path(run, t, sweep.pair_variant(config, n, t)).exists()
               for t in config["datasets"] for n in config["pairs"])


def summaries_done(config_path: Path, sender: str) -> bool:
    config, run, _ = sweep_state(config_path)
    return prepared(config_path) and all(
        sweep.summaries_cover(run, t, sender, config["prompts"], sweep.sender_variant(config, sender, t), config)
        for t in config["datasets"])


def t2t_done(config_path: Path, name: str) -> bool:
    config, _, _ = sweep_state(config_path)
    return prepared(config_path) and all((sweep.scratch_dir(config) / t / name / "t2t.done.json").exists()
                                         for t in config["datasets"])


def am_done(config_path: Path, name: str, task: str) -> bool:
    config, run, _ = sweep_state(config_path)
    return t2t_done(config_path, name) and (run / task / name / "am.json").exists()


def timings(run: Path, name: str):
    """(subset cells, sender cells, receiver cells) of the pair; each None until its file exists."""
    out = run / "timings"
    get = lambda f: read(out / f) if (out / f).exists() else None
    subset, sender, receiver = get("subset.json"), get(f"sender_hf_{name}.json"), get(f"receiver_{name}.json")
    return (subset and subset["cells"].get(name, {}), sender and sender["cells"], receiver and receiver["cells"])


def sender_timed(subset: dict, sender: dict | None, prompts) -> bool:
    return sender is not None and all(
        all(len(sender.get(t, {}).get("levels", {}).get(lv, {})) == len(cell["contexts"]) for lv in prompts)
        for t, cell in subset.items())


def receiver_timed(subset: dict, receiver: dict | None) -> bool:
    return receiver is not None and all(receiver.get(t, {}).get("complete") for t in subset)


# ----------------------------------------------------------------------------- queue

def build_queue(c: dict, run: Path) -> list:
    py = [str(PYTHON), "-u"]
    queue = []

    def add(name, cmd, gpus, after=(), done=False):
        if done:
            return  # resume: finished steps drop out, and so do dependencies on them
        queued = {job["name"] for job in queue}
        queue.append({"name": name, "cmd": shlex.join(cmd) if isinstance(cmd, list) else cmd, "gpus": gpus,
                      "after": [a for a in after if a in queued]})

    me = str(Path(__file__).resolve())
    tq = c["train_tq"]
    add("tq_features", [*py, str(MULTI / "train/head_pairings/collect_query_features.py"), *flags(tq["features"]["args"])],
        tq["features"]["gpus"], done=Path(c["paths"]["features"]).exists())
    add("tq_pairing", [*py, str(MULTI / "train/head_pairings/score_pairings.py"), *flags(tq["pairing"]["args"])],
        tq["pairing"]["gpus"], ["tq_features"], done=Path(c["paths"]["pairing"]).exists())
    add("train_tq", shlex.join([*py, str(MULTI / "train/train_query_transfer.py"), *flags(tq["train"]["args"])])
        + " && " + shlex.join([*py, me, "finalize-tq", "--run", str(run)]),
        tq["train"]["gpus"], ["tq_pairing"], done=Path(c["paths"]["query_transfer"]).exists())

    config_path = run / "sweep_config.json"
    driver = [*py, str(SWEEP / "run.py")]
    on = lambda command, *extra: shlex.join([*driver, command, "--config", str(config_path), *extra])
    gpus, name = c["gpus"], c["name"]
    add("prepare", on("prepare") + (" && " + on("fit") if "fit_filter" in c["sweep"] else ""), gpus["prepare"],
        done=prepared(config_path))
    add("summaries", sweep.summarize_command(sweep.load_config(config_path), config_path, c["sender"]), gpus["summaries"],
        ["prepare"], done=summaries_done(config_path, c["sender"]))
    add("t2t", on("eval-pair", "--pair", name), gpus["t2t"], ["summaries"], done=t2t_done(config_path, name))
    # One AM job per dataset so the queue spreads them over every free GPU (AM is the longest step, ~6 h on one GPU
    # for the seven datasets); config dataset order is launch order, so the largest (NarrativeQA) goes first
    am_jobs = [f"am_{task}" for task in c["sweep"]["datasets"]]
    for job, task in zip(am_jobs, c["sweep"]["datasets"]):
        add(job, on("am", "--pair", name, "--datasets", task), gpus["am"], ["t2t", "train_tq"],
            done=am_done(config_path, name, task))

    s = c["speed"]
    if s["enabled"]:
        speed = lambda command, *extra: shlex.join([*py, str(SWEEP / "speed.py"), command, "--config", str(config_path), *extra])
        subset, sender, receiver = timings(run, name)
        add("speed_subset", speed("subset", "--contexts", str(s["contexts"])), 1, am_jobs, done=subset is not None)
        add("speed_sender", speed("sender", "--pair", name), s["gpus"], ["speed_subset"],
            done=subset is not None and sender_timed(subset, sender, c["sweep"]["prompts"]))
        add("speed_receiver", speed("receiver", "--pair", name), s["gpus"], ["speed_sender"],
            done=subset is not None and receiver_timed(subset, receiver))
    # results always reruns (cheap) once the steps still in the queue are done
    add("results", [*py, me, "summarize", "--run", str(run)], 1, [j["name"] for j in queue])
    return queue


# ----------------------------------------------------------------------------- summarize

def summarize(run: Path) -> dict:
    """Per dataset and level: (T2T, SwaT) score, tokens and, once timed, S GPU and R prefill; means by budget rank."""
    c = read(run / "config.json")
    name, prompts = c["name"], c["sweep"]["prompts"]
    subset, sender, receiver = timings(run, name)
    timed = bool(subset) and sender_timed(subset, sender, prompts) and receiver_timed(subset, receiver)
    datasets, missing = {}, []
    for task in c["sweep"]["datasets"]:
        am_path, t2t_path = (run / "sweep" / task / name / f for f in ("am.json", "t2t.json"))
        if not am_path.exists() or not t2t_path.exists():
            missing.append(task)
            continue
        am, t2t = read(am_path), read(t2t_path)["results_by_prompt"]
        cell = subset.get(task) if timed else None
        levels = {}
        for level in prompts:
            rows = exp2_metrics.resolve(am["results_by_level"], level)["rows"]
            t2t_rows = {r["id"]: r for r in t2t[level]["rows"]}
            if any(r["id"] not in t2t_rows for r in rows):
                raise ValueError(f"{task}/{level}: AM questions without a T2T row")
            values = {
                "score": (100 * st.mean(exp2_metrics.score(t2t_rows[r["id"]]) for r in rows), 100 * st.mean(exp2_metrics.score(r) for r in rows)),
                "tokens": (st.mean(t2t_rows[r["id"]]["summarization"]["summary_tokens_receiver"] for r in rows),
                           st.mean(exp2_metrics.swat_tokens(r) for r in rows)),
            }
            if cell is not None:
                swat_level = exp2_metrics.resolve(receiver[task]["am_levels"], level)
                values["s_gpu"] = (st.mean(sender[task]["levels"][level][x]["preparation_ms"] for x in cell["contexts"]),
                                   st.mean(swat_level["sender"][x]["sender_ms"] for x in cell["contexts"]))
                values["r_prefill"] = (receiver[task]["t2t_levels"][level]["t2t_ms_per_request"], swat_level["am_ms_per_request"])
            levels[level] = {"n": len(rows), **am["budgets"][level],
                             **{k: {"t2t": t, "swat": s, "change": exp2_metrics.change_value(t, s, k)} for k, (t, s) in values.items()}}
        datasets[task] = levels

    def mean_section(cells):
        out = {}
        for k in METRICS:
            pairs = [(cell[k]["t2t"], cell[k]["swat"]) for cell in cells if k in cell]
            if len(pairs) == len(cells) and pairs:
                t, s = (exp2_metrics.average(k)(arm) for arm in zip(*pairs))
                out[k] = {"t2t": t, "swat": s, "change": exp2_metrics.change_value(t, s, k)}
        return out

    # Budget rank: each dataset's levels sorted by summary retention (as the appendix mean table)
    ranked = [sorted(levels.values(), key=lambda v: v["summary_retention"]) for levels in datasets.values()]
    means = {f"rank{i + 1}": mean_section([r[i] for r in ranked]) for i in range(len(prompts))} if ranked else {}
    if ranked:
        means["overall"] = mean_section([cell for r in ranked for cell in r])
    results = {"name": name, "sender": c["sender"], "receiver": c["receiver"], "budget": c["budget"],
               "timed": timed, "datasets": datasets, "mean": means, "missing": missing}
    write(run / "results.json", results)
    for task, levels in datasets.items():
        for level, v in levels.items():
            line = f"{task:12s} {level:30s} x{v['compression']:<3d} score T2T {v['score']['t2t']:5.1f} SwaT {v['score']['swat']:5.1f}"
            if "s_gpu" in v:
                line += f"  S GPU x{v['s_gpu']['change']:.1f}  R prefill x{v['r_prefill']['change']:.1f}"
            print(line)
    if "overall" in means:
        o = means["overall"]
        print(f"overall: score T2T {o['score']['t2t']:.1f} SwaT {o['score']['swat']:.1f} ({o['score']['change']:+.1f} pts)"
              + (f", S GPU x{o['s_gpu']['change']:.1f}, R prefill x{o['r_prefill']['change']:.1f}" if "s_gpu" in o else ""))
    print(f"timed: {timed}; missing: {missing or 'none'}")
    return results


def launch(run: Path, gpus: str | None) -> None:
    args = [str(PYTHON), str(MULTI / "gpu_queue.py"), "--commands", str(run / "queue.json"), "--name", "exp2"]
    if gpus:
        args += ["--gpus", gpus]
    with (run / "queue.log").open("a") as log:
        process = subprocess.Popen(args, cwd=MULTI, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write(run / "launch.json", {"pid": process.pid, "command": args})
    print(f"Detached exp2 queue PID {process.pid}; log: {run / 'queue.log'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["prepare", "run", "summarize", "finalize-tq"])
    parser.add_argument("--run", type=Path, required=True, help="Run directory under the repository; holds the resolved config, "
                        "query transfer, sweep tree, timings, results")
    parser.add_argument("--config", type=Path, help="Recipe config JSON (required for prepare/run)")
    parser.add_argument("--gpus", help="Physical GPU IDs the detached queue may use, e.g. 1,2 (tq_features needs two)")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == "summarize":
        summarize(run)
        return
    if args.action == "finalize-tq":
        finalize_tq(run)
        return
    if not args.config:
        parser.error(f"{args.action} needs --config")
    if not run.is_relative_to(ROOT):
        raise SystemExit(f"--run must be inside {ROOT} (speed.py records repository-relative paths)")
    config = resolve(read(args.config), run)
    if (run / "config.json").exists() and read(run / "config.json") != config:
        raise SystemExit(f"{run / 'config.json'} holds a different recipe; use a new --run directory")
    if config["budget"] != "uniform" and not Path(config["budget"]).exists():
        raise SystemExit(f"budget {config['budget']} does not exist (use \"uniform\" for receivers without one)")
    if not Path(config["renderer"]).exists():
        print(f"WARNING: renderer {config['renderer']} does not exist yet; the am step will fail until it does")
    write(run / "config.json", config)
    write(run / "sweep_config.json", sweep_config(config, run))
    queue = build_queue(config, run)
    write(run / "queue.json", queue)
    print(f"wrote {run / 'queue.json'}: {len(queue)} job(s) ({', '.join(j['name'] for j in queue)})")
    if args.action == "run":
        if len(queue) == 1:
            print("every step is done; results only")
        launch(run, args.gpus)


if __name__ == "__main__":
    main()
