#!/usr/bin/env python3
import argparse, json, math, sys
from pathlib import Path

QUICK_1X = {
    "qasper": ("eval_qasper_test100_renderer_1x.json", 100),
    "longhealth": ("eval_longhealth_renderer_1x.json", 400),
    "hotpotqa": ("eval_hotpotqa_val400_renderer_1x.json", 400),
    "musique": ("eval_musique_val400_renderer_1x.json", 400),
    "quality_boosted": ("eval_quality_boosted_renderer_1x.json", 283),
    "quality_dev": ("eval_quality_dev_renderer_1x.json", 2086),
    "squad": ("eval_squad_validation20pct_renderer_1x.json", 2114),
    "boolq": ("eval_boolq_validation_renderer_1x.json", 3270),
    "race": ("eval_race_test_renderer_1x.json", 4934),
}
AM = {
    "qasper_am10": ("eval_qasper_test100_renderer_am10.json", 100),
    "qasper_am05": ("eval_qasper_test100_renderer_am05.json", 100),
    "qasper_am02": ("eval_qasper_test100_renderer_am02.json", 100),
    "longhealth_am10": ("eval_longhealth_renderer_am10.json", 400),
    "longhealth_am05": ("eval_longhealth_renderer_am05.json", 400),
    "longhealth_am02": ("eval_longhealth_renderer_am02.json", 400),
    "longhealth_am01": ("eval_longhealth_renderer_am01.json", 400),
}
FLOORS = {"longhealth": 0.45, "quality_dev": 0.45, "hotpotqa": 0.45, "qasper": 0.15}


def _row_key(r):
    return (r.get("id"), r.get("question"))


def read_eval(path: Path):
    if not path.exists():
        return None
    try:
        j = json.load(open(path))
    except Exception:
        return None
    s = j.get("summary") or {}
    if not s:
        return None
    entry = next(iter(s.values()))
    rows = {_row_key(r): str(r.get("grade")).lower() == "correct" for r in j.get("rows", []) if r.get("grade") is not None}
    return {"seen": entry.get("seen"), "acc": entry.get("accuracy"), "f1": entry.get("answer_f1", entry.get("f1")),
            "elapsed": j.get("elapsed_human"), "rows": rows}


def paired(e, r):
    if not e or not r or not e.get("rows") or not r.get("rows"):
        return None
    common = [k for k in r["rows"] if k in e["rows"]]
    if len(common) < 50:
        return None
    return (sum(e["rows"][k] for k in common) / len(common), sum(r["rows"][k] for k in common) / len(common), len(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run output dir (contains losses.json and eval_*.json)")
    ap.add_argument("--reference", default="outputs/renderer8_densetext_qa_seed43_4b_to_llama8b_stage2_dense03")
    ap.add_argument("--previous", default=None, help="previous chain run dir for the delta table (optional)")
    ap.add_argument("--expected-steps", type=int, required=True)
    ap.add_argument("--max-mean-drop", type=float, default=5.0, help="points of mean 1x accuracy vs reference")
    ap.add_argument("--am-collapse", type=float, default=0.4)
    ap.add_argument("--max-skip-frac", type=float, default=0.10)
    ap.add_argument("--report", default=None, help="markdown report path (default <run>/gate_report.md)")
    args = ap.parse_args()
    run = Path(args.run)
    failures, notes = [], []

    losses_path = run / "losses.json"
    if not losses_path.exists():
        failures.append("losses.json missing")
        L = []
    else:
        L = json.load(open(losses_path))
        if len(L) < args.expected_steps:
            failures.append(f"only {len(L)}/{args.expected_steps} steps logged")
        last = L[-1] if L else {}
        for k in ("loss_avg_100", "ce_loss_avg_100", "kv_loss_avg_100"):
            v = last.get(k)
            if v is not None and not math.isfinite(v):
                failures.append(f"non-finite {k} at end of training")
        skipped = sum(1 for r in L if r.get("skipped"))
        if L and skipped / len(L) > args.max_skip_frac:
            failures.append(f"{skipped}/{len(L)} steps grad-skipped")
        notes.append(f"steps={len(L)} skipped={skipped} final loss_avg_100={last.get('loss_avg_100')} "
                     f"ce_avg={last.get('ce_loss_avg_100')} dense_excess_avg={last.get('dense_excess_avg_100')} "
                     f"downstream_score={last.get('downstream_score')}")
    if not (run / f"exit-checkpoint-{args.expected_steps}" / "renderer.pt").exists():
        failures.append(f"exit-checkpoint-{args.expected_steps}/renderer.pt missing")

    rows = {}
    for name, (stem, n) in {**QUICK_1X, **AM}.items():
        e = read_eval(run / stem)
        if e is None:
            failures.append(f"eval missing/unreadable: {stem}")
            continue
        if e["seen"] != n:
            failures.append(f"{stem}: seen {e['seen']} != {n}")
        rows[name] = e
    ref = {name: read_eval(Path(args.reference) / stem) for name, (stem, _) in {**QUICK_1X, **AM}.items()}
    prev = {name: read_eval(Path(args.previous) / stem) for name, (stem, _) in {**QUICK_1X, **AM}.items()} if args.previous else {}

    deltas = []
    paired_note = {}
    for name in QUICK_1X:
        if name in rows and ref.get(name) and rows[name]["acc"] is not None and ref[name]["acc"] is not None:
            if rows[name]["seen"] != ref[name]["seen"]:
                pr = paired(rows[name], ref[name])
                if pr is None:
                    failures.append(f"{name}: {rows[name]['seen']} rows vs reference {ref[name]['seen']} and no paired rows")
                    continue
                paired_note[name] = pr
                deltas.append(100 * (pr[0] - pr[1]))
            else:
                deltas.append(100 * (rows[name]["acc"] - ref[name]["acc"]))
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    if mean_delta is not None and mean_delta < -args.max_mean_drop:
        failures.append(f"mean 1x accuracy {mean_delta:+.2f} pts vs reference (limit -{args.max_mean_drop})")
    for name, floor in FLOORS.items():
        if name in rows and rows[name]["acc"] is not None and rows[name]["acc"] < floor:
            failures.append(f"{name} 1x {rows[name]['acc']:.3f} below floor {floor}")
    for base in ("qasper", "longhealth"):
        a, b = rows.get(base), rows.get(f"{base}_am10")
        if a and b and a["acc"] and b["acc"] is not None and b["acc"] < args.am_collapse * a["acc"]:
            failures.append(f"{base} am10 {b['acc']:.3f} < {args.am_collapse} x 1x {a['acc']:.3f} (compression collapse)")

    def fmt(e, key="acc"):
        return "-" if not e or e.get(key) is None else f"{100 * e[key]:.1f}"
    lines = [f"# Gate report: {run.name}", "", f"**Verdict: {'FAIL' if failures else 'PASS'}**", ""]
    lines += [f"- {f}" for f in failures] or ["- no failure conditions triggered"]
    lines += ["", "Training: " + "; ".join(notes), "",
              f"| eval | this run | reference ({Path(args.reference).name[:24]}) | delta | previous ({Path(args.previous).name[:24] if args.previous else '-'}) | delta | time |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name in list(QUICK_1X) + list(AM):
        e, r, p = rows.get(name), ref.get(name), prev.get(name)
        d = f"{100 * (e['acc'] - r['acc']):+.1f}" if e and r and e.get("acc") is not None and r.get("acc") is not None else "-"
        if name in paired_note:
            pa, pb, n_common = paired_note[name]
            d = f"{100 * (pa - pb):+.1f} (paired n={n_common}: {100 * pa:.1f} vs {100 * pb:.1f})"
        dp = f"{100 * (e['acc'] - p['acc']):+.1f}" if e and p and e.get("acc") is not None and p.get("acc") is not None else "-"
        f1 = f" (F1 {fmt(e, 'f1')})" if e and e.get("f1") is not None else ""
        lines.append(f"| {name} | {fmt(e)}{f1} | {fmt(r)} | {d} | {fmt(p)} | {dp} | {e['elapsed'] if e else '-'} |")
    if mean_delta is not None:
        lines += ["", f"Mean 1x delta vs reference over {len(deltas)} tasks: **{mean_delta:+.2f}** points"]
    report = Path(args.report) if args.report else run / "gate_report.md"
    report.write_text("\n".join(lines) + "\n")
    json.dump({"run": str(run), "verdict": "FAIL" if failures else "PASS", "failures": failures, "mean_delta_vs_reference": mean_delta,
               "rows": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in rows.items()}},
              open(run / "gate_report.json", "w"), indent=1)
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
