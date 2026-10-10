import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "multiagent/outputs_mine"
FULL = M / "specialization"
OUT = FULL / "summary"
LANGS = ["mmlu", "mmmlu-ar", "mmmlu-bn", "mmmlu-de", "mmmlu-es", "mmmlu-fr", "mmmlu-hi", "mmmlu-id", "mmmlu-it",
         "mmmlu-ja", "mmmlu-ko", "mmmlu-pt", "mmmlu-sw", "mmmlu-yo", "mmmlu-zh", "cmmlu"]
PAIRS = {
    "14E_qwen3_4b_to_llama3_8b": ("qwen3-4b", "llama3-8b-instruct"),
    "15B_qwen3_4b_to_gemma2_9b": ("qwen3-4b", "gemma-2-9b-it"),
    "15D_ministral3_8b_to_llama32_11b": ("ministral-3-8b-instruct", "llama3.2-11b-vision-instruct"),
    "15E_gemma3_4b_to_granite33_8b": ("gemma-3-4b-it", "granite-3.3-8b-instruct"),
    "15G_gemma3_4b_to_llama3_8b": ("gemma-3-4b-it", "llama3-8b-instruct"),
    "15H_gemma3_12b_to_qwen3_4b": ("gemma-3-12b-it", "qwen3-4b"),
    "exp1_granite33_8b_to_smollm3_3b": ("granite-3.3-8b-instruct", "smollm3-3b"),
}


def load_rows(arm_dir: Path):
    files = sorted(arm_dir.glob("*.json")) if arm_dir.exists() else []
    task = defaultdict(lambda: [0, 0])
    subject = defaultdict(lambda: [0, 0])
    complete = {}
    for f in files:
        d = json.loads(f.read_text())
        n = d.get("result_overview", {}).get("samples_evaluated", 0)
        if d.get("elapsed_seconds") is None or len(d.get("rows", [])) != n:
            print(f"  incomplete: {f}", file=sys.stderr)
            continue
        for r in d["rows"]:
            t = r["task"]
            ok = r["grade"] == "correct"
            task[t][0] += ok
            task[t][1] += 1
            subject[(t, r.get("subject", ""))][0] += ok
            subject[(t, r.get("subject", ""))][1] += 1
        for t in {r["task"] for r in d["rows"]}:
            complete[t] = task[t][1] >= (11582 if t == "cmmlu" else 14042)
    return task, subject, complete


def pct(c, n):
    return 100.0 * c / n if n else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {}
    for pair, (sender, receiver) in PAIRS.items():
        arms[("renderer", pair)] = load_rows(FULL / "renderer" / pair)
        for v in (sender, receiver):
            if ("vanilla", v) not in arms:
                arms[("vanilla", v)] = load_rows(FULL / "vanilla" / v)
    rows, md = [], []
    md.append("| pair | arm | " + " | ".join(LANGS) + " | mean(14 MMMLU) |")
    md.append("|---|---|" + "---:|" * (len(LANGS) + 1))
    for pair, (sender, receiver) in PAIRS.items():
        for label, key in (("vanilla sender " + sender, ("vanilla", sender)), ("vanilla receiver " + receiver, ("vanilla", receiver)), ("renderer", ("renderer", pair))):
            task, _, complete = arms[key]
            vals = {t: pct(*task[t]) if t in task else None for t in LANGS}
            mm = [vals[t] for t in LANGS if t.startswith("mmmlu-") and vals[t] is not None]
            mean = sum(mm) / len(mm) if len(mm) == 14 else None
            rows.append({"pair": pair, "arm": label, **{t: (f"{v:.4f}" if v is not None else "") for t, v in vals.items()},
                         "mean_mmmlu14": f"{mean:.4f}" if mean is not None else "",
                         "incomplete": ",".join(t for t in LANGS if t in task and not complete.get(t, False))})
            md.append(f"| {pair} | {label} | " + " | ".join((f"{vals[t]:.1f}" if vals[t] is not None else "-") for t in LANGS)
                      + f" | {mean:.1f} |" if mean is not None else f"| {pair} | {label} | " + " | ".join((f"{vals[t]:.1f}" if vals[t] is not None else "-") for t in LANGS) + " | - |")
    with (OUT / "specialization_full.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "specialization_full.md").write_text("\n".join(md) + "\n")
    with (OUT / "per_subject.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "task", "subject", "n", "accuracy"])
        for key, (_, subject, _) in arms.items():
            for (t, s), (c, n) in sorted(subject.items()):
                w.writerow([f"{key[0]}/{key[1]}", t, s, n, f"{pct(c, n):.2f}"])
    print("\n".join(md))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
