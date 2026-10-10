import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from multiagent.paths import DATASETS_ROOT

ROOT = REPO / "multiagent/outputs_mine/specialization/lora"
ROWS = ["medqa", "medmcqa", "openbookqa", "arc", "triviaqa", "nq-open"]
CLOSEDBOOK = {"triviaqa": "triviaqa", "nq-open": "nq"}
OVERLAP_DIR = DATASETS_ROOT / "qa_overlap"
GROUPS = {
    "question_overlap": lambda labels: "question_overlap" in labels,
    "answer_overlap_only": lambda labels: "answer_overlap_only" in labels,
    "no_overlap": lambda labels: "no_answer_overlap" in labels and "no_question_overlap" in labels,
    "answer_overlap(all)": lambda labels: "answer_overlap" in labels,
    "no_answer_overlap(all)": lambda labels: "no_answer_overlap" in labels,
}
PAIRS = {"14E_qwen3_4b_to_llama3_8b": ("llama3-8b-instruct", "Llama 3 8B"),
         "15B_qwen3_4b_to_gemma2_9b": ("gemma-2-9b-it", "Gemma 2 9B")}


def acc(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    ov = d.get("result_overview", {})
    if "accuracy_percent" in ov:
        return ov["accuracy_percent"], ov.get("samples_evaluated")
    s = d.get("summary", d).get("all", {})
    return 100 * s.get("accuracy", float("nan")), s.get("seen")


def norm_q(text: str) -> str:
    return " ".join(str(text).lower().split())


def overlap_labels(prefix: str) -> dict[str, set[str]]:
    questions = [line.split("\t")[0] for line in open(OVERLAP_DIR / f"{prefix}-test.qa.csv")]
    labels = [set(json.loads(line)["labels"]) for line in open(OVERLAP_DIR / f"{prefix}-annotations.jsonl")]
    return {norm_q(q): l for q, l in zip(questions, labels)}


def group_acc(path: Path, labels: dict[str, set[str]]):
    if not path.exists():
        return None
    rows = json.loads(path.read_text())["rows"]
    out = {}
    for group, test in GROUPS.items():
        hits = [r["grade"] == "correct" for r in rows if norm_q(r["question"]) in labels and test(labels[norm_q(r["question"])])]
        out[group] = (100 * sum(hits) / len(hits), len(hits)) if hits else None
    return out


def overlap_table() -> str:
    arms = [("sender_base", "vanilla/qwen3-4b"), ("sender_lora", "vanilla_lora/qwen3-4b")]
    for pair, (recv_model, recv_name) in PAIRS.items():
        short = recv_name.split()[0]
        arms += [(f"{short}_alone", f"vanilla/{recv_model}"), (f"{short}_renderer_base", f"renderer/{pair}"),
                 (f"{short}_renderer_lora", f"renderer_lora/{pair}")]
    header = ["row", "group", "n"] + [name for name, _ in arms]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row, prefix in CLOSEDBOOK.items():
        labels = overlap_labels(prefix)
        results = [group_acc(ROOT / f"{sub}/{row}.json", labels) for _, sub in arms]
        for group in GROUPS:
            cells = [r[group] if r else None for r in results]
            n = next((c[1] for c in cells if c), "-")
            lines.append("| " + " | ".join([row, group, str(n)] + ["-" if c is None else f"{c[0]:.1f}" for c in cells]) + " |")
    return "\n".join(lines) + "\n"


def fmt(v):
    return "-" if v is None else f"{v[0]:.1f}"


def main():
    out_dir = ROOT / "summary"
    out_dir.mkdir(exist_ok=True)
    header = ["row", "n", "sender_base", "sender_lora", "sender_gain",
              "receiver", "pair", "renderer_base", "renderer_lora", "renderer_gain", "transfer_frac"]
    records = []
    for row in ROWS:
        sb = acc(ROOT / f"vanilla/qwen3-4b/{row}.json")
        sl = acc(ROOT / f"vanilla_lora/qwen3-4b/{row}.json")
        for pair, (recv_model, recv_name) in PAIRS.items():
            rv = acc(ROOT / f"vanilla/{recv_model}/{row}.json")
            rb = acc(ROOT / f"renderer/{pair}/{row}.json")
            rl = acc(ROOT / f"renderer_lora/{pair}/{row}.json")
            n = next((x[1] for x in (rv, sb, rb) if x), None)
            sgain = sl[0] - sb[0] if sb and sl else None
            rgain = rl[0] - rb[0] if rb and rl else None
            frac = rgain / sgain if sgain and rgain is not None and abs(sgain) > 1e-9 else None
            records.append([row, n, fmt(sb), fmt(sl), "-" if sgain is None else f"{sgain:+.1f}",
                            fmt(rv), recv_name, fmt(rb), fmt(rl), "-" if rgain is None else f"{rgain:+.1f}",
                            "-" if frac is None else f"{frac:.2f}"])
    with open(out_dir / "lora.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(records)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in records]
    md = "\n".join(lines) + "\n"
    md += "\nClosed-book rows by train/test overlap (Lewis et al. 2021 labels; EM %):\n\n" + overlap_table()
    (out_dir / "lora.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
