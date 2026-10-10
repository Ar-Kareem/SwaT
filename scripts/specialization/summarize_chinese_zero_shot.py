import argparse
import json
from collections import Counter
from pathlib import Path


def rows_by_subject(path: Path):
    data = json.loads(path.read_text())
    rows = sorted(data["rows"], key=lambda r: r["index"])
    per_subject = {}
    for row in rows:
        per_subject.setdefault(row["subject"], []).append(row)
    return rows, per_subject


def identity(row):
    return (row["id"], row["question"], tuple(row["choices"]), row["answer"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer-label", default="Renderer14E")
    args = parser.parse_args()
    renderer_rows, renderer = rows_by_subject(args.renderer)
    llama_rows, llama = rows_by_subject(args.baseline_dir / "cmmlu_llama3_8b_instruct.json")
    qwen_rows, qwen = rows_by_subject(args.baseline_dir / "cmmlu_qwen3_4b.json")
    if not (len(renderer_rows) == len(llama_rows) == len(qwen_rows)):
        raise ValueError("The three arms have different row counts")
    if not ({identity(r) for r in renderer_rows} == {identity(r) for r in llama_rows} == {identity(r) for r in qwen_rows}):
        raise ValueError("The three arms do not share identical question IDs, questions, choices and answers")

    def accuracy(rows):
        return 100 * sum(r["grade"] == "correct" for r in rows) / len(rows)

    def entry(subject, r_rows, l_rows, q_rows):
        return dict(subject=subject, n=len(r_rows), llama=accuracy(l_rows), qwen=accuracy(q_rows),
                    renderer=accuracy(r_rows), qwen_gap_pp=accuracy(q_rows) - accuracy(l_rows),
                    renderer_gap_pp=accuracy(r_rows) - accuracy(l_rows),
                    renderer_grades=dict(Counter(r["grade"] for r in r_rows)),
                    renderer_cutoffs=sum(str(r["stop_reason"]).startswith("cutoff") for r in r_rows))

    summary = [entry("ALL", renderer_rows, llama_rows, qwen_rows)]
    subjects = [entry(s, renderer[s], llama[s], qwen[s]) for s in renderer]
    subjects.sort(key=lambda e: (-e["qwen_gap_pp"], e["subject"]))
    summary += subjects
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# {args.renderer_label} zero-shot CMMLU evaluation", "",
             "Full CMMLU test set; question supplied through the rendered cache only.",
             "Accuracy is percent; gaps are absolute percentage points over vanilla Llama.",
             "Vanilla scores reuse the saved Qwen3-4B and Llama-3-8B-Instruct evaluations.",
             "All three arms have identical question IDs, questions, choices, and answers.", "",
             f"| Subject | N | Llama | Qwen | Qwen gap | {args.renderer_label} | Renderer gap | Cutoffs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in summary:
        lines.append(f"| {e['subject']} | {e['n']} | {e['llama']:.1f} | {e['qwen']:.1f} | {e['qwen_gap_pp']:+.1f} | "
                     f"{e['renderer']:.1f} | {e['renderer_gap_pp']:+.1f} | {e['renderer_cutoffs']} |")
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"{len(summary) - 1} subjects, n={summary[0]['n']}: renderer {summary[0]['renderer']:.1f} "
          f"vs Llama {summary[0]['llama']:.1f} / Qwen {summary[0]['qwen']:.1f}")


if __name__ == "__main__":
    main()
