import csv
import json
import re
from pathlib import Path

M = Path(__file__).resolve().parents[2] / "multiagent/outputs_mine/specialization"
ROOT = M / "large_small"
ROWS = {"medqa": "mcq", "medmcqa": "mcq", "openbookqa": "mcq", "arc": "mcq", "triviaqa": "free", "nq-open": "free"}
PAIRS = {
    "15H_gemma3_12b_to_qwen3_4b": (ROOT / "vanilla/gemma-3-12b-it", M / "lora/vanilla/qwen3-4b", "Gemma 3 12B -> Qwen3-4B"),
    "exp1_granite33_8b_to_smollm3_3b": (ROOT / "vanilla/granite-3.3-8b-instruct", ROOT / "vanilla/smollm3-3b",
                                        "Granite 3.3 8B -> SmolLM3 3B"),
}


RECEIVER_OVERRIDE = {("exp1_granite33_8b_to_smollm3_3b", row): ROOT / "vanilla/smollm3-3b_512tok"
                     for row in ("medqa", "medmcqa", "openbookqa", "arc")}


def subspan(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())["result_overview"]["metrics_percent"].get("subspan_em")


FALLBACK = re.compile(r"\*\*\s*\(?([A-E])[\.\):]|\bis\s+\(?([A-E])[\.\):]")


def regrade(row: dict) -> dict:
    if row.get("grade") != "fail" or "labels" not in row:
        return row
    m = FALLBACK.search(row["output"])
    letter = next((g for g in m.groups() if g), None) if m else None
    return {**row, "grade": "correct" if letter == row["answer"] else "incorrect"}


def load(path: Path):
    return {r["id"]: regrade(r) for r in json.loads(path.read_text())["rows"]} if path.exists() else None


def acc(rows):
    return 100 * sum(r["grade"] == "correct" for r in rows.values()) / len(rows) if rows else None


def cond(X, ids):
    return f"{100 * sum(X[i]['grade'] == 'correct' for i in ids) / len(ids):.0f} ({len(ids)})" if ids else "-"


def main():
    header = ["pair", "row", "n", "sender", "receiver", "renderer", "gain_over_receiver", "gap_closed",
              "rend|only sender right", "rend|only receiver right", "subspanEM sender/receiver/renderer"]
    records = []
    for pair, (sdir, rdir, label) in PAIRS.items():
        for row, kind in ROWS.items():
            rd = RECEIVER_OVERRIDE.get((pair, row), rdir)
            if not (rd / f"{row}.json").exists():
                rd = rdir
            S, R, X = load(sdir / f"{row}.json"), load(rd / f"{row}.json"), load(ROOT / f"renderer/{pair}/{row}.json")
            s, r, x = acc(S), acc(R), acc(X)
            gain = x - r if x is not None and r is not None else None
            gap = (x - r) / (s - r) if None not in (s, r, x) and abs(s - r) > 1e-9 else None
            only_s = only_r = []
            if S and R and X:
                ids = [i for i in X if i in S and i in R]
                ok = lambda D, i: D[i]["grade"] == "correct"
                only_s = [i for i in ids if ok(S, i) and not ok(R, i)]
                only_r = [i for i in ids if ok(R, i) and not ok(S, i)]
            f = lambda v: "-" if v is None else f"{v:.1f}"
            records.append([label, row, len(X) if X else "-", f(s), f(r), f(x), "-" if gain is None else f"{gain:+.1f}",
                            "-" if gap is None else f"{gap:.2f}", cond(X, only_s) if X else "-", cond(X, only_r) if X else "-",
                            "/".join(f(subspan(d / f"{row}.json")) for d in (sdir, rd, ROOT / f"renderer/{pair}"))
                            if kind == "free" else "-"])
    out = ROOT / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "large_small.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(records)
    md = "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                   ["| " + " | ".join(str(c) for c in rec) + " |" for rec in records]) + "\n"
    (out / "large_small.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
