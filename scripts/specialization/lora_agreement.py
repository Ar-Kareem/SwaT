import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "multiagent/outputs_mine/specialization/lora"
ROWS = {"openbookqa": "mcq", "medqa": "mcq", "arc": "mcq", "medmcqa": "mcq", "triviaqa": "free", "nq-open": "free"}
PAIRS = {"14E_qwen3_4b_to_llama3_8b": "llama3-8b-instruct", "15B_qwen3_4b_to_gemma2_9b": "gemma-2-9b-it"}


def load(path):
    path = ROOT / path
    return {r["id"]: r for r in json.loads(path.read_text())["rows"]} if path.exists() else None


def answer_key(row, kind):
    if kind == "mcq":
        m = re.search(r"\b([A-E])\b", row["output"].upper())
        return m.group(1) if m else None
    return " ".join(row["output"].lower().strip().strip(".").split())


def pct(rows, ids):
    return f"{100 * sum(rows[i]['grade'] == 'correct' for i in ids) / len(ids):.0f} ({len(ids)})" if ids else "-"


def main():
    header = ["row", "pair", "rend acc: only sender right", "rend acc: only receiver right",
              "answer == sender", "answer == receiver", "LoRA-fixed q: rend+LoRA", "LoRA-fixed q: rend base"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row, kind in ROWS.items():
        S, SL = load(f"vanilla/qwen3-4b/{row}.json"), load(f"vanilla_lora/qwen3-4b/{row}.json")
        for pair, recv in PAIRS.items():
            R, X, XL = load(f"vanilla/{recv}/{row}.json"), load(f"renderer/{pair}/{row}.json"), load(f"renderer_lora/{pair}/{row}.json")
            if not (S and R and X):
                continue
            ok = lambda D, i: D[i]["grade"] == "correct"
            ids = [i for i in S if i in R and i in X]
            only_s = [i for i in ids if ok(S, i) and not ok(R, i)]
            only_r = [i for i in ids if ok(R, i) and not ok(S, i)]
            same_s = sum(answer_key(X[i], kind) == answer_key(S[i], kind) for i in ids) / len(ids)
            same_r = sum(answer_key(X[i], kind) == answer_key(R[i], kind) for i in ids) / len(ids)
            fixed = [i for i in ids if SL and XL and i in SL and i in XL and ok(SL, i) and not ok(S, i)]
            lines.append("| " + " | ".join([row, pair[:3], pct(X, only_s), pct(X, only_r), f"{100 * same_s:.0f}",
                                            f"{100 * same_r:.0f}", pct(XL, fixed) if fixed else "-",
                                            pct(X, fixed) if fixed else "-"]) + " |")
    md = "Accuracy in % with question counts in parentheses; answer agreement in %.\n\n" + "\n".join(lines) + "\n"
    (ROOT / "summary").mkdir(exist_ok=True)
    (ROOT / "summary/agreement.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
