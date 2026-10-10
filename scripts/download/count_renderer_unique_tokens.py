import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO / "multiagent")

from transformers import AutoTokenizer
from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.dataset.c2c_registry import load_c2c_dataset_specs

OLD_TEXT = "data/renderer_text/pg19_wikitext103_train_4k4k_20m.jsonl"
OLD_DENSE = ["data/renderer_text/pg19_train_4k4k_20m.jsonl", "data/renderer_text/wikitext103_train_4k4k_10m.jsonl"]
POOL_A = [OLD_TEXT, "qasper+nox:train", "quality:train", "hotpotqa:train", "musique:train"]


def pool_e(traces):
    return [OLD_TEXT, "qasper+nox:train", "quality+limit=2500:train", "hotpotqa+limit=2500:train", "musique+limit=2500:train",
            "arc-c:train", "hellaswag+limit=3000:train", "winogrande+limit=3000:train", "mmlu+limit=3000:auxiliary_train", traces]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", nargs="+", default=["Qwen3-4B", "llama3-8b-instruct"])
    ap.add_argument("--traces", default="data/phase2_traces/llama3-8b-instruct_gsm8k_train_clean.jsonl")
    ap.add_argument("--context-cap", type=int, default=8192)
    ap.add_argument("--target-cap", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = {"context": {}, "prompt": {}, "target": {}}
    per_source = {}

    def add(kind, text, src):
        text = str(text)
        if not text:
            return
        texts[kind].setdefault(text, set()).add(src)

    stages = [("stage1", POOL_A, 2500), ("stage2", pool_e(args.traces), 7500), ("dense", OLD_DENSE, 7500)]
    t0 = time.time()
    for stage, specs, limit in stages:
        for spec in specs:
            ds = load_c2c_dataset_specs([spec], include_answer_message=False, limit_per_dataset=limit)
            src = f"{stage}:{spec}"
            per_source[src] = len(ds)
            for x in ds:
                for m in (x.sender_messages or []):
                    add("context", m.get("content", ""), src)
                if stage != "dense":
                    for m in (x.receiver_messages or []):
                        if x.prompt_mode != "same_prompt_continuation":
                            add("prompt", m.get("content", ""), src)
                    add("target", x.target_text, src)
            print(f"{src}: {len(ds)} samples, uniques so far ctx={len(texts['context'])} prompt={len(texts['prompt'])} tgt={len(texts['target'])}, {time.time()-t0:.0f}s", flush=True)

    report = {"args": vars(args), "samples_per_source": per_source,
              "unique_strings": {k: len(v) for k, v in texts.items()}, "tokenizers": {}}
    caps = {"context": args.context_cap, "prompt": args.context_cap, "target": args.target_cap}
    for name in args.tokenizers:
        tok = AutoTokenizer.from_pretrained(resolve_model_path(name))
        rep = {}
        total = total_uncapped = 0
        for kind, d in texts.items():
            strings = list(d.keys())
            lens = []
            for i in range(0, len(strings), 256):
                lens += [len(ids) for ids in tok(strings[i:i + 256], add_special_tokens=False)["input_ids"]]
            capped = [min(n, caps[kind]) for n in lens]
            by_src = {}
            for s, n, c in zip(strings, lens, capped):
                for src in d[s]:
                    e = by_src.setdefault(src, {"strings": 0, "tokens_capped": 0})
                    e["strings"] += 1; e["tokens_capped"] += c
            rep[kind] = {"unique_strings": len(strings), "tokens_uncapped": sum(lens), "tokens_capped": sum(capped),
                         "n_truncated": sum(1 for n in lens if n > caps[kind]), "by_source_nonexclusive": by_src}
            total += sum(capped); total_uncapped += sum(lens)
            print(f"[{name}] {kind}: {len(strings)} unique strings, {sum(capped):,} tokens (capped), {sum(lens):,} uncapped, {time.time()-t0:.0f}s", flush=True)
        rep["total_unique_tokens_capped"] = total
        rep["total_unique_tokens_uncapped"] = total_uncapped
        report["tokenizers"][name] = rep
        print(f"[{name}] TOTAL unique tokens: {total:,} (capped) / {total_uncapped:,} (uncapped)", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
