import argparse, json, os, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO / "multiagent")

from transformers import AutoTokenizer
from multiagent.model.vanilla_loader import resolve_model_path
from multiagent.dataset.c2c_registry import load_c2c_dataset_specs
from multiagent.dataset.c2c_cross_family import CrossFamilyPhase1Collator
from multiagent.train.train_cache_renderer import RendererNextTokenCollator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--trace-length", type=int, default=4096)
    ap.add_argument("--dense-max-length", type=int, default=0, help="0 = skip dense encoding")
    ap.add_argument("--stride", type=int, default=1, help="encode every k-th sample (speed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stok = AutoTokenizer.from_pretrained(resolve_model_path(args.sender))
    rtok = AutoTokenizer.from_pretrained(resolve_model_path(args.receiver))
    ce = RendererNextTokenCollator(stok, rtok, max_source_length=args.max_length, max_trace_length=args.trace_length,
                                   device="cpu", receiver_device="cpu")
    dense = None
    if args.dense_max_length:
        dense = CrossFamilyPhase1Collator(stok, rtok, max_length=args.dense_max_length, device="cpu",
                                          receiver_device="cpu", include_repeat=False)

    report = {"args": vars(args), "datasets": {}}
    t0 = time.time()
    for spec in args.datasets:
        ds = load_c2c_dataset_specs([spec], include_answer_message=False, limit_per_dataset=args.limit)
        n = len(ds)
        rows = []
        for i in range(0, n, args.stride):
            r = ce.encode(ds[i])
            if r is None:
                rows.append(None); continue
            row = {"sender_ctx": len(r["sender_ids"]), "receiver_ctx": len(r["receiver_ids"]),
                   "decode": len(r["decode_ids"]), "supervised": sum(1 for x in r["labels"] if x != -100)}
            if dense is not None:
                d = dense.encode(ds[i])
                row["dense_sender_ctx"] = len(d["sender_ids"]) if d else 0
                row["dense_receiver_ctx"] = len(d["receiver_ids"]) if d else 0
            rows.append(row)
            if len(rows) % 200 == 0:
                print(f"{spec}: {len(rows)} encoded, {time.time()-t0:.0f}s", flush=True)
        valid = [r for r in rows if r]
        keys = list(valid[0].keys()) if valid else []
        stats = {"n_pool": n, "n_encoded": len(valid), "n_skipped": len(rows) - len(valid)}
        for k in keys:
            vals = [r[k] for r in valid]
            stats[k] = {"mean": statistics.fmean(vals), "median": statistics.median(vals), "max": max(vals),
                        "min": min(vals), "sum_encoded": sum(vals),
                        "sum_pool_est": statistics.fmean(vals) * n}
        report["datasets"][spec] = stats
        print(spec, json.dumps({k: (round(v["mean"], 1) if isinstance(v, dict) else v) for k, v in stats.items()}), flush=True)

    pool = {}
    tot_n = sum(s["n_pool"] for s in report["datasets"].values())
    for k in keys:
        pool[k] = sum(s[k]["mean"] * s["n_pool"] for s in report["datasets"].values()) / tot_n
    report["pool"] = {"n_pool": tot_n, "per_step_mean": pool}
    print("POOL", tot_n, json.dumps({k: round(v, 1) for k, v in pool.items()}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
