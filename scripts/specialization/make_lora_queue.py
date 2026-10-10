import json
import sys
from pathlib import Path

PY = "../env3.10/bin/python -u"
OUT = "outputs_mine/specialization/lora"
COMMON = "--seed 42 --max-new-tokens 64 --save-every 1000 --print-every 2000"

BENCHMARKS = {
    "medqa": (["medqa:train"], "medqa:test", 3, 4, 8),
    "openbookqa": (["openbookqa:train"], "openbookqa:test", 3, 8, 4),
    "arc": (["arc-c:train", "arc-e:train"], "arc-c:test", 3, 8, 4),
    "medmcqa": (["medmcqa:train"], "medmcqa:validation", 1, 8, 4),
}
VANILLA = {"llama3-8b-instruct": 48, "gemma-2-9b-it": 24, "qwen3-4b": 48}
RENDERERS = {
    "14E_qwen3_4b_to_llama3_8b": "outputs_mine/swat/14E/stage2/exit-checkpoint-24000",
    "15B_qwen3_4b_to_gemma2_9b": "outputs_mine/swat/15B/stage2/exit-checkpoint-24000",
}


CLOSEDBOOK = {
    "triviaqa": (["triviaqa:train"], "triviaqa:validation", 2, 8, 4),
    "nq-open": (["nq-open:train"], "nq-open:validation", 2, 32, 1),
}
CB_COMMON = "--seed 42 --max-new-tokens 32 --save-every 1000 --print-every 2000"


def train_cmd(name, tasks, epochs, batch, accum, max_length=768):
    return (f"{PY} train/train_mcq_lora.py --model qwen3-4b --tasks {' '.join(tasks)} --epochs {epochs} "
            f"--batch-size {batch} --grad-accum {accum} --rank 16 --alpha 32 --lr 2e-4 --max-length {max_length} "
            f"--seed 42 --device cuda:0 --output {OUT}/adapters/{name}")


def vanilla_cmd(model, task, out, adapter=None, common=COMMON):
    bs = VANILLA[model]
    extra = f" --adapter {adapter}" if adapter else ""
    return (f"{PY} evaluation/vanilla.py --model {model}{extra} --tasks {task} --device cuda:0 {common} "
            f"--batch-refill --batch-size {bs} --chunk-size {2 * bs} --output {out}")


def renderer_cmd(ckpt, task, out, adapter=None, common=COMMON):
    extra = f" --sender-adapter {adapter}" if adapter else ""
    return (f"{PY} evaluation/renderer.py --renderer {ckpt} --renderer-decode-prefix generation{extra} --tasks {task} "
            f"--device cuda:0 --max-length 2048 --context-cache-max 4 {common} --batch-size 12 --chunk-size 24 "
            f"--output {out}")


def closedbook_jobs():
    jobs = []
    for name, (tasks, _, epochs, batch, accum) in CLOSEDBOOK.items():
        jobs.append({"cmd": train_cmd(name, tasks, epochs, batch, accum, max_length=256), "name": f"train_{name}"})
    for name, (_, eval_task, *_rest) in CLOSEDBOOK.items():
        for model in VANILLA:
            jobs.append(vanilla_cmd(model, eval_task, f"{OUT}/vanilla/{model}/{name}.json", common=CB_COMMON))
        for pair, ckpt in RENDERERS.items():
            jobs.append(renderer_cmd(ckpt, eval_task, f"{OUT}/renderer/{pair}/{name}.json", common=CB_COMMON))
    for name, (_, eval_task, *_rest) in CLOSEDBOOK.items():
        adapter, after = f"{OUT}/adapters/{name}", [f"train_{name}"]
        jobs.append({"cmd": vanilla_cmd("qwen3-4b", eval_task, f"{OUT}/vanilla_lora/qwen3-4b/{name}.json", adapter,
                                        common=CB_COMMON), "after": after})
        for pair, ckpt in RENDERERS.items():
            jobs.append({"cmd": renderer_cmd(ckpt, eval_task, f"{OUT}/renderer_lora/{pair}/{name}.json", adapter,
                                             common=CB_COMMON), "after": after})
    return jobs


def main():
    if sys.argv[1:] == ["closedbook"]:
        jobs = closedbook_jobs()
        path = Path(__file__).resolve().parents[2] / "multiagent/gpu_queue_commands/gpu_queue_commands_spec_lora_closedbook.json"
        path.write_text(json.dumps(jobs, indent=1))
        print(f"wrote {len(jobs)} jobs to {path}")
        return
    jobs = []
    for name, (tasks, _, epochs, batch, accum) in BENCHMARKS.items():
        jobs.append({"cmd": train_cmd(name, tasks, epochs, batch, accum), "name": f"train_{name}"})
    jobs.append("# base arms (no adapter)")
    for name, (_, eval_task, *_rest) in BENCHMARKS.items():
        for model in VANILLA:
            jobs.append(vanilla_cmd(model, eval_task, f"{OUT}/vanilla/{model}/{name}.json"))
        for pair, ckpt in RENDERERS.items():
            jobs.append(renderer_cmd(ckpt, eval_task, f"{OUT}/renderer/{pair}/{name}.json"))
    jobs.append("# adapter arms (small adapters first so medmcqa training does not block them)")
    for name in ["openbookqa", "arc", "medqa", "medmcqa"]:
        eval_task = BENCHMARKS[name][1]
        adapter = f"{OUT}/adapters/{name}"
        after = [f"train_{name}"]
        jobs.append({"cmd": vanilla_cmd("qwen3-4b", eval_task, f"{OUT}/vanilla_lora/qwen3-4b/{name}.json", adapter),
                     "after": after})
        for pair, ckpt in RENDERERS.items():
            jobs.append({"cmd": renderer_cmd(ckpt, eval_task, f"{OUT}/renderer_lora/{pair}/{name}.json", adapter),
                         "after": after})
    path = Path(__file__).resolve().parents[2] / "multiagent/gpu_queue_commands/gpu_queue_commands_spec_lora.json"
    path.write_text(json.dumps(jobs, indent=1))
    print(f"wrote {len([j for j in jobs if not (isinstance(j, str) and j.startswith('#'))])} jobs to {path}")


if __name__ == "__main__":
    main()
