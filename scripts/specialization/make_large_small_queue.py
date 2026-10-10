import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagent.paths import MODELS_ROOT

PY = "../env3.10/bin/python -u evaluation/renderer.py"
RECEIVER_PY = "../env3.10/bin/python -u evaluation/vanilla.py"
OUT = "outputs_mine/specialization/large_small"
ROWS = {
    "medqa": ("medqa:test", 64),
    "medmcqa": ("medmcqa:validation", 64),
    "openbookqa": ("openbookqa:test", 64),
    "arc": ("arc-c:test", 64),
    "triviaqa": ("triviaqa:validation", 32),
    "nq-open": ("nq-open:validation", 32),
}
VANILLA = {
    "gemma-3-12b-it": ("gemma-3-12b-it", 16),
    "granite-3.3-8b-instruct": ("granite-3.3-8b-instruct", 32),
    "smollm3-3b": (str(MODELS_ROOT / "HuggingFaceTB/SmolLM3-3B"), 48),
}
RENDERERS = {
    "15H_gemma3_12b_to_qwen3_4b": ("outputs_mine/swat/15H/stage2/checkpoint-24000", 8),
    "exp1_granite33_8b_to_smollm3_3b": ("outputs_mine/swat/granite_smollm3/stage2/exit-checkpoint-24000", 12),
}


def common(tokens):
    return f"--seed 42 --max-new-tokens {tokens} --save-every 1000 --print-every 2000"


def main():
    jobs = []
    order = ["triviaqa", "medmcqa", "nq-open", "medqa", "arc", "openbookqa"]
    for row in order:
        task, tokens = ROWS[row]
        for pair, (ckpt, bs) in RENDERERS.items():
            jobs.append(f"{PY} --renderer {ckpt} --renderer-decode-prefix generation --tasks {task} --device cuda:0 "
                        f"--max-length 2048 --context-cache-max 4 {common(tokens)} --batch-size {bs} --chunk-size {2 * bs} "
                        f"--output {OUT}/renderer/{pair}/{row}.json")
        for key, (model, bs) in VANILLA.items():
            jobs.append(f"{RECEIVER_PY} --model {model} --tasks {task} --device cuda:0 {common(tokens)} --batch-refill "
                        f"--batch-size {bs} --chunk-size {2 * bs} --output {OUT}/vanilla/{key}/{row}.json")
    path = Path(__file__).resolve().parents[2] / "multiagent/gpu_queue_commands/gpu_queue_commands_spec_large_small.json"
    path.write_text(json.dumps(jobs, indent=1))
    print(f"wrote {len(jobs)} jobs to {path}")


if __name__ == "__main__":
    main()
