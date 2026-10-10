from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
TURN_END_TOKENS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>")


def stop_token_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    unk = tokenizer.unk_token_id
    for token in TURN_END_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != unk and tid >= 0:
            ids.add(int(tid))
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    args = parser.parse_args()
    req = json.loads(args.requests.read_text())
    pending = req['requests']
    answers = json.loads(args.answers.read_text()) if args.answers.exists() else {}
    pending = [r for r in pending if r['key'] not in answers]
    print(f"{args.requests.name}: {len(pending)} prompts to answer ({len(answers)} already done)", flush=True)
    if not pending:
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(req['model'])
    over = [r['key'] for r in pending if len(tokenizer(r['prompt'], add_special_tokens=False)['input_ids']) > req['max_length'] - req['max_new_tokens']]
    if over:
        raise ValueError(f"{len(over)} prompts exceed max_length - max_new_tokens; the HF path would truncate them")
    llm = LLM(model=req['model'], dtype='bfloat16', max_model_len=req['max_length'], seed=req.get('seed', 0),
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=False)
    params = SamplingParams(temperature=0.0, max_tokens=req['max_new_tokens'], stop_token_ids=stop_token_ids(tokenizer))
    started = time.time()
    outputs = llm.generate([r['prompt'] for r in pending], params)
    print(f"generated {len(outputs)} answers in {(time.time() - started) / 60:.1f} min", flush=True)
    for r, out in zip(pending, outputs):
        gen = out.outputs[0]
        answers[r['key']] = {'text': gen.text.strip(), 'finish_reason': gen.finish_reason,
                             'generated_tokens': len(gen.token_ids)}
    tmp = args.answers.with_suffix(args.answers.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(answers, ensure_ascii=False))
    tmp.replace(args.answers)


if __name__ == '__main__':
    main()
