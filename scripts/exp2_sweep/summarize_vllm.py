from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')

SUMMARY_PROMPTS = {
    "summarize": "Summarize the following text:\n\n{article_text}\n\nSummary:",
    "summarize_indepth": "Summarize the following text in depth:\n\n{article_text}\n\nSummary:",
    "summarize_keypoints_questions": (
        "Summarize the following text, providing all key points that might be necessary "
        "to answer comprehension questions:\n\n{article_text}\n\nSummary:"
    ),
    "summarize_concise": "Summarize the following text concisely:\n\n{article_text}\n\nSummary:",
    "summarize_very_concise": "Summarize the following text very concisely:\n\n{article_text}\n\nSummary:",
}
TURN_END_TOKENS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def preview(text, edge=100):
    return text if len(text) <= 2 * edge else text[:edge] + ' [...] ' + text[-edge:]


def chat_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


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
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sender', required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    args = parser.parse_args()
    config = read(args.config)
    if config.get('active_pairs'):
        config['pairs'] = {n: p for n, p in config['pairs'].items() if n in config['active_pairs']}
    run = ROOT / config['output']
    manifest = read(run / 'manifest.json')
    model_path = manifest['models'][args.sender]
    def task_variant(task=None):
        vs = {pr.get('variant_overrides', {}).get(task, pr.get('variant', 'full'))
              for pr in config['pairs'].values() if pr['sender'] == args.sender}
        if len(vs) != 1:
            raise ValueError(f'{args.sender}: pairs disagree on the context variant of {task}: {sorted(vs)}')
        return vs.pop()
    max_new = config['summary_max_new_tokens']
    mcfg = json.loads((Path(model_path) / 'config.json').read_text())
    window = (mcfg.get('text_config') or mcfg).get('max_position_embeddings')
    variants = {task: task_variant(task) for task in config['datasets']}
    lengths = {task: min(config['variants'][v]['summarizer_max_length'], int(window) if window else 10 ** 9)
               for task, v in variants.items()}
    max_length = max(lengths.values())

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    stops = stop_token_ids(tokenizer)

    my_pairs = [pr for pr in config['pairs'].values() if pr['sender'] == args.sender]
    fit_only = 'fit_filter' in config and all(pr.get('fit_filter') for pr in my_pairs)
    contexts, payloads, targets = {}, {}, {}
    for task in config['datasets']:
        seen = {}
        variant = variants[task]
        for r in read(run / 'samples' / variant / f'{task}.json')['samples']:
            seen.setdefault(r['metadata']['context_id'], r['metadata']['context'])
        if fit_only and not any(task in pr.get('fit_filter_exempt', []) for pr in my_pairs):
            side = read(run / 'samples' / variant / f'{task}.fit.json')
            seen = {c: t for c, t in seen.items() if c in set(side['fit'])}
        contexts[task] = seen
        targets[task] = run / task / 'summaries' / f'{args.sender}.json'
        payloads[task] = read(targets[task]) if targets[task].exists() else {
            'sender': args.sender, 'summarizer_model': str(model_path), 'summarizer_max_length': lengths[task],
            'variant': variant, 'summary_max_new_tokens': max_new, 'backend': 'vllm', 'seed': config['seed'],
            'sampling': {'temperature': 0.7, 'top_k': 20, 'top_p': 0.8, 'do_sample': True},
            'prompts': {}, 'complete': False}

    requests = []
    for prompt in config['prompts']:
        for task, ctxs in contexts.items():
            done = payloads[task]['prompts'].setdefault(prompt, {})
            for cid, text in ctxs.items():
                if cid in done and done[cid].get('context_sha256') == sha(text):
                    continue
                prompt_text = chat_prompt(tokenizer, SUMMARY_PROMPTS[prompt].format(article_text=text))
                prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=False)['input_ids'])
                original_tokens = len(tokenizer(text, add_special_tokens=False)['input_ids'])
                cap = min(original_tokens, lengths[task] - prompt_tokens, max_new)
                if prompt_tokens >= lengths[task] or cap <= 0:
                    raise ValueError(f'{task}/{cid}: prompt {prompt_tokens} tokens does not fit {lengths[task]}')
                requests.append(dict(prompt=prompt, task=task, cid=cid, text=text, prompt_text=prompt_text,
                                     original_tokens=original_tokens, cap=cap))
    print(f'{args.sender}: {len(requests)} pending summaries', flush=True)
    if not requests:
        for task in config['datasets']:
            payloads[task]['complete'] = True
            write(targets[task], payloads[task])
        return

    llm = LLM(model=model_path, dtype='bfloat16', max_model_len=max_length, seed=config['seed'],
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=False)
    params = [SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=r['cap'],
                             stop_token_ids=stops, seed=config['seed'] + i) for i, r in enumerate(requests)]
    started = time.time()
    outputs = llm.generate([r['prompt_text'] for r in requests], params)
    elapsed = time.time() - started
    print(f'{args.sender}: generated {len(outputs)} summaries in {elapsed / 60:.1f} min', flush=True)

    for r, out in zip(requests, outputs):
        gen = out.outputs[0]
        summary = gen.text.strip()
        if not summary:
            raise RuntimeError(f"empty summary for {r['task']}/{r['cid']}")
        metrics = getattr(out, 'metrics', None)
        arrival = getattr(metrics, 'arrival_time', None)
        first = getattr(metrics, 'first_token_time', None)
        finished = getattr(metrics, 'finished_time', None)
        timings = {
            'preparation_ms': (finished - arrival) * 1000 if arrival and finished else None,
            'sender_prefill_ms': (first - arrival) * 1000 if arrival and first else None,
            'sender_am_ms': 0.0,
            'sender_decode_ms': (finished - first) * 1000 if first and finished else None,
            'timing_note': 'vLLM continuous batching: per-request latency inside a shared batch, not an isolated sender time',
        }
        payloads[r['task']]['prompts'][r['prompt']][r['cid']] = {
            'text': summary, 'prompt': r['prompt'],
            'original_tokens_summarizer': r['original_tokens'],
            'summary_tokens_summarizer': len(tokenizer(summary, add_special_tokens=False)['input_ids']),
            'generation_limit': r['cap'],
            'generated_tokens_including_special': len(gen.token_ids),
            'finish_reason': gen.finish_reason,
            **timings,
            'context_preview': preview(r['text']), 'context_chars': len(r['text']), 'context_sha256': sha(r['text']),
        }
    for task in config['datasets']:
        payload = payloads[task]
        payload.update(variant=variants[task], summarizer_max_length=lengths[task], summary_max_new_tokens=max_new)
        payload['complete'] = all(all(cid in payload['prompts'].get(p, {}) and payload['prompts'][p][cid].get('context_sha256') == sha(text)
                                      for cid, text in contexts[task].items()) for p in config['prompts'])
        payload['generation_minutes_all_pending'] = round(elapsed / 60, 2)
        write(targets[task], payload)
        print(f'{task}/{args.sender}: complete={payload["complete"]}', flush=True)


if __name__ == '__main__':
    main()
