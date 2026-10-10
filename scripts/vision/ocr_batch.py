import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.run import image_key, sha256

SCHEMA = 'vision-ocr-v1'
BUDGET = 96


def bundles(data_dir, limit):
    groups = {}
    for path in sorted(Path(data_dir).glob('*/questions.json')):
        if any(part.startswith('.') for part in path.parts):
            continue
        manifest = json.loads(path.read_text())
        for row in manifest['rows'][:limit]:
            images = [manifest['images'][str(i)] for i in row.get('image_ids', [row.get('image_id')])]
            hashes = [im['sha256'] for im in images]
            groups.setdefault(image_key(hashes), [str((ROOT / im['path']).resolve()) for im in images])
    return groups


def shard_of(key, n):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n


def protocol(args):
    from scripts.vision.inference import OCR_PROMPT
    return {'sender': str(args.sender), 'sender_config_sha256': sha256(args.sender / 'config.json'),
            'data_dir': str(args.data_dir), 'limit_per_benchmark': args.limit, 'ocr_prompt': OCR_PROMPT,
            'numbered_images': True, 'budget': BUDGET, 'decoding': 'greedy, left-padded batches',
            'equivalent_to': 'TextScan/v6 64+32 continuation transcript', 'question_blind': True}


def run(args):
    import torch
    from PIL import Image
    from scripts.vision.inference import Engine, OCR_PROMPT, caption_inputs
    groups = bundles(args.data_dir, args.limit)
    journal = Path(f'{args.output}.shard{args.shard}of{args.shards}.jsonl')
    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                done.add(json.loads(line)['key'])
            except json.JSONDecodeError:
                pass
    todo = [k for k in sorted(groups) if shard_of(k, args.shards) == args.shard and k not in done]
    todo.sort(key=lambda k: (len(groups[k]), k))
    print(f'shard {args.shard}/{args.shards}: {len(todo)} bundles to transcribe ({len(done)} done)', flush=True)
    engine = Engine.captioner(str(args.sender))
    started = time.monotonic()
    with journal.open('a') as out:
        for i in range(0, len(todo), args.batch_size):
            keys = todo[i:i + args.batch_size]
            inputs = []
            for k in keys:
                images = []
                for p in groups[k]:
                    with Image.open(p) as im:
                        images.append(im.convert('RGB'))
                inputs.append(caption_inputs(engine.processor, images[0] if len(images) == 1 else images,
                                             OCR_PROMPT, numbered=True))
            try:
                results = engine._gemma_batch_generate(inputs, [BUDGET] * len(inputs))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results = [r for inp in inputs for r in engine._gemma_batch_generate([inp], [BUDGET])]
            for k, r in zip(keys, results):
                out.write(json.dumps({'key': k, 'output': r['output'], 'output_tokens': len(r['output_ids']),
                                      'stop_reason': r['stop_reason']}) + '\n')
            out.flush()
            n = i + len(keys)
            if (i // args.batch_size) % 20 == 0:
                el = time.monotonic() - started
                print(f'{n}/{len(todo)} bundles, {el/60:.1f} min, ~{el/n*(len(todo)-n)/60:.1f} min left', flush=True)
    print(f'shard {args.shard} done in {(time.monotonic()-started)/60:.1f} min', flush=True)


def merge(args):
    groups = bundles(args.data_dir, args.limit)
    ocr = {}
    for j in sorted(Path(args.output).parent.glob(Path(args.output).name + '.shard*of*.jsonl')):
        for line in j.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ocr[rec.pop('key')] = rec
    missing = set(groups) - set(ocr)
    if missing:
        raise SystemExit(f'{len(missing)} bundles missing')
    doc = {'schema': SCHEMA, 'complete': True, 'bundles': len(groups), 'protocol': protocol(args),
           'ocr': {k: ocr[k] for k in sorted(groups)}}
    Path(args.output).write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f'merged {len(groups)} bundles -> {args.output}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--limit', type=int, required=True)
    ap.add_argument('--sender', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    merge(args) if args.merge else run(args)


if __name__ == '__main__':
    main()
