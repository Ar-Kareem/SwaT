from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sender', required=True, help='sender model path; its tokenizer defines the window length')
    p.add_argument('--sequences', type=int, default=500)
    p.add_argument('--heldout-sequences', type=int, default=32)
    p.add_argument('--length', type=int, default=1024)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--revision', default='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9')
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError(f'Calibration already exists: {out}')
    tokenizer = AutoTokenizer.from_pretrained(args.sender)
    revision = HfApi().dataset_info('HuggingFaceFW/fineweb-edu', revision=args.revision).sha
    data = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                        revision=revision, split='train', streaming=True)
    data = data.shuffle(seed=args.seed, buffer_size=1000)
    rng = random.Random(args.seed)
    records, seen = [], set()
    count = args.sequences + args.heldout_sequences
    with open(out / 'texts.jsonl', 'w') as handle:
        for row in data:
            digest = hashlib.sha256(row['text'].encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            encoded = tokenizer(row['text'], add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded['offset_mapping']
            if len(offsets) < args.length:
                continue
            start = rng.randrange(len(offsets) - args.length + 1)
            char_start, char_end = offsets[start][0], offsets[start + args.length - 1][1]
            window = row['text'][char_start:char_end]
            handle.write(json.dumps({'text': window}) + '\n')
            records.append({'id': row.get('id'), 'url': row.get('url'), 'text_sha256': digest,
                            'document_tokens': len(offsets), 'token_start': start, 'token_end': start + args.length,
                            'char_start': char_start, 'char_end': char_end,
                            'split': 'fit' if len(records) < args.sequences else 'heldout'})
            if len(records) % 50 == 0:
                print(f'Selected {len(records)}/{count} documents', flush=True)
            if len(records) == count:
                break
    if len(records) != count:
        raise RuntimeError(f'Only found {len(records)} eligible documents')
    manifest = {'dataset': 'HuggingFaceFW/fineweb-edu', 'config': 'sample-10BT', 'revision': revision,
                'seed': args.seed, 'shuffle_buffer': 1000, 'format': 'text',
                'fit_sequences': args.sequences, 'heldout_sequences': args.heldout_sequences,
                'sequence_length': args.length, 'stride': args.stride,
                'window_tokenizer': args.sender,
                'window_tokenizer_sha256': hashlib.sha256((Path(args.sender) / 'tokenizer.json').read_bytes()).hexdigest(),
                'texts_sha256': hashlib.sha256((out / 'texts.jsonl').read_bytes()).hexdigest(),
                'sampling': 'one random contiguous window of --length sender tokens per eligible document, stored as text; '
                            'no chat template or added special tokens; both models tokenize the text natively at fit time',
                'reproduction_note': 'Paper does not specify dataset revision, subset, seed or exact documents.',
                'documents': records}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {count} text windows to {out}', flush=True)


if __name__ == '__main__':
    main()
