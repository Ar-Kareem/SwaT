from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagent.paths import MODELS_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', default=str(MODELS_ROOT / 'qwen/Qwen3-4B'))
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    revision = HfApi().dataset_info('HuggingFaceFW/fineweb-edu', revision=args.revision).sha
    data = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                        revision=revision, split='train', streaming=True)
    data = data.shuffle(seed=args.seed, buffer_size=1000)
    rng = random.Random(args.seed)
    ids, records, seen = [], [], set()
    count = args.sequences + args.heldout_sequences
    for row in data:
        digest = hashlib.sha256(row['text'].encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        tokens = tokenizer.encode(row['text'], add_special_tokens=False)
        if len(tokens) < args.length:
            continue
        start = rng.randrange(len(tokens) - args.length + 1)
        window = tokens[start:start + args.length]
        ids.append(window)
        records.append({'id': row.get('id'), 'url': row.get('url'),
                        'text_sha256': digest, 'document_tokens': len(tokens),
                        'token_start': start, 'token_end': start + args.length,
                        'split': 'fit' if len(ids) <= args.sequences else 'heldout'})
        if len(ids) % 50 == 0:
            print(f'Selected {len(ids)}/{count} documents', flush=True)
        if len(ids) == count:
            break
    if len(ids) != count:
        raise RuntimeError(f'Only found {len(ids)} eligible documents')
    np.save(out / 'input_ids.npy', np.asarray(ids, dtype=np.int64))
    manifest = {'dataset': 'HuggingFaceFW/fineweb-edu', 'config': 'sample-10BT',
                'revision': revision, 'seed': args.seed, 'shuffle_buffer': 1000,
                'fit_sequences': args.sequences, 'heldout_sequences': args.heldout_sequences,
                'sequence_length': args.length, 'stride': args.stride,
                'sampled_positions': list(range(0, args.length, args.stride)),
                'tokenizer': args.model,
                'tokenizer_sha256': hashlib.sha256((Path(args.model) / 'tokenizer.json').read_bytes()).hexdigest(),
                'input_ids_sha256': hashlib.sha256((out / 'input_ids.npy').read_bytes()).hexdigest(),
                'sampling': 'one random contiguous window per eligible document; no chat template or added special tokens',
                'reproduction_note': 'Paper does not specify dataset revision, subset, seed or exact documents.',
                'documents': records}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {count} sequences to {out}', flush=True)


if __name__ == '__main__':
    main()
