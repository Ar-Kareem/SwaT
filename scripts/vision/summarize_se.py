import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.vision.storage import load as load_json, file_sha256

import numpy as np


def digest(path):
    return file_sha256(path)


def with_summary(data):
    results = data['results']
    summary = {
        split: {
            method: f'{entry["macro"]["score"]:.2f} ± {entry["macro"]["se"]:.2f}'
            for method, entry in sorted(results.get(split, {}).items())
        }
        for split in ('eval', 'test')
    }
    return {**{key: value for key, value in data.items() if key not in ('summary', 'results')},
            'summary': summary, 'results': results}


def atomic_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(with_summary(data), stream, indent=2, allow_nan=False, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(path):
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f'vision-se-{key}.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another process is writing {path}') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def summarize(doc, captions, resamples=10000, seed=42):
    if doc.get('complete') is not True or not doc.get('datasets'):
        raise ValueError('Result is incomplete or has no datasets')
    keys = sorted(doc['datasets'])
    parents, records = {}, []

    def find(h):
        parents.setdefault(h, h)
        while parents[h] != h:
            parents[h] = parents[parents[h]]
            h = parents[h]
        return h

    for j, key in enumerate(keys):
        ds = doc['datasets'][key]
        results = ds['results']
        if (ds.get('complete') is not True or not results
                or len(results) != ds['expected_questions']
                or len({str(r['question_id']) for r in results}) != len(results)):
            raise ValueError(f'Incomplete or duplicate questions: {key}')
        for row in results:
            hashes = (row.get('image_sha256s')
                      or captions.get(row['image_sha256'], {}).get('image_sha256s')
                      or [row['image_sha256']])
            if not isinstance(hashes, list) or not all(isinstance(h, str) and h for h in hashes):
                raise ValueError(f'Invalid image hashes: {key}/{row["question_id"]}')
            for h in hashes:
                a, b = find(hashes[0]), find(h)
                parents[max(a, b)] = min(a, b)
            score = float(row['score'])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Invalid score: {key}/{row["question_id"]}')
            records.append((j, hashes[0], score))

    groups = sorted({find(h) for _, h, _ in records})
    index = {g: i for i, g in enumerate(groups)}
    sums = np.zeros((len(groups), len(keys)))
    counts = np.zeros_like(sums)
    for j, h, score in records:
        i = index[find(h)]
        sums[i, j] += score
        counts[i, j] += 1
    present = [np.flatnonzero(counts[:, j]) for j in range(len(keys))]
    if any(len(p) < 2 for p in present):
        raise ValueError('Each benchmark needs at least two independent image groups')
    scores = 100 * sums.sum(axis=0) / counts.sum(axis=0)
    for j, key in enumerate(keys):
        if not np.isclose(scores[j], doc['datasets'][key]['summary']['score'], rtol=0, atol=1e-7):
            raise ValueError(f'Saved summary disagrees with question scores: {key}')

    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(keys)))
    probabilities = np.full(len(groups), 1 / len(groups))
    for first in range(0, resamples, 256):
        last = min(first + 256, resamples)
        weights = rng.multinomial(len(groups), probabilities, size=last - first)
        for j, indices in enumerate(present):
            selected = weights[:, indices]
            denominator = np.einsum('ij,j->i', selected, counts[indices, j])
            if np.any(denominator == 0):
                raise ValueError('Bootstrap replicate has an empty benchmark; insufficient clusters')
            draws[first:last, j] = (
                100 * np.einsum('ij,j->i', selected, sums[indices, j]) / denominator)

    return {
        'benchmarks': {
            key: {'n': int(counts[:, j].sum()), 'clusters': len(present[j]),
                  'score': float(scores[j]), 'se': float(draws[:, j].std(ddof=1))}
            for j, key in enumerate(keys)
        },
        'macro': {'score': float(scores.mean()), 'se': float(draws.mean(axis=1).std(ddof=1))},
    }


def selection(values, allowed=None):
    values = list(dict.fromkeys(v for item in values for v in item.split(',') if v))
    if not values or ('all' in values and values != ['all']):
        raise ValueError('Use all alone, or a list of names')
    if allowed is not None and any(v not in allowed for v in values if v != 'all'):
        raise ValueError(f'Expected one or more of: {", ".join(allowed)}, all')
    return values


def run(args):
    root = args.input_dir.resolve()
    if not root.is_dir():
        raise ValueError(f'Input directory does not exist: {root}')
    methods = selection(args.methods)
    splits = selection(args.splits, ['eval', 'test'])
    if splits == ['all']:
        splits = ['eval', 'test']
    candidates = {}
    for split in splits:
        for area in ('no_training', 'renderer'):
            for path in sorted((root / area / split).glob('*.json')):
                if path.name.startswith('.') or path.stem.endswith('_captions'):
                    continue
                key = (split, path.stem)
                if key in candidates:
                    raise ValueError(f'Duplicate method: {split}/{path.stem}')
                candidates[key] = path
    if methods != ['all']:
        missing = set(methods) - {method for _, method in candidates}
        if missing:
            raise ValueError(f'Methods not found: {", ".join(sorted(missing))}')
        candidates = {k: p for k, p in candidates.items() if k[1] in methods}
    if not candidates:
        raise ValueError('No matching result files')

    config = {'resamples': args.resamples, 'seed': args.seed,
              'unit': 'image_connected_cluster', 'scope': 'joint_benchmarks',
              'macro': 'equal_benchmark_weight', 'se_ddof': 1}
    output = {'version': 1, 'bootstrap': config, 'results': {}}
    if args.output_file.exists():
        output = json.loads(args.output_file.read_text())
        if output.get('version') != 1 or output.get('bootstrap') != config:
            raise ValueError('Output format/settings differ; use a new output file')
        formatted = with_summary(output)
        if formatted != output or list(formatted) != list(output):
            atomic_save(args.output_file, formatted)
        output = formatted
    captions_cache = {}
    completed = skipped = 0
    for (split, method), path in sorted(candidates.items()):
        start = time.perf_counter()
        caption_path = root / 'no_training' / split / 'gemma_captions.json'
        if split not in captions_cache:
            caption_hash = digest(caption_path) if caption_path.exists() else None
            captions_cache[split] = [caption_hash, None]
        caption_hash, captions = captions_cache[split]
        signature = {'source': str(path), 'sha256': digest(path), 'captions_sha256': caption_hash}
        entries = output['results'].setdefault(split, {})
        existing = entries.get(method, {})
        if (all(existing.get(k) == v for k, v in signature.items())
                and existing.get('benchmarks') and 'se' in existing.get('macro', {})):
            print(f'Skip {split}/{method}: unchanged', flush=True)
            skipped += 1
            continue
        print(f'Compute {split}/{method}', flush=True)
        doc = load_json(path)
        if doc.get('complete') is not True:
            if method in entries:
                del entries[method]
                atomic_save(args.output_file, output)
            print(f'Skip {split}/{method}: incomplete', flush=True)
            skipped += 1
            continue
        if captions is None:
            captions = {}
            if caption_path.exists():
                captions = load_json(caption_path).get('captions', {})
            captions_cache[split][1] = captions
        entry = summarize(doc, captions, args.resamples, args.seed)
        del doc
        if digest(path) != signature['sha256'] or (caption_hash and digest(caption_path) != caption_hash):
            raise ValueError(f'Input changed during computation: {split}/{method}; rerun')
        entry = {**signature, **entry, 'seconds': round(time.perf_counter() - start, 3)}
        entries[method] = entry
        atomic_save(args.output_file, output)
        completed += 1
        macro = entry['macro']
        print(f'  {macro["score"]:.4f} ± {macro["se"]:.4f} SE ({entry["seconds"]:.2f}s)', flush=True)
    print(f'Saved {args.output_file}: {completed} computed, {skipped} skipped', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['all'])
    parser.add_argument('--splits', '--split', nargs='+', default=['all'])
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.resamples < 2 or args.seed < 0:
        parser.error('resamples must be at least 2 and seed must be nonnegative')
    try:
        with output_lock(args.output_file):
            run(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
