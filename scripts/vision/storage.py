import hashlib
import json
import os
from pathlib import Path
import tempfile

EVAL_SCHEMA = 'vision-eval-v2'
TRACKED_SCHEMA = 'vision-tracked-v1'
SCORE_UNIT = 'percent; total_score is equal-weight dataset macro average'
GROUP_NAMES = {'no_training': 'no_train'}


def load(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def is_result(doc):
    return (isinstance(doc, dict) and doc.get('schema') == EVAL_SCHEMA and isinstance(doc.get('method'), str)
            and isinstance(doc.get('datasets'), dict) and bool(doc['datasets'])
            and all(isinstance(ds, dict) and isinstance(ds.get('results'), list) for ds in doc['datasets'].values()))


def save(path, data):
    _write(path, json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    if is_result(data):
        export_tracked(path, data)


def tracked_path(path):
    path = Path(path).resolve()
    if len(path.parents) < 3:
        raise ValueError(f'Result path needs GROUP/SPLIT parents: {path}')
    split, group = path.parent.name, GROUP_NAMES.get(path.parent.parent.name, path.parent.parent.name)
    return path.parents[2] / 'tracked' / f'{split}_{group}_{path.stem}.json', split, group


def tracked_document(path, doc):
    if not is_result(doc):
        raise ValueError(f'Not a scored benchmark result: {path}')
    path = Path(path).resolve()
    target, split, group = tracked_path(path)
    if doc['method'] != path.stem:
        raise ValueError(f'File name and method differ: {path} / {doc["method"]}')
    datasets, totals, scores = {}, [], []
    for name, ds in doc['datasets'].items():
        rows = ds['results']
        if not rows:
            continue
        samples = [{'id': str(r['question_id']), 'answer': r['output'], 'score': r['score']} for r in rows]
        total = 100 * sum(s['score'] for s in samples) / len(samples)
        datasets[name] = {'n': len(samples), 'total_score': total, 'samples': samples}
        totals.append(total)
        scores.extend(s['score'] for s in samples)
    if not scores:
        raise ValueError(f'No scored samples to export: {path}')
    return target, {
        'schema': TRACKED_SCHEMA, 'method': doc['method'], 'split': split, 'group': group,
        'complete': bool(doc.get('complete', False)), 'n': len(scores),
        'total_score': sum(totals) / len(totals), 'score_unit': SCORE_UNIT,
        'sample_mean_score': 100 * sum(scores) / len(scores),
        'full_result': os.path.relpath(path, target.parent),
        'hashes': {'result_sha256': file_sha256(path), 'protocol_sha256': doc['protocol_sha256']},
        'datasets': datasets,
    }


def export_tracked(path, doc=None):
    path = Path(path)
    if doc is None:
        doc = load(path)
    target, tracked = tracked_document(path, doc)
    _write(target, json.dumps(tracked, indent=0, separators=(',', ':'), ensure_ascii=False) + '\n')
    return target
