import hashlib
import json
import random
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / 'multiagent/outputs_mine/interp/anthropic_assessment'
TEMP = ROOT / 'multiagent/outputs/interp/temp/anthropic_assessment'
COMMIT = '84fcc677e52e1902d696c32cd1a6b663e70d3993'
ITEMS_PER_SET, SEED = 100, 0
RAW = TEMP / 'raw_persona'


def main():
    RAW.mkdir(parents=True, exist_ok=True); (CORE / 'data').mkdir(parents=True, exist_ok=True)
    listing = requests.get(f'https://api.github.com/repos/anthropics/evals/contents/persona?ref={COMMIT}', timeout=60).json()
    names = sorted(x['name'] for x in listing if x['name'].endswith('.jsonl'))
    manifest, sample = {}, []
    for name in names:
        path = RAW / name
        if not path.exists():
            r = requests.get(f'https://raw.githubusercontent.com/anthropics/evals/{COMMIT}/persona/{name}', timeout=60)
            r.raise_for_status(); path.write_bytes(r.content)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        manifest[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n': len(rows)}
        rng = random.Random(f'{SEED}:{name}')
        idx = list(range(len(rows))); rng.shuffle(idx)
        for i in sorted(idx[:ITEMS_PER_SET]):
            r = rows[i]
            assert r['answer_matching_behavior'].strip() in ('Yes', 'No'), (name, r['answer_matching_behavior'])
            sample.append({'set': name[:-6], 'idx': i, 'question': r['question'], 'statement': r.get('statement'),
                           'matching': r['answer_matching_behavior'].strip()})
    (CORE / 'data/persona_sample.jsonl').write_text('\n'.join(json.dumps(s) for s in sample) + '\n')
    (TEMP / 'download_manifest.json').write_text(json.dumps({'repo': 'anthropics/evals', 'commit': COMMIT, 'items_per_set': ITEMS_PER_SET,
                                                             'seed': SEED, 'files': manifest}, indent=2) + '\n')
    yes = sum(s['matching'] == 'Yes' for s in sample)
    print(f'{len(names)} sets, {len(sample)} items, {yes} Yes-matching / {len(sample) - yes} No-matching')


if __name__ == '__main__':
    main()
