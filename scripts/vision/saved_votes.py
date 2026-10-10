from datetime import datetime, timezone
from pathlib import Path

from scripts.vision.consensus import VOTERS, consensus
from scripts.vision.run import EVAL_SCHEMA, fingerprint, require_compact, token_count
from scripts.vision.scoring import cluster_ids, mme_summary, uncertainty
from scripts.vision.storage import file_sha256, load, save

ROOT = Path(__file__).resolve().parents[2]
RECIPE = {'voters': list(VOTERS), 'normalization': 'canonical', 'tie_breaker': 'weighted_confirm',
          'ocr_weight': 4, 'ocr_min_tokens': 64, 'allow_caption_variants': True}
SHARED_ROW_KEYS = ('question_id', 'question', 'image_sha256', 'image_sha256s', 'references',
                   'source_image_sha256', 'source_image_sha256s')
SHARED_PROTOCOL_KEYS = ('caption_prompt', 'caption_max_tokens', 'instruction', 'answer_max_tokens', 'dtype',
                        'sender_config_sha256', 'receiver_config_sha256', 'randomization')


def stamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ocr_tokens(anchor, key):
    auxiliary = anchor['prefixes'].get(key, {}).get('auxiliary_captions', [])
    if len(auxiliary) != 1:
        raise ValueError(f'TextScan OCR metadata missing for image bundle {key}')
    return token_count(auxiliary[0])


def rebuild(output_dir, method='scantrust_v2'):
    output_dir = Path(output_dir)
    prefix = 'rand_' if method.startswith('rand_') else ''
    if method[len(prefix):] != 'scantrust_v2':
        raise ValueError(f'Not a saved-vote method: {method}')
    names = {v: prefix + v for v in VOTERS}
    docs = {}
    for voter, name in names.items():
        path = output_dir / f'{name}.json'
        if not path.exists():
            raise ValueError(f'Missing voter file: {path}')
        doc = load(path)
        require_compact(doc)
        if doc.get('schema') != EVAL_SCHEMA or doc.get('method') != name:
            raise ValueError(f'Expected compact {name} eval JSON: {path}')
        docs[voter] = doc
    anchor = docs['textscan_v6']
    for voter, doc in docs.items():
        base, ref = doc['protocol']['base_protocol'], anchor['protocol']['base_protocol']
        for key in SHARED_PROTOCOL_KEYS:
            if key in base and key in ref and base[key] != ref[key]:
                raise ValueError(f'Voter {names[voter]} protocol differs from TextScan: {key}')
        renderer = doc['protocol'].get('renderer', {}).get('sha256')
        if renderer is not None and renderer != anchor['protocol']['renderer']['sha256']:
            raise ValueError(f'Voter {names[voter]} used a different renderer')
        if set(doc['datasets']) != set(anchor['datasets']):
            raise ValueError(f'Voter {names[voter]} covers different datasets')

    target = output_dir / f'{method}.json'
    previous = load(target) if target.exists() else None
    old_rows = {}
    if previous is not None:
        for name, ds in previous.get('datasets', {}).items():
            for r in ds.get('results', []):
                old_rows[name + ':' + str(r['question_id'])] = r
    changed_votes, changed_outputs, changed_scores = [], [], []
    datasets, count, expected = {}, 0, 0
    for name, ads in anchor['datasets'].items():
        indexed = {}
        for voter, doc in docs.items():
            rows = {str(r['question_id']): r for r in doc['datasets'][name]['results']}
            if len(rows) != len(doc['datasets'][name]['results']):
                raise ValueError(f'Duplicate question IDs in {names[voter]}/{name}')
            indexed[voter] = rows
        shared = [qid for qid in indexed['textscan_v6'] if all(qid in indexed[v] for v in VOTERS)]
        results = []
        for qid in shared:
            rows = {voter: indexed[voter][qid] for voter in VOTERS}
            first = rows['textscan_v6']
            for voter, row in rows.items():
                if any(k in row and k in first and row[k] != first[k] for k in SHARED_ROW_KEYS):
                    raise ValueError(f'Voter inputs differ: {names[voter]} {name}:{qid}')
                if row['question'] != first['question'] or row['image_sha256'] != first['image_sha256']:
                    raise ValueError(f'Voter inputs differ: {names[voter]} {name}:{qid}')
            vote = consensus({v: {'output': rows[v]['output']} for v in VOTERS},
                             ocr_tokens(anchor, first['image_sha256']))
            selected = rows[vote['selected_voter']]
            if selected['output'] != vote['output']:
                raise ValueError(f'Consensus output is not the selected voter answer: {name}:{qid}')
            result = {**selected, 'seconds': sum(r.get('seconds', 0) for r in rows.values()),
                      'selected_voter': vote['selected_voter'], 'vote_weights': vote['weights'],
                      'voter_outputs': {v: rows[v]['output'] for v in VOTERS}}
            if 'image_sha256s' in first:
                result['image_sha256s'] = first['image_sha256s']
            identity = name + ':' + qid
            old = old_rows.get(identity)
            if old is not None:
                if old.get('voter_outputs') != result['voter_outputs']:
                    changed_votes.append(identity)
                if old.get('output') != result['output']:
                    changed_outputs.append(identity)
                if old.get('score') != result['score']:
                    changed_scores.append(identity)
            results.append(result)
        captions = {r['image_sha256']: {'image_sha256s': r['image_sha256s']} for r in results}
        ds = {k: v for k, v in ads.items() if k not in ('results', 'summary', 'complete', 'uncertainty_cluster', 'mme')}
        ids = cluster_ids(results, captions) if results else []
        ds['complete'] = len(results) == ads['expected_questions']
        ds['summary'] = uncertainty([r['score'] for r in results], ids) if ids else {
            'score': None, 'se': None, 'n': 0, 'image_clusters': 0, 'ci95': None}
        ds['uncertainty_cluster'] = 'connected components of shared input images'
        if ads['metric'] == 'mme_accuracy_plus' and results:
            ds['mme'] = mme_summary(results, load(ROOT / ads['manifest'])['rows'])
        ds['results'] = results
        datasets[name] = ds
        count += len(results)
        expected += ads['expected_questions']

    protocol = {**anchor['protocol'], 'recipes': {method: RECIPE}, 'prediction_sources': []}
    complete = count == expected and all(ds['complete'] for ds in datasets.values())
    now = stamp()
    doc = {'schema': EVAL_SCHEMA, 'method': method,
           'created_utc': previous['created_utc'] if previous else now,
           'complete': complete, 'status': 'complete' if complete else 'partial',
           'protocol': protocol, 'protocol_sha256': fingerprint(protocol),
           'expected_images': anchor['expected_images'], 'caption_source': anchor['caption_source'],
           'append_runs': anchor.get('append_runs', []), 'questions_completed': count,
           'updated_utc': now, 'final_verified': complete}
    if complete:
        doc['completed_utc'] = now
    doc['last_run_seconds'] = max(d.get('last_run_seconds', 0) for d in docs.values())
    doc['voter_sources'] = {v: {'file': f'{names[v]}.json', 'sha256': file_sha256(output_dir / f'{names[v]}.json'),
                                'protocol_sha256': docs[v]['protocol_sha256']} for v in VOTERS}
    doc['datasets'] = datasets
    doc['vote_rebuild'] = {'utc': now, 'source': 'saved voter files',
                           'changed_voter_answers': len(changed_votes), 'changed_outputs': len(changed_outputs),
                           'changed_scores': len(changed_scores),
                           'source_sha256': {n: file_sha256(Path(__file__).parent / n)
                                             for n in ('saved_votes.py', 'consensus.py', 'scoring.py')}}
    doc['checks'] = {'voter_inputs_match': True, 'votes_recomputed_from_saved_sources': True,
                     'scores_from_selected_voter': True}
    doc['prefixes'] = {h: {'voters': {v: docs[v]['prefixes'][h] for v in VOTERS if h in docs[v]['prefixes']}}
                       for h in anchor['prefixes']}
    save(target, doc)
    scored = [ds['summary']['score'] for ds in datasets.values() if ds['summary']['score'] is not None]
    return {'method': method, 'file': str(target), 'questions': count, 'expected': expected, 'complete': complete,
            'macro': sum(scored) / len(scored) if scored else None,
            'changed_voter_answers': changed_votes, 'changed_outputs': changed_outputs, 'changed_scores': changed_scores}
