import argparse
import copy
from contextlib import closing
from datetime import datetime, timezone
import json
import random
from pathlib import Path
import sys
import time
from itertools import chain
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.vision.run import (SCHEMA, EVAL_SCHEMA, CAPTION_SCHEMA, sha256, fingerprint,
    read_jsonl, image_key, predict, input_rows, validate_captions, load_captions,
    compact, require_compact, token_count, load_ocr, BASELINES, GEMMA_BASELINES,
    GEMMA_CONTROLS, CAPTION_BASELINES, OCR_BASELINES)
from scripts.vision.consensus import VOTERS
from scripts.vision.storage import load as load_json, save as save_json, export_tracked
from scripts.vision.saved_votes import rebuild as rebuild_votes
from scripts.vision.scoring import score_answer, uncertainty, cluster_ids, validate_metrics, mme_summary
from scripts.vision.parallel import gpu_list, stream, prediction_worker, caption_worker, baseline_worker, baseline_results

RECIPES = {
    'captkv': {}, 'echo_v1': {'echo': 'full'},
    'kvgain_v3': {'echo': 'full', 'visual_stride': 16, 'visibility': 'lookahead32', 'value_scale': 1.3},
    'textscan_v6': {'echo': 'full', 'visual_stride': 16, 'visibility': 'lookahead32',
                    'auxiliary_captions': [{'prompt': 'Read the text in this image. Copy visible words and numbers exactly. Do not describe the image. Output only copied text.',
                                            'max_tokens': 64, 'label': 'Additional visible text:'}], 'auxiliary_extend': 32},
    'scantrust_v2': {'voters': list(VOTERS), 'normalization': 'canonical', 'tie_breaker': 'weighted_confirm',
                     'ocr_weight': 4, 'ocr_min_tokens': 64, 'allow_caption_variants': True},
}
BASE_RECIPES = dict(RECIPES)
RECIPES.update({'rand_' + name: {**recipe, 'random_seed': 42}
                for name, recipe in BASE_RECIPES.items()})
RECIPES.update({'randimg_' + name: {**recipe, 'random_seed': 42, 'keep_caption': True}
                for name, recipe in BASE_RECIPES.items()})


def random_kind(method):
    return next((p for p in ('rand_', 'randimg_') if method.startswith(p)), '')


def base_method(method):
    return method.removeprefix(random_kind(method))


def random_inputs(plan):
    pool = {}
    for name, ds in sorted(plan['datasets'].items()):
        path = ROOT / ds['manifest']
        if sha256(path) != ds['manifest_sha256']:
            raise ValueError(f'Manifest changed: {path}')
        manifest = load_json(path)
        wanted = set(ds['question_ids'])
        for row in manifest['rows']:
            if str(row['question_id']) not in wanted:
                continue
            images = bundle(manifest, row)
            hashes = [im['sha256'] for im in images]
            pool.setdefault(image_key(hashes), {'image_sha256s': hashes,
                'images': [str((ROOT / im['path']).resolve()) for im in images]})
    keys = sorted(pool)
    if len(keys) < 2:
        raise ValueError('Random methods require at least two distinct image bundles')
    members = {key: set(pool[key]['image_sha256s']) for key in keys}
    rng = random.Random(42)
    shuffled = list(keys)
    for _ in range(10000):
        rng.shuffle(shuffled)
        if all(members[a].isdisjoint(members[b]) for a, b in zip(keys, shuffled)):
            break
    else:
        raise ValueError('Cannot construct an image-disjoint donor permutation for this plan')
    mapping = dict(zip(keys, shuffled))
    metadata = {'seed': 42, 'policy': 'full-plan image-disjoint bundle permutation v1',
                'pool_sha256': fingerprint(keys), 'mapping_sha256': fingerprint(mapping)}
    return {key: pool[donor] for key, donor in mapping.items()}, metadata


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def atomic(path, data):
    save_json(path, compact(data))


def voter_methods(methods):
    result = []
    for method in methods:
        prefix = random_kind(method)
        expanded = [prefix + m for m in VOTERS] if base_method(method) == 'scantrust_v2' else [method]
        for item in expanded:
            if item not in result:
                result.append(item)
    return result


def bundle(manifest, row):
    return [manifest['images'][str(i)] for i in row.get('image_ids', [row.get('image_id')])]


def captions(args):
    from scripts.vision.inference import Engine, CAPTION_PROMPT
    if args.limit < 1:
        raise ValueError('limit must be positive')
    batch_size = getattr(args, 'batch_size', 1) or 1
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('--batch-size must be a positive integer')
    groups, datasets = {}, {}
    for path in sorted(args.data_dir.glob('*/questions.json')):
        if any(part.startswith('.') for part in path.parts):
            continue
        manifest = load_json(path)
        datasets[path.parent.name] = {'manifest': relative(path), 'manifest_sha256': sha256(path)}
        for row in manifest['rows'][:args.limit]:
            images = bundle(manifest, row)
            hashes = [im['sha256'] for im in images]
            groups.setdefault(image_key(hashes), {'image_sha256s': hashes,
                'images': [str((ROOT / im['path']).resolve()) for im in images]})
    if not groups:
        raise ValueError('No caption inputs found')
    if args.output.exists():
        doc = load_json(args.output)
        if set(doc['captions']) - set(groups):
            raise ValueError('Existing cache belongs to a different image selection')
        validate_captions(doc, [groups[h] for h in doc['captions']], args.sender, CAPTION_PROMPT,
                          require_complete=False)
        if doc['complete'] and set(doc['captions']) == set(groups):
            print(f'Captions already complete: {len(groups)} image bundles; no model loaded.')
            return
    else:
        protocol = {'models': {'gemma': {'path': str(args.sender.resolve()),
                        'files': {'config.json': {'sha256': sha256(args.sender / 'config.json')}}}},
                    'datasets': datasets, 'caption_prompt': CAPTION_PROMPT, 'caption_max_tokens': 256,
                    'image_size': [896, 896], 'image_tokens': 256, 'pan_and_scan': False,
                    'dtype': 'bfloat16', 'decoding': 'greedy split-cache; original caption recipe',
                    'all_models_frozen': True, 'seed': 42,
                    'model_hash_scope': 'Model config only; weights supplied externally.',
                    'execution': {'gemma_batch_size': batch_size,
                                  'gemma_batch_backend': 'left_pad_split_cache_v1' if batch_size > 1 else 'serial_split_cache',
                                  'batched_seconds_scope': 'batch wall time divided by actual batch size'}}
        doc = {'schema': CAPTION_SCHEMA, 'method': 'gemma_captions', 'created_utc': stamp(),
               'protocol': protocol, 'protocol_sha256': fingerprint(protocol), 'captions': {},
               'complete': False, 'caption_unit': 'ordered image bundle'}
    missing = [h for h in groups if h not in doc['captions']]
    doc.update(expected_images=len(groups), complete=False, status='partial')
    doc.pop('completed_utc', None)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        doc.update(images_completed=len(doc['captions']), updated_utc=stamp())
        doc['complete'] = set(doc['captions']) == set(groups)
        doc['status'] = 'complete' if doc['complete'] else 'partial'
        if doc['complete']:
            doc['completed_utc'] = stamp()
        atomic(args.output, doc)

    if not missing:
        save()
        return
    if getattr(args, 'gpus', None):
        jobs = [{'key': h, **groups[h], 'weight': len(groups[h]['images'])} for h in missing]
        started = last_saved = time.monotonic()
        try:
            for h, result in stream(caption_worker, args, jobs, args.gpus):
                doc['captions'][h] = result
                print(f'Captions: {len(doc["captions"])}/{len(groups)}', flush=True)
                if len(doc['captions']) == 1 or time.monotonic() - last_saved >= 60:
                    save()
                    last_saved = time.monotonic()
        finally:
            doc['last_run_seconds'] = time.monotonic() - started
            save()
        return
    import torch
    from PIL import Image
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Caption generation requires a CUDA GPU')
    engine = Engine.captioner(str(args.sender), args.device)
    started = last_saved = time.monotonic()
    import hashlib
    try:
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            bundles = []
            for h in chunk:
                group = groups[h]
                images = []
                for path, digest in zip(group['images'], group['image_sha256s']):
                    if sha256(path) != digest:
                        raise ValueError(f'Image hash mismatch: {path}')
                    with Image.open(path) as im:
                        images.append(im.convert('RGB'))
                bundles.append(images[0] if len(images) == 1 else images)
            then = time.monotonic()
            results = engine.describe_batch(bundles)
            seconds = (time.monotonic() - then) / len(chunk)
            for h, result in zip(chunk, results):
                result = compact(result)
                if not result['output'].strip():
                    raise ValueError(f'Empty caption: {h}')
                doc['captions'][h] = {**result, 'image_sha256': h,
                    'image_sha256s': groups[h]['image_sha256s'], 'image_paths': groups[h]['images'],
                    'caption_sha256': hashlib.sha256(result['output'].encode()).hexdigest(),
                    'seconds': seconds}
            print(f'Captions: {len(doc["captions"])}/{len(groups)}', flush=True)
            if len(doc['captions']) == len(chunk) or time.monotonic() - last_saved >= 60:
                save()
                last_saved = time.monotonic()
    finally:
        doc['last_run_seconds'] = time.monotonic() - started
        save()


def prepare(args):
    if args.shards < 1 or args.limit < 1:
        raise ValueError('shards and limit must be positive')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir / 'plan.json').exists():
        raise ValueError('Plan already exists; use its existing input shards')
    groups, datasets = {}, {}
    for path in sorted(args.data_dir.glob('*/questions.json')):
        if any(part.startswith('.') for part in path.parts):
            continue
        manifest = load_json(path)
        rows = manifest['rows'][:args.limit]
        name = path.parent.name
        datasets[name] = {'name': manifest['dataset'], 'source_split': manifest['split'],
                          'manifest': relative(path), 'manifest_sha256': sha256(path),
                          'metric': manifest['metric'], 'expected_questions': len(rows),
                          'question_ids': [str(r['question_id']) for r in rows]}
        if len(set(datasets[name]['question_ids'])) != len(rows):
            raise ValueError(f'Duplicate question IDs: {path}')
        for row in rows:
            images = bundle(manifest, row)
            hashes = tuple(im['sha256'] for im in images)
            group = groups.setdefault(hashes, [])
            group.append({'id': name + ':' + str(row['question_id']),
                          'images': [str((ROOT / im['path']).resolve()) for im in images],
                          'question': row['question']})
    if not groups:
        raise ValueError('No questions found')
    shards = [[] for _ in range(args.shards)]
    loads = [0] * args.shards
    for hashes in sorted(groups, key=lambda h: (-(4 * len(h) + len(groups[h])), h)):
        owner = min(range(args.shards), key=loads.__getitem__)
        shards[owner].extend(groups[hashes])
        loads[owner] += 4 * len(hashes) + len(groups[hashes])
    inputs = []
    for i, shard in enumerate(shards):
        path = args.work_dir / f'input{i}.jsonl'
        with path.open('x') as stream:
            for row in shard:
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
        inputs.append({'file': relative(path), 'sha256': sha256(path), 'questions': len(shard)})
    plan = {'created_utc': stamp(), 'datasets': datasets,
            'expected_images': len(groups), 'expected_questions': sum(map(len, shards)),
            'input_shards': inputs, 'data_dir': relative(args.data_dir)}
    atomic(args.work_dir / 'plan.json', plan)
    print(json.dumps({'questions': plan['expected_questions'], 'images': len(groups),
                      'shard_questions': list(map(len, shards))}))


def summarize(doc, plan, captions):
    doc.pop('comparison_sources', None)
    for name, ds in doc['datasets'].items():
        order = {qid: i for i, qid in enumerate(plan['datasets'][name]['question_ids'])}
        ds['results'].sort(key=lambda r: order[r['question_id']])
        ids = cluster_ids(ds['results'], captions)
        ds['summary'] = uncertainty([r['score'] for r in ds['results']], ids) if ids else {
            'score': None, 'se': None, 'n': 0, 'image_clusters': 0, 'ci95': None}
        ds['complete'] = len(ds['results']) == ds['expected_questions']
        ds['uncertainty_cluster'] = 'connected components of shared input images'
        if ds['metric'] == 'mme_accuracy_plus':
            manifest = load_json((ROOT / plan['datasets'][name]['manifest']))
            ds['mme'] = mme_summary(ds['results'], manifest['rows'])
        ds.pop('comparisons', None)


def export(args, records=None, replace=False, initial_documents=None):
    plan = load_json(args.plan)
    if len(args.methods) != len(set(args.methods)):
        raise ValueError('Duplicate output methods')
    requested_votes = [m for m in args.methods if base_method(m) == 'scantrust_v2']
    methods = voter_methods(args.methods)
    kinds = {random_kind(m) for m in methods}
    if len(kinds) != 1:
        raise ValueError('Export regular, rand and randimg predictions separately')
    keep_caption = kinds == {'randimg_'}
    donors, randomization = random_inputs(plan) if kinds != {''} else (None, None)
    if len(methods) != len(set(methods)):
        raise ValueError('Duplicate output methods')
    for m in methods:
        if not replace and (args.output_dir / f'{m}.json').exists():
            raise ValueError(f'Output already exists: {m}.json')
    expected, metadata = {}, {}
    for name, ds in plan['datasets'].items():
        path = ROOT / ds['manifest']
        if sha256(path) != ds['manifest_sha256']:
            raise ValueError(f'Manifest changed: {path}')
        manifest = load_json(path)
        lookup = {str(r['question_id']): r for r in manifest['rows']}
        metadata[name] = {k: v for k, v in ds.items() if k != 'question_ids'}
        for qid in ds['question_ids']:
            row = lookup[qid]
            hashes = [im['sha256'] for im in bundle(manifest, row)]
            expected[name + ':' + qid] = (name, row, hashes)
    source_files = ([{'file': relative(p), 'sha256': sha256(p)} for p in args.input]
                    if records is None else [])
    if records is None:
        records = chain.from_iterable(read_jsonl(p) for p in args.input)
    runtime_protocol = None
    documents, captions, seen = {}, {}, set()
    caption_reference = None
    if args.reference_captions:
        reference = load_json(args.reference_captions)
        if reference.get('schema') != CAPTION_SCHEMA:
            raise ValueError(f'Expected caption schema {CAPTION_SCHEMA}')
        require_compact(reference)
        caption_reference = reference['captions']
    caption_matches, caption_mismatches = 0, []
    for record in records:
        require_compact(record)
        token_count(record['caption'])
        if record['ocr'] is not None:
            token_count(record['ocr'])
        identity = record['id']
        if record.get('schema') != SCHEMA or identity not in expected or identity in seen:
            raise ValueError(f'Unexpected or duplicate prediction: {identity}')
        seen.add(identity)
        name, row, hashes = expected[identity]
        h = image_key(hashes)
        source_hashes = donors[h]['image_sha256s'] if donors is not None else hashes
        source_key = image_key(source_hashes)
        if donors is not None:
            if (record.get('source_image_sha256s') != source_hashes
                    or record['protocol'].get('randomization') != randomization
                    or bool(record['protocol'].get('keep_caption')) != keep_caption):
                raise ValueError(f'Wrong randomized image mapping: {identity}')
        elif 'source_image_sha256s' in record or 'randomization' in record['protocol']:
            raise ValueError('Cannot export randomized predictions as a regular method')
        if (record['question'] != row['question'] or record['image_sha256s'] != hashes
                or record['input_sha256'] != fingerprint({'question': row['question'], 'images': hashes})):
            raise ValueError(f'Prediction belongs to different inputs: {identity}')
        if runtime_protocol is None:
            runtime_protocol = record['protocol']
            base_protocol = {**runtime_protocol, 'datasets': metadata, 'image_size': [896, 896],
                'image_tokens': 256, 'pan_and_scan': False, 'all_models_frozen': True, 'seed': 42,
                'models': {name: {'path': runtime_protocol['models'][name],
                    'files': {'config.json': {'sha256': runtime_protocol[key]}}}
                    for name, key in (('gemma', 'sender_config_sha256'), ('granite', 'receiver_config_sha256'))},
                'model_hash_scope': 'Model config files only; model weights are supplied externally.'}
            for method in methods:
                protocol = {'base_protocol': base_protocol, 'recipes': {method: RECIPES[method]},
                    'renderer': {'path': runtime_protocol['renderer_path'], 'sha256': runtime_protocol['renderer_sha256'], 'step': 24000},
                    'prediction_sources': source_files, 'plan_source': {'file': relative(args.plan), 'sha256': sha256(args.plan)},
                    'source_sha256': {relative(p): sha256(p) for p in Path(__file__).parent.glob('*.py')},
                    'timing_scope': 'Result seconds measure Granite answer generation only; shared caption/OCR/renderer work is excluded.',
                    'uncertainty': 'Image-connected components; 10000 bootstrap draws, seed 42',
                    'question_policy': 'All sender/renderer work completes before Granite sees Q; reference answers only used for scoring.'}
                documents[method] = {'schema': EVAL_SCHEMA, 'method': method, 'created_utc': getattr(args, 'started_utc', stamp()),
                    'complete': False, 'status': 'scoring fresh predictions', 'protocol': protocol,
                    'protocol_sha256': fingerprint(protocol), 'expected_images': plan['expected_images'],
                    'prefixes': {}, 'checks': {},
                    'caption_source': runtime_protocol['caption_source'],
                    'datasets': {n: {**ds, 'results': [], 'complete': False} for n, ds in metadata.items()}}
        if record['protocol'] != runtime_protocol:
            raise ValueError('Cannot mix inference protocols')
        if h not in captions:
            captions[h] = {**record['caption'], 'image_sha256s': hashes}
            if caption_reference is not None:
                old = caption_reference.get(h if keep_caption else source_key, {})
                if all(old.get(k) == record['caption'][k] for k in ('output', 'output_tokens', 'output_sha256')):
                    caption_matches += 1
                else:
                    caption_mismatches.append(h)
        elif any(captions[h][k] != record['caption'][k] for k in ('output', 'output_tokens', 'output_sha256')):
            raise ValueError(f'Inconsistent caption for image bundle {h}')
        for method in methods:
            chosen = base_method(method)
            prediction = record['predictions'][chosen]
            prefix = dict(record['prefixes'][chosen])
            if prefix['transferred_sha256'] != prediction['prefix_sha256']:
                raise ValueError(f'Wrong prefix: {identity} / {chosen}')
            token_count(prediction)
            token_count(prediction, 'receiver_input')
            token_count(prefix, 'receiver_prefix')
            if chosen == 'textscan_v6':
                prefix['auxiliary_captions'] = [{**record['ocr'], 'prompt': runtime_protocol['ocr_prompt'],
                                                 'label': 'Additional visible text:', 'budget': 96}]
            result = {**prediction, 'question_id': str(row['question_id']), 'question': row['question'],
                'image_sha256': h, 'image_sha256s': hashes, 'caption_sha256': prefix['caption_sha256'],
                'references': row['answers'], 'score': score_answer(prediction['output'], row)}
            if donors is not None:
                result.update(source_image_sha256=source_key, source_image_sha256s=source_hashes)
            old_prefix = documents[method]['prefixes'].setdefault(h, prefix)
            if old_prefix != prefix:
                raise ValueError(f'Inconsistent image prefix: {identity} / {method}')
            documents[method]['datasets'][name]['results'].append(result)
    if not seen:
        raise ValueError('No predictions to export')
    complete = seen == set(expected) and len(captions) == plan['expected_images']
    if not complete and not args.allow_partial:
        raise ValueError(f'Incomplete predictions: {len(seen)}/{len(expected)} questions')
    for method, old in (initial_documents or {}).items():
        doc = documents[method]
        appended_protocol = doc['protocol']
        for name, old_ds in old['datasets'].items():
            new_rows = doc['datasets'][name]['results']
            old_ids = {str(r['question_id']) for r in old_ds['results']}
            if old_ids & {r['question_id'] for r in new_rows}:
                raise ValueError(f'Refusing to replace existing answers: {method}/{name}')
            doc['datasets'][name]['results'] = copy.deepcopy(old_ds['results']) + new_rows
            for r in old_ds['results']:
                hashes = expected[name + ':' + str(r['question_id'])][2]
                captions.setdefault(r['image_sha256'], {'image_sha256s': hashes})
        for h, prefix in old['prefixes'].items():
            if h in doc['prefixes'] and prefix != doc['prefixes'][h]:
                raise ValueError(f'Existing prefix changed: {method}/{h}')
            doc['prefixes'][h] = copy.deepcopy(prefix)
        doc['created_utc'] = old['created_utc']
        doc['protocol'], doc['protocol_sha256'] = old['protocol'], old['protocol_sha256']
        doc['caption_source'] = old['caption_source']
        doc['append_runs'] = old.get('append_runs', []) + [{
            'started_utc': args.started_utc, 'protocol': appended_protocol,
            'new_questions': len(seen), 'previous_questions': old.get('questions_completed', 0)}]
    for method, doc in documents.items():
        count = sum(len(ds['results']) for ds in doc['datasets'].values())
        complete = count == len(expected)
        summarize(doc, plan, captions)
        doc.update(complete=complete, status='complete' if complete else 'partial', questions_completed=count,
                   updated_utc=stamp(), final_verified=complete)
        if complete:
            doc['completed_utc'] = stamp()
        if hasattr(args, 'started_monotonic'):
            doc['last_run_seconds'] = time.monotonic() - args.started_monotonic
        saved = doc['caption_source']['kind'] == 'saved'
        doc['checks'] = {'all_inputs_match_manifests': True, 'all_predictions_scored': True,
                          'fresh_captions': 0 if saved else len(captions),
                          'reused_captions': len(captions) if saved else 0,
                          'caption_matches_reference': caption_matches,
                          'caption_mismatches_reference': caption_mismatches,
                          'reference_captions': {'file': relative(args.reference_captions), 'sha256': sha256(args.reference_captions)} if args.reference_captions else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method, doc in documents.items():
        atomic(args.output_dir / f'{method}.json', doc)
        count = doc['questions_completed']
        if doc['complete']:
            macro = sum(ds['summary']['score'] for ds in doc['datasets'].values()) / len(doc['datasets'])
            tqdm.write(f'{method}: {count} questions, macro {macro:.6f}')
        else:
            tqdm.write(f'{method}: saved {count}/{len(expected)} questions (partial)')

    for method in requested_votes:
        rebuild_votes(args.output_dir, method)


def run(args):
    if args.checkpoint_seconds <= 0:
        raise ValueError('checkpoint-seconds must be positive')
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('Duplicate output methods')
    requested_votes = [m for m in args.methods if base_method(m) == 'scantrust_v2']
    args = argparse.Namespace(**{**vars(args), 'methods': voter_methods(args.methods)})
    inference = argparse.Namespace(**vars(args))
    inference.image = inference.question_file = None
    rows = input_rows(inference)
    plan = load_json(args.plan)
    donors, randomization = random_inputs(plan) if any(random_kind(m) for m in args.methods) else (None, None)
    if donors is not None:
        if not args.captions:
            raise ValueError('Random methods require --captions; they never regenerate captions')
        args.randomization = randomization
        args.donors = donors
    expected, samples = {}, {}
    for name, ds in plan['datasets'].items():
        path = ROOT / ds['manifest']
        if sha256(path) != ds['manifest_sha256']:
            raise ValueError(f'Manifest changed: {path}')
        manifest = load_json(path)
        wanted = set(ds['question_ids'])
        for row in manifest['rows']:
            if str(row['question_id']) in wanted:
                samples[name + ':' + str(row['question_id'])] = (row, [im['sha256'] for im in bundle(manifest, row)])
                expected[name + ':' + str(row['question_id'])] = fingerprint({
                    'question': row['question'], 'images': [im['sha256'] for im in bundle(manifest, row)]})
    for row in rows:
        if expected.get(row['id']) != row['input_sha256']:
            raise ValueError(f'Input does not match evaluation plan: {row["id"]}')
    validate_metrics(row for row, _ in samples.values())
    if len(rows) != len(expected) and not args.allow_partial:
        raise ValueError('Input covers only part of the plan; pass --allow-partial for a shard')
    existing, batches = {}, {}
    requested = {r['id'] for r in rows}
    for method in args.methods:
        path = args.output_dir / f'{method}.json'
        done = set()
        if path.exists():
            doc = load_json(path)
            check_existing(doc, method, args, plan, samples)
            existing[method] = doc
            done = {n + ':' + str(r['question_id']) for n, ds in doc['datasets'].items() for r in ds['results']}
        missing = frozenset(requested - done)
        if missing:
            batches.setdefault((missing, random_kind(method)), []).append(method)
        else:
            if path.exists():
                export_tracked(path, doc)
            print(f'{method}: all requested answers already saved; skipped.')
    if any(randomized for _, randomized in batches):
        from scripts.vision.inference import CAPTION_PROMPT, OCR_PROMPT
        source_rows = [donors[image_key(r['image_sha256s'])] for r in rows] + rows
        load_captions(args.captions, source_rows, args.sender, CAPTION_PROMPT)
        if any(random_kind(m) and base_method(m) in ('textscan_v6', 'scantrust_v2') for ms in batches.values() for m in ms):
            args.ocr = getattr(args, 'ocr', None) or args.output_dir / 'textscan_v6.json'
            load_ocr(args.ocr, source_rows, args.sender, OCR_PROMPT)
    for (missing, kind), methods in batches.items():
        randomized = bool(kind)
        inference.keep_caption = kind == 'randimg_'
        inference.only_ids = missing
        inference.consensus = False
        inference.methods = [base_method(m) for m in methods]
        inference.progress_methods = methods
        inference.donors = donors if randomized else None
        inference.randomization = randomization if randomized else None
        inference.ocr = args.ocr if randomized and hasattr(args, 'ocr') else None
        inference.resume = False
        inference.output = None
        args.started_utc, args.started_monotonic = stamp(), time.monotonic()
        scoring = argparse.Namespace(**vars(args))
        scoring.methods = methods
        scoring.reference_captions = args.captions
        scoring.allow_partial = True
        initial = {m: existing[m] for m in methods if m in existing}
        records, saved_count = [], 0
        last_saved = time.monotonic()

        def save():
            nonlocal last_saved, saved_count
            export(scoring, records=records, replace=True, initial_documents=initial)
            saved_count, last_saved = len(records), time.monotonic()

        def accept(record):
            records.append(record)
            if len(records) == 1 or time.monotonic() - last_saved >= args.checkpoint_seconds:
                save()

        try:
            if getattr(args, 'gpus', None):
                jobs = {}
                for row in rows:
                    if row['id'] in missing:
                        source = donors[image_key(row['image_sha256s'])] if randomized else row
                        h = image_key(source['image_sha256s'])
                        job = jobs.setdefault(h, {'ids': [], 'weight': 4 * len(source['image_sha256s'])})
                        job['ids'].append(row['id'])
                        job['weight'] += 1
                with closing(stream(prediction_worker, inference, list(jobs.values()), args.gpus)) as predictions:
                    for record in predictions:
                        accept(record)
            else:
                predict(inference, sink=accept)
        finally:
            if len(records) > saved_count:
                save()

    for method in requested_votes:
        rebuild_votes(args.output_dir, method)


def check_existing(doc, method, args, plan, samples):
    from scripts.vision.inference import CAPTION_PROMPT, INSTRUCTION, SYSTEM_DATE, gemma_control_recipe
    if doc.get('schema') != EVAL_SCHEMA:
        raise ValueError(f'Expected eval schema {EVAL_SCHEMA}')
    require_compact(doc)
    protocol = doc['protocol']
    base = protocol if method in BASELINES else protocol['base_protocol']
    if random_kind(method) and base.get('randomization') != args.randomization:
        raise ValueError('Existing randomized donor mapping differs')
    if bool(base.get('keep_caption')) != (random_kind(method) == 'randimg_'):
        raise ValueError('Existing caption-source policy differs')
    if random_kind(method):
        if base['caption_source']['sha256'] != sha256(args.captions):
            raise ValueError('Existing donor caption cache differs')
        if base_method(method) in ('textscan_v6', 'scantrust_v2'):
            ocr_path = getattr(args, 'ocr', None) or args.output_dir / 'textscan_v6.json'
            if base.get('ocr_source', {}).get('sha256') != sha256(ocr_path):
                raise ValueError('Existing donor OCR cache differs')
    if doc['method'] != method:
        raise ValueError('Existing method differs')
    if method == 't2tplus' or method in GEMMA_CONTROLS:
        sources = [('caption_source', args.captions)]
        if method in OCR_BASELINES:
            sources.append(('ocr_source', args.ocr))
        for key, path in sources:
            if doc.get(key, {}).get('sha256') != sha256(path):
                raise ValueError(f'Existing {method} {key} differs')
    if method in GEMMA_CONTROLS and base.get('gemma_control') != gemma_control_recipe(method):
        raise ValueError(f'Existing {method} recipe differs')
    fixed = {'caption_prompt': CAPTION_PROMPT, 'caption_max_tokens': 256,
             'instruction': INSTRUCTION, 'answer_max_tokens': 32,
             'dtype': 'bfloat16'}
    if method not in GEMMA_BASELINES:
        fixed['granite_system_date'] = SYSTEM_DATE
    if any(base.get(k) != v for k, v in fixed.items()):
        raise ValueError('Existing prompt/decoding configuration differs')
    for name, path in (('gemma', args.sender), ('granite', args.receiver)):
        if base['models'][name]['files']['config.json']['sha256'] != sha256(path / 'config.json'):
            raise ValueError(f'Existing {name} configuration differs')
    if method not in BASELINES:
        if protocol['renderer']['sha256'] != sha256(args.renderer):
            raise ValueError('Existing renderer differs')
        recipe = protocol['recipes'][method]
        if any(recipe.get(k) != v for k, v in RECIPES[method].items()):
            raise ValueError(f'Existing recipe differs: {method}')
    seen = set()
    for name, ds in doc['datasets'].items():
        if name not in plan['datasets']:
            raise ValueError(f'Existing dataset is absent from the plan: {name}')
        for r in ds['results']:
            token_count(r)
            identity = name + ':' + str(r['question_id'])
            if identity in seen or identity not in samples:
                raise ValueError(f'Duplicate or out-of-plan saved question: {identity}')
            seen.add(identity)
            row, hashes = samples[identity]
            if random_kind(method):
                donor = args.donors[image_key(hashes)]['image_sha256s']
                if r.get('source_image_sha256s') != donor or r.get('source_image_sha256') != image_key(donor):
                    raise ValueError(f'Saved donor image changed: {identity}')
            if r['question'] != row['question'] or r['references'] != row['answers'] or r['image_sha256'] != image_key(hashes):
                raise ValueError(f'Saved question inputs changed: {identity}')


def baselines(args):
    from scripts.vision.inference import Engine, CAPTION_PROMPT, OCR_PROMPT, INSTRUCTION, SYSTEM_DATE, gemma_control_recipe
    if args.limit < 1 or len(args.methods) != len(set(args.methods)):
        raise ValueError('Positive limit and unique methods required')
    batch_size = getattr(args, 'batch_size', 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('--batch-size must be a positive integer')
    datasets, samples, groups = {}, {}, {}
    for path in sorted(args.data_dir.glob('*/questions.json')):
        if any(part.startswith('.') for part in path.parts):
            continue
        manifest = load_json(path)
        rows = manifest['rows'][:args.limit]
        name = path.parent.name
        datasets[name] = {'name': manifest['dataset'], 'source_split': manifest['split'],
            'manifest': relative(path), 'manifest_sha256': sha256(path), 'metric': manifest['metric'],
            'expected_questions': len(rows), 'question_ids': [str(r['question_id']) for r in rows]}
        for row in rows:
            identity = name + ':' + str(row['question_id'])
            if identity in samples:
                raise ValueError(f'Duplicate question: {identity}')
            images = bundle(manifest, row)
            hashes = [im['sha256'] for im in images]
            samples[identity] = (row, hashes)
            group = groups.setdefault(image_key(hashes), {'image_sha256s': hashes,
                'paths': [ROOT / im['path'] for im in images], 'questions': []})
            group['questions'].append((name, identity, row))
    if not samples:
        raise ValueError('No questions found')
    validate_metrics(row for row, _ in samples.values())
    transcripts = {}
    if set(args.methods) & set(OCR_BASELINES):
        args.ocr = getattr(args, 'ocr', None) or args.output_dir.parent.parent / 'renderer' / args.output_dir.name / 'textscan_v6.json'
        transcripts = load_ocr(args.ocr, list(groups.values()), args.sender, OCR_PROMPT)
    plan = {'datasets': datasets}
    metadata = {n: {k: v for k, v in ds.items() if k != 'question_ids'} for n, ds in datasets.items()}
    protocol = {'models': {name: {'path': str(path.resolve()),
                    'files': {'config.json': {'sha256': sha256(path / 'config.json')}}}
                    for name, path in (('gemma', args.sender), ('granite', args.receiver))},
                'datasets': metadata, 'caption_prompt': CAPTION_PROMPT, 'caption_max_tokens': 256,
                'instruction': INSTRUCTION, 'answer_max_tokens': 32, 'granite_system_date': SYSTEM_DATE,
                'dtype': 'bfloat16', 'decoding': 'greedy split-cache; original caption recipe',
                'image_size': [896, 896], 'image_tokens': 256, 'pan_and_scan': False,
                'all_models_frozen': True, 'seed': 42,
                'execution': {'gemma_batch_size': batch_size, 'gemma_batch_backend': 'left_pad_split_cache_v1',
                              'batched_seconds_scope': 'batch wall time divided by actual batch size'},
                'source_sha256': {relative(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    documents, done = {}, {}
    for method in args.methods:
        method_protocol = copy.deepcopy(protocol)
        if method in GEMMA_CONTROLS:
            method_protocol['gemma_control'] = gemma_control_recipe(method)
        path = args.output_dir / f'{method}.json'
        if path.exists():
            doc = load_json(path)
            check_existing(doc, method, args, plan, samples)
            have = {n + ':' + str(r['question_id']) for n, ds in doc['datasets'].items() for r in ds['results']}
            if have == set(samples):
                export_tracked(path, doc)
                print(f'{method}: all {len(have)} answers already saved; skipped.')
                continue
            doc.setdefault('append_runs', []).append({'started_utc': stamp(), 'protocol': method_protocol})
        else:
            doc = {'schema': EVAL_SCHEMA, 'method': method, 'created_utc': stamp(),
                   'protocol': method_protocol, 'protocol_sha256': fingerprint(method_protocol), 'datasets': {}}
            have = set()
        for name, ds in metadata.items():
            doc['datasets'].setdefault(name, {'results': []}).update(ds)
        documents[method], done[method] = doc, have
    if not documents:
        return
    cached = load_captions(args.captions, list(groups.values()), args.sender, CAPTION_PROMPT) if set(documents) & set(CAPTION_BASELINES) else {}
    for method in CAPTION_BASELINES:
        if method in documents:
            documents[method]['caption_source'] = {'file': relative(args.captions), 'sha256': sha256(args.captions)}
    for method in OCR_BASELINES:
        if method in documents:
            documents[method]['ocr_source'] = {'file': relative(args.ocr), 'sha256': sha256(args.ocr),
                'prompt': OCR_PROMPT, 'ocr_tokens': [64, 32], 'label': 'Additional visible text:'}
    import torch
    if not getattr(args, 'gpus', None) and (not args.device.startswith('cuda') or not torch.cuda.is_available()):
        raise ValueError('Baseline inference requires a CUDA GPU')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = last_saved = time.monotonic()
    dirty = set()

    def save():
        for method in list(dirty):
            doc = documents[method]
            summarize(doc, plan, groups)
            complete = done[method] == set(samples)
            doc.update(complete=complete, status='complete' if complete else 'partial',
                       questions_completed=len(done[method]), updated_utc=stamp(),
                       final_verified=complete, last_run_seconds=time.monotonic()-started)
            doc.pop('completed_utc', None)
            if complete:
                doc['completed_utc'] = stamp()
            atomic(args.output_dir / f'{method}.json', doc)
        dirty.clear()

    def accept(identity, method, result):
        result = compact(result)
        row, hashes = samples[identity]
        name = identity.split(':', 1)[0]
        h = image_key(hashes)
        result.update(question_id=str(row['question_id']), question=row['question'],
                      image_sha256=h, image_sha256s=hashes,
                      references=row['answers'], score=score_answer(result['output'], row))
        if method in CAPTION_BASELINES:
            import hashlib
            result['caption_sha256'] = hashlib.sha256(cached[h]['output'].encode()).hexdigest()
            if method in OCR_BASELINES:
                result['ocr_sha256'] = hashlib.sha256(transcripts[h]['output'].encode()).hexdigest()
            if method == 'gemma_cot':
                expected_budget = token_count(cached[h]) + token_count(transcripts[h])
                if result['reasoning_budget'] != expected_budget or token_count(result['reasoning']) > expected_budget:
                    raise ValueError('Gemma reasoning does not match the saved S+OCR budget')
        documents[method]['datasets'][name]['results'].append(result)
        done[method].add(identity)
        dirty.add(method)
        print(f'{method}: {len(done[method])}/{len(samples)}', flush=True)

    jobs = []
    for h, group in groups.items():
        questions = [{'id': identity, 'question': row['question'],
                      'methods': [m for m in documents if identity not in done[m]]}
                     for _, identity, row in group['questions']]
        questions = [q for q in questions if q['methods']]
        if questions:
            jobs.append({'images': [str(p) for p in group['paths']],
                         'image_sha256s': group['image_sha256s'], 'questions': questions,
                         'caption': cached[h]['output'] if h in cached else None,
                         'ocr': transcripts[h]['output'] if h in transcripts else None,
                         'reasoning_budget': (token_count(cached[h]) + token_count(transcripts[h])
                                              if 'gemma_cot' in documents else None),
                         'weight': 4 * len(group['paths']) + sum(len(q['methods']) for q in questions)})
    if getattr(args, 'gpus', None):
        results = stream(baseline_worker, args, jobs, args.gpus)
    else:
        engine = Engine.baseliner(str(args.sender), str(args.receiver), list(documents), args.device)
        results = baseline_results(engine, jobs, batch_size)
    try:
        with closing(results):
            for identity, method, result in results:
                accept(identity, method, result)
                if time.monotonic() - last_saved >= 60:
                    save()
                    last_saved = time.monotonic()
    finally:
        save()


def merge(args):
    requested_votes = [m for m in args.methods if base_method(m) == 'scantrust_v2']
    args = argparse.Namespace(**{**vars(args), 'methods': voter_methods(args.methods)})
    documents = {}
    for directory in args.input_dirs:
        for method in args.methods:
            path = directory / f'{method}.json'
            shard = load_json(path)
            if shard.get('schema') != EVAL_SCHEMA:
                raise ValueError(f'Expected eval schema {EVAL_SCHEMA}: {path}')
            require_compact(shard)
            if shard['method'] != method:
                raise ValueError(f'Wrong method: {path}')
            if (args.output_dir / f'{method}.json').exists():
                raise ValueError(f'Output already exists: {method}.json')
            if method not in documents:
                documents[method] = shard
                continue
            doc = documents[method]
            for key in ('protocol', 'protocol_sha256', 'caption_source', 'expected_images'):
                if doc[key] != shard[key]:
                    raise ValueError(f'Shard {key} mismatch: {path}')
            if set(doc['datasets']) != set(shard['datasets']):
                raise ValueError(f'Shard dataset mismatch: {path}')
            for name, ds in shard['datasets'].items():
                for key in ('manifest_sha256', 'expected_questions', 'metric'):
                    if ds[key] != doc['datasets'][name][key]:
                        raise ValueError(f'Shard dataset metadata mismatch: {path}/{name}')
                doc['datasets'][name]['results'].extend(ds['results'])
            for h, prefix in shard['prefixes'].items():
                if doc['prefixes'].setdefault(h, prefix) != prefix:
                    raise ValueError(f'Shard prefix mismatch: {h}')
    for method, doc in documents.items():
        src = doc['protocol']['plan_source']
        plan_path = ROOT / src['file']
        if sha256(plan_path) != src['sha256']:
            raise ValueError('Evaluation plan changed')
        plan = load_json(plan_path)
        captions = {}
        for name, ds in doc['datasets'].items():
            wanted = plan['datasets'][name]
            if sha256(ROOT / wanted['manifest']) != wanted['manifest_sha256']:
                raise ValueError(f'Manifest changed: {name}')
            ids = [r['question_id'] for r in ds['results']]
            if len(set(ids)) != len(ids) or set(ids) != set(wanted['question_ids']):
                raise ValueError(f'Incomplete or overlapping shards: {method}/{name}')
            for row in ds['results']:
                captions[row['image_sha256']] = {'image_sha256s': row['image_sha256s']}
        if len(captions) != plan['expected_images'] or set(captions) != set(doc['prefixes']):
            raise ValueError(f'Image coverage mismatch: {method}')
        summarize(doc, plan, captions)
        doc.update(complete=True, status='complete', final_verified=True,
                   questions_completed=sum(len(ds['results']) for ds in doc['datasets'].values()),
                   updated_utc=stamp(), completed_utc=stamp())
        doc.pop('last_run_seconds', None)
        doc['checks'] = {'all_inputs_match_manifests': True, 'all_predictions_scored': True,
                         'disjoint_shards_complete': True,
                         'fresh_captions': len(captions) if doc['caption_source']['kind'] == 'freshly_generated' else 0,
                         'reused_captions': len(captions) if doc['caption_source']['kind'] == 'saved' else 0}
        doc['merged_sources'] = [{'file': relative(d / f'{method}.json'), 'sha256': sha256(d / f'{method}.json')}
                                 for d in args.input_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method, doc in documents.items():
        atomic(args.output_dir / f'{method}.json', doc)
        print(f'{method}: merged {doc["questions_completed"]} questions')

    for method in requested_votes:
        rebuild_votes(args.output_dir, method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    vote = sub.add_parser('consensus', help='Recompute ScanTrust from saved per-method voter files; CPU only')
    vote.add_argument('--output-dir', type=Path, required=True)
    vote.add_argument('--methods', nargs='+', choices=['scantrust_v2', 'rand_scantrust_v2'], default=['scantrust_v2'])
    cap = sub.add_parser('captions', help='Generate only missing Gemma captions; skip a complete cache')
    cap.add_argument('--data-dir', type=Path, required=True)
    cap.add_argument('--output', type=Path, required=True)
    cap.add_argument('--sender', type=Path, required=True)
    cap.add_argument('--device', default='cuda:0')
    cap.add_argument('--limit', type=int, default=200)
    cap.add_argument('--batch-size', type=int, default=1,
                     help='Image bundles per Gemma forward pass; 1 keeps the original serial decoding (default: 1)')
    base = sub.add_parser('baselines', help='Append missing vanilla baselines or Gemma S/ST/CoT controls')
    base.add_argument('--data-dir', type=Path, required=True)
    base.add_argument('--output-dir', type=Path, required=True)
    base.add_argument('--sender', type=Path, required=True)
    base.add_argument('--receiver', type=Path, required=True)
    base.add_argument('--captions', type=Path, required=True)
    base.add_argument('--methods', nargs='+', choices=BASELINES, default=['granite_only', 'gemma_only', 't2t'])
    base.add_argument('--ocr', type=Path, help='Saved TextScan OCR for t2tplus, gemma_st and gemma_cot; default: sibling renderer/MODE/textscan_v6.json')
    base.add_argument('--limit', type=int, default=200)
    base.add_argument('--device', default='cuda:0')
    base.add_argument('--batch-size', type=int, default=1,
                      help='Gemma examples per GPU forward pass; Granite baselines remain unbatched (default: 1)')
    prep = sub.add_parser('prepare', help='Make question-only JSONL shards and a manifest plan')
    prep.add_argument('--data-dir', type=Path, required=True)
    prep.add_argument('--work-dir', type=Path, required=True)
    prep.add_argument('--shards', type=int, default=1)
    prep.add_argument('--limit', type=int, default=200)
    score = sub.add_parser('export', help='Score complete fresh prediction shards into per-method eval JSONs')
    score.add_argument('--plan', type=Path, required=True)
    score.add_argument('--input', type=Path, nargs='+', required=True)
    score.add_argument('--output-dir', type=Path, required=True)
    score.add_argument('--methods', nargs='+', choices=RECIPES, default=['kvgain_v3', 'textscan_v6', 'scantrust_v2'])
    score.add_argument('--reference-captions', type=Path, help='Optional caption text/token-hash verification; never used for inference')
    score.add_argument('--allow-partial', action='store_true', help='Export available records with complete=false')
    direct = sub.add_parser('run', help='Run inference and checkpoint per-method eval JSONs directly')
    direct.add_argument('--plan', type=Path, required=True)
    direct.add_argument('--input', type=Path, required=True, help='Question-only JSONL from prepare')
    direct.add_argument('--output-dir', type=Path, required=True)
    direct.add_argument('--methods', nargs='+', choices=RECIPES, default=['kvgain_v3', 'textscan_v6', 'scantrust_v2'])
    source = direct.add_mutually_exclusive_group(required=True)
    source.add_argument('--captions', type=Path, help='Existing no_training/eval/gemma_captions.json; reused verbatim')
    source.add_argument('--fresh-captions', action='store_true', help='Explicitly regenerate captions for new images')
    direct.add_argument('--sender', type=Path, required=True)
    direct.add_argument('--receiver', type=Path, required=True)
    direct.add_argument('--renderer', type=Path, required=True)
    direct.add_argument('--ocr', type=Path, help='Saved TextScan OCR for rand methods; default: OUTPUT_DIR/textscan_v6.json')
    direct.add_argument('--device', default='cuda:0')
    direct.add_argument('--allow-partial', action='store_true', help='Input is one shard of the plan; retain complete=false')
    direct.add_argument('--checkpoint-seconds', type=float, default=60)
    joined = sub.add_parser('merge', help='Merge disjoint per-method eval JSON shards; CPU only')
    joined.add_argument('--input-dirs', type=Path, nargs='+', required=True)
    joined.add_argument('--output-dir', type=Path, required=True)
    joined.add_argument('--methods', nargs='+', choices=RECIPES, default=['kvgain_v3', 'textscan_v6', 'scantrust_v2'])
    for child in (cap, base, direct):
        child.add_argument('--gpus', type=gpu_list, help='Physical GPUs, e.g. 0,1,2; one worker per GPU, automatic image sharding')
    args = parser.parse_args()
    try:
        if args.command == 'consensus':
            for method in args.methods:
                report = rebuild_votes(args.output_dir, method)
                print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in report.items()}))
            return
        {'captions': captions, 'baselines': baselines, 'prepare': prepare, 'export': export, 'run': run, 'merge': merge}[args.command](args)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
