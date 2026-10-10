import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.vision.consensus import VOTERS, consensus
from scripts.vision.storage import load as load_json, file_sha256

SCHEMA = 'scantrust-predictions-v2'
EVAL_SCHEMA = 'vision-eval-v2'
CAPTION_SCHEMA = 'vision-captions-v2'
GEMMA_CONTROLS = ('gemma_s', 'gemma_st', 'gemma_cot')
GEMMA_BASELINES = ('gemma_only', *GEMMA_CONTROLS)
CAPTION_BASELINES = ('t2t', 't2tplus', *GEMMA_CONTROLS)
OCR_BASELINES = ('t2tplus', 'gemma_st', 'gemma_cot')
BASELINES = ('granite_only', 'gemma_only', 't2t', 't2tplus', *GEMMA_CONTROLS)
COMPACT_FIELDS = {
    'output_ids': ('output_tokens', 'output_sha256'),
    'input_ids': ('input_tokens', 'input_sha256'),
    'receiver_input_ids': ('receiver_input_tokens', 'receiver_input_sha256'),
    'receiver_prefix_ids': ('receiver_prefix_tokens', 'receiver_prefix_sha256'),
    'permitted_source_positions': ('permitted_source_count', 'permitted_source_sha256'),
    'selected_source_positions': ('selected_source_count', 'selected_source_sha256'),
}


def sha256(path):
    return file_sha256(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(value):
    if isinstance(value, list):
        return [compact(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: compact(item) for key, item in value.items() if key not in COMPACT_FIELDS}
    for key, (count_key, hash_key) in COMPACT_FIELDS.items():
        if key not in value:
            continue
        items = value[key]
        if not isinstance(items, list) or any(type(item) is not int for item in items):
            raise ValueError(f'Expected integer array: {key}')
        for target, summary in ((count_key, len(items)), (hash_key, fingerprint(items))):
            if target in result and result[target] != summary:
                raise ValueError(f'Inconsistent runtime metadata: {target}')
            result[target] = summary
    return result


def require_compact(value):
    if isinstance(value, dict):
        if COMPACT_FIELDS.keys() & value.keys():
            raise ValueError('Expected compact data; token/position arrays are not supported in saved files')
        for item in value.values():
            require_compact(item)
    elif isinstance(value, list):
        for item in value:
            require_compact(item)


def token_count(record, name='output'):
    count, digest = record[name + '_tokens'], record[name + '_sha256']
    if (type(count) is not int or count < 0 or not isinstance(digest, str)
            or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError(f'Invalid compact {name} token metadata')
    return count


def image_key(hashes):
    return hashes[0] if len(hashes) == 1 else hashlib.sha256(
        json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()


def load_captions(path, rows, sender, caption_prompt):
    doc = load_json(path)
    return validate_captions(doc, rows, sender, caption_prompt)


def load_ocr(path, rows, sender, prompt):
    doc = load_json(path)
    require_compact(doc)
    if doc.get('schema') != EVAL_SCHEMA or doc.get('method') != 'textscan_v6':
        raise ValueError(f'Expected compact textscan_v6 eval JSON: {path}')
    base = doc['protocol']['base_protocol']
    if (base['models']['gemma']['files']['config.json']['sha256'] != sha256(sender / 'config.json')
            or base['ocr_prompt'] != prompt or base['ocr_tokens'] != [64, 32]):
        raise ValueError('Saved OCR does not match the fixed Gemma OCR recipe')
    transcripts = {}
    for row in rows:
        key = image_key(row['image_sha256s'])
        if key in transcripts:
            continue
        auxiliary = doc['prefixes'].get(key, {}).get('auxiliary_captions', [])
        if len(auxiliary) != 1:
            raise ValueError(f'OCR missing for donor image bundle {key}; refusing to regenerate')
        saved = auxiliary[0]
        if (not isinstance(saved['output'], str) or token_count(saved) > 96
                or saved['prompt'] != prompt or saved['budget'] != 96):
            raise ValueError(f'Invalid saved OCR for image bundle {key}')
        transcripts[key] = {k: saved[k] for k in ('output', 'output_tokens', 'output_sha256', 'stop_reason')}
    return transcripts


def validate_captions(doc, rows, sender, caption_prompt, require_complete=True):
    if doc.get('schema') != CAPTION_SCHEMA:
        raise ValueError(f'Expected caption schema {CAPTION_SCHEMA}')
    require_compact(doc)
    protocol = doc['protocol']
    if ((require_complete and not doc['complete']) or protocol['caption_prompt'] != caption_prompt
            or protocol['caption_max_tokens'] != 256
            or protocol['models']['gemma']['files']['config.json']['sha256'] != sha256(sender / 'config.json')):
        raise ValueError('Saved captions do not match the fixed Gemma caption recipe')
    captions = {}
    for row in rows:
        key = image_key(row['image_sha256s'])
        if key in captions:
            continue
        if key not in doc['captions']:
            raise ValueError(f'Caption missing for image bundle {key}; refusing to regenerate')
        saved = doc['captions'][key]
        if (not isinstance(saved['output'], str) or not saved['output'].strip()
                or not 0 < token_count(saved) <= 256
                or hashlib.sha256(saved['output'].encode()).hexdigest() != saved['caption_sha256']
                or saved['image_sha256'] != key):
            raise ValueError(f'Invalid saved caption: {key}')
        captions[key] = {k: saved[k] for k in ('output', 'output_tokens', 'output_sha256', 'stop_reason')}
    return captions


def read_jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Expected a JSON object')
                    yield row
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{path}:{number}: {exc}') from exc


def write_row(stream, row):
    stream.write(json.dumps(compact(row), ensure_ascii=False, separators=(',', ':')) + '\n')
    stream.flush()


def input_rows(args):
    if args.input:
        if args.image or args.question_file:
            raise ValueError('Use --input OR --image and --question-file')
        rows = list(read_jsonl(args.input))
        base = args.input.resolve().parent
    else:
        if not args.image or not args.question_file:
            raise ValueError('Supply --input or both --image and --question-file')
        rows = [{'id': '0', 'images': args.image, 'question': args.question_file.read_text().strip()}]
        base = Path.cwd()
    if not rows:
        raise ValueError('No input questions')
    seen = set()
    for row in rows:
        if 'id' not in row or not isinstance(row['id'], (str, int)):
            raise ValueError('Every question needs a string or integer id')
        row['id'] = str(row['id'])
        if row['id'] in seen:
            raise ValueError(f'Duplicate input id: {row["id"]}')
        seen.add(row['id'])
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'Empty or invalid question: {row["id"]}')
        paths = row.get('images') or ([row['image']] if 'image' in row else [])
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise ValueError('Supply image (string) or images (nonempty ordered list)')
        row['images'] = [str((base / p).resolve()) for p in paths]
        row['image_sha256s'] = [sha256(p) for p in row['images']]
        row['input_sha256'] = fingerprint({'question': row['question'], 'images': row['image_sha256s']})
    return rows


def predict(args, sink=None):
    rows = input_rows(args)
    if hasattr(args, 'only_ids'):
        rows = [r for r in rows if r['id'] in args.only_ids]
    if not rows:
        return
    methods = args.methods or list(VOTERS)
    if len(set(methods)) != len(methods):
        raise ValueError('Duplicate methods')
    if args.consensus and set(methods) != set(VOTERS):
        raise ValueError('--consensus requires all four voters (omit --methods)')
    for path in (args.sender, args.receiver):
        if not (path / 'config.json').is_file():
            raise ValueError(f'Expected a local model directory: {path}')
    if not args.renderer.is_file():
        raise ValueError(f'Renderer not found: {args.renderer}')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    from PIL import Image
    import torch
    import transformers
    from scripts.vision.inference import Engine, CAPTION_PROMPT, OCR_PROMPT, INSTRUCTION, SYSTEM_DATE
    donors = getattr(args, 'donors', None)
    keep_caption = donors is not None and getattr(args, 'keep_caption', False)
    source_rows = [donors[image_key(r['image_sha256s'])] for r in rows] if donors is not None else rows
    if keep_caption:
        source_rows = rows
    captions = load_captions(args.captions, source_rows, args.sender, CAPTION_PROMPT) if args.captions else None
    ocr_path = getattr(args, 'ocr', None) if 'textscan_v6' in methods else None
    transcripts = load_ocr(ocr_path, source_rows, args.sender, OCR_PROMPT) if ocr_path else None
    if donors is not None and (captions is None or ('textscan_v6' in methods and transcripts is None)):
        raise ValueError('Random methods require saved donor captions and, for TextScan, saved OCR')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Fresh inference requires a CUDA GPU; saved consensus does not')
    protocol = {
        'recipe': 'scantrust-v2-voters-v1', 'renderer_sha256': sha256(args.renderer),
        'renderer_path': str(args.renderer.resolve()),
        'sender_config_sha256': sha256(args.sender / 'config.json'),
        'receiver_config_sha256': sha256(args.receiver / 'config.json'),
        'caption_prompt': CAPTION_PROMPT, 'caption_max_tokens': 256,
        'ocr_prompt': OCR_PROMPT, 'ocr_tokens': [64, 32], 'instruction': INSTRUCTION,
        'answer_max_tokens': 32, 'granite_system_date': SYSTEM_DATE,
        'decoding': 'greedy', 'dtype': 'bfloat16',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'models': {'gemma': str(args.sender.resolve()), 'granite': str(args.receiver.resolve())},
        'caption_source': {'kind': 'saved', 'file': str(args.captions.resolve()), 'sha256': sha256(args.captions)}
                          if args.captions else {'kind': 'freshly_generated'},
        'source_sha256': {str(p.resolve().relative_to(ROOT)): sha256(p) for p in
                         (Path(__file__), Path(__file__).with_name('inference.py'), Path(__file__).with_name('consensus.py'))},
    }
    if donors is not None:
        protocol['randomization'] = args.randomization
    if keep_caption:
        protocol['keep_caption'] = True
    if ocr_path:
        protocol['ocr_source'] = {'file': str(ocr_path.resolve()), 'sha256': sha256(ocr_path)}
    completed_ids = set()
    if args.resume and args.output.exists():
        expected = {r['id']: r for r in rows}
        for old in read_jsonl(args.output):
            require_compact(old)
            if (old.get('schema') != SCHEMA or old['id'] not in expected or old['input_sha256'] != expected[old['id']]['input_sha256']
                    or old['protocol'] != protocol or set(old['predictions']) != set(methods)
                    or old['id'] in completed_ids or bool(old.get('consensus')) != args.consensus):
                raise ValueError('Resume file does not match the requested inputs and protocol')
            completed_ids.add(old['id'])
    if len(completed_ids) == len(rows):
        print('All predictions already complete.', file=sys.stderr)
        return
    engine = Engine(str(args.sender), str(args.receiver), args.renderer, args.device)
    groups = {}
    for row in rows:
        if row['id'] not in completed_ids:
            source = donors[image_key(row['image_sha256s'])] if donors is not None else row
            groups.setdefault(tuple(source['image_sha256s']), []).append(row)
    from scripts.vision.parallel import prediction_progress
    with (nullcontext(None) if sink else args.output.open('a' if args.resume else 'x')) as stream, \
            prediction_progress(args, len(rows), initial=len(completed_ids)) as progress:
        for group in groups.values():
            source = donors[image_key(group[0]['image_sha256s'])] if donors is not None else group[0]
            source_key = image_key(source['image_sha256s'])
            images = []
            for path, expected_hash in zip(source['images'], source['image_sha256s']):
                if sha256(path) != expected_hash:
                    raise ValueError(f'Image hash mismatch: {path}')
                with Image.open(path) as im:
                    images.append(im.convert('RGB'))
            text_key = image_key(group[0]['image_sha256s']) if keep_caption else source_key
            caption = captions[text_key] if captions is not None else None
            extra = {'ocr': transcripts[text_key]} if transcripts is not None else {}
            prepared = engine.prepare(images[0] if len(images) == 1 else images, methods, caption=caption, **extra)
            for row in group:
                predictions = engine.answer(prepared, row['question'])
                result = compact({'schema': SCHEMA, 'id': row['id'], 'input_sha256': row['input_sha256'],
                    'question': row['question'], 'image_sha256s': row['image_sha256s'],
                    'protocol': protocol, 'caption': prepared['caption'], 'ocr': prepared['ocr'],
                    'prefixes': {m: info for m, (_, info, _) in prepared['prefixes'].items()},
                    'predictions': predictions})
                if donors is not None:
                    result['source_image_sha256s'] = source['image_sha256s']
                if args.consensus:
                    result['consensus'] = consensus(result['predictions'], token_count(result['ocr']))
                if sink:
                    sink(result)
                else:
                    write_row(stream, result)
                progress.update()
            del prepared


def saved_consensus(args):
    combined = {}
    for path in args.input:
        seen = set()
        for row in read_jsonl(path):
            require_compact(row)
            if row.get('schema') != SCHEMA:
                raise ValueError(f'{path}: expected predictions generated by this runner')
            identity = str(row['id'])
            if identity in seen:
                raise ValueError(f'{path}: duplicate id {identity}')
            seen.add(identity)
            if not row.get('predictions') or set(row['predictions']) - set(VOTERS):
                raise ValueError(f'{path}: invalid voter predictions for {identity}')
            expected = fingerprint({'question': row['question'], 'images': row['image_sha256s']})
            if row['input_sha256'] != expected:
                raise ValueError(f'{path}: inconsistent input fingerprint for {identity}')
            ocr = row.get('ocr')
            if 'textscan_v6' in row['predictions'] and not isinstance(ocr, dict):
                raise ValueError(f'{path}: TextScan prediction requires OCR metadata')
            if ocr is not None:
                token_count(ocr)
            for prediction in row['predictions'].values():
                token_count(prediction)
            if identity not in combined:
                combined[identity] = row
                continue
            dest = combined[identity]
            for key in ('input_sha256', 'question', 'image_sha256s', 'source_image_sha256s', 'protocol', 'caption'):
                if dest.get(key) != row.get(key):
                    raise ValueError(f'Cannot combine {identity}: mismatched {key}')
            if row.get('ocr') is not None:
                if dest.get('ocr') is not None and dest['ocr'] != row['ocr']:
                    raise ValueError(f'Cannot combine {identity}: mismatched OCR')
                dest['ocr'] = row['ocr']
            for method, prediction in row['predictions'].items():
                stable = lambda p: {k: v for k, v in p.items() if k != 'seconds'}
                if method in dest['predictions'] and stable(dest['predictions'][method]) != stable(prediction):
                    raise ValueError(f'Cannot combine {identity}: conflicting {method} predictions')
                dest['predictions'][method] = prediction
            for method, prefix in row.get('prefixes', {}).items():
                old = dest.setdefault('prefixes', {}).get(method)
                if old is not None and old != prefix:
                    raise ValueError(f'Cannot combine {identity}: conflicting {method} prefix')
                dest['prefixes'][method] = prefix
    if not combined:
        raise ValueError('No saved predictions')
    for row in combined.values():
        if not row.get('ocr'):
            raise ValueError(f'Missing TextScan OCR metadata for {row["id"]}')
        row['consensus'] = consensus(row['predictions'], token_count(row['ocr']))
    with args.output.open('x') as stream:
        for row in combined.values():
            write_row(stream, row)
    print(f'Voted on {len(combined)} questions; no model inference.', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    fresh = sub.add_parser('predict', help='Generate voter predictions, optionally also consensus')
    fresh.add_argument('--input', type=Path, help='JSONL: id, image or images, question')
    fresh.add_argument('--image', nargs='+', help='One image, or an ordered image bundle')
    fresh.add_argument('--question-file', type=Path)
    fresh.add_argument('--sender', type=Path, required=True)
    fresh.add_argument('--receiver', type=Path, required=True)
    fresh.add_argument('--renderer', type=Path, required=True)
    fresh.add_argument('--captions', type=Path, help='Reuse gemma_captions.json verbatim; missing captions are an error')
    fresh.add_argument('--device', default='cuda:0')
    fresh.add_argument('--methods', nargs='+', choices=VOTERS, help='Default: all four voters')
    fresh.add_argument('--consensus', action='store_true', help='Also output ScanTrust/v2 consensus')
    fresh.add_argument('--resume', action='store_true', help='Continue a complete-line JSONL journal with matching inputs and protocol')
    saved = sub.add_parser('consensus', help='Vote using saved voter JSONL files; CPU only')
    saved.add_argument('--input', type=Path, nargs='+', required=True)
    for child in (fresh, saved):
        child.add_argument('--output', type=Path, required=True, help='New JSONL output file')
    args = parser.parse_args()
    if args.output.exists() and not getattr(args, 'resume', False):
        parser.error(f'Output already exists: {args.output}')
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (predict if args.action == 'predict' else saved_consensus)(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
