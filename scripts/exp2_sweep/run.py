from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multiagent import paths
from multiagent.paths import DATASETS_ROOT, PYTHON

CONFIG = Path(__file__).with_name('config.json')
DATASET_ROOT = str(DATASETS_ROOT)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def configure_renderer_models(pair):
    from multiagent.model import cache_renderer
    original = cache_renderer.load_renderer_checkpoint

    def load(*args, **kwargs):
        renderer, payload = original(*args, **kwargs)
        return renderer, {**payload, 'sender_model': pair['sender'], 'receiver_model': pair['receiver']}

    cache_renderer.load_renderer_checkpoint = load


def excerpt_sample(sample, tokenizers, cap, cache):
    old = sample.metadata['context']
    key = hashlib.sha256(old.encode()).hexdigest()
    if key not in cache:
        text = old
        while (longest := max(len(t.encode(text, add_special_tokens=False)) for t in tokenizers)) > cap:
            keep = int(len(text) * min(.97, (cap - 100) / longest))
            if keep < 2:
                raise ValueError('Context cap is too small')
            text = text[:keep//2] + '\n\n[Context excerpt: middle omitted]\n\n' + text[-keep//2:]
        cache[key] = text
    text = cache[key]
    data = dataclasses.asdict(sample)
    for field in ('sender_messages', 'phase1_messages', 'messages', 'receiver_messages'):
        for message in data.get(field) or []:
            message['content'] = message['content'].replace(old, text)
    data['metadata'].update(context=text, original_context_sha256=key,
                            frozen_context_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            context_excerpted=text != old)
    return data


def load_config(path):
    config = read(path)
    if config.get('active_pairs'):
        config['pairs'] = {n: p for n, p in config['pairs'].items() if n in config['active_pairs']}
    config['senders'] = sorted({p['sender'] for p in config['pairs'].values()})
    return config


def out_dir(config):
    return ROOT / config['output']


def scratch_dir(config):
    return ROOT / config['scratch']


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def preview(text, edge=100):
    if len(text) <= 2 * edge:
        return text
    return text[:edge] + ' [...] ' + text[-edge:]


def prepare(config_path):
    config = load_config(config_path)
    run = out_dir(config)
    manifest_path = run / 'manifest.json'
    previous = read(manifest_path) if manifest_path.exists() else None
    if previous is not None:
        if previous['config'] == read(config_path):
            print('already prepared', flush=True)
            return
        changed = {k for k in set(previous['config']) | set(read(config_path))
                   if previous['config'].get(k) != read(config_path).get(k)}
        if changed - EXTENDABLE_KEYS:
            raise ValueError(f'{manifest_path}: protocol keys changed {sorted(changed - EXTENDABLE_KEYS)}; use a new output dir')
        print(f'extending prepared run (changed: {sorted(changed)})', flush=True)
    from transformers import AutoTokenizer
    from multiagent.dataset.c2c_registry import load_c2c_task
    from multiagent.model.vanilla_loader import resolve_model_path

    def freeze_sample(sample, settings, cache):
        if settings.get('head_tokens'):
            return head_truncate_sample(sample, tokenizers_by_name[settings['head_tokenizer']], settings['head_tokens'], cache)
        if settings['context_tokens'] is not None:
            return excerpt_sample(sample, tokenizers, settings['context_tokens'], cache)
        data = dataclasses.asdict(sample)
        text = data['metadata']['context']
        data['metadata'].update(original_context_sha256=sha(text), frozen_context_sha256=sha(text), context_excerpted=False)
        return data
    variants = {v: config['variants'][v] for v in used_variants(config)}
    names = sorted({p[k] for p in config['pairs'].values() for k in ('sender', 'receiver')})
    models = {name: str(resolve_model_path(name)) for name in names}
    for name, path in models.items():
        if not Path(path).is_dir():
            raise FileNotFoundError(f'{name}: {path}')
    tokenizers_by_name = {name: AutoTokenizer.from_pretrained(path) for name, path in models.items()}
    tokenizers = list(tokenizers_by_name.values())
    datasets = {}
    for task, spec in config['datasets'].items():
        dataset = load_c2c_task(spec, root=DATASET_ROOT).dataset
        groups: dict[str, list[int]] = {}
        for i in range(len(dataset)):
            m = dataset[i].metadata
            cid = str(m.get('context_id') or f"{dataset[i].dataset}/{dataset[i].split}/{sha(str(m['context']))[:16]}")
            groups.setdefault(cid, []).append(i)
        chosen = sorted(groups)
        random.Random(f"{config['seed']}:{spec}").shuffle(chosen)
        chosen = chosen[:config.get('contexts_per_dataset_overrides', {}).get(task, config['contexts_per_dataset'])]
        datasets[task] = {}
        for variant, settings in variants.items():
            cache, records = {}, []
            for cid in chosen:
                for i in groups[cid]:
                    sample = dataset[i]
                    sample.metadata['context_id'] = cid
                    records.append(freeze_sample(sample, settings, cache))
            records.sort(key=lambda r: (str(r['metadata']['context_id']), r['id']))
            target = samples_path(run, task, variant)
            if target.exists():
                old = {r['id']: r for r in read(target)['samples']}
                new = {r['id']: r for r in records}
                missing = sorted(set(old) - set(new))
                altered = sorted(i for i in old if i in new and old[i]['metadata']['frozen_context_sha256'] != new[i]['metadata']['frozen_context_sha256'])
                if missing or altered:
                    raise ValueError(f'{task}/{variant}: extension would drop {len(missing)} / alter {len(altered)} frozen samples')
            write(target, {'task': spec, 'variant': variant, **{k: v for k, v in settings.items() if k != 'summarizer_max_length'},
                           'samples': records})
            datasets[task][variant] = {
                'questions': len(records),
                'contexts': len(chosen),
                'excerpted_contexts': len({r['metadata']['context_id'] for r in records if r['metadata']['context_excerpted']}),
                'dataset_questions': len(dataset),
                'dataset_contexts': len(groups),
            }
            print(task, variant, json.dumps(datasets[task][variant]), flush=True)
    write(manifest_path, {'config': read(config_path), 'models': models, 'datasets': datasets,
                          'dataset_root': DATASET_ROOT})


def head_truncate_sample(sample, tokenizer, head_tokens, cache):
    old = sample.metadata['context']
    key = sha(old)
    if key not in cache:
        offsets = tokenizer(old, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        keep = len(offsets)
        text = old
        while len(tokenizer(text, add_special_tokens=False)['input_ids']) > head_tokens:
            keep = min(keep, head_tokens) - (0 if keep > head_tokens else 1)
            text = old[:offsets[keep - 1][1]].rstrip()
        cache[key] = text
    text = cache[key]
    data = dataclasses.asdict(sample)
    for field in ('sender_messages', 'phase1_messages', 'messages', 'receiver_messages'):
        for message in data.get(field) or []:
            message['content'] = message['content'].replace(old, text)
    data['metadata'].update(context=text, original_context_sha256=key, frozen_context_sha256=sha(text),
                            context_excerpted=text != old)
    return data


EXTENDABLE_KEYS = {'contexts_per_dataset', 'contexts_per_dataset_overrides', 'vllm_python_overrides', 'pairs', 'fit_filter', 'am', 'am_max_length', 'active_pairs',
                   'am_fit_receiver_tokens', 'am_decode_backend', 'receiver_engine', 'receiver_batch_size', 'receiver_batch_size_overrides',
                   'summarizer_batch_size', 'summarizer_batch_size_overrides', 'summarizer_backend',
                   'vllm_gpu_memory_utilization', 'notes', 'scratch'}


def samples_sha(run, task, variant):
    keys = sorted((r['id'], r['metadata']['frozen_context_sha256']) for r in read(samples_path(run, task, variant))['samples'])
    return sha(json.dumps(keys))


def summaries_cover(run, task, sender, prompts, variant, config=None):
    target = run / task / 'summaries' / f'{sender}.json'
    if not target.exists():
        return False
    payload = read(target)
    contexts = contexts_of(run, task, variant)
    required = sender_context_ids(config, run, task, sender) if config else set(contexts)
    wanted = {cid: sha(text) for cid, text in contexts.items() if cid in required}
    return all(all(cid in payload['prompts'].get(p, {}) and payload['prompts'][p][cid].get('context_sha256') == h
                   for cid, h in wanted.items()) for p in prompts)


def fit_sidecar_path(run, task, variant):
    return run / 'samples' / variant / f'{task}.fit.json'


def write_fit_sidecar(config, run, manifest, task, variant, tokenizer=None):
    from transformers import AutoTokenizer
    ff = config['fit_filter']
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(manifest['models'][ff['tokenizer']])
    fit, over = [], []
    for cid, text in contexts_of(run, task, variant).items():
        n = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        (fit if n <= ff['tokens'] else over).append(cid)
    write(fit_sidecar_path(run, task, variant), {'tokens': ff['tokens'], 'tokenizer': ff['tokenizer'],
                                                 'samples_sha': samples_sha(run, task, variant), 'fit': fit, 'over': over})
    return fit, over


def fit_ids(config, run, task, variant):
    if 'fit_filter' not in config:
        return None
    side = fit_sidecar_path(run, task, variant)
    if not side.exists() or read(side)['samples_sha'] != samples_sha(run, task, variant):
        raise FileNotFoundError(f'{side} missing or stale: run `run.py fit`')
    return set(read(side)['fit'])


def fit_filtered(config, pair_name, task):
    pair = config['pairs'][pair_name]
    return bool(pair.get('fit_filter')) and task not in pair.get('fit_filter_exempt', [])


def pair_context_ids(config, run, task, pair_name):
    variant = pair_variant(config, pair_name, task)
    ids = set(contexts_of(run, task, variant))
    if fit_filtered(config, pair_name, task):
        ids &= fit_ids(config, run, task, variant)
    return ids


def sender_context_ids(config, run, task, sender):
    ids = set()
    for name, pair in config['pairs'].items():
        if pair['sender'] == sender:
            ids |= pair_context_ids(config, run, task, name)
    return ids


def fit(config_path):
    config = load_config(config_path)
    run = out_dir(config)
    manifest = read(run / 'manifest.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest['models'][config['fit_filter']['tokenizer']])
    for task in config['datasets']:
        for variant in used_variants(config):
            f, o = write_fit_sidecar(config, run, manifest, task, variant, tokenizer)
            print(f'{task}/{variant}: fit {len(f)} over {len(o)}', flush=True)


def contexts_of(run, task, variant):
    seen = {}
    for r in read(samples_path(run, task, variant))['samples']:
        seen.setdefault(r['metadata']['context_id'], r['metadata']['context'])
    return seen


def samples_path(run, task, variant):
    return run / 'samples' / variant / f'{task}.json'


def pair_variant(config, pair_name, task=None):
    pair = config['pairs'][pair_name]
    return pair.get('variant_overrides', {}).get(task, pair.get('variant', 'full'))


def used_variants(config):
    return sorted({pair_variant(config, n, t) for n in config['pairs'] for t in config['datasets']})


def sender_variant(config, sender, task=None):
    variants = {pair_variant(config, name, task) for name, pair in config['pairs'].items() if pair['sender'] == sender}
    if len(variants) != 1:
        raise ValueError(f'sender {sender} is used with several context variants: {sorted(variants)}')
    return variants.pop()


def summarize(config_path, sender):
    config = load_config(config_path)
    run = out_dir(config)
    manifest = read(run / 'manifest.json')
    import torch
    from multiagent.evaluation import summaries as ev
    path = Path(manifest['models'][sender])
    tokenizer = ev.load_tokenizer(path)
    model = ev.load_model(path, torch.bfloat16, 'cuda:0')
    variant = sender_variant(config, sender)
    max_length = config['variants'][variant]['summarizer_max_length']
    batch_size = config.get('summarizer_batch_size_overrides', {}).get(sender, config['summarizer_batch_size'])
    contexts = {task: {c: t for c, t in contexts_of(run, task, variant).items()
                       if c in sender_context_ids(config, run, task, sender)} for task in config['datasets']}
    payloads, targets = {}, {}
    for task in config['datasets']:
        targets[task] = run / task / 'summaries' / f'{sender}.json'
        payloads[task] = read(targets[task]) if targets[task].exists() else {
            'sender': sender, 'summarizer_model': str(path), 'summarizer_max_length': max_length,
            'variant': variant, 'summary_max_new_tokens': config['summary_max_new_tokens'],
            'summarizer_batch_size': batch_size, 'seed': config['seed'],
            'sampling': {'temperature': 0.7, 'top_k': 20, 'top_p': 0.8, 'do_sample': True},
            'prompts': {}, 'complete': False}
    for prompt in config['prompts']:
        pending = []
        for task, ctxs in contexts.items():
            done = payloads[task]['prompts'].setdefault(prompt, {})
            pending += [(f'{task}|{cid}', text) for cid, text in ctxs.items() if cid not in done]
        if not pending:
            continue
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        print(f'{sender}/{prompt}: summarizing {len(pending)} contexts in batches of {batch_size}', flush=True)

        def save(key, record):
            task, cid = key.split('|', 1)
            text = contexts[task][cid]
            payloads[task]['prompts'][prompt][cid] = {
                **record, 'context_preview': preview(text), 'context_chars': len(text), 'context_sha256': sha(text)}

        ev.summarize_contexts_batched(
            model=model, tokenizer=tokenizer, receiver_tokenizer=tokenizer,
            contexts=pending, prompt_name=prompt, device='cuda:0', max_length=max_length,
            batch_size=batch_size, on_context=save, max_new_tokens=config['summary_max_new_tokens'])
        for task in config['datasets']:
            write(targets[task], payloads[task])
    for task in config['datasets']:
        payload = payloads[task]
        payload['complete'] = all(set(payload['prompts'].get(p, {})) == set(contexts[task]) for p in config['prompts'])
        payload['gpu'] = torch.cuda.get_device_name(0)
        write(targets[task], payload)
        print(f'{task}/{sender}: complete={payload["complete"]}', flush=True)


def wait_for_summaries(config, run, task, sender, prompts, variant, timeout_s=12 * 3600):
    started = time.time()
    while not summaries_cover(run, task, sender, prompts, variant, config):
        if time.time() - started > timeout_s:
            raise TimeoutError(f'summaries never covered the samples: {task}/{sender}')
        print(f'waiting for {task}/summaries/{sender}.json to cover all samples', flush=True)
        time.sleep(60)


def purge_stale_rows(config, output, prepared, task, pair_name):
    if not output.exists():
        return
    current = {r['id']: r['metadata']['frozen_context_sha256'] for r in prepared['samples']}
    payload = read(output)
    dropped, kept = [], 0
    for level, res in payload.get('results_by_prompt', {}).items():
        keep = []
        for row in res.get('rows', []):
            if current.get(row['id']) == row.get('frozen_context_sha256'):
                keep.append(row)
            else:
                dropped.append({'level': level, **{k: row.get(k) for k in ('id', 'context_id', 'frozen_context_sha256', 'grade', 'output')}})
        kept += len(keep)
        res['rows'] = keep
        res['summarized_contexts'] = {}
    if dropped:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        write(scratch_dir(config) / task / pair_name / f'purged_rows_{stamp}.json', dropped)
        write(output, payload)
        print(f'{task}/{pair_name}: purged {len(dropped)} stale rows (kept {kept}); see scratch purged_rows_{stamp}.json', flush=True)


def vllm_answers(config, run, manifest, pair_name, task, prepared, output):
    from multiagent.evaluation import summaries as ev
    from multiagent.dataset.c2c_types import C2CExample
    pair = config['pairs'][pair_name]
    tokenizer = ev.load_tokenizer(Path(manifest['models'][pair['receiver']]))
    summaries = read(run / task / 'summaries' / f"{pair['sender']}.json")['prompts']
    have = set()
    if output.exists():
        for level, res in read(output)['results_by_prompt'].items():
            have |= {(level, r['id']) for r in res['rows']}
    work = scratch_dir(config) / task / pair_name
    answers_path = work / 't2t_vllm_answers.json'
    answers = read(answers_path) if answers_path.exists() else {}
    requests = {}
    for level in config['prompts']:
        for r in prepared['samples']:
            if (level, r['id']) in have:
                continue
            sample = C2CExample(**r)
            summary = summaries[level][r['metadata']['context_id']]['text']
            messages = ev.messages_with_summarized_context(sample, summary)
            text = ev.apply_chat_template(tokenizer, messages, add_generation_prompt=True)
            key = sha(text)
            if key not in answers:
                requests[key] = text
    if requests:
        req_path = work / 't2t_vllm_requests.json'
        write(req_path, {'model': manifest['models'][pair['receiver']], 'max_length': config['max_length'],
                         'max_new_tokens': config['max_new_tokens'], 'seed': config['seed'],
                         'requests': [{'key': k, 'prompt': v} for k, v in requests.items()]})
        python = vllm_python(config, pair['receiver'])
        print(f'{task}/{pair_name}: vLLM answering {len(requests)} prompts', flush=True)
        subprocess.run([python, '-u', str(Path(__file__).with_name('t2t_vllm.py')), '--requests', str(req_path),
                        '--answers', str(answers_path)], cwd=ROOT, check=True)
        answers = read(answers_path)
    return answers


def evaluate(config_path, pair_name, task):
    config = load_config(config_path)
    run = out_dir(config)
    manifest = read(run / 'manifest.json')
    pair = config['pairs'][pair_name]
    summaries = run / task / 'summaries' / f"{pair['sender']}.json"
    output = run / task / pair_name / 't2t.json'
    done = scratch_dir(config) / task / pair_name / 't2t.done.json'
    variant = pair_variant(config, pair_name, task)
    current_sha = samples_sha(run, task, variant)
    if fit_filtered(config, pair_name, task):
        current_sha = sha(current_sha + json.dumps(sorted(pair_context_ids(config, run, task, pair_name))))
    if done.exists() and read(done).get('samples_sha') == current_sha:
        print(f'{task}/{pair_name}: done', flush=True)
        return
    if done.exists():
        print(f'{task}/{pair_name}: sample set changed since the done marker; resuming', flush=True)
    wait_for_summaries(config, run, task, pair['sender'], config['prompts'], variant)
    import torch
    from multiagent.evaluation import summaries as ev
    from multiagent.dataset.c2c_registry import C2CTask
    from multiagent.dataset.c2c_types import C2CExample
    prepared = read(samples_path(run, task, variant))
    allowed = pair_context_ids(config, run, task, pair_name)
    prepared['samples'] = [r for r in prepared['samples'] if r['metadata']['context_id'] in allowed]
    purge_stale_rows(config, output, prepared, task, pair_name)
    resume = ['--resume-from', str(output)] if output.exists() else []
    answers = vllm_answers(config, run, manifest, pair_name, task, prepared, output) if config.get('receiver_engine') == 'vllm' else None

    class Dataset:
        preserve_order = True
        def __init__(self): self.rows = [C2CExample(**r) for r in prepared['samples']]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    if not hasattr(ev, '_exp2_sweep_model_cache'):
        ev._exp2_sweep_model_cache, ev._exp2_sweep_load_model = {}, ev.load_model
        def cached_load_model(model_path, dtype, device, **kwargs):
            key = (str(model_path), str(dtype), device, str(kwargs.get("adapter")))
            if key not in ev._exp2_sweep_model_cache:
                ev._exp2_sweep_model_cache[key] = ev._exp2_sweep_load_model(model_path, dtype, device, **kwargs)
            return ev._exp2_sweep_model_cache[key]
        ev.load_model = cached_load_model
    original_load = getattr(ev, '_exp2_sweep_original_load_task', ev.load_c2c_task)
    ev._exp2_sweep_original_load_task = original_load
    def load_task(spec, **kwargs):
        if spec != prepared['task']:
            raise ValueError(f'Unexpected task {spec}')
        return C2CTask(spec=spec, dataset=Dataset(), grade=original_load(spec, **kwargs).grade)
    ev.load_c2c_task = load_task
    device = 'cpu' if answers is not None else 'cuda:0'
    batch = '1' if answers is not None else str(config.get('receiver_batch_size_overrides', {}).get(pair_name, config.get('receiver_batch_size', 1)))
    sys.argv = [str(ROOT / 'multiagent/evaluation/summaries.py'),
                '--model', manifest['models'][pair['receiver']], '--summaries-from', str(summaries),
                '--summary-prompts', *config['prompts'], '--tasks', prepared['task'],
                '--dataset-root', manifest['dataset_root'], '--max-length', str(config['max_length']),
                '--max-new-tokens', str(config['max_new_tokens']), '--device', device, '--dtype', 'bfloat16',
                '--seed', str(config['seed']), '--save-every', str(config.get('save_every', 500)), '--output', str(output),
                '--batch-size', batch, *resume]
    output.parent.mkdir(parents=True, exist_ok=True)
    if answers is not None:
        original_generate = getattr(ev, '_exp2_sweep_original_generate_vanilla', ev.generate_vanilla)
        ev._exp2_sweep_original_generate_vanilla = original_generate
        def lookup_generate(model, tokenizer, messages, device, max_length, max_new_tokens, timings=None):
            key = sha(ev.apply_chat_template(tokenizer, messages, add_generation_prompt=True))
            a = answers[key]
            return a['text'], {'stop': 'eos', 'length': 'cutoff'}.get(a['finish_reason'], a['finish_reason']), a['generated_tokens']
        ev.generate_vanilla = lookup_generate
    ev.main()
    if answers is not None:
        payload = read(output)
        payload['receiver_engine'] = {'backend': 'vllm', 'script': 'scripts/exp2_sweep/t2t_vllm.py', 'decoding': 'greedy',
                                      'note': 'answers generated by vLLM from the same chat prompts; graded and assembled by evaluation.summaries on CPU'}
        write(output, payload)
    journal = Path(str(output) + '.measurements.jsonl')
    if journal.exists():
        done.parent.mkdir(parents=True, exist_ok=True)
        journal.replace(done.parent / journal.name)
    write(done, {'pair': pair_name, 'task': task, 'sender': pair['sender'], 'receiver': pair['receiver'],
                 'samples_sha': current_sha, 'variant': variant, 'questions': len(prepared['samples']),
                 'fit_filter': bool(pair.get('fit_filter')), 'receiver_engine': config.get('receiver_engine', 'hf'),
                 'command': sys.argv, 'gpu': torch.cuda.get_device_name(0)})


def evaluate_pair(config_path, pair_name):
    for task in load_config(config_path)['datasets']:
        evaluate(config_path, pair_name, task)


def vllm_python(config, model):
    env = os.environ.get('MULTIKV_VLLM_PYTHON_' + ''.join(c if c.isalnum() else '_' for c in model).upper())
    return env or config.get('vllm_python_overrides', {}).get(model) or str(paths.VLLM_PYTHON)


def summarize_command(config, config_path, sender):
    me = str(Path(__file__).resolve())
    if config.get('summarizer_backend', 'hf') != 'vllm':
        return shlex.join([str(PYTHON), '-u', me, 'summarize', '--config', str(config_path), '--sender', sender])
    import importlib.util
    spec = importlib.util.spec_from_file_location('summarize_vllm', Path(me).with_name('summarize_vllm.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from multiagent.evaluation.summaries import SUMMARY_PROMPTS
    if module.SUMMARY_PROMPTS != SUMMARY_PROMPTS:
        raise ValueError('summarize_vllm.SUMMARY_PROMPTS differs from evaluation.summaries.SUMMARY_PROMPTS')
    util = config.get('vllm_gpu_memory_utilization', {}).get(sender, 0.92)
    python = vllm_python(config, sender)
    return shlex.join([python, '-u', str(Path(me).with_name('summarize_vllm.py')), '--config', str(config_path),
                       '--sender', sender, '--gpu-memory-utilization', str(util)])


def queue(config_path, gpus, only_pairs=None, with_am=False):
    config = load_config(config_path)
    run = out_dir(config)
    me = str(Path(__file__).resolve())
    if only_pairs:
        config['pairs'] = {n: p for n, p in config['pairs'].items() if n in only_pairs}
        config['senders'] = sorted({p['sender'] for p in config['pairs'].values()})
    def summaries_complete(sender):
        return all(summaries_cover(run, task, sender, config['prompts'], sender_variant(config, sender, task), config)
                   for task in config['datasets'])
    ready = [s for s in config['senders'] if summaries_complete(s)]
    commands = []
    for phase in ('ready', 'summarize', 'rest'):
        for sender in config['senders']:
            if phase == 'summarize' and sender not in ready:
                commands.append(summarize_command(config, config_path, sender))
            if phase in ('ready', 'rest') and ((sender in ready) == (phase == 'ready')):
                for pair_name, pair in config['pairs'].items():
                    if pair['sender'] == sender:
                        commands.append(shlex.join([str(PYTHON), '-u', me, 'eval-pair', '--config', str(config_path),
                                                    '--pair', pair_name]))
    if with_am:
        for pair_name in config['pairs']:
            if pair_name in config.get('am', {}):
                commands.append(shlex.join([str(PYTHON), '-u', me, 'am', '--config', str(config_path), '--pair', pair_name]))
    tag = Path(config['output']).name
    queue_file = ROOT / f'multiagent/gpu_queue_commands/gpu_queue_commands_{tag}.json'
    write(queue_file, commands)
    write(scratch_dir(config) / 'queue.json', {'commands': commands, 'gpus': gpus})
    launch = (f'cd {ROOT / "multiagent"} && setsid nohup ../env3.10/bin/python gpu_queue.py '
              f'--commands {queue_file.relative_to(ROOT / "multiagent")} --gpus {gpus} --name {tag} '
              f'> gpu_queue_logs/{tag}.nohup.log 2>&1 &')
    print(f'{len(commands)} commands -> {queue_file}\n{launch}')
    return launch


def report(config_path):
    config = load_config(config_path)
    run = out_dir(config)
    lines = [f'# exp2_sweep report ({time.strftime("%Y-%m-%d %H:%M")})', '']
    manifest = read(run / 'manifest.json')
    lines += ['| dataset | variant | contexts | questions | excerpted |', '|---|---|---|---|---|']
    for task, variants in manifest['datasets'].items():
        for variant, d in variants.items():
            lines.append(f"| {task} | {variant} | {d['contexts']} / {d['dataset_contexts']} | {d['questions']} / {d['dataset_questions']} | {d['excerpted_contexts']} |")
    for task in config['datasets']:
        lines += ['', f'## {task}', '']
        lines += ['| pair | sender -> receiver | variant | ' + ' | '.join(p.replace('summarize_', '') or 'summarize' for p in config['prompts']) + ' |',
                  '|---|---|---|' + '---|' * len(config['prompts'])]
        for pair_name, pair in config['pairs'].items():
            path = run / task / pair_name / 't2t.json'
            cells = []
            for prompt in config['prompts']:
                r = read(path)['results_by_prompt'].get(prompt) if path.exists() else None
                if not r or not r['rows']:
                    cells.append('-')
                    continue
                s = r['summary']['all']
                score = s['metrics'].get('f1', s['metrics'].get('answer_f1')) if s['metrics'] else None
                score = score if score is not None else s['accuracy']
                tok = sum(x['summarization']['summary_tokens_receiver'] for x in r['rows']) / len(r['rows'])
                cells.append(f"{100 * score:.1f} ({tok:.0f} tok, n={s['seen']})")
            lines.append(f"| {pair_name} | {pair['sender']} -> {pair['receiver']} | {pair_variant(config, pair_name)} | " + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    scratch_dir(config).mkdir(parents=True, exist_ok=True)
    (scratch_dir(config) / 'report.md').write_text(text)
    print(text)


def am_budgets(run, task, pair_name, prompts):
    results = read(run / task / pair_name / 't2t.json')['results_by_prompt']
    out = {}
    for prompt in prompts:
        recs = {r['summarization']['context_id']: r['summarization'] for r in results[prompt]['rows']}
        retention = (sum(v['summary_tokens_receiver'] for v in recs.values())
                     / sum(v['original_tokens_receiver'] for v in recs.values()))
        compression = math.ceil(1 / retention)
        out[prompt] = {'summary_retention': retention, 'compression': compression, 'am_ratio': 1 / compression}
    return out


def am_context_seed(base_seed, task_key, index, sample):
    return int(hashlib.sha256(sample.metadata['context_id'].encode()).hexdigest()[:8], 16)


def am(config_path, pair_name, datasets, contexts, out, decode_backend=None, arm='transfer', query_transfer=None,
       only_t2t_contexts=True, levels=None):
    config = load_config(config_path)
    if decode_backend:
        config['am_decode_backend'] = decode_backend
    run = out_dir(config)
    manifest = read(run / 'manifest.json')
    pair = config['pairs'][pair_name]
    variant = pair_variant(config, pair_name)
    am_cfg = dict(config['am'][pair_name])
    if query_transfer:
        am_cfg['query_transfer'] = query_transfer
    final_root = Path(out) if out else run
    if out and 'outputs_mine' not in Path(out).resolve().parts:
        work_root = Path(out)
    else:
        work_root = (ROOT / 'multiagent/outputs' / Path(out).name if out else scratch_dir(config))
    datasets = datasets or list(config['datasets'])
    if levels:
        unknown = set(levels) - set(config['prompts'])
        if unknown:
            raise ValueError(f'unknown levels {sorted(unknown)}')
        config['prompts'] = [p for p in config['prompts'] if p in levels]
    import torch
    from transformers import AutoTokenizer
    from multiagent.evaluation import native_am, renderer_am, beyond_context
    if arm == 'native' and decode_backend is not None:
        raise ValueError('--am-decode-backend applies only to rendered AM')
    from multiagent.dataset.c2c_registry import C2CTask
    from multiagent.dataset.c2c_types import C2CExample
    if arm != 'native':
        configure_renderer_models(pair)
    receiver_tok = AutoTokenizer.from_pretrained(manifest['models'][pair['receiver']])
    for task in datasets:
        variant = pair_variant(config, pair_name, task)
        t2t_done = scratch_dir(config) / task / pair_name / 't2t.done.json'
        allowed_ids = pair_context_ids(config, run, task, pair_name)
        expected_questions = len([r for r in read(samples_path(run, task, variant))['samples']
                                  if r['metadata']['context_id'] in allowed_ids])
        if expected_questions == 0:
            print(f'{task}/{pair_name}: no fitting contexts, AM skipped', flush=True)
            continue
        waited = 0
        while not only_t2t_contexts and not (t2t_done.exists() and read(t2t_done).get('questions', -1) == expected_questions):
            if waited == 0:
                print(f'{task}/{pair_name}: waiting for the T2T cell to finish (budgets come from it)', flush=True)
            time.sleep(60); waited += 1
            if waited > 12 * 60:
                raise TimeoutError(f'{task}/{pair_name}: T2T never finished')
        prepared = read(samples_path(run, task, variant))
        fits, skipped, order = {}, [], []
        allowed = pair_context_ids(config, run, task, pair_name) if pair.get('fit_filter') else None
        overrides = pair.get('am_overrides', {}).get(task, {})
        long_context = overrides.get('beyond_context')
        if long_context and (long_context != 'hybrid' or arm == 'oracle'):
            raise ValueError('long-context sweeps require hybrid transfer or local native AM; oracle is full-context only')
        ev = beyond_context if long_context else native_am if arm == 'native' else renderer_am
        ev.deterministic_am_context_seed = am_context_seed
        if not hasattr(ev, '_exp2_sweep_model_cache'):
            ev._exp2_sweep_model_cache, ev._exp2_sweep_load_model = {}, ev.load_model
            def cached_load_model(model_path, dtype, device, _entry=ev, **kwargs):
                key = (str(model_path), str(dtype), device, str(kwargs.get("adapter")))
                if key not in _entry._exp2_sweep_model_cache:
                    _entry._exp2_sweep_model_cache[key] = _entry._exp2_sweep_load_model(model_path, dtype, device, **kwargs)
                return _entry._exp2_sweep_model_cache[key]
            ev.load_model = cached_load_model
        original_load = getattr(ev, '_exp2_sweep_original_load_task', ev.load_c2c_task)
        ev._exp2_sweep_original_load_task = original_load
        am_max_length = overrides.get('am_max_length', config['am_max_length'])
        route_prefix = (['local' if arm == 'native' else 'hybrid'] if long_context else [])
        route_args = (['--bc-chunk-tokens', str(overrides['bc_chunk_tokens'])]
                      if long_context else ['--am-chunk-size', '4096'])
        for r in prepared['samples']:
            cid = r['metadata']['context_id']
            if cid not in fits:
                if allowed is not None:
                    fits[cid] = cid in allowed
                else:
                    n = len(receiver_tok(r['metadata']['context'], add_special_tokens=False)['input_ids'])
                    fits[cid] = n <= config['am_fit_receiver_tokens']
                (order if fits[cid] else skipped).append(cid)
        if only_t2t_contexts:
            covered = {r['context_id'] for r in read(run / task / pair_name / 't2t.json')['results_by_prompt'][config['prompts'][0]]['rows']}
            order = [c for c in order if c in covered]
        chosen = set(order[:contexts] if contexts else order)
        samples = [r for r in prepared['samples'] if r['metadata']['context_id'] in chosen]
        budgets = am_budgets(run, task, pair_name, config['prompts'])
        suffix = '' if arm == 'transfer' else f'_{arm}'
        work = work_root / task / pair_name / f'am_levels{suffix}'
        final = final_root / task / pair_name / f'am{suffix}.json'
        print(f'{task}/{pair_name}: {len(chosen)} contexts, {len(samples)} questions, skipped over fit: {len(skipped)}', flush=True)

        class Dataset:
            preserve_order = True
            def __init__(self): self.rows = [C2CExample(**r) for r in samples]
            def __len__(self): return len(self.rows)
            def __getitem__(self, i): return self.rows[i]
        def load_task(spec, **kwargs):
            if spec != prepared['task']:
                raise ValueError(f'Unexpected task {spec}')
            return C2CTask(spec=spec, dataset=Dataset(), grade=original_load(spec, **kwargs).grade)
        ev.load_c2c_task = load_task
        computed = {}
        for prompt in config['prompts']:
            compression = budgets[prompt]['compression']
            if compression in computed:
                print(f'{task}/{pair_name}/{prompt}: x{compression} already computed for {computed[compression]}', flush=True)
                continue
            computed[compression] = prompt
            output = work / f'{prompt}.json'
            done = output.with_suffix('.done.json')
            if done.exists() and read(done).get('compression') == compression and read(done).get('questions') == len(samples):
                print(f'{task}/{pair_name}/{prompt}: done', flush=True)
                continue
            if output.exists():
                output.unlink()
            ratio = budgets[prompt]['am_ratio']
            if arm == 'transfer':
                model_args = ['--renderer', str(ROOT / am_cfg['renderer']), '--am-query-transfer', str(ROOT / am_cfg['query_transfer'])]
            elif arm == 'oracle':
                model_args = ['--renderer', str(ROOT / am_cfg['renderer']), '--am-oracle-queries']
            elif arm == 'native':
                model_args = ['--model', manifest['models'][pair['receiver']]]
            else:
                raise ValueError(f'unknown arm {arm}')
            budget_args = (['--am-head-budget-mode', 'uniform'] if am_cfg['budget'] == 'uniform' else
                           ['--am-head-budget-mode', 'precomputed', '--am-head-budget-path', str(ROOT / am_cfg['budget'])])
            renderer_options = ([] if arm == 'native' else [
                '--am-query-storage', 'device', '--am-attention-backend', 'eager',
                '--am-decode-backend', config.get('am_decode_backend', 'torchscript_prefill')])
            sys.argv = [str(Path(ev.__file__).resolve()), *route_prefix, *model_args,
                        '--am-target-ratio', str(ratio), *budget_args, '--am-query-samples', '10000',
                        '--am-nnls-iters', '2', '--am-score-method', 'rms', *route_args,
                        '--context-cache-max', '1', '--am-compaction-workers', '4', *renderer_options,
                        '--tasks', prepared['task'], '--dataset-root', manifest['dataset_root'],
                        '--max-length', str(am_max_length), '--max-new-tokens', str(config['max_new_tokens']),
                        '--device', 'cuda:0', '--seed', str(config['seed']), '--save-every', '1', '--output', str(output)]
            print(f'{task}/{pair_name}/{prompt}: AM ratio {ratio:.4f} (x{compression})', flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            ev.main()
            write(done, {'pair': pair_name, 'task': task, 'prompt': prompt, **budgets[prompt],
                         'contexts': len(chosen), 'questions': len(samples), 'command': sys.argv,
                         'gpu': torch.cuda.get_device_name(0)})
        merged = {'pair': pair_name, 'arm': arm, 'sender': pair['sender'], 'receiver': pair['receiver'], 'variant': variant,
                  'task': prepared['task'], 'am': {k: str(v) for k, v in am_cfg.items()},
                  'am_max_length': am_max_length, 'am_overrides': overrides, 'am_fit_receiver_tokens': config['am_fit_receiver_tokens'],
                  'contexts': sorted(chosen), 'questions': len(samples), 'skipped_contexts_over_fit': skipped,
                  'budgets': budgets, 'results_by_level': {}}
        for prompt in config['prompts']:
            holder = computed[budgets[prompt]['compression']]
            if holder != prompt:
                merged['results_by_level'][prompt] = {**budgets[prompt], 'alias_of': holder}
                continue
            payload = read(work / f'{prompt}.json')
            rows = [r for r in payload['rows'] if not r['id'].startswith('warmup/')]
            merged['results_by_level'][prompt] = {**budgets[prompt], 'summary': payload['summary'],
                                                  'evaluation_config': payload.get('evaluation_config'),
                                                  'compression_stats': payload.get('compression'), 'rows': rows}
        write(final, merged)
        print(f'{task}/{pair_name}: wrote {final}', flush=True)


def oracle_report(config_path, out, contexts):
    import statistics as st
    config = load_config(config_path)
    run = out_dir(config); out = Path(out)
    scratch = ROOT / 'multiagent/outputs' / out.name
    def score(rows):
        gr = rows[0].get('grade_result') if rows else None
        for k in ('f1', 'answer_f1'):
            if isinstance(gr, dict) and k in gr:
                return k, 100 * st.mean(r['grade_result'][k] for r in rows)
        return 'acc', (100 * st.mean(r['grade'] == 'correct' for r in rows) if rows else float('nan'))
    lines = [f'# exp2_sweep_oracle: AM query/cache source ablation on the first {contexts} fitting contexts per dataset', '',
             'Arms: T2T = receiver reads the sender summary; transfer = rendered cache + transferred sender queries (ours); '
             'oracle = rendered cache + receiver\'s own on-policy queries (receiver reads the text); native = receiver\'s own cache + own queries. '
             'Same contexts, questions and per-level budgets (summary retention rounded up).', '']
    manifest = {'contexts_per_dataset': contexts, 'source_run': str(run.relative_to(ROOT)), 'pairs': {}, 'datasets': {}}
    for pair_name in config['pairs']:
        for task in config['datasets']:
            ours = run / task / pair_name / 'am.json'
            if not ours.exists():
                continue
            arms = {'transfer': read(ours)}
            for arm in ('oracle', 'native'):
                f = out / task / pair_name / f'am_{arm}.json'
                if f.exists():
                    arms[arm] = read(f)
            t2t = read(run / task / pair_name / 't2t.json')['results_by_prompt']
            sample_ctx = set(arms.get('oracle', arms.get('native', arms['transfer']))['contexts'])
            if 'oracle' in arms or 'native' in arms:
                sample_ctx = set(next(a for k, a in arms.items() if k != 'transfer')['contexts'])
            compare = {'pair': pair_name, 'task': task, 'contexts': sorted(sample_ctx), 'levels': {}}
            lines += [f'## {task} / {pair_name} ({len(sample_ctx)} contexts)', '',
                      '| level | budget | n | T2T | transfer | oracle | native |', '|---|---|---|---|---|---|---|']
            for level in config['prompts']:
                row = {}
                for arm, d in arms.items():
                    r = d['results_by_level'][level]; rows = d['results_by_level'][r.get('alias_of', level)]['rows']
                    rows = [x for x in rows if x['context_id'] in sample_ctx]
                    m, v = score(rows); row[arm] = v; row['n'] = len(rows); row['metric'] = m; row['compression'] = r['compression']
                ids = {x['id'] for x in arms['transfer']['results_by_level'][arms['transfer']['results_by_level'][level].get('alias_of', level)]['rows'] if x['context_id'] in sample_ctx}
                m, v = score([x for x in t2t[level]['rows'] if x['id'] in ids]); row['t2t'] = v
                compare['levels'][level] = row
                fmt = lambda k: f"{row[k]:.1f}" if k in row else '-'
                lines.append(f"| {level.replace('summarize_', '') or 'summarize'} | x{row['compression']} | {row['n']} | {fmt('t2t')} | {fmt('transfer')} | {fmt('oracle')} | {fmt('native')} |")
            lines.append('')
            write(out / task / pair_name / 'compare.json', compare)
            manifest['datasets'].setdefault(task, {})[pair_name] = {'contexts': len(sample_ctx), 'arms': sorted(arms)}
    write(out / 'manifest.json', manifest)
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def ckpt_report(config_path, out, pair_name):
    import random, statistics as st
    random.seed(0)
    config = load_config(config_path)
    run = out_dir(config); out = Path(out)
    ckpts = sorted([d for d in out.iterdir() if d.is_dir() and d.name.startswith('ckpt-')], key=lambda d: int(d.name.split('-')[1]))
    def val(r):
        g = r.get('grade_result')
        for k in ('f1', 'answer_f1'):
            if isinstance(g, dict) and k in g:
                return float(g[k])
        return 1.0 if r['grade'] == 'correct' else 0.0
    def ci(d, B=2000):
        if not d:
            return float('nan'), float('nan'), float('nan')
        bs = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(B))
        return 100 * st.mean(d), 100 * bs[int(B * .025)], 100 * bs[int(B * .975) - 1]
    rows, t2t = {}, {}
    for ck in ckpts:
        rows[ck.name] = {}
        for task in config['datasets']:
            f = ck / task / pair_name / 'am.json'
            if not f.exists():
                continue
            d = read(f); rows[ck.name][task] = {}
            for level in config['prompts']:
                r = d['results_by_level'].get(level)
                if r is None:
                    continue
                rows[ck.name][task][level] = {x['id']: val(x) for x in d['results_by_level'][r.get('alias_of', level)]['rows']}
            if task not in t2t:
                t2t[task] = {level: {x['id']: val(x) for x in read(run / task / pair_name / 't2t.json')['results_by_prompt'][level]['rows']}
                             for level in config['prompts']}
    complete = [c for c in rows if rows[c] and all(t in rows[c] for t in t2t)]
    if complete:
        levels = [p for p in config['prompts'] if all(p in rows[c][t] and rows[c][t][p] for c in complete for t in rows[c])]
        config['prompts'] = levels
    last = complete[-1] if complete else None
    lines = [f'# {pair_name} query-transfer checkpoint sweep ({out.name})', '',
             f'Checkpoints with all datasets done: {", ".join(complete) or "none"}. Paired per question; 95% bootstrap CI; points of acc/F1.', '']
    breakdown = {'pair': pair_name, 'checkpoints': {}, 'reference_checkpoint': last}
    lines += ['## AM - T2T per checkpoint (pooled over all datasets and levels)', '', '| checkpoint | n | AM - T2T | vs ' + (last or '-') + ' |', '|---|---|---|---|']
    for c in complete:
        diffs, vs_last = [], []
        for task in rows[c]:
            for level in config['prompts']:
                a = rows[c][task][level]; t = t2t[task][level]
                diffs += [a[i] - t[i] for i in a]
                if last and last != c:
                    b = rows[last][task][level]; vs_last += [a[i] - b[i] for i in a]
        m, lo, hi = ci(diffs); cell = f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]'
        if vs_last:
            m2, lo2, hi2 = ci(vs_last); cell2 = f'{m2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]'
        else:
            cell2 = 'reference'
        breakdown['checkpoints'][c] = {'n': len(diffs), 'am_minus_t2t': [m, lo, hi], 'vs_reference': None if not vs_last else [m2, lo2, hi2]}
        lines.append(f'| {c} | {len(diffs)} | {cell} | {cell2} |')
    lines += ['', '## Mean score per dataset (average over the 5 levels)', '', '| dataset | n/level | T2T | ' + ' | '.join(complete) + ' |', '|---|---|---|' + '---|' * len(complete)]
    for task in config['datasets']:
        if task not in t2t or not all(task in rows[c] for c in complete):
            continue
        ids = set(rows[complete[0]][task][config['prompts'][0]])
        tm = st.mean(st.mean(t2t[task][l][i] for i in ids) for l in config['prompts']) * 100
        cells = [f"{st.mean(st.mean(rows[c][task][l][i] for i in ids) for l in config['prompts']) * 100:.1f}" for c in complete]
        lines.append(f'| {task} | {len(ids)} | {tm:.1f} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Mean score per level (all datasets pooled, per-row average)', '', '| level | T2T | ' + ' | '.join(complete) + ' |', '|---|---|' + '---|' * len(complete)]
    for level in config['prompts']:
        tv = [v for task in t2t if all(task in rows[c] for c in complete) for i, v in t2t[task][level].items() if i in rows[complete[0]][task][level]]
        cells = [f"{100 * st.mean(v for task in rows[c] for v in rows[c][task][level].values()):.1f}" for c in complete]
        lines.append(f"| {level.replace('summarize_', '') or 'summarize'} | {100 * st.mean(tv):.1f} | " + ' | '.join(cells) + ' |')
    write(out / 'breakdown.json', breakdown)
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


AM_RECAPS = {'am.json': 'recap_am.json', 'am_oracle.json': 'recap_am_oracle.json'}


def recap(config_path, out=None):
    config = read(config_path)
    config['senders'] = sorted({p['sender'] for p in config['pairs'].values()})
    if out:
        for task in config['datasets']:
            for arm_file, name in AM_RECAPS.items():
                recap_am(config, (ROOT / out).resolve(), task, arm_file, name)
        return
    run = out_dir(config)
    manifest = read(run / 'manifest.json')
    for task in config['datasets']:
        variants = sorted({pair_variant(config, n) for n in config['pairs']})
        lines = ['{', f'"dataset": {json.dumps(task)},', f'"levels": {json.dumps(config["prompts"])},',
                 '"row": "[correct, f1 or null, receiver summary tokens, answer]",',
                 '"summary_row": "[context tokens under the sender, generated tokens, stop reason]",',
                 '"samples_sha": ' + json.dumps({v: samples_sha(run, task, v) for v in variants}) + ',',
                 '"summaries": {']
        senders = config['senders']
        for si, sender in enumerate(senders):
            path = run / task / 'summaries' / f'{sender}.json'
            payload = read(path) if path.exists() else {'prompts': {}}
            lines.append(f'{json.dumps(sender)}: {{"variant": {json.dumps(sender_variant(config, sender, task))}, "levels": {{')
            for li, level in enumerate(config['prompts']):
                records = payload['prompts'].get(level, {})
                lines.append(f'{json.dumps(level)}: {{')
                for ci, cid in enumerate(sorted(records)):
                    r = records[cid]
                    cell = [r['original_tokens_summarizer'], r['generated_tokens_including_special'], r.get('finish_reason')]
                    lines.append(f'{json.dumps(cid)}: {json.dumps(cell)}' + (',' if ci < len(records) - 1 else ''))
                lines.append('}' + (',' if li < len(config['prompts']) - 1 else ''))
            lines.append('}}' + (',' if si < len(senders) - 1 else ''))
        lines += ['},', '"pairs": {']
        pair_names = list(config['pairs'])
        for pi, pair_name in enumerate(pair_names):
            pair = config['pairs'][pair_name]
            path = run / task / pair_name / 't2t.json'
            lines.append(f'{json.dumps(pair_name)}: {{"sender": {json.dumps(pair["sender"])}, '
                         f'"receiver": {json.dumps(pair["receiver"])}, "variant": {json.dumps(pair_variant(config, pair_name, task))}, "levels": {{')
            results = read(path)['results_by_prompt'] if path.exists() else {}
            for li, level in enumerate(config['prompts']):
                rows = sorted(results.get(level, {}).get('rows', []), key=lambda r: r['id'])
                lines.append(f'{json.dumps(level)}: {{')
                for ri, r in enumerate(rows):
                    f1 = r['grade_result'].get('f1') if isinstance(r.get('grade_result'), dict) else None
                    cell = [1 if r['grade'] == 'correct' else 0, None if f1 is None else round(float(f1), 4),
                            r['summarization']['summary_tokens_receiver'], r['output']]
                    lines.append(f'{json.dumps(r["id"])}: {json.dumps(cell, ensure_ascii=False)}' + (',' if ri < len(rows) - 1 else ''))
                lines.append('}' + (',' if li < len(config['prompts']) - 1 else ''))
            lines.append('}}' + (',' if pi < len(pair_names) - 1 else ''))
        lines += ['}', '}']
        target = run / task / 'recap.json'
        target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        json.loads(target.read_text(encoding='utf-8'))
        print(f'{target.relative_to(ROOT)}: {target.stat().st_size / 1e6:.2f} MB', flush=True)
        for arm_file, name in AM_RECAPS.items():
            recap_am(config, run, task, arm_file, name)


def recap_am(config, run, task, arm_file='am.json', name='recap_am.json'):
    pairs = [n for n in config['pairs'] if (run / task / n / arm_file).exists()]
    lines = ['{', f'"dataset": {json.dumps(task)},', f'"source": {json.dumps(arm_file)},', f'"levels": {json.dumps(config["prompts"])},',
             '"row": "[correct, f1 or null, article fraction kept by AM, answer]",',
             '"budget_rule": "AM ratio = 1 / ceil(1 / summary retention of the same pair, dataset and level)",',
             '"pairs": {']
    for pi, pair_name in enumerate(pairs):
        d = read(run / task / pair_name / arm_file)
        head = {k: d[k] for k in ('sender', 'receiver', 'variant', 'questions', 'contexts', 'skipped_contexts_over_fit', 'budgets')}
        lines.append(f'{json.dumps(pair_name)}: {{' + json.dumps(head, sort_keys=True)[1:-1] + ', "levels": {')
        for li, level in enumerate(config['prompts']):
            r = d['results_by_level'][level]
            if 'alias_of' in r:
                lines.append(f'{json.dumps(level)}: {{"alias_of": {json.dumps(r["alias_of"])}}}' + (',' if li < len(config['prompts']) - 1 else ''))
                continue
            rows = sorted(r['rows'], key=lambda x: x['id'])
            lines.append(f'{json.dumps(level)}: {{')
            for ri, row in enumerate(rows):
                f1 = row['grade_result'].get('f1') if isinstance(row.get('grade_result'), dict) else None
                kept = row.get('compression', {}).get('article_retained_fraction')
                kept = None if kept is None else round(float(kept), 4)
                cell = [1 if row['grade'] == 'correct' else 0, None if f1 is None else round(float(f1), 4),
                        kept, row['output']]
                lines.append(f'{json.dumps(row["id"])}: {json.dumps(cell, ensure_ascii=False)}' + (',' if ri < len(rows) - 1 else ''))
            lines.append('}' + (',' if li < len(config['prompts']) - 1 else ''))
        lines.append('}}' + (',' if pi < len(pairs) - 1 else ''))
    lines += ['}', '}']
    target = run / task / name
    if not pairs:
        if target.exists():
            target.unlink()
        return
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    json.loads(target.read_text(encoding='utf-8'))
    print(f'{target.relative_to(ROOT)}: {target.stat().st_size / 1e6:.2f} MB', flush=True)


def publish(config_path, dest, suffix):
    import shutil
    config = load_config(config_path)
    run, dest = out_dir(config), ROOT / dest
    copies = []
    for task in config['datasets']:
        for pair_name, pair in config['pairs'].items():
            for arm in ('t2t', 'am'):
                copies.append((run / task / pair_name / f'{arm}.json', dest / task / pair_name / f'{arm}_{suffix}.json'))
            variant = pair_variant(config, pair_name, task)
            copies.append((samples_path(run, task, variant), samples_path(dest, task, variant)))
        for sender in config['senders']:
            copies.append((run / task / 'summaries' / f'{sender}.json', dest / task / f'summaries_{suffix}' / f'{sender}.json'))
    missing = [str(a) for a, _ in copies if not a.exists()]
    if missing:
        raise FileNotFoundError(f'side run incomplete: {missing}')
    for source, target in dict.fromkeys(copies):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f'{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'summarize', 'eval', 'eval-pair', 'queue', 'report', 'recap', 'am', 'fit', 'oracle-report', 'ckpt-report', 'publish'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--sender')
    parser.add_argument('--pair')
    parser.add_argument('--task')
    parser.add_argument('--gpus', default='1,2')
    parser.add_argument('--datasets', nargs='*', help='am: datasets to run (default all)')
    parser.add_argument('--contexts', type=int, default=0, help='am: first N fitting contexts per dataset (0 = all)')
    parser.add_argument('--out', help='am: output root (default: the run directory); recap: only regenerate the AM recaps of this dir')
    parser.add_argument('--am-decode-backend', help='am: override the decode backend (eager, torchscript_prefill, ...)')
    parser.add_argument('--arm', default='transfer', choices=['transfer', 'oracle', 'native'], help='am: query/cache source')
    parser.add_argument('--query-transfer', help='am: override the query-transfer checkpoint (checkpoint sweeps)')
    parser.add_argument('--levels', nargs='*', help='am: restrict to these summary levels (cheap sweeps)')
    parser.add_argument('--pairs', nargs='*', help='queue: restrict to these pairs (and their senders)')
    parser.add_argument('--with-am', action='store_true', help='queue: append an AM job per pair that has an am config')
    parser.add_argument('--dest', default='multiagent/outputs_mine/exp2_sweep', help='publish: main sweep dir to copy into')
    parser.add_argument('--suffix', default='truncated', help='publish: file-name suffix for the copied finals')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.config)
    elif args.command == 'summarize':
        summarize(args.config, args.sender)
    elif args.command == 'eval':
        evaluate(args.config, args.pair, args.task)
    elif args.command == 'eval-pair':
        evaluate_pair(args.config, args.pair)
    elif args.command == 'queue':
        queue(args.config, args.gpus, args.pairs, args.with_am)
    elif args.command == 'report':
        report(args.config)
    elif args.command == 'recap':
        recap(args.config, args.out)
    elif args.command == 'fit':
        fit(args.config)
    elif args.command == 'ckpt-report':
        ckpt_report(args.config, args.out, args.pair)
    elif args.command == 'oracle-report':
        oracle_report(args.config, args.out, args.contexts)
    elif args.command == 'publish':
        publish(args.config, args.dest, args.suffix)
    elif args.command == 'am':
        am(args.config, args.pair, args.datasets, args.contexts, args.out, args.am_decode_backend, args.arm, args.query_transfer,
           levels=args.levels)


if __name__ == '__main__':
    main()
