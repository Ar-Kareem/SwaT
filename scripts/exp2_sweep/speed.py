from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multiagent.paths import PYTHON
CONFIG = Path(__file__).with_name('config.json')
CONFIG_TRUNCATED = Path(__file__).with_name('config_truncated.json')
TRUNCATED = {('14E', 'longhealth'), ('15A', 'longhealth'), ('15H', 'longhealth')}
PAIRS = ('14E', '15A', '15H')
CONTEXTS = 20
BATCH = 16
MAX_BATCHES = 8
FOLD_DIM = 136
INACTIVE = -1.0e4


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load('exp2_sweep_run', Path(__file__).with_name('run.py'))
read, write = sweep.read, sweep.write


def dirs(config):
    return ROOT / config.get('timings_output', 'multiagent/outputs_mine/exp2_sweep_timings'), sweep.scratch_dir(config) / 'timings'


def speed_pairs(config):
    if 'speed_pairs' not in config:
        return PAIRS, TRUNCATED
    return tuple(config['speed_pairs']), {tuple(c) for c in config.get('speed_truncated', [])}


def interleave(ids, context_of, max_batches=MAX_BATCHES):
    groups = {}
    for i in ids:
        groups.setdefault(context_of(i), []).append(i)
    order = [i for rank in range(max(map(len, groups.values()))) for g in groups.values() if rank < len(g) for i in [g[rank]]]
    return order[:BATCH * min(max_batches, len(order) // BATCH)]


def flash_varlen(q, k, v, cu_q, cu_k, max_q, max_k, lengths_q, seqused_k, causal, scale):
    import torch
    H = q.shape[1]
    if q.device.type == "cuda":
        out, lse, *_ = torch.ops.aten._flash_attention_forward(
            q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal, False, scale=scale, seqused_k=seqused_k)
        if lse.dim() == 2:
            lse = lse.transpose(0, 1)
        else:
            lse = torch.cat([lse[s, :, :n].transpose(0, 1) for s, n in enumerate(lengths_q)], 0)
        return out, lse
    groups = H // k.shape[1]
    outs, lses = [], []
    cu_q_l, cu_k_l = cu_q.tolist(), cu_k.tolist()
    used = seqused_k.tolist() if seqused_k is not None else [cu_k_l[s + 1] - cu_k_l[s] for s in range(len(cu_k_l) - 1)]
    for s in range(len(cu_q_l) - 1):
        qs = q[cu_q_l[s]:cu_q_l[s + 1]].float().transpose(0, 1)
        ks = k[cu_k_l[s]:cu_k_l[s] + used[s]].float().repeat_interleave(groups, 1).transpose(0, 1)
        vs = v[cu_k_l[s]:cu_k_l[s] + used[s]].float().repeat_interleave(groups, 1).transpose(0, 1)
        scores = qs @ ks.transpose(-1, -2) * scale
        if causal:
            L, S = scores.shape[-2:]
            mask = torch.arange(S).view(1, -1) > torch.arange(L).view(-1, 1)
            scores = scores.masked_fill(mask.view(1, L, S), float("-inf"))
        lse = torch.logsumexp(scores, -1)
        outs.append(((scores - lse.unsqueeze(-1)).exp() @ vs).transpose(0, 1).to(q.dtype))
        lses.append(lse.transpose(0, 1))
    return torch.cat(outs, 0), torch.cat(lses, 0)


def am_varlen_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    import torch
    p = module._am
    _, heads, total, dim = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    p["out_k"][:, :, :total].copy_(key[0].unsqueeze(0))
    p["out_v"][:, :, :total].copy_(value[0].unsqueeze(0))
    pieces = []
    start = 0
    for n in p["lengths"]:
        block = query[0, :, start:start + n]
        pieces.append(block.view(kv_heads, groups, n, dim).permute(0, 2, 1, 3).reshape(kv_heads * n, groups, dim))
        start += n
    packed = torch.cat(pieces, 0)
    qf = packed.new_zeros(packed.shape[0], groups, FOLD_DIM)
    qf[..., :dim] = packed
    qf[..., dim] = 1.0
    out1, lse1 = flash_varlen(qf, p["kf"], p["vf"], p["cu_q1"], p["cu_k"], p["max_q"], p["width"],
                              p["lengths1"], p["seqused"], False, scaling)
    outs, lses = [], []
    start = 0
    for n in p["lengths"]:
        o = out1[start:start + kv_heads * n, :, :dim].view(kv_heads, n, groups, dim).permute(1, 0, 2, 3).reshape(n, heads, dim)
        l = lse1[start:start + kv_heads * n].view(kv_heads, n, groups).permute(1, 0, 2).reshape(n, heads)
        outs.append(o)
        lses.append(l)
        start += kv_heads * n
    out1 = torch.cat(outs, 0)
    lse1 = torch.cat(lses, 0).unsqueeze(-1)
    q2 = query[0].transpose(0, 1).contiguous()
    k2 = key[0].transpose(0, 1).contiguous()
    v2 = value[0].transpose(0, 1).contiguous()
    out2, lse2 = flash_varlen(q2, k2, v2, p["cu_q2"], p["cu_q2"], p["max_q"], p["max_q"], p["lengths"], None, True, scaling)
    lse2 = lse2.unsqueeze(-1)
    m = torch.maximum(lse1, lse2)
    w1 = (lse1 - m).exp()
    w2 = (lse2 - m).exp()
    out = (w1 * out1.float() + w2 * out2.float()) / (w1 + w2)
    return out.to(query.dtype).unsqueeze(0).contiguous(), None


def t2t_varlen_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    p = module._t2t
    q = query[0].transpose(0, 1).contiguous()
    k = key[0].transpose(0, 1).contiguous()
    v = value[0].transpose(0, 1).contiguous()
    out, _ = flash_varlen(q, k, v, p["cu"], p["cu"], p["max_len"], p["max_len"], p["lengths"], None, True, scaling)
    return out.unsqueeze(0).contiguous(), None


def fold_prefix(key, value, beta, scaling):
    import torch
    kv_heads, width, dim = key.shape
    kf = key.new_zeros(kv_heads, width, FOLD_DIM)
    kf[..., :dim] = key
    kf[..., dim] = (beta.float() / scaling).nan_to_num(neginf=INACTIVE).clamp(min=INACTIVE).to(key.dtype)
    vf = key.new_zeros(kv_heads, width, FOLD_DIM)
    vf[..., :dim] = value
    return kf, vf


def build_am_batch(model, requests, device):
    import torch
    config = model.config
    layers = config.num_hidden_layers
    kv_heads = config.num_key_value_heads
    modules = [layer.self_attn for layer in model.model.layers]
    scaling = float(modules[0].scaling)
    B = len(requests)
    packing_stats = dict(heads=0, repacked_heads=0, active_entries=0, stored_entries=0)
    lengths = [len(r["ids"]) for r in requests]
    total = sum(lengths)
    ids = torch.tensor([sum((r["ids"] for r in requests), [])], device=device)
    positions = torch.cat([torch.arange(r["position"], r["position"] + len(r["ids"])) for r in requests]).view(1, -1).to(device)
    cu_q2 = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32)
    lengths1 = [n for n in lengths for _ in range(kv_heads)]
    cu_q1 = torch.tensor([0] + list(torch.tensor(lengths1).cumsum(0).tolist()), device=device, dtype=torch.int32)
    per_layer = []
    for layer in range(layers):
        width = max(r["keys"][layer].shape[-2] for r in requests)
        dtype = requests[0]["keys"][layer].dtype
        kf = torch.zeros(B * kv_heads, width, FOLD_DIM, device=device, dtype=dtype)
        vf = torch.zeros_like(kf)
        seqused = torch.ones(B * kv_heads, device=device, dtype=torch.int32)
        for b, r in enumerate(requests):
            key = r["keys"][layer].to(device, non_blocking=True)[0]
            value = r["values"][layer].to(device, non_blocking=True)[0]
            beta = r["betas"][layer].to(device, non_blocking=True)
            w = key.shape[-2]
            finite = torch.isfinite(beta.float())
            active = finite.sum(-1)
            k_f, v_f = fold_prefix(key, value, beta, scaling)
            for h in range(kv_heads):
                idx = finite[h].nonzero(as_tuple=False).flatten()
                n = idx.numel()
                if n < w:
                    packing_stats["repacked_heads"] += 1
                kf[b * kv_heads + h, :n] = k_f[h, idx]
                vf[b * kv_heads + h, :n] = v_f[h, idx]
                if n == 0:
                    kf[b * kv_heads + h, 0, key.shape[-1]] = INACTIVE
                packing_stats["heads"] += 1
            packing_stats["active_entries"] += int(active.sum())
            packing_stats["stored_entries"] += int(w * kv_heads)
            seqused[b * kv_heads:(b + 1) * kv_heads] = active.clamp(min=1).to(torch.int32)
        cu_k = torch.arange(B * kv_heads + 1, device=device, dtype=torch.int32) * width
        per_layer.append(dict(kf=kf.view(-1, 1, FOLD_DIM), vf=vf.view(-1, 1, FOLD_DIM), cu_k=cu_k, width=width, seqused=seqused,
                              cu_q1=cu_q1, cu_q2=cu_q2, max_q=max(lengths), lengths=lengths, lengths1=lengths1,
                              out_k=torch.zeros(1, kv_heads, total, key.shape[-1], device=device, dtype=dtype),
                              out_v=torch.zeros(1, kv_heads, total, key.shape[-1], device=device, dtype=dtype)))
    return dict(ids=ids, positions=positions, per_layer=per_layer, lengths=lengths, modules=modules, packing=packing_stats)


def build_t2t_batch(prompts, device):
    import torch
    lengths = [len(p) for p in prompts]
    ids = torch.tensor([sum(prompts, [])], device=device)
    positions = torch.cat([torch.arange(n) for n in lengths]).view(1, -1).to(device)
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32)
    return dict(ids=ids, positions=positions, cu=cu, lengths=lengths, max_len=max(lengths))


class Graph:
    def __init__(self, fn):
        import torch
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = fn()

    def __call__(self):
        self.graph.replay()
        return self.output


def timeit(fn, warm=3, reps=9):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(median=statistics.median(samples), samples=samples)


def set_impl(model, name):
    model.config._attn_implementation = name
    model.model.config._attn_implementation = name


def first_token_graph(model, batch, use_cache):
    import torch
    last = torch.tensor(batch['lengths'], device=batch['ids'].device).cumsum(0) - 1

    def step():
        hidden = model.model(input_ids=batch['ids'], position_ids=batch['positions'], use_cache=use_cache).last_hidden_state
        return model.lm_head(hidden[0, last]).argmax(-1)
    return Graph(step)


def level_rows(am_payload, level):
    r = am_payload['results_by_level'][level]
    return am_payload['results_by_level'][r['alias_of']] if 'alias_of' in r else r


def subset(config_path, contexts):
    config = sweep.load_config(config_path)
    run = sweep.out_dir(config)
    out, _ = dirs(config)
    pairs, truncated_cells = speed_pairs(config)
    cells = {}
    for pair in pairs:
        for task in config['datasets']:
            truncated = (pair, task) in truncated_cells
            if not truncated and config['pairs'][pair].get('am_overrides', {}).get(task, {}).get('beyond_context'):
                print(f'{pair}/{task}: chunked AM route, skipped', flush=True)
                continue
            suffix = '_truncated' if truncated else ''
            variant = 'truncated' if truncated else sweep.pair_variant(config, pair, task)
            files = {'am_file': run / task / pair / f'am{suffix}.json', 't2t_file': run / task / pair / f't2t{suffix}.json',
                     'summaries': run / task / f'summaries{suffix}' / f"{config['pairs'][pair]['sender']}.json"}
            am = read(files['am_file'])
            if am['variant'] != variant:
                raise ValueError(f"{pair}/{task}: {files['am_file'].name} is variant {am['variant']}, expected {variant}")
            by_id = {s['id']: s['metadata']['context_id'] for s in read(sweep.samples_path(run, task, variant))['samples']}
            rows = level_rows(am, config['prompts'][0])['rows']
            chosen = list(dict.fromkeys(by_id[row['id']] for row in rows))[:contexts + 1]
            summaries = read(files['summaries'])['prompts']
            missing = [(p, c) for p in config['prompts'] for c in chosen if c not in summaries[p]]
            if missing:
                raise ValueError(f'{pair}/{task}: {len(missing)} subset summaries missing')
            questions = sum(by_id[r['id']] in chosen[1:] for r in rows)
            if questions < BATCH:
                raise ValueError(f'{pair}/{task}: {questions} questions, not one full batch of {BATCH}')
            cells.setdefault(pair, {})[task] = {'warmup': chosen[0], 'contexts': chosen[1:], 'questions': questions,
                                                'variant': variant, 'source': 'truncated' if truncated else 'sweep',
                                                **{k: str(v.relative_to(ROOT)) for k, v in files.items()}}
            print(f'{pair}/{task} ({variant}): {len(chosen) - 1} contexts, {questions} questions', flush=True)
    write(out / 'subset.json', {'contexts_per_cell': contexts, 'rule': 'first 1+N contexts of the am.json row order; '
                                'the first is a warmup', 'cells': cells})
    print(out / 'subset.json')


def sender_hf(config, run, manifest, pair_name, cells, target):
    import torch
    from multiagent.evaluation import summaries as ev
    pair = config['pairs'][pair_name]
    device = 'cuda:0'
    tokenizer = ev.load_tokenizer(Path(manifest['models'][pair['sender']]))
    receiver_tokenizer = ev.load_tokenizer(Path(manifest['models'][pair['receiver']]))
    model = ev.load_model(Path(manifest['models'][pair['sender']]), torch.bfloat16, device)
    if hasattr(model, 'set_attn_implementation'):
        model.set_attn_implementation('sdpa')
    model.eval()
    original_generate = model.generate
    forced = {}

    def generate(*args, **kwargs):
        kwargs['max_new_tokens'] = kwargs['min_new_tokens'] = forced['n']
        return original_generate(*args, **kwargs)

    model.generate = generate
    result = read(target) if target.exists() else {'pair': pair_name, 'sender': pair['sender'], 'engine': 'hf',
                                                   'gpu': torch.cuda.get_device_name(0), 'cells': {}}
    torch.manual_seed(config['seed'])

    def one(task, level, cid, contexts, summaries, max_length):
        record = summaries[level][cid]
        if record['context_sha256'] != sweep.sha(contexts[cid]):
            raise ValueError(f'{task}/{cid}: summary was made from different context text')
        forced['n'] = record['generated_tokens_including_special']
        timed = ev.summarize_context(model, tokenizer, receiver_tokenizer, contexts[cid], level, device, max_length)
        if timed['generated_tokens_including_special'] != forced['n']:
            raise RuntimeError(f"{task}/{level}/{cid}: generated {timed['generated_tokens_including_special']} != {forced['n']}")
        return {k: timed[k] for k in ('preparation_ms', 'sender_prefill_ms', 'sender_decode_ms',
                                      'generated_tokens_including_special', 'original_tokens_summarizer')}

    for task, cell in cells.items():
        contexts = sweep.contexts_of(run, task, cell['variant'])
        payload = read(ROOT / cell['summaries'])
        summaries, max_length = payload['prompts'], payload['summarizer_max_length']
        done = result['cells'].setdefault(task, {'warmup': cell['warmup'], 'levels': {}})['levels']
        if all(len(done.get(level, {})) == len(cell['contexts']) for level in config['prompts']):
            print(f'{pair_name}/{task}: done', flush=True)
            continue
        one(task, 'summarize', cell['warmup'], contexts, summaries, max_length)
        for cid in cell['contexts']:
            for level in config['prompts']:
                if cid not in done.get(level, {}):
                    done.setdefault(level, {})[cid] = one(task, level, cid, contexts, summaries, max_length)
            write(target, result)
        means = {level: statistics.mean(v['preparation_ms'] for v in done[level].values()) for level in config['prompts']}
        print(f'{pair_name}/{task}: ' + ', '.join(f'{k} {v:.0f} ms' for k, v in means.items()), flush=True)
    write(target, result)


def sender(config_path, pair_name, datasets=None):
    config = sweep.load_config(config_path)
    run = sweep.out_dir(config)
    manifest = read(run / 'manifest.json')
    out, _ = dirs(config)
    cells = {k: v for k, v in read(out / 'subset.json')['cells'][pair_name].items() if not datasets or k in datasets}
    target = out / f'sender_hf_{pair_name}.json'
    sender_hf(config, run, manifest, pair_name, cells, target)
    print(target)


def am_source(config_path, run, scratch, task, cell):
    if cell['source'] == 'sweep':
        return config_path, scratch / 'am'
    staging = scratch / 'truncated_run'
    config = read(CONFIG_TRUNCATED)
    config.update(output=str(staging.relative_to(ROOT)), scratch=str((scratch / 'truncated_scratch').relative_to(ROOT)))
    staged_config = scratch / 'config_truncated_staged.json'
    write(staged_config, config)
    manifest = read(run / 'manifest.json')
    write(staging / 'manifest.json', manifest)
    write(sweep.samples_path(staging, task, 'truncated'), read(sweep.samples_path(run, task, 'truncated')))
    sweep.write_fit_sidecar(sweep.load_config(staged_config), staging, manifest, task, 'truncated')
    for pair_name in read(CONFIG_TRUNCATED)['pairs']:
        t2t = ROOT / f'{Path(cell["t2t_file"]).parent.parent}/{pair_name}/t2t_truncated.json'
        if t2t.exists() or t2t == ROOT / cell['t2t_file']:
            write(staging / task / pair_name / 't2t.json', read(t2t))
    return staged_config, scratch / 'am_truncated'


def receiver(config_path, pair_name, datasets=None, redo=False, decode_check=False, ttft=False, max_batches=MAX_BATCHES,
             out_dir=None):
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.cache_utils import DynamicCache
    from multiagent.evaluation import renderer_am as ev
    from multiagent.evaluation.native_am import _generate_from_beta_cache_eager
    from multiagent.evaluation.summaries import messages_with_summarized_context
    from multiagent.model.attention_matching import beta_attention
    from multiagent.dataset.c2c import chat_prompt_ids
    from multiagent.dataset.c2c_types import C2CExample
    from multiagent.model.unaware_comm import cache_layer
    config = sweep.load_config(config_path)
    run = sweep.out_dir(config)
    out, scratch = dirs(config)
    cells = {k: v for k, v in read(out / 'subset.json')['cells'][pair_name].items() if not datasets or k in datasets}
    pair = config['pairs'][pair_name]
    target = (Path(out_dir) if out_dir else out) / f'receiver_{pair_name}.json'
    result = read(target) if target.exists() else {'pair': pair_name, 'receiver': pair['receiver'], 'batch': BATCH, 'cells': {}}
    ALL_ATTENTION_FUNCTIONS.register('real_am_varlen', am_varlen_attention)
    ALL_ATTENTION_FUNCTIONS.register('real_t2t_varlen', t2t_varlen_attention)

    captured, contexts, holder, current = {}, {}, {}, {}
    original_renderer = ev.generate_attention_matching_renderer

    def renderer(*args, **kwargs):
        current['sample'] = kwargs['sample']
        return original_renderer(*args, **kwargs)

    @torch.no_grad()
    def capture(model, tokenizer, cache, betas, receiver_decode_ids, logical_position, device, max_new_tokens,
                min_new_tokens=0, timings=None, attention_backend='eager', decode_backend='eager'):
        sample = current['sample']
        holder.update(model=model, tokenizer=tokenizer, device=device)
        cid = str(sample.metadata['context_id'])
        if cid not in contexts:
            keys, values = zip(*(cache_layer(cache, i) for i in range(len(betas))))
            contexts[cid] = dict(keys=[k.to('cpu', copy=True).pin_memory() for k in keys],
                                 values=[v.to('cpu', copy=True).pin_memory() for v in values],
                                 betas=[b.to('cpu', copy=True).pin_memory() for b in betas])
        captured[sample.id] = dict(context_id=cid, ids=list(receiver_decode_ids), position=int(logical_position))
        if timings is not None:
            timings.update(receiver_prefill_ms=0.0, receiver_decode_ms=0.0, captured=True)
        return '', 'captured', 0

    if not hasattr(ev, '_exp2_sweep_model_cache'):
        ev._exp2_sweep_model_cache, ev._exp2_sweep_load_model = {}, ev.load_model

        def cached_load_model(model_path, dtype, device, **kwargs):
            key = (str(model_path), str(dtype), device, str(kwargs.get('adapter')))
            if key not in ev._exp2_sweep_model_cache:
                ev._exp2_sweep_model_cache[key] = ev._exp2_sweep_load_model(model_path, dtype, device, **kwargs)
            return ev._exp2_sweep_model_cache[key]

        ev.load_model = cached_load_model
    ev.generate_attention_matching_renderer = renderer
    ev.generate_from_beta_cache = capture
    original_main = ev.main
    level_timings = {}

    def am_level(level, task_key):
        cell = cells[task_key]
        model, device = holder['model'], holder['device']
        modules = [layer.self_attn for layer in model.model.layers]
        previous = model.config._attn_implementation
        order = interleave([i for i, c in captured.items() if c['context_id'] in cell['contexts']],
                           lambda i: captured[i]['context_id'], max_batches)
        batches, check = [], None
        for index in range(0, len(order), BATCH):
            ids_ = order[index:index + BATCH]
            requests = [dict(**contexts[captured[i]['context_id']], ids=captured[i]['ids'][:-1],
                             position=captured[i]['position']) for i in ids_]
            batch = build_am_batch(model, requests, device)
            for module, p in zip(modules, batch['per_layer']):
                module._am = p
            set_impl(model, 'real_am_varlen')
            graph = Graph(lambda: model.model(input_ids=batch['ids'], position_ids=batch['positions'], use_cache=False))
            timed = timeit(graph)
            first = None
            if ttft:
                del graph
                full = build_am_batch(model, [dict(r, ids=captured[i]['ids']) for r, i in zip(requests, ids_)], device)
                for module, p in zip(modules, full['per_layer']):
                    module._am = p
                graph = first_token_graph(model, full, use_cache=False)
                first = dict(ttft_ms=timeit(graph)['median'], ttft_tokens=sum(full['lengths']), tokens=graph().tolist())
                for module, p in zip(modules, batch['per_layer']):
                    module._am = p
                del full
            if check is None:
                set_impl(model, previous)
                errors, start, decodes = [], 0, []
                for j, req in enumerate(requests if decode_check or ttft else requests[:4]):
                    n = len(req['ids'])
                    cache = DynamicCache(config=model.config)
                    for layer in range(len(req['keys'])):
                        cache.update(req['keys'][layer].to(device), req['values'][layer].to(device), layer)
                    betas = [b.to(device) for b in req['betas']]
                    with beta_attention(model, betas):
                        ref = model.model(input_ids=torch.tensor([req['ids']], device=device),
                                          position_ids=torch.arange(req['position'], req['position'] + n, device=device).view(1, -1),
                                          past_key_values=cache, use_cache=True).past_key_values
                    num = den = 0.0
                    got_cache = DynamicCache(config=model.config) if decode_check else None
                    for layer, p in enumerate(batch['per_layer']):
                        k_ref, v_ref = cache_layer(ref, layer)
                        w = req['keys'][layer].shape[-2]
                        got_k, got_v = p['out_k'][:, :, start:start + n], p['out_v'][:, :, start:start + n]
                        for got, want in ((got_k, k_ref[:, :, w:]), (got_v, v_ref[:, :, w:])):
                            num += (got.float() - want.float()).square().sum().item()
                            den += want.float().square().sum().item()
                        if got_cache is not None:
                            got_cache.update(torch.cat([k_ref[:, :, :w], got_k], dim=2), torch.cat([v_ref[:, :, :w], got_v], dim=2), layer)
                    if j < 4:
                        errors.append((num / den) ** 0.5)
                    if decode_check:
                        last, at = captured[ids_[j]]['ids'][-1], req['position'] + n
                        texts = [_generate_from_beta_cache_eager(model, holder['tokenizer'], c, betas, [last], at, device,
                                                                   config['max_new_tokens'])[0] for c in (ref, got_cache)]
                        decodes.append(dict(id=ids_[j], eager=texts[0], batched=texts[1]))
                    if first is not None:
                        last, at = captured[ids_[j]]['ids'][-1], req['position'] + n
                        with beta_attention(model, betas):
                            logits = model(input_ids=torch.tensor([[last]], device=device),
                                           position_ids=torch.tensor([[at]], device=device), past_key_values=ref,
                                           use_cache=True).logits
                        first.setdefault('eager_tokens', []).append(int(logits[0, -1].argmax()))
                    del cache, ref, got_cache
                    start += n
                check = {'am_relative_rmse_vs_eager': max(errors)}
                if first is not None:
                    check.update(first_token_identical=sum(a == b for a, b in zip(first['tokens'], first['eager_tokens'])),
                                 first_token_n=len(first['eager_tokens']))
                if decode_check:
                    check.update(decode_identical=sum(d['eager'] == d['batched'] for d in decodes), decode_n=len(decodes),
                                 decodes=decodes)
            del graph
            for module in modules:
                del module._am
            set_impl(model, previous)
            batches.append(dict(am_ms=timed['median'], am_tokens=sum(batch['lengths']), packing=batch['packing'], ids=ids_,
                                **({'am_ttft_ms': first['ttft_ms'], 'am_ttft_tokens': first['ttft_tokens'],
                                    'am_first_tokens': first['tokens']} if first else {})))
            del batch
            gc.collect()
            torch.cuda.empty_cache()
        return batches, check

    def main_wrapper():
        captured.clear()
        contexts.clear()
        original_main()
        level = Path(sys.argv[sys.argv.index('--output') + 1]).stem
        level_timings[level] = (*am_level(level, current['task']), dict(captured))
        captured.clear()
        contexts.clear()
        gc.collect()

    ev.main = main_wrapper
    for task_key, cell in cells.items():
        if task_key in result['cells'] and result['cells'][task_key].get('complete') and not redo:
            print(f'{pair_name}/{task_key}: done', flush=True)
            continue
        current['task'] = task_key
        level_timings.clear()
        cell_config, work = am_source(config_path, run, scratch, task_key, cell)
        if (work / task_key / pair_name).exists():
            shutil.rmtree(work / task_key / pair_name)
        sweep.am(cell_config, pair_name, [task_key], len(cell['contexts']) + 1, str(work), config.get('am_decode_backend'))
        merged = read(work / task_key / pair_name / 'am.json')
        if sorted(merged['contexts']) != sorted([cell['warmup'], *cell['contexts']]):
            raise RuntimeError(f'{pair_name}/{task_key}: AM chose different contexts than subset.json')
        samples = {s['id']: s for s in read(sweep.samples_path(run, task_key, cell['variant']))['samples']}
        entry = {'am_levels': {}, 't2t_levels': {}, 'budgets': merged['budgets']}
        for level in config['prompts']:
            block = merged['results_by_level'][level]
            if 'alias_of' in block:
                entry['am_levels'][level] = {'alias_of': block['alias_of']}
                continue
            batches, check, _ = level_timings[level]
            senders = {}
            for row in block['rows']:
                cid = samples[row['id']]['metadata']['context_id']
                if cid in cell['contexts'] and not row['timings'].get('context_cache_hit'):
                    senders[cid] = {k: row['timings'][k] for k in ('sender_ms', 'sender_prefill_ms', 'sender_am_ms')}
            if len(senders) != len(cell['contexts']):
                raise RuntimeError(f'{pair_name}/{task_key}/{level}: {len(senders)} sender timings')
            entry['am_levels'][level] = {
                'sender': senders, 'batches': batches, 'check': check,
                'am_ms_per_request': statistics.mean(b['am_ms'] / BATCH for b in batches),
                'am_tokens_per_request': statistics.mean(b['am_tokens'] / BATCH for b in batches)}
            if ttft:
                entry['am_levels'][level]['am_ttft_ms_per_request'] = statistics.mean(b['am_ttft_ms'] / BATCH for b in batches)
        first = next(level for level in config['prompts'] if 'alias_of' not in entry['am_levels'][level])
        first_captured = level_timings[first][2]
        order = interleave([i for i, c in first_captured.items() if c['context_id'] in cell['contexts']],
                           lambda i: first_captured[i]['context_id'], max_batches)
        model, tokenizer, device = holder['model'], holder['tokenizer'], holder['device']
        modules = [layer.self_attn for layer in model.model.layers]
        previous = model.config._attn_implementation
        summaries = read(ROOT / cell['summaries'])['prompts']
        for level in config['prompts']:
            batches = []
            for index in range(0, len(order), BATCH):
                ids_ = order[index:index + BATCH]
                prompts = []
                for i in ids_:
                    sample = C2CExample(**samples[i])
                    messages = messages_with_summarized_context(sample, summaries[level][sample.metadata['context_id']]['text'])
                    prompts.append(chat_prompt_ids(tokenizer, messages, config['max_length'], add_generation_prompt=True))
                batch = build_t2t_batch([p[:-1] for p in prompts], device)
                for module in modules:
                    module._t2t = batch
                set_impl(model, 'real_t2t_varlen')
                graph = Graph(lambda: model.model(input_ids=batch['ids'], position_ids=batch['positions'], use_cache=True))
                timed = timeit(graph)
                del graph
                extra = {}
                if ttft:
                    full = build_t2t_batch(prompts, device)
                    for module in modules:
                        module._t2t = full
                    graph = first_token_graph(model, full, use_cache=True)
                    extra = dict(t2t_ttft_ms=timeit(graph)['median'], t2t_ttft_tokens=sum(full['lengths']))
                    del graph, full
                for module in modules:
                    del module._t2t
                set_impl(model, previous)
                batches.append(dict(t2t_ms=timed['median'], t2t_tokens=sum(batch['lengths']), ids=ids_, **extra))
                gc.collect()
                torch.cuda.empty_cache()
            entry['t2t_levels'][level] = {'batches': batches,
                                          't2t_ms_per_request': statistics.mean(b['t2t_ms'] / BATCH for b in batches),
                                          't2t_tokens_per_request': statistics.mean(b['t2t_tokens'] / BATCH for b in batches)}
            if ttft:
                entry['t2t_levels'][level]['t2t_ttft_ms_per_request'] = statistics.mean(b['t2t_ttft_ms'] / BATCH for b in batches)
        if task_key in result['cells']:
            entry['previous'] = {k: v for k, v in result['cells'][task_key].items() if k != 'previous'}
        entry['complete'] = True
        entry['gpu'] = torch.cuda.get_device_name(0)
        result['cells'][task_key] = entry
        write(target, result)
        print(f'{pair_name}/{task_key}: ' + ', '.join(
            f"{lv} T2T {entry['t2t_levels'][lv]['t2t_ms_per_request']:.1f} / AM "
            f"{level_rows({'results_by_level': entry['am_levels']}, lv)['am_ms_per_request']:.1f} ms"
            for lv in config['prompts']), flush=True)
    print(target)


def queue(config_path, gpus, datasets=None):
    me = str(Path(__file__).resolve())
    commands = []
    for pair in speed_pairs(sweep.load_config(config_path))[0]:
        for step in ('sender', 'receiver'):
            commands.append(shlex.join([str(PYTHON), '-u', me, step, '--config', str(config_path), '--pair', pair,
                                        *(['--datasets', *datasets] if datasets else [])]))
    tag = 'exp2_sweep_timings' + (f"_{'_'.join(datasets)}" if datasets else '')
    queue_file = ROOT / f'multiagent/gpu_queue_commands/gpu_queue_commands_{tag}.json'
    write(queue_file, commands)
    print(f'{len(commands)} commands -> {queue_file}')
    print(f'cd {ROOT / "multiagent"} && setsid nohup ../env3.10/bin/python gpu_queue.py '
          f'--commands {queue_file.relative_to(ROOT / "multiagent")} --gpus {gpus} --name {tag} '
          f'> gpu_queue_logs/{tag}.nohup.log 2>&1 &')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['subset', 'sender', 'receiver', 'queue'])
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--pair', help=f'timed pair (default config: {", ".join(PAIRS)}; else its speed_pairs)')
    parser.add_argument('--contexts', type=int, default=CONTEXTS)
    parser.add_argument('--datasets', nargs='*', help='sender/receiver/queue: only these datasets')
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--redo', action='store_true', help='receiver: re-measure complete cells (old entry kept as previous)')
    parser.add_argument('--decode-check', action='store_true', help='receiver: greedy answers from batched vs eager K/V')
    parser.add_argument('--ttft', action='store_true', help='receiver: also time the first answer token on the same batches')
    parser.add_argument('--max-batches', type=int, default=MAX_BATCHES, help='receiver: full batches timed per level')
    parser.add_argument('--out', type=Path, help='receiver: write receiver_<pair>.json here instead of the results dir')
    args = parser.parse_args()
    if args.command == 'subset':
        subset(args.config, args.contexts)
    elif args.command == 'sender':
        sender(args.config, args.pair, args.datasets)
    elif args.command == 'receiver':
        receiver(args.config, args.pair, args.datasets, args.redo, args.decode_check, args.ttft, args.max_batches, args.out)
    else:
        queue(args.config, args.gpus, args.datasets)


if __name__ == '__main__':
    main()
