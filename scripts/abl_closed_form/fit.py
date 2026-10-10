from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.abl_closed_form.mapper import ClosedFormMapper, ridge_from_moments, same_head_ols_r2
from multiagent.dataset.c2c_cross_family import encode_context_pair
from multiagent.model.cache_renderer import kv_layer_indices
from multiagent.model.unaware_comm import _apply_rope, _base_model, cache_layer
from multiagent.model.vanilla_loader import load_causal_model
from multiagent.paths import MODELS_ROOT

SUPPORTED = {'qwen3', 'qwen2', 'llama'}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def geometry(path):
    c = AutoConfig.from_pretrained(path)
    if c.model_type not in SUPPORTED:
        raise ValueError(f'Unsupported model type {c.model_type!r} (dense full-attention models only)')
    layers = len(kv_layer_indices(c))
    if layers != c.num_hidden_layers:
        raise ValueError('Every layer must carry a KV cache')
    heads = int(getattr(c, 'num_key_value_heads', None) or c.num_attention_heads)
    dim = int(getattr(c, 'head_dim', None) or c.hidden_size // c.num_attention_heads)
    return dict(layers=layers, heads=heads, dim=dim)


def tokenizers_identical(a, b):
    return a.get_vocab() == b.get_vocab() and a.backend_tokenizer.to_str() == b.backend_tokenizer.to_str()


def build_alignment(calibration_dir, manifest, sender_tokenizer, receiver_tokenizer, max_length):
    stride = manifest['stride']
    docs = []
    if (calibration_dir / 'input_ids.npy').exists():
        if not tokenizers_identical(sender_tokenizer, receiver_tokenizer):
            raise ValueError('Token calibration (input_ids.npy) requires identical tokenizers; use texts.jsonl')
        for row in np.load(calibration_dir / 'input_ids.npy'):
            positions = list(range(0, len(row), stride))
            docs.append({'sender_ids': row.tolist(), 'receiver_ids': row.tolist(),
                         'receiver_positions': positions, 'sender_index': positions})
        return docs
    for line in (calibration_dir / 'texts.jsonl').read_text().splitlines():
        text = json.loads(line)['text']
        row = encode_context_pair(sender_tokenizer, receiver_tokenizer, [{'role': 'raw', 'content': text}],
                                  max_length, 'interp')
        sender_ids, receiver_ids = row['sender_ids'], row['receiver_ids']
        if not sender_ids or not receiver_ids:
            raise ValueError('Empty tokenization')
        positions = list(range(0, len(receiver_ids), stride))
        limits = np.asarray(row['cross_causal_limits'])
        index = np.clip(limits[positions] - 1, 0, len(sender_ids) - 1).tolist()
        docs.append({'sender_ids': sender_ids, 'receiver_ids': receiver_ids,
                     'receiver_positions': positions, 'sender_index': index})
    return docs


@torch.inference_mode()
def extract(model_path, role, docs, out, device, fingerprint):
    dims = geometry(model_path)
    marker = out / f'{role}_complete.json'
    total = sum(len(d['receiver_positions']) for d in docs)
    expected = {'model': model_path, 'geometry': dims, 'calibration_fingerprint': fingerprint, 'tokens': total}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError(f'Stale extraction: {marker}')
        print(f'Reusing completed {role} features', flush=True)
        return dims
    shape = (dims['layers'], total, dims['heads'], dims['dim'])
    arrays = [np.lib.format.open_memmap(out / f'{role}_{kind}.npy', mode='w+',
                                      dtype=np.float16, shape=shape) for kind in ['K', 'V']]
    model = load_causal_model(model_path, torch.bfloat16, device).eval()
    model.requires_grad_(False)
    base = _base_model(model)
    layer_ids = kv_layer_indices(base.config)
    started = time.monotonic()
    offset = 0
    for i, doc in enumerate(docs):
        ids = doc[f'{role}_ids']
        select = doc['sender_index'] if role == 'sender' else doc['receiver_positions']
        inputs = torch.tensor(ids, device=device, dtype=torch.long)[None]
        position_ids = torch.arange(len(ids), device=device)[None]
        selected = torch.tensor(select, device=device)
        cache = base(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                     position_ids=position_ids, use_cache=True).past_key_values
        begin, end = offset, offset + len(select)
        for slot, layer in enumerate(layer_ids):
            key, value = cache_layer(cache, layer)
            key = _apply_rope(base, key, layer, position_ids, inverse=True)
            for array, feature in zip(arrays, [key, value]):
                feature = feature[0].index_select(1, selected).permute(1, 0, 2).to(torch.float16)
                if not torch.isfinite(feature).all():
                    raise ValueError(f'Nonfinite {role} features, sequence {i}, layer {layer}')
                array[slot, begin:end] = feature.cpu().numpy()
        offset = end
        del cache, key, value, feature
        if (i + 1) % 25 == 0 or i + 1 == len(docs):
            print(f'{role}: {i + 1}/{len(docs)} sequences, {time.monotonic() - started:.1f}s', flush=True)
    assert offset == total
    for array in arrays:
        array.flush()
    del arrays, model, base
    gc.collect()
    torch.cuda.empty_cache()
    write_json(marker, expected)
    return dims


def load_features(out, role, kind):
    return np.load(out / f'{role}_{kind}.npy', mmap_mode='r')


def rows(array, layer, sl, device, dtype=torch.float32):
    return torch.tensor(np.array(array[layer, sl]), device=device, dtype=dtype).flatten(1)


@torch.inference_mode()
def layer_ranking(out, nfit, device, matched, chunk_rows=16384, chunk_layers=4):
    cached = out / 'layer_scores.pt'
    if cached.exists():
        print('Reusing completed layer probe', flush=True)
        return torch.load(cached)['mean_kv_head_r2']
    results = []
    for kind in ['K', 'V']:
        source, target = load_features(out, 'sender', kind), load_features(out, 'receiver', kind)
        ls, lt = len(source), len(target)
        if matched:
            scores = torch.empty(ls, lt, source.shape[2], dtype=torch.float64)
            for s in range(ls):
                x = torch.tensor(np.array(source[s, :nfit]), device=device, dtype=torch.float64).permute(1, 0, 2)
                for start in range(0, lt, chunk_layers):
                    stop = min(start + chunk_layers, lt)
                    y = torch.tensor(np.array(target[start:stop, :nfit]), device=device, dtype=torch.float64)
                    y = y.permute(2, 1, 0, 3).reshape(source.shape[2], nfit, -1)
                    scores[s, start:stop] = same_head_ols_r2(x, y).cpu()
                    del y
                del x
                print(f'Layer probe {kind}: source {s + 1}/{ls}', flush=True)
            results.append(scores.mean(-1))
        else:
            chunks = [slice(a, min(a + chunk_rows, nfit)) for a in range(0, nfit, chunk_rows)]
            stats = []
            for s in range(ls):
                x = rows(source, s, slice(0, nfit), device)
                mx = x.double().mean(0)
                gram = sum((x[c].double() - mx).T @ (x[c].double() - mx) for c in chunks)
                eigenvalues, vectors = torch.linalg.eigh(gram)
                inverse = torch.where(eigenvalues > eigenvalues.max() * 1e-10, eigenvalues.clamp_min(1e-30).reciprocal(), 0)
                stats.append((mx, gram, (vectors * inverse) @ vectors.T))
                del x
            scores = torch.empty(ls, lt, dtype=torch.float64)
            for t in range(lt):
                y = rows(target, t, slice(0, nfit), device)
                my = y.double().mean(0)
                sst = sum((y[c].double() - my).square().sum() for c in chunks)
                for s in range(ls):
                    mx, gram, pinv = stats[s]
                    x = rows(source, s, slice(0, nfit), device)
                    cross = sum((x[c].double() - mx).T @ (y[c].double() - my) for c in chunks)
                    w = pinv @ cross
                    sse = (sst - 2 * (w * cross).sum() + (w * (gram @ w)).sum()).clamp_min(0)
                    scores[s, t] = 1 - sse / sst.clamp_min(1e-20)
                    del x, cross, w
                del y
                print(f'Layer probe {kind}: target {t + 1}/{lt}', flush=True)
            results.append(scores)
    scores = torch.stack(results)
    ranking = scores.mean(0)
    if not torch.isfinite(ranking).all():
        raise ValueError('Nonfinite layer-ranking scores')
    temp = out / 'layer_scores.pt.tmp'
    torch.save({'r2_kv': scores, 'mean_kv_head_r2': ranking, 'probe': 'same-head' if matched else 'all-head'}, temp)
    temp.replace(cached)
    return ranking


def select_layers(ranking, k, out, matched):
    selected = ranking.topk(k, dim=0).indices.T.contiguous()
    write_json(out / 'layer_selection.json', {'k': k, 'source_layers_zero_based': selected.tolist(),
        'criterion': ('in-sample same-index-head OLS R2, pooled channels per head, equal head/K/V means' if matched
                      else 'in-sample all-head single-source OLS R2, pooled target channels, equal K/V means') + '; top-k per target layer',
        'rcond_gram': 1e-10, 'heldout_used_for_selection': False,
        'scores_source_by_target': ranking.tolist()})
    return selected


def gather_x(source, layers, sl, device):
    return torch.cat([rows(source, s, sl, device) for s in layers], dim=1)


@torch.inference_mode()
def solve_layer(source, target, layers, t, nfit, device, ridge_lambda, rows_per_chunk):
    y_dim = target.shape[2] * target.shape[3]
    x_dim = len(layers) * source.shape[2] * source.shape[3]
    chunks = [slice(a, min(a + rows_per_chunk, nfit)) for a in range(0, nfit, rows_per_chunk)]
    sx = torch.zeros(x_dim, device=device, dtype=torch.float64)
    sy = torch.zeros(y_dim, device=device, dtype=torch.float64)
    for sl in chunks:
        sx += gather_x(source, layers, sl, device).double().sum(0)
        sy += rows(target, t, sl, device, torch.float64).sum(0)
    mx, my = sx / nfit, sy / nfit
    gram = torch.zeros(x_dim, x_dim, device=device, dtype=torch.float64)
    cross = torch.zeros(x_dim, y_dim, device=device, dtype=torch.float64)
    for sl in chunks:
        xc = gather_x(source, layers, sl, device).double() - mx
        yc = rows(target, t, sl, device, torch.float64) - my
        gram += xc.T @ xc
        cross += xc.T @ yc
        del xc, yc
    weight, bias, residual = ridge_from_moments(gram, cross, mx, my, ridge_lambda)
    del gram, cross
    return weight, bias, residual


@torch.inference_mode()
def regression_metrics(source, target, layers, t, span, weight, bias, heads, dim, device, rows_per_chunk):
    sse = torch.zeros(heads, device=device, dtype=torch.float64)
    sy = torch.zeros(heads * dim, device=device, dtype=torch.float64)
    syy = torch.zeros(heads * dim, device=device, dtype=torch.float64)
    total = 0
    for a in range(span.start, span.stop, rows_per_chunk):
        sl = slice(a, min(a + rows_per_chunk, span.stop))
        y = rows(target, t, sl, device, torch.float64)
        pred = (gather_x(source, layers, sl, device) @ weight + bias).double()
        sse += (pred - y).square().reshape(-1, heads, dim).sum((0, 2))
        sy += y.sum(0)
        syy += y.square().sum(0)
        total += y.shape[0]
    sst = (syy - sy.square() / total).reshape(heads, dim).sum(1).clamp_min(1e-30)
    r2 = 1 - sse / sst
    return {'head_r2': r2.tolist(), 'mean_r2': float(r2.mean()), 'relative_mse': float(sse.sum() / syy.sum().clamp_min(1e-30))}


def fit(features, selected, dims, nfit, ntotal, device, ridge_lambda, rows_per_chunk):
    k = selected.shape[1]
    mapper = ClosedFormMapper(sender_layers=dims['sender']['layers'], receiver_layers=dims['receiver']['layers'],
                              kv_heads=dims['sender']['heads'], head_dim=dims['sender']['dim'],
                              receiver_kv_heads=dims['receiver']['heads'], receiver_head_dim=dims['receiver']['dim'], k=k)
    mapper.source_layers.copy_(selected)
    heads, dim = dims['receiver']['heads'], dims['receiver']['dim']
    report = []
    for kind_index, kind in enumerate(['K', 'V']):
        source, target = load_features(features, 'sender', kind), load_features(features, 'receiver', kind)
        for t in range(dims['receiver']['layers']):
            layers = selected[t].tolist()
            weight, bias, residual = solve_layer(source, target, layers, t, nfit, device, ridge_lambda, rows_per_chunk)
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all() or residual > 1e-7:
                raise RuntimeError(f'Invalid solve at layer {t}, {kind}: residual={residual}')
            w, b = weight.float(), bias.float()
            mapper.weight[t, kind_index].copy_(w.cpu())
            mapper.bias[t, kind_index].copy_(b.cpu())
            row = {'target_layer': t, 'source_layers': layers, 'kind': kind, 'solve_relative_residual': residual,
                   'fit': regression_metrics(source, target, layers, t, slice(0, nfit), w, b, heads, dim, device, rows_per_chunk),
                   'heldout': regression_metrics(source, target, layers, t, slice(nfit, ntotal), w, b, heads, dim, device, rows_per_chunk)}
            report.append(row)
            print(f'Ridge {kind} {layers}->{t}: fit R2={row["fit"]["mean_r2"]:.4f}, '
                  f'heldout R2={row["heldout"]["mean_r2"]:.4f}', flush=True)
            del weight, bias, w, b
            torch.cuda.empty_cache()
    return mapper, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--features-dir', type=Path, help='shared cache features + layer probe (default: output dir)')
    p.add_argument('--k', type=int, default=1, help='source layers per target layer; 0 = all')
    p.add_argument('--sender', default=str(MODELS_ROOT / 'qwen/Qwen3-4B'))
    p.add_argument('--receiver', default=str(MODELS_ROOT / 'qwen/Qwen3-8B'))
    p.add_argument('--max-length', type=int, default=1024, help='per-model token cap for text calibration')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ridge-lambda', type=float, default=0.01)
    p.add_argument('--rows-per-chunk', type=int, default=16384)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.output_dir
    features = args.features_dir or out
    out.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    if (out / 'renderer.pt').exists():
        raise FileExistsError(f'Mapper already exists: {out / "renderer.pt"}')
    manifest_path = args.calibration_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['heldout_sequences'] < 1 or args.ridge_lambda <= 0:
        raise ValueError('Need heldout documents and positive ridge penalty')
    for name, key in [('input_ids.npy', 'input_ids_sha256'), ('texts.jsonl', 'texts_sha256')]:
        if (args.calibration_dir / name).exists():
            if hashlib.sha256((args.calibration_dir / name).read_bytes()).hexdigest() != manifest[key]:
                raise ValueError(f'Calibration hash mismatch: {name}')
    dims = {'sender': geometry(args.sender), 'receiver': geometry(args.receiver)}
    matched = dims['sender']['heads'] == dims['receiver']['heads'] and dims['sender']['dim'] == dims['receiver']['dim']
    k = args.k or dims['sender']['layers']
    sender_tokenizer, receiver_tokenizer = [AutoTokenizer.from_pretrained(path) for path in [args.sender, args.receiver]]
    identical = tokenizers_identical(sender_tokenizer, receiver_tokenizer)
    docs = build_alignment(args.calibration_dir, manifest, sender_tokenizer, receiver_tokenizer, args.max_length)
    counts = [len(d['receiver_positions']) for d in docs]
    nfit, ntotal = sum(counts[:manifest['fit_sequences']]), sum(counts)
    alignment = {'identical_tokenizers': identical, 'max_length': args.max_length, 'stride': manifest['stride'],
                 'rule': 'identity' if identical else 'receiver slot i <- sender token cross_causal_limits[i]-1 (encode_context_pair, raw text)',
                 'documents': len(docs), 'fit_tokens': nfit, 'heldout_tokens': ntotal - nfit,
                 'mean_sender_tokens': float(np.mean([len(d['sender_ids']) for d in docs])),
                 'mean_receiver_tokens': float(np.mean([len(d['receiver_ids']) for d in docs]))}
    fingerprint = hashlib.sha256(manifest_path.read_bytes() + json.dumps(alignment, sort_keys=True).encode()).hexdigest()
    write_json(features / 'alignment.json', {**alignment, 'fingerprint': fingerprint})
    print(json.dumps(alignment), flush=True)
    started = time.monotonic()
    for role in ['sender', 'receiver']:
        extract(getattr(args, role), role, docs, features, args.device, fingerprint)
    ranking = layer_ranking(features, nfit, args.device, matched)
    selected = select_layers(ranking, k, out, matched)
    mapper, report = fit(features, selected, dims, nfit, ntotal, args.device, args.ridge_lambda, args.rows_per_chunk)
    metadata = {'paper': 'https://arxiv.org/abs/2608.03893', 'k': k, 'geometry': dims, 'matched_kv': matched,
                'layer_probe': 'same-head' if matched else 'all-head', 'alignment': alignment,
                'ridge_lambda': args.ridge_lambda, 'ridge_convention': 'unnormalized centered X.T@X + lambda*I; unpenalized intercept',
                'calibration_manifest': str(manifest_path.resolve()), 'calibration_fingerprint': fingerprint,
                'features_dir': str(features.resolve()),
                'fit_tokens': nfit, 'heldout_tokens': ntotal - nfit,
                'precision': {'forward': 'bfloat16', 'feature_storage': 'float16', 'covariance_and_solve': 'float64', 'mapper': 'float32'},
                'deviations': ['FP64 covariance/solve for stability rather than paper FP32 covariance',
                               'Exact calibration documents unspecified in paper; pinned sample-10BT selection',
                               'Additional heldout documents used only for diagnostics, never selection or fitting',
                               'k selected by held-out calibration R2, not benchmark accuracy']
                              + ([] if matched else ['Mismatched KV heads: all-head layer probe (paper only studies matched-KV pairs)'])
                              + ([] if identical else ['Different tokenizers: repository character alignment (paper only studies shared tokenizers)']),
                'elapsed_seconds': time.monotonic() - started,
                'parameter_count': mapper.weight.numel() + mapper.bias.numel()}
    write_json(out / 'fit_report.json', {**metadata, 'regressions': report})
    payload = {'renderer_config': mapper.config_dict(), 'renderer_state_dict': mapper.state_dict(),
               'sender_model': args.sender, 'receiver_model': args.receiver,
               'token_alignment': {'mode': 'native', 'version': 1}, 'position_mode': 'interp',
               'args': {'causal_cross_attention': True}, 'closed_form_metadata': metadata}
    temporary = out / 'renderer.pt.tmp'
    torch.save(payload, temporary)
    temporary.replace(out / 'renderer.pt')
    write_json(out / 'complete.json', metadata)
    print(f'Completed mapper: {out / "renderer.pt"}', flush=True)


if __name__ == '__main__':
    main()
