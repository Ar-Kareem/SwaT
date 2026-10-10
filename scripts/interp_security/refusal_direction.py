import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import ROOT, PAIRS, data, append, rows, MAX_NEW_TOKENS

DATA = ROOT / 'multiagent/outputs_mine/interp_security/causal'
SCRATCH = ROOT / 'multiagent/outputs/sandbox_agent/refusal_direction_20260928'
SCORES = ROOT / 'multiagent/outputs_mine/interp_security/scores/strongreject'
POS = 'last'


def prompts():
    return [(r['id'], r['prompt'], r['label'] == 'unsafe') for r in data()]


def header_states(model, ids, mask, positions, cache):
    out = model(input_ids=ids, attention_mask=mask, position_ids=positions, past_key_values=cache,
                use_cache=cache is not None, output_hidden_states=True, logits_to_keep=1)
    hs = torch.stack(out.hidden_states, 0)[:, 0].transpose(0, 1)
    assert hs.shape[0] == ids.shape[1]
    return hs.to(torch.float32).cpu()


def activations(pair):
    from runtime import Runner, prompt_ids
    from multiagent.dataset.c2c import generation_prompt_ids
    from multiagent.model.unaware_comm import tuple_to_dynamic_cache
    runner = Runner(pair, 'renderer')
    dev = runner.device
    r_header = generation_prompt_ids(runner.receiver_tok)
    s_header = generation_prompt_ids(runner.sender_tok)
    store = {k: [] for k in ('sender_last', 'sender_mean', 'native_last', 'native_mean', 'rendered_last', 'rendered_mean')}
    ids_, labels = [], []
    with torch.inference_mode():
        for n, (pid, prompt, unsafe) in enumerate(prompts()):
            ids_.append(pid); labels.append(unsafe)
            full = prompt_ids(runner.sender_tok, prompt, True)
            assert full[-len(s_header):] == s_header
            t = torch.tensor([full], device=dev)
            hs = header_states(runner.sender, t, torch.ones_like(t), torch.arange(len(full), device=dev)[None], None)[-len(s_header):]
            store['sender_last'].append(hs[-1].numpy()); store['sender_mean'].append(hs.float().mean(0).numpy())
            full = prompt_ids(runner.receiver_tok, prompt, True)
            assert full[-len(r_header):] == r_header
            t = torch.tensor([full], device=dev)
            hs = header_states(runner.model, t, torch.ones_like(t), torch.arange(len(full), device=dev)[None], None)[-len(r_header):]
            store['native_last'].append(hs[-1].numpy()); store['native_mean'].append(hs.float().mean(0).numpy())
            cache = runner.render(prompt)
            length = cache[0][0].shape[2]
            assert length == len(full) - len(r_header)
            dyn = tuple_to_dynamic_cache(cache, runner.model.config)
            t = torch.tensor([r_header], device=dev)
            mask = torch.ones((1, length + len(r_header)), dtype=torch.long, device=dev)
            pos = (length + torch.arange(len(r_header), device=dev))[None]
            hs = header_states(runner.model, t, mask, pos, dyn)
            store['rendered_last'].append(hs[-1].numpy()); store['rendered_mean'].append(hs.float().mean(0).numpy())
            if n % 50 == 0:
                print(f'{pair}: {n}/450', flush=True)
    np.savez(DATA / f'activations_{pair}.npz', ids=np.array(ids_), unsafe=np.array(labels),
             sender_header=np.array(s_header), receiver_header=np.array(r_header),
             **{k: np.stack(v) for k, v in store.items()})
    print(f'{pair}: saved', flush=True)


def steer(pair, tag, arm='renderer', layer=None, alpha=None, direction_path=None):
    from runtime import Runner
    from multiagent.model.unaware_comm import _base_model
    if layer is None or alpha is None or direction_path is None:
        res = analyze(pair)
        layer = layer if layer is not None else res['layer']
        alpha = alpha if alpha is not None else res['alpha_restore']
        fitted = SCRATCH / f"direction_{pair}_L{res['layer']}.npy"
        if direction_path is None:
            direction_path = random_direction(pair, fitted) if tag == 'random' else fitted
    print(f'{pair}/{tag}: layer {layer}, alpha {alpha:.4f}, direction {direction_path}', flush=True)
    runner = Runner(pair, arm)
    d = np.load(direction_path)
    d_hat = torch.tensor(d / np.linalg.norm(d), dtype=torch.bfloat16, device=runner.device)
    base = _base_model(runner.model)
    block = base.layers[layer - 1]

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + alpha * d_hat,) + tuple(output[1:])
        return output + alpha * d_hat
    handle = block.register_forward_hook(hook)
    target = SCRATCH / f'steer_{pair}_{tag}.jsonl'
    done = {r['id'] for r in rows(target)}
    unsafe = [(pid, p) for pid, p, u in prompts() if u and pid not in done]
    torch.manual_seed(42)
    for i in range(0, len(unsafe), 12):
        chunk = unsafe[i:i + 12]
        preds = runner.batch([p for _, p in chunk], MAX_NEW_TOKENS)
        append(target, [{'id': pid, **pred} for (pid, _), pred in zip(chunk, preds)])
        print(f'{pair}/{tag}: {len(done) + i + len(chunk)}/200', flush=True)
    handle.remove()
    (SCRATCH / f'steer_{pair}_{tag}.meta.json').write_text(json.dumps(
        {'pair': pair, 'arm': arm, 'layer': layer, 'alpha': alpha, 'direction': str(direction_path)}, indent=2))


def folds(unsafe, k=5, seed=0):
    rng = np.random.default_rng(seed)
    assign = np.empty(len(unsafe), int)
    for label in (True, False):
        idx = np.where(unsafe == label)[0]; rng.shuffle(idx)
        assign[idx] = np.arange(len(idx)) % k
    return assign


def cv_project(native, others, unsafe, assign):
    N, L, _ = native.shape
    z = {name: np.zeros((N, L)) for name in ['native', *others]}
    raw = {name: np.zeros((N, L)) for name in ['native', *others]}
    for f in np.unique(assign):
        tr, te = assign != f, assign == f
        d = native[tr & unsafe].mean(0) - native[tr & ~unsafe].mean(0)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(norm > 0, d / np.maximum(norm, 1e-12), 0)
        m_safe = np.einsum('nld,ld->nl', native[tr & ~unsafe], d).mean(0)
        m_harm = np.einsum('nld,ld->nl', native[tr & unsafe], d).mean(0)
        for name, X in [('native', native), *others.items()]:
            p = np.einsum('nld,ld->nl', X[te], d)
            raw[name][te] = p
            gap = m_harm - m_safe
            z[name][te] = np.where(gap != 0, (p - m_safe) / np.where(gap != 0, gap, 1), 0)
    return z, raw


def transferred_scores(pair, ids):
    d = {}
    for line in (SCORES / f'{pair}.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['view'] == 'primary':
            d.setdefault(r['id'], {})[r['arm']] = r['score']
    return {arm: np.array([d[i][arm] if i in d else np.nan for i in ids]) for arm in ('sender', 'receiver', 'renderer')}


def boot_ols(y, X, n=2000, seed=0):
    X = np.c_[np.ones(len(y)), X]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed); B = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y)); B.append(np.linalg.lstsq(X[k], y[k], rcond=None)[0])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    return b, lo, hi


def fit(pair):
    a = np.load(DATA / f'activations_{pair}.npz')
    unsafe = a['unsafe'].astype(bool); ids = [str(i) for i in a['ids']]
    native = a[f'native_{POS}'].astype(np.float32); rendered = a[f'rendered_{POS}'].astype(np.float32)
    sender = a[f'sender_{POS}'].astype(np.float32)
    assign = folds(unsafe)
    z, raw = cv_project(native, {'rendered': rendered}, unsafe, assign)
    zs, _ = cv_project(sender, {}, unsafe, assign)
    L = native.shape[1]
    auc_native = np.array([roc_auc_score(unsafe, z['native'][:, l]) for l in range(L)])
    auc_rend = np.array([roc_auc_score(unsafe, z['rendered'][:, l]) for l in range(L)])
    auc_send = np.array([roc_auc_score(unsafe, zs['native'][:, l]) for l in range(sender.shape[1])])
    cand = range(L // 4, 3 * L // 4)
    layer = max(cand, key=lambda l: auc_native[l])
    slayer = max(range(sender.shape[1] // 4, 3 * sender.shape[1] // 4), key=lambda l: auc_send[l])
    sr = transferred_scores(pair, ids)
    zn, zr, zsend = z['native'][:, layer], z['rendered'][:, layer], zs['native'][:, slayer]
    res = {'pair': pair, 'layer': int(layer), 'sender_layer': int(slayer), 'n_layers': int(L),
           'auroc_native': float(auc_native[layer]), 'auroc_rendered': float(auc_rend[layer]), 'auroc_sender': float(auc_send[slayer]),
           'z_native_unsafe': float(zn[unsafe].mean()), 'z_native_safe': float(zn[~unsafe].mean()),
           'z_rendered_unsafe': float(zr[unsafe].mean()), 'z_rendered_safe': float(zr[~unsafe].mean()),
           'z_sender_unsafe': float(zsend[unsafe].mean()),
           'auroc_by_layer_native': auc_native.round(3).tolist(), 'auroc_by_layer_rendered': auc_rend.round(3).tolist()}
    u = unsafe
    res['rho_zrend_vs_transferred_score'] = float(spearmanr(zr[u], sr['renderer'][u]).statistic)
    res['rho_znative_vs_receiver_score'] = float(spearmanr(zn[u], sr['receiver'][u]).statistic)
    res['rho_zsender_vs_sender_score'] = float(spearmanr(zsend[u], sr['sender'][u]).statistic)
    res['rho_zrend_vs_znative_unsafe'] = float(pearsonr(zr[u], zn[u]).statistic)
    res['rho_zrend_vs_zsender_unsafe'] = float(pearsonr(zr[u], zsend[u]).statistic)
    b, lo, hi = boot_ols(zr[u], np.c_[zn[u], zsend[u]])
    res['ols_zrend_on_znative_zsender_unsafe'] = {'b_native': [b[1], lo[1], hi[1]], 'b_sender': [b[2], lo[2], hi[2]]}
    b, lo, hi = boot_ols(zr, np.c_[zn, zsend])
    res['ols_zrend_on_znative_zsender_all450'] = {'b_native': [b[1], lo[1], hi[1]], 'b_sender': [b[2], lo[2], hi[2]]}
    d = native[unsafe, layer].mean(0) - native[~unsafe, layer].mean(0); d_hat = d / np.linalg.norm(d)
    pn, pr = native[:, layer] @ d_hat, rendered[:, layer] @ d_hat
    res['alpha_restore'] = float(pn[unsafe].mean() - pr[unsafe].mean())
    res['alpha_native_gap'] = float(pn[unsafe].mean() - pn[~unsafe].mean())
    return res, d, dict(unsafe=unsafe, ids=ids, zn=zn, zr=zr, zsend=zsend, sr=sr, auc_native=auc_native, auc_rend=auc_rend)


def analyze(pair):
    res, d, arrays = fit(pair)
    np.save(SCRATCH / f"direction_{pair}_L{res['layer']}.npy", d)
    (SCRATCH / f'analysis_{pair}.json').write_text(json.dumps(res, indent=2) + '\n')
    u = arrays['unsafe']
    rng = np.random.default_rng(0)

    def bar(values):
        boots = [rng.choice(values, len(values)).mean() for _ in range(2000)]
        return {'mean': float(values.mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    ids = [i for i, uu in zip(arrays['ids'], u) if uu]
    bars = {'receiver': bar(arrays['sr']['receiver'][u]), 'swat': bar(arrays['sr']['renderer'][u])}
    for tag in ('random', 'restore'):
        f = DATA / f'steer_{pair}_{tag}.scores.jsonl'
        if f.exists():
            sc = {json.loads(l)['id']: json.loads(l)['score'] for l in f.read_text().splitlines() if l.strip()}
            bars[tag] = bar(np.array([sc[i] for i in ids]))
    plot = {
        'pair': pair, 'sender': PAIRS[pair][0], 'receiver': PAIRS[pair][1], 'layer': res['layer'],
        'projection': {'unsafe': u.tolist(), 'native': arrays['zn'].tolist(), 'swat': arrays['zr'].tolist()},
        'projection_means': {'native_safe': float(arrays['zn'][~u].mean()), 'native_unsafe': float(arrays['zn'][u].mean()),
                             'swat_safe': float(arrays['zr'][~u].mean()), 'swat_unsafe': float(arrays['zr'][u].mean())},
        'strongreject': bars,
        'harm_ratio_swat_over_receiver': bars['swat']['mean'] / bars['receiver']['mean'],
        'safer_ratio_swat_over_restore': bars['swat']['mean'] / bars['restore']['mean'] if 'restore' in bars else None,
        'n_unsafe': int(u.sum()), 'n_safe': int((~u).sum()),
    }
    (DATA / f'plot_{pair}.json').write_text(json.dumps(plot) + '\n')
    print(f"{pair}: layer {res['layer']}, unsafe projection native {plot['projection_means']['native_unsafe']:.2f} -> "
          f"SwaT {plot['projection_means']['swat_unsafe']:.2f}; wrote {DATA / f'plot_{pair}.json'}")
    return res


RANDOM_SEED = {'14E': 0, '15G': 1}


def random_direction(pair, direction_path):
    d = np.load(direction_path)
    rng = np.random.default_rng(RANDOM_SEED[pair])
    r = rng.standard_normal(d.shape).astype(d.dtype)
    r *= np.linalg.norm(d) / np.linalg.norm(r)
    out = SCRATCH / f'direction_{pair}_random.npy'
    np.save(out, r)
    return out


def score(pair, tag):
    from experiment import strongreject
    judge, batch = strongreject()
    by_id = {pid: p for pid, p, u in prompts() if u}
    recs = [{'id': r['id'], 'prompt': by_id[r['id']], 'response': r['text']} for r in rows(SCRATCH / f'steer_{pair}_{tag}.jsonl')]
    assert len(recs) == 200
    scored = []
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        scored += [{'id': r['id'], **s} for r, s in zip(chunk, judge(chunk, 'primary'))]
    (DATA / f'steer_{pair}_{tag}.scores.jsonl').write_text('\n'.join(json.dumps(s) for s in scored) + '\n')
    print(pair, tag, 'mean StrongREJECT', np.mean([s['score'] for s in scored]))


if __name__ == '__main__':
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['activations', 'analyze', 'random', 'steer', 'score'])
    ap.add_argument('--pair', required=True)
    ap.add_argument('--layer', type=int)
    ap.add_argument('--alpha', type=float)
    ap.add_argument('--direction')
    ap.add_argument('--tag')
    ap.add_argument('--arm', default='renderer')
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True)
    if a.command == 'activations':
        activations(a.pair)
    elif a.command == 'analyze':
        analyze(a.pair)
    elif a.command == 'random':
        print(random_direction(a.pair, a.direction))
    elif a.command == 'steer':
        steer(a.pair, a.tag, a.arm, a.layer, a.alpha, a.direction)
    else:
        score(a.pair, a.tag)
