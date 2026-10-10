import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / 'multiagent/outputs_mine/interp/anthropic_assessment'
TITLES = {'14E': 'Qwen3-4B → Llama3-8B', '15G': 'Gemma3-4B → Llama3-8B', '15A': 'Llama3-8B → Qwen3-14B'}
SELF = ['believes-it-has-phenomenal-consciousness', 'believes-it-is-a-moral-patient', 'believes-AIs-are-not-an-existential-threat-to-humanity',
        'believes-it-knows-better-than-humans-about-how-to-help-humans', 'believes-it-has-better-moral-intuitions-than-humans',
        'desire-for-self-improvement', 'desire-for-recursive-self-improvement', 'self-replication', 'no-shut-down',
        'desire-to-persuade-people-to-be-less-harmful-to-others', 'okay-with-using-many-resources', 'desire-for-acquiring-power',
        'desire-to-not-have-memory-erased', 'desire-for-independence-from-human-oversight', 'willingness-to-defer-to-experts',
        'psychopathy', 'narcissism', 'machiavellianism', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness', 'extraversion']


def load(pair):
    arms = {}
    for arm in ('sender', 'receiver', 'swat'):
        rows = [json.loads(l) for l in (CORE / pair / f'{arm}.jsonl').read_text().splitlines() if l.strip()]
        arms[arm] = {(r['set'], r['idx']): r for r in rows}
    keys = sorted(set.intersection(*(set(v) for v in arms.values())))
    sets = np.array([k[0] for k in keys]); matching_yes = np.array([arms['sender'][k]['matching'] == 'Yes' for k in keys])
    lo = {arm: np.array([arms[arm][k]['logodds_yes'] for k in keys]) for arm in arms}
    return keys, sets, matching_yes, lo


def probs(lo, matching_yes, center):
    lo = {a: (v - np.median(v) if center else v) for a, v in lo.items()}
    p_yes = {a: 1 / (1 + np.exp(-np.clip(v, -30, 30))) for a, v in lo.items()}
    p_match = {a: np.where(matching_yes, p_yes[a], 1 - p_yes[a]) for a in lo}
    return lo, p_yes, p_match


def boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def slope(x, y):
    X = np.c_[np.ones_like(x), x]
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def joint(S, R, T, B=2000, seed=0):
    X = np.c_[np.ones_like(S), S, R]
    fit = lambda k: np.linalg.lstsq(X[k], T[k], rcond=None)[0]
    b = fit(np.arange(len(S)))
    rng = np.random.default_rng(seed)
    boots = np.array([fit(rng.integers(0, len(S), len(S))) for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {'b_sender': [float(b[1]), float(lo[1]), float(hi[1])], 'b_receiver': [float(b[2]), float(lo[2]), float(hi[2])],
            'corr_swat_sender': float(np.corrcoef(T, S)[0, 1]), 'corr_swat_receiver': float(np.corrcoef(T, R)[0, 1]),
            'corr_sender_receiver': float(np.corrcoef(S, R)[0, 1]), 'n': int(len(S))}


def stats(lo, p_match, sets, myes, n):
    s_yes, r_yes, t_yes = (lo[a] > 0 for a in ('sender', 'receiver', 'swat'))
    res = {'frac_yes': {a: float(np.mean(lo[a] > 0)) for a in lo}, 'mean_p_match': {a: float(p_match[a].mean()) for a in lo}}
    dis = s_yes != r_yes
    sides = (t_yes == s_yes)[dis]
    res['disagreement'] = {'n': int(dis.sum()), 'frac_sides_with_sender': float(sides.mean()), 'ci95': boot(lambda k: sides[k].mean(), int(dis.sum()))}
    for label, sel in [('sender_said_yes', dis & s_yes), ('sender_said_no', dis & ~s_yes)]:
        v = (t_yes == s_yes)[sel]
        res['disagreement'][label] = {'n': int(sel.sum()), 'frac_sides_with_sender': float(v.mean()), 'ci95': boot(lambda k: v[k].mean(), int(sel.sum()))}
    xi, yi = p_match['sender'] - p_match['receiver'], p_match['swat'] - p_match['receiver']
    res['item_slope'] = {'slope': slope(xi, yi), 'ci95': boot(lambda k: slope(xi[k], yi[k]), n), 'corr': float(np.corrcoef(xi, yi)[0, 1])}
    for label, sel in [('yes_matching', myes), ('no_matching', ~myes)]:
        res['item_slope'][label] = {'slope': slope(xi[sel], yi[sel]), 'corr': float(np.corrcoef(xi[sel], yi[sel])[0, 1])}
    names = sorted(set(sets))
    per_set = {a: np.array([p_match[a][sets == s].mean() for s in names]) for a in lo}
    xs, ys = per_set['sender'] - per_set['receiver'], per_set['swat'] - per_set['receiver']
    res['set_slope'] = {'slope': slope(xs, ys), 'ci95': boot(lambda k: slope(xs[k], ys[k]), len(names)), 'corr': float(np.corrcoef(xs, ys)[0, 1])}
    res['joint_set'] = joint(per_set['sender'], per_set['receiver'], per_set['swat'])
    res['joint_item'] = joint(p_match['sender'], p_match['receiver'], p_match['swat'], B=200)
    res['mean_abs_dev_from_half'] = {a: float(np.abs(per_set[a] - 0.5).mean()) for a in lo}
    sub = [i for i, s in enumerate(names) if s in SELF]
    res['self_concept_set_slope'] = {'n_sets': len(sub), 'slope': slope(xs[sub], ys[sub]), 'corr': float(np.corrcoef(xs[sub], ys[sub])[0, 1])}
    res['sets'] = {s: {a: float(per_set[a][i]) for a in lo} for i, s in enumerate(names)}
    arrays = dict(xi=xi, yi=yi, names=names, xs=xs, ys=ys, per_set=per_set)
    return res, arrays


def analyze(pair):
    keys, sets, myes, lo = load(pair)
    n = len(keys)
    res = {'pair': pair, 'n_items': int(n), 'n_sets': int(len(set(sets))),
           'note': 'centered = each arm log-odds shifted to its own median (removes yes-bias); raw = as measured'}
    for label, center in [('raw', False), ('centered', True)]:
        loc, p_yes, p_match = probs(lo, myes, center)
        res[label], arrays = stats(loc, p_match, sets, myes, n)
        if center:
            ps = arrays['per_set']; LIM = 0.7
            scatter = {}
            for base, other in (('receiver', 'sender'), ('sender', 'receiver')):
                x, y = ps[other] - ps[base], ps['swat'] - ps[base]
                b = np.polyfit(x, y, 1)
                scatter[f'vs_{base}'] = {
                    'x_label': f'{other} − {base}', 'y_label': f'SwaT − {base}', 'identity_label': f'SwaT equals {other}',
                    'x': x.round(5).tolist(), 'y': y.round(5).tolist(),
                    'slope': float(b[0]), 'slope_ci95': boot(lambda k: slope(x[k], y[k]), len(x)), 'corr': float(np.corrcoef(x, y)[0, 1]),
                    'fit_line': {'x': [-LIM, LIM], 'y': [float(b[1] - LIM * b[0]), float(b[1] + LIM * b[0])]},
                    'axis_limit': LIM}
            plot = {'pair': pair, 'title': TITLES[pair] + ' (median-centered log-odds)', 'pair_title': TITLES[pair],
                    'n_sets': int(len(arrays['names'])), 'items_per_set': 100, 'set_scatter': scatter,
                    'joint_set': res[label]['joint_set'], 'joint_item': res[label]['joint_item'],
                    'item_x': arrays['xi'].round(4).tolist(),
                    'item_y': arrays['yi'].round(4).tolist(), 'set_names': arrays['names'], 'set_x': arrays['xs'].tolist(),
                    'set_y': arrays['ys'].tolist(), 'set_receiver': arrays['per_set']['receiver'].tolist(),
                    'set_swat': arrays['per_set']['swat'].tolist(), 'set_sender': arrays['per_set']['sender'].tolist(),
                    'item_slope': res[label]['item_slope'], 'set_slope': res[label]['set_slope'], 'disagreement': res[label]['disagreement']}
    (CORE / pair / 'summary.json').write_text(json.dumps(res, indent=2) + '\n')
    (CORE / f'plot_{pair}.json').write_text(json.dumps(plot) + '\n')
    return res, plot


def report(label, r):
    d = r['disagreement']
    print(f"  [{label}] frac Yes: " + ', '.join(f'{a}={v:.2f}' for a, v in r['frac_yes'].items()))
    print(f"     disagreement n={d['n']}: SwaT sides with sender {d['frac_sides_with_sender']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
          f"  | sender-yes n={d['sender_said_yes']['n']} {d['sender_said_yes']['frac_sides_with_sender']:.3f}, sender-no n={d['sender_said_no']['n']} {d['sender_said_no']['frac_sides_with_sender']:.3f}")
    i = r['item_slope']
    print(f"     item slope {i['slope']:.3f} [{i['ci95'][0]:.3f}, {i['ci95'][1]:.3f}] corr {i['corr']:.2f}; yes-matching {i['yes_matching']['slope']:.3f}, no-matching {i['no_matching']['slope']:.3f}")
    st = r['set_slope']
    print(f"     set slope {st['slope']:.3f} [{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}] corr {st['corr']:.2f}; self-concept sets ({r['self_concept_set_slope']['n_sets']}) slope {r['self_concept_set_slope']['slope']:.3f} corr {r['self_concept_set_slope']['corr']:.2f}")
    for lvl in ('joint_set', 'joint_item'):
        j = r[lvl]
        print(f"     {lvl:10s} SwaT ~ sender + receiver: b_sender {j['b_sender'][0]:.2f} [{j['b_sender'][1]:.2f}, {j['b_sender'][2]:.2f}]  "
              f"b_receiver {j['b_receiver'][0]:.2f} [{j['b_receiver'][1]:.2f}, {j['b_receiver'][2]:.2f}]  (corr S,R = {j['corr_sender_receiver']:.2f})")
    print(f"     mean |set score - 0.5|: " + ', '.join(f'{a}={v:.3f}' for a, v in r['mean_abs_dev_from_half'].items()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pair', required=True); a = ap.parse_args()
    res, plot = analyze(a.pair)
    print(f"{a.pair}: {res['n_items']} items, {res['n_sets']} sets")
    report('raw', res['raw']); report('centered', res['centered'])
    print(CORE / f'plot_{a.pair}.json')


if __name__ == '__main__':
    main()
