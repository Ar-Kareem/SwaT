import numpy as np


CHANGE_KIND = {'score': 'points', 'tokens': 'relative', 's_gpu': 'speedup', 'r_prefill': 'speedup'}


def average(metric):
    return (lambda values: float(np.mean(list(values)))) if metric == 'score' else geo


def geo(values):
    values = np.asarray(list(values), dtype=float)
    if (values <= 0).any():
        raise ValueError('geometric mean of a non-positive value')
    return float(np.exp(np.log(values).mean()))


def score(row):
    g = row.get('grade_result')
    f1 = g.get('f1', g.get('answer_f1')) if isinstance(g, dict) else None
    return float(f1) if f1 is not None else (1.0 if row['grade'] == 'correct' else 0.0)


def swat_tokens(row):
    return float(row['compression']['tokens'])


def resolve(blocks, level):
    block = blocks[level]
    return blocks[block['alias_of']] if 'alias_of' in block else block


def change_value(t2t, swat, metric, kind=None):
    kind = kind or CHANGE_KIND[metric]
    return {'points': lambda: swat - t2t, 'relative': lambda: 100 * (swat / t2t - 1), 'speedup': lambda: t2t / swat}[kind]()
