from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-dir', type=Path, required=True)
    p.add_argument('--ks', type=int, nargs='+', required=True, help='every k that must be present')
    args = p.parse_args()
    table = []
    for k in args.ks:
        path = args.pair_dir / f'k{k}_fineweb' / 'fit_report.json'
        if not path.exists():
            raise FileNotFoundError(path)
        rows = json.loads(path.read_text())['regressions']
        mean = lambda kind: sum(r['heldout']['mean_r2'] for r in rows if r['kind'] == kind) / sum(r['kind'] == kind for r in rows)
        table.append({'k': k, 'heldout_r2_K': mean('K'), 'heldout_r2_V': mean('V')})
        table[-1]['heldout_r2_mean'] = (table[-1]['heldout_r2_K'] + table[-1]['heldout_r2_V']) / 2
    best = max(table, key=lambda r: (round(r['heldout_r2_mean'], 6), r['k']))
    for r in table:
        print(f"k={r['k']:>2}  K {r['heldout_r2_K']:.4f}  V {r['heldout_r2_V']:.4f}  mean {r['heldout_r2_mean']:.4f}" + ('  <-- selected' if r is best else ''))
    (args.pair_dir / 'k_selection.json').write_text(json.dumps({
        'criterion': 'argmax over k of mean(held-out K R2, held-out V R2) averaged over layers; ties (1e-6) toward larger k',
        'selected_k': best['k'], 'table': table}, indent=2) + '\n')
    link = args.pair_dir / 'final'
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(f"k{best['k']}_fineweb", link)
    print(f"final -> k{best['k']}_fineweb", flush=True)


if __name__ == '__main__':
    main()
