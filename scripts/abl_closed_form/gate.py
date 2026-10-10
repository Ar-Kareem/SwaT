from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-dir', type=Path, required=True)
    p.add_argument('--timeout-minutes', type=float, default=180)
    p.add_argument('--min-heldout-r2', type=float, default=0.0,
                   help='fail if any (layer, K|V) held-out mean R2 is below this (0 = worse than predicting the mean)')
    p.add_argument('--max-gap', type=float, default=1.0, help='fail if fit R2 - heldout R2 exceeds this (overfit)')
    args = p.parse_args()
    started = time.monotonic()
    while not (args.fit_dir / 'complete.json').exists():
        if time.monotonic() - started > args.timeout_minutes * 60:
            sys.exit(f'Timed out waiting for {args.fit_dir / "complete.json"}')
        time.sleep(30)
    report = json.loads((args.fit_dir / 'fit_report.json').read_text())
    failures = []
    print(f'{"layer":>5} {"src":>4} {"kind":>4} {"fit R2":>8} {"held R2":>8} {"gap":>7}')
    for row in report['regressions']:
        fit_r2, held_r2 = row['fit']['mean_r2'], row['heldout']['mean_r2']
        gap = fit_r2 - held_r2
        flag = ''
        if held_r2 < args.min_heldout_r2 or gap > args.max_gap:
            flag = '  <-- FAIL'
            failures.append(row)
        print(f'{row["target_layer"]:>5} {str(row["source_layers"]):>4} {row["kind"]:>4} {fit_r2:8.4f} {held_r2:8.4f} {gap:7.4f}{flag}')
    for kind in ['K', 'V']:
        rows = [r for r in report['regressions'] if r['kind'] == kind]
        print(f'{kind}: mean held-out R2 = {sum(r["heldout"]["mean_r2"] for r in rows) / len(rows):.4f}, '
              f'min = {min(r["heldout"]["mean_r2"] for r in rows):.4f}')
    if failures:
        sys.exit(f'GATE FAILED: {len(failures)} regressions below threshold')
    print('GATE PASSED', flush=True)


if __name__ == '__main__':
    main()
