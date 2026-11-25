"""Collect eval summaries into success-rate tables (with Wilson 95% intervals).

  python scripts/summarize.py                 # reads results/<method>/eval/<split>/
  python scripts/summarize.py --root runs     # reads your own runs/<method>/<split>/
"""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['act_k50', 'act_k10', 'pa_act']
SPLITS = ['Standard', 'P2', 'P1']


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def find_summary(root, method, split):
    for path in (root / method / 'eval' / split / 'summary.json', root / method / split / 'summary.json'):
        if path.exists():
            return json.loads(path.read_text())
    return None


def main(root, out):
    out.mkdir(parents=True, exist_ok=True)
    overall, per_task = [], []
    for split in SPLITS:
        for method in METHODS:
            s = find_summary(root, method, split)
            if s is None:
                continue
            lo, hi = wilson(s['successes'], s['episodes'])
            overall.append(dict(method=method, split=split, successes=s['successes'], episodes=s['episodes'],
                                success_rate_pct=round(s['success_rate_pct'], 1),
                                ci95_low=round(max(lo, 0.), 1), ci95_high=round(hi, 1)))
            for task, v in s['per_task'].items():
                per_task.append(dict(method=method, split=split, task=task, **v,
                                     success_rate_pct=round(100 * v['successes'] / v['episodes'], 1)))
    for name, rows in (('success_rates.csv', overall), ('task_success_rates.csv', per_task)):
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"{'method':10s}" + ''.join(f'{s:>12s}' for s in SPLITS))
    for method in METHODS:
        cells = []
        for split in SPLITS:
            r = next((r for r in overall if r['method'] == method and r['split'] == split), None)
            cells.append(f"{r['successes']}/{r['episodes']}" if r else '-')
        print(f'{method:10s}' + ''.join(f'{c:>12s}' for c in cells))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT / 'results', type=Path)
    ap.add_argument('--out', default=ROOT / 'results' / 'summary', type=Path)
    args = ap.parse_args()
    main(args.root, args.out)
