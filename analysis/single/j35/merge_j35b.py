"""Merge the j35b function shards (CEC2022 10D, official budget 200000 evals)
into one run directory so scripts/analyze_quick.py can aggregate them in one
pass.

Every run's seed is fixed by its run index alone (core/runner.py), so running
the 12 functions as separate quick invocations gives the same numbers as a
single invocation; concatenating the shards' summary.csv / wilcoxon.csv (and
the per-function stats/ files that [5] needs) reproduces the single-run output
up to row order.

Usage: python3 analysis/single/j35/merge_j35b.py <shard> [<shard> ...] <out>
"""
import csv, shutil, sys
from pathlib import Path

DIM = 'dim10'
SHARDS = [Path(p) for p in sys.argv[1:-1]]
OUT = Path(sys.argv[-1])
(OUT / DIM / 'stats').mkdir(parents=True, exist_ok=True)

for name in ('summary.csv', 'wilcoxon.csv'):
    hdr, rows, seen = None, [], set()
    for sh in SHARDS:
        with open(sh / DIM / name) as fh:
            rd = csv.DictReader(fh)
            if hdr is None:
                hdr = rd.fieldnames
            assert rd.fieldnames == hdr, f'{sh}/{name}: header mismatch'
            for r in rd:
                k = (r['function'], r['method'])
                assert k not in seen, f'duplicate {k} in {name}'
                seen.add(k)
                rows.append(r)
    with open(OUT / DIM / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    funcs = sorted({k[0] for k in seen})
    meths = sorted({k[1] for k in seen})
    print(f'{name}: {len(rows)} rows = {len(funcs)} functions x {len(meths)} methods'
          f'  ({"complete" if len(rows) == len(funcs)*len(meths) else "INCOMPLETE"})')
    print(f'  functions: {", ".join(f.split("-")[0] for f in funcs)}')

for sh in SHARDS:
    for p in (sh / DIM / 'stats').glob('*.csv'):
        dst = OUT / DIM / 'stats' / p.name
        assert not dst.exists(), f'duplicate stats file {p.name}'
        shutil.copy2(p, dst)
print(f'stats files: {len(list((OUT / DIM / "stats").glob("*.csv")))}')
