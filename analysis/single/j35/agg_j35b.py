"""Job 5 (j35b) aggregation: CEC2022 10D at the official budget 200000 evals.

Prints the parts of the standard aggregation that analyze_quick.py does not:
per-function SR@1e-7 / median error for MC-ESO vs the best baseline, the
evals_succ_mean comparison on the functions where both have successes, and the
same numbers from the 25000-eval run (analysis/single/j35/summary_e25k.csv) for
the same 7 methods.

Usage: python3 analysis/single/j35/agg_j35b.py <merged_run_dir>
"""
import csv, math, sys
from collections import defaultdict
from pathlib import Path

RUN = Path(sys.argv[1])
METH = ["MC-ESO", "EA4eig", "LSHADE-SPACMA", "L-SHADE", "L-SRTDE",
        "EBOwithCMAR", "jSO"]
E25K = Path('analysis/single/j35/summary_e25k.csv')


def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[r['function']][r['method']] = r
    return d


def ev(d, f, m):
    try:
        x = float(d[f][m]['evals_succ_mean'])
    except Exception:
        return None
    return None if math.isinf(x) or math.isnan(x) else x


def sr(d, f, m, lvl='sr_1e-10'):
    return float(d[f][m][lvl].rstrip('%'))


cur = load(RUN / 'dim10' / 'summary.csv')
funcs = sorted(cur)
print(f"### functions measured at 200000 evals: {len(funcs)} "
      f"({', '.join(f.split('-')[0] for f in funcs)})")

print("\n### per-function SR@1e-10 (%), 7 methods, CEC2022 10D n=20 max_evals=200000")
print("| func | " + " | ".join(METH) + " |")
for f in funcs:
    print(f"| {f} | " + " | ".join(f"{sr(cur, f, m):.0f}" for m in METH) + " |")

print("\n### evals_succ_mean, mean over functions where BOTH have successes (ref=MC-ESO)")
print("| method | common funcs | method evals | MC-ESO evals |")
for m in METH:
    if m == 'MC-ESO':
        continue
    com = [f for f in funcs if ev(cur, f, m) is not None and ev(cur, f, 'MC-ESO') is not None]
    if not com:
        print(f"| {m} | 0 | - | - |")
        continue
    print(f"| {m} | {len(com)} | {sum(ev(cur, f, m) for f in com)/len(com):.0f} "
          f"| {sum(ev(cur, f, 'MC-ESO') for f in com)/len(com):.0f} |")

print("\n### SR@1e-7 / median error: MC-ESO vs best baseline (of the 6)")
BASE = [m for m in METH if m != 'MC-ESO']
print("| func | MC-ESO | best baseline |")
for f in funcs:
    bb = sorted(BASE, key=lambda m: (-sr(cur, f, m, 'sr_1e-7'),
                                     float(cur[f][m]['median_best_f'])))[0]
    print(f"| {f} | {sr(cur, f, 'MC-ESO', 'sr_1e-7'):.0f}% / "
          f"{float(cur[f]['MC-ESO']['median_best_f']):.3e} | {bb} "
          f"{sr(cur, f, bb, 'sr_1e-7'):.0f}% / {float(cur[f][bb]['median_best_f']):.3e} |")

print("\n### same functions at 25000 evals (analysis/single/j35/summary_e25k.csv), SR@1e-10 (%)")
old = load(E25K)
print("| func | " + " | ".join(METH) + " |")
for f in funcs:
    if f not in old:
        print(f"| {f} | (not in e25k) |")
        continue
    print(f"| {f} | " + " | ".join(
        f"{sr(old, f, m):.0f}" if m in old[f] else "-" for m in METH) + " |")

print("\n### overall SR@1e-10 (%) over these functions: 200000 vs 25000 evals")
print("| method | 200000 | 25000 | diff |")
for m in METH:
    a = sum(sr(cur, f, m) for f in funcs) / len(funcs)
    ofs = [f for f in funcs if f in old and m in old[f]]
    b = sum(sr(old, f, m) for f in ofs) / len(ofs) if ofs else float('nan')
    print(f"| {m} | {a:.1f} | {b:.1f} | {a-b:+.1f} |")

print("\n### CEC mean-error rank proxy from mean_best_f (<1e-8 -> 0), same 7 methods")
print("(the authoritative rank over per-run errors is analyze_quick.py [5]; this")
print(" proxy lets the 25000-eval run, whose per-run stats are not in the repo,")
print(" be ranked over the same 7 methods)")
for label, d in (('200000', cur), ('25000', old)):
    fs = [f for f in funcs if f in d and all(m in d[f] for m in METH)]
    tot = defaultdict(float)
    for f in fs:
        vals = []
        for m in METH:
            x = float(d[f][m]['mean_best_f'])
            vals.append((0.0 if x < 1e-8 else x, m))
        vals.sort()
        # average ranks for ties
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
                j += 1
            rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                tot[vals[k][1]] += rank
            i = j + 1
    order = sorted(METH, key=lambda m: tot[m] / len(fs))
    print(f"  {label} evals ({len(fs)} functions): " +
          ", ".join(f"{m} {tot[m]/len(fs):.2f}" for m in order))
