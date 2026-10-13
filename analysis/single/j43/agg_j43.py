"""j43a+j43b (20D, official budget 200000 evals = 10000*D, F01-F16) aggregation.

Standard aggregation for the measurement queue: overall SR ladder, paired
evals_succ_mean, per-function SR@1e-10, Wilcoxon counts with A12. Also diffs
MC-ESO against its own 2500*D (50000-eval) canonical reference measured on
2026-10-13 with the same default and the same container spec
(analysis/single/j42/summary_d20_e50k.csv).

Only 16 of the 24 functions are present: shard j43c (F17-F24) did not fit in
this invocation's wall-time budget.
"""
import csv, collections, math, sys

RUN = sys.argv[1] if len(sys.argv) > 1 else None
REF = 'MC-ESO'
J42 = 'analysis/single/j42/summary_d20_e50k.csv'

rows = list(csv.DictReader(open(f'{RUN}/dim20/summary.csv')))
wil = list(csv.DictReader(open(f'{RUN}/dim20/wilcoxon.csv')))
funcs = sorted({r['function'] for r in rows})
meths = sorted({r['method'] for r in rows})
S = {(r['function'], r['method']): r for r in rows}

def f(v):
    """summary.csv writes SR as '100%' and a missing evals_succ_mean as '---'."""
    if v is None: return None
    v = v.strip().rstrip('%')
    try: x = float(v)
    except Exception: return None
    return x if math.isfinite(x) else None

print(f'[0] functions {len(funcs)}  methods {len(meths)}  rows {len(rows)}  '
      f'(expected {len(funcs)*len(meths)})')

print('\n[1] OVERALL (mean over the 16 functions, %)')
print(f'  {"method":<16} {"1e-2":>7} {"1e-4":>7} {"1e-7":>7} {"1e-10":>7} {"evals":>9} {"nf":>3}')
order = []
for m in meths:
    sr = {lv: sum(f(S[(fn, m)][f'sr_{lv}']) for fn in funcs)/len(funcs)
          for lv in ('1e-2', '1e-4', '1e-7', '1e-10')}
    ev = [f(S[(fn, m)]['evals_succ_mean']) for fn in funcs]
    ev = [e for e in ev if e is not None]
    order.append((sr['1e-10'], m, sr, sum(ev)/len(ev) if ev else None, len(ev)))
for sr10, m, sr, ev, nf in sorted(order, reverse=True):
    print(f'  {m:<16} {sr["1e-2"]:7.2f} {sr["1e-4"]:7.2f} {sr["1e-7"]:7.2f} {sr10:7.2f} '
          f'{ev if ev else float("nan"):9.1f} {nf:3d}')

print('\n[1b] evals_succ_mean, paired (only functions where BOTH have successes)')
for m in meths:
    if m == REF: continue
    both = [fn for fn in funcs
            if f(S[(fn, REF)]['evals_succ_mean']) is not None
            and f(S[(fn, m)]['evals_succ_mean']) is not None]
    a = sum(f(S[(fn, REF)]['evals_succ_mean']) for fn in both)/len(both)
    b = sum(f(S[(fn, m)]['evals_succ_mean']) for fn in both)/len(both)
    print(f'  vs {m:<16} {REF} {a:9.1f}  /  {m} {b:9.1f}   (nf={len(both)})'
          + ('   <-- baseline faster' if b < a else ''))

print('\n[2] PER-FUNCTION SR@1e-10 (%)')
hdr = ' '.join(f'{m[:13]:>14}' for m in meths)
print(f'  {"function":<22}{hdr}')
for fn in funcs:
    cells = ' '.join(f'{f(S[(fn, m)]["sr_1e-10"]):14.0f}' for m in meths)
    print(f'  {fn:<22}{cells}')

print('\n[3] BUDGET DIFF for MC-ESO: j42 @50000 (2500*D) -> j43 @200000 (10000*D)')
j42 = {r['function']: r for r in csv.DictReader(open(J42)) if r['method'] == REF}
tot_old = tot_new = 0.0
moved, missing = [], []
for fn in funcs:
    if fn not in j42:
        missing.append(fn); continue
    o, n = f(j42[fn]['sr_1e-10']), f(S[(fn, REF)]['sr_1e-10'])
    tot_old += o; tot_new += n
    if abs(n - o) > 1e-9: moved.append((n - o, fn, o, n))
for d, fn, o, n in sorted(moved, key=lambda x: -x[0]):
    print(f'  {fn:<22} {o:6.0f} -> {n:6.0f}   ({d:+.0f})')
k = len(funcs) - len(missing)
print(f'  {k}-function mean: {tot_old/k:.2f} -> {tot_new/k:.2f} '
      f'({tot_new/k - tot_old/k:+.2f} pt);  functions moved: {len(moved)}'
      + (f'; missing in j42: {missing}' if missing else ''))

print('\n[3b] BUDGET DIFF for the baselines (same two runs), 16-function mean SR@1e-10')
for m in meths:
    if m == REF: continue
    b42 = {r['function']: r for r in csv.DictReader(open(J42)) if r['method'] == m}
    both = [fn for fn in funcs if fn in b42]
    if not both: print(f'  {m:<16} not in j42'); continue
    o = sum(f(b42[fn]['sr_1e-10']) for fn in both)/len(both)
    n = sum(f(S[(fn, m)]['sr_1e-10']) for fn in both)/len(both)
    print(f'  {m:<16} {o:6.2f} -> {n:6.2f}  ({n-o:+.2f} pt, nf={len(both)})')

print('\n[4] WILCOXON (ref=MC-ESO, p_two<0.05, direction by A12)')
cnt = collections.defaultdict(lambda: [0, 0])
sig = collections.defaultdict(list)
for r in wil:
    p, a12 = f(r['p_value_two_sided']), f(r['a12'])
    if p is None or a12 is None or p >= 0.05 or a12 == 0.5: continue
    m = r['method']
    better = a12 > 0.5
    cnt[m][0 if better else 1] += 1
    sig[m].append((r['function'], 'better' if better else 'WORSE', a12,
                   r['a12_magnitude'], p))
for m in sorted(cnt, key=lambda k: -cnt[k][0]):
    b, w = cnt[m]
    print(f'  {m:<16} better {b:2d} / worse {w:2d}')
    for fn, d, a12, mag, p in sorted(sig[m], key=lambda x: x[0]):
        print(f'      {fn:<22} {d:<6} A12={a12:.2f} ({mag})  p={p:.3g}')
