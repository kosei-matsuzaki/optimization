"""j41a (10D, official budget 100000 evals, F01-F12) aggregation.

Standard aggregation for the measurement queue: overall SR ladder,
paired evals_succ_mean, per-function SR@1e-10, Wilcoxon counts with A12.
Also diffs the current default against its own 25000-eval (2500*D) canonical
reference (analysis/single/j36/summary_dim10.csv, method `v2_scan2d`, which is
the configuration that became the default on 2026-10-13).
"""
import csv, collections, math, sys

RUN = 'results/20261013_024004_j41a_quick/dim10'
REF = 'MC-ESO'
J36 = 'analysis/single/j36/summary_dim10.csv'

rows = list(csv.DictReader(open(f'{RUN}/summary.csv')))
wil = list(csv.DictReader(open(f'{RUN}/wilcoxon.csv')))
funcs = sorted({r['function'] for r in rows})
meths = sorted({r['method'] for r in rows})
S = {(r['function'], r['method']): r for r in rows}

def f(v):
    """summary.csv writes SR as '100%' (already a percentage) and a missing
    evals_succ_mean as '---'."""
    if v is None: return None
    v = v.strip().rstrip('%')
    try: x = float(v)
    except Exception: return None
    return x if math.isfinite(x) else None

print(f'[0] functions {len(funcs)}  methods {len(meths)}  rows {len(rows)}')

print('\n[1] OVERALL (mean over the 12 functions, %)')
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
hdr = ' '.join(f'{m[:13]:>13}' for m in meths)
print(f'  {"function":<22}{hdr}')
for fn in funcs:
    cells = ' '.join(f'{f(S[(fn, m)]["sr_1e-10"]):13.0f}' for m in meths)
    print(f'  {fn:<22}{cells}')

print('\n[3] BUDGET DIFF for the current default: j36 v2_scan2d @25000 -> j41a MC-ESO @100000')
j36 = {r['function']: r for r in csv.DictReader(open(J36)) if r['method'] == 'v2_scan2d'}
tot_old = tot_new = 0.0
moved = []
for fn in funcs:
    if fn not in j36:
        print(f'  {fn}: missing in j36'); continue
    o, n = f(j36[fn]['sr_1e-10']), f(S[(fn, REF)]['sr_1e-10'])
    tot_old += o; tot_new += n
    if abs(n - o) > 1e-9: moved.append((n - o, fn, o, n))
for d, fn, o, n in sorted(moved, key=lambda x: -x[0]):
    print(f'  {fn:<22} {o:6.0f} -> {n:6.0f}   ({d:+.0f})')
print(f'  12-function mean: {tot_old/len(funcs):.2f} -> {tot_new/len(funcs):.2f} '
      f'({tot_new/len(funcs)-tot_old/len(funcs):+.2f} pt);  functions moved: {len(moved)}')

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
