"""j40 (5D, CEC official budget 10000*D = 50000 evals, instance 1) aggregation.

Lines the 6 methods of this run up against the same methods in the 12500-eval
reference run named by the job (runs/20261012_020743_allmethods2_d5_quick).
Reporting only -- no adoption judgement.
"""
import csv, gzip, io, statistics as st

NEW = 'results/20261013_003317_j40d5_quick/dim5'
REF = 'runs/20261012_020743_allmethods2_d5_quick/dim5'
METHODS = ['MC-ESO', 'ELSHADE-SPACMA', 'LSHADE-SPACMA', 'EBOwithCMAR', 'IPOP-CMA-ES', 'L-SHADE']


def load(path, gz=False):
    op = (lambda p: io.TextIOWrapper(gzip.open(p, 'rb'))) if gz else open
    with op(path) as f:
        return list(csv.DictReader(f))


def num(v):
    """summary.csv stores SR as '95%' and a missing evals mean as '' or '---'."""
    if v is None:
        return None
    v = v.strip().rstrip('%')
    if v in ('', '---', 'nan', 'inf', 'Infinity'):
        return None
    try:
        return float(v)
    except ValueError:
        return None


new = load(f'{NEW}/summary.csv')
ref = load(f'{REF}/summary.csv.gz', gz=True)
wil = load(f'{NEW}/wilcoxon.csv')

N = {(r['function'], r['method']): r for r in new}
R = {(r['function'], r['method']): r for r in ref}
FUNCS = sorted({r['function'] for r in new})
out = []
P = out.append

P('=' * 78)
P('j40 / j40d5  BBOB-24 dim=5  n_runs=20  max_evals=50000 (CEC official 10000*D)  instance 1')
P(f'run: 20261013_003317_j40d5_quick   functions: {len(FUNCS)}   methods: {len(METHODS)}')
P('reference (12500 evals): 20261012_020743_allmethods2_d5_quick')
P('=' * 78)

P('')
P('[A] SR@1e-10 RANK, 6 methods, 50000 evals vs the 12500-eval reference run')
P('    (the reference run was measured on Windows local -- NOT the canonical env --')
P('     and its "MC-ESO" is the pre-2026-10-13 default (v1_w4q3E), not this run\'s default.)')
P('')
P('  method           SR@1e-10  SR@1e-7  SR@1e-4  SR@1e-2   evals_own   nf   | ref SR@1e-10  delta')
P('  ' + '-' * 96)
rows = []
for m in METHODS:
    srs = {lv: st.mean(num(N[(f, m)][f'sr_{lv}']) for f in FUNCS)
           for lv in ('1e-10', '1e-7', '1e-4', '1e-2')}
    ev = [num(N[(f, m)]['evals_succ_mean']) for f in FUNCS]
    ev = [e for e in ev if e is not None]
    refsr = st.mean(num(R[(f, m)]['sr_1e-10']) for f in FUNCS) if (FUNCS[0], m) in R else None
    rows.append((m, srs, st.mean(ev), len(ev), refsr))
for m, srs, ev, nf, refsr in sorted(rows, key=lambda t: -t[1]['1e-10']):
    rs = f'{refsr:8.2f}    {srs["1e-10"] - refsr:+6.2f}' if refsr is not None else '     ---       ---'
    P(f'  {m:<16} {srs["1e-10"]:7.2f}  {srs["1e-7"]:7.2f}  {srs["1e-4"]:7.2f}  {srs["1e-2"]:7.2f}  '
      f'{ev:9.1f}  {nf:3d}   | {rs}')

P('')
P('[A2] CROSS-REFERENCE: canonical-environment 12500-eval values for the SAME config')
P('   The job-named reference run above is Windows-local and its "MC-ESO" row is the')
P('   pre-2026-10-13 default, so its delta mixes budget + default change + environment.')
P('   analysis/single/j36/summary_dim5.csv (canonical env, 12500 evals, instance 1) holds')
P('   the current default under its arm name v2_scan2d:')
P('     v2_scan2d (= this run\'s MC-ESO)  SR@1e-10 78.33  evals_own 3926.5  nf 22   @12500 evals')
P('     MC-ESO (pre-10-13 default)       SR@1e-10 68.12  evals_own 4143.7  nf 22   @12500 evals')
P('     v2_hlC                           SR@1e-10 73.54  evals_own 4047.8  nf 21   @12500 evals')
P('   => same config, same environment, budget 12500 -> 50000: 78.33 -> 81.67 (+3.34pt),')
P('      evals_own 3926.5 -> 8619.3.')

P('')
P('[B] evals_succ_mean, paired: averaged ONLY over functions where BOTH have successes')
P('  baseline          nf   MC-ESO   baseline   (own-set means in [A])')
P('  ' + '-' * 60)
for m in METHODS[1:]:
    pair = [(num(N[(f, 'MC-ESO')]['evals_succ_mean']), num(N[(f, m)]['evals_succ_mean']))
            for f in FUNCS]
    pair = [(a, b) for a, b in pair if a is not None and b is not None]
    P(f'  {m:<16} {len(pair):4d} {st.mean(a for a, _ in pair):9.1f} {st.mean(b for _, b in pair):10.1f}')

P('')
P('[C] PER-FUNCTION BEST METHOD on SR@1e-10 (ties listed; tie-break shown by evals)')
P('  function                 best SR@1e-10  method(s)')
P('  ' + '-' * 70)
for f in FUNCS:
    vals = {m: num(N[(f, m)]['sr_1e-10']) for m in METHODS}
    top = max(vals.values())
    best = [m for m in METHODS if vals[m] == top]
    P(f'  {f:<24} {top:7.1f}%     ' + ', '.join(best))

P('')
P('[D] PER-FUNCTION SR@1e-10: this run (50000 evals) vs reference (12500 evals), all 24 functions')
P('  function                 ' + '  '.join(f'{m[:13]:>13}' for m in METHODS))
P('  ' + '-' * 108)
for f in FUNCS:
    cells = []
    for m in METHODS:
        a = num(N[(f, m)]['sr_1e-10'])
        b = num(R[(f, m)]['sr_1e-10']) if (f, m) in R else None
        cells.append(f'{b:3.0f}>{a:3.0f}' if b is not None else f'  ->{a:3.0f}')
    P(f'  {f:<24} ' + '  '.join(f'{c:>13}' for c in cells))
P('  (format: 12500-eval value > 50000-eval value, in SR@1e-10 %)')

P('')
P('[E] functions whose SR@1e-10 CHANGED from the 12500-eval reference, per method')
for m in METHODS:
    ch = []
    for f in FUNCS:
        if (f, m) not in R:
            continue
        a = num(N[(f, m)]['sr_1e-10'])
        b = num(R[(f, m)]['sr_1e-10'])
        if a != b:
            ch.append(f'{f} {b:.0f}->{a:.0f} ({a - b:+.0f})')
    tag = ' (ref row = pre-10-13 default)' if m == 'MC-ESO' else ''
    P(f'  {m}{tag}: {len(ch)} changed')
    for c in ch:
        P(f'      {c}')

P('')
P('[E2] PER-FUNCTION SR@1e-10, canonical same-config 12500 -> 50000 evals')
P('     (analysis/single/j36/summary_dim5.csv arm v2_scan2d  ->  this run\'s MC-ESO)')
_r = {(r['function'], r['method']): r
      for r in load('analysis/single/j36/summary_dim5.csv')}
_ch = []
for f in FUNCS:
    a = num(_r[(f, 'v2_scan2d')]['sr_1e-10'])
    b = num(N[(f, 'MC-ESO')]['sr_1e-10'])
    if a != b:
        _ch.append(f'{f} {a:.0f}->{b:.0f} ({b - a:+.0f})')
P(f'  {len(_ch)} of 24 functions changed')
for c in _ch:
    P(f'      {c}')

P('')
P('[F] WILCOXON (ref=MC-ESO, p_two<0.05, direction by A12)')
P('  baseline          better(MC-ESO wins) / worse(baseline wins)')
P('  ' + '-' * 60)
W = {(r['function'], r['method']): r for r in wil}
for m in METHODS[1:]:
    bet, wor = [], []
    for f in FUNCS:
        r = W.get((f, m))
        if not r:
            continue
        p = num(r.get('p_value_two_sided'))
        a12 = num(r.get('a12'))
        if p is None or a12 is None or p >= 0.05:
            continue
        (bet if a12 > 0.5 else wor).append(f'{f} A12={a12:.2f}')
    P(f'  {m:<16} {len(bet)} / {len(wor)}')
    P(f'      better: ' + (', '.join(bet) if bet else '-'))
    P(f'      worse : ' + (', '.join(wor) if wor else '-'))

txt = '\n'.join(out)
open('analysis/single/j40/detail_d5_e50k.txt', 'w').write(txt + '\n')
print(txt)
