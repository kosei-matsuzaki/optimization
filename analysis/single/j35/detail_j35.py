import csv, math
from collections import defaultdict
P='results/20261012_merged_j35_e25k_quick/dim10/summary.csv'
rows=list(csv.DictReader(open(P)))
d=defaultdict(dict)
for r in rows: d[r['function']][r['method']]=r
funcs=sorted(d)
METH=["MC-ESO","v2_hlC","MC-ESO-v1","CMA-ES","IPOP-CMA-ES","BIPOP-CMA-ES","DE","L-SHADE","jSO","LSHADE-SPACMA","ELSHADE-SPACMA","EBOwithCMAR","EA4eig","L-SRTDE","IMODE"]
def ev(f,m):
    try: x=float(d[f][m]['evals_succ_mean'])
    except Exception: return None
    return None if math.isinf(x) or math.isnan(x) else x
print("### per-function SR@1e-10 (%), 15 methods, CEC2022 10D n=20 max_evals=25000")
print("| func | " + " | ".join(METH) + " |")
for f in funcs:
    print(f"| {f} | " + " | ".join(d[f][m]['sr_1e-10'].rstrip('%') for m in METH) + " |")
print("\n### evals_succ_mean, mean over functions where BOTH have successes (ref=MC-ESO)")
print("| method | common funcs | method | MC-ESO |")
for m in METH:
    if m=="MC-ESO": continue
    com=[f for f in funcs if ev(f,m) is not None and ev(f,'MC-ESO') is not None]
    if not com: print(f"| {m} | 0 | - | - |"); continue
    print(f"| {m} | {len(com)} | {sum(ev(f,m) for f in com)/len(com):.0f} | {sum(ev(f,'MC-ESO') for f in com)/len(com):.0f} |")
print("\n### SR@1e-7 / median error: MC-ESO / v2_hlC / best baseline")
BASE=[m for m in METH if m not in ("MC-ESO","v2_hlC")]
for f in funcs:
    bb=sorted(BASE,key=lambda m:(-float(d[f][m]['sr_1e-7'].rstrip('%')),float(d[f][m]['median_best_f'])))[0]
    cells=[f"{m} {d[f][m]['sr_1e-7'].rstrip('%')}% / {float(d[f][m]['median_best_f']):.3e}" for m in ("MC-ESO","v2_hlC",bb)]
    print(f"| {f} | " + " | ".join(cells) + " |")
