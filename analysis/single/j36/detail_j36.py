#!/usr/bin/env python3
"""j36: per-function SR@1e-10 changes and paired evals_succ_mean for v2_scan2d."""
import csv, sys, math
from pathlib import Path

def pct(s):
    s=(s or "").strip().rstrip("%")
    try: return float(s)
    except ValueError: return float("nan")

def num(s):
    s=(s or "").strip()
    if s in ("","nan","---","inf"): return float("nan")
    try: return float(s)
    except ValueError: return float("nan")

METHODS=["MC-ESO","v2_hlC","v2_scan2d"]
for dim, run in [(2,sys.argv[1]),(5,sys.argv[2]),(10,sys.argv[3])]:
    rows=list(csv.DictReader(open(f"{run}/dim{dim}/summary.csv")))
    T={}
    for r in rows: T.setdefault(r["function"],{})[r["method"]]=r
    funcs=sorted(T)
    print(f"\n{'='*78}\n=== dim{dim}  ({run})\n{'='*78}")
    # overall
    print("overall SR (%, mean over 24 funcs): method  1e-2 / 1e-4 / 1e-7 / 1e-10")
    for m in METHODS:
        vals=[ [pct(T[f][m][k]) for f in funcs] for k in ["sr_1e-2","sr_1e-4","sr_1e-7","sr_1e-10"] ]
        print("  %-10s %6.2f / %6.2f / %6.2f / %6.2f" % (m, *[sum(v)/len(v) for v in vals]))
    # paired evals_succ_mean
    print("paired evals_succ_mean (funcs where BOTH have successes):")
    for a,b in [("MC-ESO","v2_hlC"),("MC-ESO","v2_scan2d"),("v2_hlC","v2_scan2d")]:
        pf=[f for f in funcs if not math.isnan(num(T[f][a]["evals_succ_mean"])) and not math.isnan(num(T[f][b]["evals_succ_mean"]))]
        if pf:
            ea=sum(num(T[f][a]["evals_succ_mean"]) for f in pf)/len(pf)
            eb=sum(num(T[f][b]["evals_succ_mean"]) for f in pf)/len(pf)
            print("  %-10s %8.1f  vs  %-10s %8.1f   (nf=%d)" % (a,ea,b,eb,len(pf)))
    # own evals
    print("own evals_succ_mean (own successful funcs):")
    for m in METHODS:
        v=[num(T[f][m]["evals_succ_mean"]) for f in funcs]
        v=[x for x in v if not math.isnan(x)]
        print("  %-10s %8.1f  (nf=%d)" % (m, sum(v)/len(v), len(v)))
    # per-function changes
    for base,arm in [("MC-ESO","v2_scan2d"),("v2_hlC","v2_scan2d")]:
        ch=[]
        for f in funcs:
            a,b=pct(T[f][base]["sr_1e-10"]),pct(T[f][arm]["sr_1e-10"])
            if a!=b: ch.append((f,a,b,b-a))
        print(f"per-function SR@1e-10 changes {base} -> {arm}: {len(ch)} funcs")
        for f,a,b,d in sorted(ch,key=lambda x:x[3]):
            print("    %-24s %5.0f -> %5.0f  (%+.0f)" % (f,a,b,d))
        if not ch: print("    (none - all 24 identical)")
    # all-zero funcs
    z=[f for f in funcs if all(pct(T[f][m]["sr_1e-10"])==0 for m in METHODS)]
    print("SR@1e-10 = 0%% in all 3 methods: %s" % (", ".join(z) if z else "none"))
    # job-specified function table
    want=["F01-Sphere","F02-EllipsoidalSep","F03-RastriginSep","F04-BucheRastrigin","F05-LinearSlope","F20-Schwefel"]
    print("job-specified funcs (SR@1e-10 %% / evals_succ_mean) MC-ESO / v2_hlC / v2_scan2d:")
    for f in want:
        if f not in T: continue
        cells=[]
        for m in METHODS:
            e=num(T[f][m]["evals_succ_mean"])
            cells.append("%3.0f%% / %s" % (pct(T[f][m]["sr_1e-10"]), "---" if math.isnan(e) else "%.0f"%e))
        print("    %-22s %s" % (f, "   |   ".join(cells)))
    # wilcoxon
    wrows=list(csv.DictReader(open(f"{run}/dim{dim}/wilcoxon.csv")))
    print("Wilcoxon (ref=MC-ESO, p_two<0.05, direction by A12):")
    for m in ["v2_hlC","v2_scan2d"]:
        win=[];los=[];nan=0
        for r in wrows:
            if r.get("method")!=m: continue
            p=num(r.get("p_value_two_sided") or r.get("p_two") or "")
            a=num(r.get("a12",""))
            if math.isnan(p): nan+=1; continue
            if p<0.05:
                (win if a>0.5 else los).append((r["function"],a,r.get("a12_magnitude","")))
        print("  %-10s MC-ESO wins %d / loses %d  (p=nan %d)" % (m,len(win),len(los),nan))
        for lbl,lst in [("win",win),("loss",los)]:
            for f,a,mag in lst: print("      %-5s %-24s A12=%.2f %s" % (lbl,f,a,mag))
