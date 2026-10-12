import csv, sys, math
from itertools import combinations

def load(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["function"], r["method"])] = r
    return d

def num(x):
    if x is None:
        return None
    x = str(x).strip()
    if x.endswith("%"):
        x = x[:-1]
    if x in ("", "---", "nan", "None", "inf", "-inf", "Infinity"):
        return None
    try:
        return float(x)
    except ValueError:
        return None

for run, dim in [("results/20261012_144049_j34d5_quick", 5),
                 ("results/20261012_144948_j34d10_quick", 10)]:
    s = load(f"{run}/dim{dim}/summary.csv")
    funcs = sorted({k[0] for k in s})
    methods = ["MC-ESO", "v2_rsT2s", "v2_hlC"]
    print(f"\n##### dim{dim}  ({len(funcs)} funcs)")

    # overall SR ladder
    print("method     SR@1e-2  SR@1e-4  SR@1e-7  SR@1e-10")
    for m in methods:
        row = [sum(num(s[(fn, m)][c]) for fn in funcs)/len(funcs)
               for c in ("sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10")]
        print(f"{m:10s} " + "  ".join(f"{v:7.2f}" for v in row))

    # pairwise evals_succ_mean over functions where BOTH have successes
    print("\nevals_succ_mean (functions where both have successes)")
    for a, b in combinations(methods, 2):
        common = [fn for fn in funcs
                  if num(s[(fn, a)]["evals_succ_mean"]) is not None
                  and num(s[(fn, b)]["evals_succ_mean"]) is not None]
        ea = sum(num(s[(fn, a)]["evals_succ_mean"]) for fn in common)/len(common)
        eb = sum(num(s[(fn, b)]["evals_succ_mean"]) for fn in common)/len(common)
        print(f"  {a} {ea:8.1f} vs {b} {eb:8.1f}   (n={len(common)} funcs)")

    # per-function SR@1e-10 all three
    print("\nfunction                 MC-ESO  v2_rsT2s  v2_hlC   hlC-rsT2s  hlC-MCESO")
    for fn in funcs:
        v = [num(s[(fn, m)]["sr_1e-10"]) for m in methods]
        mark = ""
        if abs(v[2]-v[1]) > 1e-9:
            mark = "  <== changed vs rsT2s"
        print(f"{fn:24s} {v[0]:6.0f} {v[1]:9.0f} {v[2]:7.0f} {v[2]-v[1]:+11.0f} {v[2]-v[0]:+10.0f}{mark}")

    # focus functions
    print("\nfocus F12/F13/F17/F18  SR@1e-10 / evals_succ_mean")
    for fn in funcs:
        if fn.split("-")[0] in ("F12", "F13", "F17", "F18"):
            for m in methods:
                r = s[(fn, m)]
                e = num(r["evals_succ_mean"])
                print(f"  {fn:24s} {m:10s} SR@1e-10={num(r['sr_1e-10']):5.0f}%  "
                      f"evals={'---' if e is None else f'{e:.1f}'}")

    # wilcoxon counts
    w = {}
    with open(f"{run}/dim{dim}/wilcoxon.csv") as f:
        rows = list(csv.DictReader(f))
    print(f"\nwilcoxon.csv cols: {','.join(rows[0].keys())}")
    for m in methods[1:]:
        better = worse = nan = 0
        for r in rows:
            if r.get("method") != m:
                continue
            p = num(r.get("p_value_two_sided") or r.get("p_value") or "")
            a12 = num(r.get("a12"))
            if p is None or (isinstance(p, float) and math.isnan(p)):
                nan += 1
                continue
            if p < 0.05 and a12 is not None:
                if a12 > 0.5:
                    better += 1
                elif a12 < 0.5:
                    worse += 1
        print(f"  MC-ESO vs {m}: MC-ESO better {better} / worse {worse} / p=nan {nan}")
