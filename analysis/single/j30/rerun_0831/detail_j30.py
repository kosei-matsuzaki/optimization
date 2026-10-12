#!/usr/bin/env python3
"""j30 detail: 7-method comparison table per dimension (sections A-G)."""
import argparse, csv, math
from pathlib import Path

SR = ["sr_1e-2", "sr_1e-4", "sr_1e-7", "sr_1e-10"]
P = "sr_1e-10"

def pct(s):
    s = (s or "").strip().rstrip("%")
    try: return float(s)
    except ValueError: return float("nan")

def num(s):
    s = (s or "").strip()
    if not s or s in ("N/A", "-"): return float("nan")
    try:
        v = float(s)
    except ValueError:
        return float("nan")
    return v if math.isfinite(v) else float("nan")

def load_summary(p):
    rows = list(csv.DictReader(open(p, newline="")))
    d = {}
    for r in rows:
        d[(r["function"], r["method"])] = r
    return d, rows

def load_wil(p):
    if not Path(p).exists(): return []
    return list(csv.DictReader(open(p, newline="")))

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--dim", type=int, required=True)
ap.add_argument("--ref", default="MC-ESO")
ap.add_argument("--order", default="MC-ESO,MC-ESO-v1,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,DE,L-SHADE")
ap.add_argument("--baselines", default="CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,DE,L-SHADE")
ap.add_argument("--prev", default=None, help="prior run summary csv for consistency check")
ap.add_argument("--prev-map", default=None, help="thisMethod=prevMethod,...")
a = ap.parse_args()

rd = Path(a.run_dir)
sm, rows = load_summary(rd / f"dim{a.dim}" / "summary.csv")
wil = load_wil(rd / f"dim{a.dim}" / "wilcoxon.csv")
funcs = sorted({f for f, _ in sm})
methods = [m for m in a.order.split(",") if any(k[1] == m for k in sm)]
missing = [m for m in a.order.split(",") if m not in methods]
ref = a.ref

out = []
W = out.append
W(f"===== dim{a.dim}  functions={len(funcs)}  methods={len(methods)} =====")
if missing: W(f"  MISSING METHODS: {missing}")
# completeness
holes = [(f, m) for f in funcs for m in methods if (f, m) not in sm]
if holes: W(f"  MISSING (function, method) CELLS: {holes}")

W("")
W(f"[A] OVERALL (mean over {len(funcs)} functions, %)  + pairwise evals_succ_mean vs {ref}")
W(f"  {'method':<16}{'1e-2':>8}{'1e-4':>8}{'1e-7':>8}{'1e-10':>8}{'evalsREF':>12}{'evalsM':>10}{'nFunc':>6}")
W("  " + "-" * 76)
overall = {}
for m in methods:
    vals = [sum(pct(sm[(f, m)][k]) for f in funcs) / len(funcs) for k in SR]
    overall[m] = vals[-1]
    er = em = 0.0; nf = 0
    for f in funcs:
        x, y = num(sm[(f, ref)]["evals_succ_mean"]), num(sm[(f, m)]["evals_succ_mean"])
        if not math.isnan(x) and not math.isnan(y):
            er += x; em += y; nf += 1
    if nf:
        W(f"  {m:<16}{vals[0]:8.2f}{vals[1]:8.2f}{vals[2]:8.2f}{vals[3]:8.2f}{er/nf:12.1f}{em/nf:10.1f}{nf:6d}")
    else:
        W(f"  {m:<16}{vals[0]:8.2f}{vals[1]:8.2f}{vals[2]:8.2f}{vals[3]:8.2f}{'-':>12}{'-':>10}{0:6d}")

W("")
W("[B] RANKING by SR@1e-10")
for i, (m, v) in enumerate(sorted(overall.items(), key=lambda kv: -kv[1]), 1):
    W(f"  {i}. {m:<16}{v:6.2f}%")

W("")
W("[C] PER-FUNCTION SR@1e-10 (%) all methods")
W("  " + f"{'function':<26}" + "".join(f"{m[:12]:>12}" for m in methods))
for f in funcs:
    W("  " + f"{f:<26}" + "".join(f"{pct(sm[(f,m)][P]):12.0f}" for m in methods))

W("")
W("[C2] PER-FUNCTION evals_succ_mean ('-' = no successful run)")
W("  " + f"{'function':<26}" + "".join(f"{m[:12]:>12}" for m in methods))
for f in funcs:
    cells = []
    for m in methods:
        v = num(sm[(f, m)]["evals_succ_mean"])
        cells.append(f"{'-':>12}" if math.isnan(v) else f"{v:12.0f}")
    W("  " + f"{f:<26}" + "".join(cells))

if "MC-ESO-v1" in methods:
    W("")
    W(f"[D] SR@1e-10 CHANGE  MC-ESO-v1 (old default) -> {ref} (new default)")
    ch = imp = wor = 0
    for f in funcs:
        o, n = pct(sm[(f, "MC-ESO-v1")][P]), pct(sm[(f, ref)][P])
        if o != n:
            ch += 1
            if n > o: imp += 1
            else: wor += 1
            W(f"  {'+' if n>o else '-'} {f:<26}{o:5.0f} -> {n:5.0f}  ({n-o:+.0f})")
    W(f"  changed: {ch}  improved: {imp}  worsened: {wor}  unchanged: {len(funcs)-ch}")

bl = [m for m in a.baselines.split(",") if m in methods]
W("")
W(f"[E] {ref} behind BEST of the {len(bl)} comparison methods by >=15pt on SR@1e-10")
n_e = 0
for f in funcs:
    r = pct(sm[(f, ref)][P])
    best_m = max(bl, key=lambda m: pct(sm[(f, m)][P]))
    b = pct(sm[(f, best_m)][P])
    if b - r >= 15:
        n_e += 1
        W(f"  {f:<24} {ref} {r:5.0f}  best {best_m:<14}{b:5.0f}  gap {b-r:+.0f}")
if not n_e: W("  (none)")

W("")
W(f"[F] WILCOXON vs {ref} (ref), alpha=0.05, two-sided")
for m in methods:
    if m == ref: continue
    rws = [r for r in wil if r.get("method") == m or r.get("method_b") == m]
    bw = bl_ = 0; gl = []; ll = []
    for r in rws:
        p = num(r.get("p_value_two_sided") or r.get("p_value") or "")
        a12 = num(r.get("a12") or "")
        if math.isnan(p) or p >= 0.05 or math.isnan(a12): continue
        fn = r["function"].split("-")[0]
        eff = (r.get("a12_magnitude") or "").strip() or "?"
        if a12 > 0.5: bw += 1; gl.append(f"{fn} A12={a12:.2f} {eff}")
        else: bl_ += 1; ll.append(f"{fn} A12={a12:.2f} {eff}")
    W(f"  {m:<14} {ref} better {bw:2d} / {m} better {bl_:2d}")
    if gl: W(f"      {ref}>: " + ", ".join(gl))
    if ll: W(f"      {m}>: " + ", ".join(ll))

if a.prev and a.prev_map:
    W("")
    W(f"[G] CONSISTENCY vs prior run ({Path(a.prev).name})")
    psm, _ = load_summary(a.prev)
    for pair in a.prev_map.split(","):
        cur, old = pair.split("=")
        if cur not in methods: continue
        diffs = []
        for f in funcs:
            if (f, old) not in psm: diffs.append((f, None, None)); continue
            x, y = pct(sm[(f, cur)][P]), pct(psm[(f, old)][P])
            if x != y: diffs.append((f, y, x))
        cv = sum(pct(sm[(f, cur)][P]) for f in funcs)/len(funcs)
        ov = sum(pct(psm[(f, old)][P]) for f in funcs if (f, old) in psm)/len(funcs)
        if not diffs:
            W(f"  {cur} (this {cv:.2f}%) vs {old} (prior {ov:.2f}%): ALL {len(funcs)} FUNCTIONS MATCH")
        else:
            W(f"  {cur} (this {cv:.2f}%) vs {old} (prior {ov:.2f}%): {len(diffs)} FUNCTION(S) DIFFER")
            for f, y, x in diffs:
                W(f"      {f}: prior {y} -> this {x}")

print("\n".join(out))
