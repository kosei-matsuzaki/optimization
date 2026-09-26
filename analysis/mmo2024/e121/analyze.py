"""e121 analysis -- the pre-registered decision rule, then the mechanism split.

Reads by_problem/*.csv (one row per descent) and prints:
  0. calibration (r = 0), the ceiling of the measurement
  1. the reach curves, members vs matched isolated control, both sigma0 conditions
  2. the pre-registered thresholds on the `deployed` member curve
  3. the falsifier: members vs isolated at the same r, paired over the 8 problems
  4. r_half for each curve
"""
from __future__ import annotations
import csv
import glob
import gzip
import pathlib

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = pathlib.Path(__file__).resolve().parent
RADII = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def a12(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def load():
    rows = []
    paths = sorted(glob.glob(str(HERE / "by_problem" / "*.csv.gz")))
    if not paths:                       # entry 170 deleted the row-level dumps
        raise SystemExit(
            "e121/by_problem/*.csv.gz was removed by entry 170 (the basin-radius "
            "route closed at entry 125).  Every number this script printed is in "
            "docs/acceptance_topology.md, section 121; see DUMPS_REMOVED.md.  "
            "To re-measure: python3 analysis/mmo2024/e121/basin_radius.py --pid M01")
    for p in paths:
        for r in csv.DictReader(gzip.open(p, 'rt')):
            r["r"] = float(r["r"])
            r["clustered"] = int(r["clustered"])
            r["hit"] = int(r["hit"])
            r["hit_loose"] = int(r["hit_loose"])
            r["best_f"] = float(r["best_f"])
            r["dist_target"] = float(r["dist_target"])
            rows.append(r)
    return rows


def rate(rows, cond, clustered, r, key="hit"):
    s = [x[key] for x in rows
         if x["cond"] == cond and x["clustered"] == clustered and x["r"] == r]
    return (np.mean(s) if s else float("nan")), len(s)


def r_half(rows, cond, clustered, key="hit"):
    """Largest r at which the reach rate is still >= 0.5 (nan if never)."""
    ok = [r for r in RADII if rate(rows, cond, clustered, r, key)[0] >= 0.5]
    return max(ok) if ok else float("nan")


def main():
    rows = load()
    pids = sorted({r["pid"] for r in rows})
    print(f"e121: {len(rows)} descents, {len(pids)} problems ({', '.join(pids)})")
    nm = len({(r['pid'], r['idx']) for r in rows if r['clustered'] == 1})
    ni = len({(r['pid'], r['idx']) for r in rows if r['clustered'] == 0})
    print(f"targets: {nm} members, {ni} isolated controls\n")

    print("0. CALIBRATION -- r = 0 (start ON the optimum). Ceiling of the measurement.")
    for cond in ("deployed", "matched"):
        for c, lab in ((1, "member "), (0, "isolated")):
            h, n = rate(rows, cond, c, 0.0)
            hl, _ = rate(rows, cond, c, 0.0, "hit_loose")
            bf = np.median([x["best_f"] for x in rows if x["cond"] == cond
                            and x["clustered"] == c and x["r"] == 0.0])
            print(f"   {cond:9s} {lab}: hit {h:.3f}  loose {hl:.3f}  "
                  f"median best_f {bf:.3g}  (n={n})")

    print("\n1. REACH CURVES  P(reach target | start distance r)")
    for cond in ("deployed", "matched"):
        print(f"\n   sigma0 = {cond}")
        print("     r     member  isolated   |  member(loose) isolated(loose)")
        for r in RADII:
            hm, n1 = rate(rows, cond, 1, r)
            hi_, n0 = rate(rows, cond, 0, r)
            lm, _ = rate(rows, cond, 1, r, "hit_loose")
            li, _ = rate(rows, cond, 0, r, "hit_loose")
            tag = "  <- calibration" if r == 0.0 else ""
            print(f"   {r:5.2f}   {hm:.3f}     {hi_:.3f}    |    {lm:.3f}        "
                  f"{li:.3f}{tag}")
        print(f"     r_half: member {r_half(rows, cond, 1):.2f}  "
              f"isolated {r_half(rows, cond, 0):.2f}")

    print("\n2. PRE-REGISTERED RULE (deployed curve, members, strict hit)")
    p025 = rate(rows, "deployed", 1, 0.25)[0]
    p010 = rate(rows, "deployed", 1, 0.1)[0]
    print(f"   P(reach | r=0.25) = {p025:.3f}   threshold for (alpha): >= 0.5")
    print(f"   P(reach | r=0.10) = {p010:.3f}   threshold for (beta):  <  0.2")
    if p025 >= 0.5:
        verdict = "(alpha) -- basin is there, resolution is the issue"
    elif p010 < 0.2:
        verdict = "(beta) -- this descent operator cannot enter the members"
    else:
        verdict = "INTERMEDIATE -- report the curve shape (r_half)"
    print(f"   => {verdict}")

    print("\n3. FALSIFIER -- is the member curve actually below the isolated curve?")
    print("   (if the isolated control is equally flat, the measurement is about")
    print("    the 3125-eval budget and nothing above may be concluded)")
    for cond in ("deployed", "matched"):
        print(f"\n   sigma0 = {cond}")
        for r in RADII[1:]:
            xm = [x["hit"] for x in rows if x["cond"] == cond
                  and x["clustered"] == 1 and x["r"] == r]
            xi = [x["hit"] for x in rows if x["cond"] == cond
                  and x["clustered"] == 0 and x["r"] == r]
            if len(set(xm + xi)) == 1:
                print(f"   r={r:4.2f}  member {np.mean(xm):.3f} vs iso {np.mean(xi):.3f}"
                      f"   (degenerate: all identical)")
                continue
            u, p = mannwhitneyu(xm, xi, alternative="less")
            print(f"   r={r:4.2f}  member {np.mean(xm):.3f} vs iso {np.mean(xi):.3f}"
                  f"   MWU p={p:.3g}  A12={a12(xm, xi):.3f}")
        # paired over the 8 problems, pooled over r > 0
        pm, pi = [], []
        for pid in pids:
            sm = [x["hit"] for x in rows if x["pid"] == pid and x["cond"] == cond
                  and x["clustered"] == 1 and x["r"] > 0]
            si = [x["hit"] for x in rows if x["pid"] == pid and x["cond"] == cond
                  and x["clustered"] == 0 and x["r"] > 0]
            pm.append(np.mean(sm))
            pi.append(np.mean(si))
        d = np.array(pm) - np.array(pi)
        if np.any(d != 0):
            w, pw = wilcoxon(pm, pi, alternative="less")
            print(f"   paired over {len(pids)} problems (r>0 pooled): "
                  f"member {np.mean(pm):.3f} vs iso {np.mean(pi):.3f}  "
                  f"Wilcoxon p={pw:.4g}  ({int((d < 0).sum())}/{int((d > 0).sum())}"
                  f"/{int((d == 0).sum())} down/up/tie)")
        else:
            print("   paired: all differences zero")

    print("\n4. MECHANISM -- matched vs deployed on the SAME targets (members only)")
    for r in RADII[1:]:
        key = {}
        for x in rows:
            if x["clustered"] == 1 and x["r"] == r:
                key.setdefault((x["pid"], x["idx"]), {})[x["cond"]] = x["hit"]
        dep = [v["deployed"] for v in key.values() if len(v) == 2]
        mat = [v["matched"] for v in key.values() if len(v) == 2]
        gain = np.mean(mat) - np.mean(dep)
        if np.any(np.array(mat) != np.array(dep)):
            w, pw = wilcoxon(mat, dep, alternative="greater")
            ps = f"p={pw:.3g}"
        else:
            ps = "p=1 (identical)"
        print(f"   r={r:4.2f}  deployed {np.mean(dep):.3f} -> matched "
              f"{np.mean(mat):.3f}  (+{gain:.3f})  {ps}  n={len(dep)}")

    print("\n5. PER-PROBLEM (deployed, strict hit, r>0 pooled)")
    print("   pid    member  isolated   n_mem")
    for pid in pids:
        sm = [x["hit"] for x in rows if x["pid"] == pid
              and x["cond"] == "deployed" and x["clustered"] == 1 and x["r"] > 0]
        si = [x["hit"] for x in rows if x["pid"] == pid
              and x["cond"] == "deployed" and x["clustered"] == 0 and x["r"] > 0]
        print(f"   {pid}   {np.mean(sm):.3f}    {np.mean(si):.3f}    {len(sm) // 5}")


if __name__ == "__main__":
    main()
