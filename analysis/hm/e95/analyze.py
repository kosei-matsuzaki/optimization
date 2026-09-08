#!/usr/bin/env python3
"""Score the hill-valley arm against the class null (entry 95, question 2).

Question 2 asks whether adding HillVallEA's mechanism -- the segment test that
splits a population into basins -- moves the class ceiling that entries 88/89
measured for "uniform restart + local descent".  The dumps this reads are
budget-matched replicates: every arm got the same evaluations on the same
function, so one replicate is one run at the suite's normal budget and its
`cov_<eps>` column is the class ceiling in the units entry 88 used
(distinct optima reached with best_f <= eps, per run).

Pairing is by replicate index.  The absolute level of a ceiling estimate is
biased down by finite draws (entry 94), and that bias is common to the arms,
so the paired difference is the quantity to read -- not the levels.
"""
from __future__ import annotations
import csv
import gzip
import sys
from pathlib import Path

import numpy as np
from scipy import stats

EPS = ("0.1", "0.01", "0.001", "0.0001", "1e-05")
# HillVallEA GECCO'18 table 4, best of 6 methods, PR@1e-5, normal budget
# (related_work.md).  F13 is at the suite's ceiling already.
PUBLISHED = {"N13-CF3-2D": 1.000, "N14-CF3-3D": 0.793,
             "N16-CF3-5D": 0.677, "N18-CF3-10D": 0.667}


def a12(x, y):
    """Vargha-Delaney A12: P(x > y) + 0.5 P(x == y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    g = sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    return float(g) / (len(x) * len(y))


def paired(a, b):
    """Wilcoxon signed-rank on the paired difference, with the w/t/l split."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    w, t, l = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    if w + l == 0:
        return w, t, l, 1.0
    return w, t, l, float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def load(paths):
    rows = []
    for p in paths:
        op = gzip.open if str(p).endswith(".gz") else open
        with op(p, "rt", newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def main(paths):
    rows = load(paths)
    funcs = sorted({r["func"] for r in rows})
    arms = ["iso", "hv", "split"]

    for fn in funcs:
        fr = [r for r in rows if r["func"] == fn]
        K = int(fr[0]["K"])
        pub = PUBLISHED.get(fn)
        print(f"\n=== {fn}  K = {K}  published best PR@1e-5 = {pub}"
              f"  -> threshold {pub * K:.2f} optima/run")
        by = {a: sorted([r for r in fr if r["arm"] == a],
                        key=lambda r: int(r["rep"])) for a in arms}
        n = len(by["iso"])
        print(f"  {n} replicates/arm, "
              f"{int(np.mean([int(r['evals']) for r in fr]))} evals each; "
              + ", ".join(f"{a}: {np.mean([int(r['descents']) for r in by[a]]):.0f}"
                          f" descents/{np.mean([int(r['units']) for r in by[a]]):.1f}"
                          f" units, hv_ev "
                          f"{np.mean([int(r['hv_ev']) for r in by[a]]):.0f}"
                          for a in arms))

        for e in EPS:
            col = f"cov_{e}"
            vals = {a: np.array([float(r[col]) for r in by[a]]) for a in arms}
            line = f"  eps {e:>6}: " + "  ".join(
                f"{a} {vals[a].mean():.3f}" for a in arms)
            for a in ("hv", "split"):
                w, t, l, p = paired(vals[a], vals["iso"])
                line += (f"   | {a}-iso {vals[a].mean() - vals['iso'].mean():+.3f}"
                         f" ({w}/{t}/{l}, p={p:.4f}, A12={a12(vals[a], vals['iso']):.2f})")
            print(line)

        # Which optima ever open, and does any arm clear the published bar?
        for e in ("0.1", "1e-05"):
            print(f"  union of optima reached at eps {e}:")
            for a in arms:
                seen, per = set(), []
                for r in by[a]:
                    s = {int(v) for v in r[f"set_{e}"].split("|") if v != ""}
                    seen |= s
                    per.append(s)
                hit = {j: sum(j in s for s in per) for j in range(K)}
                mx = max(len(s) for s in per)
                print(f"    {a:<5} union {sorted(seen)}  "
                      f"per-optimum hits/{n}: "
                      + " ".join(f"{j}:{hit[j]}" for j in range(K))
                      + f"  best single run {mx}/{K}")
        if pub is not None:
            print(f"  ceiling vs published ({pub * K:.2f}):")
            for a in arms:
                v = np.array([float(r["cov_1e-05"]) for r in by[a]])
                print(f"    {a:<5} mean {v.mean():.3f}  max over runs {v.max():.0f}"
                      f"  -> {'EXCEEDS' if v.mean() > pub * K else 'below'}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(Path(__file__).parent.glob("hv_*.csv.gz")))
