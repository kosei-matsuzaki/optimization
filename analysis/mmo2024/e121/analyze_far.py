"""e121 far field -- POST-HOC, not the pre-registered decision. Read as such.

The pre-registered sweep stops at r = 2.0, and every level of it came back at
0.69-1.00 with members indistinguishable from isolated controls.  In D = 10 that
is not a surprise: the mean Voronoi cell of a member has volume share 0.01915 of
a box of side 10, so its linear scale is (0.01915 * 10^10)^(1/10) ~ 6.7, and a
ball of radius 2 is ~5.6e-6 of it.  The pre-registered sweep therefore probed a
vanishing fraction of the cell and cannot locate the 13x deficit of entry 119.

This extends the DEPLOYED curve to r in {3, 4, 6, 8} on the same 110 targets and
asks where the reach actually collapses, and whether it collapses earlier for
members than for isolated optima.  Starts are clipped to the box, so `r_eff`
(the realised distance) is reported next to the nominal r.
"""
from __future__ import annotations
import csv
import glob
import gzip
import pathlib

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = pathlib.Path(__file__).resolve().parent


def a12(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def load(sub):
    rows = []
    for p in sorted(glob.glob(str(HERE / sub / "*.csv.gz"))):
        for r in csv.DictReader(gzip.open(p, 'rt')):
            r["r"] = float(r["r"])
            r["r_eff"] = float(r["r_eff"])
            r["clustered"] = int(r["clustered"])
            r["hit"] = int(r["hit"])
            r["hit_loose"] = int(r["hit_loose"])
            rows.append(r)
    return rows


def main():
    rows = [r for r in load("by_problem") if r["cond"] == "deployed"]
    rows += load("by_problem_far")
    pids = sorted({r["pid"] for r in rows})
    radii = sorted({r["r"] for r in rows})
    print(f"e121 far field (POST-HOC): deployed sigma0 = 1.0, {len(rows)} descents, "
          f"{len(pids)} problems")
    print("box [-5,5]^10; member mean Voronoi share 0.01915 -> linear scale ~6.7\n")

    print("  r    r_eff(med)   member  isolated    MWU p    A12     n")
    for r in radii:
        xm = [x["hit"] for x in rows if x["clustered"] == 1 and x["r"] == r]
        xi = [x["hit"] for x in rows if x["clustered"] == 0 and x["r"] == r]
        re = np.median([x["r_eff"] for x in rows if x["r"] == r])
        if len(set(xm + xi)) == 1:
            print(f"{r:5.2f}   {re:6.3f}     {np.mean(xm):.3f}    {np.mean(xi):.3f}"
                  f"    (degenerate)        {len(xm)}")
            continue
        u, p = mannwhitneyu(xm, xi, alternative="less")
        print(f"{r:5.2f}   {re:6.3f}     {np.mean(xm):.3f}    {np.mean(xi):.3f}"
              f"    {p:7.4f}  {a12(xm, xi):.3f}   {len(xm)}")

    print("\npaired over the 8 problems, at each far radius:")
    for r in [x for x in radii if x >= 3.0]:
        pm, pi = [], []
        for pid in pids:
            pm.append(np.mean([x["hit"] for x in rows if x["pid"] == pid
                               and x["clustered"] == 1 and x["r"] == r]))
            pi.append(np.mean([x["hit"] for x in rows if x["pid"] == pid
                               and x["clustered"] == 0 and x["r"] == r]))
        d = np.array(pm) - np.array(pi)
        if np.any(d != 0):
            w, pw = wilcoxon(pm, pi, alternative="less")
            ps = f"p={pw:.4f}"
        else:
            ps = "p=1"
        print(f"  r={r:4.1f}  member {np.mean(pm):.3f} vs iso {np.mean(pi):.3f}  {ps}"
              f"  ({int((d < 0).sum())}/{int((d > 0).sum())}/{int((d == 0).sum())}"
              f" down/up/tie)")

    print("\nwhere the reach halves (first r with member reach < 0.5):")
    for c, lab in ((1, "member  "), (0, "isolated")):
        curve = [(r, np.mean([x["hit"] for x in rows
                              if x["clustered"] == c and x["r"] == r])) for r in radii]
        below = [r for r, v in curve if v < 0.5]
        print(f"  {lab}: {below[0] if below else 'never in this sweep'}"
              f"   curve = {', '.join(f'{v:.2f}' for _, v in curve)}")

    print("\nwhere the descent goes instead when it misses (far radii, members):")
    for r in [x for x in radii if x >= 3.0]:
        miss = [x for x in rows if x["clustered"] == 1 and x["r"] == r and not x["hit"]]
        if not miss:
            continue
        d = np.median([float(x["dist"]) for x in miss])
        dt = np.median([float(x["dist_target"]) for x in miss])
        print(f"  r={r:4.1f}  n_miss={len(miss):3d}  median dist to its landing "
              f"{d:.3f}, to the target {dt:.3f}")


if __name__ == "__main__":
    main()
