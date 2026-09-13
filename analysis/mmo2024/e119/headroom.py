"""e119 headroom: if branch (ii) holds, how much PR is sitting in the clustered optima?

Not a decision rule -- it prices the arm that the pre-registered (ii) branch licenses.
Counterfactual: bring clustered optima up to the reach rate the SAME run already
achieves on isolated optima in the SAME problem, change nothing else.
"""
import csv
import gzip
import pathlib

import numpy as np
from scipy.spatial.distance import cdist

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIDS = [f"M{p:02d}" for p in range(1, 17)]
z = np.load(ROOT / "analysis/mmo2024/e118/optima_d10_pin01.npz")
nullmed = {r["pid"]: float(r["null_med"])
           for r in csv.DictReader(open(ROOT / "analysis/mmo2024/e118/structure_d10.csv"))}


def components(members, dmat, tau):
    rem, comps = set(members), []
    while rem:
        s = rem.pop()
        comp, stack = [s], [s]
        while stack:
            u = stack.pop()
            for v in [v for v in list(rem) if dmat[u, v] < tau]:
                rem.discard(v); comp.append(v); stack.append(v)
        comps.append(sorted(comp))
    return comps


rows = []
print("pid   K  ncl  |  touch: cl   iso   all -> cf  |  deep: cl   iso   all -> cf")
for pid in PIDS:
    pts = z[pid]; K = len(pts)
    d = cdist(pts, pts); np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1); tau = 0.60 * nullmed[pid]
    in_cl = {i for c in components([i for i in range(K) if nn[i] < tau], d, tau)
             if len(c) >= 2 for i in c}
    iso = [i for i in range(K) if i not in in_cl]
    cl = sorted(in_cl)

    p = ROOT / f"analysis/mmo2024/e115/descents/{pid}-D10-PIN01_seed0.csv.gz"
    with gzip.open(p, "rt") as f:
        dump = [(int(r["land_opt"]), float(r["best_f"])) for r in csv.DictReader(f)]
    touch = np.zeros(K, bool); deep = np.zeros(K, bool)
    for lo, bf in dump:
        touch[lo] = True
        if bf <= 1e-5:
            deep[lo] = True

    out = {}
    for lab, m in (("touch", touch), ("deep", deep)):
        r_cl = m[cl].mean() if cl else float("nan")
        r_is = m[iso].mean() if iso else float("nan")
        r_all = m.mean()
        # counterfactual: clustered optima reached at the isolated rate of this problem
        cf = (len(cl) * (r_is if cl else 0) + m[iso].sum()) / K if cl else r_all
        out[lab] = (r_cl, r_is, r_all, cf)
    rows.append(dict(pid=pid, K=K, n_cl=len(cl), n_iso=len(iso),
                     **{f"{l}_{k}": v for l, t in out.items()
                        for k, v in zip(("cl", "iso", "all", "cf"), t)}))
    t, dp = out["touch"], out["deep"]
    print(f"{pid}  {K:2d}  {len(cl):3d}  | {t[0]:9.3f} {t[1]:5.3f} {t[2]:5.3f} -> {t[3]:.3f} "
          f"| {dp[0]:8.3f} {dp[1]:5.3f} {dp[2]:5.3f} -> {dp[3]:.3f}")

with open(HERE / "headroom_d10.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

for lab in ("touch", "deep"):
    a = np.mean([r[f"{lab}_all"] for r in rows])
    c = np.mean([r[f"{lab}_cf"] for r in rows])
    aA = np.mean([r[f"{lab}_all"] for r in rows[:8]])
    cA = np.mean([r[f"{lab}_cf"] for r in rows[:8]])
    print(f"\n{lab:5s} 16-problem mean: observed {a:.4f} -> counterfactual {c:.4f} "
          f"(+{c-a:.4f})   group A only: {aA:.4f} -> {cA:.4f} (+{cA-aA:.4f})")
n_cl = sum(r["n_cl"] for r in rows); n_tot = sum(r["K"] for r in rows)
print(f"\nclustered optima: {n_cl}/{n_tot} = {n_cl/n_tot:.3f} of the pool, all of them in group A")
print("published best (D=10, 15 instances): MPR 0.651 (RR-CMA-ES) / Score 0.6080 (TRDE-LR)")
