"""e121 rescaling -- POST-HOC. Is the member deficit intrinsic, or just crowding?

The far-field sweep shows members losing reach from r = 3 while isolated optima
hold to r = 8.  Two readings again:

  (I)  members have intrinsically defective basins (something about the landscape
       near a cluster member resists this descent), or
  (II) the basin radius is set by the distance to the nearest neighbouring optimum
       -- a start beyond roughly NN/2 is captured by the neighbour -- and members
       simply have small NN (entry 118: 4.512 vs 8.165).

(II) predicts that plotting reach against r / NN(target) collapses the two curves
onto one.  (I) predicts the member curve stays below after rescaling.

`nn` comes from entry 119's enrichment_d10.csv (distance to the nearest other
optimum of the same problem).
"""
from __future__ import annotations
import csv
import glob
import gzip
import pathlib

import numpy as np
from scipy.stats import mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent
BINS = [(0.0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
        (1.0, 1.5), (1.5, 10.0)]


def a12(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return ((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) \
        / (len(x) * len(y))


def main():
    nn = {(r["pid"], int(r["idx"])): float(r["nn"]) for r in
          csv.DictReader(open(HERE.parents[0] / "e119/enrichment_d10.csv"))}
    rows = []
    for sub in ("by_problem", "by_problem_far"):
        for p in sorted(glob.glob(str(HERE / sub / "*.csv.gz"))):
            for r in csv.DictReader(gzip.open(p, 'rt')):
                if r["cond"] != "deployed" or float(r["r"]) == 0.0:
                    continue
                r["clustered"] = int(r["clustered"])
                r["hit"] = int(r["hit"])
                r["u"] = float(r["r_eff"]) / nn[(r["pid"], int(r["idx"]))]
                rows.append(r)

    mnn = np.mean([nn[k] for k in {(x["pid"], int(x["idx"])) for x in rows
                                   if x["clustered"] == 1}])
    inn = np.mean([nn[k] for k in {(x["pid"], int(x["idx"])) for x in rows
                                   if x["clustered"] == 0}])
    print(f"e121 rescaled by NN (POST-HOC). mean NN: member {mnn:.3f}, "
          f"isolated {inn:.3f} (ratio {inn / mnn:.2f})")
    print(f"{len(rows)} deployed descents at r > 0\n")
    print(" r/NN bin      member        isolated       MWU p    A12")
    for lo, hi in BINS:
        xm = [x["hit"] for x in rows if x["clustered"] == 1 and lo <= x["u"] < hi]
        xi = [x["hit"] for x in rows if x["clustered"] == 0 and lo <= x["u"] < hi]
        if len(xm) < 5 or len(xi) < 5:
            print(f" [{lo:.3f},{hi:.2f})  n too small (m={len(xm)}, i={len(xi)})")
            continue
        if len(set(xm + xi)) == 1:
            print(f" [{lo:.3f},{hi:.2f})  {np.mean(xm):.3f} (n={len(xm):3d})  "
                  f"{np.mean(xi):.3f} (n={len(xi):3d})   (degenerate)")
            continue
        u, p = mannwhitneyu(xm, xi, alternative="less")
        print(f" [{lo:.3f},{hi:.2f})  {np.mean(xm):.3f} (n={len(xm):3d})  "
              f"{np.mean(xi):.3f} (n={len(xi):3d})   {p:6.3f}  {a12(xm, xi):.3f}")

    # the same comparison pooled, and the half-reach point in NN units
    xm = [x["hit"] for x in rows if x["clustered"] == 1]
    xi = [x["hit"] for x in rows if x["clustered"] == 0]
    u, p = mannwhitneyu(xm, xi, alternative="less")
    print(f"\npooled (unrescaled): member {np.mean(xm):.3f} vs iso {np.mean(xi):.3f}, "
          f"MWU p={p:.3g}, A12={a12(xm, xi):.3f}")
    print("half-reach in NN units (last bin with reach >= 0.5):")
    for c, lab in ((1, "member  "), (0, "isolated")):
        ok = [hi for lo, hi in BINS
              if len([x for x in rows if x["clustered"] == c and lo <= x["u"] < hi]) >= 5
              and np.mean([x["hit"] for x in rows
                           if x["clustered"] == c and lo <= x["u"] < hi]) >= 0.5]
        print(f"  {lab}: {max(ok) if ok else 'none'}")


if __name__ == "__main__":
    main()
