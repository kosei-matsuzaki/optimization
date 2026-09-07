"""Paired read of the two arms: did the basin-relative counter buy depth?

Reports, per accuracy: the paired mean, seed-direction counts (win/tie/loss),
the Wilcoxon signed-rank p (MC-ESO base as reference) and the A12 effect size,
on the official PR and on the true-attribution PR (entry 76: PR@1e-1 is a rho
artefact on this function). Hunt count / length come along because longer hunts
are paid for out of the hunt budget, so the coverage side has to be visible.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EPS = ["0.1", "0.01", "0.001", "0.0001", "1e-05"]
LABEL = {"0.1": "1e-1", "0.01": "1e-2", "0.001": "1e-3",
         "0.0001": "1e-4", "1e-05": "1e-5"}


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x = y), x = variant, y = base."""
    gt = sum((a > b) for a in x for b in y)
    eq = sum((a == b) for a in x for b in y)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def paired(v: np.ndarray, b: np.ndarray, name: str) -> None:
    d = v - b
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    if np.any(d != 0):
        p = wilcoxon(v, b, zero_method="wilcox").pvalue
    else:
        p = 1.0
    print(f"{name:<16} base {b.mean():.4f}  basin {v.mean():.4f}  "
          f"delta {d.mean():+.4f}  {w}/{t}/{l}  p={p:.4g}  A12={a12(v, b):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", default="analysis/hm/e77/N18_*_off*_pr.csv")
    ap.add_argument("--hunts", default="analysis/hm/e77/N18_*_off*_hunts.csv.gz")
    a = ap.parse_args()

    pr = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(a.pr))],
                   ignore_index=True).sort_values(["arm", "seed"])
    b = pr[pr.arm == "base"].set_index("seed").sort_index()
    v = pr[pr.arm == "basin"].set_index("seed").sort_index()
    seeds = sorted(set(b.index) & set(v.index))
    b, v = b.loc[seeds], v.loc[seeds]
    print(f"== {len(seeds)} paired seeds: {seeds}")

    print("\n== official PR (core.runner scorer)")
    for e in EPS:
        paired(v[f"pr_{e}"].values, b[f"pr_{e}"].values, f"PR@{LABEL[e]}")
    print("\n== PR with the reported points attributed to the six true optima")
    for e in EPS:
        paired(v[f"prtrue_{e}"].values, b[f"prtrue_{e}"].values,
               f"PRtrue@{LABEL[e]}")

    print("\n== the mechanism (hunt budget)")
    for col in ("n_hunts", "median_hunt_len", "n_rep", "best_f"):
        paired(v[col].values.astype(float), b[col].values.astype(float), col)
    print(f"basin resets / run: median {v.n_basin_resets.median():.0f} "
          f"(min {v.n_basin_resets.min()}, max {v.n_basin_resets.max()})")

    h = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(a.hunts))],
                  ignore_index=True)
    print("\n== hunt length, hunts after the first (evals)")
    for arm, g in h[h.hunt > 0].groupby("arm"):
        print(f"{arm:<6} n={len(g):<6} median={g.n_evals.median():.0f} "
              f"q25={g.n_evals.quantile(.25):.0f} q75={g.n_evals.quantile(.75):.0f} "
              f"max={g.n_evals.max():.0f}")
    print("\n== hunts reaching a depth, by arm (share of all hunts)")
    for arm, g in h.groupby("arm"):
        for e in (1e-1, 1e-3, 1e-5):
            n = int((g.bb_end <= e).sum())
            print(f"{arm:<6} bb_end <= {e:g}: {n:5d} ({n / len(g) * 100:5.1f}%)")
    print("\n== distinct true optima hit by any hunt (per seed, dist <= 0.05)")
    for arm, g in h.groupby("arm"):
        per = g[g.dist <= 0.05].groupby("seed").gopt.nunique()
        deep = (h[(h.arm == arm) & (h.bb_end <= 1e-3)]
                .groupby("seed").gopt.nunique().reindex(per.index).fillna(0))
        print(f"{arm:<6} any-depth median {per.median():.1f}  "
              f"hunts reaching 1e-3 cover median {deep.median():.1f} optima")


if __name__ == "__main__":
    main()
