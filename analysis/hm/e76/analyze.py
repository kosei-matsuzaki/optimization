"""Read the descent probe's per-hunt rows and decide between shapes (i) and (ii).

(i)  the hunt's descent has flattened before the release  -> the basin is not
     being descended any more when the hunt is cut (entry 75's reading);
(ii) the descent is still running at the release          -> the cut is
     premature and entry 75's "timer" reading must be withdrawn.

The discriminators, per hunt that got within eps of a global optimum:

  tail_frac_idle  fraction of the hunt spent after its last `basin_best`
                  improvement. ~0 means it was still improving when cut.
  slope_last25    decades of `basin_best` per 1000 evals over the last quarter.
  need_evals      evals still needed to reach 1e-3 if that final slope held.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunts", default="analysis/hm/e76/N18_off*_hunts.csv.gz")
    ap.add_argument("--reported", default="analysis/hm/e76/N18_off*_reported.csv.gz")
    ap.add_argument("--eps", type=float, default=1e-1)
    a = ap.parse_args()

    h = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(a.hunts))],
                  ignore_index=True)
    print(f"== hunts: {len(h)} rows, {h.seed.nunique()} seeds "
          f"({sorted(h.seed.unique())})")
    print(h.groupby("seed").agg(hunts=("hunt", "count"),
                                evals=("n_evals", "sum"),
                                median_len=("n_evals", "median")).to_string())

    #  the first hunt of a run is the one that drills the global best: it is the
    #  only one whose stagnation counter is reset by its own progress.
    first = h[h.hunt == 0]
    later = h[h.hunt > 0]
    print("\n== hunt length: first hunt vs the rest (evals)")
    print(f"first : n={len(first)} median={first.n_evals.median():.0f} "
          f"min={first.n_evals.min():.0f} max={first.n_evals.max():.0f} "
          f"bb_end median={first.bb_end.median():.3g}")
    print(f"later : n={len(later)} median={later.n_evals.median():.0f} "
          f"q25={later.n_evals.quantile(.25):.0f} q75={later.n_evals.quantile(.75):.0f}")

    deep = h[(h.bb_end <= a.eps) & (h.n_improve > 0)].copy()
    print(f"\n== hunts reaching bb_end <= {a.eps:g}: {len(deep)} "
          f"({len(deep) / len(h) * 100:.1f}% of hunts)")
    deep["need_evals"] = np.where(
        deep.slope_last25 > 0,
        (np.log10(deep.bb_end) + 3.0) / deep.slope_last25 * 1000.0, np.inf)
    deep["reached_1e3"] = deep.bb_end <= 1e-3

    print("\n== per true global optimum (all seeds pooled)")
    rows = []
    for j, g in deep.groupby("gopt"):
        rows.append(dict(
            gopt=j, hunts=len(g), seeds=g.seed.nunique(),
            best_bb=g.bb_end.min(), med_bb=g.bb_end.median(),
            med_dist=g.dist.median(),
            med_len=g.n_evals.median(),
            med_idle=g.tail_frac_idle.median(),
            med_slope_last=g.slope_last25.median(),
            hunts_le_1e3=int(g.reached_1e3.sum()),
            med_need=g.need_evals.median()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    stall = deep[~deep.reached_1e3]
    print(f"\n== the stalled hunts ({len(stall)} of {len(deep)}): shape test")
    print(f"tail_frac_idle   median={stall.tail_frac_idle.median():.3f} "
          f"q25={stall.tail_frac_idle.quantile(.25):.3f} "
          f"q75={stall.tail_frac_idle.quantile(.75):.3f} "
          f"frac(idle > 0.25)={np.mean(stall.tail_frac_idle > 0.25):.3f}")
    print(f"slope_last25     median={stall.slope_last25.median():.4g} "
          f"decades/1000ev   frac(slope == 0)={np.mean(stall.slope_last25 <= 0):.3f}")
    print(f"need_evals(1e-3) median={stall.need_evals.median():.4g} "
          f"vs hunt length median={stall.n_evals.median():.0f} "
          f"-> ratio={stall.need_evals.median() / max(stall.n_evals.median(), 1):.3g}")
    print(f"frac(need <= own hunt length)  = "
          f"{np.mean(stall.need_evals <= stall.n_evals):.3f}")
    print(f"frac(need <= 10x hunt length)  = "
          f"{np.mean(stall.need_evals <= 10 * stall.n_evals):.3f}")

    #  per-seed direction agreement, so the reading is not a pooled artefact
    print("\n== per seed (stalled hunts only)")
    ps = stall.groupby("seed").agg(n=("hunt", "count"),
                                   med_idle=("tail_frac_idle", "median"),
                                   med_slope=("slope_last25", "median"),
                                   med_len=("n_evals", "median"))
    print(ps.to_string(float_format=lambda v: f"{v:.4g}"))

    r = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(a.reported))],
                  ignore_index=True)
    print("\n== reported set: is the 1e-1 coverage six *different* global basins?")
    for eps, col in ((1e-1, "eps_1e-1"), (1e-3, "eps_1e-3"), (1e-5, "eps_1e-5")):
        per_seed = []
        for s, g in r.groupby("seed"):
            sel = g[(g[col] == 1) & (g.is_seed == 1)]
            per_seed.append((sel.gopt.nunique(), len(sel)))
        nun = np.array([p[0] for p in per_seed])
        npt = np.array([p[1] for p in per_seed])
        print(f"eps={eps:<6g} scored points/seed median={np.median(npt):.1f}  "
              f"distinct TRUE optima/seed median={np.median(nun):.1f} "
              f"(min {nun.min()}, max {nun.max()})  PR_true={nun.mean() / 6:.4f}")
    near = r[(r["eps_1e-1"] == 1) & (r.is_seed == 1)]
    print(f"distance to nearest true optimum, scored 1e-1 points: "
          f"median={near.dist.median():.4g} q90={near.dist.quantile(.9):.4g} "
          f"max={near.dist.max():.4g}")


if __name__ == "__main__":
    main()
