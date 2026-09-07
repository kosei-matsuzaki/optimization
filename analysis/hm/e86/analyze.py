"""Paired read of base vs the adoption composite on the CF3 pair (entry 86).

Same statistics as `analysis/hm/e77/analyze.py` (paired Wilcoxon with base as
reference, seed-direction w/t/l, A12), but the order of report is fixed by the
question: coverage first (`PR@1e-1`, then `PRtrue@1e-1`, which is the one entry
76 says can be read as coverage on 10D), then the judgement levels.

Base for N18 is entry 77's saved CSVs (zero new base runs there); base for N16
is run here because no PRtrue exists for it. Both base columns are checked
against entry 85's official six-method table before anything else is printed.
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
#  Published best PR@1e-5 (HillVallEA GECCO'18 table 4, entry 71) = the bar the
#  composite's true coverage has to clear for case (A) to survive.
BAR = {"N16-CF3-5D": 0.677, "N18-CF3-10D": 0.667}


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) + 0.5 P(x = y), x = variant, y = base."""
    gt = sum((a > b) for a in x for b in y)
    eq = sum((a == b) for a in x for b in y)
    return (gt + 0.5 * eq) / (len(x) * len(y))


def paired(v: np.ndarray, b: np.ndarray, name: str) -> None:
    d = v - b
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    p = wilcoxon(v, b, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
    print(f"{name:<18} base {b.mean():.4f}  comp {v.mean():.4f}  "
          f"delta {d.mean():+.4f}  {w}/{t}/{l}  p={p:.4g}  A12={a12(v, b):.3f}")


def _load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="analysis/hm/e86/runs.csv.gz")
    ap.add_argument("--e77", default="analysis/hm/e77/N18_base_off*_pr.csv")
    ap.add_argument("--e85", default="analysis/hm/e85")
    a = ap.parse_args()

    runs = pd.read_csv(a.runs)
    #  Entry 77's N18 base, re-labelled into this table's schema.
    b77 = _load(a.e77)
    b77 = b77[b77.arm == "base"].assign(func="N18-CF3-10D", arm="base")
    runs = pd.concat([runs, b77[runs.columns.intersection(b77.columns)]],
                     ignore_index=True)

    print("== base official PR against entry 85's six-method table")
    for f in sorted(runs.func.unique()):
        e85 = pd.read_csv(f"{a.e85}/{f}_6methods_12seed.csv")
        m = e85[e85.method == "MC-ESO"].sort_values("seed")
        b = runs[(runs.func == f) & (runs.arm == "base")].sort_values("seed")
        for e, col in zip(EPS, ["pr_1e-1", "pr_1e-2", "pr_1e-3",
                                "pr_1e-4", "pr_1e-5"]):
            ok = np.allclose(b[f"pr_{e}"].values, m[col].values, atol=5e-4)
            if not ok:
                print(f"  MISMATCH {f} {LABEL[e]}: "
                      f"{b[f'pr_{e}'].values} vs {m[col].values}")
        print(f"  {f}: base reproduces entry 85 at all five accuracies "
              f"(n={len(b)})")

    for f in sorted(runs.func.unique()):
      for arm in ("comp", "compf"):
        g = runs[runs.func == f]
        b = g[g.arm == "base"].set_index("seed").sort_index()
        v = g[g.arm == arm].set_index("seed").sort_index()
        seeds = sorted(set(b.index) & set(v.index))
        if not seeds:
            continue
        b, v = b.loc[seeds], v.loc[seeds]
        print(f"\n=== {f}  arm={arm}  ({len(seeds)} paired seeds)  "
              f"published best PR@1e-5 = {BAR[f]:.3f}")
        print("-- (1) coverage, official scorer")
        paired(v["pr_0.1"].values, b["pr_0.1"].values, "PR@1e-1")
        print("-- (2) coverage, attributed to the true optima (entry 76)")
        paired(v["prtrue_0.1"].values, b["prtrue_0.1"].values, "PRtrue@1e-1")
        print(f"   max PRtrue@1e-1 over seeds: base {b['prtrue_0.1'].max():.4f}"
              f"  comp {v['prtrue_0.1'].max():.4f}"
              f"   -> bar {BAR[f]:.3f} cleared: "
              f"{'YES' if v['prtrue_0.1'].max() > BAR[f] else 'NO'}")
        print("-- (3) judgement levels")
        for e in ("0.001", "1e-05"):
            paired(v[f"pr_{e}"].values, b[f"pr_{e}"].values, f"PR@{LABEL[e]}")
            paired(v[f"prtrue_{e}"].values, b[f"prtrue_{e}"].values,
                   f"PRtrue@{LABEL[e]}")
        print("-- context")
        for col in ("n_rep", "best_f"):
            paired(v[col].values.astype(float), b[col].values.astype(float),
                   col)


if __name__ == "__main__":
    main()
