#!/usr/bin/env python3
"""Is the `sigma_only` peak-ratio gain the sigma, or just the extra hunts?

Entry 23 left one confounder open. Taking the post-restart sigma down to the
local basin spacing also makes each hunt cheaper, so the number of hunts a run
fits into its budget rises (N07: 35.5 -> 49.1). The gain could be that, and
nothing about the sigma.

The release knobs cannot separate them: on Vincent every hunt ends through
`hunt_level_tol` (all global optima sit at the same height), so shortening the
stagnation window or loosening the sigma-floor test does not add hunts --
measured, base holds at 35-36 for `hunt_no_improve_mult` in 0.05..0.5 and
`exhausted_sigma_tol` in 1.5..25.

So the control is budget: run *base* at a larger budget until it holds as many
hunts as `sigma_only` does at 20k, and read its peak ratio there. This hands
base the extra evaluations for free, so it is biased towards the null -- if
base still does not reach `sigma_only`'s peak ratio at a matched (or higher)
hunt count, the hunts are not the explanation.

Usage:
  python3 scripts/hunt_confound.py --func N07-Vincent2D \
      base20k=analysis/hm/commit15_base.csv sigma_only=analysis/hm/commit15_sigma_only.csv \
      base28k=analysis/hm/base28k_n07.csv
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def load(path: str, func: str) -> dict[float, dict[int, dict]]:
    """{eps: {seed: row}} for one function out of a diagnose_niching CSV."""
    out: dict[float, dict[int, dict]] = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["function"] != func:
                continue
            out[float(r["eps"])][int(r["seed"])] = {
                "pr": float(r["reported"]) / float(r["K"]),
                "visited": float(r["visited"]) / float(r["K"]),
                "hunts": float(r["hunts"]),
                "evals": float(r["evals"]),
                # Entry 40 read the *mechanism* off these three (a sigma arm
                # moves hunts/landed, a commit arm moves distinct alone), so
                # carry them here rather than re-parsing the same CSV elsewhere.
                "distinct": float(r["distinct"]),
                "landed": float(r["landed"]),
                "blocked": float(r["blocked"]),
            }
    return out


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney on the paired differences: share of seeds where x wins."""
    d = x - y
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / len(d))


def paired(x: np.ndarray, y: np.ndarray) -> tuple[str, str, float]:
    """w/t/l, p, A12 for x against reference y over the shared seeds."""
    d = x - y
    w, t, ll = int(np.sum(d > 0)), int(np.sum(d == 0)), int(np.sum(d < 0))
    if np.all(d == 0):
        p = 1.0
    else:
        p = float(wilcoxon(x, y, zero_method="zsplit").pvalue)
    return f"{w}/{t}/{ll}", f"{p:.4f}", a12(x, y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--func", default="N07-Vincent2D")
    ap.add_argument("--ref", default=None,
                    help="label to test against (default: the first one given)")
    ap.add_argument("--metric", default="pr",
                    choices=("pr", "visited", "distinct", "landed", "hunts"),
                    help="which column the paired test is run on (default pr). "
                         "Entry 68's rejection condition is stated on `visited` "
                         "(the coverage ceiling), so the test has to follow the "
                         "column the question names, not always PR.")
    ap.add_argument("specs", nargs="+", metavar="LABEL=CSV")
    args = ap.parse_args()

    data = {}
    for spec in args.specs:
        label, _, path = spec.partition("=")
        data[label] = load(path, args.func)
    ref = args.ref or args.specs[0].split("=")[0]

    print(f"\n{args.func}   reference = {ref}   tested column = {args.metric}   "
          f"(paired Wilcoxon over shared seeds, A12 = share of seeds the row wins)")
    for eps in sorted(next(iter(data.values())), reverse=True):
        print(f"\n  eps = {eps:g}")
        print(f"    {'config':<18}{'hunts':>7}{'PR':>7}{'visited':>8}"
              f"{'dist':>7}{'land':>7}{'blk':>7}"
              f"{'w/t/l':>10}{'p':>9}{'A12':>7}{'seeds':>7}")
        for label, d in data.items():
            if eps not in d:
                continue
            seeds = sorted(set(d[eps]) & set(data[ref][eps]))
            pr = np.array([d[eps][s]["pr"] for s in seeds])
            tst = np.array([d[eps][s][args.metric] for s in seeds])
            rtst = np.array([data[ref][eps][s][args.metric] for s in seeds])
            hunts = np.mean([d[eps][s]["hunts"] for s in seeds])
            vis = np.mean([d[eps][s]["visited"] for s in seeds])
            dis = np.mean([d[eps][s]["distinct"] for s in seeds])
            lan = np.mean([d[eps][s]["landed"] for s in seeds])
            blk = np.mean([d[eps][s]["blocked"] for s in seeds])
            if label == ref:
                wtl, p, a = "--", "--", float("nan")
            else:
                wtl, p, a = paired(tst, rtst)
            print(f"    {label:<18}{hunts:>7.1f}{np.mean(pr):>7.3f}"
                  f"{vis:>8.2f}{dis:>7.1f}{lan:>7.1f}{blk:>7.1f}{wtl:>10}{p:>9}"
                  f"{'' if label == ref else f'{a:>7.2f}'}{len(seeds):>7}")


if __name__ == "__main__":
    main()
