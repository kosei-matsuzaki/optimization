#!/usr/bin/env python3
"""Does a forbidden region hurt a reported set by wiping all of it out at once?

Under a forbidden region the rank verdict and the mean verdict disagree on half
the cells: a diverse set wins most scenarios and loses enormously on the rest.
The proposed mechanism is that this model, unlike a tilt or an instance shift,
can delete every solution in the set at once, and a loss with nothing left to
fall back on has no ceiling.

That is directly checkable. scenario_value.py records, per scenario, whether
every point of the reported set fell inside the forbidden half-space. This asks
whether those wipeouts carry the losses, and whether a set's spatial spread is
what protects it -- across several sizes of the forbidden region, since a
mechanism that only appears at one size is not a mechanism.

Refuted if the wipeout rate does not fall as spread rises, or if the losses are
the same size whether or not the set was wiped out.

Usage:
  python3 scripts/wipeout.py analysis/wipe_cut*.csv [--props analysis/set_properties.csv]
"""
from __future__ import annotations
import argparse
import collections
import csv
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:                                   # pragma: no cover
    spearmanr = None

BASE = "quality"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path, nargs="+")
    ap.add_argument("--props", type=Path, default=Path("analysis/set_properties.csv"))
    args = ap.parse_args()

    spread: dict[tuple[str, str], float] = {}
    if args.props.exists():
        acc: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
        with open(args.props, newline="") as fh:
            for r in csv.DictReader(fh):
                acc[(r["function"], r["method"])].append(float(r["spread"]))
        spread = {k: float(np.mean(v)) for k, v in acc.items()}

    print("wipeout = every point of the reported set fell inside the forbidden region")
    print("Losses are in units of the function's own 75th-percentile baseline regret.\n")
    print(f"{'file':<22}{'wipe%':>7}{'loss|wiped':>12}{'loss|kept':>11}"
          f"{'rho(spread,wipe)':>19}")
    print("-" * 71)

    for path in args.rows:
        reg: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
        wipe: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
        base_reg: dict[str, list[float]] = collections.defaultdict(list)
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            if "wiped" not in (rd.fieldnames or []):
                print(f"{path.name:<22}(no wiped column; rerun with the patched script)")
                continue
            for r in rd:
                key = (r["function"], r["method"])
                reg[key].append(float(r["regret"]))
                wipe[key].append(int(r["wiped"]))
                if r["method"] == BASE:
                    base_reg[r["function"]].append(float(r["regret"]))

        scale = {fn: (float(np.percentile(v, 75)) or float(np.max(v)) or 1.0)
                 for fn, v in base_reg.items()}

        w_all, lw, lk = [], [], []
        rhos = []
        by_fn: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for k in reg:
            by_fn[k[0]].append(k)
        for fn, ks in by_fn.items():
            xs, ys = [], []
            for k in ks:
                w = np.array(wipe[k])
                e = np.array(reg[k]) / scale[fn]
                w_all.append(w)
                if w.any():
                    lw.append(e[w == 1])
                if (~w.astype(bool)).any():
                    lk.append(e[w == 0])
                if k in spread:
                    xs.append(spread[k])
                    ys.append(float(w.mean()))
            if spearmanr is not None and len(xs) >= 5 and np.ptp(xs) > 0 and np.ptp(ys) > 0:
                r = spearmanr(xs, ys).statistic
                if not np.isnan(r):
                    rhos.append(float(r))

        w_all = np.concatenate(w_all)
        lw = np.concatenate(lw) if lw else np.array([np.nan])
        lk = np.concatenate(lk) if lk else np.array([np.nan])
        rho = (f"{np.mean(rhos):>6.2f} ({sum(1 for r in rhos if (r > 0) == (np.mean(rhos) > 0))}"
               f"/{len(rhos)})" if rhos else "-")
        print(f"{path.name:<22}{100 * w_all.mean():>6.0f}%{np.median(lw):>12.2f}"
              f"{np.median(lk):>11.2f}{rho:>19}")

    print("\nrho is between a method's spread and its wipeout rate, taken within each")
    print("function and averaged; negative means spread protects. loss columns are")
    print("median excess regret in the two cases, so their gap is what a wipeout costs.")


if __name__ == "__main__":
    main()
