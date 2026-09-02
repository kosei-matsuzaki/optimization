#!/usr/bin/env python3
"""Read the per-hunt dump from ``diagnose_niching.py --hunt-csv`` and say which
of three things stops MC-ESO from descending the basins it grazes.

A "hunt" is one spillover cycle: the search settles in a basin and is then
released. The dump holds, for each hunt, how deep it got (``f``), why it was
released (``exhausted``: the sigma floor / level tolerance fired, vs. the plain
stagnation window), and whether its endpoint is a rho-separated basin nobody
else landed in (``distinct``).

The split this answers:

  (i)   hunts land deep but there are too few of them -> the number of hunts is
        the ceiling on peak ratio, whatever the descent does.
  (ii)  hunts land shallow with exhausted=1 -> the stopping rule releases the
        basin before the descent finishes.
  (ii') hunts land shallow with exhausted=0 -> the stagnation window cuts the
        hunt off mid-descent.
  (iii) endpoints duplicate (distinct << hunts) -> the descent is fine and the
        hunts keep re-landing in basins already held.

Usage:  python3 scripts/analyze_hunts.py analysis/hunts_*.csv
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict

import numpy as np

LEVELS = (1e-1, 1e-3, 1e-5)


def _load(paths: list[str]) -> dict[tuple[str, int, int], list[dict]]:
    """(function, K, seed) -> its hunts, in order."""
    out: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for p in paths:
        with open(p) as fh:
            for row in csv.DictReader(fh):
                out[(row["function"], int(row["K"]), int(row["seed"]))].append({
                    "eval": int(row["eval"]),
                    "f": float(row["f"]),
                    "switch": int(row["switch"]),
                    "exhausted": int(row["exhausted"]),
                    "sigma_span": float(row["sigma_span"]),
                    "distinct": int(row["distinct"]),
                })
    return out


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        raise SystemExit(2)
    runs = _load(paths)

    # ── per (function, seed): the shape of the hunt population ───────────────
    print("Per-run hunt yield.  'deep@e' = hunts whose endpoint reached f <= e.")
    print(f"{'function':<20}{'K':>5}{'seed':>5}{'hunts':>7}{'switch':>7}"
          f"{'exh':>6}{'distinct':>9}"
          + "".join(f"{f'deep@{e:g}':>11}" for e in LEVELS)
          + f"{'med f':>10}")
    print("-" * 110)
    agg: dict[tuple[str, int], list[list[float]]] = defaultdict(list)
    for (name, k, seed), hs in sorted(runs.items()):
        f = np.array([h["f"] for h in hs])
        deep = [int((f <= e).sum()) for e in LEVELS]
        row = [len(hs), sum(h["switch"] for h in hs), sum(h["exhausted"] for h in hs),
               sum(h["distinct"] for h in hs), *deep, float(np.median(f))]
        agg[(name, k)].append(row)
        print(f"{name:<20}{k:>5}{seed:>5}{row[0]:>7.0f}{row[1]:>7.0f}{row[2]:>6.0f}"
              f"{row[3]:>9.0f}" + "".join(f"{d:>11.0f}" for d in deep)
              + f"{row[7]:>10.1e}")

    # ── the ceiling argument, per function ──────────────────────────────────
    print("\nCeiling: even a perfect descent cannot report more distinct optima "
          "than it has\nhunts that ended in a fresh basin.  PR_ceiling = "
          "distinct_deep / K.")
    print(f"{'function':<20}{'K':>5}{'hunts':>7}{'distinct':>9}{'hunts/K':>9}"
          + "".join(f"{f'deep@{e:g}':>11}" for e in LEVELS)
          + f"{'exh frac':>10}")
    print("-" * 104)
    for (name, k), rows in sorted(agg.items()):
        m = np.mean(np.array(rows, dtype=float), axis=0)
        print(f"{name:<20}{k:>5}{m[0]:>7.1f}{m[3]:>9.1f}{m[0]/k:>9.2f}"
              + "".join(f"{m[4 + i]:>11.1f}" for i in range(len(LEVELS)))
              + f"{m[2]/max(m[0], 1e-9):>10.2f}")

    # ── why hunts that stayed shallow were released ─────────────────────────
    print("\nRelease reason vs. depth reached (all runs pooled per function).")
    print("A shallow hunt with exhausted=1 means the stopping rule let go of a "
          "basin the\ndescent had not finished; with exhausted=0 the stagnation "
          "window cut it off.")
    print(f"{'function':<20}{'depth band':>16}{'n':>6}{'exh=1':>7}{'exh=0':>7}"
          f"{'med sigma/span':>16}")
    print("-" * 72)
    for (name, k) in sorted(agg):
        hs = [h for (n2, _, _), lst in runs.items() if n2 == name for h in lst]
        bands = [("f <= 1e-5", lambda f: f <= 1e-5),
                 ("1e-5 < f <=1e-3", lambda f: 1e-5 < f <= 1e-3),
                 ("1e-3 < f <=1e-1", lambda f: 1e-3 < f <= 1e-1),
                 ("f > 1e-1", lambda f: f > 1e-1)]
        for label, pred in bands:
            sel = [h for h in hs if pred(h["f"])]
            if not sel:
                continue
            e1 = sum(h["exhausted"] for h in sel)
            ss = float(np.median([h["sigma_span"] for h in sel]))
            print(f"{name:<20}{label:>16}{len(sel):>6}{e1:>7}{len(sel) - e1:>7}"
                  f"{ss:>16.1e}")

    # ── how the budget is spent across hunts ────────────────────────────────
    print("\nEvaluations per hunt (gaps between consecutive spillovers).")
    print(f"{'function':<20}{'K':>5}{'first hunt':>12}{'median gap':>12}"
          f"{'max gap':>10}{'last spill @':>16}")
    print("-" * 76)
    for (name, k) in sorted(agg):
        firsts, gaps, tails = [], [], []
        for (n2, _, _), hs in runs.items():
            if n2 != name or not hs:
                continue
            ev = [h["eval"] for h in hs]
            firsts.append(ev[0])
            gaps.extend(np.diff(ev).tolist())
            tails.append(ev[-1])
        print(f"{name:<20}{k:>5}{np.mean(firsts):>12.0f}"
              f"{np.median(gaps) if gaps else float('nan'):>12.0f}"
              f"{max(gaps) if gaps else float('nan'):>10.0f}"
              f"{np.mean(tails):>16.0f}")


if __name__ == "__main__":
    main()
