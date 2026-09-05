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

Usage:  python3 scripts/analyze_hunts.py analysis/hm/hunts_*.csv
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVELS = (1e-1, 1e-3, 1e-5)


def _open(path: str):
    """Row-level dumps are stored gzipped (the repository rule); read both."""
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def _load(paths: list[str]) -> dict[tuple[str, int, int], list[dict]]:
    """(function, K, seed) -> its hunts, in order."""
    out: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for p in paths:
        with _open(p) as fh:
            for row in csv.DictReader(fh):
                out[(row["function"], int(row["K"]), int(row["seed"]))].append({
                    "eval": int(row["eval"]),
                    "f": float(row["f"]),
                    "switch": int(row["switch"]),
                    "exhausted": int(row["exhausted"]),
                    "sigma_span": float(row["sigma_span"]),
                    "distinct": int(row["distinct"]),
                    # endpoint coordinates, when the dump carries them (the
                    # timeline mode needs them; the older summaries do not)
                    **({"x": np.array([float(row[k]) for k in row
                                       if k.startswith("x") and k[1:].isdigit()])}
                       if any(k.startswith("x") and k[1:].isdigit() for k in row)
                       else {}),
                })
    return out


def _rho(name: str) -> float:
    """The scorer's niche radius for this function."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.benchmarks import NICHING_BENCHMARKS_BY_NAME  # noqa: E402
    return float(NICHING_BENCHMARKS_BY_NAME[name].niche_rho)


def _K(name: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.benchmarks import NICHING_BENCHMARKS_BY_NAME  # noqa: E402
    return int(NICHING_BENCHMARKS_BY_NAME[name].n_global_optima)


def timeline(runs: dict, budget: int) -> None:
    """Does the run stop landing in *new* basins before the budget ends?

    The `distinct` column in the dump is an offline, best-f-first greedy over
    the whole run, so it cannot say *when* a basin was first reached. This walks
    the hunts in evaluation order instead and applies the same rho-greedy rule
    incrementally: a hunt opens a new niche if its endpoint is deeper than eps
    and more than rho away from every endpoint already held. The cumulative
    count is then the coverage the run has actually built by that point in the
    budget — an upper bound on what any reporting rule could score from it.

    Read it against the rejection condition: coverage that is still climbing at
    the end of the budget means the ceiling is not coverage.
    """
    fracs = (0.1, 0.25, 0.5, 0.75, 1.0)
    for name in sorted({n for n, _, _ in runs}):
        rho, k = _rho(name), _K(name)
        print(f"\n=== {name}  K={k}  rho={rho}  budget={budget}  "
              f"({len({s for n, _, s in runs if n == name})} seeds) ===")
        print("Cumulative distinct basins landed in, by fraction of the budget "
              "(time-ordered rho-greedy\nover hunt endpoints). 'hunts' is how "
              "many hunts had ended by then.")
        for eps in LEVELS:
            per_seed = {}
            for (n2, _, seed), hs in sorted(runs.items()):
                if n2 != name:
                    continue
                hs = sorted(hs, key=lambda h: h["eval"])
                held: list[np.ndarray] = []
                # (eval at which a new niche opened)
                opened: list[int] = []
                nh: list[int] = []
                for h in hs:
                    nh.append(h["eval"])
                    if h["f"] > eps or "x" not in h:
                        continue
                    x = h["x"]
                    if not held or min(float(np.linalg.norm(y - x))
                                       for y in held) > rho:
                        held.append(x)
                        opened.append(h["eval"])
                per_seed[seed] = (np.array(opened), np.array(nh))
            print(f"\n  eps = {eps:g}")
            print(f"    {'seed':>5}" + "".join(f"{f'@{f:.0%}':>9}" for f in fracs)
                  + f"{'hunts':>8}{'new 2nd half':>14}{'last new @':>12}")
            rows = []
            for seed, (opened, nh) in sorted(per_seed.items()):
                cums = [int((opened <= budget * f).sum()) for f in fracs]
                half = int((opened <= budget * 0.5).sum())
                rows.append(cums + [len(nh), cums[-1] - half,
                                    int(opened.max()) if len(opened) else 0])
                print(f"    {seed:>5}" + "".join(f"{c:>9d}" for c in cums)
                      + f"{len(nh):>8d}{cums[-1] - half:>14d}"
                      + f"{rows[-1][-1]:>12d}")
            a = np.array(rows, dtype=float)
            m = a.mean(axis=0)
            print(f"    {'mean':>5}" + "".join(f"{v:>9.1f}" for v in m[:len(fracs)])
                  + f"{m[len(fracs)]:>8.1f}{m[len(fracs) + 1]:>14.1f}"
                  + f"{m[-1]:>12.0f}")
            first = a[:, 2]                          # opened in the first half
            second = a[:, len(fracs) - 1] - a[:, 2]  # opened in the second half
            try:
                from scipy.stats import wilcoxon
                w = wilcoxon(first, second)
                p = float(w.pvalue)
            except Exception:
                p = float("nan")
            wins = int((second < first).sum())
            print(f"    first half {first.mean():.2f} new basins vs second half "
                  f"{second.mean():.2f}  "
                  f"({wins}/{len(first)} seeds slower in the 2nd half, "
                  f"paired Wilcoxon p = {p:.2g})")
            print(f"    PR ceiling from coverage = {m[len(fracs) - 1] / k:.3f} "
                  f"of K = {k}")


def main() -> None:
    args = sys.argv[1:]
    tl = "--timeline" in args
    budget = 0
    if "--budget" in args:
        budget = int(args[args.index("--budget") + 1])
        args = [a for i, a in enumerate(args)
                if a != "--budget" and args[i - 1] != "--budget"]
    paths = [a for a in args if not a.startswith("--")]
    if not paths:
        print(__doc__)
        raise SystemExit(2)
    runs = _load(paths)
    if tl:
        if not budget:
            budget = max(h["eval"] for hs in runs.values() for h in hs)
        timeline(runs, budget)
        return

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
