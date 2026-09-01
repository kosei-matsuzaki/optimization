#!/usr/bin/env python3
"""Why does MC-ESO's peak ratio saturate? Found-vs-reported diagnosis.

Peak ratio only scores the solutions a run *reports* (final population plus
archives). If a run visits an optimum, walks away and never records it, the
optimum is invisible to the metric. This script separates the two:

  visited   distinct global optima the run touched at accuracy eps, counted over
            the whole evaluation history with the CEC2013 rho rule
  reported  the same count over result.final_solutions, i.e. what PR sees
  distinct  how many rho-separated points the reported set holds at all,
            regardless of accuracy — the duplicate-report check

visited >> reported means the search finds optima and forgets them (a recording
problem). visited ~= reported means it genuinely stops finding new ones (a
search problem). Spillover / basin-switch / exhaustion counters come along so
the restart loop can be read at the same time.

Usage:
  python3 scripts/diagnose_niching.py [--evals 25000] [--seeds 5]
                                      [--funcs N06-Shubert2D,...]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME          # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer        # noqa: E402
from core.runner import _seed_indices, count_goptima             # noqa: E402


class _CountingMCESO(MultiChannelEpidemicOptimizer):
    """MC-ESO with counters on the restart hooks. Behaviour is unchanged: every
    override calls super() and only tallies."""

    def optimize(self, max_evals: int = 5000):
        self.n_spillover = 0
        self.n_basin_switch = 0
        self.n_exhausted = 0
        self.hunts: list[tuple[int, float]] = []   # (eval index, basin best f)
        return super().optimize(max_evals)

    def _basin_exhausted(self, st) -> bool:
        out = super()._basin_exhausted(st)
        self.n_exhausted += int(bool(out))
        return out

    def _on_spillover_start(self, st, basin_switch: bool) -> None:
        self.n_spillover += 1
        self.n_basin_switch += int(bool(basin_switch))
        # What the hunt that is ending here achieved, and when.
        self.hunts.append((len(st.history_f), float(min(st.pop_f))))
        return super()._on_spillover_start(st, basin_switch)


def _distinct_points(X: np.ndarray, F: np.ndarray, rho: float) -> int:
    """rho-separated points in a set, ignoring accuracy (duplicate check)."""
    if len(X) == 0:
        return 0
    order = np.argsort(F)
    return len(_seed_indices(X[order], rho))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", type=int, default=25000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--variant", type=str, default="base",
                    help="base | localwin (post-exhaustion pacing on the basin)")
    ap.add_argument("--funcs", type=str,
                    default="N04-Himmelblau,N06-Shubert2D,N07-Vincent2D,N10-ModRastrigin2D")
    args = ap.parse_args()

    print(f"MC-ESO niching diagnosis   evals={args.evals}  seeds={args.seeds}  "
          f"eps={args.eps:g}")
    print(f"{'function':<20}{'K':>4}{'visited':>9}{'reported':>9}{'distinct':>9}"
          f"{'|rep|':>7}{'spill':>7}{'switch':>7}{'hunts':>7}{'landed':>7}"
          f"{'yield':>7}{'best_f':>11}")
    print("-" * 104)
    for name in [s.strip() for s in args.funcs.split(",")]:
        b = NICHING_BENCHMARKS_BY_NAME[name]
        rows = []
        for seed in range(args.seeds):
            kw = {"localwin": {"exhausted_local_window": True},
                  "fast": {"hunt_no_improve_mult": 0.5},
                  "base": {}}[args.variant]
            opt = _CountingMCESO(b, seed=seed * 100, **kw)
            r = opt.optimize(args.evals)

            hx = np.asarray(r.history_x, dtype=float)
            hf = np.asarray(r.history_f, dtype=float)
            visited = count_goptima(hx, hf, b.n_global_optima, b.niche_rho, args.eps)

            sx = np.asarray(r.final_solutions or [r.best_x], dtype=float)
            sf = np.array([float(b.func(x)) for x in sx])
            reported = count_goptima(sx, sf, b.n_global_optima, b.niche_rho, args.eps)

            # Hunt yield: a hunt "landed" if the basin it abandoned was within
            # eps of the global value — i.e. the restart cycle actually produced
            # a solution rather than being cut off mid-descent.
            hunts = opt.hunts
            landed = sum(1 for _, f in hunts if f <= args.eps)
            rows.append((visited, reported, _distinct_points(sx, sf, b.niche_rho),
                         len(sx), opt.n_spillover, opt.n_basin_switch,
                         len(hunts), landed, r.best_f))
        m = np.mean(np.array(rows, dtype=float), axis=0)
        y = m[7] / m[6] if m[6] else float("nan")
        print(f"{name:<20}{b.n_global_optima:>4}{m[0]:>9.1f}{m[1]:>9.1f}{m[2]:>9.1f}"
              f"{m[3]:>7.0f}{m[4]:>7.1f}{m[5]:>7.1f}{m[6]:>7.1f}{m[7]:>7.1f}"
              f"{y:>7.2f}{m[8]:>11.1e}")

    print("\nvisited >> reported -> optima are found and then dropped from the "
          "reported set (a recording problem).")
    print("visited ~= reported -> the search itself stops finding new optima.")
    print("distinct << |rep|   -> the reported set is mostly duplicates of the "
          "same basins.")


if __name__ == "__main__":
    main()
