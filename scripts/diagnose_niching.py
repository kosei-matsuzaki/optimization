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
        self._last_exhausted = False
        self.hunts: list[tuple[int, float]] = []   # (eval index, basin best f)
        # One row per hunt: where it ended, how deep it got, where, and *why* it
        # was released. `exhausted` is read from the cached value of the last
        # `_basin_exhausted` call (the spillover path always evaluates it on this
        # same state just before this hook), so nothing is re-evaluated here.
        self.hunt_rows: list[dict] = []
        return super().optimize(max_evals)

    def _basin_exhausted(self, st) -> bool:
        out = super()._basin_exhausted(st)
        self.n_exhausted += int(bool(out))
        self._last_exhausted = bool(out)
        return out

    def _on_spillover_start(self, st, basin_switch: bool) -> None:
        self.n_spillover += 1
        self.n_basin_switch += int(bool(basin_switch))
        # What the hunt that is ending here achieved, and when.
        best_i = int(np.argmin(st.pop_f))
        self.hunts.append((len(st.history_f), float(min(st.pop_f))))
        self.hunt_rows.append({
            "eval": len(st.history_f),
            "f": float(st.pop_f[best_i]),
            "x": np.asarray(st.pop_x[best_i], dtype=float).copy(),
            "switch": bool(basin_switch),
            "exhausted": bool(self._last_exhausted),
            "sigma_span": float(st.sigma) / float(st.span),
            "no_improve": int(st.no_improve),
        })
        return super()._on_spillover_start(st, basin_switch)


def _distinct_points(X: np.ndarray, F: np.ndarray, rho: float) -> int:
    """rho-separated points in a set, ignoring accuracy (duplicate check)."""
    if len(X) == 0:
        return 0
    order = np.argsort(F)
    return len(_seed_indices(X[order], rho))


def _greedy_seeds_capped(X: np.ndarray, rho: float, cap: int) -> np.ndarray:
    """The first ``cap`` seeds of the CEC rho-greedy rule over ``X`` (already
    sorted best-f first), vectorised and stopped early.

    Identical to ``_seed_indices(X, rho)[:cap]`` — greedy selection is prefix
    stable, so stopping once ``cap`` seeds are held cannot change the ones
    already kept. The early stop is what makes this usable on a 40k-point
    history: the reference loop keeps every rho-separated point and so grows to
    thousands of seeds on Vincent3D.
    """
    if len(X) == 0 or cap <= 0:
        return np.zeros(0, dtype=int)
    kept_idx: list[int] = [0]
    kept = [X[0]]
    for i in range(1, len(X)):
        if len(kept_idx) >= cap:
            break
        d = np.linalg.norm(np.asarray(kept) - X[i], axis=1)
        if d.min() > rho:
            kept_idx.append(i)
            kept.append(X[i])
    return np.asarray(kept_idx[:cap], dtype=int)


def reselect_from_history(hx: np.ndarray, hf: np.ndarray, rho: float,
                          cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a *legal* reported set from the evaluation history, zero extra
    evaluations.

    Sort every point the run evaluated best-f first, walk it with the same
    rho-greedy rule the scorer uses, and keep at most ``cap = max(100, 2K)``
    points — the cap ``core.runner._niching_counts`` enforces. Reporting the
    whole history is not a legal answer (it would reward dense sampling); this
    is a selection rule of the size the competition allows, so any method could
    adopt it as its output rule.
    """
    if len(hx) == 0:
        return hx, hf
    order = np.argsort(hf)
    sx, sf = hx[order], hf[order]
    keep = _greedy_seeds_capped(sx, rho, cap)
    return sx[keep], sf[keep]


def _cap_by_f(X: np.ndarray, F: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """The best-f trim ``core.runner._niching_counts`` applies before scoring."""
    if len(F) <= cap:
        return X, F
    keep = np.argsort(F)[:cap]
    return X[keep], F[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", type=int, default=25000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps", type=str, default="1e-4",
                    help="one accuracy, or a comma list. Several values are all "
                         "scored off the *same* runs, so relaxing the accuracy "
                         "costs no extra evaluations.")
    ap.add_argument("--variant", type=str, default="base",
                    help="base | localwin (post-exhaustion pacing on the basin)")
    ap.add_argument("--funcs", type=str,
                    default="N04-Himmelblau,N06-Shubert2D,N07-Vincent2D,N10-ModRastrigin2D")
    ap.add_argument("--csv", type=str, default=None,
                    help="write the per-(function, seed, eps) rows here")
    ap.add_argument("--hunt-csv", type=str, default=None,
                    help="write one row per hunt (spillover event) here: how deep "
                         "it got, where, and why it was released. Splits 'the "
                         "search never reaches the basin' from 'it reaches it and "
                         "is cut off mid-descent'.")
    args = ap.parse_args()

    eps_list = [float(s) for s in args.eps.split(",")]

    print(f"MC-ESO niching diagnosis   evals={args.evals}  seeds={args.seeds}  "
          f"eps={','.join(f'{e:g}' for e in eps_list)}")
    print(f"{'function':<20}{'K':>4}{'eps':>8}{'visited':>9}{'reported':>9}"
          f"{'resel':>8}{'|res|':>8}{'PR_now':>8}{'PR_res':>8}{'distinct':>9}"
          f"{'blocked':>8}{'|rep|':>7}{'spill':>7}{'hunts':>7}{'best_f':>11}")
    print("-" * 132)
    csv_rows = []
    hunt_rows: list[list] = []
    for name in [s.strip() for s in args.funcs.split(",")]:
        b = NICHING_BENCHMARKS_BY_NAME[name]
        cap = max(100, 2 * b.n_global_optima)   # core.runner._niching_counts
        # One run per seed; every accuracy is scored off these same runs.
        runs = []
        for seed in range(args.seeds):
            kw = {"localwin": {"exhausted_local_window": True},
                  "fast": {"hunt_no_improve_mult": 0.5},
                  "base": {}}[args.variant]
            opt = _CountingMCESO(b, seed=seed * 100, **kw)
            r = opt.optimize(args.evals)

            hx = np.asarray(r.history_x, dtype=float)
            hf = np.asarray(r.history_f, dtype=float)
            sx = np.asarray(r.final_solutions or [r.best_x], dtype=float)
            sf = np.array([float(b.func(x)) for x in sx])
            # The scorer trims the reported set to the cap before counting;
            # do the same here so `reported` is exactly what PR sees.
            sx, sf = _cap_by_f(sx, sf, cap)
            # The reselected set depends only on rho and the cap, not on eps,
            # so build it once and score it at every accuracy.
            rx, rf = reselect_from_history(hx, hf, b.niche_rho, cap)
            runs.append((seed, hx, hf, sx, sf, rx, rf, opt, r))

            if args.hunt_csv:
                # rho-greedy over the hunt endpoints, best-f first: how many
                # *distinct* basins the hunts actually ended in. Duplication here
                # is the question's rejection condition (descent is fine, the
                # hunts keep landing in basins already held).
                hr = opt.hunt_rows
                if hr:
                    hxs = np.array([h["x"] for h in hr], dtype=float)
                    hfs = np.array([h["f"] for h in hr], dtype=float)
                    order = np.argsort(hfs)
                    keep = set(int(order[i]) for i in
                               _seed_indices(hxs[order], b.niche_rho))
                    for j, h in enumerate(hr):
                        hunt_rows.append([name, b.n_global_optima, seed, j,
                                          h["eval"], h["f"], int(h["switch"]),
                                          int(h["exhausted"]), h["sigma_span"],
                                          h["no_improve"], int(j in keep),
                                          *[f"{v:.10g}" for v in h["x"]]])

        for eps in eps_list:
            rows = []
            for seed, hx, hf, sx, sf, rx, rf, opt, r in runs:
                visited = count_goptima(hx, hf, b.n_global_optima, b.niche_rho, eps)
                reported = count_goptima(sx, sf, b.n_global_optima, b.niche_rho, eps)
                # Same scorer, same cap, on the set reselected from history.
                resel = count_goptima(rx, rf, b.n_global_optima, b.niche_rho, eps)
                # rho-separated points in the reported set regardless of accuracy;
                # `blocked` is how many of those niches are held by a point that
                # misses eps, which is what count_goptima refuses to score.
                distinct = _distinct_points(sx, sf, b.niche_rho)

                # Hunt yield: a hunt "landed" if the basin it abandoned was within
                # eps of the global value — i.e. the restart cycle actually produced
                # a solution rather than being cut off mid-descent.
                hunts = opt.hunts
                landed = sum(1 for _, f in hunts if f <= eps)
                rows.append((visited, reported, resel, len(rx), distinct,
                             distinct - reported, len(sx), opt.n_spillover,
                             opt.n_basin_switch, len(hunts), landed, r.best_f))
                csv_rows.append([name, b.n_global_optima, eps, seed, args.evals,
                                 cap, *rows[-1]])
            m = np.mean(np.array(rows, dtype=float), axis=0)
            k = b.n_global_optima
            print(f"{name:<20}{k:>4}{eps:>8.0e}{m[0]:>9.1f}{m[1]:>9.1f}"
                  f"{m[2]:>8.1f}{m[3]:>8.1f}{m[1]/k:>8.2f}{m[2]/k:>8.2f}"
                  f"{m[4]:>9.1f}{m[5]:>8.1f}{m[6]:>7.0f}{m[7]:>7.1f}"
                  f"{m[8]:>7.1f}{m[11]:>11.1e}")

    if args.csv:
        import csv as _csv
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["function", "K", "eps", "seed", "evals", "cap",
                        "visited", "reported", "resel", "n_resel_pts",
                        "distinct", "blocked", "n_reported_pts",
                        "spillover", "basin_switch", "hunts", "landed", "best_f"])
            w.writerows(csv_rows)
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")

    if args.hunt_csv:
        import csv as _csv
        Path(args.hunt_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.hunt_csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            ndim = max((len(r) - 11 for r in hunt_rows), default=0)
            w.writerow(["function", "K", "seed", "hunt", "eval", "f", "switch",
                        "exhausted", "sigma_span", "no_improve", "distinct"]
                       + [f"x{i}" for i in range(ndim)])
            w.writerows(hunt_rows)
        print(f"wrote {args.hunt_csv} ({len(hunt_rows)} hunts)")

    print("\nvisited >> reported -> optima are found and then dropped from the "
          "reported set (a recording problem).")
    print("visited ~= reported -> the search itself stops finding new optima.")
    print("distinct << |rep|   -> the reported set is mostly duplicates of the "
          "same basins.")
    print("blocked > 0        -> reported niches are held by points that miss "
          "eps, so they score nothing.")
    print("PR_res >> PR_now   -> the loss is in the reporting rule: the same run, "
          "rescored off a legal\n                      cap-sized set reselected from "
          "its own history, is worth this much more.")
    print("|res| < cap        -> the history held fewer rho-separated points than "
          "the cap allows, so the\n                      cap is not what limits "
          "the reselected set.")


if __name__ == "__main__":
    main()
