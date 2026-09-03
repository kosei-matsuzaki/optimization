#!/usr/bin/env python3
"""Is there anything for a recovery phase to go back to?

Question 1 (allocation side) proposed re-entering the basins the run left
shallow, on the premise that the inventory sits in ``st.ir_basin_centroids``.
Before spending a measurement on the variant, this probe asks whether that
inventory exists at all, and where.

Base MC-ESO reports (``mceso.py:755``) the union of three sets:

  pop          the live population of the hunt in progress when the budget ends
  ir_archive   the strain reservoir (niche-separated elites, capped and
               best-f-trimmed at every spillover)
  sol_archive  the best point of every basin drilled and abandoned

Only points that are *distinct niches yet miss eps* (`blocked`, the column
entry 29 counted) are candidates for recovery, and only if they are still
recoverable -- a niche some later hunt drilled anyway costs nothing to abandon.
So this dumps, per run:

  blocked_<component>   rho-distinct points of that component missing eps
  disc_shallow          rho-distinct elites *discarded* at spillovers with
                        f > hunt_level_tol * f_init_scale, over the whole run --
                        the stream that never reaches any archive because the
                        ir_archive keep-loop is a best-f trim
  disc_lost             of those, the ones rho-distinct from every point in the
                        final reported set: niches the run entered, threw away,
                        and never came back to. **This is the recoverable
                        inventory**; if it is ~0 there is nothing to recover.

Everything is recording only -- the search is untouched, so these are base's own
numbers.

Usage:
  python3 scripts/blocked_inventory.py --funcs N06-Shubert2D --seeds 15 \
      --evals 20000 --eps 1e-1,1e-3,1e-5 --csv analysis/hm/inventory_n06.csv
"""
from __future__ import annotations
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME          # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer        # noqa: E402
from core.runner import _seed_indices                            # noqa: E402


class _DiscardProbe(MultiChannelEpidemicOptimizer):
    """Base MC-ESO, plus a tap on the elites discarded at each spillover.

    ``_on_spillover_start`` folds the population's niche elites into
    ``ir_archive`` and then trims that archive to the best ``n_elite_max`` by f.
    Everything the trim drops is gone. This records the population's niche
    elites *before* the fold, so the discarded stream can be reconstructed by
    differencing against what the archives ended up holding.
    """

    def _init_state(self, max_evals):
        st = super()._init_state(max_evals)
        self._st = st
        self.disc_x: list[np.ndarray] = []
        self.disc_f: list[float] = []
        return st

    def _on_spillover_start(self, st, basin_switch: bool) -> None:
        if len(st.pop_f):
            for i in self._niche_elites(st.pop_x, st.pop_f, st.niche_radius):
                self.disc_x.append(np.asarray(st.pop_x[i], dtype=float).copy())
                self.disc_f.append(float(st.pop_f[i]))
        return super()._on_spillover_start(st, basin_switch)


def _distinct(X: np.ndarray, F: np.ndarray, rho: float) -> np.ndarray:
    """Indices of the rho-greedy distinct set, best-f first (the CEC rule)."""
    if len(X) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(F)
    return order[_seed_indices(X[order], rho)]


def _blocked(X: np.ndarray, F: np.ndarray, rho: float, eps: float) -> int:
    """rho-distinct points of a set that miss eps -- what scores nothing."""
    idx = _distinct(X, F, rho)
    return int(np.sum(F[idx] > eps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--funcs", default="N06-Shubert2D,N07-Vincent2D")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--evals", type=int, default=20000)
    ap.add_argument("--eps", default="1e-1,1e-3,1e-5")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    eps_list = [float(s) for s in args.eps.split(",")]
    rows = []
    print(f"{'function':<16}{'eps':>8}{'blk_pop':>9}{'blk_ir':>8}{'blk_sol':>9}"
          f"{'disc_sh':>9}{'disc_lost':>11}{'deep_rep':>10}{'sh_bestf':>11}")
    print("-" * 92)
    for name in [s.strip() for s in args.funcs.split(",")]:
        b = NICHING_BENCHMARKS_BY_NAME[name]
        rho = b.niche_rho
        runs = []
        for seed in range(args.seeds):
            o = _DiscardProbe(b, seed=seed * 100)
            o.optimize(args.evals)
            st = o._st
            pop_x = np.asarray(st.pop_x, dtype=float)
            pop_f = np.asarray(st.pop_f, dtype=float)
            ir_x = np.asarray(st.ir_archive_x, dtype=float).reshape(-1, b.dim)
            ir_f = np.asarray(st.ir_archive_f, dtype=float)
            so_x = np.asarray(st.sol_archive_x, dtype=float).reshape(-1, b.dim)
            so_f = np.asarray(st.sol_archive_f, dtype=float)
            rep_x = np.vstack([pop_x, ir_x, so_x])
            rep_f = np.concatenate([pop_f, ir_f, so_f])
            dx = np.asarray(o.disc_x, dtype=float).reshape(-1, b.dim)
            df = np.asarray(o.disc_f, dtype=float)
            level = o.hunt_level_tol * st.f_init_scale
            runs.append((seed, pop_x, pop_f, ir_x, ir_f, so_x, so_f,
                         rep_x, rep_f, dx, df, level))

        for eps in eps_list:
            acc = []
            for (seed, pop_x, pop_f, ir_x, ir_f, so_x, so_f,
                 rep_x, rep_f, dx, df, level) in runs:
                # The discarded stream, shallow end only, de-duplicated.
                sh = df > max(level, eps)
                sx, sf = dx[sh], df[sh]
                sidx = _distinct(sx, sf, rho)
                # Lost = rho-distinct from every reported point, so no later hunt
                # covered that niche. This is what a recovery phase could add.
                lost = 0
                for i in sidx:
                    if len(rep_x) == 0 or np.min(
                            np.linalg.norm(rep_x - sx[i], axis=1)) > rho:
                        lost += 1
                ridx = _distinct(rep_x, rep_f, rho)
                row = [_blocked(pop_x, pop_f, rho, eps),
                       _blocked(ir_x, ir_f, rho, eps),
                       _blocked(so_x, so_f, rho, eps),
                       len(sidx), lost,
                       int(np.sum(rep_f[ridx] <= eps)),
                       float(sf.min()) if len(sf) else float("nan")]
                acc.append(row)
                rows.append([name, b.n_global_optima, eps, seed, args.evals] + row)
            m = np.nanmean(np.array(acc, dtype=float), axis=0)
            print(f"{name:<16}{eps:>8.0e}{m[0]:>9.1f}{m[1]:>8.1f}{m[2]:>9.1f}"
                  f"{m[3]:>9.1f}{m[4]:>11.1f}{m[5]:>10.1f}{m[6]:>11.2e}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["function", "K", "eps", "seed", "evals",
                        "blocked_pop", "blocked_ir", "blocked_sol",
                        "disc_shallow", "disc_lost", "deep_reported",
                        "disc_best_f"])
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")

    print("\ndisc_lost ~ 0 -> nothing to recover: every niche the run entered is "
          "either still\n              held in the reported set or was never "
          "distinct to begin with.")


if __name__ == "__main__":
    main()
