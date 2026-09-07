"""Make the stagnation counter basin-relative and see whether N18 gets deeper.

Entry 76 closed the previous question: on N18-CF3-10D every hunt after the first
is cut at exactly 1499 evaluations and is *still descending* when it is cut
(1.14 decades/1000 evals over its last quarter, 12/12 seeds). The mechanism is
that `no_improve` is only reset by an improvement of the **global** best
(`core/optimizers/mceso.py:1560-1571`), so once hunt 0 has drilled to 1e-12 no
later hunt can ever reset the counter, and `_spillover_should_fire`'s exhausted
branch (`no_improve >= window`, window = 1500 at dim 10) chops each of them into
exactly one window. Entry 76 measured the shortfall at 694 evaluations.

This is the one diagnostic arm that question 1 pre-registers: reset `no_improve`
when the **basin** best improves, using the same log-slope gate the global path
uses (`_meaningful_improvement`), and change nothing else. `core/` is untouched;
the variant lives here and the defaults ship unchanged.

  H1  hunts run past 1499 evals and the extra ~700 evals carry the descent
      across 1e-3, so PR@1e-3 / 1e-5 rise.
  Refuted if hunt length grows but PR@1e-3 / 1e-5 does not move over 12 seeds
      -> depth trades against hunt *count*, not against the budget inside a
      hunt, and N18's depth is not reachable by re-allocating the budget.

Coverage has to be reported alongside: longer hunts mean fewer hunts, so the
distinct side can pay for the depth. Both arms are scored identically, with the
official scorer (`core.runner._niching_counts`) and with attribution to the six
true optima the suite ships (entry 76: the scored PR@1e-1 of 1.000 is a rho
artefact, the true coverage is 4/6).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer           # noqa: E402
from core.runner import _niching_counts                             # noqa: E402

EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)


class BasinResetMCESO(MultiChannelEpidemicOptimizer):
    """`no_improve` is reset by the basin's own progress, not only the global's.

    `basin_reset=False` disables the override and must reproduce base bit for
    bit (identity check, entry 74's seed 0: best_f = 6.22882e-12, n_rep = 246).
    """

    def __init__(self, *a, basin_reset: bool = True, **kw) -> None:
        super().__init__(*a, **kw)
        self.basin_reset = bool(basin_reset)

    def _init_state(self, max_evals):
        st = super()._init_state(max_evals)
        self._b_ref = math.log10(max(st.basin_best, 1e-300))
        self._b_since = 0
        self._b_seen = st.basin_best
        self._n_basin_resets = 0
        return st

    def _record_eval(self, st, x, f, sigma_used) -> None:
        bb_pre = st.basin_best
        ni_pre = st.no_improve
        super()._record_eval(st, x, f, sigma_used)
        if not self.basin_reset:
            return
        if st.basin_best > self._b_seen:
            #  A re-seed raised basin_best: a new hunt starts here.
            self._b_ref = math.log10(max(st.basin_best, 1e-300))
            self._b_since = 0
        elif st.no_improve == 0 and ni_pre != 0:
            #  The global path already reset the counter; re-anchor so the two
            #  references cannot drift apart.
            self._b_ref = math.log10(max(st.basin_best, 1e-300))
            self._b_since = 0
        else:
            self._b_since += 1
            if f < bb_pre and self._meaningful_improvement(
                    f, self._b_ref, self._b_since):
                #  Only the stagnation counter moves. `log_best_ref` and
                #  `evals_since_reset` belong to the global path and are left
                #  alone, so this arm changes exactly one thing.
                st.no_improve = 0
                self._b_ref = math.log10(max(f, 1e-300))
                self._b_since = 0
                self._n_basin_resets += 1
        self._b_seen = st.basin_best


class _HuntTap:
    """Read-only tap on hunt boundaries and basin improvements (entry 76's)."""

    def _init_state(self, max_evals):
        st = super()._init_state(max_evals)
        self.hunts: list[dict] = []
        self._cur: dict | None = None
        self._pending = True
        return st

    def _open(self, st, bb0: float) -> None:
        self._cur = {"start": len(st.history_f), "bb0": float(bb0),
                     "ev": [], "bb": [], "bx": None}
        self._pending = False

    def _close(self, st, reason: str) -> None:
        if self._cur is None:
            return
        c = self._cur
        c["end"] = len(st.history_f)
        c["reason"] = reason
        c["bb_end"] = float(st.basin_best)
        self.hunts.append(c)
        self._cur = None
        self._pending = True

    def _record_eval(self, st, x, f, sigma_used) -> None:
        bb_pre = st.basin_best
        super()._record_eval(st, x, f, sigma_used)
        if self._pending:
            self._open(st, bb_pre)
        if st.basin_best < bb_pre:
            self._cur["ev"].append(len(st.history_f) - self._cur["start"])
            self._cur["bb"].append(float(st.basin_best))
            self._cur["bx"] = np.asarray(x, dtype=float).copy()

    def _spillover_should_fire(self, st) -> bool:
        fire = super()._spillover_should_fire(st)
        if fire:
            self._close(st, "release")
        return fire


class Probe(_HuntTap, BasinResetMCESO):
    pass


def _attribute(x, opt: np.ndarray) -> tuple[int, float]:
    if x is None:
        return -1, float("nan")
    d = np.linalg.norm(opt - np.asarray(x, dtype=float)[None, :], axis=1)
    j = int(np.argmin(d))
    return j, float(d[j])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="N18-CF3-10D")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--arm", choices=("base", "basin"), default="basin")
    ap.add_argument("--out", required=True, help="prefix for the CSVs")
    a = ap.parse_args()

    b = NICHING_BENCHMARKS_BY_NAME[a.func]
    ev_budget = a.evals or int(b.suite_max_evals)
    opt = np.asarray(b.optima_pos, dtype=float)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hf = open(f"{out}_hunts.csv", "w", newline="")
    hw = csv.writer(hf)
    hw.writerow(["arm", "seed", "hunt", "start", "n_evals", "reason", "bb0",
                 "bb_end", "n_improve", "last_improve_ev", "gopt", "dist"])
    pf = open(f"{out}_pr.csv", "w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["arm", "seed", "best_f", "n_rep", "n_hunts", "n_basin_resets",
                 "median_hunt_len", "wall_s"]
                + [f"pr_{e:g}" for e in EPS]
                + [f"prtrue_{e:g}" for e in EPS])

    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        t0 = time.time()
        o = Probe(b, seed=s * 100, basin_reset=(a.arm == "basin"))
        r = o.optimize(ev_budget)
        if o._cur is not None:
            o._cur["end"] = len(r.history_f)
            o._cur["reason"] = "budget"
            o._cur["bb_end"] = float(min(o._cur["bb"])) if o._cur["bb"] else float("nan")
            o.hunts.append(o._cur)
            o._cur = None

        lens = []
        for i, c in enumerate(o.hunts):
            n = c["end"] - c["start"]
            lens.append(n)
            j, d = _attribute(c["bx"], opt)
            last = float(c["ev"][-1]) if c["ev"] else 0.0
            hw.writerow([a.arm, s, i, c["start"], n, c["reason"],
                         f"{c['bb0']:.6g}", f"{c['bb_end']:.6g}", len(c["ev"]),
                         f"{last:.0f}", j, f"{d:.4g}"])

        #  Official scoring, exactly as core.runner does it (cap + rho-greedy).
        counts, n_rep = _niching_counts([r], b, EPS)
        pr = [counts[0, j] / b.n_global_optima for j in range(len(EPS))]

        #  True coverage: distinct real optima among the *scored* set.
        X = np.asarray(r.final_solutions, dtype=float)
        F = np.array([float(b.func(x)) for x in X])
        cap = max(100, 2 * b.n_global_optima)
        if len(F) > cap:
            keep = np.argsort(F)[:cap]
            X, F = X[keep], F[keep]
        prt = []
        for e in EPS:
            got = {(_attribute(X[k], opt)[0]) for k in range(len(X)) if F[k] <= e}
            prt.append(len(got) / b.n_global_optima)

        pw.writerow([a.arm, s, f"{r.best_f:.6g}", n_rep[0], len(o.hunts),
                     o._n_basin_resets if a.arm == "basin" else 0,
                     f"{np.median(lens):.0f}", f"{time.time() - t0:.1f}"]
                    + [f"{v:.6g}" for v in pr] + [f"{v:.6g}" for v in prt])
        pf.flush()
        hf.flush()
        print(f"{a.arm} seed={s} hunts={len(o.hunts)} "
              f"median_len={np.median(lens):.0f} best_f={r.best_f:.6g} "
              f"n_rep={n_rep[0]} PR@1e-3={pr[2]:.3f} PR@1e-5={pr[4]:.3f} "
              f"wall={time.time() - t0:.1f}s", flush=True)

    hf.close()
    pf.close()


if __name__ == "__main__":
    main()
