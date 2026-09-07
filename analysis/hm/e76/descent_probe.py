"""How does `basin_best` descend *inside* a hunt on N18-CF3-10D?

Entry 75 closed the release-rule line on this function: all three depth knobs
(the level clause and the two sigma clauses) leave PR@1e-4 / 1e-5 bit-identical
over 12 seeds, and `_fl08` was shown to fire (hunts -23%) without moving the
depth at which hunts are released (median basin_best 0.1125 -> 0.1122). The
reading was "the sigma clause is a timer that runs after progress has already
stopped", i.e. the descent itself stops at f ~ 0.1.

That reading was inferred from release-time snapshots only. This tap measures
the descent directly: per hunt, the running `basin_best` against evaluations
since the hunt started, plus which true global optimum (the suite ships all six
coordinates) the hunt's best point actually sits next to.

Two shapes are pre-registered (research_loop question 1):
  (i)  the curve flattens and asymptotes near 0.1  -> the basin cannot be
       descended with the isotropic sigma this optimiser uses (CF3 composes
       rotated, differently conditioned components), and entry 75 stands;
  (ii) the curve is still descending when the hunt is cut -> the release is
       premature after all and entry 75's "timer" reading has to be withdrawn.

The tap is read-only: `_spillover_should_fire` and `_record_eval` are called
exactly as often as in the shipped optimiser and their return values are
untouched, so the search is bit-identical to base (checked against entry 75's
`best_f` / `n_rep` for seed 0).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer           # noqa: E402
from core.runner import _seed_indices                               # noqa: E402


class _DescentProbe(MultiChannelEpidemicOptimizer):
    """The shipped optimiser, tapped on hunt boundaries and on improvements."""

    def _init_state(self, max_evals):
        st = super()._init_state(max_evals)
        self.hunts: list[dict] = []
        self._cur: dict | None = None
        self._pending = True          # first hunt starts at the first offspring
        return st

    # -- hunt bookkeeping ---------------------------------------------------
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

    # -- taps (no behaviour change) ----------------------------------------
    def _record_eval(self, st, x, f, sigma_used) -> None:
        bb_pre = st.basin_best
        super()._record_eval(st, x, f, sigma_used)
        if self._pending:
            #  A fresh hunt: the re-seed loop already reset basin_best to
            #  min(pop_f), which is `bb_pre` as seen by this first offspring.
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


def _attribute(x: np.ndarray | None, opt: np.ndarray) -> tuple[int, float]:
    """Nearest true global optimum and the distance to it (-1 / nan if no x)."""
    if x is None:
        return -1, float("nan")
    d = np.linalg.norm(opt - x[None, :], axis=1)
    j = int(np.argmin(d))
    return j, float(d[j])


def _slope(ev: np.ndarray, bb: np.ndarray, lo: float, hi: float,
           bb0: float = float("nan")) -> float:
    """Decades of `basin_best` lost per 1000 evals over the [lo, hi] fraction of
    a hunt's length. Uses the running-min curve, so it is monotone by
    construction and the slope is >= 0."""
    if len(ev) == 0:
        return 0.0
    n = ev[-1]
    a, b = lo * n, hi * n
    if b - a <= 0:
        return 0.0
    #  running-min value at an arbitrary eval offset
    def val(t: float) -> float:
        k = int(np.searchsorted(ev, t, side="right")) - 1
        return bb[k] if k >= 0 else bb0
    va, vb = val(a), val(b)
    if not np.isfinite(va) or not np.isfinite(vb) or va <= 0 or vb <= 0:
        return float("nan")
    return float((np.log10(va) - np.log10(vb)) / (b - a) * 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="N18-CF3-10D")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--out", required=True, help="prefix for the two CSVs")
    ap.add_argument("--curve-eps", type=float, default=1e-1,
                    help="dump the full curve for hunts that reach this depth")
    a = ap.parse_args()

    b = NICHING_BENCHMARKS_BY_NAME[a.func]
    ev_budget = a.evals or int(b.suite_max_evals)
    opt = np.asarray(b.optima_pos, dtype=float)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hf = open(f"{out}_hunts.csv", "w", newline="")
    hw = csv.writer(hf)
    hw.writerow(["seed", "hunt", "start", "n_evals", "reason", "bb0", "bb_end",
                 "n_improve", "last_improve_ev", "tail_frac_idle",
                 "slope_first50", "slope_last25", "gopt", "dist"])
    cf = gzip.open(f"{out}_curves.csv.gz", "wt", newline="")
    cw = csv.writer(cf)
    cw.writerow(["seed", "hunt", "ev_since_start", "basin_best"])
    rf = open(f"{out}_reported.csv", "w", newline="")
    rw = csv.writer(rf)
    rw.writerow(["seed", "rank", "f", "gopt", "dist", "is_seed", "eps_1e-1",
                 "eps_1e-3", "eps_1e-5"])

    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        t0 = time.time()
        o = _DescentProbe(b, seed=s * 100)
        r = o.optimize(ev_budget)
        #  close whatever hunt was live when the budget ran out
        if o._cur is not None:
            o._cur["end"] = len(r.history_f)
            o._cur["reason"] = "budget"
            o._cur["bb_end"] = float(min(o._cur["bb"])) if o._cur["bb"] else float("nan")
            o.hunts.append(o._cur)
            o._cur = None

        deep = 0
        for i, c in enumerate(o.hunts):
            ev = np.asarray(c["ev"], dtype=float)
            bb = np.asarray(c["bb"], dtype=float)
            n = c["end"] - c["start"]
            j, d = _attribute(c["bx"], opt)
            last = float(ev[-1]) if len(ev) else 0.0
            hw.writerow([s, i, c["start"], n, c["reason"],
                         f"{c['bb0']:.6g}", f"{c['bb_end']:.6g}", len(ev),
                         f"{last:.0f}", f"{(n - last) / n:.4f}" if n > 0 else "",
                         f"{_slope(ev, bb, 0.0, 0.5, c['bb0']):.6g}",
                         f"{_slope(ev, bb, 0.75, 1.0, c['bb0']):.6g}",
                         j, f"{d:.4g}"])
            if c["bb_end"] <= a.curve_eps and len(ev):
                deep += 1
                for e, v in zip(c["ev"], c["bb"]):
                    cw.writerow([s, i, e, f"{v:.6g}"])

        #  Is the reported 6/6 coverage at 1e-1 really six *different* global
        #  basins? The scorer only requires the points to be rho apart.
        sx = np.asarray([np.asarray(x, dtype=float) for x in r.final_solutions])
        sfv = np.asarray([b.func(x) for x in sx])
        order = np.argsort(sfv)
        sx, sfv = sx[order], sfv[order]
        seeds = set(_seed_indices(sx, b.niche_rho))
        for k in range(len(sx)):
            j, d = _attribute(sx[k], opt)
            rw.writerow([s, k, f"{sfv[k]:.6g}", j, f"{d:.4g}", int(k in seeds),
                         int(sfv[k] <= 1e-1), int(sfv[k] <= 1e-3),
                         int(sfv[k] <= 1e-5)])

        print(f"{a.func} seed={s} evals={ev_budget} hunts={len(o.hunts)} "
              f"deep(<= {a.curve_eps:g})={deep} best_f={r.best_f:.6g} "
              f"n_rep={len(r.final_solutions)} wall={time.time() - t0:.1f}s",
              flush=True)

    hf.close()
    cf.close()
    rf.close()


if __name__ == "__main__":
    main()
