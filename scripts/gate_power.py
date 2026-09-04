#!/usr/bin/env python3
"""Can a BBOB safety gate distinguish `hunt_level_tol` arms at all?

Entries 29 and 32 both ran the gate (BBOB-24 dim2, 5000 then 20000 evals) and
got arms that were *numerically identical*. Identical output has two very
different causes and the gate cannot tell them apart:

  (i)  the tightened release rule fires and the run ends up in the same place
       anyway  -> the variant is genuinely safe here;
  (ii) the rule never gets to decide anything  -> the gate measured nothing.

This probe measures (ii) directly, on base MC-ESO only, so the search is
untouched and these are base's own numbers. Per run it records:

  exhausted      did `has_exhausted` ever become True? The level clause at
                 `mceso.py:919` is dead code before that.
  f_at_exh       best_so_far at the first exhaustion.
  post_gain      f_at_exh - best_f. Zero means the run banked its final answer
                 before the clause was ever live, so no release rule downstream
                 of exhaustion can move best_f. This is entry 29's measurement.
  n_div_t07/08   generations where base's release decision is *decided by the
                 level clause* and a tightened tol would have decided the other
                 way: stagnated, sigma not bottomed on its own, and
                 tol_tight * scale < basin_best <= 1e-6 * scale. Each one is a
                 hunt base releases and the variant keeps drilling.

A function can only supply gate power when both are non-zero: the arms have to
diverge (n_div > 0) *and* what happens after the divergence has to be able to
reach best_f (post_gain > 0). Everything else is a tie by construction, and a
tie by construction is not evidence of safety.

Usage:
  python3 scripts/gate_power.py --funcs F17-SchafferF7,F23-Katsuura \
      --dim 2 --seeds 10 --evals 20000 --csv analysis/hm/gatepower_dim2.csv
"""
from __future__ import annotations
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import (BENCHMARKS_BY_NAME, BENCHMARKS_3D_BY_NAME,     # noqa: E402
                             BENCHMARKS_5D_BY_NAME, BENCHMARKS_10D_BY_NAME)
from core.optimizers import MultiChannelEpidemicOptimizer                    # noqa: E402
from core.optimizers.mceso_rel_level import RelLevelMCESO                    # noqa: E402

_REG = {2: BENCHMARKS_BY_NAME, 3: BENCHMARKS_3D_BY_NAME,
        5: BENCHMARKS_5D_BY_NAME, 10: BENCHMARKS_10D_BY_NAME}

#  The tightened values under test (entry 32 recommends 1e-7, 1e-8 is the peak).
_TIGHT = (1e-7, 1e-8)


class _ProbeMixin:
    """A read-only tap on the basin-release decision.

    `_basin_exhausted` is re-implemented rather than wrapped because the two
    clauses it ORs together have to be told apart: only a release that the level
    clause *decided on its own* can be flipped by changing `hunt_level_tol`.
    The logic below is `mceso.py:910-940` with the shipped defaults, plus
    counters; it returns the same value, so the search is unchanged. It reads
    `self.hunt_level_tol` rather than the shipped constant, so mixing it into
    `RelLevelMCESO` (whose `_init_state` rewrites that attribute from the run's
    own `f_init_scale`) probes the eps-relative arm with the same code.
    """

    def _init_state(self, max_evals):
        st = super()._init_state(max_evals)
        self._st = st
        self.f_at_exh: float | None = None
        self.n_div = dict.fromkeys(_TIGHT, 0)
        self.n_level_release = 0
        self.f_init_scale = float(st.f_init_scale)
        return st

    def _basin_exhausted(self, st) -> bool:
        sigma_bottomed_raw = (st.sigma
                              <= self.exhausted_sigma_tol * st.span * self.sigma_floor_ratio)
        sigma_bottomed = sigma_bottomed_raw
        level_decided = False
        if st.has_exhausted and self.hunt_level_tol > 0.0:
            by_level = st.basin_best <= self.hunt_level_tol * st.f_init_scale
            level_decided = by_level and not sigma_bottomed_raw
            sigma_bottomed = sigma_bottomed or by_level
        mult = self.exhausted_no_improve_mult
        if st.has_exhausted and self.hunt_no_improve_mult > 0.0:
            mult = self.hunt_no_improve_mult
        stagnated = st.no_improve >= mult * self._stagnation_window()
        out = sigma_bottomed and stagnated
        if out and level_decided:
            #  base lets this hunt go because the level clause fired. A tighter
            #  tol keeps drilling iff the basin has not reached *its* level.
            self.n_level_release += 1
            for tol in _TIGHT:
                if st.basin_best > tol * st.f_init_scale:
                    self.n_div[tol] += 1
        was = st.has_exhausted
        if sigma_bottomed and stagnated:
            st.has_exhausted = True
        if not was and st.has_exhausted and self.f_at_exh is None:
            self.f_at_exh = float(st.best_so_far)
        return out


class _GateProbe(_ProbeMixin, MultiChannelEpidemicOptimizer):
    """The shipped optimiser, tapped."""


class _RelGateProbe(_ProbeMixin, RelLevelMCESO):
    """The eps-relative diagnostic variant (entry 36), tapped."""


def _compare(reg, names, seeds, evals, tol, csv_path) -> None:
    """Where does the tie come from -- same search, or same answer?

    Runs base and the tightened arm on the same seed and reports the evaluation
    at which their histories first diverge (`div_ev`) against the evaluation at
    which base banks its final best_f (`bank_ev`). `bank_ev < div_ev` in every
    run means the arms cannot differ in anything the gate scores: the answer is
    already fixed before the two searches part company, so an identical
    SR@1e-10 is a tie by construction, not a safety measurement.
    """
    rows = []
    print(f"{'function':<24}{'runs':>6}{'differ':>8}{'bank<div':>10}"
          f"{'med_bank':>10}{'med_div':>9}{'same_best_f':>13}")
    print("-" * 82)
    for name in names:
        b = reg[name]
        per = []
        for seed in range(seeds):
            r0 = MultiChannelEpidemicOptimizer(b, seed=seed * 100).optimize(evals)
            r1 = MultiChannelEpidemicOptimizer(b, seed=seed * 100,
                                               hunt_level_tol=tol).optimize(evals)
            h0, h1 = np.asarray(r0.history_f), np.asarray(r1.history_f)
            n = min(len(h0), len(h1))
            neq = np.nonzero(h0[:n] != h1[:n])[0]
            div = int(neq[0]) if len(neq) else (n if len(h0) != len(h1) else -1)
            hb = np.asarray(r0.history_best)
            bank = int(np.argmax(hb <= float(r0.best_f)))
            per.append(dict(function=name, seed=seed, div_ev=div, bank_ev=bank,
                            base_best_f=float(r0.best_f), var_best_f=float(r1.best_f),
                            same=float(r0.best_f) == float(r1.best_f)))
        differ = sum(1 for r in per if r["div_ev"] >= 0)
        bank_first = sum(1 for r in per if r["div_ev"] >= 0 and r["bank_ev"] < r["div_ev"])
        same = sum(1 for r in per if r["same"])
        med_b = float(np.median([r["bank_ev"] for r in per]))
        dv = [r["div_ev"] for r in per if r["div_ev"] >= 0]
        med_d = float(np.median(dv)) if dv else float("nan")
        print(f"{name:<24}{seeds:>6}{differ:>8}{bank_first:>10}"
              f"{med_b:>10.0f}{med_d:>9.0f}{same:>13}")
        rows.extend(per)
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--funcs", required=True, help="comma-separated exact names")
    ap.add_argument("--dim", type=int, default=2, choices=sorted(_REG))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index, for splitting a run across workers")
    ap.add_argument("--evals", type=int, default=20000)
    ap.add_argument("--rel-level", type=float, default=0.0, metavar="L",
                    help="probe RelLevelMCESO(rel_level=L) instead of the "
                         "shipped optimiser (entry 36's eps-relative arm; "
                         "L = c * eps_target, e.g. 1e-6 for c=0.1)")
    ap.add_argument("--fis-floor", type=float, default=0.0, metavar="S",
                    help="clamp shape (a), entry 44: lower-bound f_init_scale "
                         "at S before dividing, so the endpoint f_init_scale "
                         "-> 1e-300 cannot blow the quotient up. No-op at 0.")
    ap.add_argument("--tol-cap", type=float, default=float("inf"), metavar="T",
                    help="clamp shape (b), entry 44: upper-bound the resulting "
                         "hunt_level_tol at T. Entry 37 wrote T = the shipped "
                         "default 1e-6, which entry 43 showed returns the "
                         "variant to base wherever f_init_scale < 1.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--compare", type=float, default=None, metavar="TOL",
                    help="instead of probing base, run base against "
                         "hunt_level_tol=TOL and locate the tie")
    args = ap.parse_args()

    reg = _REG[args.dim]
    if args.compare is not None:
        _compare(reg, [s.strip() for s in args.funcs.split(",")],
                 args.seeds, args.evals, args.compare, args.csv)
        return
    rows = []
    if args.rel_level > 0.0:
        print(f"probing RelLevelMCESO(rel_level={args.rel_level:g}, "
              f"fis_floor={args.fis_floor:g}, tol_cap={args.tol_cap:g}) "
              f"-- effective release level L is fixed, hunt_level_tol = L / f_init_scale")
    print(f"{'function':<24}{'exh':>5}{'post_gain>0':>12}{'lvl_rel':>9}"
          f"{'div_1e-7':>10}{'div_1e-8':>10}{'power_t07':>11}{'power_t08':>11}")
    print("-" * 92)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    for name in [s.strip() for s in args.funcs.split(",")]:
        b = reg[name]
        per = []
        for seed in seeds:
            if args.rel_level > 0.0:
                o = _RelGateProbe(b, seed=seed * 100, rel_level=args.rel_level,
                                  fis_floor=args.fis_floor, tol_cap=args.tol_cap)
            else:
                o = _GateProbe(b, seed=seed * 100)
            res = o.optimize(args.evals)
            gain = (float("nan") if o.f_at_exh is None
                    else o.f_at_exh - float(res.best_f))
            per.append(dict(seed=seed, exhausted=o.f_at_exh is not None,
                            f_at_exh=o.f_at_exh, best_f=float(res.best_f),
                            post_gain=gain, n_level_release=o.n_level_release,
                            div_t07=o.n_div[1e-7], div_t08=o.n_div[1e-8],
                            f_init_scale=o.f_init_scale,
                            eff_tol=float(o.hunt_level_tol)))
        n_exh = sum(r["exhausted"] for r in per)
        n_gain = sum(1 for r in per if r["exhausted"] and r["post_gain"] > 0)
        lvl = float(np.mean([r["n_level_release"] for r in per]))
        d7 = float(np.mean([r["div_t07"] for r in per]))
        d8 = float(np.mean([r["div_t08"] for r in per]))
        #  A run supplies gate power only if the arms diverge AND best_f can
        #  still move after the first exhaustion.
        p7 = sum(1 for r in per if r["div_t07"] > 0 and r["post_gain"] > 0)
        p8 = sum(1 for r in per if r["div_t08"] > 0 and r["post_gain"] > 0)
        print(f"{name:<24}{n_exh:>5}{n_gain:>12}{lvl:>9.1f}{d7:>10.1f}{d8:>10.1f}"
              f"{p7:>11}{p8:>11}")
        for r in per:
            rows.append(dict(function=name, dim=args.dim, evals=args.evals, **r))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
