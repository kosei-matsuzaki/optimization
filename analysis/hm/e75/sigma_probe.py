"""Which clause ends the hunts on N18-CF3-10D, and would the two sigma arms move it?

Entry 74 established that on N18 the *level* clause decides 0-20 releases per run,
so `hunt_level_tol` is a dead knob there; the remaining depth tools are the two
sigma-side arms entry 55 closed for free on N08-Shubert3D
(`_sig10` = exhausted_sigma_tol 1.5 -> 1.0, `_fl08` = sigma_floor_ratio 1e-6 -> 1e-8).

Entry 74's lesson is that a knob's *sign* being right does not make it binding, so
this counts the firings first (2 minutes) before any 11-minute arm is run. Same
device as `e74/fire_probe.py`: `_basin_exhausted` is re-implemented with counters
and returns the same value, so the search is unchanged.

Per release event (`out == True`) it records the counterfactuals the two arms
actually install:

  would_sig10_block   sigma > 1.0 x floor at release  ==> `_sig10` delays this hunt
  would_fl08_block    sigma > 1.5 x (floor/100)       ==> `_fl08` delays this hunt
  level_would_cover   basin_best <= L_eff             ==> the level clause releases
                      it anyway once the sigma clause is tightened, so the arm
                      buys no extra drilling on this hunt

and separately splits the *non*-release calls into "sigma satisfied, waiting on
stagnation" vs "stagnated, waiting on sigma", which is the entry-54 reading of
which side is binding.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/optimization")
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME  # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer  # noqa: E402


class _SigmaProbe(MultiChannelEpidemicOptimizer):
    """The shipped optimiser, tapped on the release decision (search unchanged)."""

    #  The two arms under test, as multipliers on the shipped release threshold
    #  `exhausted_sigma_tol * span * sigma_floor_ratio` (= 1.5 floor units).
    _SIG10 = 1.0 / 1.5          # exhausted_sigma_tol 1.5 -> 1.0
    _FL08 = 1e-8 / 1e-6         # sigma_floor_ratio 1e-6 -> 1e-8

    def _init_state(self, max_evals):
        st = super()._init_state(max_evals)
        self._st = st
        self.f_init_scale = float(st.f_init_scale)
        self.n_calls = 0
        self.n_release = 0
        self.n_sig_only = 0       # sigma bottomed, stagnation not yet -> stagnation binding
        self.n_stag_only = 0      # stagnated, sigma not yet -> sigma binding
        self.n_level_release = 0  # release the level clause decided on its own
        self.n_sig10_block = 0
        self.n_fl08_block = 0
        self.n_level_covers = 0   # release where basin_best is already <= L_eff
        self.n_arm_level_covers = 0  # ... <= 1e-5, the level the arms install
        self.sig_units = []       # sigma / floor at each release
        self.basin_at_rel = []
        return st

    def _basin_exhausted(self, st) -> bool:
        floor = st.span * self.sigma_floor_ratio
        thresh = self.exhausted_sigma_tol * floor
        sigma_bottomed_raw = st.sigma <= thresh
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

        self.n_calls += 1
        if not out:
            if sigma_bottomed and not stagnated:
                self.n_sig_only += 1
            elif stagnated and not sigma_bottomed:
                self.n_stag_only += 1
        else:
            self.n_release += 1
            if level_decided:
                self.n_level_release += 1
            self.sig_units.append(float(st.sigma / floor) if floor > 0 else float("nan"))
            self.basin_at_rel.append(float(st.basin_best))
            #  `_sig10` / `_fl08` only matter where the sigma clause is what let
            #  this hunt go, i.e. where the level clause is not already true.
            l_eff = self.hunt_level_tol * st.f_init_scale
            covered = st.basin_best <= l_eff
            if covered:
                self.n_level_covers += 1
            #  The arms are `level_rel_c100_*`, i.e. they carry c = 1.0 on top
            #  (L_eff = 1e-5, a proven no-op on N18 by entry 74). Under them the
            #  level clause is 1e-5, not base's 1e-6 x f_init_scale.
            arm_covered = st.basin_best <= 1e-5
            if arm_covered:
                self.n_arm_level_covers += 1
            #  Under the arm the level clause sits at 1e-5, so a hunt already at
            #  that depth is released by it whatever the sigma clause says.
            if st.sigma > self._SIG10 * thresh and not arm_covered:
                self.n_sig10_block += 1
            if st.sigma > self._FL08 * thresh and not arm_covered:
                self.n_fl08_block += 1

        if sigma_bottomed and stagnated:
            st.has_exhausted = True
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--arm", default="base", choices=("base", "sig10", "fl08"),
                    help="probe the shipped gate, or one of the two sigma arms")
    a = ap.parse_args()
    b = NICHING_BENCHMARKS_BY_NAME[a.func]
    ev = a.evals or int(b.suite_max_evals)
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        kw = {"sig10": {"exhausted_sigma_tol": 1.0},
              "fl08": {"sigma_floor_ratio": 1e-8}}.get(a.arm, {})
        o = _SigmaProbe(b, seed=s * 100, **kw)
        r = o.optimize(ev)
        su = np.asarray(o.sig_units) if o.sig_units else np.asarray([np.nan])
        ba = np.asarray(o.basin_at_rel) if o.basin_at_rel else np.asarray([np.nan])
        l_eff = o.hunt_level_tol * o.f_init_scale
        print(
            f"{a.func:<13} {a.arm:<5} seed={s} evals={ev} fis={o.f_init_scale:.4g} L_eff={l_eff:.4g} "
            f"calls={o.n_calls} releases={o.n_release} level_only={o.n_level_release} "
            f"sig_ok_wait_stag={o.n_sig_only} stag_ok_wait_sig={o.n_stag_only} "
            f"sig_units[min/med/max]={np.nanmin(su):.4g}/{np.nanmedian(su):.4g}/{np.nanmax(su):.4g} "
            f"basin_at_rel[med]={np.nanmedian(ba):.4g} level_covers={o.n_level_covers} "
            f"arm_level_covers={o.n_arm_level_covers} "
            f"sig10_blocks={o.n_sig10_block} fl08_blocks={o.n_fl08_block} "
            f"best_f={r.best_f:.6g} n_rep={len(r.final_solutions)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
