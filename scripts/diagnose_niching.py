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
from core.optimizers.mceso_adaptive_repel import AdaptiveRepelMCESO  # noqa: E402
from core.optimizers.mceso_commit_reseed import CommitReseedMCESO    # noqa: E402
from core.optimizers.mceso_phased_accept import PhasedAcceptMCESO    # noqa: E402
from core.optimizers.mceso_crowding import MCESOCrowding             # noqa: E402
from core.optimizers.mceso_recover import RecoverMCESO               # noqa: E402
from core.optimizers.mceso_rel_level import RelLevelMCESO            # noqa: E402
from core.optimizers.mceso_sol_archive import SolArchiveTrimMCESO    # noqa: E402
from core.runner import _seed_indices, count_goptima             # noqa: E402


class _HuntCounters:
    """Counters on the restart hooks. Behaviour is unchanged: every override
    calls super() and only tallies. Mixed in front of whichever MC-ESO class a
    variant selects, so base and variant runs are dumped identically."""

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


class _CountingMCESO(_HuntCounters, MultiChannelEpidemicOptimizer):
    """The shipped optimiser, instrumented."""


class _CountingAdaptiveMCESO(_HuntCounters, AdaptiveRepelMCESO):
    """The adaptive-repel diagnostic variant, instrumented the same way."""


class _CountingCommitMCESO(_HuntCounters, CommitReseedMCESO):
    """The commit-to-one-draw diagnostic variant, instrumented the same way."""


class _CountingPhasedMCESO(_HuntCounters, PhasedAcceptMCESO):
    """The phased acceptance-rule variant, instrumented the same way."""


class _CountingCrowdingMCESO(_HuntCounters, MCESOCrowding):
    """Pure crowding replacement, instrumented the same way."""


class _CountingRecoverMCESO(_HuntCounters, RecoverMCESO):
    """The late recovery-phase variant, instrumented the same way."""


class _CountingRelLevelMCESO(_HuntCounters, RelLevelMCESO):
    """The eps-relative hunt-release-level variant, instrumented the same way."""


class _CountingSolArchiveMCESO(_HuntCounters, SolArchiveTrimMCESO):
    """The niche-greedy answer-archive trim variant, instrumented the same way."""


class _CountingCommitSolArchiveMCESO(_HuntCounters, CommitReseedMCESO,
                                     SolArchiveTrimMCESO):
    """Both diagnostics at once: the committed restart (entry 62, which raises
    the *coverage ceiling*) and the niche-greedy answer-archive trim (entry 60,
    which stops the report from throwing that coverage away).

    The two touch disjoint hooks — ``CommitReseedMCESO`` overrides
    ``_diversified_reseed`` / ``_maybe_spillover``, ``SolArchiveTrimMCESO``
    overrides ``_on_spillover_start`` — so the MRO
    (counters -> commit -> adaptive-repel -> archive trim -> shipped) runs each
    exactly where it runs on its own; the archive override still snapshots the
    pre-trim archive before delegating, so its semantics are unchanged.
    ``commit_mode="off", sol_trim_mode="off"`` is the identity check."""


class _CountingRelCommitSolArchiveMCESO(_HuntCounters, RelLevelMCESO,
                                        CommitReseedMCESO,
                                        SolArchiveTrimMCESO):
    """Entry 69: the composite of entry 68 (`ct_soltrim`) with the *depth*
    tuning underneath it.

    Entry 68 left 122 of `ct_soltrim`'s 195 rho-separated report points short of
    eps=1e-5: the tight run sigma buys landings and pays for them in descent.
    The two depth dials that entries 51 and 55 closed for free -- the
    eps-relative hunt-release level (``rel_level``) and the sigma floor
    (``sigma_floor_ratio``, a constructor argument of the shipped class) -- have
    never been measured on top of the descent arm.

    The three overrides are on disjoint hooks (``RelLevelMCESO._init_state``,
    ``CommitReseedMCESO._diversified_reseed`` / ``_maybe_spillover``,
    ``SolArchiveTrimMCESO._on_spillover_start``), so the MRO
    (counters -> rel-level -> commit -> adaptive-repel -> archive trim ->
    shipped) runs each exactly where it runs on its own. ``rel_level=0.0``
    disables only the new layer, which is the identity check against
    ``ct_soltrim``."""


# variant name -> (class, constructor kwargs)
_VARIANTS: dict[str, tuple[type, dict]] = {
    "base": (_CountingMCESO, {}),
    "localwin": (_CountingMCESO, {"exhausted_local_window": True}),
    # NOTE: `fast` is a no-op — mceso.py's default hunt_no_improve_mult is
    # already 0.5, so this variant reproduces `base` exactly. Kept as an
    # identity check; use the hunts_* controls below to actually add hunts.
    "fast": (_CountingMCESO, {"hunt_no_improve_mult": 0.5}),
    # Hunt-count controls for the sigma_only confounder test: release hunts
    # earlier *without* touching the post-restart sigma, so base can be run at
    # a matched number of hunts. Two independent release knobs:
    #   hunt_no_improve_mult  - shorter stagnation window at the sigma floor
    #   exhausted_sigma_tol   - call the basin bottomed at a larger sigma
    **{f"hunts_m{int(round(m * 100)):03d}": (_CountingMCESO,
                                             {"hunt_no_improve_mult": m})
       for m in (0.35, 0.25, 0.15, 0.05)},
    **{f"hunts_tol{int(round(t)):02d}": (_CountingMCESO,
                                         {"exhausted_sigma_tol": t})
       for t in (3.0, 6.0, 12.0, 25.0)},
    # Restart repulsion scaled by the observed distances between basins already
    # drilled, instead of a fixed 0.02 * span. See mceso_adaptive_repel.py.
    "adaptive": (_CountingAdaptiveMCESO, {"repel_mode": "adaptive"}),
    "adaptive_maxmin": (_CountingAdaptiveMCESO, {"repel_mode": "adaptive_maxmin"}),
    # Identity check: the variant class with its override disabled must
    # reproduce `base` exactly.
    "adaptive_off": (_CountingAdaptiveMCESO, {"repel_mode": "off"}),
    # The basin switch commits the whole population to a single repelled draw
    # instead of racing n_pop independent ones. See mceso_commit_reseed.py.
    "commit": (_CountingCommitMCESO, {"commit_sigma_mode": "place"}),
    "commit_sigma": (_CountingCommitMCESO, {"commit_sigma_mode": "run"}),
    # Commitment on top of the basin-scale repel radius (entry 22's draw rule).
    "commit_adaptive": (_CountingCommitMCESO, {"commit_sigma_mode": "run",
                                               "repel_mode": "adaptive"}),
    # How tight the commitment has to be: 0.1 x the local basin spacing instead
    # of 0.25 x. Separates "commit" from "commit *narrowly*".
    "commit_tight": (_CountingCommitMCESO, {"commit_sigma_mode": "run",
                                            "commit_sigma_ratio": 0.1}),
    # The other half of `commit_tight`, at the *same* ratio: the population is
    # committed to one anchor (the best-of-n_pop race is gone) but the
    # post-restart sigma stays at the base sigma_init, so the hunt still
    # searches at the box scale. With `sigma_only_r010` and `commit_tight` this
    # closes a 2x2 -- commit on/off x run-sigma tight/base -- at a matched
    # ratio, which `commit` (place mode, ratio 0.25) does not (entry 40).
    "commit_place_r010": (_CountingCommitMCESO, {"commit_sigma_mode": "place",
                                                 "commit_sigma_ratio": 0.1}),
    # Identity check for the commit class.
    "commit_off": (_CountingCommitMCESO, {"commit_mode": "off"}),
    # Control: base restart draws (the best-of-n_pop race is kept) with only the
    # post-restart sigma taken down to the local basin scale.
    "sigma_only": (_CountingCommitMCESO, {"commit_mode": "sigma_only",
                                          "commit_sigma_mode": "run",
                                          "commit_sigma_ratio": 0.1}),
    # Two-phase acceptance rule (question 1): the shipped host competition while
    # the first basin is being drilled, nearest-neighbour crowding for every
    # later hunt. Entry 22 showed the restart's coverage is limited by the
    # best-of-n_pop selection, not by the draw; this removes that pressure
    # without touching sigma (the lever entries 26/27 closed).
    "phased": (_CountingPhasedMCESO, {"accept_phase": "exhausted"}),
    # Controls. `phased_off` must reproduce `base` exactly; `crowd_always` is
    # the wholesale swap the audit found destroys depth, and `phased_always`
    # must reproduce it exactly (shared body).
    "phased_off": (_CountingPhasedMCESO, {"accept_phase": "off"}),
    "phased_always": (_CountingPhasedMCESO, {"accept_phase": "always"}),
    "crowd_always": (_CountingCrowdingMCESO, {}),
    # Question 1 (allocation side): in the last 20% of the budget an exhausted
    # basin switch stops repelling away and re-enters a recorded under-drilled
    # basin instead. See mceso_recover.py.
    "recover": (_CountingRecoverMCESO, {"recover_mode": "blocked"}),
    # Control: same phase, same tight sigma, anchored on the ordinary repelled
    # draw. Separates "return to the blocked basins" from "restart tightly late"
    # -- entries 23/25 showed the basin-scale sigma alone already buys depth in 2D.
    "recover_ctrl": (_CountingRecoverMCESO, {"recover_mode": "fresh"}),
    # Identity check for the recover class: must reproduce `base` exactly.
    "recover_off": (_CountingRecoverMCESO, {"recover_mode": "off"}),
    # The release level itself. `blocked_inventory.py` found that on
    # N06-Shubert2D every abandoned basin stops in (1e-5, 1e-3] -- i.e. at
    # hunt_level_tol * f_init_scale ~ 1e-4, which passes eps=1e-3 and fails
    # eps=1e-5. Entry 25 swept the other two release knobs
    # (hunt_no_improve_mult, exhausted_sigma_tol) and neither moved anything;
    # this one is the path that actually fires on equal-height multi-global
    # problems (mceso.py:919-928). Lowering it buys depth per hunt and must cost
    # hunts, so the coarse levels are the rejection condition.
    **{f"level_t{int(round(-np.log10(t))):02d}": (_CountingMCESO,
                                                  {"hunt_level_tol": t})
       for t in (1e-7, 1e-8, 1e-10)},
    # Question 1 (entry 36): the release level made *dimension invariant*. The
    # fixed level_t* sweep gave two different best values (2D 1e-7, 3D 1e-8)
    # because the effective release level is `hunt_level_tol * f_init_scale` and
    # f_init_scale is ~145 in 2D but ~2493 in 3D (entry 35). This variant fixes
    # the *effective* level L = c * eps_target directly and lets hunt_level_tol
    # track f_init_scale, so the same c should be best in both dimensions.
    # eps_target = 1e-5 (the deepest scoring accuracy); c = 0.1 -> L = 1e-6
    # (matches the 2D win the fixed 1e-8 bought, 1.4e-6) and c = 0.01 -> L = 1e-7.
    # `level_rel_off` (L=0) must reproduce base exactly (identity check).
    "level_rel_off": (_CountingRelLevelMCESO, {"rel_level": 0.0}),
    # Entry 46 added the loose side. c = 0.1 buys N06's PR@1e-5 (+0.55) but pays
    # PR@1e-3 (-0.063, p = 0.0062, 30 seeds), and base sits at an effective
    # c = 1e-6 * f_init_scale / 1e-5 = 14.2 in 2D, so everything between c = 0.1
    # and base was blank. The sweep asks whether the two move together (one line,
    # entry 29's finding again) or whether an interior c keeps the deep gain
    # without the 1e-3 loss.
    **{f"level_rel_c{int(round(c * 100)):02d}": (_CountingRelLevelMCESO,
                                                 {"rel_level": c * 1e-5})
       for c in (0.1, 0.01, 0.3, 1.0, 3.0, 10.0)},
    # Entry 44: the three shapes of the clamp entry 37 required, at c = 0.1.
    # `_dflt` is the shape entry 37 wrote down (cap at the shipped default
    # 1e-6); on any function with f_init_scale < 1 it returns the variant to
    # base, so on N05 / N07 / N09 it must reproduce base row for row -- that is
    # the measurement, not an identity check. `_fis` and `_cap` are the two
    # shapes that touch only the endpoint (f_init_scale pinned at 1e-300); on
    # every niching function here (f_init_scale 0.17 .. 2511) neither binds, so
    # both must reproduce `level_rel_c10` row for row.
    "level_rel_c10_dflt": (_CountingRelLevelMCESO,
                           {"rel_level": 1e-6, "tol_cap": 1e-6}),
    "level_rel_c10_fis": (_CountingRelLevelMCESO,
                          {"rel_level": 1e-6, "fis_floor": 1e-12}),
    "level_rel_c10_cap": (_CountingRelLevelMCESO,
                          {"rel_level": 1e-6, "tol_cap": 1e-2}),
    # Entry 55 (question 1): the *other* release clause, on top of the adoption
    # candidate (c = 1.0, `level_rel_c100`). Entry 54 counted the clause that
    # ends the hunts behind N08-Shubert3D's residual step (PR 0.860 at 1e-4
    # against 0.754 at 1e-5) and found it is always the sigma-floor clause
    # (129/129 reported step points, level clause 0/129), with sigma stopped at
    # 1.449x its floor -- i.e. released with 0.45 floor-units of step size still
    # unspent. These two arms give the hunt that room, in the two ways it can be
    # given, and are paired against `level_rel_c100` on the same seeds:
    #   _sig10  the clause itself, at the one point the queue pre-registered:
    #           released only once sigma is *at* the floor (tol 1.5 -> 1.0).
    #           This is the arm the rejection condition is written against.
    #   _fl08   diagnostic only: the floor moved down two decades (1e-6 ->
    #           1e-8) with the clause left at its default. `_sig10` can only
    #           buy the 0.45 floor-units that exist; if it comes back null,
    #           this arm separates "depth is not sigma-limited" from "the room
    #           `_sig10` could give was too small to see".
    "level_rel_c100_sig10": (_CountingRelLevelMCESO,
                             {"rel_level": 1e-5, "exhausted_sigma_tol": 1.0}),
    "level_rel_c100_fl08": (_CountingRelLevelMCESO,
                            {"rel_level": 1e-5, "sigma_floor_ratio": 1e-8}),
    # Entry 58 / question 2: the answer archive's f-only trim
    # (`np.argsort(sol_archive_f)[:200]`, mceso.py:822) is blind to niches, and
    # on N09-Vincent3D it drops 59.20 rho-separated endpoints to 37.33 (PR@1e-5
    # 0.274 -> 0.173). `sol_archive` is read only when the reported set is
    # assembled and never re-enters the search, so all three arms below run the
    # *same* trajectory: `best_f`, `hunts` and every diagnostic except the
    # reported-set columns must match base seed for seed. That equality is the
    # measurement's built-in identity check, not an assumption.
    #   solcap2000  the ceiling: capacity raised past the hunt count (676 at the
    #               suite budget) so the trim never fires. No new code — it is a
    #               constructor argument of the shipped class.
    #   soltrim_rho capacity left at 200, selection made niche-greedy at
    #               0.02 x span (the reseed's own fine repel radius, not the
    #               scoring rho). Matching solcap2000 here means the fix costs
    #               no memory, so this is the adoption shape.
    #   soltrim_off identity check for the variant class: must reproduce base.
    "solcap2000": (_CountingMCESO, {"solution_archive_max": 2000}),
    "soltrim_rho": (_CountingSolArchiveMCESO, {"sol_trim_mode": "rho"}),
    "soltrim_off": (_CountingSolArchiveMCESO, {"sol_trim_mode": "off"}),
    # Entry 63 / question 1: entry 62 raised N09's coverage ceiling
    # (`visited@1e-5` 0.275 -> 0.355) but the report still passes a fixed
    # fraction of it (`reported / visited` 0.63 vs 0.62), so PR@1e-5 stopped at
    # 0.221. `soltrim_rho` is the tool that recovered the report up to the
    # *base* ceiling (entry 60); this arm asks whether it still does once the
    # ceiling has moved. Only the composite is new — both single arms are on
    # disk (entries 60 and 62) and pair with this one by seed.
    "cp_soltrim": (_CountingCommitSolArchiveMCESO,
                   {"commit_sigma_mode": "place", "commit_sigma_ratio": 0.1,
                    "sol_trim_mode": "rho"}),
    "cp_soltrim_off": (_CountingCommitSolArchiveMCESO,
                       {"commit_mode": "off", "sol_trim_mode": "off"}),
    # Entry 68 / question 1: entry 67 showed the descent (`commit_tight`, run
    # sigma at 0.1 x the local basin spacing) doubles the *landed* basins
    # (63 -> 123) where `commit_place` only reaches 81. `ct_soltrim` is the same
    # composite as `cp_soltrim` with the descent arm underneath, so the pair
    # asks whether that extra landing survives to the report once the answer
    # archive's f-only trim is out of the way. Both single arms (`commit_tight`
    # here, `cp_soltrim` from entry 63) pair with it by seed.
    "ct_soltrim": (_CountingCommitSolArchiveMCESO,
                   {"commit_sigma_mode": "run", "commit_sigma_ratio": 0.1,
                    "sol_trim_mode": "rho"}),
    # Entry 69 / question 1: entry 68 showed the descent doubles the landings
    # but 122 of the 195 report points miss eps=1e-5, so the coverage ceiling
    # did not move (0.319 vs `cp_soltrim`'s 0.355). These put the two *free*
    # depth dials underneath that arm. `rel_level = 1e-5` is c = 1.0 (entry 46's
    # corner, L = eps_target) with the adoption-shape clamp `fis_floor = 1e-12`
    # (entry 44); `_fl08` adds entry 55's sigma floor. `ct_soltrim_rel_off`
    # disables only the new layer and must reproduce `ct_soltrim` exactly.
    "ct_soltrim_rel": (_CountingRelCommitSolArchiveMCESO,
                       {"commit_sigma_mode": "run", "commit_sigma_ratio": 0.1,
                        "sol_trim_mode": "rho",
                        "rel_level": 1e-5, "fis_floor": 1e-12}),
    "ct_soltrim_rel_fl08": (_CountingRelCommitSolArchiveMCESO,
                            {"commit_sigma_mode": "run",
                             "commit_sigma_ratio": 0.1,
                             "sol_trim_mode": "rho",
                             "rel_level": 1e-5, "fis_floor": 1e-12,
                             "sigma_floor_ratio": 1e-8}),
    "ct_soltrim_rel_off": (_CountingRelCommitSolArchiveMCESO,
                           {"commit_sigma_mode": "run",
                            "commit_sigma_ratio": 0.1,
                            "sol_trim_mode": "rho", "rel_level": 0.0}),
}
# How tight the commitment has to be: the same variant at a sweep of spreads,
# as a fraction of the locally observed basin spacing. The suffix is the ratio
# x1000, so `commit_tight` (ratio 0.1) is r100, not r010.
# Entry 42 added 0.5: paired against `sigma_only_r500` it is the loose end of
# the commit-increment ladder, whose other end is `commit_place_r010` (run sigma
# left at base). The mechanism entry 41 proposed -- commit only binds once the
# run sigma is tight enough to keep the population near the anchor -- predicts
# that increment shrinks monotonically to zero as the ratio grows.
_VARIANTS.update({
    f"commit_r{int(round(r * 1000)):03d}": (
        _CountingCommitMCESO,
        {"commit_sigma_mode": "run", "commit_sigma_ratio": r})
    for r in (0.5, 0.25, 0.10, 0.05, 0.02, 0.01)
})
# Entry 26 rejected `sigma_only` (ratio 0.1) for the body: on N08-Shubert3D the
# PR@1e-1 sign reverses (0.34 -> 0.20, 0/0/5). The mechanism read off the
# diagnostics is that a tight sigma makes each hunt short, and in 3D a short
# hunt does not reach 1e-1 at all (`blocked` 29.0 -> 54.6). If that is the whole
# story, the ratio is simply mistuned for 3D -- basins there are wider relative
# to the spacing -- and a looser ratio should recover. These are the same
# control as `sigma_only` (base draws, best-of-n_pop race kept; only the
# post-restart sigma moves) at 0.25x and 0.5x the local basin spacing.
_VARIANTS.update({
    f"sigma_only_r{int(round(r * 1000)):03d}": (
        _CountingCommitMCESO,
        {"commit_mode": "sigma_only", "commit_sigma_mode": "run",
         "commit_sigma_ratio": r})
    for r in (0.25, 0.5)
})
# Entry 42: the commit arms move the placement spread and the run sigma with the
# *same* ratio, so `commit_r500 - commit_r100` is not a sigma dial on its own.
# These hold the run sigma at base while the placement spread takes the same
# ratios, which makes the sigma effect readable at a fixed placement:
# `commit_r{r} - commit_place_r{r}`. `commit_place_r010` is the r=0.1 member.
_VARIANTS.update({
    f"commit_place_r{int(round(r * 1000)):03d}": (
        _CountingCommitMCESO,
        {"commit_sigma_mode": "place", "commit_sigma_ratio": r})
    for r in (0.25, 0.5)
})


def _distinct_points(X: np.ndarray, F: np.ndarray, rho: float) -> int:
    """rho-separated points in a set, ignoring accuracy (duplicate check)."""
    if len(X) == 0:
        return 0
    order = np.argsort(F)
    return len(_seed_indices(X[order], rho))


def visited_fast(hx: np.ndarray, hf: np.ndarray, k: int, rho: float,
                 eps: float) -> int:
    """``count_goptima`` over a whole evaluation history, without the O(n^2) walk.

    Exactly equivalent to ``count_goptima(hx, hf, k, rho, eps)``, not an
    approximation. The reference sorts by f ascending, greedily keeps
    rho-separated points over the *whole* set, then counts the kept ones with
    ``f <= eps`` and stops at k. Because the sort is ascending, every point with
    ``f > eps`` comes after every point with ``f <= eps``, so an inaccurate point
    can never occupy a niche ahead of an accurate one — the blocking the
    reference models only bites when the *reported* set is scored, where the cap
    lets bad points in. Dropping ``f > eps`` first therefore cannot change the
    count, and the greedy is prefix stable, so it can stop at k seeds.

    Why it matters: ``_seed_indices`` keeps *every* rho-separated point, which on
    a 40000-point Vincent3D history grows to thousands of seeds and costs
    minutes per run — 40x the 6 s the optimisation itself takes (entry 39).
    """
    m = np.asarray(hf) <= eps
    if not m.any():
        return 0
    X, F = np.asarray(hx)[m], np.asarray(hf)[m]
    order = np.argsort(F)
    return len(_greedy_seeds_capped(X[order], rho, k))


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
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index (default 0). Seeds run are "
                         "[seed_start, seed_start + seeds); the RNG seed is "
                         "index*100 as always, so an earlier run's rows can be "
                         "extended with more seeds instead of re-measured.")
    ap.add_argument("--eps", type=str, default="1e-4",
                    help="one accuracy, or a comma list. Several values are all "
                         "scored off the *same* runs, so relaxing the accuracy "
                         "costs no extra evaluations.")
    ap.add_argument("--variant", type=str, default="base",
                    choices=sorted(_VARIANTS),
                    help="base | localwin | fast | adaptive | adaptive_maxmin "
                         "| adaptive_off (identity check of the variant class)")
    ap.add_argument("--funcs", type=str,
                    default="N04-Himmelblau,N06-Shubert2D,N07-Vincent2D,N10-ModRastrigin2D")
    ap.add_argument("--fast-scoring", action="store_true",
                    help="score `visited` with visited_fast() (exactly "
                         "equivalent, see its docstring) and skip the "
                         "reselected-set columns, which are written as -1. On "
                         "N09-Vincent3D the two skipped walks cost ~4 min per "
                         "run against 6 s of optimisation; without this flag a "
                         "paired 2-arm measurement there does not fit a 40 min "
                         "budget at any useful seed count (entry 39).")
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
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            cls, kw = _VARIANTS[args.variant]
            opt = cls(b, seed=seed * 100, **kw)
            r = opt.optimize(args.evals)

            hx = np.asarray(r.history_x, dtype=float)
            hf = np.asarray(r.history_f, dtype=float)
            sx = np.asarray(r.final_solutions or [r.best_x], dtype=float)
            sf = np.array([float(b.func(x)) for x in sx])
            # The scorer trims the reported set to the cap before counting;
            # do the same here so `reported` is exactly what PR sees. Keep the
            # untrimmed set too: the trim is by f alone, so once the run
            # reports more than `cap` points it can drop whole niches in favour
            # of duplicates of the ones it holds many points in. That only
            # starts to bite at the full suite budget, where the archive
            # outgrows the cap (entry 57).
            sx_all, sf_all = sx, sf
            sx, sf = _cap_by_f(sx, sf, cap)
            # The reselected set depends only on rho and the cap, not on eps,
            # so build it once and score it at every accuracy.
            if args.fast_scoring:
                rx, rf = np.zeros((0, hx.shape[1])), np.zeros(0)
            else:
                rx, rf = reselect_from_history(hx, hf, b.niche_rho, cap)
            runs.append((seed, hx, hf, sx, sf, rx, rf, opt, r, sx_all, sf_all))

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
            for seed, hx, hf, sx, sf, rx, rf, opt, r, sx_all, sf_all in runs:
                if args.fast_scoring:
                    visited = visited_fast(hx, hf, b.n_global_optima,
                                           b.niche_rho, eps)
                    resel = -1
                else:
                    visited = count_goptima(hx, hf, b.n_global_optima,
                                            b.niche_rho, eps)
                    # Same scorer, same cap, on the set reselected from history.
                    resel = count_goptima(rx, rf, b.n_global_optima,
                                          b.niche_rho, eps)
                reported = count_goptima(sx, sf, b.n_global_optima, b.niche_rho, eps)
                # rho-separated points in the reported set regardless of accuracy;
                # `blocked` is how many of those niches are held by a point that
                # misses eps, which is what count_goptima refuses to score.
                distinct = _distinct_points(sx, sf, b.niche_rho)

                # Hunt yield: a hunt "landed" if the basin it abandoned was within
                # eps of the global value — i.e. the restart cycle actually produced
                # a solution rather than being cut off mid-descent.
                hunts = opt.hunts
                landed = sum(1 for _, f in hunts if f <= eps)
                # The same two counts on the *untrimmed* reported set. If these
                # exceed `reported` / `distinct`, the cap's best-f trim is
                # throwing away niches the run already held.
                reported_all = count_goptima(sx_all, sf_all, b.n_global_optima,
                                             b.niche_rho, eps)
                distinct_all = _distinct_points(sx_all, sf_all, b.niche_rho)
                rows.append((visited, reported, resel,
                             -1 if args.fast_scoring else len(rx), distinct,
                             distinct - reported, len(sx), opt.n_spillover,
                             opt.n_basin_switch, len(hunts), landed, r.best_f))
                csv_rows.append([name, b.n_global_optima, eps, seed, args.evals,
                                 cap, *rows[-1], reported_all, distinct_all,
                                 len(sx_all)])
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
                        "spillover", "basin_switch", "hunts", "landed", "best_f",
                        "reported_uncapped", "distinct_uncapped",
                        "n_reported_pts_uncapped"])
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
