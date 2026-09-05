"""Diagnostic MC-ESO variant: **records which clause released each hunt**, and
where every reported solution came from. Recording only — no state is written,
no RNG is drawn, so the arm is bit-identical to :class:`RelLevelMCESO` with the
same keywords (entry 53's arm). ``core/optimizers/mceso.py`` is untouched.

Why (research_loop question 1). Entry 51 left exactly one step in the N08-Shubert3D
profile — PR 0.860 at eps=1e-4 against 0.754 at eps=1e-5, i.e. ~8 of the 81 peaks
are reported only to within (1e-5, 1e-4]. Entry 53 killed the obvious explanation
(the release level L sitting *on* the deepest scored threshold): moving L a decade
down left the step in the same place at the same size. What has never been counted
is **which of the release paths actually ends those hunts**.

The release rule (``mceso.py:910-940``) is a conjunction, not three parallel doors::

    sigma_bottomed = sigma <= exhausted_sigma_tol * span * sigma_floor_ratio
                     or (has_exhausted and basin_best <= hunt_level_tol * f_init_scale)
    stagnated      = no_improve >= mult * stagnation_window     # mult = hunt_no_improve_mult
    released       = sigma_bottomed and stagnated

so a hunt ends either because it *reached the level* L (the level clause) or
because it *ran out of step size* (the sigma-floor clause) — and in both cases
only once the stagnation counter has also filled. This probe records, at every
spillover, which sub-conditions held, what ``basin_best`` was, and how the archived
point was produced, so the depth of a reported solution can be traced back to the
clause that stopped the hunt that produced it.

Provenance. ``optimize()`` reports ``pop + ir_archive + sol_archive`` in that
order (``mceso.py:753-756``), which are three different things:

* ``pop``          — the live population of the hunt that was running when the
                     budget ran out. Its depth is bounded by the budget, not by
                     any release clause.
* ``ir_archive``   — niched elites harvested at each spillover (a *breadth*
                     record, kept at ``n_elite_max``).
* ``sol_archive``  — one point per spillover: the best point of the basin being
                     abandoned. **These are the released hunts**, and the only
                     entries a release clause can be attributed to.

``solution_tags()`` returns that labelling aligned with ``final_solutions``, and
``spill_records()`` the per-spillover release records, joined to the surviving
``sol_archive`` entries by object identity (the archive prunes by f, so the two
lists are not 1:1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mceso import _MCESOState
from .mceso_rel_level import RelLevelMCESO


class ReleaseProbeMCESO(RelLevelMCESO):
    """``RelLevelMCESO`` plus read-only instrumentation of the release clauses.

    Adds no parameters. After ``optimize()`` returns, ``solution_tags()`` and
    ``spill_records()`` describe the run that just finished.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._st: _MCESOState | None = None
        self._exh_this_gen: list[dict] = []
        self._records: list[dict] = []
        self._rec_by_id: dict[int, dict] = {}
        self._watch: dict[str, int | None] = dict.fromkeys(
            ("start", "sigma", "level", "stag"))

    # ── recording hooks (no writes to st, no RNG) ───────────────────────────
    def _init_state(self, max_evals: int) -> _MCESOState:
        st = super()._init_state(max_evals)
        self._st = st
        self._exh_this_gen = []
        self._records = []
        self._rec_by_id = {}
        self._watch = dict.fromkeys(("start", "sigma", "level", "stag"))
        return st

    def _basin_exhausted(self, st: _MCESOState) -> bool:
        """Same decision as the base class, with the sub-conditions written down.

        ``has_exhausted`` is snapshotted *before* the super() call because the
        base flips it to True inside the very call that first returns True, and
        both the level clause and the stagnation multiplier are gated on the
        pre-flip value. Reading it afterwards would attribute the first release
        to the level clause, which the code did not consult.
        """
        pre_exhausted = bool(st.has_exhausted)
        out = super()._basin_exhausted(st)
        sigma_floor_ok = bool(
            st.sigma <= self.exhausted_sigma_tol * st.span * self.sigma_floor_ratio)
        level = self.hunt_level_tol * st.f_init_scale
        level_ok = bool(pre_exhausted and self.hunt_level_tol > 0.0
                        and st.basin_best <= level)
        mult = (self.hunt_no_improve_mult
                if (pre_exhausted and self.hunt_no_improve_mult > 0.0)
                else self.exhausted_no_improve_mult)
        stagnated = bool(st.no_improve >= mult * self._stagnation_window())
        self._exh_this_gen.append({
            "released": bool(out),
            "pre_exhausted": pre_exhausted,
            "sigma_floor_ok": sigma_floor_ok,
            "level_ok": level_ok,
            "stagnated": stagnated,
            "level": float(level),
            "basin_best": float(st.basin_best),
            "sigma_over_floor": float(st.sigma / (st.span * self.sigma_floor_ratio)),
            "no_improve": int(st.no_improve),
            "stag_need": float(mult * self._stagnation_window()),
        })
        return out

    def _hunt_watch(self, st: _MCESOState) -> None:
        """Note the first evaluation count at which each sub-condition became
        true inside the current hunt. The release is a conjunction, so the clause
        that turned true **last** is the binding one — the one that decides how
        deep the hunt gets. Comparing only the states at the release cannot tell
        a binding clause from one that has been satisfied for thousands of evals.
        """
        w = self._watch
        e = len(st.history_f)
        if w["start"] is None:
            w["start"] = e
        if w["sigma"] is None and (
                st.sigma <= self.exhausted_sigma_tol * st.span * self.sigma_floor_ratio):
            w["sigma"] = e
        if w["level"] is None and st.has_exhausted and self.hunt_level_tol > 0.0 \
                and st.basin_best <= self.hunt_level_tol * st.f_init_scale:
            w["level"] = e
        mult = (self.hunt_no_improve_mult
                if (st.has_exhausted and self.hunt_no_improve_mult > 0.0)
                else self.exhausted_no_improve_mult)
        if w["stag"] is None and st.no_improve >= mult * self._stagnation_window():
            w["stag"] = e
        elif w["stag"] is not None and st.no_improve == 0:
            w["stag"] = None            # an improvement reset the counter

    def _record_generation(self, st: _MCESOState) -> None:
        super()._record_generation(st)
        self._hunt_watch(st)
        self._exh_this_gen = []

    def _on_spillover_start(self, st: _MCESOState, basin_switch: bool) -> None:
        """Snapshot the release decision, then let the base archive the basin.

        The first ``_basin_exhausted`` call of this generation is the one
        ``_spillover_should_fire`` made the decision with; later calls (the
        basin-switch test, the reseed) see the same generation but a possibly
        flipped ``has_exhausted``.
        """
        dec = dict(self._exh_this_gen[0]) if self._exh_this_gen else {}
        w = self._watch
        # Binding side of the conjunction: the depth side (sigma floor OR level)
        # became satisfiable at the earlier of the two, and the release needs
        # stagnation as well, so whichever of the two turned true last decided
        # when the hunt stopped.
        depth_at = [v for v in (w["sigma"], w["level"]) if v is not None]
        depth_at = min(depth_at) if depth_at else None
        binding = ""
        if depth_at is not None and w["stag"] is not None:
            binding = "stagnation" if w["stag"] > depth_at else "depth"
        rec = {
            "spill_i": len(self._records),
            "evals": len(st.history_f),
            "basin_switch": bool(basin_switch),
            "pop_best": float(np.min(st.pop_f)) if len(st.pop_f) else float("nan"),
            "hunt_start": w["start"],
            "first_sigma": w["sigma"],
            "first_level": w["level"],
            "first_stag": w["stag"],
            "binding": binding,
            **dec,
        }
        before = {id(a) for a in st.sol_archive_x}
        super()._on_spillover_start(st, basin_switch)
        for a in st.sol_archive_x:
            if id(a) not in before:
                self._rec_by_id[id(a)] = rec
        self._records.append(rec)
        # The reseed below starts a fresh hunt: clear the per-hunt watch.
        self._watch = dict.fromkeys(("start", "sigma", "level", "stag"))

    # ── read-out ────────────────────────────────────────────────────────────
    def spill_records(self) -> list[dict]:
        """One record per spillover, in order."""
        return list(self._records)

    def solution_tags(self) -> list[dict]:
        """Provenance of every entry of ``final_solutions``, in the same order.

        Each entry has ``source`` in ``{"pop", "ir_archive", "sol_archive"}``;
        ``sol_archive`` entries carry the release record of the spillover that
        produced them (``None`` only if the archive was pruned in a way that
        loses identity, which the base class does not do).
        """
        st = self._st
        assert st is not None, "call optimize() first"
        tags: list[dict] = [{"source": "pop", "rec": None} for _ in st.pop_x]
        tags += [{"source": "ir_archive", "rec": None} for _ in st.ir_archive_x]
        tags += [{"source": "sol_archive", "rec": self._rec_by_id.get(id(a))}
                 for a in st.sol_archive_x]
        return tags
