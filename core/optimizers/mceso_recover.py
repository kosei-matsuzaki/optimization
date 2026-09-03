"""Diagnostic MC-ESO variant: a **recovery phase** that goes back into the
basins the run found but never drilled out, instead of always repelling away.

Why. Entries 26/27 (sigma scale) and 29 (acceptance rule) closed the rule-side
levers: two structurally unrelated interventions move along the *same*
depth<->breadth line, so nothing that only rewrites a rule escapes it while the
evaluations landing in one hunt stay the same. What is left is the *allocation*.

The shipped restart only ever leaves: ``_diversified_reseed`` on an exhausted
basin is a repelled uniform draw, so a niche that was entered and abandoned
shallow is never visited again. Entry 29 counted that inventory directly at
eps=1e-3: N06 holds 20.2 and N07 15.7 reported niches that miss eps (`blocked`)
under the crowding phase, against 5.2 / 3.2 for base.

What this variant does. Once the run has spent ``recover_start_frac`` of its
budget, an exhausted basin switch stops drawing a fresh repelled point and
instead **re-enters a recorded under-drilled basin**, placing the whole
population in a tight cloud around it and searching at the basin's own scale.
No extra evaluations are spent: the anchors are the centroids
``_on_spillover_start`` already records.

  candidate   a basin abandoned with ``f > hunt_level_tol * f_init_scale`` --
              i.e. the hunt was released without matching the depth the run had
              already banked ("found the basin, did not drill it out"). This is
              the optimiser's own equal-height landmark (``mceso.py:928``), not
              a reference to eps or to the known optima, so the rule is legal
              on a black-box problem.
  choice      the best-f candidate not already recovered (nearest to being
              drilled out), de-duplicated against earlier re-entries with the
              same 0.02*span radius the base repel rule uses.
  sigma       ``commit_sigma_ratio * d_local`` from
              ``CommitReseedMCESO._commit_scale`` -- the same de-duplicated
              centroid geometry, so no new estimator is introduced. Setting the
              post-restart sigma is not optional here: at the base sigma_init
              (0.2*span) the population is blown straight back out of the basin
              and the re-entry is a no-op.

``recover_mode`` selects the arm:

  ``"off"``      inherited behaviour exactly (identity check against base).
  ``"blocked"``  the treatment: re-enter recorded under-drilled basins.
  ``"fresh"``    the control that isolates *where* the restart goes. Identical
                 phase, identical tight sigma, but the anchor is the ordinary
                 repelled uniform draw. Entries 23/25 showed a basin-scale sigma
                 alone already buys depth in 2D, so a "blocked" gain only counts
                 as a gain of the *return* if it beats this arm.

MC-ESO's defaults are untouched: this file subclasses ``CommitReseedMCESO`` with
``commit_mode="off"`` / ``repel_mode="off"``, which is the shipped optimiser.
"""
from __future__ import annotations

import numpy as np

from .mceso_commit_reseed import CommitReseedMCESO


class RecoverMCESO(CommitReseedMCESO):
    """MC-ESO whose late basin switches return to under-drilled basins."""

    def __init__(self, *args, recover_mode: str = "blocked",
                 recover_start_frac: float = 0.8,
                 recover_sigma_ratio: float = 0.25,
                 recover_sigma_fallback: float = 0.02,
                 **kwargs) -> None:
        kwargs.setdefault("commit_mode", "off")
        kwargs.setdefault("repel_mode", "off")
        # _commit_scale reads commit_sigma_ratio; the commitment itself is off,
        # so the field is free to carry the recovery spread.
        kwargs.setdefault("commit_sigma_ratio", recover_sigma_ratio)
        super().__init__(*args, **kwargs)
        if recover_mode not in ("off", "blocked", "fresh"):
            raise ValueError(f"unknown recover_mode {recover_mode!r}")
        self.recover_mode = recover_mode
        self.recover_start_frac = recover_start_frac
        self.recover_sigma_fallback = recover_sigma_fallback
        # Recording only -- the abandoned-basin inventory, uncapped. st.sol_archive
        # cannot serve: it is trimmed to the best solution_archive_max by f, which
        # drops exactly the shallow basins this variant is looking for.
        self._ab_x: list[np.ndarray] = []
        self._ab_f: list[float] = []
        self._recovered: list[np.ndarray] = []
        self._rec_anchor: np.ndarray | None = None
        self._rec_sigma: float | None = None
        self._rec_failed = False
        # Diagnostics: a variant that silently fell back every time must not be
        # read as a null result about recovery.
        self.n_recover = 0
        self.n_recover_fallback = 0
        self.recover_sigmas: list[float] = []
        self.recover_depths: list[float] = []

    # -- the inventory ------------------------------------------------------
    def _on_spillover_start(self, st, basin_switch: bool) -> None:
        """Record the basin being abandoned (position and depth), then defer."""
        if len(st.pop_f):
            best_i = int(np.argmin(st.pop_f))
            self._ab_x.append(np.asarray(st.pop_x[best_i], dtype=float).copy())
            self._ab_f.append(float(st.pop_f[best_i]))
        return super()._on_spillover_start(st, basin_switch)

    def _in_recovery(self, st) -> bool:
        return (self.recover_mode != "off"
                and len(st.history_f) >= self.recover_start_frac * st.max_evals
                and self._basin_exhausted(st))

    def _recover_anchor(self, st) -> np.ndarray | None:
        """Best under-drilled basin not yet re-entered, or ``None``."""
        if not self._ab_x:
            return None
        level = self.hunt_level_tol * st.f_init_scale
        dedup_r = 0.02 * float(st.span)
        prev = np.asarray(self._recovered) if self._recovered else None
        for j in np.argsort(self._ab_f):
            if self._ab_f[j] <= level:
                continue                      # drilled out already -- nothing owed
            x = self._ab_x[j]
            if prev is not None and np.min(np.linalg.norm(prev - x, axis=1)) <= dedup_r:
                continue                      # this basin has had its second visit
            return x.copy()
        return None

    # -- the restart --------------------------------------------------------
    def _diversified_reseed(self, st, x_best_snap) -> np.ndarray:
        if self._rec_failed or not self._in_recovery(st):
            return super()._diversified_reseed(st, x_best_snap)
        if self._rec_anchor is None:
            if self.recover_mode == "fresh":
                # Control: the ordinary repelled draw, tightened. Consumes the
                # same RNG the base path would.
                anchor = np.asarray(super()._diversified_reseed(st, x_best_snap),
                                    dtype=float)
                depth = float("nan")
            else:
                anchor = self._recover_anchor(st)
                if anchor is None:
                    # No under-drilled basin on record: leave the event on the
                    # base path. Counted once per event, not once per slot.
                    self._rec_failed = True
                    self.n_recover_fallback += 1
                    return super()._diversified_reseed(st, x_best_snap)
                depth = float(min(self._ab_f[j] for j, x in enumerate(self._ab_x)
                                  if np.array_equal(x, anchor)))
            sigma = self._commit_scale(st, anchor)
            if sigma is None:
                # Fewer than three distinct basins drilled: no local spacing to
                # read, so fall back to a fixed tight spread rather than to the
                # box scale, which would undo the re-entry.
                sigma = self.recover_sigma_fallback * float(st.span)
            self._rec_anchor = anchor
            self._rec_sigma = float(sigma)
            self._recovered.append(anchor.copy())
            self.n_recover += 1
            self.recover_sigmas.append(float(sigma))
            self.recover_depths.append(depth)
            return anchor
        return self._reflect(
            self._rec_anchor + self._rec_sigma * st.rng.standard_normal(self.dim),
            st.lo, st.hi)

    def _maybe_spillover(self, st) -> bool:
        # Anchor scope is one spillover event.
        self._rec_anchor = None
        self._rec_sigma = None
        self._rec_failed = False
        out = super()._maybe_spillover(st)
        if self._rec_sigma is not None:
            # The base restart has just set sigma to sigma_init (0.2 * span),
            # which would throw the population straight back out of the basin
            # being re-entered.
            st.sigma = float(self._rec_sigma)
        return out
