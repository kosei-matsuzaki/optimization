"""Diagnostic MC-ESO variant: a basin-switch restart that **commits to one
draw** instead of racing twenty of them.

Why. On an exhausted basin switch ``MultiChannelEpidemicOptimizer.
_maybe_spillover`` re-seeds every one of the ``n_pop = max(20, 4*dim)`` slots
(``div_ratio = 1.0``) with an *independent* repelled uniform draw, then the
mu+lambda greedy of the following generations keeps the best of them. Entry 22
of research_loop measured both halves of that on N07-Vincent2D (3 seeds, 20k
evals, ``scripts/reseed_to_landing.py``):

  draws touched   36 / 36 distinct optima        (coverage is already there)
  landings        11 / 36 distinct optima        (coverage is lost downstream)
  spacing of the optimum landed in  median 1.918, drawn near  median 1.023

and it did not move when the repel radius was made basin-adaptive, nor when
sigma_init was cut from 0.2*span to 0.05*span (landings 11 -> 14 -> 15). So the
draw distribution is not the binding constraint: twenty independent draws land
in many basins, and the best-of-twenty selection systematically keeps the one in
the widest basin. Whatever the draws cover, the selection un-covers.

Fix under test. Make the restart a *commitment*: draw once, then place the whole
population around that single anchor, so there is no cross-basin competition for
the greedy to resolve. The spread is set by the basin scale the run has already
observed, not by the box:

  anchor      the first ``_diversified_reseed`` call of the event, drawn with
              whichever repel rule ``repel_mode`` selects (``"off"`` reproduces
              the shipped fixed 0.02*span radius, so ``commit`` isolates the
              commitment itself from entry 22's radius change).
  sigma_place ``commit_sigma_ratio * d_local``, where ``d_local`` is the
              nearest-neighbour distance, among the de-duplicated basin
              centroids in ``ir_basin_centroids``, of the centroid nearest the
              anchor -- reusing ``AdaptiveRepelMCESO._adaptive_radii``'s
              geometry, so it costs no extra evaluations. Clipped to
              ``[0.005, 0.2] * span``; falls back to the base per-slot draw
              while fewer than three distinct basins have been drilled.

``commit_sigma_mode`` chooses how far the commitment reaches:

  ``"place"`` only the placement of the slots is committed; the generation
              sigma stays at the sigma_init the base restart sets. Isolates
              "stop racing 20 draws".
  ``"run"``   the post-restart sigma is set to sigma_place too, so the hunt
              *searches* at the basin scale rather than the box scale.

This file exists so the diagnosis never touches the shipped optimiser: MC-ESO's
defaults are unchanged, and ``mceso.py`` / ``mceso_adaptive_repel.py`` are not
edited. ``commit_mode="off"`` is the inherited behaviour exactly.
"""
from __future__ import annotations

import numpy as np

from .mceso_adaptive_repel import AdaptiveRepelMCESO


class CommitReseedMCESO(AdaptiveRepelMCESO):
    """MC-ESO whose exhausted basin switch commits the population to one draw.

    Every other code path is inherited unchanged, so ``commit_mode="off"`` with
    ``repel_mode="off"`` is bit-identical to the shipped optimiser.
    """

    def __init__(self, *args, commit_mode: str = "on",
                 commit_sigma_ratio: float = 0.25,
                 commit_sigma_mode: str = "place",
                 commit_sigma_lo: float = 0.005,
                 commit_sigma_hi: float = 0.2,
                 repel_mode: str = "off", **kwargs) -> None:
        super().__init__(*args, repel_mode=repel_mode, **kwargs)
        if commit_mode not in ("off", "on", "sigma_only"):
            raise ValueError(f"unknown commit_mode {commit_mode!r}")
        if commit_sigma_mode not in ("place", "run"):
            raise ValueError(f"unknown commit_sigma_mode {commit_sigma_mode!r}")
        self.commit_mode = commit_mode
        self.commit_sigma_ratio = commit_sigma_ratio
        self.commit_sigma_mode = commit_sigma_mode
        self.commit_sigma_lo = commit_sigma_lo
        self.commit_sigma_hi = commit_sigma_hi
        self._commit_anchor: np.ndarray | None = None
        self._commit_sigma: float | None = None
        self._commit_failed = False
        # Diagnostics: how often the commitment actually engaged, and how wide
        # it was. A variant that silently fell back every time must not be read
        # as a null result about commitment.
        self.n_commit = 0
        self.n_commit_fallback = 0
        self.commit_sigmas: list[float] = []

    # -- the local basin scale, from centroids already observed --------------
    def _commit_scale(self, st, anchor: np.ndarray) -> float | None:
        """Spread for the committed population, or ``None`` if not estimable.

        Uses the same de-duplicated centroid geometry as ``_adaptive_radii``:
        collapse repeat landings (a second landing in a basin already held must
        not drive that basin's nearest-neighbour distance to zero), then take
        the nearest-neighbour distance of the surviving centroid closest to the
        anchor. That is a *local* estimate of how far apart basins are in this
        part of the box, which is the point -- Vincent's optima are log-spaced,
        so a single global number is wrong at one end or the other.
        """
        if not st.ir_basin_centroids:
            return None
        span = float(st.span)
        base_r = 0.02 * span
        reps: list[np.ndarray] = []
        for c in np.asarray(st.ir_basin_centroids, dtype=float):
            if reps and np.min(np.linalg.norm(np.asarray(reps) - c, axis=1)) <= base_r:
                continue
            reps.append(c)
        if len(reps) < 3:
            return None
        R = np.asarray(reps)
        D = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        d_nn = D.min(axis=1)
        near = int(np.argmin(np.linalg.norm(R - anchor, axis=1)))
        return float(np.clip(self.commit_sigma_ratio * d_nn[near],
                             self.commit_sigma_lo * span,
                             self.commit_sigma_hi * span))

    # -- the restart --------------------------------------------------------
    def _diversified_reseed(self, st, x_best_snap) -> np.ndarray:
        if self.commit_mode == "off" or not self._basin_exhausted(st):
            return super()._diversified_reseed(st, x_best_snap)
        if self.commit_mode == "sigma_only":
            # Control: keep the base restart exactly -- every slot draws
            # independently, so the best-of-n_pop race is still there -- and only
            # take the post-restart sigma down to the local basin scale. Isolates
            # "search at the basin scale" from "commit to one draw".
            cand = np.asarray(super()._diversified_reseed(st, x_best_snap),
                              dtype=float)
            if self._commit_sigma is None and not self._commit_failed:
                sigma = self._commit_scale(st, cand)
                if sigma is None:
                    self._commit_failed = True
                    self.n_commit_fallback += 1
                else:
                    self._commit_sigma = sigma
                    self.n_commit += 1
                    self.commit_sigmas.append(sigma)
            return cand
        if self._commit_anchor is None:
            # First slot of this event: draw the anchor with the configured
            # repel rule and read the local basin scale off it.
            anchor = np.asarray(super()._diversified_reseed(st, x_best_snap),
                                dtype=float)
            sigma = self._commit_scale(st, anchor)
            if sigma is None:
                # Too few distinct basins drilled to estimate a scale: leave the
                # event on the base per-slot path. Counted once per event, not
                # once per slot (every slot re-enters this branch).
                if not self._commit_failed:
                    self._commit_failed = True
                    self.n_commit_fallback += 1
                return anchor
            self._commit_anchor = anchor
            self._commit_sigma = sigma
            self.n_commit += 1
            self.commit_sigmas.append(sigma)
            return anchor
        # Every other slot: a tight cloud around the anchor, no second draw and
        # so no cross-basin race for the greedy to resolve.
        return self._reflect(
            self._commit_anchor + self._commit_sigma * st.rng.standard_normal(self.dim),
            st.lo, st.hi)

    def _maybe_spillover(self, st) -> bool:
        # Anchor scope is one spillover event.
        self._commit_anchor = None
        self._commit_sigma = None
        self._commit_failed = False
        out = super()._maybe_spillover(st)
        if self.commit_sigma_mode == "run" and self._commit_sigma is not None:
            # The base restart has just set sigma to sigma_init (0.2 * span);
            # search the committed basin at the basin's own scale instead.
            st.sigma = float(self._commit_sigma)
        return out
