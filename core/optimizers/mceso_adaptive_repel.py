"""Diagnostic MC-ESO variant: a restart repel radius set by the *observed*
basin scale instead of a fixed fraction of the box.

Why. On Vincent the exhausted-restart path of ``MultiChannelEpidemicOptimizer.
_diversified_reseed`` draws uniformly from the box and rejects a candidate that
falls within ``0.02 * span`` of a basin already drilled. Vincent's optima are
log-spaced (adjacent optima 0.291 to 3.595 apart on [0.25, 10], a 12x range),
so that one radius is wrong at both ends: it is smaller than a *wide* basin, so
it does not push the draw out of a basin already held, and the uniform draw
lands in a basin in proportion to its width anyway. Measured (research_loop
entry 21): 105 of 105 hunts on N07 descend to f <= 1e-5, but 98 of them land in
just 9 of the 36 optima, and the reported set holds 8.7 distinct basins.

Fix under test. The exclusion radius around each drilled basin is read off the
geometry the run has already observed — the mutual distances between the basin
centroids in ``ir_basin_centroids`` — so it costs no extra evaluations:

  1. Collapse centroids that are within ``base_r = 0.02 * span`` of each other.
     That fixed radius is *reliable* as a duplicate test (it is below the
     smallest optimum spacing on both landscapes) even though it is useless as
     a basin-coverage radius; repeat landings in one basin must not drive its
     own nearest-neighbour distance to zero.
  2. For each surviving centroid, ``d_i`` = distance to the nearest other one.
  3. Repel radius ``r_i = clip(alpha * d_i, base_r, max_ratio * span)``.
     With ``alpha = 0.5`` the exclusion balls of two adjacent drilled basins
     just touch: a drilled basin is covered, its neighbour is not blocked.

Two draw rules are offered, both spending the same number of draws the base
method already budgets (``ir_repel_max_tries``):

  ``adaptive``         reject-and-redraw, exactly the base loop with r_i in
                       place of the fixed radius (isolates the radius change).
  ``adaptive_maxmin``  draw them all and keep the candidate maximising
                       ``min_i ||c_i - x|| / r_i`` (isolates what the *draw*
                       adds on top: late in a run, when much of the box is
                       excluded, rejection returns its last failed candidate
                       while max-min returns the most-outside one).

This file exists so the diagnosis never touches the shipped optimiser:
MC-ESO's defaults are unchanged and ``mceso.py`` is not edited.
"""
from __future__ import annotations

import numpy as np

from .mceso import MultiChannelEpidemicOptimizer


class AdaptiveRepelMCESO(MultiChannelEpidemicOptimizer):
    """MC-ESO with a basin-scale-adaptive repel radius on the exhausted restart.

    Every other code path — the informed restart, the reservoir re-ignition, the
    generation loop, the acceptance rule — is inherited unchanged, so a run with
    ``repel_mode="off"`` is bit-identical to the base optimiser.
    """

    def __init__(self, *args, repel_mode: str = "adaptive",
                 repel_alpha: float = 0.5, repel_max_ratio: float = 0.25,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if repel_mode not in ("off", "adaptive", "adaptive_maxmin"):
            raise ValueError(f"unknown repel_mode {repel_mode!r}")
        self.repel_mode = repel_mode
        self.repel_alpha = repel_alpha
        self.repel_max_ratio = repel_max_ratio

    # -- the observed basin scale ------------------------------------------
    def _adaptive_radii(self, centroids: np.ndarray, span: float
                        ) -> np.ndarray | None:
        """Per-centroid repel radius from the mutual centroid distances.

        Returns ``None`` when the run has not yet seen enough *distinct* basins
        to estimate a scale (fewer than three), in which case the caller falls
        back to the base fixed-radius path rather than guessing off one pair.
        """
        base_r = 0.02 * span
        # 1. Duplicate collapse: greedily keep centroids >= base_r apart and
        # remember, for each original centroid, which survivor represents it.
        reps: list[np.ndarray] = []
        owner = np.empty(len(centroids), dtype=int)
        for i, c in enumerate(centroids):
            if reps:
                d = np.linalg.norm(np.asarray(reps) - c, axis=1)
                j = int(np.argmin(d))
                if d[j] <= base_r:
                    owner[i] = j
                    continue
            owner[i] = len(reps)
            reps.append(c)
        if len(reps) < 3:
            return None
        R = np.asarray(reps)
        # 2. Nearest *other* survivor, per survivor.
        D = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        d_nn = D.min(axis=1)
        # 3. Half the way to the neighbour, floored at the duplicate scale and
        # capped so two far-apart early basins cannot black out the box.
        r_rep = np.clip(self.repel_alpha * d_nn, base_r,
                        self.repel_max_ratio * span)
        return r_rep[owner]

    # -- the restart draw ---------------------------------------------------
    def _diversified_reseed(self, st, x_best_snap) -> np.ndarray:
        if self.repel_mode == "off" or not self._basin_exhausted(st):
            return super()._diversified_reseed(st, x_best_snap)
        if not st.ir_basin_centroids:
            return super()._diversified_reseed(st, x_best_snap)

        centroids = np.asarray(st.ir_basin_centroids, dtype=float)
        radii = self._adaptive_radii(centroids, float(st.span))
        if radii is None:
            return super()._diversified_reseed(st, x_best_snap)

        rng, lo, hi, dim = st.rng, st.lo, st.hi, self.dim
        tries = max(1, int(self.ir_repel_max_tries))
        if self.repel_mode == "adaptive":
            cand = rng.uniform(lo, hi, dim)
            for _ in range(tries):
                if np.all(np.linalg.norm(centroids - cand, axis=1) > radii):
                    break
                cand = rng.uniform(lo, hi, dim)
            return cand
        # adaptive_maxmin: same draw budget, keep the most-outside candidate.
        best, best_score = None, -np.inf
        for _ in range(tries):
            cand = rng.uniform(lo, hi, dim)
            score = float(np.min(np.linalg.norm(centroids - cand, axis=1) / radii))
            if score > best_score:
                best, best_score = cand, score
            if best_score > 1.0:
                break
        return best
