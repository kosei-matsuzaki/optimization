"""Diagnostic variant: make MC-ESO's answer archive trim niche-aware.

Entry 58 located the third stage of N09-Vincent3D's loss chain inside MC-ESO
itself. ``_on_spillover_start`` appends the best point of every abandoned basin
to ``sol_archive`` and, once the archive passes ``solution_archive_max`` (200,
``mceso.py:381``), trims it with ``np.argsort(sol_archive_f)[:200]``
(``mceso.py:822-825``) — **by objective value alone, blind to which niche a
point sits in**. On Vincent every global optimum has the same f, so the trim
keeps 200 numerically-best points that are free to be duplicates of a handful
of wide basins. Measured: the 676 hunt endpoints of a 4e5-evaluation run hold
59.20 rho-separated niches, the f-only top-200 keeps 37.33 (15/15 seeds,
p = 0.00064), and PR@1e-5 falls 0.274 -> 0.173.

Two arms, both costing **zero extra evaluations** — ``sol_archive`` is read
only when the reported set is assembled (``mceso.py:757``) and never feeds back
into the search, so the trajectory, ``best_f`` and every hunt are bit-identical
to base:

  ``capacity``  raise ``solution_archive_max`` so the trim never fires. Needs no
                code at all (it is a constructor argument of the shipped class);
                it is the upper bound on what any trim rule can recover, and is
                run from ``diagnose_niching.py`` as a plain kwarg.
  ``rho``       keep the capacity at 200 and make the trim niche-greedy, which
                is what this file implements. If it matches ``capacity`` the fix
                costs no memory, which is why it is the adoption shape.

The radius is **not** the benchmark's scoring rho — using that would be reading
the answer key. It is ``sol_trim_radius_ratio x span`` with the default 0.02,
i.e. the same fine repulsion radius the reseed draw already uses
(``mceso.py:874``), so the rule stays scale-free and benchmark-blind. On Vincent
that is 0.195 against a scoring rho of 0.2, but nothing here depends on the two
agreeing: the trim only has to stop the archive from spending its 200 slots on
near-duplicates.

``sol_trim_mode="off"`` restores the shipped f-only trim exactly and is the
identity check for the class.
"""
from __future__ import annotations

import numpy as np

from .mceso import MultiChannelEpidemicOptimizer


class SolArchiveTrimMCESO(MultiChannelEpidemicOptimizer):
    """MC-ESO with a niche-greedy answer-archive trim.

    Only ``_on_spillover_start``'s archive-keeping block is overridden. The
    override runs *after* ``super()`` has done everything else, so the search
    state, the RNG stream and the number of evaluations are untouched.
    """

    def __init__(self, *args,
                 sol_trim_mode: str = "rho",     # "rho" | "off" (shipped f-only)
                 sol_trim_radius_ratio: float = 0.02,   # x span
                 **kwargs):
        super().__init__(*args, **kwargs)
        if sol_trim_mode not in ("rho", "off"):
            raise ValueError(f"sol_trim_mode={sol_trim_mode!r}")
        self.sol_trim_mode = sol_trim_mode
        self.sol_trim_radius_ratio = float(sol_trim_radius_ratio)

    def _on_spillover_start(self, st, basin_switch: bool) -> None:
        # Let the shipped path append this basin's best point and apply its
        # f-only trim; then, if the trim just fired, redo the *selection* on the
        # pre-trim archive. Redoing it needs the points the base trim discarded,
        # so snapshot the archive first (cheap: <= max + 1 references).
        pre_x = list(st.sol_archive_x)
        pre_f = list(st.sol_archive_f)
        out = super()._on_spillover_start(st, basin_switch)
        if self.sol_trim_mode == "off" or self.solution_archive_max <= 0:
            return out
        cap = self.solution_archive_max
        # The append happened inside super(); reconstruct the pre-trim archive
        # exactly as the base path saw it (snapshot + the point it appended).
        if len(pre_f) + 1 <= cap:
            return out                      # trim did not fire — nothing to redo
        if not len(st.pop_f):
            return out
        best_i = int(np.argmin(st.pop_f))
        cand_x = pre_x + [np.asarray(st.pop_x[best_i], dtype=float).copy()]
        cand_f = pre_f + [float(st.pop_f[best_i])]

        keep = self._rho_greedy_keep(np.asarray(cand_x, dtype=float),
                                     np.asarray(cand_f, dtype=float),
                                     cap, self.sol_trim_radius_ratio * st.span)
        st.sol_archive_x = [cand_x[i].copy() for i in keep]
        st.sol_archive_f = [float(cand_f[i]) for i in keep]
        return out

    @staticmethod
    def _rho_greedy_keep(X: np.ndarray, F: np.ndarray, cap: int,
                         radius: float) -> list[int]:
        """Indices of the `cap` points to keep: best-f first, a point accepted
        only if it is further than `radius` from every point already accepted.

        Separation is the *only* thing that changes; depth still orders the
        candidates, so the deepest point of the archive is always kept and the
        rule degenerates to the shipped f-only trim when every point is mutually
        separated. If fewer than `cap` mutually separated points exist the
        remaining slots are filled by f, so the arm never reports *fewer* points
        than base — the comparison is about which points, not how many.
        """
        order = np.argsort(F, kind="stable")
        kept: list[int] = []
        kept_x: list[np.ndarray] = []
        rejected: list[int] = []
        for i in order:
            i = int(i)
            if len(kept) >= cap:
                rejected.append(i)
                continue
            if kept_x and np.min(np.linalg.norm(
                    np.asarray(kept_x) - X[i], axis=1)) <= radius:
                rejected.append(i)
                continue
            kept.append(i)
            kept_x.append(X[i])
        if len(kept) < cap:                 # top up by f, best first
            kept.extend(rejected[:cap - len(kept)])
        return kept
