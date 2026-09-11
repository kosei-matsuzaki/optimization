"""MC-ESO with a per-spillover trace. Recording only — the search is untouched.

Why this exists (entry 113, queue 2). MC-ESO loses to the memoryless multistart
null by 4.5x in MPR and 5.7x in Score on the GECCO'2024 suite, and nothing in
the record says *which* stage loses it. The null's own dump
(``RESTART_LANDER_DUMP``, core/optimizers/restart_lander.py) already writes one
row per descent — draw, evals, landing optimum — so the only thing missing is
the same row for MC-ESO. This supplies it.

The unit. The null's "hunt" is one uniform draw plus one CMA descent. MC-ESO has
no such object: it runs a population of ``n_pop`` slots and re-seeds them at a
*spillover*. The matching unit is therefore **the search segment between two
spillovers** — a segment starts with a fresh set of re-seed draws and ends when
the population stalls and is re-seeded again. Segment count is MC-ESO's "hunt
count"; the population best at the moment of the next spillover is the segment's
landing point, which is what ``_on_spillover_start`` already archives into
``sol_archive_x`` (that archive is capped at 200 and re-sorted, so it cannot be
used to count segments — hence a separate, uncapped list here).

Two dumps per run, both mirroring restart_lander's column names so one reader
serves both methods:

* ``<problem>_seed<N>_segments.csv.gz`` — one row per spillover, plus a terminal
  row for the segment the budget ends inside:
  ``segment, evals, best_f, land_opt, dist, basin_switch``
  (``basin_switch`` is -1 on the terminal row).
* ``<problem>_seed<N>_draws.csv.gz`` — one row per population slot immediately
  after each re-seed: ``segment, slot, f, land_opt, dist, kind``. ``kind`` is
  ``reseed`` for a fresh draw and ``retained`` for the single slot an ordinary
  spillover keeps (the pre-spillover best). This is the draw stage: what the
  segment was *offered*, before selection narrowed it to one basin.

Nothing here consumes the RNG, spends an evaluation, or changes any default: the
two overridden hooks call ``super()`` and only read state, and ``optimize`` is
inherited. The identity check against the base class is in
``analysis/mmo2024/e113/identity_check.py``.
"""
from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path

import numpy as np

from core.optimizers.mceso import MultiChannelEpidemicOptimizer, _MCESOState


class TracedMCESO(MultiChannelEpidemicOptimizer):
    """MC-ESO plus a per-spillover / per-draw trace. Search is bit-identical."""

    def __init__(self, benchmark, seed: int = 42, **kw):
        super().__init__(benchmark, seed, **kw)
        self.segments: list[dict] = []
        self.draws: list[dict] = []
        self._seg_start_evals = 0
        self._fired = False
        self._retained_slot = -1
        self._last_st: _MCESOState | None = None

    # ── hooks (recording only) ──────────────────────────────────────────────
    def _on_spillover_start(self, st: _MCESOState, basin_switch: bool) -> None:
        """Fires exactly once per spillover, before the re-seed overwrites the
        population — so this is the end of a segment."""
        self._fired = True
        if len(st.pop_f):
            i = int(np.argmin(st.pop_f))
            self._retained_slot = -1 if basin_switch else i
            self.segments.append({
                "segment": len(self.segments),
                "evals": len(st.history_f) - self._seg_start_evals,
                "best_f": float(st.pop_f[i]),
                "x": st.pop_x[i].copy(),
                "basin_switch": int(basin_switch),
            })
        self._seg_start_evals = len(st.history_f)
        super()._on_spillover_start(st, basin_switch)

    def _maybe_spillover(self, st: _MCESOState) -> bool:
        self._fired = False
        out = super()._maybe_spillover(st)
        if self._fired:
            seg = len(self.segments)        # the segment these draws open
            for i in range(len(st.pop_f)):
                self.draws.append({
                    "segment": seg, "slot": i, "f": float(st.pop_f[i]),
                    "x": st.pop_x[i].copy(),
                    "kind": "retained" if i == self._retained_slot else "reseed",
                })
        return out

    def _record_generation(self, st: _MCESOState) -> None:
        self._last_st = st                  # for the terminal segment
        super()._record_generation(st)

    def optimize(self, max_evals: int = 5000):
        result = super().optimize(max_evals)
        st = self._last_st
        if st is not None and len(st.pop_f):
            i = int(np.argmin(st.pop_f))
            self.segments.append({
                "segment": len(self.segments),
                "evals": len(st.history_f) - self._seg_start_evals,
                "best_f": float(st.pop_f[i]),
                "x": st.pop_x[i].copy(),
                "basin_switch": -1,         # marks the terminal segment
            })
        self._dump()
        return result

    # ── dump ────────────────────────────────────────────────────────────────
    def _dump(self) -> None:
        d = os.environ.get("MCESO_HUNT_DUMP")
        if not d or not self.segments:
            return
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)
        opts = np.asarray(self.benchmark.optima_pos, dtype=float)
        stem = f"{self.benchmark.name}_seed{self.seed}"

        def nearest(x):
            dd = np.linalg.norm(opts - np.asarray(x, dtype=float), axis=1)
            j = int(np.argmin(dd))
            return j, float(dd[j])

        with gzip.open(out / f"{stem}_segments.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["segment", "evals", "best_f", "land_opt", "dist",
                        "basin_switch"])
            for r in self.segments:
                j, dist = nearest(r["x"])
                w.writerow([r["segment"], r["evals"], f"{r['best_f']:.12g}",
                            j, f"{dist:.6g}", r["basin_switch"]])

        if not self.draws:
            return
        with gzip.open(out / f"{stem}_draws.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["segment", "slot", "f", "land_opt", "dist", "kind"])
            for r in self.draws:
                j, dist = nearest(r["x"])
                w.writerow([r["segment"], r["slot"], f"{r['f']:.12g}",
                            j, f"{dist:.6g}", r["kind"]])
