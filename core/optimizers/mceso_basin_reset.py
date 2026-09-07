"""`no_improve` reset by the basin's own progress (entry 77's diagnostic arm).

Entry 77 introduced this arm in `analysis/hm/e77/basin_reset.py` and measured it
on N18-CF3-10D: PR@1e-3 0.194 -> 0.306 (6/6/0, p = 0.031), PR@1e-5 0.167 ->
0.292 (7/5/0, p = 0.016, A12 = 0.79), the first time the depth side moved on the
multimodal theme. What it did not measure is the price: `no_improve` also drives
the restart path, the sigma control and `_basin_exhausted`, so the arm has to go
through the BBOB-24 dim2 gate (entries 34 / 37 / 47 / 56) before it can be an
adoption candidate. `scripts/gate_power.py --basin-reset` runs that pairing, and
this module is where the gate reads the class from; the class body below is a
verbatim copy of entry 77's, so the two measurements are of the same arm.

**Nothing here changes a shipped default.** `MultiChannelEpidemicOptimizer` is
untouched; this is a diagnostic subclass in the sense of CLAUDE.md's rule.
"""
from __future__ import annotations

import math

from .mceso import MultiChannelEpidemicOptimizer


class BasinResetMCESO(MultiChannelEpidemicOptimizer):
    """`no_improve` is reset by the basin's own progress, not only the global's.

    `basin_reset=False` disables the override and must reproduce base bit for
    bit (identity check, entry 74's seed 0: best_f = 6.22882e-12, n_rep = 246).
    """

    def __init__(self, *a, basin_reset: bool = True, **kw) -> None:
        super().__init__(*a, **kw)
        self.basin_reset = bool(basin_reset)

    def _init_state(self, max_evals):
        st = super()._init_state(max_evals)
        self._b_ref = math.log10(max(st.basin_best, 1e-300))
        self._b_since = 0
        self._b_seen = st.basin_best
        self._n_basin_resets = 0
        return st

    def _record_eval(self, st, x, f, sigma_used) -> None:
        bb_pre = st.basin_best
        ni_pre = st.no_improve
        super()._record_eval(st, x, f, sigma_used)
        if not self.basin_reset:
            return
        if st.basin_best > self._b_seen:
            #  A re-seed raised basin_best: a new hunt starts here.
            self._b_ref = math.log10(max(st.basin_best, 1e-300))
            self._b_since = 0
        elif st.no_improve == 0 and ni_pre != 0:
            #  The global path already reset the counter; re-anchor so the two
            #  references cannot drift apart.
            self._b_ref = math.log10(max(st.basin_best, 1e-300))
            self._b_since = 0
        else:
            self._b_since += 1
            if f < bb_pre and self._meaningful_improvement(
                    f, self._b_ref, self._b_since):
                #  Only the stagnation counter moves. `log_best_ref` and
                #  `evals_since_reset` belong to the global path and are left
                #  alone, so this arm changes exactly one thing.
                st.no_improve = 0
                self._b_ref = math.log10(max(f, 1e-300))
                self._b_since = 0
                self._n_basin_resets += 1
        self._b_seen = st.basin_best

