"""MC-ESO with a *phased* acceptance rule — depth first, breadth after.

The audit theme closed on "one knob cannot hold depth and breadth at the same
time": parent comparison buys depth, nearest-neighbour (crowding) replacement
buys breadth, and moving the single knob trades one for the other (MC-ESO's
SR@1e-10 goes 100% -> 0% when the acceptance rule is swapped wholesale, see
``mceso_crowding.py``). Entry 27 re-confirmed that in 3D on the sigma side: the
local-basin-spacing ratio turned out to be a monotone trade-off dial with no
interior optimum.

This variant does not move a knob — it splits the run into two phases and gives
each phase its own rule:

* **before the first exhaustion** (``st.has_exhausted`` False): the shipped
  global mu+lambda host competition. This is the phase that has to reach
  FP precision on unimodal problems, so it is left untouched — the BBOB-24
  dim2 SR@1e-10 asset lives here.
* **after it** (later hunts): crowding, i.e. a child must beat its *nearest*
  host rather than the one it displaced. Entry 22 measured that what limits the
  restart's coverage is the selection, not the draw: a basin switch redraws all
  20 slots repelled and the draws do cover all 36 optima, yet the best-of-20
  greedy lands on 11-15 because it systematically prefers points in wide
  basins. Crowding removes exactly that pressure without touching sigma.

``accept_phase="off"`` inherits every path unchanged and is therefore
bit-identical to the shipped optimiser; ``accept_phase="always"`` delegates to
``MCESOCrowding`` and must reproduce it exactly. MC-ESO's defaults are
unchanged and ``mceso.py`` is not edited.
"""
from __future__ import annotations

import numpy as np

from .mceso import MultiChannelEpidemicOptimizer, _MCESOState
from .mceso_crowding import MCESOCrowding


class PhasedAcceptMCESO(MultiChannelEpidemicOptimizer):
    """Host competition while the first basin is being drilled, crowding after.

    ``accept_phase``:
      ``off``        always the base rule (identity check)
      ``exhausted``  crowding once the run has exhausted a basin at least once
      ``always``     crowding from the first generation (= ``MCESOCrowding``)
    """

    def __init__(self, *args, accept_phase: str = "exhausted", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if accept_phase not in ("off", "exhausted", "always"):
            raise ValueError(f"unknown accept_phase {accept_phase!r}")
        self.accept_phase = accept_phase
        # Diagnostics: a variant that never switched phase must not be read as a
        # null result about phasing.
        self.n_gen_base_rule = 0
        self.n_gen_crowd_rule = 0

    def _crowding_active(self, st: _MCESOState) -> bool:
        if self.accept_phase == "off":
            return False
        if self.accept_phase == "always":
            return True
        return bool(getattr(st, "has_exhausted", False))

    def _place_and_compete(
        self, st: _MCESOState, new_xs: np.ndarray, sigma_children: np.ndarray,
        n_dead: int, dead_global: np.ndarray,
        dead_orig_x: np.ndarray | None, dead_orig_f: np.ndarray | None,
    ) -> None:
        if self._crowding_active(st):
            self.n_gen_crowd_rule += 1
            # Reuse the audited crowding body verbatim rather than restating it,
            # so `always` is bit-identical to MCESOCrowding by construction.
            return MCESOCrowding._place_and_compete(
                self, st, new_xs, sigma_children, n_dead, dead_global,
                dead_orig_x, dead_orig_f)
        self.n_gen_base_rule += 1
        return super()._place_and_compete(
            st, new_xs, sigma_children, n_dead, dead_global,
            dead_orig_x, dead_orig_f)
