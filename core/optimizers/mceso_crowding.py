"""MC-ESO with crowding replacement — testing the acceptance-rule principle.

The audit found that what makes a method's reported set fragile under an
unmodelled criterion is not its mutation or its diversity mechanism but its
acceptance rule: DE, whose trial competes with its parent, loses 13 net
functions as the unmodelled component grows, while Crowding-DE, identical
except that a trial competes with its nearest neighbour, loses nothing.

MC-ESO sits on the fragile side (host competition rolls a child back against
the host it replaced, so the population is free to pile into one basin). If the
principle is right, swapping in nearest-neighbour replacement should buy tilt
robustness. If it does not, the principle is incomplete — which is the more
interesting outcome, since MC-ESO also pulls toward strain elites through the
droplet channel, and that pull would then be the part that matters.

This subclass changes exactly one thing: which incumbent a child must beat.
"""
from __future__ import annotations
import numpy as np

from .base import OptimizeResult                              # noqa: F401
from .mceso import MultiChannelEpidemicOptimizer, _MCESOState


class MCESOCrowding(MultiChannelEpidemicOptimizer):
    """Children compete with their nearest host instead of the one they replaced."""

    def _place_and_compete(
        self, st: _MCESOState, new_xs: np.ndarray, sigma_children: np.ndarray,
        n_dead: int, dead_global: np.ndarray,
        dead_orig_x: np.ndarray | None, dead_orig_f: np.ndarray | None,
    ) -> None:
        # Snapshot the population before any placement: "nearest host" has to
        # mean nearest in the population the child was generated from, not one
        # that already contains this generation's other children.
        pop_before_x = st.pop_x.copy()
        pop_before_f = st.pop_f.copy()

        for k in range(min(n_dead, len(new_xs))):
            slot = int(dead_global[k])
            x = new_xs[k]
            f = float(self.func(x))
            sigma_used_k = (float(sigma_children[k])
                            if k < len(sigma_children) else float(st.sigma))
            self._record_eval(st, x, f, sigma_used_k)

            # Crowding: the child displaces its nearest host if it is better,
            # otherwise the slot keeps whoever was in it. A child that improves
            # on a distant basin therefore cannot evict a good host nearby, and
            # separated basins survive together.
            d = np.linalg.norm(pop_before_x - x, axis=1)
            near = int(np.argmin(d))
            if f < pop_before_f[near]:
                st.pop_x[near] = x
                st.pop_f[near] = f
                st.pop_age[near] = 0
            elif dead_orig_x is not None:
                st.pop_x[slot] = dead_orig_x[k]
                st.pop_f[slot] = dead_orig_f[k]
            if not st.budget_left:
                break

        # The learned covariance is fed by children that beat a host; with
        # crowding the relevant comparison is against the nearest one.
        if self._cc_dim_gate() > 0.0:
            self._update_cc_cov(st, [])

        st.pop_age += 1
