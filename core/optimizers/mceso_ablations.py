"""MC-ESO diagnostic ablations.

These two variants exist to test a single hypothesis raised during the
outbreak-dynamics analysis: **is MC-ESO's performance carried by the three
transmission channels + strain coexistence, or merely by the frequent
uniform-reseed restarts (spillover)?**

They are *diagnostic* subclasses, not improvement candidates — they decompose
the full method into its two suspected drivers so a fair 3-way comparison
(full MC-ESO vs each ablation) can attribute the SR.

  • ``MCESONoSpillover`` — channels ON, restart OFF.
        Disables the spillover event entirely (``_maybe_spillover`` always
        returns False). If SR collapses relative to full MC-ESO, the restarts
        are doing the heavy lifting and the channels alone cannot reach the
        optimum on multimodal / weak-structure landscapes.

  • ``MCESORandomRestart`` — restart ON, channels stripped to vanilla local.
        Keeps the *identical* spillover schedule, σ-adaptation, drilling mode,
        μ+λ greedy + rollback, and softmax host selection, but replaces all
        three transmission channels (rotation-aware close-contact, DE droplet,
        airborne) and the niched-elite strain pool with a single **plain
        isotropic Gaussian** local search around softmax-selected parents at
        σ_global. If this matches full MC-ESO, the channels add nothing and
        performance is "best-anchored random restart + vanilla local search"
        — i.e. structurally an IPOP-style restart heuristic.

Together: full ≈ RandomRestart  ⇒ channels redundant (hypothesis confirmed).
          full ≫ NoSpillover    ⇒ restarts carry the SR (hypothesis confirmed).
"""
from __future__ import annotations
import numpy as np

from .base import OptimizeResult
from .mceso import MultiChannelEpidemicOptimizer, _MCESOState


class MCESONoSpillover(MultiChannelEpidemicOptimizer):
    """MC-ESO with the spillover (stagnation re-seed) event disabled.

    Everything else — the three channels, strain coexistence, host competition,
    σ-adaptation, drilling mode — is inherited unchanged. Once the population
    converges into one basin it can only refine there; there is no mechanism to
    escape, so this isolates the channels' standalone reach.
    """

    def _maybe_spillover(self, st: _MCESOState) -> bool:  # noqa: D401
        # Never spill over. no_improve keeps growing but nothing reads it for
        # control once spillover is off, so the run just keeps grinding the
        # current basin until the budget is spent.
        return False


class MCESORandomRestart(MultiChannelEpidemicOptimizer):
    """MC-ESO restart machinery with the three channels stripped to a single
    plain isotropic Gaussian local search.

    Mechanisms KEPT (identical to full MC-ESO):
      • spillover / basin-switch restart schedule
      • σ-adaptation (sigma_up / sigma_down) + drilling mode
      • μ+λ greedy replacement with rollback
      • softmax host (parent) selection

    Mechanisms REMOVED:
      • droplet (DE/current-to-best) channel    → h2h_ratio forced to 0
      • airborne channel                        → air_ratio forced to 0
      • rotation-aware empirical covariance      → close-contact overridden
      • per-host σ_i quality/age scaling         → close-contact overridden
      • niched-elite strain coexistence pool     → _niche_elites returns ∅

    The sole offspring operator becomes ``x_parent + σ_global · N(0, I)``,
    a vanilla (μ+λ)-ES step. The RNG draw order (choice → standard_normal) is
    kept parallel to the original close-contact channel so the comparison is
    apples-to-apples.
    """

    def __init__(self, *args, **kwargs):
        # Force the channel split to close-contact-only; the override below
        # then makes that single channel a plain isotropic Gaussian.
        kwargs["air_ratio"] = 0.0
        kwargs["h2h_ratio"] = 0.0
        super().__init__(*args, **kwargs)

    def _niche_elites(self, pop_x, pop_f, niche_radius):  # noqa: D401
        # Strain coexistence off — the droplet channel (its only consumer) is
        # disabled anyway, but we return ∅ so n_elite logging reflects it.
        return set()

    def _close_contact_children(self, st: _MCESOState, n_local: int,
                                weights: np.ndarray, log_f_max: float,
                                log_f_spread: float) -> tuple[np.ndarray, np.ndarray]:
        """Plain isotropic Gaussian: x_parent + σ_global · N(0, I). No empirical
        covariance, no per-host σ scaling — the vanilla local-search baseline."""
        if n_local <= 0:
            return np.empty((0, self.dim)), np.empty(0)
        rng = st.rng
        gi_arr = rng.choice(self.n_pop, size=n_local, p=weights)
        noise = rng.standard_normal((n_local, self.dim))
        parent_x = st.pop_x[gi_arr].copy()
        new_local = self._reflect(parent_x + noise * st.sigma, st.lo, st.hi)
        sigma_i = np.full(n_local, float(st.sigma))
        return new_local, sigma_i

    # Base MC-ESO now performs an *informed* restart (reservoir re-ignition +
    # basin-memory repulsion). Pin this diagnostic to the original **blind
    # uniform restart** so it keeps testing "uninformed uniform restart +
    # vanilla local search" as it did when the ablation was first run.
    def _on_spillover_start(self, st: _MCESOState, basin_switch: bool) -> None:
        return None

    def _diversified_reseed(self, st: _MCESOState, x_best_snap) -> np.ndarray:
        return st.rng.uniform(st.lo, st.hi, self.dim)


# ── simplification-audit ablations ───────────────────────────────────────────
# Each isolates a *single* mechanism with no standalone ablation record, to
# decide whether it pays its way or can be deleted to simplify MC-ESO. The
# parameter-toggleable ones (airborne channel → air_ratio=0, per-host σ →
# host_sigma_min_scale=1.0, streak basin-switch →
# basin_switch_after_failed_spillovers=∞, convergence-adaptive airborne σ →
# air_sigma_amplifier=0) are expressed as kwargs in quick_check.py; the boundary
# snap has no parameter, so it needs a subclass.
class MCESONoBoundarySnap(MultiChannelEpidemicOptimizer):
    """MC-ESO with the boundary 'snap' clause removed from ``_reflect``.

    Base ``_reflect`` snaps a candidate to the bound when it overshoots by less
    than span×1e-3 (so boundary-optimum landscapes like F05 LinearSlope can land
    *exactly* on the corner instead of orbiting it at σ_floor). This variant uses
    pure reflection on every overshoot, isolating whether the snap actually
    earns its keep now that the axis sweep (its companion boundary mechanism) is
    gone."""

    @staticmethod
    def _reflect(x, lo, hi):
        span = hi - lo
        x_rel = (x - lo) % (2 * span)
        x_rel = np.where(x_rel > span, 2 * span - x_rel, x_rel)
        return x_rel + lo
