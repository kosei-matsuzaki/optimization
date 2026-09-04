"""Diagnostic MC-ESO variant: the hunt-release level expressed **relative to the
scoring threshold** instead of as an absolute ``hunt_level_tol``.

Why. Entry 35 (research_loop) established that the release rule fires when

    basin_best <= hunt_level_tol * f_init_scale                 (mceso.py:928)

so a discarded basin scores at accuracy ``eps`` exactly when the *effective
release level* ``L = hunt_level_tol * f_init_scale`` is below ``eps``. The catch
is that ``f_init_scale`` (the initial-population best |f|) is **not** dimension
invariant: on N06-Shubert2D it is ~145, on N08-Shubert3D it is ~2493 (17x). So
one fixed ``hunt_level_tol`` yields a *different* effective release level per
problem, and the fixed sweep found two different "best" values (2D 1e-7,
3D 1e-8) that were really one rule read at two cross-sections.

This variant removes the dimension dependence: it fixes the effective release
level directly,

    L = c * eps_target                     (a chosen absolute release level)
    hunt_level_tol = L / f_init_scale       (set once f_init_scale is known)

so ``basin_best <= L`` regardless of the problem's f range. ``eps_target`` is the
deepest scoring accuracy the theme aims at (1e-5); ``c`` is the dimensionless
margin below it. ``c = 0.1`` -> ``L = 1e-6`` (≈ the 2D win the fixed 1e-8 bought:
1.4e-6), and crucially the *same* ``L`` in 3D, where fixed 1e-8 only reached
2.5e-5 and so could not score at eps=1e-5.

Every other code path is inherited unchanged. ``hunt_level_tol`` is recomputed at
the start of each run from that run's own ``f_init_scale``, so a fresh instance
per seed is not required for correctness, but the diagnostic driver makes one
anyway. With ``rel_level`` left at 0 the class is bit-identical to the shipped
optimiser (it never touches ``hunt_level_tol``), which is the identity check.
"""

from __future__ import annotations

from .mceso import MultiChannelEpidemicOptimizer, _MCESOState


class RelLevelMCESO(MultiChannelEpidemicOptimizer):
    """MC-ESO whose ``hunt_level_tol`` tracks ``f_init_scale`` so that the
    effective hunt-release level is a fixed absolute value ``rel_level``.

    ``rel_level = 0.0`` disables the override (identity check).
    """

    def __init__(self, *args, rel_level: float = 0.0, fis_floor: float = 0.0,
                 tol_cap: float = float("inf"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if rel_level < 0.0:
            raise ValueError(f"rel_level must be >= 0, got {rel_level!r}")
        if fis_floor < 0.0:
            raise ValueError(f"fis_floor must be >= 0, got {fis_floor!r}")
        if tol_cap <= 0.0:
            raise ValueError(f"tol_cap must be > 0, got {tol_cap!r}")
        # The target *effective* release level L = c * eps_target, on the same
        # (f - f_opt) scale as the eps scoring thresholds.
        self.rel_level = float(rel_level)
        # Two shapes of the clamp entry 37 asked for, both no-ops at their
        # defaults so the unclamped arm stays reachable from the same class.
        #
        # The endpoint to stop is `f_init_scale` pinned at its own floor 1e-300
        # (mceso.py:795), which happens when the initial population lands
        # exactly on f = 0 -- F07-StepEllipsoidal does it. There the quotient is
        # 1e294 and the level clause stops being a *level* at all.
        #
        # Entry 43 showed the clamp entry 37 wrote down, `min(quotient, the
        # shipped default 1e-6)`, is far stronger than that: it binds on every
        # problem with `f_init_scale < 1` and returns the variant to base there,
        # which is exactly where the variant's only significant gain lives.
        # So the clamp has to be placed where only the endpoint reaches it:
        #
        #   fis_floor  a lower bound on the denominator. The effective release
        #              level becomes `L * f_init_scale / fis_floor`, i.e. it
        #              *shrinks* with f_init_scale below the floor, which is the
        #              conservative direction (fewer releases, like base).
        #   tol_cap    an upper bound on the quotient itself. The effective
        #              level becomes `tol_cap * f_init_scale`, i.e. base's own
        #              shape with a looser constant.
        #
        # Neither binds for `f_init_scale` in the range the benchmarks actually
        # occupy (0.1 .. 1e5) as long as fis_floor << L and tol_cap >> L.
        self.fis_floor = float(fis_floor)
        self.tol_cap = float(tol_cap)

    def _init_state(self, max_evals: int) -> _MCESOState:
        st = super()._init_state(max_evals)
        if self.rel_level > 0.0:
            # L = hunt_level_tol * f_init_scale  =>  hunt_level_tol = L / f_init_scale.
            # f_init_scale is >= 1e-300 by construction (mceso.py:795).
            denom = max(float(st.f_init_scale), self.fis_floor)
            self.hunt_level_tol = min(self.rel_level / denom, self.tol_cap)
        return st
