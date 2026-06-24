"""MC-ESO sequential niching — now built into the base optimizer.

The multi-solution (sequential niching) behaviour that this module used to add as
a separate ``MCESOEndemic`` subclass is, as of 2026-06, **integrated into the base
``MultiChannelEpidemicOptimizer``** (see ``mceso.py``): once a basin is drilled to
the algorithm's resolution limit (``_basin_exhausted`` — σ bottomed out + stagnant,
a scale-/shift-invariant signal), MC-ESO restarts repelled away from it to discover
further optima. SR@1e-10 is preserved because a basin is only left once base could
drill no deeper there.

``MCESOEndemic`` is kept as a thin alias for backward compatibility (older result
configs / imports). It is now byte-identical to base MC-ESO.
"""
from __future__ import annotations

from .mceso import MultiChannelEpidemicOptimizer


class MCESOEndemic(MultiChannelEpidemicOptimizer):
    """Deprecated alias — sequential niching is now part of base MC-ESO."""
