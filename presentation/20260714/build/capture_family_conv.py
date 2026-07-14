"""Convergence-with-spread on four functions, one per method-family weakness,
reproducing the 20260706 existing-methods comparison (all 8 methods, dim 2,
n=20, seed=i*100, max_evals=5000). For each function we record every method's
running-best gap-to-optimum on a shared eval grid (all 20 seeds) plus the 2-D
map / 3-D landscape, so fig_family_conv() can draw the p21-style rows:
[2-D map | 3-D landscape | convergence mean ± 1σ band].

Functions (family that struggles):
  F03-RastriginSep    multimodal   — CMA-ES family
  F11-Discus          ill-cond     — PSO / SaVOA
  F08-Rosenbrock      bent-valley  — DE family (DE / L-SHADE)
  F15-RastriginRot    (all fall short)
Saves to figs/family_conv.npz.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.benchmarks import _make_bbob
from core.optimizers import (
    MultiChannelEpidemicOptimizer, IPOPCMAESOptimizer,
    PSOOptimizer, DEOptimizer, SaVOAOptimizer, MultistartNelderMeadOptimizer,
)

MAX_EVALS = 5000
FLOOR = 1e-16          # low enough that converged runs show their true depth,
                       # not a flat line pinned at the clamp
GRID = np.linspace(1, MAX_EVALS, 200).astype(int)
SEEDS = [i * 100 for i in range(20)]

# Unified comparison set used across the Comparison chapter (p_family_conv +
# the category/Wilcoxon table): MC-ESO vs IPOP-CMA-ES / PSO / DE / SaVOA /
# NM-Restart. The remaining baselines (CMA-ES, BIPOP, L-SHADE, NCDE) are moved
# to the appendix.
METHODS = [
    ("MC-ESO", MultiChannelEpidemicOptimizer),
    ("IPOP-CMA-ES", IPOPCMAESOptimizer),
    ("PSO", PSOOptimizer),
    ("DE", DEOptimizer),
    ("SaVOA", SaVOAOptimizer),
    ("NM-Restart", MultistartNelderMeadOptimizer),
]
FUNCS = [("a", 3, "F03-RastriginSep"), ("b", 11, "F11-Discus"),
         ("c", 8, "F08-Rosenbrock"), ("d", 15, "F15-RastriginRot")]


def curve(bench, cls, seed):
    opt = cls(bench, seed=seed)
    res = opt.optimize(max_evals=MAX_EVALS)
    f = np.maximum(np.minimum.accumulate(np.asarray(res.history_f)), FLOOR)
    return f[np.clip(GRID - 1, 0, len(f) - 1)]


data = {"grid": GRID, "methods": np.array([m for m, _ in METHODS])}
for tag, fid, name in FUNCS:
    bench = _make_bbob(fid, name, "x", 2)
    data[f"{tag}_name"] = name
    # landscape for the 2-D map + 3-D surface
    lo, hi = bench.bounds
    gx = np.linspace(lo, hi, 110)
    GZ = np.array([[bench.func(np.array([x, y])) for x in gx] for y in gx])
    data[f"{tag}_land"] = np.log1p(GZ - GZ.min())
    data[f"{tag}_ext"] = np.array([lo, hi, lo, hi])
    oi = np.unravel_index(np.argmin(GZ), GZ.shape)
    data[f"{tag}_opt"] = np.array([gx[oi[1]], gx[oi[0]]])
    for mname, cls in METHODS:
        traj = np.array([curve(bench, cls, sd) for sd in SEEDS])  # (nseed, ngrid)
        data[f"{tag}_{mname}"] = traj
        print(f"{name:22s} {mname:12s} median final {np.median(traj[:, -1]):.2e}",
              flush=True)

OUT = Path(__file__).resolve().parent / "figs" / "family_conv.npz"
np.savez(OUT, **data)
print("saved", OUT)
