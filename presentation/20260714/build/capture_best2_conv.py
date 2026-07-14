"""best2 version of capture_floor_conv: BBOB functions where a seed FAILS with
the single-difference droplet (abl3_router, droplet_variant='cur2best') but
SUCCEEDS with the route-gated 2nd-difference droplet (MC-ESO default,
'best2_droplet'). Reproduces the change-ablation runs (seed = run_index*100).
Saves each function's two trajectories + 2-D map + 3-D landscape to
figs/best2_conv.npz for fig_best2_conv_panels().
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.benchmarks import _make_bbob
from core.optimizers.mceso import MultiChannelEpidemicOptimizer

MAX_EVALS = 5000
TARGET = 1e-10
GRID = np.linspace(1, MAX_EVALS, 200).astype(int)
ABL3 = {"droplet_variant": "cur2best"}   # single difference (router on)
MCE = {}                                 # default = best2_droplet (route-gated)
SEEDS = [i * 100 for i in range(20)]
CANDIDATES = [(13, "F13-SharpRidge"), (14, "F14-DiffPowers")]


def curve(bench, kw, seed):
    opt = MultiChannelEpidemicOptimizer(bench, seed=seed, **kw)
    res = opt.optimize(max_evals=MAX_EVALS)
    f = np.maximum(np.minimum.accumulate(np.asarray(res.history_f)), 1e-12)
    return f[np.clip(GRID - 1, 0, len(f) - 1)]


data = {"grid": GRID}
used, chosen = set(), []
for fid, name in CANDIDATES:
    if len(chosen) == 2:
        break
    b = _make_bbob(fid, name, "x", 2)
    for sd in SEEDS:
        if sd in used:
            continue
        off = curve(b, ABL3, sd)
        if off[-1] > TARGET:
            on = curve(b, MCE, sd)
            if on[-1] <= TARGET:
                used.add(sd)
                chosen.append((fid, name, sd, off, on, b))
                break

for tag, (fid, name, sd, off, on, b) in zip(("a", "b"), chosen):
    data[f"{tag}_off"], data[f"{tag}_on"], data[f"{tag}_seed"] = off, on, sd
    data[f"{tag}_name"] = name
    lo, hi = b.bounds
    gx = np.linspace(lo, hi, 110)
    GZ = np.array([[b.func(np.array([x, y])) for x in gx] for y in gx])
    data[f"{tag}_land"] = np.log1p(GZ - GZ.min())
    data[f"{tag}_ext"] = np.array([lo, hi, lo, hi])
    oi = np.unravel_index(np.argmin(GZ), GZ.shape)
    data[f"{tag}_opt"] = np.array([gx[oi[1]], gx[oi[0]]])
    print(f"{name}: seed {sd}  off {off[-1]:.1e} (fail)  on {on[-1]:.1e} (ok)")

OUT = Path(__file__).resolve().parent / "figs" / "best2_conv.npz"
np.savez(OUT, **data)
print("saved", OUT)
