"""Convergence with vs without the adaptive anisotropy floor.

Both variants already succeed on the *median* seed (SR ~90% without the floor),
so a median curve hides the point. The floor's real value is eliminating the
rare failures: seeds where the fixed floor stalls *above* the 1e-10 target. So
for each function we pick a seed that FAILS without the floor but SUCCEEDS with
it, and plot that single seed's two trajectories. Saved to figs/floor_conv.npz.
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

# Exactly the change-ablation configs (quick_check.py) and its seeding
# (core/runner.py: seed = run_index * 100).  abl1_ir = without the adaptive
# floor; abl2_floornich = with it. So this reproduces the very runs behind
# results/20260707_進捗報告データ_変更点ablation/.
ABL1 = {"droplet_variant": "cur2best", "channel_schedule": False,
        "cov_floor_low": 0.01, "exhausted_no_improve_mult": 1e9}   # without floor
ABL2 = {"droplet_variant": "cur2best", "channel_schedule": False}  # adaptive floor
SEEDS = [i * 100 for i in range(20)]   # the 20 ablation runs
FORCE_SEED = {"f19": 1400}             # hand-picked panel seeds (skip the search)


def curve(bench, kw, seed):
    opt = MultiChannelEpidemicOptimizer(bench, seed=seed, **kw)
    res = opt.optimize(max_evals=MAX_EVALS)
    f = np.maximum(np.minimum.accumulate(np.asarray(res.history_f)), 1e-12)
    return f[np.clip(GRID - 1, 0, len(f) - 1)]


data = {"grid": GRID}
used = set()   # keep the two panels on DISTINCT seeds (avoid a coincidental tie)
for tag, fid, name in [("f10", 10, "F10-EllipsoidalRot"),
                       ("f19", 19, "F19-GriewankRosenbrock")]:
    b = _make_bbob(fid, name, "x", 2)
    chosen = None
    if tag in FORCE_SEED:
        sd = FORCE_SEED[tag]
        chosen = (sd, curve(b, ABL1, sd), curve(b, ABL2, sd))
    for sd in SEEDS if chosen is None else []:
        if sd in used:
            continue
        off = curve(b, ABL1, sd)          # abl1_ir  (without adaptive floor)
        if off[-1] > TARGET:              # failed without the floor
            on = curve(b, ABL2, sd)       # abl2_floornich  (adaptive floor)
            if on[-1] <= TARGET:          # rescued by the floor
                chosen = (sd, off, on)
                break
    if chosen is None:                    # fallback: worst-off seed
        sd = max((s for s in SEEDS if s not in used),
                 key=lambda s: curve(b, ABL1, s)[-1])
        chosen = (sd, curve(b, ABL1, sd), curve(b, ABL2, sd))
    sd, off, on = chosen
    used.add(sd)
    data[f"{tag}_off"], data[f"{tag}_on"], data[f"{tag}_seed"] = off, on, sd
    # landscape shape of the function (log1p for contrast) + its optimum
    lo, hi = b.bounds
    gx = np.linspace(lo, hi, 110)
    GZ = np.array([[b.func(np.array([x, y])) for x in gx] for y in gx])
    data[f"{tag}_land"] = np.log1p(GZ - GZ.min())
    data[f"{tag}_ext"] = np.array([lo, hi, lo, hi])
    oi = np.unravel_index(np.argmin(GZ), GZ.shape)
    data[f"{tag}_opt"] = np.array([gx[oi[1]], gx[oi[0]]])
    print(f"{name}: seed {sd}  off {off[-1]:.1e} (fail)  on {on[-1]:.1e} (ok)")

OUT = Path(__file__).resolve().parent / "figs" / "floor_conv.npz"
np.savez(OUT, **data)
print("saved", OUT)
