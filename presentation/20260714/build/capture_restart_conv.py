"""Restart version of capture_floor_conv: BBOB functions where a seed FAILS with
the blind uniform restart (abl0_base2018) but SUCCEEDS with the informed restart
(abl1_ir). Reproduces the change-ablation runs (seed = run_index*100). For each
chosen function, saves the two convergence trajectories + its 2-D map and 3-D
landscape. Saved to figs/restart_conv.npz for fig_restart_conv_panels().
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
# blind uniform restart (no reservoir re-ignition, no basin repulsion)
ABL0 = {"droplet_variant": "cur2best", "channel_schedule": False,
        "cov_floor_low": 0.01, "exhausted_no_improve_mult": 1e9,
        "ir_archive_frac": 0.0, "ir_repel_max_tries": 0}
# informed restart (reservoir + herd-immunity repulsion)
ABL1 = {"droplet_variant": "cur2best", "channel_schedule": False,
        "cov_floor_low": 0.01, "exhausted_no_improve_mult": 1e9}
SEEDS = [i * 100 for i in range(20)]
# functions the informed restart lifts (abl0 -> abl1, SR@1e-10 up)
CANDIDATES = [(4, "F04-BucheRastrigin"), (18, "F18-SchafferF7ill"),
              (11, "F11-Discus"), (14, "F14-DiffPowers"),
              (24, "F24-LunacekRastrigin")]


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
        off = curve(b, ABL0, sd)
        if off[-1] > TARGET:
            on = curve(b, ABL1, sd)
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

OUT = Path(__file__).resolve().parent / "figs" / "restart_conv.npz"
np.savez(OUT, **data)
print("saved", OUT)
