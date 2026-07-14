"""Capture real re-seed data from MC-ESO's restart, old (blind) vs new (informed).

Runs both variants on a 2-D multimodal landscape with a fixed seed, logging every
spillover re-seed point (tagged reservoir / repelled / blind), plus the reservoir
hosts and abandoned-basin centroids the informed variant remembers. Saved to
figs/ir_data.npz for fig_restart() to plot. This is a one-off visualization data
generator, not an evaluation run.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.benchmarks import _make_bbob
from core.optimizers.mceso import MultiChannelEpidemicOptimizer

# BBOB F04 (Büche-Rastrigin, 2-D): a real multimodal benchmark where informed
# restart actually helps, and it triggers a good mix of reservoir re-ignitions
# and basin-repelled re-seeds. func(x) = f(x) - f_opt, so the global min is 0.
BENCH = _make_bbob(4, "F04-BucheRastrigin", "multimodal", 2)
LO, HI = BENCH.bounds


class _StashState:
    def _init_state(self, m):
        st = super()._init_state(m)
        self._st = st
        return st


class NewLogged(_StashState, MultiChannelEpidemicOptimizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reseeds = []

    def _diversified_reseed(self, st, x_best_snap):
        rng, lo, hi, dim = st.rng, st.lo, st.hi, self.dim
        if self._basin_exhausted(st):
            repel_r = 0.02 * st.span
            cand = rng.uniform(lo, hi, dim)
            if st.ir_basin_centroids:
                centroids = np.array(st.ir_basin_centroids)
                for _ in range(self.ir_repel_max_tries):
                    if np.all(np.linalg.norm(centroids - cand, axis=1) > repel_r):
                        break
                    cand = rng.uniform(lo, hi, dim)
            self.reseeds.append((cand.copy(), "repelled"))
            return cand
        if st.ir_archive_x and rng.random() < self.ir_archive_frac:
            k = rng.integers(0, len(st.ir_archive_x))
            sigma = self.ir_reignite_sigma_ratio * st.span
            cand = self._reflect(
                st.ir_archive_x[k] + sigma * rng.standard_normal(dim), lo, hi)
            self.reseeds.append((cand.copy(), "reservoir"))
            return cand
        repel_r = self.ir_repel_radius_ratio * st.span
        cand = rng.uniform(lo, hi, dim)
        if st.ir_basin_centroids:
            centroids = np.array(st.ir_basin_centroids)
            for _ in range(self.ir_repel_max_tries):
                if np.all(np.linalg.norm(centroids - cand, axis=1) > repel_r):
                    break
                cand = rng.uniform(lo, hi, dim)
        self.reseeds.append((cand.copy(), "repelled"))
        return cand


class OldLogged(_StashState, MultiChannelEpidemicOptimizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reseeds = []

    def _on_spillover_start(self, st, basin_switch):
        return None  # blind: no reservoir harvest, no basin memory

    def _diversified_reseed(self, st, x_best_snap):
        cand = st.rng.uniform(st.lo, st.hi, self.dim)
        self.reseeds.append((cand.copy(), "blind"))
        return cand


def run(cls, seed, max_evals=4000):
    opt = cls(BENCH, seed=seed)
    opt.optimize(max_evals=max_evals)
    st = opt._st
    reseeds = np.array([p for p, _ in opt.reseeds]) if opt.reseeds else np.empty((0, 2))
    tags = np.array([t for _, t in opt.reseeds]) if opt.reseeds else np.empty(0, dtype=object)
    reservoir = np.array(st.ir_archive_x) if st.ir_archive_x else np.empty((0, 2))
    basins = np.array(st.ir_basin_centroids) if st.ir_basin_centroids else np.empty((0, 2))
    return reseeds, tags, reservoir, basins


SEED = 7
new_rs, new_tags, reservoir, basins = run(NewLogged, SEED)
old_rs, old_tags, _, _ = run(OldLogged, SEED)

# landscape grid + approximate global optimum (argmin of f - f_opt)
g = np.linspace(LO, HI, 240)
GX, GY = np.meshgrid(g, g)
GZ = np.array([[BENCH.func(np.array([xx, yy])) for xx in g] for yy in g])
opt_ij = np.unravel_index(np.argmin(GZ), GZ.shape)
opt = np.array([GX[opt_ij], GY[opt_ij]])

OUT = Path(__file__).resolve().parent / "figs" / "ir_data.npz"
np.savez(OUT, gx=GX, gy=GY, gz=np.log1p(GZ - GZ.min()),  # log scale for contrast
         old_rs=old_rs, new_rs=new_rs, new_tags=new_tags.astype(str),
         reservoir=reservoir, basins=basins, opt=opt,
         repel_r=0.1 * (HI - LO), lo=LO, hi=HI)
print("optimum ≈", np.round(opt, 2).tolist())
print(f"old reseeds: {len(old_rs)}  new reseeds: {len(new_rs)} "
      f"(reservoir {int((new_tags=='reservoir').sum())}, "
      f"repelled {int((new_tags=='repelled').sum())})")
print(f"reservoir hosts: {len(reservoir)}  abandoned basins: {len(basins)}")
print("saved", OUT)
