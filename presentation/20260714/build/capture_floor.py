"""Capture real close-contact sampling shapes on an ill-conditioned vs a rugged
2-D BBOB function. Logs the population and the effective close-contact Gaussian
(population covariance eigen-decomposition after the adaptive anisotropy floor)
at a representative generation. Saved to figs/floor_data.npz for fig_floor().
One-off visualization data generator, not an evaluation run.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.benchmarks import _make_bbob
from core.optimizers.mceso import MultiChannelEpidemicOptimizer


class Logged(MultiChannelEpidemicOptimizer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.snaps = []

    def _close_contact_children(self, st, n_local, weights, log_f_max, log_f_spread):
        # read-only snapshot of the sampling shape (no RNG consumed)
        if self.dim >= 2 and n_local > 0 and self.n_pop >= 4:
            cov = np.cov(st.pop_x, rowvar=False)
            if isinstance(cov, np.ndarray) and cov.shape == (self.dim, self.dim):
                eigvals, eigvecs = np.linalg.eigh(cov)
                floor_eff = self._adaptive_cov_floor(st, eigvals)
                ratio = float(eigvals[-1] / max(eigvals[0], 1e-300))
                me = float(eigvals.mean())
                ev = eigvals / me if me > 1e-30 else np.ones(self.dim)
                ev = np.maximum(ev, floor_eff)
                ev = ev / float(ev.mean())          # floored sampling eigenvalues
                self.snaps.append(dict(
                    pop=st.pop_x.copy(), mean=st.pop_x.mean(axis=0),
                    eigvecs=eigvecs.copy(), ev=ev.copy(),
                    sigma=float(st.sigma), ratio=ratio, floor=float(floor_eff),
                    best=st.pop_x[int(np.argmin(st.pop_f))].copy()))
        return super()._close_contact_children(
            st, n_local, weights, log_f_max, log_f_spread)


def local_grid(bench, center, win, n=140):
    xs = np.linspace(center[0] - win, center[0] + win, n)
    ys = np.linspace(center[1] - win, center[1] + win, n)
    GZ = np.array([[bench.func(np.array([x, y])) for x in xs] for y in ys])
    return np.log1p(GZ - GZ.min()), xs[0], xs[-1], ys[0], ys[-1]


def pick(snaps):
    """Snapshot ~10% into the run: the population has settled into the local
    structure (so the covariance reflects the regime — elongated in an
    ill-conditioned valley, round on rugged ground) while σ is still large
    enough for the close-contact Gaussian to be a visible size."""
    return snaps[max(1, int(0.1 * len(snaps)))]


def capture(fid, name, seed=3, max_evals=4000):
    bench = _make_bbob(fid, name, "x", 2)
    opt = Logged(bench, seed=seed)
    opt.optimize(max_evals=max_evals)
    s = pick(opt.snaps)
    center = s["best"]
    win = 4.0 * s["sigma"] * float(np.sqrt(s["ev"].max()))  # ~major-axis extent
    lgz, gx0, gx1, gy0, gy1 = local_grid(bench, center, win)
    return dict(name=name, win=win, lgz=lgz,
                gx0=gx0, gx1=gx1, gy0=gy0, gy1=gy1, **s)


ILL = capture(10, "F10-EllipsoidalRot")     # ill-conditioned rotated valley
RUG = capture(15, "F15-RastriginRot")        # rugged / multimodal

OUT = Path(__file__).resolve().parent / "figs" / "floor_data.npz"


def pack(prefix, s):
    return {f"{prefix}_{k}": s[k] for k in
            ("lgz", "best", "eigvecs", "ev", "sigma", "ratio", "win",
             "gx0", "gx1", "gy0", "gy1")}


np.savez(OUT, **pack("ill", ILL), **pack("rug", RUG))
print(f"ill  ratio {ILL['ratio']:.1e}  ev {np.round(ILL['ev'],3)}  sigma {ILL['sigma']:.3f}")
print(f"rug  ratio {RUG['ratio']:.1e}  ev {np.round(RUG['ev'],3)}  sigma {RUG['sigma']:.3f}")
print("saved", OUT)
