"""Capture the per-landscape channel router's committed decision for each 2-D
BBOB function: the two scale-invariant covariance signals at the commit
checkpoint (cond = log10 λmax/λmin, algA = axis-alignment, mgap = coordinate
gap) and the route it locked (droplet / close / keepair). Saved to
figs/router_data.npz for fig_router() to draw the decision-region map.
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
        self.commit = None

    def _channel_ratios(self, st):
        out = super()._channel_ratios(st)
        if self.commit is None and st.channel_route is not None:
            self.commit = dict(route=st.channel_route,
                               cond=float(st.cc_logratio_ema or 0.0),
                               algA=float(st.cc_align_ema or 0.0),
                               mgap=float(st.cc_mgap_ema or 0.0))
        return out


rows = []
for fid in range(1, 25):
    b = _make_bbob(fid, f"F{fid:02d}", "x", 2)
    opt = Logged(b, seed=0)
    opt.optimize(max_evals=3000)
    c = opt.commit or dict(route="keepair", cond=0.0, algA=0.0, mgap=0.0)
    rows.append((fid, c["route"], c["cond"], c["algA"], c["mgap"]))
    print(f"F{fid:02d}  {c['route']:8s}  cond {c['cond']:.2f}  "
          f"algA {c['algA']:.3f}  mgap {c['mgap']:.3f}")

fid = np.array([r[0] for r in rows])
route = np.array([r[1] for r in rows])
cond = np.array([r[2] for r in rows])
algA = np.array([r[3] for r in rows])
mgap = np.array([r[4] for r in rows])
OUT = Path(__file__).resolve().parent / "figs" / "router_data.npz"
np.savez(OUT, fid=fid, route=route, cond=cond, algA=algA, mgap=mgap,
         cond_thresh=3.0, align_thresh=0.965, mgap_thresh=0.36)
for r in ("droplet", "close", "keepair"):
    print(f"{r}: {[f'F{f:02d}' for f, rr in zip(fid, route) if rr == r]}")
print("saved", OUT)
