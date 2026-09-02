#!/usr/bin/env python3
"""Does the restart *draw* decide where the hunt *lands*?

``_diversified_reseed`` returns one candidate per re-seeded population slot, and
a basin switch re-seeds every slot and resets sigma to sigma_init = 0.2 * span.
On Vincent2D that is 1.95 against optimum spacings of 0.291 to 3.595, so a
population scattered with that sigma can roll out of the basin it was dropped
in. If it does, repelling the *draw* away from basins already drilled cannot fix
coverage however well the radius is chosen — the funnel is downstream of it.

This measures the link directly. For each hunt it records every diversified
draw, maps both the draws and the hunt's endpoint onto the nearest true Vincent
optimum, and reports:

  match      the endpoint optimum was also the nearest optimum of at least one
             of that hunt's draws (the draw reached the landing)
  d(draw)    distance from the endpoint to the nearest draw of its own hunt
  width      the nearest-neighbour spacing of the optimum landed in vs. drawn
             near — the wide-basin bias, measured on the draw and on the
             landing separately

Estimator direction: `match` is generous — with n_pop draws per hunt, some draw
lands near the endpoint's optimum by chance, so a low match rate is strong
evidence and a high one is weak.

Usage:  python3 scripts/reseed_to_landing.py [--variant base|adaptive] [--seeds 3]
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer           # noqa: E402
from core.optimizers.mceso_adaptive_repel import AdaptiveRepelMCESO  # noqa: E402
from core.optimizers.mceso_commit_reseed import CommitReseedMCESO      # noqa: E402
from scripts.hunt_coverage import vincent_optima                    # noqa: E402


def _tracer(base_cls):
    class _Traced(base_cls):
        """Records the diversified draws of the hunt in progress, and files them
        against that hunt's endpoint when the next spillover fires."""

        def optimize(self, max_evals: int = 5000):
            self._draws: list[np.ndarray] = []
            self.records: list[dict] = []
            return super().optimize(max_evals)

        def _diversified_reseed(self, st, x_best_snap):
            cand = super()._diversified_reseed(st, x_best_snap)
            self._draws.append(np.asarray(cand, dtype=float).copy())
            return cand

        def _on_spillover_start(self, st, basin_switch: bool) -> None:
            best_i = int(np.argmin(st.pop_f))
            self.records.append({
                "end": np.asarray(st.pop_x[best_i], dtype=float).copy(),
                "f": float(st.pop_f[best_i]),
                "draws": list(self._draws),
            })
            self._draws = []
            return super()._on_spillover_start(st, basin_switch)
    return _Traced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base",
                    choices=["base", "adaptive", "commit_tight"])
    ap.add_argument("--func", default="N07-Vincent2D")
    ap.add_argument("--evals", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=None,
                    help="override sigma_init/span (default 0.2). Mechanism probe: "
                         "does shrinking the restart sigma below the basin spacing "
                         "keep the population in the basin it was dropped into?")
    args = ap.parse_args()

    b = NICHING_BENCHMARKS_BY_NAME[args.func]
    dim = b.dim
    opt_pts = vincent_optima(dim)
    D = np.linalg.norm(opt_pts[:, None, :] - opt_pts[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    width = D.min(axis=1)          # nearest-neighbour spacing per optimum
    assert len(opt_pts) == b.n_global_optima, (len(opt_pts), b.n_global_optima)

    _CLS = {"base": (MultiChannelEpidemicOptimizer, {}),
            "adaptive": (AdaptiveRepelMCESO, {"repel_mode": "adaptive"}),
            # The whole population committed to one draw, spread at 0.1x the
            # locally observed basin spacing (scripts/diagnose_niching.py's
            # `commit_tight`). Note the tracer records *every* diversified draw,
            # so for this variant the "draws" of a hunt are the anchor plus the
            # cloud placed around it -- `distinct optima drawn near` therefore
            # measures the committed placement, not an independent draw per slot.
            "commit_tight": (CommitReseedMCESO,
                             {"commit_sigma_mode": "run",
                              "commit_sigma_ratio": 0.1})}
    base_cls, kw = _CLS[args.variant]
    cls = _tracer(base_cls)
    kw = dict(kw)
    if args.sigma is not None:
        kw["sigma"] = args.sigma

    n_hunt = n_match = 0
    d_draw: list[float] = []
    land_w: list[float] = []
    draw_w: list[float] = []
    land_ct: Counter = Counter()
    draw_ct: Counter = Counter()
    span = 0.0
    for seed in range(args.seeds):
        o = cls(b, seed=seed * 100, **kw)
        o.optimize(args.evals)
        span = float(b.bounds[1] - b.bounds[0])
        for rec in o.records:
            if not rec["draws"]:
                continue
            n_hunt += 1
            end_i = int(np.argmin(np.linalg.norm(opt_pts - rec["end"], axis=1)))
            dr = np.asarray(rec["draws"])
            draw_i = np.argmin(np.linalg.norm(
                opt_pts[None, :, :] - dr[:, None, :], axis=-1), axis=1)
            n_match += int(end_i in set(draw_i.tolist()))
            d_draw.append(float(np.min(np.linalg.norm(dr - rec["end"], axis=1))))
            land_w.append(float(width[end_i]))
            land_ct[end_i] += 1
            for j in draw_i:
                draw_w.append(float(width[int(j)]))
                draw_ct[int(j)] += 1

    sig = (args.sigma if args.sigma is not None else 0.2) * span
    print(f"{args.func}  variant={args.variant}  seeds={args.seeds}  "
          f"evals={args.evals}  K={b.n_global_optima}")
    print(f"  hunts with draws            {n_hunt}")
    print(f"  endpoint optimum was drawn  {n_match}  ({n_match / max(n_hunt,1):.2f})")
    print(f"  median d(endpoint, nearest draw of its hunt)  {np.median(d_draw):.3f}"
          f"   sigma_init = {sig:.3f}")
    print(f"  basin width (NN spacing) landed in   median {np.median(land_w):.3f}")
    print(f"  basin width (NN spacing) drawn near  median {np.median(draw_w):.3f}")
    print(f"  distinct optima drawn near  {len(draw_ct)} / {b.n_global_optima}")
    print(f"  distinct optima landed in   {len(land_ct)} / {b.n_global_optima}")
    wide = width >= np.median(width)
    dw = sum(c for i, c in draw_ct.items() if wide[i]) / max(sum(draw_ct.values()), 1)
    lw = sum(c for i, c in land_ct.items() if wide[i]) / max(sum(land_ct.values()), 1)
    print(f"  share on the wide half of the optima:  draws {dw:.2f}   landings {lw:.2f}")


if __name__ == "__main__":
    main()
