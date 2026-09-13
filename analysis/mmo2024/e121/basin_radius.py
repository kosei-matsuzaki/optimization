"""e121 -- basin-radius curves for cluster members vs isolated optima.

One problem per invocation (``--pid M01``), so 4-wide xargs works and a run cut
short leaves finished problems behind (entry 85's rule).

The descent is Restart-Lander's, copied rather than imported because that class
chains descents to fill a run budget and draws its own uniform starts; here the
start is placed by hand on a sphere around a known optimum and exactly one
descent is run.  The CMA options below are the same dict as
``core/optimizers/restart_lander.py:82`` minus the chaining.

See prereg.md for the design and the decision rule.
"""
from __future__ import annotations
import argparse
import csv
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from core.benchmarks import niching_by_name          # noqa: E402

RADII = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
DESCENT_BUDGET = 3125                                 # 12500 / 4
SIGMA_RATIO = 0.1                                     # the null's default
NEAR = 0.1                                            # entry 119's convention
SEED = 20260913


def descent(bench, x0, sigma0, budget=DESCENT_BUDGET, seed=1):
    """Restart-Lander's descent, run once from a given start."""
    import cma
    lo, hi = bench.bounds
    opts = {"bounds": [lo, hi], "maxfevals": budget, "seed": seed,
            "verbose": -9, "tolfun": 0, "tolfunhist": 0, "tolx": 0,
            "CMA_on": 0}                              # step size only, no rotation
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    best_f, best_x, used = float("inf"), np.asarray(x0, dtype=float), 0
    while not es.stop():
        xs = es.ask()
        fs = [float(bench.func(np.asarray(x))) for x in xs]
        used += len(xs)
        es.tell(xs, fs)
        i = int(np.argmin(fs))
        if fs[i] < best_f:
            best_f, best_x = fs[i], np.asarray(xs[i], dtype=float)
    return best_f, best_x, used, "|".join(sorted(es.stop())) or "budget"


def targets_for(pid, rng):
    """(idx, clustered) pairs: this problem's members + a matched isolated control."""
    rows = [r for r in csv.DictReader(open(ROOT / "analysis/mmo2024/e119/enrichment_d10.csv"))
            if r["pid"] == pid]
    mem = [int(r["idx"]) for r in rows if r["clustered"] == "1"]
    iso = [int(r["idx"]) for r in rows if r["clustered"] == "0"]
    if not mem:
        return []
    ctrl = list(rng.choice(iso, size=min(len(mem), len(iso)), replace=False))
    return [(i, 1) for i in mem] + [(int(i), 0) for i in ctrl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--out", default=str(HERE / "by_problem"))
    # the pre-registered sweep stops at r = 2.0; --radii/--conds run the
    # far-field extension (see scored_far.txt) without touching those files
    ap.add_argument("--radii", default=None,
                    help="comma-separated radii (default: the pre-registered set)")
    ap.add_argument("--conds", default="deployed,matched")
    a = ap.parse_args()
    radii = RADII if a.radii is None else [float(v) for v in a.radii.split(",")]
    conds = a.conds.split(",")

    name = f"{a.pid}-D10-PIN01"
    bench = niching_by_name(name)
    opt = np.asarray(bench.optima_pos, dtype=float)
    lo, hi = bench.bounds
    dim = bench.dim
    # one stream per problem so a rerun of one problem reproduces byte for byte
    rng = np.random.default_rng(SEED + int(a.pid[1:]))
    tgts = targets_for(a.pid, rng)
    if not tgts:
        print(f"{a.pid}: no cluster members, skipped")
        return

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for idx, clustered in tgts:
        z = opt[idx]
        for r in radii:
            u = rng.normal(size=dim)
            u /= np.linalg.norm(u)
            x0 = np.clip(z + r * u, lo, hi)
            r_eff = float(np.linalg.norm(x0 - z))     # clipping can shorten it
            all_conds = {"deployed": SIGMA_RATIO * (hi - lo),
                         "matched": max(r, 0.1) / 2.0}
            for cond, s0 in [(c, all_conds[c]) for c in conds]:
                k += 1
                bf, bx, used, stop = descent(bench, x0, s0, seed=SEED + k)
                d = np.linalg.norm(opt - bx, axis=1)
                land = int(np.argmin(d))
                rows.append({
                    "pid": a.pid, "idx": idx, "clustered": clustered,
                    "r": r, "r_eff": round(r_eff, 6), "cond": cond,
                    "sigma0": s0, "land_opt": land,
                    "dist": round(float(d[land]), 6),
                    "dist_target": round(float(np.linalg.norm(bx - z)), 6),
                    "best_f": bf, "evals": used, "stop": stop,
                    "hit": int(land == idx and float(np.linalg.norm(bx - z)) < NEAR),
                    "hit_loose": int(land == idx),
                })
    p = out / f"{a.pid}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{a.pid}: {len(rows)} descents -> {p}")


if __name__ == "__main__":
    main()
