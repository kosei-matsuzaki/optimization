"""Where does MC-ESO spend its budget, and what actually makes progress?

Runs MC-ESO (current default) on BBOB-24 at the given dims with the quick
budget (2500·D) and seeds (run i → seed 100·i), and reduces each run's trace
(core/optimizers/mceso.py, recording only) to one row:

* reached 1e-10?  first evaluation at which it did; final f − f*
* router: committed route, evaluation at commit
* restarts: spillovers, basin switches, first "dug out" event
* budget: share of evaluations per channel, and the same counted only up to
  the first hit of 1e-10 (what it took to get there)
* progress credit: every evaluation that improved the best-so-far is credited
  with the decades it gained (log10 old gap − log10 new gap, floor 1e-12), per
  channel, up to the first hit — which channel actually moves the best
* σ: evaluation at which drilling first starts; learned-C condition (≥3D)

    python scripts/single/diag_trace.py --dims 2 5 10 --runs 20 --jobs 16 \
        --out analysis/single/local_trace/runs.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FLOOR = 1e-12
CH = {0: "close", 1: "droplet", 2: "air", 5: "reseed", 6: "init", 7: "ls"}


def _task(args):
    dim, func, runs, method = args
    sys.argv = ["quick_check"]
    import quick_check as q
    b = q._DIM_REGISTRIES[dim][func]
    results, _ = q._run_one(b, method, runs, 2500 * dim, None)
    rows = []
    for i, r in enumerate(results):
        t = r.trace
        g = t["gen"]
        routes = t["routes"]
        ch = np.frombuffer(t["eval_channel"], dtype=np.uint8)
        f = np.maximum(np.asarray(r.history_f, dtype=float) - b.optimum, FLOOR)
        best = np.minimum.accumulate(f)
        hit = np.nonzero(best <= 1e-10)[0]
        first_hit = int(hit[0]) + 1 if len(hit) else None
        upto = first_hit or len(f)
        # progress credit per channel (up to first hit)
        lb = np.log10(best[:upto])
        gain = np.concatenate([[0.0], lb[:-1] - lb[1:]])
        row = {
            "dim": dim, "function": func, "seed": 100 * i,
            "reached": int(first_hit is not None), "first_hit": first_hit or "",
            "final_gap": f"{best[-1]:.3e}",
            "route": routes[g["route"][-1]],
            "commit_evals": next((g["evals"][k] for k, v in enumerate(g["route"]) if v), ""),
            "n_spill": sum(1 for e in t["events"] if e[1] == "spillover"),
            "n_switch": sum(1 for e in t["events"] if e[1] == "basin_switch"),
            "exhausted": next((e[0] for e in t["events"] if e[1] == "exhausted"), ""),
            "drill_start": next((g["evals"][k] for k, v in enumerate(g["drilling"]) if v), ""),
            "cc_cond_max": (f"{np.nanmax(np.asarray(g['cc_cond'], dtype=float)):.2f}"
                            if any(v == v for v in g["cc_cond"]) else ""),
            "n_pop_at_hit": g["n_pop"][min(np.searchsorted(g["evals"], upto), len(g["n_pop"]) - 1)],
        }
        for code, name in CH.items():
            m_all = ch == code
            m_pre = m_all[:upto]
            row[f"share_{name}"] = f"{m_all.mean():.4f}"
            row[f"share_pre_{name}"] = f"{m_pre.mean():.4f}"
            row[f"gain_{name}"] = f"{gain[m_pre].sum():.3f}"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--method", default="MC-ESO")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.argv = ["quick_check"]
    import quick_check as q
    tasks = [(d, fn, a.runs, a.method) for d in a.dims for fn in q._ALL_FUNCTIONS
             if fn in q._DIM_REGISTRIES[d]]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, rs in enumerate(ex.map(_task, tasks)):
            rows.extend(rs)
            print(f"{k + 1}/{len(tasks)} {rs[0]['dim']}D {rs[0]['function']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
