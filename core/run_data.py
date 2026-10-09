"""Small per-run data the results UI draws from (instead of pre-rendered images).

Written next to summary.csv for every run, figures or not:

* ``curves/{Func}.json.gz`` — per method, the median and quartiles of the
  best-so-far f − f* over the runs, on a thinned evaluation grid (log10, 3
  decimals). Enough for the convergence chart; a few KB per function.
* ``mceso_runs.csv`` — one row per (function, method with a trace, seed): the
  route the channel router committed to, spillover / basin-switch counts, the
  share of evaluations per channel, the final population size. The router and
  restart behaviour across all runs at a glance.

Everything heavier (every evaluated point, population snapshots, per-generation
internals) is not stored: runs are seeded, so the UI re-runs the one run it
needs (web/app_lib/replay.py).
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

GAP_FLOOR = 1e-12            # f − f* = 0 is drawn here on the log axis


def _grid(n: int, k: int = 90) -> np.ndarray:
    """Evaluation indices (0-based) dense at the start, even later on."""
    if n <= 2 * k:
        return np.arange(n)
    pts = np.concatenate([np.geomspace(1, n, k), np.linspace(1, n, k)]) - 1
    return np.unique(np.clip(pts.astype(int), 0, n - 1))


def save_curves(dim_dir: Path, bench, results_per_method: dict) -> None:
    n = max(max(len(r.history_best) for r in res) for res in results_per_method.values())
    idx = _grid(n)
    methods = {}
    for name, results in results_per_method.items():
        rows = np.array([list(r.history_best) + [r.history_best[-1]] * (n - len(r.history_best))
                         for r in results], dtype=float)[:, idx]
        g = np.log10(np.maximum(rows - bench.optimum, GAP_FLOOR))
        q1, med, q3 = np.percentile(g, [25, 50, 75], axis=0)
        methods[name] = {"median": np.round(med, 3).tolist(),
                         "q1": np.round(q1, 3).tolist(), "q3": np.round(q3, 3).tolist()}
    out = dim_dir / "curves" / f"{bench.name}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"function": bench.name, "dim": bench.dim, "evals": (idx + 1).tolist(),
               "floor": GAP_FLOOR, "n_runs": len(next(iter(results_per_method.values()))),
               "methods": methods}
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


MCESO_RUN_FIELDS = ["function", "method", "seed", "best_f", "route", "route_commit_evals",
                    "n_spillover", "n_basin_switch", "exhausted_evals",
                    "frac_close", "frac_droplet", "frac_airborne", "frac_reseed",
                    "n_pop_first", "n_pop_last"]


def append_mceso_runs(dim_dir: Path, bench, results_per_method: dict) -> None:
    rows = []
    for name, results in results_per_method.items():
        for i, r in enumerate(results):
            t = getattr(r, "trace", None) or {}
            g = t.get("gen")
            if not g or not g.get("evals"):
                continue
            routes = t.get("routes", ())
            rc = g["route"]
            commit = next((k for k, v in enumerate(rc) if v != 0), None)
            ch = np.frombuffer(t.get("eval_channel", b""), dtype=np.uint8)
            total = max(len(ch), 1)
            ev = [e for e in t.get("events", [])]
            rows.append({
                "function": bench.name, "method": name, "seed": i * 100,
                "best_f": f"{r.best_f:.6e}",
                "route": routes[rc[-1]] if routes else rc[-1],
                "route_commit_evals": g["evals"][commit] if commit is not None else "",
                "n_spillover": sum(1 for e in ev if e[1] == "spillover"),
                "n_basin_switch": sum(1 for e in ev if e[1] == "basin_switch"),
                "exhausted_evals": next((e[0] for e in ev if e[1] == "exhausted"), ""),
                "frac_close": f"{np.sum(ch == 0) / total:.3f}",
                "frac_droplet": f"{np.sum(ch == 1) / total:.3f}",
                "frac_airborne": f"{np.sum(ch == 2) / total:.3f}",
                "frac_reseed": f"{np.sum(ch == 5) / total:.3f}",
                "n_pop_first": g["n_pop"][0], "n_pop_last": g["n_pop"][-1],
            })
    if not rows:
        return
    path = dim_dir / "mceso_runs.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MCESO_RUN_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
