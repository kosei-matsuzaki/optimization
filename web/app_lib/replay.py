"""Re-run one seeded run on demand, and compute landscape grids.

A results directory keeps only small data. When the UI needs everything a run
did — every evaluated point, the population per generation, MC-ESO's internals —
it asks for the run to be repeated here. Runs are seeded by index
(core.runner.run_experiment: seed = run index × 100), so the replay is the run,
provided the code has not changed since. ``verified`` says whether the replayed
final best matches the one recorded in stats/{Func}.csv for that seed.

Methods that draw from an unseeded global RNG (the mealpy wrappers) cannot be
replayed faithfully; they come back with ``verified = False``.
"""
from __future__ import annotations

import base64
import dataclasses
import functools
import json
import zlib
from pathlib import Path

import numpy as np

from . import results

MAX_POP_FRAMES = 240          # population snapshots sent to the browser
GAP_FLOOR = 1e-12


@functools.lru_cache(maxsize=1)
def _qc():
    import sys
    argv, sys.argv = sys.argv, ["quick_check"]
    try:
        import quick_check
    finally:
        sys.argv = argv
    return quick_check


def find_benchmark(dim: int, name: str):
    qc = _qc()
    regs = [qc._DIM_REGISTRIES.get(dim, {}), qc.NICHING_BENCHMARKS_BY_NAME,
            qc.BENCHMARKS_CEC2022_10D_BY_NAME]
    for reg in regs:
        b = reg.get(name)
        if b is not None and b.dim == dim:
            return b
    return None


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _recorded_best(run_dir: Path, dim: str, func: str, method: str, seed: int):
    for row in results.read_stats(run_dir.name, dim, func).get("rows", []):
        if row.get("method") == method and str(row.get("seed")) == str(seed):
            try:
                return float(row["best_f"])
            except (KeyError, ValueError):
                return None
    return None


@functools.lru_cache(maxsize=24)
def replay(run_id: str, dim_name: str, func: str, method: str, seed: int) -> str:
    """JSON (as a string, so the LRU cache holds the encoded payload)."""
    run_dir = results.run_path(run_id)
    if run_dir is None:
        raise LookupError("run not found")
    meta = results.read_result_meta(run_dir)
    dim = int(dim_name.replace("dim", ""))
    bench = find_benchmark(dim, func)
    if bench is None:
        raise LookupError(f"benchmark {func} (dim {dim}) not found")
    qc = _qc()
    if method not in qc._OPTIMIZERS:
        raise LookupError(f"method {method} is not registered any more")
    max_evals = int(meta.get("max_evals") or 2500 * dim)

    cls, kwargs = qc._OPTIMIZERS[method]
    kw = {**kwargs}
    if cls in qc._SIGMA_USERS:
        kw["sigma0"] = 0.2 * (bench.bounds[1] - bench.bounds[0])
    bench_run = bench
    noise = meta.get("noise")
    if noise:
        from core.benchmarks import make_noisy_func
        i = seed // 100
        rng = np.random.default_rng(zlib.crc32(f"{bench.name}|{noise}|{i}".encode()))
        bench_run = dataclasses.replace(bench, func=make_noisy_func(bench.func, noise, rng))
    r = cls(bench_run, seed=seed, **kw).optimize(max_evals=max_evals)
    if noise:
        from core.runner import _rescore_noiseless
        r = _rescore_noiseless(r, bench.func)

    recorded = _recorded_best(run_dir, dim_name, func, method, seed)
    verified = recorded is not None and float(f"{r.best_f:.6e}") == recorded

    X = np.asarray(r.history_x, dtype=float)
    F = np.asarray(r.history_f, dtype=float)
    gap = np.maximum(F - bench.optimum, GAP_FLOOR)
    best_gap = np.maximum(np.asarray(r.history_best, dtype=float) - bench.optimum, GAP_FLOOR)
    opt = np.asarray(bench.optima_pos[0], dtype=float) if bench.optima_pos else None
    dist = np.linalg.norm(X - opt, axis=1) if opt is not None else None

    # Population: thinned snapshots, padded into one array + per-frame sizes.
    pops = r.history_pop or []
    if pops:
        sel = np.unique(np.linspace(0, len(pops) - 1, min(len(pops), MAX_POP_FRAMES)).astype(int))
        frames = [np.asarray(pops[k], dtype=float) for k in sel]
        sizes = [len(p) for p in frames]
        pop_flat = np.concatenate([p.reshape(-1, dim) for p in frames]) if frames else np.zeros((0, dim))
        n_gen = len(pops)
        evals_per_gen = (np.asarray(r.history_eval_count, dtype=float)
                         if len(r.history_eval_count) >= n_gen - 1 else None)
        if evals_per_gen is not None and len(evals_per_gen) == n_gen - 1:
            # history_pop[0] is the initial pool (before generation 1)
            evals_per_gen = np.concatenate([[len(pops[0])], evals_per_gen])
        if evals_per_gen is None or len(evals_per_gen) != n_gen:
            evals_per_gen = np.round((np.arange(n_gen) + 1) / n_gen * len(F))
        frame_evals = evals_per_gen[sel].astype(int).tolist()
        spread = np.array([p.std(axis=0) for p in frames]) / (bench.bounds[1] - bench.bounds[0])
        sig = None
        if r.history_pop_sigma and len(r.history_pop_sigma) >= len(pops):
            sig = np.concatenate([np.asarray(r.history_pop_sigma[k], dtype=float).reshape(-1)[:sizes[j]]
                                  for j, k in enumerate(sel)])
    else:
        sizes, pop_flat, frame_evals, spread, sig = [], np.zeros((0, dim)), [], np.zeros((0, dim)), None

    trace = None
    t = r.trace or {}
    if t.get("gen"):
        trace = {
            "gen": {k: [None if (isinstance(v, float) and v != v) else v for v in vals]
                    for k, vals in t["gen"].items()},
            "channels": list(t.get("channels", ())),
            "routes": list(t.get("routes", ())),
            "events": [list(e) for e in t.get("events", [])],
        }
    payload = {
        "run": run_id, "function": func, "method": method, "seed": seed, "dim": dim,
        "bounds": list(bench.bounds), "optimum": opt.tolist() if opt is not None else None,
        "optima": [list(map(float, o)) for o in (bench.optima_pos or [])],
        "n_evals": int(len(F)), "best_f": float(r.best_f), "f_opt": float(bench.optimum), "best_gap": float(best_gap[-1]),
        "recorded_best_f": recorded, "verified": verified,
        "x": _b64(X, np.float32),
        "log_gap": _b64(np.log10(gap), np.float32),
        "log_best_gap": _b64(np.log10(best_gap), np.float32),
        "dist": _b64(dist, np.float32) if dist is not None else None,
        "channel": base64.b64encode(t.get("eval_channel", b"")).decode() if t.get("eval_channel") else None,
        "pop": {"sizes": sizes, "evals": frame_evals, "x": _b64(pop_flat, np.float32),
                "sigma": _b64(sig, np.float32) if sig is not None else None,
                "spread": _b64(spread, np.float32)},
        "trace": trace,
    }
    return json.dumps(payload, separators=(",", ":"))


@functools.lru_cache(maxsize=64)
def landscape(dim: int, func: str, a: int, b: int, n: int = 120) -> str:
    """f over the plane through the (first) optimum spanned by coordinates a, b.

    In 2D this is the whole landscape; in higher dimensions a slice, so the
    background of a 2-coordinate projection shows the function where the other
    coordinates sit at the optimum. Values are log10(f − f_min + 1e-12) on the grid.
    """
    bench = find_benchmark(dim, func)
    if bench is None:
        raise LookupError("benchmark not found")
    lo, hi = bench.bounds
    base = (np.asarray(bench.optima_pos[0], dtype=float) if bench.optima_pos
            else np.full(dim, (lo + hi) / 2))
    g = np.linspace(lo, hi, n)
    Z = np.empty((n, n))
    x = base.copy()
    for i, yv in enumerate(g):
        x[b] = yv
        for j, xv in enumerate(g):
            x[a] = xv
            Z[i, j] = bench.func(x)
    L = np.log10(np.maximum(Z - Z.min(), 0) + 1e-12)
    return json.dumps({"function": func, "dim": dim, "a": a, "b": b, "n": n,
                       "bounds": [lo, hi], "zmin": float(Z.min()),
                       "log_f": _b64(L, np.float32), "slice_at": base.tolist()},
                      separators=(",", ":"))
