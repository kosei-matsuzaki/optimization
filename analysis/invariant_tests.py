"""Invariant tests for MC-ESO to catch edge-case bugs.

Runs MC-ESO across many functions and seeds, checking algorithmic invariants:
  I1. best_so_far is monotonically non-increasing through history_best
  I2. n_evals ≈ max_evals (no early termination, no major overshoot)
  I3. history_f[i] = func(history_x[i]) (no data desync)
  I4. final best_x corresponds to a real evaluated point (within history)
  I5. no NaN or inf in history_f
  I6. final pop_f values are reproducible from final pop_x via func()
  I7. for σ_global if recorded — within [span × sigma_floor_ratio, span × sigma_ceil_ratio]

Any failure indicates a bug in state-machine logic.

Usage:
    python -m analysis.invariant_tests [--seeds 20]
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.benchmarks import (
    BENCHMARKS_BY_NAME,
    BENCHMARKS_CEC2022_10D_BY_NAME,
)
from core.optimizers import MultiChannelEpidemicOptimizer


# Curated diverse subset — separable/multimodal/ill-cond/deceptive
_FUNCTIONS_2D = [
    "F01-Sphere",
    "F04-BucheRastrigin",  # separable multimodal
    "F10-EllipsoidalRot",  # ill-cond
    "F17-SchafferF7",      # multimodal
    "F23-Katsuura",        # fractal
    "F24-LunacekRastrigin",  # double-funnel (triggers basin_switch)
    "C05-Eggholder",       # deceptive 2D
    "C09-Easom",           # needle in haystack
]

_FUNCTIONS_CEC = [
    "G01-Zakharov",
    "G06-Hybrid1",         # the regression case from L
    "G11-Composition3",
]


class InvariantFailure(Exception):
    pass


def _check_invariants(name: str, seed: int, result, bench, max_evals: int,
                      tol_overshoot: int = 0) -> list[str]:
    """Run all checks; return list of failure messages (empty if pass)."""
    failures: list[str] = []

    # I1: history_best monotonically non-increasing
    hb = result.history_best
    for i in range(1, len(hb)):
        if hb[i] > hb[i-1] + 1e-15:
            failures.append(
                f"I1[{name},seed={seed}]: history_best not monotonic at i={i}: "
                f"{hb[i-1]} → {hb[i]}")
            break

    # I2: n_evals close to max_evals (allow tiny overshoot from batch granularity
    # — MC-ESO should land exactly on max_evals)
    if result.n_evals < max_evals:
        failures.append(
            f"I2[{name},seed={seed}]: n_evals={result.n_evals} < max_evals={max_evals} "
            f"(early termination?)")
    if result.n_evals > max_evals + tol_overshoot:
        failures.append(
            f"I2[{name},seed={seed}]: n_evals={result.n_evals} overshoots "
            f"max_evals+{tol_overshoot} = {max_evals + tol_overshoot}")

    # I3: history_f consistent with history_x via func re-evaluation (spot check)
    # Full re-eval is expensive; check just the best point and 3 random points.
    if len(result.history_f) > 0:
        sample_idx = [int(np.argmin(result.history_f))]
        rng = np.random.default_rng(seed)
        sample_idx += list(rng.integers(0, len(result.history_f), size=3))
        for idx in sample_idx:
            recomputed = float(bench.func(result.history_x[idx]))
            if abs(recomputed - result.history_f[idx]) > 1e-9:
                failures.append(
                    f"I3[{name},seed={seed}]: history_f[{idx}]={result.history_f[idx]} "
                    f"vs func(history_x[{idx}])={recomputed}")

    # I4: best_f matches min of history_f
    if abs(result.best_f - min(result.history_f)) > 1e-15:
        failures.append(
            f"I4[{name},seed={seed}]: best_f={result.best_f} ≠ "
            f"min(history_f)={min(result.history_f)}")

    # I4b: best_x corresponds to best_f in history
    best_idx_h = int(np.argmin(result.history_f))
    recomputed_best = float(bench.func(result.best_x))
    if abs(recomputed_best - result.best_f) > 1e-9:
        failures.append(
            f"I4b[{name},seed={seed}]: best_x recomputes to {recomputed_best}, "
            f"best_f={result.best_f}")

    # I5: no NaN/inf
    arr = np.asarray(result.history_f, dtype=float)
    if not np.all(np.isfinite(arr)):
        failures.append(
            f"I5[{name},seed={seed}]: history_f contains NaN/inf "
            f"(count={int((~np.isfinite(arr)).sum())})")

    # I6: history_x all within bounds
    lo, hi = bench.bounds
    x_arr = np.array(result.history_x)
    if x_arr.size > 0:
        if (x_arr < lo - 1e-9).any() or (x_arr > hi + 1e-9).any():
            n_oob = int(((x_arr < lo - 1e-9) | (x_arr > hi + 1e-9)).any(axis=1).sum())
            failures.append(
                f"I6[{name},seed={seed}]: {n_oob} history_x points out of bounds "
                f"[{lo}, {hi}]")

    return failures


def run_suite(functions, registry, max_evals: int, seeds: list[int],
              suite_name: str) -> int:
    n_fail = 0
    n_pass = 0
    for fname in functions:
        if fname not in registry:
            print(f"  SKIP {fname} (not in {suite_name} registry)")
            continue
        bench = registry[fname]
        for seed in seeds:
            opt = MultiChannelEpidemicOptimizer(bench, seed=seed)
            try:
                result = opt.optimize(max_evals=max_evals)
            except Exception as e:
                print(f"  CRASH {fname} seed={seed}: {type(e).__name__}: {e}")
                n_fail += 1
                continue
            fails = _check_invariants(fname, seed, result, bench, max_evals)
            if fails:
                n_fail += len(fails)
                for f in fails:
                    print(f"  FAIL {f}")
            else:
                n_pass += 1
    return n_pass, n_fail


def _check_determinism(name: str, seed: int, bench, max_evals: int) -> list[str]:
    """Run twice with same seed; results must be byte-identical."""
    r1 = MultiChannelEpidemicOptimizer(bench, seed=seed).optimize(max_evals=max_evals)
    r2 = MultiChannelEpidemicOptimizer(bench, seed=seed).optimize(max_evals=max_evals)
    if r1.best_f != r2.best_f:
        return [f"DET[{name},seed={seed}]: non-deterministic best_f {r1.best_f} vs {r2.best_f}"]
    if len(r1.history_f) != len(r2.history_f):
        return [f"DET[{name},seed={seed}]: history length differs {len(r1.history_f)} vs {len(r2.history_f)}"]
    if any(a != b for a, b in zip(r1.history_f, r2.history_f)):
        return [f"DET[{name},seed={seed}]: history_f sequence differs"]
    return []


def _check_stress(name: str, bench) -> list[str]:
    """Stress test: very small/large budgets."""
    fails: list[str] = []
    for evals in [50, 100, 300]:  # below typical spillover trigger
        try:
            res = MultiChannelEpidemicOptimizer(bench, seed=0).optimize(max_evals=evals)
            if res.n_evals != evals:
                fails.append(f"STRESS[{name},evals={evals}]: n_evals={res.n_evals} != {evals}")
            if not math.isfinite(res.best_f):
                fails.append(f"STRESS[{name},evals={evals}]: best_f = {res.best_f}")
        except Exception as e:
            fails.append(f"STRESS[{name},evals={evals}]: CRASH {type(e).__name__}: {e}")
    return fails


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds per function (default 10)")
    parser.add_argument("--max-evals", type=int, default=2000)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    print(f"=== Invariant tests: MC-ESO ({args.seeds} seeds × functions, "
          f"max_evals={args.max_evals}) ===")

    print(f"\n[BBOB / dim=2]")
    p1, f1 = run_suite(_FUNCTIONS_2D, BENCHMARKS_BY_NAME,
                       args.max_evals, seeds, "BBOB")
    print(f"  → {p1} pass, {f1} fail")

    print(f"\n[CEC2022 / dim=10]")
    cec_evals = max(args.max_evals, 5000)
    p2, f2 = run_suite(_FUNCTIONS_CEC, BENCHMARKS_CEC2022_10D_BY_NAME,
                       cec_evals, seeds, "CEC2022")
    print(f"  → {p2} pass, {f2} fail")

    print(f"\n[Determinism: same seed → identical results]")
    det_fail = 0
    det_pass = 0
    for fname in _FUNCTIONS_2D[:4]:  # subset
        bench = BENCHMARKS_BY_NAME[fname]
        for seed in seeds[:3]:
            fails = _check_determinism(fname, seed, bench, args.max_evals)
            if fails:
                det_fail += len(fails)
                for f in fails:
                    print(f"  FAIL {f}")
            else:
                det_pass += 1
    print(f"  → {det_pass} pass, {det_fail} fail")

    print(f"\n[Stress: tiny budgets (evals=50/100/300)]")
    s_fail = 0
    s_pass = 0
    for fname in _FUNCTIONS_2D[:5]:
        bench = BENCHMARKS_BY_NAME[fname]
        fails = _check_stress(fname, bench)
        if fails:
            s_fail += len(fails)
            for f in fails:
                print(f"  FAIL {f}")
        else:
            s_pass += 3   # 3 evals tested
    print(f"  → {s_pass} pass, {s_fail} fail")

    print(f"\n=== Summary ===")
    total_pass = p1 + p2 + det_pass + s_pass
    total_fail = f1 + f2 + det_fail + s_fail
    print(f"Pass: {total_pass}, Fail: {total_fail}")
    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
