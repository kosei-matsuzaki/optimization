"""CMA-ES based HP optimization for MC-ESO sensitive HPs.

Optimizes 20 sensitive HPs identified by hp_sensitivity. The 2 insensitive
HPs (restart_quality_rel_floor, basin_switch_quality_rel_floor) are kept
at defaults.

Objective: minimize mean log10(max(best_f, 1e-300)) across a curated
BBOB+custom training subset. After CMA-ES converges, the best config is
printed for validation on the full BBOB + CEC2022 suites.

To avoid touching baselines fairness, only MC-ESO is run during HP
optimization. Validation against baselines happens separately via the
normal quick_check pipeline.

Usage:
    python -m analysis.hp_optimize [--budget 200] [--seeds 3] [--evals 1500]
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import cma
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.benchmarks import BENCHMARKS_BY_NAME
from core.optimizers import MultiChannelEpidemicOptimizer


# Training subset spanning function classes — keeps eval cost down vs full BBOB.
_TRAIN_FUNCTIONS = [
    "F01-Sphere",           # separable unimodal sanity
    "F04-BucheRastrigin",   # separable multimodal
    "F10-EllipsoidalRot",   # ill-cond
    "F14-DiffPowers",       # ill-cond, diminishing returns
    "F17-SchafferF7",       # multimodal
    "F23-Katsuura",         # weak structure
    "F24-LunacekRastrigin", # double funnel (basin_switch target)
    "C09-Easom",            # needle in haystack
]


# (HP name, lower bound, upper bound, integer?, log-scale?)
_HP_SPACE: list[tuple[str, float, float, bool, bool]] = [
    # Population/niching
    ("n_pop",                                10,     40,    True,  False),
    ("n_elite_max",                          3,      10,    True,  False),
    ("niche_radius_ratio",                   0.03,   0.30,  False, True),
    # σ
    ("sigma",                                0.05,   0.40,  False, True),
    ("host_sigma_min_scale",                 0.01,   0.20,  False, True),
    # Channels
    ("air_ratio",                            0.10,   0.50,  False, False),
    ("h2h_ratio",                            0.20,   0.60,  False, False),
    ("h2h_F",                                0.30,   0.90,  False, False),
    ("air_sigma_amplifier",                  1.5,    7.0,   False, False),
    ("kill_fraction",                        0.10,   0.50,  False, False),
    # Restart
    ("restart_no_improve_threshold",         100,    600,   True,  False),
    ("restart_sigma_ratio",                  0.10,   0.60,  False, False),
    ("basin_switch_after_failed_spillovers", 1,      4,     True,  False),
    # σ adaptation
    ("sigma_up",                             1.02,   1.25,  False, False),
    ("sigma_down",                           0.80,   0.99,  False, False),
    ("precision_sigma_ratio",                1e-4,   1e-2,  False, True),
    ("sigma_drill_down",                     0.70,   0.95,  False, False),
    # Misc
    ("log_slope_threshold",                  1e-5,   1e-3,  False, True),
    ("h2h_CR",                               0.50,   1.00,  False, False),
    ("empirical_cov_floor",                  1e-3,   0.10,  False, True),
]


def _normalize(value: float, lo: float, hi: float, log: bool) -> float:
    """Map value → [0, 1] for CMA-ES."""
    if log:
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)


def _denormalize(z: float, lo: float, hi: float, log: bool, is_int: bool) -> float:
    """Map [0, 1] → HP value."""
    z = max(0.0, min(1.0, z))
    if log:
        v = math.exp(math.log(lo) + z * (math.log(hi) - math.log(lo)))
    else:
        v = lo + z * (hi - lo)
    if is_int:
        v = int(round(v))
    return v


def _vec_to_kwargs(vec: np.ndarray) -> dict:
    """Translate CMA-ES vector (in [0,1]^d) to HP kwargs."""
    kw: dict = {}
    for z, (name, lo, hi, is_int, log) in zip(vec, _HP_SPACE):
        kw[name] = _denormalize(float(z), lo, hi, log, is_int)
    return kw


def _evaluate(kwargs: dict, seeds: list[int], max_evals: int) -> float:
    """Mean log10(best_f) across training functions and seeds (lower = better)."""
    logs: list[float] = []
    for fname in _TRAIN_FUNCTIONS:
        bench = BENCHMARKS_BY_NAME[fname]
        for seed in seeds:
            try:
                opt = MultiChannelEpidemicOptimizer(bench, seed=seed, **kwargs)
                res = opt.optimize(max_evals=max_evals)
                logs.append(math.log10(max(res.best_f, 1e-300)))
            except Exception:
                logs.append(100.0)  # penalty for crash
    return float(np.mean(logs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=200,
                        help="CMA-ES total candidate evaluations")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Inner seeds per (config, function)")
    parser.add_argument("--evals", type=int, default=1500,
                        help="Inner max_evals per MC-ESO run")
    parser.add_argument("--sigma0", type=float, default=0.15,
                        help="CMA-ES initial σ on normalized [0,1] space")
    args = parser.parse_args()

    # Baseline from defaults
    import inspect
    sig = inspect.signature(MultiChannelEpidemicOptimizer.__init__)
    defaults = {n: p.default for n, p in sig.parameters.items()
                if p.default is not inspect.Parameter.empty}

    seeds = list(range(args.seeds))

    # Initial point = current defaults, normalised
    x0 = []
    for name, lo, hi, is_int, log in _HP_SPACE:
        z = _normalize(float(defaults[name]), lo, hi, log)
        x0.append(max(0.05, min(0.95, z)))  # clip to interior

    print(f"=== CMA-ES HP optimization ===")
    print(f"  HPs: {len(_HP_SPACE)}, train funcs: {len(_TRAIN_FUNCTIONS)},")
    print(f"  inner seeds: {args.seeds}, inner evals: {args.evals},")
    print(f"  CMA-ES candidate budget: {args.budget}, σ0: {args.sigma0}")

    print("\nBaseline (current defaults)...")
    t0 = time.time()
    base_score = _evaluate({}, seeds, args.evals)
    print(f"  → score {base_score:+.3f} ({time.time()-t0:.0f}s)")

    opts = cma.CMAOptions()
    opts["bounds"] = [[0.0] * len(_HP_SPACE), [1.0] * len(_HP_SPACE)]
    opts["maxfevals"] = args.budget
    opts["verbose"] = -1   # quiet
    opts["popsize"] = max(8, 4 + int(3 * math.log(len(_HP_SPACE))))
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)

    best_score = base_score
    best_kw = dict()  # empty = defaults
    seen = 0
    while not es.stop():
        candidates = es.ask()
        scores = []
        for c in candidates:
            kw = _vec_to_kwargs(np.array(c))
            s = _evaluate(kw, seeds, args.evals)
            scores.append(s)
            seen += 1
            marker = ""
            if s < best_score:
                best_score = s
                best_kw = kw
                marker = "  ← NEW BEST"
            print(f"  [{seen:3d}/{args.budget}] score={s:+.3f}{marker}")
        es.tell(candidates, scores)

    print(f"\n=== Best config (Δ vs baseline = {best_score - base_score:+.3f}) ===")
    print(f"Baseline score: {base_score:+.3f}")
    print(f"Best score:     {best_score:+.3f}")
    print("\nBest HP values (only those changed from default shown):")
    for name, _, _, _, _ in _HP_SPACE:
        if name not in best_kw:
            continue
        v_best = best_kw[name]
        v_def = defaults[name]
        if v_best != v_def:
            print(f"  {name}: {v_def} → {v_best}")


if __name__ == "__main__":
    main()
