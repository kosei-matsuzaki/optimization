"""HP sensitivity test for MC-ESO.

For each HP, perturb the default by ±factor and measure the change in
mean best_f on a curated BBOB subset. HPs whose perturbation produces
small changes are "insensitive" — candidates for hard-coding.

The metric used is mean log10(max(best_f, 1e-300)) averaged over the
test functions, which is shift-tolerant for very small / 0 best values.
Each (HP, perturbation) combination is evaluated with a small seed pool
to keep the cost tractable.

Usage:
    python -m analysis.hp_sensitivity [--seeds 5] [--max-evals 2000]
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.benchmarks import BENCHMARKS_BY_NAME
from core.optimizers import MultiChannelEpidemicOptimizer


# Subset spanning separable / ill-cond / multimodal / weak-structure
# / 2D custom — chosen so each HP has at least one function class it can affect.
_FUNCTIONS = [
    "F01-Sphere",
    "F04-BucheRastrigin",
    "F10-EllipsoidalRot",
    "F17-SchafferF7",
    "F23-Katsuura",
    "F24-LunacekRastrigin",
    "C09-Easom",
]

# HPs and the perturbation factor to apply. "perturb" of 0.5 means test
# at default×0.5 and default×2 (so ±factor on multiplicative scale).
# For integer HPs the perturbation is rounded.
_HPS_TO_TEST: list[tuple[str, float, str]] = [
    # (hp_name, perturbation_factor, type: "mult" or "add")
    ("n_pop",                                0.5, "mult"),
    ("n_elite_max",                          0.5, "mult"),
    ("niche_radius_ratio",                   0.5, "mult"),
    ("sigma",                                0.5, "mult"),
    ("host_sigma_min_scale",                 0.5, "mult"),
    ("air_ratio",                            0.5, "mult"),
    ("h2h_ratio",                            0.5, "mult"),
    ("h2h_F",                                0.5, "mult"),
    ("air_sigma_amplifier",                  0.5, "mult"),
    ("kill_fraction",                        0.5, "mult"),
    ("restart_no_improve_threshold",         0.5, "mult"),
    ("restart_sigma_ratio",                  0.5, "mult"),
    ("restart_quality_rel_floor",            0.01, "mult"),  # 1e-8 → 1e-10 / 1e-6
    ("basin_switch_after_failed_spillovers", 1.0, "add"),    # 2 → 1 / 3
    ("basin_switch_quality_rel_floor",       0.1, "mult"),
    ("sigma_up",                             0.05, "add"),   # 1.1 → 1.05 / 1.15
    ("sigma_down",                           0.05, "add"),
    ("precision_sigma_ratio",                0.1, "mult"),   # 1e-3 → 1e-4 / 1e-2
    ("sigma_drill_down",                     0.05, "add"),
    ("log_slope_threshold",                  0.1, "mult"),
    ("h2h_CR",                               0.1, "add"),    # 0.9 → 0.8 / 1.0
    ("empirical_cov_floor",                  0.1, "mult"),
]

# HPs whose ±50% multiplicative perturbation isn't well-defined
# (numerical floors / ceilings). Skip.
_HPS_TO_SKIP = ["sigma_floor_ratio", "sigma_ceil_ratio"]


def _eval_config(kwargs: dict, seeds: list[int], max_evals: int) -> float:
    """Return mean log10(best_f) across functions and seeds for given kwargs."""
    log_bests: list[float] = []
    for fname in _FUNCTIONS:
        bench = BENCHMARKS_BY_NAME[fname]
        for seed in seeds:
            opt = MultiChannelEpidemicOptimizer(bench, seed=seed, **kwargs)
            res = opt.optimize(max_evals=max_evals)
            log_bests.append(math.log10(max(res.best_f, 1e-300)))
    return float(np.mean(log_bests))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-evals", type=int, default=2000)
    parser.add_argument("--sensitive-threshold", type=float, default=0.5,
                        help="Δ log10(best_f) threshold to call an HP sensitive")
    args = parser.parse_args()

    import inspect
    sig = inspect.signature(MultiChannelEpidemicOptimizer.__init__)
    defaults = {n: p.default for n, p in sig.parameters.items()
                if p.default is not inspect.Parameter.empty}

    seeds = list(range(args.seeds))
    print(f"=== HP sensitivity ({args.seeds} seeds × {len(_FUNCTIONS)} funcs, "
          f"max_evals={args.max_evals}) ===")

    print("\nBaseline (default HPs)...")
    base = _eval_config({}, seeds, args.max_evals)
    print(f"  baseline mean log10(best_f) = {base:+.3f}\n")

    print(f"{'HP':<40} {'lo':>10} {'hi':>10} {'Δ_lo':>8} {'Δ_hi':>8} {'verdict':>10}")
    print("-" * 100)
    results: list[tuple[str, float, float, float, float]] = []
    for hp, perturb, kind in _HPS_TO_TEST:
        if hp not in defaults:
            print(f"{hp:<40}  (not in signature — skipping)")
            continue
        default = defaults[hp]
        if kind == "mult":
            lo_val = default * perturb
            hi_val = default / perturb
        else:  # "add"
            lo_val = default - perturb
            hi_val = default + perturb
        # Integer HPs: round
        if isinstance(default, int):
            lo_val = max(1, int(round(lo_val)))
            hi_val = int(round(hi_val))

        lo_score = _eval_config({hp: lo_val}, seeds, args.max_evals)
        hi_score = _eval_config({hp: hi_val}, seeds, args.max_evals)
        d_lo = lo_score - base
        d_hi = hi_score - base
        max_dev = max(abs(d_lo), abs(d_hi))
        verdict = "SENSITIVE" if max_dev >= args.sensitive_threshold else "insensitive"
        results.append((hp, default, lo_val, hi_val, d_lo, d_hi, verdict))
        print(f"{hp:<40} {str(lo_val):>10} {str(hi_val):>10} "
              f"{d_lo:+8.2f} {d_hi:+8.2f} {verdict:>10}")

    print("\n=== Summary: insensitive HPs (candidates to hard-code) ===")
    for hp, default, lo, hi, d_lo, d_hi, verdict in results:
        if verdict == "insensitive":
            print(f"  {hp} = {default} (Δ_lo={d_lo:+.2f}, Δ_hi={d_hi:+.2f})")


if __name__ == "__main__":
    main()
