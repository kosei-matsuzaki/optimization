"""Lightweight local sanity check.

Runs a small subset of BBOB functions with reduced settings so results
are visible in under a minute. Full experiments go through GitHub Actions.

Usage:
    python quick_check.py
    python quick_check.py --n-runs 5 --max-evals 3000
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
from pathlib import Path

from core.benchmarks import (
    BENCHMARKS_BY_NAME, BENCHMARKS_3D_BY_NAME,
    BENCHMARKS_5D_BY_NAME, BENCHMARKS_10D_BY_NAME, BENCHMARKS_20D_BY_NAME,
    BENCHMARKS_CEC2022_10D_BY_NAME, NOISE_MODELS,
)
from core.optimizers import (
    CMAESOptimizer, MultiChannelEpidemicOptimizer, PSOOptimizer,
    DEOptimizer, SaVOAOptimizer,
    MultistartNelderMeadOptimizer, NCDEOptimizer,
    LSHADEOptimizer, IPOPCMAESOptimizer, BIPOPCMAESOptimizer,
)
from core.optimizers.mceso_ablations import (
    MCESONoSpillover, MCESONoHostCompetition,
)
from core.runner import (run_experiment, summarize, wilcoxon_vs_reference,
                         peak_metrics)
from core.visualize import (
    save_landscape_svg, save_convergence_svg,
    save_method_runs_anim, save_method_evals_anim, save_method_population_anim,
    save_method_3devals_anim, save_method_3dpopulation_anim,
    save_method_vso_svg, save_stats,
)


def _append_wilcoxon(dim_dir: Path, bench_name: str,
                     results_per_method: dict, reference: str = "MC-ESO") -> None:
    """Append paired Wilcoxon signed-rank rows comparing ``reference`` vs each
    other method, for this benchmark's run results."""
    if reference not in results_per_method:
        return
    ref_bests = np.array([r.best_f for r in results_per_method[reference]])
    path = dim_dir / "wilcoxon.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "function", "reference", "method", "n", "win_count", "tie_count",
            "p_value_two_sided", "p_value_ref_better",
            "a12", "a12_magnitude",
        ])
        if write_header:
            writer.writeheader()
        for method, results in results_per_method.items():
            if method == reference:
                continue
            method_bests = np.array([r.best_f for r in results])
            # We test "is reference better than method?" → reference (cand) < method (ref)
            stat = wilcoxon_vs_reference(ref_bests, method_bests)
            writer.writerow({
                "function": bench_name,
                "reference": reference,
                "method": method,
                "n": stat["n"],
                "win_count": stat["win_count"],
                "tie_count": stat["tie_count"],
                "p_value_two_sided": f"{stat['p_value']:.4g}",
                "p_value_ref_better": f"{stat['p_less']:.4g}",
                "a12":                f"{stat['a12']:.4f}",
                "a12_magnitude":      stat["a12_magnitude"],
            })

# Curated 12-function subset — two representatives per BBOB group. BBOB-only:
# custom benchmarks are opt-in (--custom) per the 2D-BBOB evaluation standard.
# Used as the default quick set.
_QUICK_FUNCTIONS: list[str] = [
    "F01-Sphere",            # separable        — unimodal baseline
    "F03-RastriginSep",      # separable        — separable multimodal
    "F08-Rosenbrock",        # moderate-cond    — banana valley
    "F09-RosenbrockRot",     # moderate-cond    — rotated, harder
    "F10-EllipsoidalRot",    # ill-cond         — cond ≈ 10^6
    "F12-BentCigar",         # ill-cond         — extreme cond ≈ 10^6
    "F15-RastriginRot",      # multimodal       — structured landscape
    "F16-Weierstrass",       # multimodal       — highly rugged
    "F17-SchafferF7",        # multimodal       — irregular rough landscape
    "F20-Schwefel",          # weak-structure   — deceptive optima
    "F21-Gallagher101",      # weak-structure   — 101 Gaussian peaks
    "F24-LunacekRastrigin",  # weak-structure   — deceptive double funnel
]

# Full BBOB-24 set (F01-F24) + 2D-only custom benchmarks. Built programmatically
# so adding new benchmarks doesn't require touching this list.
_BBOB_NAMES: list[str] = sorted(
    {n for n in BENCHMARKS_BY_NAME if n.startswith("F")},
    key=lambda n: int(n[1:3]),
)
_CUSTOM_2D_ONLY: list[str] = sorted(
    {n for n in BENCHMARKS_BY_NAME if n.startswith("C")},
    key=lambda n: int(n[1:3]),
)
# Standard evaluation set (--all): 2D BBOB-24 only. Custom benchmarks are
# opt-in via --custom (multimodal / multi-optima focus) — see docs/experiments.md.
_ALL_FUNCTIONS: list[str] = _BBOB_NAMES

# Held-out CEC2022 suite (dim=10). Independent of BBOB transformations —
# used to test whether MC-ESO mechanisms generalize beyond the BBOB suite
# they were tuned against. Selected with --suite cec2022 (forces --dim 10).
_CEC2022_NAMES: list[str] = sorted(BENCHMARKS_CEC2022_10D_BY_NAME)

# BBOB registries keyed by dimension. n = 2, 3, 5, 10, 20 are supported for the
# BBOB suite (dimension-scaling snapshot). The CEC2022 hold-out is a separate
# suite (--suite cec2022) with its own dim=10 registry, so BBOB's dim=10 no
# longer collides with it.
_DIM_REGISTRIES: dict[int, dict[str, object]] = {
    2:  BENCHMARKS_BY_NAME,
    3:  BENCHMARKS_3D_BY_NAME,
    5:  BENCHMARKS_5D_BY_NAME,
    10: BENCHMARKS_10D_BY_NAME,
    20: BENCHMARKS_20D_BY_NAME,
}

# MC-ESO (Multi-Channel Epidemic Spread Optimizer): all core mechanisms
# — 3-channel transmission with h2h CR=0.9, rotation-aware close-contact
# (empirical covariance + adaptive anisotropy floor), drilling-mode airborne
# suppression, informed-restart spillover (reservoir re-ignition + basin-memory
# repulsion), sequential niching (σ-exhaustion), host competition with rollback,
# σ adapt — are baked into the base implementation. This dict is the standard
# comparison: MC-ESO vs the 9 baselines. Diagnostic / ablation variants are NOT
# registered here (they live in core/optimizers/mceso_ablations.py and can be
# added back temporarily when isolating a mechanism's contribution).
_OPTIMIZERS = {
    "CMA-ES":       (CMAESOptimizer,                {}),
    "IPOP-CMA-ES":  (IPOPCMAESOptimizer,            {}),
    "BIPOP-CMA-ES": (BIPOPCMAESOptimizer,           {}),
    "PSO":          (PSOOptimizer,                  {}),
    "DE":           (DEOptimizer,                   {}),
    "L-SHADE":      (LSHADEOptimizer,               {}),
    "SaVOA":        (SaVOAOptimizer,                {}),
    # Multistart local-search floor: in 2D BBOB a restarted Nelder-Mead is a
    # strong reference — any metaheuristic gain must clear this bar.
    "NM-Restart":   (MultistartNelderMeadOptimizer, {}),
    # Dedicated niching baseline: the multi-solution (PR / MMOsr) comparison
    # partner for MC-ESO's sequential niching.
    "NCDE":         (NCDEOptimizer,                 {}),
    "MC-ESO":       (MultiChannelEpidemicOptimizer, {}),
    # Temporary entries: cumulative ablation ladder for the progress report
    # (each rung re-enables one committed improvement; MC-ESO = all on).
    #  abl0 = 5/18-era base: blind-uniform restart, fixed high cov floor,
    #         no niching, no router, single-difference droplet.
    "abl0_base2018":  (MultiChannelEpidemicOptimizer, {
        "droplet_variant": "cur2best", "channel_schedule": False,
        "cov_floor_low": 0.01, "exhausted_no_improve_mult": 1e9,
        "ir_archive_frac": 0.0, "ir_repel_max_tries": 0}),
    #  abl1 = + informed restart (reservoir re-ignition + basin repulsion)
    "abl1_ir":        (MultiChannelEpidemicOptimizer, {
        "droplet_variant": "cur2best", "channel_schedule": False,
        "cov_floor_low": 0.01, "exhausted_no_improve_mult": 1e9}),
    #  abl2 = + adaptive anisotropy floor + sequential niching
    "abl2_floornich": (MultiChannelEpidemicOptimizer, {
        "droplet_variant": "cur2best", "channel_schedule": False}),
    #  abl3 = + per-landscape channel router (full MC-ESO minus best2)
    "abl3_router":    (MultiChannelEpidemicOptimizer, {
        "droplet_variant": "cur2best"}),
    # Mechanism-necessity ablations (2026-07): each turns OFF one of the three
    # population-level mechanisms (docs/mceso.md 集団レベルの 3 機構) vs full MC-ESO.
    #  系統共存 OFF: single best strain (no multi-basin donor pool)
    "abl_noStrain":   (MultiChannelEpidemicOptimizer, {"n_elite_max": 1}),
    #  宿主競合 OFF: keep worst-K kill + placement, drop rollback (accept all)
    "abl_noHostComp": (MCESONoHostCompetition,         {}),
    #  スピルオーバー OFF: no stagnation restart (channels grind one basin)
    "abl_noSpill":    (MCESONoSpillover,               {}),
    #  Drilling OFF: no accelerated σ contraction in drilling mode
    "abl_noDrill":    (MultiChannelEpidemicOptimizer, {"sigma_drill_down": 0.95}),
    # Pre-fix reference for the dimension-scaled stagnation window (2026-08-23):
    # the old fixed 300-eval window. Bit-identical to MC-ESO at dim 2 by
    # construction; diverges only at dim ≥ 3. Kept as the regression pin for the
    # high-dimension work (see docs/history.md「次元スケーリングの計測と高次元崩壊」).
    "hd_win0":        (MultiChannelEpidemicOptimizer, {"restart_window_dim_scale": 0.0}),
    # Pre-fix reference for the scale-invariant parent selection (2026-08-24):
    # the raw-f softmax, which flattens to uniform once the population converges
    # (measured effective parent count 20.0 of 20 at dim 2). Regression pin for
    # the audit — see docs/history.md「全パラメータの次元不変性 監査」.
    "dimf_softmax0":  (MultiChannelEpidemicOptimizer, {"softmax_beta": 0.0}),
    # σ-pinning detector (2026-08-25): σ equilibrates at a dimension-independent
    # improvement rate (0.350), which the realised rate meets at dim 10, pinning
    # σ above the drilling threshold — F08/F09 never drill at all. Detected
    # directly as "no drilling for 30% of the budget", which fires on 68-74% of
    # generations there and on 0% of the multimodal functions that every
    # dimension-scaled variant regressed.
    "pin30d5":        (MultiChannelEpidemicOptimizer,
                       {"sigma_pin_evals_frac": 0.30, "sigma_pin_damp": 0.5}),
    # Split close-contact stream (2026-08-26): half the close-contact offspring
    # are shaped by the instantaneous C_pop and half by a persistent rank-μ C
    # started at the identity, with host competition deciding which shape was
    # right. No matrix is blended (additive blending caps anisotropy at ~dim/w
    # and destroys F02), and the close share is raised so the learner is not
    # sample-starved. dim10 SR@1e-10 13.3 → 25.4 at n=10.
    "split70":        (MultiChannelEpidemicOptimizer,
                       {"cc_learning_rate": 0.05, "cc_persist_frac": 0.5}),
    #  Pre-fix reference: the persistent stream off (pure C_pop close-contact).
    "split_off":      (MultiChannelEpidemicOptimizer, {"cc_learning_rate": 0.0}),
}


def _run_dim(benchmarks: list, dim_dir: Path, n_runs: int, max_evals: int,
             optimizers: dict | None = None, noise: str | None = None) -> None:
    """Run all functions in a dimension group and save results to dim_dir."""
    dim_dir.mkdir(parents=True, exist_ok=True)
    if optimizers is None:
        optimizers = _OPTIMIZERS
    # Methods that take a per-benchmark `sigma0` initial step.
    _SIGMA_USERS = (CMAESOptimizer, IPOPCMAESOptimizer, BIPOPCMAESOptimizer)
    print(f"\n{'Function':<22} {'Method':<12} {'Mean':>12} "
          f"{'SR@1e-1':>7} {'SR@1e-2':>7} {'SR@1e-4':>7} {'SR@1e-7':>7} {'SR@1e-10':>8} {'EvalsSucc':>10}")
    print("-" * 102)
    for bench in benchmarks:
        sigma0 = 0.2 * (bench.bounds[1] - bench.bounds[0])
        results_per_method: dict = {}
        times_per_method: dict = {}
        for method, (cls, kwargs) in optimizers.items():
            kw = {**kwargs, **({"sigma0": sigma0} if cls in _SIGMA_USERS else {})}
            results, times = run_experiment(
                cls, bench, n_runs=n_runs, max_evals=max_evals,
                noise_model=noise, **kw
            )
            results_per_method[method] = results
            times_per_method[method] = times
            s = summarize(results)
            ev = s['evals_succ_mean']
            ev_str = f"{ev:>10.0f}" if ev < float('inf') else "       ---"
            # Multi-modal report: for functions with >1 known global optimum,
            # append peak ratio (fraction of optima found) and MMO success rate.
            span = bench.bounds[1] - bench.bounds[0]
            pm = peak_metrics(results, bench.optima_pos, span)
            mmo_str = ""
            if pm["n_optima"] > 1:
                mmo_str = (f"  | K={pm['n_optima']:>2} "
                           f"PR@1e-2={pm['pr_1e-2']:>5.0%} "
                           f"PR@1e-4={pm['pr_1e-4']:>5.0%} "
                           f"MMOsr@1e-4={pm['mmo_sr_1e-4']:>4.0%}")
            print(
                f"{bench.name:<22} {method:<10} "
                f"{s['mean']:>12.4e} "
                f"{s['sr_1e-1']:>6.0%} {s['sr_1e-2']:>6.0%} {s['sr_1e-4']:>6.0%} "
                f"{s['sr_1e-7']:>6.0%} {s['sr_1e-10']:>7.0%}{ev_str}{mmo_str}"
            )

        # Per-method visualizations
        for method_name, results in results_per_method.items():
            if bench.dim == 2:
                save_method_runs_anim(bench, results, method_name, output_dir=dim_dir)
                save_method_evals_anim(bench, results, method_name, output_dir=dim_dir, best=True)
                save_method_evals_anim(bench, results, method_name, output_dir=dim_dir, best=False)
                save_method_population_anim(bench, results, method_name, output_dir=dim_dir, best=True)
                save_method_population_anim(bench, results, method_name, output_dir=dim_dir, best=False)
            elif bench.dim == 3:
                save_method_3devals_anim(bench, results, method_name, output_dir=dim_dir, best=True)
                save_method_3devals_anim(bench, results, method_name, output_dir=dim_dir, best=False)
                save_method_3dpopulation_anim(bench, results, method_name, output_dir=dim_dir, best=True)
                save_method_3dpopulation_anim(bench, results, method_name, output_dir=dim_dir, best=False)
            save_method_vso_svg(bench, results, method_name, output_dir=dim_dir, best=True)
            save_method_vso_svg(bench, results, method_name, output_dir=dim_dir, best=False)

        # Function-level outputs
        save_landscape_svg(bench, output_dir=dim_dir)
        save_convergence_svg(bench, results_per_method, output_dir=dim_dir)
        save_stats(bench, results_per_method, times_per_method, output_dir=dim_dir)
        _append_wilcoxon(dim_dir, bench.name, results_per_method, reference="MC-ESO")

    print(f"Saved → {dim_dir.resolve()}/")


def main(
    n_runs: int = 20,
    max_evals: int = 5000,
    output_dir: Path = Path("results/quick"),
    funcs: list[str] | None = None,
    use_all: bool = False,
    dim: int = 2,
    suite: str = "bbob",
    methods: list[str] | None = None,
    with_custom: bool = False,
    noise: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    if suite == "cec2022":
        if dim != 10:
            print(f"--suite cec2022 forces dim=10 (got {dim})")
            dim = 10
        registry = BENCHMARKS_CEC2022_10D_BY_NAME
        func_set = _CEC2022_NAMES
    else:
        if dim not in _DIM_REGISTRIES:
            raise SystemExit(
                f"--dim {dim} not supported for BBOB suite "
                f"(available: {sorted(_DIM_REGISTRIES)}). "
                "Use --suite cec2022 for the CEC2022 dim=10 hold-out set.")
        registry = _DIM_REGISTRIES[dim]
        # Standard: 2D BBOB-only. Custom benchmarks are opt-in via --custom (or by
        # naming them explicitly with --funcs, which selects from the full registry).
        func_set = _ALL_FUNCTIONS if use_all else _QUICK_FUNCTIONS
        if with_custom:
            func_set = func_set + [n for n in _CUSTOM_2D_ONLY if n not in func_set]
        # Custom benchmarks are 2D-only — drop them for higher dims
        func_set = [n for n in func_set if dim == 2 or n not in _CUSTOM_2D_ONLY]
        # Explicit --funcs may name custom benchmarks not in the BBOB-only set;
        # make them selectable at dim=2 without requiring --custom.
        if funcs and dim == 2:
            named_custom = [n for n in _CUSTOM_2D_ONLY
                            if n in funcs and n not in func_set]
            func_set = func_set + named_custom
    # Filter optimizers by --methods (preserve order from _OPTIMIZERS).
    if methods:
        method_filter = {m.strip() for m in methods if m and m.strip()}
        unknown = method_filter - set(_OPTIMIZERS)
        if unknown:
            raise SystemExit(
                f"Unknown method(s): {sorted(unknown)}.  "
                f"Available: {list(_OPTIMIZERS)}")
        optimizers = {k: v for k, v in _OPTIMIZERS.items() if k in method_filter}
    else:
        optimizers = _OPTIMIZERS

    print(f"quick_check  suite={suite}  dim={dim}  n_runs={n_runs}  "
          f"max_evals={max_evals}  "
          f"set={'all' if use_all else 'quick'}  custom={with_custom}  "
          f"noise={noise or 'off'}  "
          f"funcs={funcs or 'all'}  methods={list(optimizers)}")
    if noise:
        print("NOISE MODE: optimizers observe the noisy f; all reported metrics "
              "are re-scored on the noise-free f of visited points "
              "(COCO-noisy convention, noise-free below 1e-8).")

    func_filter = set(funcs) if funcs else None
    benchmarks: list = []
    for fname in func_set:
        if func_filter is not None and fname not in func_filter:
            continue
        if fname not in registry:
            print(f"  skip {fname} (not in dim{dim} registry)")
            continue
        benchmarks.append(registry[fname])

    if not benchmarks:
        raise SystemExit(f"No matching functions for filter {funcs} at dim={dim}")

    print(f"\n=== dim{dim} ===")
    _run_dim(benchmarks, output_dir / f"dim{dim}", n_runs, max_evals,
             optimizers=optimizers, noise=noise)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",     type=int, default=20,                   help="Number of runs per method")
    parser.add_argument("--max-evals",  type=int, default=5000,                 help="Max function evaluations per run")
    parser.add_argument("--output-dir", type=Path, default=Path("results/quick"), help="Output directory")
    parser.add_argument("--funcs",      type=str, default=None,
                        help="Comma-separated function names to run (default: all in selected set)")
    parser.add_argument("--all",        action="store_true",
                        help="Use the full 2D BBOB-24 set (F01-F24) instead of the quick subset. "
                             "BBOB-only — add --custom to also run the C01-C11 custom benchmarks.")
    parser.add_argument("--custom",     action="store_true",
                        help="Also run the 2D-only custom benchmarks (C01-C11) — opt-in for "
                             "multimodal / multi-optima focus. Ignored for dim != 2.")
    parser.add_argument("--dim",        type=int, default=2, choices=sorted(_DIM_REGISTRIES),
                        help="BBOB problem dimension (default 2; one of 2/3/5/10/20). "
                             "Custom C01-C11 are 2D-only and skipped for higher dims. "
                             "For the CEC2022 hold-out use --suite cec2022 (forces dim=10).")
    parser.add_argument("--suite",      type=str, default="bbob", choices=["bbob", "cec2022"],
                        help="Benchmark suite. 'bbob' (default) uses BBOB-24 + custom; "
                             "'cec2022' uses the 12-function CEC2022 hold-out at dim=10.")
    parser.add_argument("--methods",    type=str, default=None,
                        help="Comma-separated optimizer names to run "
                             "(default: all registered methods).  "
                             "Available: " + ", ".join(_OPTIMIZERS.keys()))
    parser.add_argument("--noise",      type=str, default=None, choices=list(NOISE_MODELS),
                        help="Evaluation-noise mode: optimizers observe a noisy f "
                             "(multiplicative, BBOB-noisy style); metrics are re-scored "
                             "on the noise-free f of visited points.")
    args = parser.parse_args()
    funcs_list = [s.strip() for s in args.funcs.split(",")] if args.funcs else None
    methods_list = [s.strip() for s in args.methods.split(",")] if args.methods else None
    main(n_runs=args.n_runs, max_evals=args.max_evals, output_dir=args.output_dir,
         funcs=funcs_list, use_all=args.all, dim=args.dim, suite=args.suite,
         methods=methods_list, with_custom=args.custom, noise=args.noise)
