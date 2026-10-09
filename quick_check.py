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
    BENCHMARKS_CEC2022_10D_BY_NAME, NICHING_BENCHMARKS_BY_NAME, NOISE_MODELS,
)
from core.optimizers import (
    CMAESOptimizer, MultiChannelEpidemicOptimizer, PSOOptimizer,
    DEOptimizer, SaVOAOptimizer,
    MultistartNelderMeadOptimizer, NCDEOptimizer,
    RingPSOOptimizer, NMMSOOptimizer, MAPElitesOptimizer,
    LSHADEOptimizer, IPOPCMAESOptimizer, BIPOPCMAESOptimizer,
    RepellingCMAESOptimizer,
)
from core.optimizers.mceso_ablations import (
    MCESONoSpillover, MCESONoHostCompetition,
)
from core.optimizers.mceso_commit_reseed import CommitReseedMCESO
from core.optimizers.mceso_phased_accept import PhasedAcceptMCESO
# Hybrid / state-of-the-art comparison candidates (2026-10-06). Registered so
# they can be run by name; not in any default method list yet.
from core.optimizers.ea4eig import (EA4eigOptimizer, EA4eigJsoIdebdOptimizer,
                                    EA4eigSimplifiedOptimizer)
from core.optimizers.lshade_spacma import LSHADESPACMAOptimizer
from core.optimizers.lshade_port import LSHADEPortOptimizer
from core.optimizers.imode import IMODEOptimizer as IMODEPortOptimizer
from core.optimizers.elshade_spacma import ELSHADESPACMAOptimizer
from core.optimizers.apgsk_imode import APGSKIMODEOptimizer
from core.optimizers.sps_lshade_eig import SPSLSHADEEIGOptimizer
from core.optimizers.cobide import CoBiDEOptimizer
from core.optimizers.amalgam_so import AMALGAMSOOptimizer
from core.optimizers.hses import HSESOptimizer
from core.optimizers.icmaes_ils import ICMAESILSOptimizer
from core.optimizers.mos import MOSOptimizer
from core.optimizers.umoea import UMOEAIIOptimizer
from core.optimizers.ebowithcmar import EBOwithCMAROptimizer
from core.optimizers.hmhh import HMHHOptimizer
from core.optimizers.ps_cmaes import PSCMAESOptimizer
from core.optimizers.depso import DEPSOOptimizer
from core.optimizers.jso import JSOPortOptimizer
from core.optimizers.lsrtde import LSRTDEOptimizer as LSRTDEPortOptimizer
from core.optimizers.lib_wrappers import (IMODEOptimizer, LSHADEcnEpSinOptimizer,
                                          JSOOptimizer as JSOLibOptimizer,
                                          LSRTDEOptimizer as LSRTDELibOptimizer,
                                          NGOptOptimizer, NGPortfolioOptimizer)
from core.env_info import write_env
from core.run_data import save_curves, append_mceso_runs
from core.runner import (run_experiment, summarize, wilcoxon_vs_reference,
                         peak_metrics, niching_peak_metrics, niching_peak_counts)
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

def _append_wilcoxon_pr(dim_dir: Path, bench, results_per_method: dict,
                        reference: str = "MC-ESO") -> None:
    """Same paired test as ``_append_wilcoxon`` but on the per-run peak count
    (averaged over the accuracy levels) instead of best_f — the multi-solution
    side needs its own significance test, since winning on depth says nothing
    about how many optima a method reported.

    Counts are negated before the test so that "lower is better" still holds and
    a12 > 0.5 keeps meaning "the reference is better", i.e. finds more peaks.
    Niching suite only; other suites have a single global optimum.
    """
    if reference not in results_per_method or not getattr(bench, "n_global_optima", None):
        return
    ref_counts = -niching_peak_counts(results_per_method[reference], bench)
    path = dim_dir / "wilcoxon_pr.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "function", "n_optima", "reference", "method", "n",
            "win_count", "tie_count", "p_value_two_sided", "p_value_ref_better",
            "a12", "a12_magnitude", "mean_peaks_ref", "mean_peaks_method",
        ])
        if write_header:
            writer.writeheader()
        for method, results in results_per_method.items():
            if method == reference:
                continue
            method_counts = -niching_peak_counts(results, bench)
            stat = wilcoxon_vs_reference(ref_counts, method_counts)
            writer.writerow({
                "function": bench.name,
                "n_optima": bench.n_global_optima,
                "reference": reference,
                "method": method,
                "n": stat["n"],
                "win_count": stat["win_count"],
                "tie_count": stat["tie_count"],
                "p_value_two_sided": f"{stat['p_value']:.4g}",
                "p_value_ref_better": f"{stat['p_less']:.4g}",
                "a12":                f"{stat['a12']:.4f}",
                "a12_magnitude":      stat["a12_magnitude"],
                "mean_peaks_ref":     f"{-ref_counts.mean():.2f}",
                "mean_peaks_method":  f"{-method_counts.mean():.2f}",
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

# CEC2013 niching suite, 2-D/3-D subset (--suite niching). Mixed dimensions in
# one suite, so --dim does not apply: each function runs at its own dimension
# and lands in the matching dim{N} directory.
_NICHING_NAMES: list[str] = sorted(NICHING_BENCHMARKS_BY_NAME)

# Default line-up for --suite niching. One row per question worth answering,
# not one row per available method: single-solution methods tuned for
# higher-dimensional black-box search (CMA-ES, PSO, DE, L-SHADE, SaVOA) lose at
# multi-solution for known reasons, so running them here buys nothing. Compute
# saved this way goes into the budget axis instead (docs/related_work.md).
#
#   MC-ESO        the proposal
#   NM-Restart    is a metaheuristic needed at all in 2-3D?
#   IPOP-CMA-ES   how much does a plain restart pick up — and the control for
#                 Repel-CMA-ES, which is IPOP plus repelling
#   Repel-CMA-ES  what does MC-ESO add over a published repelling restart?
#   NCDE          parallel crowding niches vs sequential niching
#   r3pso         is a niche radius worth its cost, against radius-free niching?
#   NMMSO         the competition-grade ceiling
#
# BIPOP-CMA-ES (a second restart-ES row) and Crowding-DE (an ablation of NCDE's
# neighbourhood mutation, not a competitor) stay selectable via --methods.
_NICHING_METHODS: list[str] = [
    "MC-ESO", "NM-Restart", "IPOP-CMA-ES", "Repel-CMA-ES",
    "NCDE", "r3pso", "NMMSO", "MAP-Elites",
]

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
    # ── Hybrid / SOTA comparison candidates (2026-10-06; docs/baselines.md) ──
    "EA4eig":           (EA4eigOptimizer,           {}),
    "EA4eig-jSO-IDEbd": (EA4eigJsoIdebdOptimizer,   {}),
    "EA4eig-Simpl":     (EA4eigSimplifiedOptimizer, {}),
    "LSHADE-SPACMA":    (LSHADESPACMAOptimizer,     {}),
    "AMALGAM-SO":       (AMALGAMSOOptimizer,        {}),
    "AMALGAM-SO-DE":    (AMALGAMSOOptimizer,        {"methods": ("CMA", "GA", "DE")}),
    "HSES":             (HSESOptimizer,             {}),
    "ICMAES-ILS":       (ICMAESILSOptimizer,        {}),
    "MOS":              (MOSOptimizer,              {}),
    "UMOEA-II":         (UMOEAIIOptimizer,          {}),
    "EBOwithCMAR":      (EBOwithCMAROptimizer,      {}),
    "HMHH":             (HMHHOptimizer,             {}),
    "HMHH-random":      (HMHHOptimizer,             {"allocation": "random"}),
    "PS-CMA-ES":        (PSCMAESOptimizer,          {}),
    "DEPSO":            (DEPSOOptimizer,            {}),
    "jSO":              (JSOPortOptimizer,          {}),
    "jSO-minionpy":     (JSOLibOptimizer,           {"backend": "minionpy"}),
    "L-SRTDE":          (LSRTDEPortOptimizer,       {}),
    "L-SRTDE-minionpy": (LSRTDELibOptimizer,        {}),
    "IMODE":            (IMODEPortOptimizer,        {}),
    "ELSHADE-SPACMA":   (ELSHADESPACMAOptimizer,    {}),
    "APGSK-IMODE":      (APGSKIMODEOptimizer,       {}),
    "SPS-L-SHADE-EIG":  (SPSLSHADEEIGOptimizer,     {}),
    "CoBiDE":           (CoBiDEOptimizer,           {}),
    "IMODE-mealpy":     (IMODEOptimizer,            {}),
    "LSHADE-cnEpSin":   (LSHADEcnEpSinOptimizer,    {}),
    "NGOpt":            (NGOptOptimizer,            {}),
    "NG-Portfolio":     (NGPortfolioOptimizer,      {}),
    "CMA-ES":       (CMAESOptimizer,                {}),
    "IPOP-CMA-ES":  (IPOPCMAESOptimizer,            {}),
    "BIPOP-CMA-ES": (BIPOPCMAESOptimizer,           {}),
    "PSO":          (PSOOptimizer,                  {}),
    "DE":           (DEOptimizer,                   {}),
    # Faithful numpy port (2026-10-06). The mealpy wrapper drew F/CR from the
    # unseeded global RNG and stalled on 10D F10 at ~1e3; kept for reference.
    "L-SHADE":        (LSHADEPortOptimizer,         {}),
    "L-SHADE-mealpy": (LSHADEOptimizer,             {}),
    "SaVOA":        (SaVOAOptimizer,                {}),
    # Multistart local-search floor: in 2D BBOB a restarted Nelder-Mead is a
    # strong reference — any metaheuristic gain must clear this bar.
    "NM-Restart":   (MultistartNelderMeadOptimizer, {}),
    # Dedicated niching baselines: the multi-solution comparison partners for
    # MC-ESO's sequential niching. They are in the registry for every suite but
    # only the niching suite runs them by default (_NICHING_METHODS below).
    "NCDE":         (NCDEOptimizer,                 {}),
    #  Plain crowding DE (Thomsen 2004) = NCDE with global donors; isolates
    #  what NCDE's neighbourhood mutation is worth.
    "Crowding-DE":  (NCDEOptimizer,                 {"m": 30}),
    #  Ring-topology lbest PSO (Li 2010): niching with no radius parameter.
    "r3pso":        (RingPSOOptimizer,              {}),
    #  Multi-swarm niching (Fieldsend 2014) through the published pynmmso.
    "NMMSO":        (NMMSOOptimizer,                {}),
    #  Quality-diversity (Mouret & Clune 2015), descriptor = first two coords.
    "MAP-Elites":   (MAPElitesOptimizer,            {}),
    #  Repelling restart CMA-ES (de Nobel+ 2024): the published analogue of
    #  MC-ESO's basin-memory spillover.
    "Repel-CMA-ES": (RepellingCMAESOptimizer,       {}),
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
    # Channel / router ablations (2026-10-05): which part carries the 2D lead.
    "abl_noAir":      (MultiChannelEpidemicOptimizer, {"air_ratio": 0.0}),
    "abl_noDroplet":  (MultiChannelEpidemicOptimizer, {"h2h_ratio": 0.0}),
    "abl_closeOnly":  (MultiChannelEpidemicOptimizer, {"air_ratio": 0.0, "h2h_ratio": 0.0}),
    "abl_noRouter":   (MultiChannelEpidemicOptimizer, {"channel_schedule": False}),
    #  close-contact isotropic: floor every normalised eigenvalue at 1 (no C_pop shape)
    "abl_isoClose":   (MultiChannelEpidemicOptimizer,
                       {"empirical_cov_floor": 1.0, "cov_floor_low": 1.0}),
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
    # Learned-C sample starvation (2026-09-29, その182): the rank-μ update is fed
    # only by close-contact children that beat their own parent (~1 sample per
    # generation at dim 20), which is the diagnosed cause of the ill-conditioned
    # functions staying at 0% at dim 5/10/20. These arms replace the
    # beat-your-parent test by CMA-ES's own rule — the best μ share of this
    # generation's placed close-contact children. Bit-identical to MC-ESO at
    # dim 2 (_cc_dim_gate() is exactly 0 there).
    "ccmu50":         (MultiChannelEpidemicOptimizer, {"cc_mu_frac": 0.50}),
    "ccmu25":         (MultiChannelEpidemicOptimizer, {"cc_mu_frac": 0.25}),
    "ccmu100":        (MultiChannelEpidemicOptimizer, {"cc_mu_frac": 1.00}),
    # High-dim arms (2026-09-29, local session). Traces at d10 separated the two
    # 0%-vs-100% functions into different faults: F12-BentCigar's learned C
    # grows to the needed cond ~1e6 only at the very end of the budget (learning
    # speed), F07-StepEllipsoidal freezes on a plateau with σ at the floor (a σ
    # rule fault, not a C fault).
    #   npop8        — n_pop = 8·dim (twice the children, hence C samples, per gen)
    #   npop8_ccmu50 — the same plus the top-half rank-μ selection
    #   ccmu50_lr10  — more samples per update, so afford twice the rate
    #   flat         — CMA-ES flat-fitness rule: ties to the best expand σ
    # All but `flat` are bit-identical to MC-ESO at dim 2.
    "npop8":          (MultiChannelEpidemicOptimizer, {"n_pop_dim_mult": 8.0}),
    "npop8_ccmu50":   (MultiChannelEpidemicOptimizer,
                       {"n_pop_dim_mult": 8.0, "cc_mu_frac": 0.50}),
    "ccmu50_lr10":    (MultiChannelEpidemicOptimizer,
                       {"cc_mu_frac": 0.50, "cc_learning_rate": 0.10}),
    "flat":           (MultiChannelEpidemicOptimizer, {"sigma_flat_expand": True}),
    # 2026-10-05 (local session). npop8 opened F07 but cost every easy function
    # half its generations; IPOP-style growth enlarges the population only on a
    # restart. On F07 d10 the restarts are ordinary spillovers (basin switches are
    # rare), so the growth is triggered by any spillover.
    "ipopS15":        (MultiChannelEpidemicOptimizer,
                       {"ipop_growth": 1.5, "ipop_trigger": "spillover", "ipop_max_mult": 4.0}),
    "ipopS15_ccmu50": (MultiChannelEpidemicOptimizer,
                       {"ipop_growth": 1.5, "ipop_trigger": "spillover", "ipop_max_mult": 4.0,
                        "cc_mu_frac": 0.50}),
    # Freeze the learned C for N generations after an ordinary spillover. On
    # F12 d10 the uniform reseed's first successes are isotropic and wash the
    # learned elongation out (cond stuck at 1e2-1e4; 1e5-1e6 without spillovers).
    "ccfrz25":        (MultiChannelEpidemicOptimizer, {"cc_spill_freeze_gens": 25}),
    "ccfrz50":        (MultiChannelEpidemicOptimizer, {"cc_spill_freeze_gens": 50}),
    # Provenance gate: learn C only from children of parents within
    # 2·σ·sqrt(dim) of the best in the learned metric (no timer).
    "ccgate2":        (MultiChannelEpidemicOptimizer, {"cc_gate_mahal": 2.0}),
    "ccgate3":        (MultiChannelEpidemicOptimizer, {"cc_gate_mahal": 3.0}),
    "ccgate2_ccmu50": (MultiChannelEpidemicOptimizer,
                       {"cc_gate_mahal": 2.0, "cc_mu_frac": 0.50}),
    # 2026-10-06 (local session). Cross-channel learning: droplet successes also
    # update the learned C (sample starvation: close-contact alone yields
    # 0.1-0.3 successes per generation at d10).
    "ccdrop":         (MultiChannelEpidemicOptimizer, {"cc_learn_droplet": True}),
    "ccdrop_gate2":   (MultiChannelEpidemicOptimizer,
                       {"cc_learn_droplet": True, "cc_gate_mahal": 2.0}),
    # Momentum channel: continue a lineage's last move (κ·dx, κ ~ U(1, 2)).
    "mom10":          (MultiChannelEpidemicOptimizer, {"mom_ratio": 0.10}),
    "mom10_noAir":    (MultiChannelEpidemicOptimizer,
                       {"mom_ratio": 0.10, "air_ratio": 0.0, "cc_air_ratio": 0.0}),
    "mom10_noAir_gate2": (MultiChannelEpidemicOptimizer,
                       {"mom_ratio": 0.10, "air_ratio": 0.0, "cc_air_ratio": 0.0,
                        "cc_gate_mahal": 2.0}),
    # Pre-2026-10-08 default (fixed population, no learned-C freeze, beat-parent
    # rule) — the reference every arm below was measured against. Arms defined
    # before 2026-10-08 set only the parameters they change, so they now run ON
    # TOP of the new default; their recorded numbers are relative to MC-ESO-v0.
    "MC-ESO-v0":      (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "fixed", "cc_spill_freeze_gens": 0, "cc_mu_frac": 0.0}),
    # Ablation of the 2026-10-08 default: remove one of its three changes.
    "v1_noPop":       (MultiChannelEpidemicOptimizer, {"pop_schedule": "fixed"}),
    # 2026-10-09 arms from the internal-state diagnosis (docs/history.md); all on
    # top of the current default, none changes 2D except v1_popfin8.
    "v1_rtC":         (MultiChannelEpidemicOptimizer, {"router_signal": "learned"}),
    "v1_cpath":       (MultiChannelEpidemicOptimizer, {"cc_path": True}),
    "v1_noairHD":     (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0}),
    "v1_h2h50HD":     (MultiChannelEpidemicOptimizer, {"cc_h2h_ratio": 0.5}),
    "v1_popfin8":     (MultiChannelEpidemicOptimizer, {"pop_final_mult": 8.0}),
    # round 2: does removing airborne stack with the router / the evolution path?
    "v1_noair_cpath": (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "cc_path": True}),
    "v1_noair_rtC":   (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "router_signal": "learned"}),
    # On top of v1_noairHD: stronger selection / smaller start from 3D up (2D unchanged).
    "v1_k50":         (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "hd_kill_fraction": 0.5}),
    "v1_k35":         (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "hd_kill_fraction": 0.35}),
    "v1_pi8":         (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "hd_pop_init_mult": 8.0}),
    "v1_k50_pi8":     (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "hd_kill_fraction": 0.5,
                                                       "hd_pop_init_mult": 8.0}),
    # jDE-style CR self-adaptation from 3D up, off the DROPLET route only.
    "v1_cr":          (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "hd_cr_heritable": True}),
    "v1_crR":         (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "hd_cr_heritable": True,
                                                       "router_signal": "learned"}),
    "v1_noair_all":   (MultiChannelEpidemicOptimizer, {"cc_air_ratio": 0.0, "cc_path": True,
                                                       "router_signal": "learned"}),
    "v1_noFrz":       (MultiChannelEpidemicOptimizer, {"cc_spill_freeze_gens": 0}),
    "v1_noMu":        (MultiChannelEpidemicOptimizer, {"cc_mu_frac": 0.0}),
    # Candidates on top of the 2026-10-08 default (v1_* = relative to new default).
    "v1_crmix":       (MultiChannelEpidemicOptimizer, {"h2h_cr_mix": (0.1, 0.9)}),
    "v1_crlow":       (MultiChannelEpidemicOptimizer, {"h2h_CR": 0.2}),
    "v1_h2hA":        (MultiChannelEpidemicOptimizer, {"h2h_adapt": True}),
    "v1_pop32":       (MultiChannelEpidemicOptimizer, {"pop_init_mult": 32.0}),
    "v1_gate2":       (MultiChannelEpidemicOptimizer, {"cc_gate_mahal": 2.0}),
    "v1_rcf30":       (MultiChannelEpidemicOptimizer, {"route_commit_frac": 0.30}),
    "v1_crher":       (MultiChannelEpidemicOptimizer, {"h2h_cr_heritable": True}),
    "v1_crher_rcf30": (MultiChannelEpidemicOptimizer,
                       {"h2h_cr_heritable": True, "route_commit_frac": 0.30}),
    # 2026-10-06 improvement candidates (all defaults unchanged).
    # A. population-size schedules
    "pop_lin16":      (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0}),
    "pop_lin8":       (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 8.0, "pop_final_mult": 4.0}),
    "pop_lin16_2":    (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 2.0}),
    # A2. front-loaded shrink (power > 1 hands the late phase more generations)
    "pop_pow2":       (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0,
                        "pop_shrink_power": 2.0}),
    "pop_pow3":       (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0,
                        "pop_shrink_power": 3.0}),
    "pop_pow2_2":     (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 2.0,
                        "pop_shrink_power": 2.0}),
    "pop_pow2_frzmu": (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0,
                        "pop_shrink_power": 2.0, "cc_spill_freeze_gens": 50, "cc_mu_frac": 0.50}),
    "pop_pow3_frzmu": (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0,
                        "pop_shrink_power": 3.0, "cc_spill_freeze_gens": 50, "cc_mu_frac": 0.50}),
    "pop_pow2_gate2": (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0,
                        "pop_shrink_power": 2.0, "cc_gate_mahal": 2.0}),
    "pop_lin16_gate2": (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0,
                        "cc_gate_mahal": 2.0}),
    "pop_lin16_frzmu": (MultiChannelEpidemicOptimizer,
                       {"pop_schedule": "linear", "pop_init_mult": 16.0, "pop_final_mult": 4.0,
                        "cc_spill_freeze_gens": 50, "cc_mu_frac": 0.50}),
    "ipop_fail3":     (MultiChannelEpidemicOptimizer,
                       {"ipop_growth": 1.5, "ipop_trigger": "failstreak",
                        "ipop_fail_streak": 3, "ipop_max_mult": 4.0}),
    # B. router v2: no airborne; per-route (droplet, momentum) shares
    "rv2":            (MultiChannelEpidemicOptimizer,
                       {"route_mix": {"pre": (0.4, 0.1), "droplet": (0.3, 0.2),
                                      "close": (0.3, 0.1), "keepair": (0.5, 0.0)}}),
    "rv2_flat":       (MultiChannelEpidemicOptimizer,
                       {"route_mix": {"pre": (0.4, 0.1), "droplet": (0.4, 0.1),
                                      "close": (0.4, 0.1), "keepair": (0.4, 0.1)}}),
    # C. droplet F / CR success-history adaptation
    "h2hA":           (MultiChannelEpidemicOptimizer, {"h2h_adapt": True}),
    # D. end-phase SLSQP local search
    "lsfin":          (MultiChannelEpidemicOptimizer,
                       {"ls_final_frac": 0.10, "ls_budget_frac": 0.05}),
    "ccfrz50_ccmu50": (MultiChannelEpidemicOptimizer,
                       {"cc_spill_freeze_gens": 50, "cc_mu_frac": 0.50}),
    # cc_mu_frac only in runs routed as ill-conditioned (droplet): at d5 the
    # ungated rule cost the Rastrigin family (その183).
    "ccmu50_drop":    (MultiChannelEpidemicOptimizer,
                       {"cc_mu_frac": 0.50, "cc_mu_droplet_only": True}),
    "ccmu100_drop":   (MultiChannelEpidemicOptimizer,
                       {"cc_mu_frac": 1.00, "cc_mu_droplet_only": True}),
    #  Pre-fix reference: reset the learned covariance on every spillover.
    #  Tracing F12-BentCigar showed that reset destroying a covariance that was
    #  on its way to the extreme elongation the function needs (effective rank
    #  1.00, condition 2.2e6 in the run that succeeds).
    "resetC":         (MultiChannelEpidemicOptimizer, {"cc_keep_on_spillover": False}),
    #  Pre-fix reference: the persistent stream off (pure C_pop close-contact).
    "split_off":      (MultiChannelEpidemicOptimizer, {"cc_learning_rate": 0.0}),
    # Reported-set ceiling sweep (2026-08-30). MC-ESO reports surviving hosts
    # (n_pop) plus the strain archive (n_elite_max), i.e. 23 points by default —
    # a hard cap on peak ratio when K = 36 or 216. n_elite_max also widens the
    # droplet channel's strain pool, so these are not reporting-only changes.
    "elite20":        (MultiChannelEpidemicOptimizer, {"n_elite_max": 20}),
    "pop50":          (MultiChannelEpidemicOptimizer, {"n_pop": 50}),
    "pop50_elite20":  (MultiChannelEpidemicOptimizer, {"n_pop": 50, "n_elite_max": 20}),
    #  Answer-archive control: pre-2026-08-31 reporting (population + strain
    #  reservoir only). Search is identical, so any SR difference is a bug.
    "arch_off":       (MultiChannelEpidemicOptimizer, {"solution_archive_max": 0}),
    #  Pre-2026-08-31 hunt pacing: every hunt drills back down to the σ floor.
    "hunt_off":       (MultiChannelEpidemicOptimizer, {"hunt_level_tol": 0.0,
                                                       "hunt_no_improve_mult": 0.0}),
    #  Niching lever under test (research_loop 問い 1c, 2026-09-03): the restart
    #  sigma taken to a ratio of the *locally observed basin spacing* instead of
    #  a fixed ratio of the box. Raises N06/N07 peak ratio at the deep accuracy
    #  levels; this entry exists only to check it does not cost BBOB-24 dim2
    #  SR@1e-10 (93.5% / evals_succ_mean 798) before it goes into mceso.py.
    "sigma_only":     (CommitReseedMCESO, {"commit_mode": "sigma_only",
                                           "commit_sigma_mode": "run",
                                           "commit_sigma_ratio": 0.1}),
    #  Niching lever under test (research_loop 問い 1, 2026-09-03): the phased
    #  acceptance rule — the shipped host competition while the first basin is
    #  drilled, nearest-neighbour crowding for every later hunt. The first phase
    #  is what has to hold SR@1e-10, so this entry checks the phase boundary is
    #  where it is claimed to be: BBOB-24 dim2 must not lose SR@1e-10.
    "phased":         (PhasedAcceptMCESO, {"accept_phase": "exhausted"}),
    #  Niching lever under test (research_loop 問い 1c, 2026-09-03): the basin
    #  release level. `hunt_level_tol` decides how deep an exhausted hunt has to
    #  get before the basin is abandoned; the default 1e-6 lands the release
    #  between eps=1e-3 and eps=1e-5 on Shubert (entry 30). This entry exists
    #  only to gate the lever on BBOB-24 dim2 SR@1e-10 before it goes anywhere
    #  near mceso.py. NOTE: the path only fires after `has_exhausted`, and entry
    #  29 proved the default 5000-eval gate is structurally blind to that — run
    #  this at a raised budget (--max-evals 20000), never at 5000.
    "level_t08":      (MultiChannelEpidemicOptimizer, {"hunt_level_tol": 1e-8}),
    #  The value entry 32 actually recommends for adoption: 1e-7 keeps 4/5 of the
    #  depth gain on N06 while costing nothing at the eps <= 1e-3 decision levels,
    #  so the gate has to clear this arm too, not only the more extreme 1e-8.
    "level_t07":      (MultiChannelEpidemicOptimizer, {"hunt_level_tol": 1e-7}),
    #  Adoption candidate under gate (research_loop キュー 2, 2026-09-12): the
    #  committed basin-switch restart, placement only (`commit_place_r010` in
    #  scripts/diagnose_niching.py). Five of the six stalled adoption candidates
    #  had already been through the BBOB-24 dim2 gate; this one never had, which
    #  is the single measurement 方針欄 2026-09-11 (3) asks for. Entry exists
    #  only to run that gate — mceso.py defaults are untouched either way.
    "commit_place_r010": (CommitReseedMCESO, {"commit_mode": "on",
                                              "commit_sigma_mode": "place",
                                              "commit_sigma_ratio": 0.1}),
}

# Every MC-ESO arm defined before the 2026-10-08 default change keeps meaning
# what it meant when it was measured: the parameters it does not set are filled
# with the pre-change default (MC-ESO-v0), so its recorded numbers reproduce.
# Arms of the new default are "MC-ESO" itself and the v1_* ablations.
_V0_DEFAULTS = {"pop_schedule": "fixed", "pop_shrink_power": 1.0,
                "cc_spill_freeze_gens": 0, "cc_mu_frac": 0.0}
for _name, (_cls, _kw) in list(_OPTIMIZERS.items()):
    if (isinstance(_cls, type) and issubclass(_cls, MultiChannelEpidemicOptimizer)
            and _name != "MC-ESO" and not _name.startswith("v1_")):
        _OPTIMIZERS[_name] = (_cls, {**_V0_DEFAULTS, **_kw})


# Methods that take a per-benchmark `sigma0` initial step.
_SIGMA_USERS = (CMAESOptimizer, IPOPCMAESOptimizer, BIPOPCMAESOptimizer,
                RepellingCMAESOptimizer)

# Result fields that only the renderers read. A --jobs worker drops them when
# rendering is off: an MC-ESO run at 10D pickles to ~12 MB, almost all of it
# population snapshots, and every run crosses a process boundary.
_VIZ_ONLY_FIELDS = ("history_pop", "history_pop_sigma", "history_sigma_global",
                    "history_n_elite", "history_no_improve", "history_eval_count",
                    "history_sigma_eval")


def _run_one(bench, method: str, n_runs: int, max_evals: int,
             noise: str | None) -> tuple[list, list]:
    """All runs of one method on one function (the unit of work for --jobs)."""
    cls, kwargs = _OPTIMIZERS[method]
    sigma0 = 0.2 * (bench.bounds[1] - bench.bounds[0])
    kw = {**kwargs, **({"sigma0": sigma0} if cls in _SIGMA_USERS else {})}
    return run_experiment(cls, bench, n_runs=n_runs, max_evals=max_evals,
                          noise_model=noise, **kw)


def _bench_key(bench) -> tuple[str, int, str]:
    """Benchmarks hold closures and do not pickle; workers look them up by name."""
    for kind, regs in (("bbob", _DIM_REGISTRIES.items()),
                       ("niching", [(0, NICHING_BENCHMARKS_BY_NAME)]),
                       ("cec2022", [(10, BENCHMARKS_CEC2022_10D_BY_NAME)])):
        for d, reg in regs:
            if reg.get(bench.name) is bench:
                return kind, d, bench.name
    raise ValueError(f"benchmark {bench.name!r} is not in a registry; --jobs cannot ship it")


def _bench_from_key(key: tuple[str, int, str]):
    kind, d, name = key
    if kind == "bbob":
        return _DIM_REGISTRIES[d][name]
    return (NICHING_BENCHMARKS_BY_NAME if kind == "niching"
            else BENCHMARKS_CEC2022_10D_BY_NAME)[name]


def _worker_task(key, method, n_runs, max_evals, noise, slim):
    import dataclasses
    results, times = _run_one(_bench_from_key(key), method, n_runs, max_evals, noise)
    if slim:
        results = [dataclasses.replace(
            r, history_x=np.asarray(r.history_x, dtype=float),
            **{k: [] for k in _VIZ_ONLY_FIELDS}) for r in results]
    return results, times


def _parallel_results(benchmarks: list, methods: list[str], n_runs: int,
                      max_evals: int, noise: str | None, jobs: int, slim: bool):
    """Yield (bench index, method, results, times) in the serial order.

    Every run is seeded by its index (core.runner), so splitting the work across
    processes gives the same numbers, except for methods that draw from an
    unseeded global RNG (the mealpy wrappers), which are not reproducible in any
    mode. Results are consumed in order and submission runs at most `window`
    tasks ahead, which bounds memory when one slow task holds up the queue.
    """
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(bi, m) for bi in range(len(benchmarks)) for m in methods]
    keys = [_bench_key(b) for b in benchmarks]
    window = max(4 * jobs, 32)
    pending: dict = {}
    nxt = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, (bi, m) in enumerate(tasks):
            while nxt < len(tasks) and len(pending) < window:
                b2, m2 = tasks[nxt]
                pending[nxt] = ex.submit(_worker_task, keys[b2], m2, n_runs,
                                         max_evals, noise, slim)
                nxt += 1
            results, times = pending.pop(i).result()
            yield bi, m, results, times


def _run_dim(benchmarks: list, dim_dir: Path, n_runs: int, max_evals: int,
             optimizers: dict | None = None, noise: str | None = None,
             no_viz: bool = False, jobs: int = 1) -> None:
    """Run all functions in a dimension group and save results to dim_dir."""
    dim_dir.mkdir(parents=True, exist_ok=True)
    if optimizers is None:
        optimizers = _OPTIMIZERS
    print(f"\n{'Function':<22} {'Method':<12} {'Mean':>12} "
          f"{'SR@1e-1':>7} {'SR@1e-2':>7} {'SR@1e-4':>7} {'SR@1e-7':>7} {'SR@1e-10':>8} {'EvalsSucc':>10}")
    print("-" * 102)
    methods = list(optimizers)
    if jobs > 1:
        stream = _parallel_results(benchmarks, methods, n_runs, max_evals, noise,
                                   jobs, slim=no_viz)
    else:
        stream = ((bi, m, *_run_one(b, m, n_runs, max_evals, noise))
                  for bi, b in enumerate(benchmarks) for m in methods)
    stream = iter(stream)
    for bench in benchmarks:
        results_per_method: dict = {}
        times_per_method: dict = {}
        for _ in methods:
            _, method, results, times = next(stream)
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
            # Niching suite: peak ratio over the reported solution set, scored
            # by the CEC2013 rules (see core.runner.niching_peak_metrics).
            npm = niching_peak_metrics(results, bench)
            if npm["n_optima"] > 0:
                mmo_str = (f"  | K={npm['n_optima']:>3} "
                           f"PR@1e-2={npm['cec_pr_1e-2']:>5.0%} "
                           f"PR@1e-4={npm['cec_pr_1e-4']:>5.0%} "
                           f"PRmean={npm['cec_pr_mean']:>5.0%} "
                           f"rep={npm['n_reported']:>4.0f}")
            elif pm["n_optima"] > 1:
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

        # Per-method visualizations. Rendering animates every evaluation, so at
        # large budgets it costs far more wall time (and disk) than the search
        # itself — --no-viz keeps the CSVs and drops the pictures.
        for method_name, results in ({} if no_viz else results_per_method).items():
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
        if not no_viz:
            save_landscape_svg(bench, output_dir=dim_dir)
            save_convergence_svg(bench, results_per_method, output_dir=dim_dir)
        save_stats(bench, results_per_method, times_per_method, output_dir=dim_dir)
        # Small data the results UI draws from (convergence, router / restarts);
        # always written, figures or not (core/run_data.py).
        save_curves(dim_dir, bench, results_per_method)
        append_mceso_runs(dim_dir, bench, results_per_method)
        _append_wilcoxon(dim_dir, bench.name, results_per_method, reference="MC-ESO")
        _append_wilcoxon_pr(dim_dir, bench, results_per_method, reference="MC-ESO")

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
    suite_budget: bool = False,
    no_viz: bool = False,
    jobs: int = 1,
) -> None:
    output_dir = Path(output_dir)
    # Results reproduce bit for bit on one machine, not across machines
    # (core/env_info.py): record where this run was computed.
    write_env(output_dir)
    if suite == "niching":
        registry = NICHING_BENCHMARKS_BY_NAME
        func_set = _NICHING_NAMES
    elif suite == "cec2022":
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
    # Default method set. BBOB / CEC2022 keep the historical line-up; the
    # niching suite swaps in the multi-solution baselines instead (running all
    # 14 methods everywhere would just make every BBOB judgement slower).
    if not methods and suite == "niching":
        methods = list(_NICHING_METHODS)
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

    dim_label = "per-function" if suite == "niching" else dim
    print(f"quick_check  suite={suite}  dim={dim_label}  n_runs={n_runs}  "
          f"max_evals={'suite' if suite_budget else max_evals}  "
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
            print(f"  skip {fname} (not in the {suite} dim{dim_label} registry)")
            continue
        benchmarks.append(registry[fname])

    if not benchmarks:
        raise SystemExit(f"No matching functions for filter {funcs} at dim={dim}")

    if suite == "niching":
        # One group per dimension in the suite. --suite-budget switches from the
        # project's flat budget to each function's competition MaxFEs
        # (5e4 for N04/N05, 2e5 for N06/N07/N10, 4e5 for N08/N09).
        for d in sorted({b.dim for b in benchmarks}):
            group = [b for b in benchmarks if b.dim == d]
            print(f"\n=== dim{d} ===")
            if suite_budget:
                for b in group:
                    _run_dim([b], output_dir / f"dim{d}", n_runs,
                             int(b.suite_max_evals), optimizers=optimizers,
                             noise=noise, no_viz=no_viz, jobs=jobs)
            else:
                _run_dim(group, output_dir / f"dim{d}", n_runs, max_evals,
                         optimizers=optimizers, noise=noise, no_viz=no_viz, jobs=jobs)
        return

    print(f"\n=== dim{dim} ===")
    _run_dim(benchmarks, output_dir / f"dim{dim}", n_runs, max_evals,
             optimizers=optimizers, noise=noise, no_viz=no_viz, jobs=jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",     type=int, default=20,                   help="Number of runs per method")
    parser.add_argument("--max-evals",  type=int, default=5000,                 help="Max function evaluations per run")
    parser.add_argument("--output-dir", type=Path, default=Path("results/quick"), help="Output directory")
    parser.add_argument("--funcs",      type=str, default=None,
                        help="Comma-separated function names to run (default: all in selected set). "
                             "Filters within the selected set: without --all that is quick-12.")
    parser.add_argument("--all",        action="store_true",
                        help="Use the full BBOB-24 set (F01-F24) at the selected --dim instead of "
                             "the quick-12 subset. BBOB-only — add --custom to also run the "
                             "C01-C11 custom benchmarks (2D only). Note: --funcs alone only filters "
                             "within the quick-12 subset; add --all to name functions outside it.")
    parser.add_argument("--custom",     action="store_true",
                        help="Also run the 2D-only custom benchmarks (C01-C11) — opt-in for "
                             "multimodal / multi-optima focus. Ignored for dim != 2.")
    parser.add_argument("--dim",        type=int, default=2, choices=sorted(_DIM_REGISTRIES),
                        help="BBOB problem dimension (default 2; one of 2/3/5/10/20). "
                             "Custom C01-C11 are 2D-only and skipped for higher dims. "
                             "For the CEC2022 hold-out use --suite cec2022 (forces dim=10).")
    parser.add_argument("--suite",      type=str, default="bbob",
                        choices=["bbob", "cec2022", "niching"],
                        help="Benchmark suite. 'bbob' (default) uses BBOB-24 + custom; "
                             "'cec2022' uses the 12-function CEC2022 hold-out at dim=10; "
                             "'niching' uses the CEC2013 niching 2D/3D subset (N04-N10), "
                             "which ignores --dim and runs each function at its own.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip landscape / convergence / animation rendering and "
                             "write only the CSVs. Rendering scales with the number of "
                             "evaluations, so large-budget runs need this.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Worker processes. Each (function, method) pair runs in "
                             "its own task; outputs are written in the serial order and "
                             "every seeded method gives the same numbers as --jobs 1.")
    parser.add_argument("--suite-budget", action="store_true",
                        help="Use each function's own competition budget instead of "
                             "--max-evals. Only meaningful with --suite niching "
                             "(MaxFEs 5e4 / 2e5 / 4e5).")
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
         methods=methods_list, with_custom=args.custom, noise=args.noise,
         suite_budget=args.suite_budget, no_viz=args.no_viz, jobs=max(1, args.jobs))
