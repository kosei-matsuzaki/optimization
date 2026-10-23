"""MC-ESO — Multi-Channel Epidemic Spread Optimizer (the proposed method)."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


@dataclass
class _MCESOState:
    """Loop-carried state for one MC-ESO run.

    Bundles the RNG, the search-domain constants, the population, the
    optimisation bookkeeping, and the per-eval/per-generation history buffers
    so the generation loop can be split into ``_maybe_spillover`` /
    ``_run_generation`` / ``_record_generation`` without threading two dozen
    locals through every call. Scalars are reassigned through ``state.x = ...``
    and arrays/lists are mutated in place, so the RNG draw order is identical
    to the original single-method implementation.
    """
    rng: "np.random.Generator"
    lo: float
    hi: float
    span: float
    max_evals: int
    sigma: float          # σ_global (mutable)
    sigma_init: float
    niche_radius: float
    pop_x: np.ndarray
    pop_f: np.ndarray
    pop_age: np.ndarray
    best_so_far: float
    f_init_scale: float
    no_improve: int = 0
    # Stagnation of the *current* basin: best f seen since the last spillover,
    # and evaluations since that improved. `no_improve` tracks the global best,
    # which stops moving once one optimum is drilled — after that it only
    # measures time, not whether the basin being searched is finished.
    basin_best: float = float("inf")
    has_exhausted: bool = False   # a basin has been drilled out at least once
    # Per-host step size, used only by niching variants (e.g. MC-ESO-Endemic) for
    # independent per-basin drilling; empty in base MC-ESO (single global σ).
    pop_sigma: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Adaptive close-contact anisotropy floor: EMA of log10 of the population
    # covariance's natural eigenvalue ratio. A *sustained* huge ratio means the
    # population is genuinely stretched along an ill-conditioned valley (relax the
    # floor); a small ratio with only transient spikes means rugged/multimodal
    # structure (keep the floor high so spurious anisotropy is clamped).
    cc_logratio_ema: "float | None" = None
    # Axis-alignment EMA (separability signal for the channel router): mean over
    # the population-covariance eigenvectors of their max |component|. ≈1 means
    # axis-aligned / separable (→ close-contact route); lower means rotated.
    cc_align_ema: "float | None" = None
    # Max-normalized-gap EMA (a second separability signal for the router): the
    # largest nearest-neighbour gap along any coordinate, normalized by the axis
    # range. Regular separable multimodals (F04/F16) leave wide coordinate gaps
    # (≈0.42) → close route; irregular/deceptive multimodals (F17/F20) pack
    # tighter (≈0.30) → keep-air. Separates the close vs keep-air classes that
    # axis-alignment alone cannot (F04 algA 0.975 ≈ F17 algA 0.974).
    cc_mgap_ema: "float | None" = None
    # Committed channel route ("droplet"/"close"/"keepair") for the router. Locked
    # once, after a warmup, from the stabilized EMAs — conditioning/separability
    # are run-invariant landscape properties, so committing removes the
    # generation-to-generation route flip-flop that perturbs threshold-borderline
    # functions off their best (median-signal) route.
    channel_route: "str | None" = None
    # Persistent close-contact covariance (rank-μ style). Starts at I and is
    # updated from the steps of infections that took hold, so it stays full rank
    # by construction — unlike C_pop, which is re-estimated every generation from
    # a population drawn from it and collapses to rank ≈ 2 of 10 in high dim.
    cc_C: "np.ndarray | None" = None
    # cc_chain: the transmission chain — the cumulated, σ-normalised drift of the
    # outbreak's centre (mean of its best hosts) — and the previous centre.
    cc_chain: "np.ndarray | None" = None
    cc_center: "np.ndarray | None" = None
    cc_cond_hist: list = field(default_factory=list)   # (evals, log10 cond of learned C)
    # switch_regrow: the population-size schedule restarts at a basin switch.
    sched_start: int = 0            # evaluations spent when the schedule started
    sched_n0: int = 0               # its initial size (0 = the run's initial size)
    # switch_pace: after a basin switch the new outbreak is paced on its own best.
    paced: bool = False
    med_hist: list = field(default_factory=list)   # spill_med_veto: (evals, log10 median f)
    dead_x: list = field(default_factory=list)     # h2h_archive: hosts displaced by children
    trace_spent: bool = False       # chain_trace: traced since the last meaningful improvement
    pace_ref: float = 0.0           # log10 of the outbreak's best at its last meaningful step
    pace_since: int = 0
    # The learned C has reached 10^router_learned_cond (latched; σ floor and
    # the containment hand-over read it).
    cc_ill: bool = False
    # hd_wave_lambda: a wave has run since the last meaningful improvement.
    wave_spent: bool = False
    # reassort_sweep: a sweep has run since the last meaningful improvement.
    sweep_spent: bool = False
    sweep_dead: bool = False        # a sweep found nothing: the strain died out
    handover_done: bool = False
    # Parent positions of this generation's offspring (same order as the
    # concatenated children) and the σ used, so the accepted steps can be
    # recovered for the update. Written only when the persistent covariance is on.
    gen_parent_x: list = field(default_factory=list)
    gen_sigma_used: list = field(default_factory=list)
    # How many of this generation's children came from close-contact — only they
    # are drawn from C, so only they may update it.
    gen_n_local: int = 0
    gen_parent_f: "np.ndarray | None" = None
    # len(history_f) when this generation's children started being evaluated —
    # lets σ adaptation see the generation's own f values (hd_sigma_success)
    # and the trace assign each evaluation its channel.
    gen_eval_start: int = 0
    # Generation index (len(history_sigma_global)) before which the learned C
    # is frozen after a spillover (cc_spill_freeze_gens).
    cc_freeze_until: int = 0
    log_best_ref: float = 0.0
    evals_since_reset: int = 0
    consecutive_failed_spillovers: int = 0
    last_n_elite: int = 0  # elite count of the current generation (for recording)
    history_x: list[np.ndarray] = field(default_factory=list)
    history_f: list[float] = field(default_factory=list)
    history_pop: list[np.ndarray] = field(default_factory=list)
    history_pop_sigma: list[np.ndarray] = field(default_factory=list)
    history_sigma_global: list[float] = field(default_factory=list)
    history_n_elite: list[int] = field(default_factory=list)
    history_no_improve: list[int] = field(default_factory=list)
    history_eval_count: list[int] = field(default_factory=list)
    history_sigma_eval: list[float] = field(default_factory=list)
    # Informed-restart state (reservoir re-ignition + basin-memory repulsion).
    # Persist across the whole run; harvested at each spillover.
    ir_archive_x: list[np.ndarray] = field(default_factory=list)   # niche-separated reservoir hosts
    ir_archive_f: list[float] = field(default_factory=list)
    # Answer archive: the best point of every basin the run drilled and then
    # abandoned. Separate from ir_archive (a re-ignition reservoir capped at
    # n_elite_max) and from basin_memory (centroids used for repulsion) — this
    # one exists so found optima survive into the reported solution set.
    sol_archive_x: list[np.ndarray] = field(default_factory=list)
    sol_archive_f: list[float] = field(default_factory=list)
    ir_basin_centroids: list[np.ndarray] = field(default_factory=list)  # abandoned-basin memory
    # Diagnostics for the results UI (recording only — nothing here is read by
    # the search, no RNG is drawn and no evaluation is spent to fill it).
    # eval_channel: one code per evaluation (TRACE_CHANNELS); trace_gen: one row
    # per generation (TRACE_GEN_KEYS); trace_events: (eval count, kind).
    eval_channel: bytearray = field(default_factory=bytearray)
    trace_gen: dict = field(default_factory=dict)
    trace_events: list = field(default_factory=list)
    gen_counts: tuple = (0, 0, 0, 0, 0)
    follow_done: bool = False     # wave_follow: one extension per run
    strain_owns: bool = False     # strain_handover: the strain has the whole budget
    probe_done: bool = False      # probe_at: the early CMA probe has run
    # strain_compete: a CMA strain co-circulating with the outbreak.
    strain_es: object = None
    strain_lam: int = 0
    strain_evals: int = 0
    strain_start: int = 0
    strain_rate: float = 0.0      # EMA of decades per evaluation, CMA strain
    outbreak_rate: float = 0.0    # EMA of decades per evaluation, outbreak
    record_eval: int = 0          # evaluation count at the last new global best
    strain_record_eval: int = -1  # ... set by the CMA strain
    outbreak_record_eval: int = 0 # ... set by anything else (the outbreak)

    @property
    def budget_left(self) -> bool:
        return len(self.history_f) < self.max_evals


# Channel that produced each evaluation (trace["eval_channel"]). "momentum" and
# "migratory" (codes 3 / 4, and n_mom / n_mig in the generation rows) belong to
# removed experimental channels: they are always 0 now, kept so that recorded
# traces and the results UI keep decoding with the same layout.
TRACE_CHANNELS = ("close", "droplet", "airborne", "momentum", "migratory",
                  "reseed", "init", "wave", "reassort")
# 7 was "local_search" (the end-phase SLSQP, removed 2026-10-12); since then it
# is the recombination wave (stall wave and containment hand-over). 8 is the
# reassortment sweep. Codes 3 / 4 (momentum / migratory) are never written.
_CH_RESEED, _CH_INIT, _CH_LS, _CH_SWEEP, _CH_UNKNOWN = 5, 6, 7, 8, 255
# Router route per generation (trace["gen"]["route"]); 0 = not committed yet.
TRACE_ROUTES = ("pending", "keepair", "droplet", "close")
TRACE_GEN_KEYS = ("evals", "n_pop", "sigma", "best_f", "n_elite", "no_improve",
                  "drilling", "route", "cond", "align", "mgap",
                  "n_close", "n_droplet", "n_air", "n_mom", "n_mig", "cc_cond")


class MultiChannelEpidemicOptimizer(BaseOptimizer):
    """MC-ESO — Multi-Channel Epidemic Spread Optimizer.

    A population-based black-box optimizer that models the search as the
    progression of an epidemic, in which infected hosts spread the pathogen
    through **multiple distinct transmission channels** running in parallel.

    Where conventional metaheuristics rely on a single reproduction operator
    (DE = differential mutation, ES = Gaussian mutation, PSO = velocity-driven
    move, GA = crossover), real epidemics propagate simultaneously via
    qualitatively different routes — close-contact, droplet, and airborne —
    each with a characteristic distance scale and directional structure.
    MC-ESO mirrors this by mixing three transmission channels per generation:

      • **Close-contact transmission** (local Gaussian, rotation-aware):
            x_child = x_parent + N(0, σ_i² · C_pop)
            Tight neighborhood spread; the noise is sampled from a Gaussian
            with the **instantaneous empirical covariance** C_pop of the
            population (eigenvalues mean-normalized to 1). This gives
            close-contact rotation- and anisotropy-awareness without any
            history accumulation (cf. CMA-ES rank-μ updates). σ_i further
            adapts to the host's relative fitness and age so well-adapted
            hosts probe finer.

      • **Droplet transmission** (host-to-host, DE/current-to-best/1):
            x_child = x_parent + F·(x_elite − x_parent) + F·(x_a − x_b)
            A productive (niched-elite) strain donates its "phenotype" to the
            parent, and a difference between two random hosts injects
            population-shape-aware drift, reinforcing the same anisotropy
            signal that close-contact picks up via C_pop.

      • **Airborne transmission** (population-independent spread):
            x_child = x_random_host + N(0, σ_air I)
            Long-range aerosol spread for escaping local maxima; σ_air
            inflates as the outbreak clusters. **Suppressed in drilling
            mode** (σ < span × precision_sigma_ratio) so precision grinding
            isn't disrupted by random long jumps.

    These channels are complemented by three population-level mechanisms:

      • **Strain coexistence** (niched elite pool): spatially separated
        productive lineages are maintained as transmission seeds for the
        droplet channel, preserving multi-basin coverage.

      • **Host competition** (greedy μ+λ with rollback): each generation the
        worst ``kill_fraction`` of hosts die and are replaced by offspring;
        children that fail to outcompete the host they replaced are rolled
        back. The outbreak monotonically improves.

      • **Spillover event** (stagnation-triggered *informed* restart): when the
        outbreak stalls (no improvement for ``restart_no_improve_threshold``
        evals) and ``best_so_far`` still exceeds
        ``restart_quality_rel_floor × |f_init|`` (i.e. the run has not yet
        reduced the initial-population best by ~8 orders of magnitude), the
        population spills over to a fresh host pool. Unlike a blind uniform
        restart, the re-seed **reuses the search structure**: the current
        basin's niched elites are harvested into a persistent strain archive
        and its centroid is remembered, then a fraction ``ir_archive_frac`` of
        the new hosts re-ignite as a tight Gaussian around a surviving archived
        reservoir while the rest are uniform draws repelled away from every
        remembered basin (herd immunity → explore susceptible regions). After a
        streak of failed spillovers the next event escalates to a full basin
        switch (best discarded, σ reset to σ_init). Quality floors are relative
        to the initial-population best so they remain meaningful under
        multiplicative rescaling of f. (Closes the "uninformed restart" gap
        found by the 2026-06 ablation; the genuine differentiation from IPOP's
        blind restart.)

    Step-size adaptation is always on: σ is multiplied by ``sigma_up`` on
    improvement and ``sigma_down`` on stagnation. Once σ falls below
    ``span × precision_sigma_ratio`` (i.e. the search has localised inside a
    basin) the contraction switches to the stronger ``sigma_drill_down`` to
    drill toward the floating-point optimum.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        # ── Population / niching ────────────────────────────────────────
        # None → with pop_schedule="linear" (default) the schedule's initial size
        # max(20, pop_init_mult·dim); with "fixed", max(20, 4·dim): 20 up to dim 5, then
        # scales up (e.g. 40 at dim 10). A fixed 20 underfills the population in
        # higher dimensions — at dim 10 it let niching restarts wander badly on
        # CEC2022 G06-Hybrid1 (best_f 2140 → 40 once n_pop reached 40). Low-dim
        # (BBOB dim 2/3) is unchanged. Pass an int to override.
        n_pop: "int | None" = None,
        # Slope of the dimension-aware default: n_pop = max(20, this · dim).
        # 4 is the historical default. The 2026-06 sweep that chose it predates
        # the learned covariance, whose sample supply scales with the children
        # per generation (kill_fraction · n_pop) — hence re-measured as an arm.
        # Any value ≤ 10 leaves dim 2 at n_pop = 20 (bit-identical there).
        n_pop_dim_mult: float = 4.0,
        # Population-size schedule. "linear" (default since 2026-10-08) starts at
        # max(20, pop_init_mult·D) and shrinks with the budget spent to
        # max(pop_min, pop_final_mult·D), removing the worst hosts (L-SHADE's
        # LPSR applied to the host pool). A larger pool opened F07 at d10 but cost
        # easy functions half their generations; shrinking keeps the early breadth
        # and gives the late phase back its generations.
        # Default since 2026-10-08 (adopted from the pop_pow2_frzmu arm): shrink.
        pop_schedule: str = "linear",
        pop_init_mult: float = 16.0,
        pop_final_mult: float = 4.0,
        pop_min: int = 10,
        # Shape of the shrink: size = final + (init − final)·(1 − t)^power, t =
        # fraction of the budget spent. 1 = linear. Larger values shrink early and
        # hand the late phase more generations — at d10 the linear 16·D → 4·D
        # schedule lost 5pt (F08 / F09 / F14) while it gained 15.8pt at d5.
        pop_shrink_power: float = 2.0,
        n_elite_max: int = 6,
        niche_radius_ratio: float = 0.1,       # min mutual elite distance, × span
                                               # (scale-invariant; on BBOB span=10
                                               # → 1.0, identical to the previous
                                               # absolute niche_radius=1.0)
        # ── σ (global) ────────────────────────────────────────────────
        sigma: float = 0.2,                    # σ_init relative to span
        host_sigma_min_scale: float = 0.05,    # per-host σ_i scaling floor — σ_i is
                                               # σ_global × host_sigma_min_scale ** (lq · (0.7+0.3·ar))
                                               # i.e. high-quality / old hosts probe finer
        # ── Infection modes ────────────────────────────────────────────
        air_ratio: float = 0.3,                # share of children = pure-random "air"
        h2h_ratio: float = 0.4,                # share of children = host-to-host (DE-style)
        h2h_F: float = 0.5,                    # h2h differential scale
        # Airborne σ inflates linearly with population convergence:
        #   factor = 1.5 + air_sigma_amplifier × (1 - diversity_ratio)
        # i.e. factor = 1.5 at full diversity, 1.5+amplifier at full convergence.
        # Default 3.5 reproduces the prior min=1.5, max=5.0 behaviour.
        air_sigma_amplifier: float = 3.5,
        # ── Droplet difference structure (route-gated best2, 2026-07) ────
        # Swaps the *math inside* the droplet channel without adding a 4th
        # channel; at its default the run is bit-identical to base off the
        # droplet route. A 2026-07 sweep of heavier-tailed / oppositional
        # airborne and Cauchy close-contact / rand1 / cur2pbest / global-best2
        # droplet kernels was rejected (all regressed overall SR@1e-10); only
        # the route-gated best2 below generalised. Details in docs/history.md.
        #   droplet_variant: host-to-host DE difference structure.
        #     "best2_droplet" (default) — current-to-best/2 on the committed
        #                     DROPLET route, else current-to-best/1. The 2nd
        #                     difference vector rescues stalled ill-conditioned runs
        #                     (F13/F14 at dim2; F02/F10/F11/F12/F13/F14 all large at
        #                     dim3, +12.9pt overall) while off-route multimodals stay
        #                     bit-identical to base. On droplet only (route-gated):
        #                     a *global* 2nd vector regresses keep-air multimodals.
        #     "cur2best"    — x_p + F(x_strain−x_p) + F(x_a−x_b) everywhere
        #                     (single-difference; pre-2026-07 behaviour, regression
        #                     reference).
        droplet_variant: str = "best2_droplet",
        # Per-landscape channel router. When True, each generation is classified
        # from two f-independent, scale-invariant covariance signals and the
        # airborne budget is routed to the channel that landscape needs — instead
        # of a uniform reallocation (all four uniform variants lost SR@1e-10, 2026-
        # 07; see docs/history.md "チャネル割合スケジューリングの探索"):
        #   • conditioning cond = log10(λmax/λmin), EMA  — cond > cond_droplet_thresh
        #     ⇒ ill-conditioned valley ⇒ DROPLET route (air tapered → droplet).
        #   • axis-alignment algA (mean max|eigvec comp|), EMA — else algA >
        #     align_close_thresh ⇒ separable/axis-aligned ⇒ CLOSE route (air → close).
        #   • otherwise ⇒ KEEP-AIR route ⇒ base ratios (multimodal escape untouched).
        # The default route is KEEP-AIR = base, so any function that doesn't clearly
        # signal droplet/close stays bit-identical to base (bounded risk). The
        # taper reuses the validated σ-ramp, so a routed function starts at base at
        # full σ and only diverges as it drills. Deterministic (no reward bandit,
        # cf. rejected MC-ESO-V2a). On by default: verified +0.6pt overall SR@1e-10
        # on BBOB dim2 (87.9→88.4) and dim3 (43.5→44.2, no significant regression)
        # plus best_f gains on the CEC2022 dim10 hold-out (G06 364→202). Set False
        # to recover the exact pre-router flat-ratio behaviour.
        channel_schedule: bool = True,
        # Route thresholds are applied to the EMA *values* at the commit
        # checkpoint (measured: droplet cond@120 ≥ 3.3, non-droplet ≤ 2.66; close
        # algA@120 ≥ 0.975, keep-air ≤ 0.903 — clean gaps).
        cond_droplet_thresh: float = 3.0,      # cond@commit above → droplet route
        # Close route needs BOTH high axis-alignment AND a wide coordinate gap:
        # axis-alignment alone can't separate separable multimodals that want
        # close (F04 algA 0.975, mgap 0.41) from deceptive ones that need air
        # (F17 algA 0.974, mgap 0.29). Requiring mgap too routes F17/F20 to
        # keep-air while keeping F04/F16 on close.
        align_close_thresh: float = 0.965,     # algA@commit above … (with mgap) → close
        close_mgap_thresh: float = 0.36,       # … AND mgap@commit above → close
                                               # (between F20≈0.31 and F04≈0.41)
        # Early droplet latch: a genuinely ill-conditioned valley drives cond high
        # *fast* (F11/F12/F13 to 8-11 by ~gen 20-30, F14 by ~gen 85) — routing it
        # to droplet early captures the gain (F13 SharpRidge +30 needs droplet from
        # the start). Rugged multimodals (F16/F23) only spike cond transiently to
        # ~3, so a HIGH bar avoids latching them. Above this at any pre-commit gen
        # ⇒ latch droplet immediately.
        cond_droplet_early: float = 4.0,
        # Route-commit checkpoint (generation). Before it, run base keep-air; at it,
        # commit the route from the stabilized EMA values and lock for the run. Set
        # inside the exploration phase (functions have ~260-470 explore gens) and
        # late enough that the conditioning EMA has developed (droplet functions
        # reach cond ~3-7 by here, others stay ≤ ~2.7).
        route_commit_gen: int = 120,
        # ── Greedy (μ+λ) replacement ───────────────────────────────────
        kill_fraction: float = 0.25,           # fraction of active killed per gen (by f)
        # σ floor once the learned C is ill-conditioned (≥3D; None = always
        # sigma_floor_ratio). A floor of 1e-10 lifted 2D F23 75→100 / F18 95→100
        # and 5D F23 10→60, but 5D F13 (sharp ridge) fell 100→70: σ collapses to
        # 1e-9 at gap 5e-10 and the ridge stalls, whereas the 1e-6 floor keeps σ
        # large enough to travel along the learned C's long axis (2026-10-10).
        sigma_floor_ill_ratio: "float | None" = 1e-6,  # default since 2026-10-12 (v1_w4q3E)
        # σ success rule from dimension 3 up (None = "the best improved this
        # generation", the 2D rule). With ~40 births per generation at 10D the
        # chance that one of them improves the best stays high, so σ hovers at
        # 0.02–0.1 while 10D F12 gains one decade per ~8500 evaluations
        # (2026-10-10 trace). Here σ grows when more than this fraction of the
        # close-contact children beat their own parent (a 1/5-type rule).
        hd_sigma_success: "float | None" = 0.05,  # default since 2026-10-12 (v1_w4q3E)
        # Recombination wave on a stall (≥3D; None = off). When a spillover is
        # about to fire on a basin that is not drilled out, and no wave has run
        # since the last meaningful improvement, run a non-elitist weighted-
        # recombination wave instead: (μ/μ_w, λ) sampling with λ = this·D around
        # the best host, σ0 = wave_sigma_ratio·span, until it stops itself or
        # spends wave_budget_frac of the budget; its best replaces the worst
        # host. 2026-10-11: from MC-ESO's stuck points on 5D F17 / F18, CMA-ES
        # with λ = 8 fails but λ = 40 (8·D) reaches 1e-10 in 8/8 and 7/8 runs,
        # while the spillover re-drills the same point ~7 times per run.
        hd_wave_lambda: "float | None" = 4.0,  # default since 2026-10-12 (v1_w4q3E)
        wave_sigma_ratio: float = 0.01,
        wave_budget_frac: float = 0.25,
        # Run the wave only once best_so_far ≤ this × |f_init| (None = always).
        # Gallagher (F21 / F22) stalls at local optima with f ≈ 1e-2·|f_init| and
        # needs the spillovers the wave replaces; Schaffer stalls ~1e-6·|f_init|.
        wave_quality: "float | None" = 1e-3,  # default since 2026-10-12 (v1_w4q3E)
        # Whiten the wave by the learned C when it is ill-conditioned (the wave
        # then starts from the shape the outbreak has learned instead of the
        # identity; 10D F12 spent ~3000 evaluations re-learning it).
        wave_use_C: bool = True,  # default since 2026-10-12 (v1_w4q3E)
        # Containment hand-over (≥3D, needs hd_wave_lambda): once the learned C
        # is ill-conditioned and σ has entered the drilling regime, run one wave
        # (whitened, σ0 = current σ) without waiting for a stall.
        wave_ill_handover: bool = True,  # default since 2026-10-12 (v1_w4q3E)
        # Hand over as soon as the learned C is ill-conditioned (False = wait for
        # σ to enter the drilling regime as well).
        wave_handover_drill: bool = False,
        # Population of the containment hand-over wave: None = the same λ as the
        # stall wave (hd_wave_lambda·D); 0 = CMA-ES's default 4 + ⌊3 ln D⌋;
        # a negative −k = k × that default.
        # 10D F12: every failing run spent its ~18800 hand-over evaluations and
        # stopped at 1e-8..1e-9 with λ = 40; a single valley wants a small λ.
        wave_handover_lambda: "float | None" = -2.0,  # default since 2026-10-13 (v2_scan2d); None until then
        # IPOP-style restarts inside the hand-over: when the wave stops itself
        # with budget left, restart it from its best point with λ doubled.
        # A small λ suits a single valley (10D F12 50→100 at λ = 2·D) and a
        # large one the Schaffer ripples (5D F18 75→55 at λ = 2·D).
        wave_handover_restart: bool = False,  # default since 2026-10-12 (v1_w4q3E)
        # ── Restart on stagnation ──────────────────────────────────────
        restart_no_improve_threshold: int = 300,  # no_improve count that triggers restart
        # ``no_improve`` counts EVALUATIONS, but stagnation is a *generational*
        # signal and a generation costs kill_fraction·n_pop evals — 5 at dim 2,
        # 10 at dim 10, 20 at dim 20. A fixed eval window therefore shrinks in
        # generations exactly where more generations are needed, and the spillover
        # degenerates into a limit cycle: σ is reset to σ_init·restart_sigma_ratio
        # every window, so it can never contract past
        # σ_init·restart_sigma_ratio·sigma_down^(window/children) and drilling mode
        # is never reached (measured at dim 10: 46-65 spillovers/run, σ bottoming at
        # 1.3e-2·span, drilling 0% of generations). The window is therefore scaled
        # by (dim / 2) ** restart_window_dim_scale — 1.0 at dim 2 for ANY exponent,
        # so the low-dimensional behaviour is bit-identical; 0.0 disables scaling.
        restart_window_dim_scale: float = 1.0,
        # ── Dimension-invariance flags (audit 2026-08-24) ─────────────────────
        # Every one defaults to the pre-audit behaviour, and each is a no-op at
        # dim 2 by construction, so dim-2 judgement runs stay bit-identical.
        #
        # softmax_beta: parent ("infectivity") selection weight. The original
        #   form exp(f_max − f_i) uses RAW f differences, so once the population
        #   converges and the f spread falls below 1 the weights flatten to
        #   uniform — the measured effective parent count was 20.0 of 20 at
        #   dim 2, i.e. the selection was inert on every problem. The default
        #   uses the range-normalised, scale-invariant form
        #   w ∝ exp(−beta · (f_i − f_min)/(f_max − f_min)), which gives the same
        #   pressure whatever the f scale or the dimension. Verified at every
        #   dimension (SR@1e-10 vs the raw form): dim2 +0.21 / dim3 +6.67 /
        #   dim5 +8.12 / dim10 +3.12 / dim20 +6.87. beta = 0 restores the raw,
        #   scale-dependent form (regression pin). beta = 8 was also measured and
        #   rejected: better in high dim but −2.29 at dim 2.
        softmax_beta: float = 5.0,
        restart_sigma_ratio: float = 0.3,      # σ after restart, relative to σ_init
        # Quality floors are RELATIVE to the initial-population best
        # f_init_scale = max(|best_of_initial_population|, ε). On problems
        # normalised so the optimum is at 0 (BBOB/CEC2022 after our f - f_opt
        # transform) the relative gate is equivalent to a fixed absolute
        # threshold when f_init is large; under multiplicative rescaling of f
        # the gate adapts automatically. Shift invariance (f_opt unknown) is
        # not achievable from f alone and is documented as a limitation.
        restart_quality_rel_floor: float = 1e-8,     # skip restart if
                                                     # best_so_far / |f_init| ≤ this
        # Spillover: on every restart, all non-best slots are re-seeded uniformly
        # across the search domain. Best is preserved unless the streak of failed
        # spillovers triggers a full basin switch below.
        basin_switch_after_failed_spillovers: int = 2,  # streak → wipe best & reset σ
        basin_switch_quality_rel_floor: float = 1e-2,   # basin switch suppressed when
                                                        # best_so_far / |f_init| ≤ this
        solution_archive_max: int = 200,                # answer-archive capacity; 0 = off
        # ── Multi-solution (sequential niching) ───────────────────────
        # Once a basin is drilled to the algorithm's resolution limit, restart
        # repelled away from it to discover further optima (raises peak ratio on
        # multi-global functions). "Drilled out" is detected scale-/shift-free:
        # σ has bottomed at its floor AND the run has stagnated — no reference to
        # the (unknown) optimum value. SR@1e-10 is preserved: a basin is only left
        # once base could drill no deeper there. Set exhausted_no_improve_mult to
        # a very large value to disable niching (pure single-basin MC-ESO).
        exhausted_sigma_tol: float = 1.5,       # σ ≤ this × σ-floor ⇒ bottomed out
        exhausted_no_improve_mult: float = 3.0, # stagnation (× restart threshold) at the floor
        hunt_no_improve_mult: float = 0.5,      # stagnation multiplier for hunts after
                                                # the first exhaustion (0 = reuse
                                                # exhausted_no_improve_mult)
        hunt_level_tol: float = 1e-6,           # a later hunt ends once its basin reaches
                                                # this × |f_init| — as good as the optimum
                                                # already banked. 0 = off (pre-2026-08-31)
        # ── σ adaptation ──────────────────────────────────────────────
        # Multiplicative step-size adaptation. Once σ < span × precision_sigma_ratio
        # the contraction switches to sigma_drill_down to drill to FP precision.
        sigma_up: float = 1.1,                 # gentle expansion when improving
        sigma_down: float = 0.95,              # gentle contraction when not
        sigma_floor_ratio: float = 1e-10,      # σ_global absolute floor (× span); 1e-6 until 2026-10-12
        sigma_ceil_ratio: float = 1.0,         # σ_global absolute ceiling (× span)
        precision_sigma_ratio: float = 1e-3,   # σ < span × this → drilling mode
                                               # (scale-invariant; replaces absolute
                                               # drilling_threshold)
        sigma_drill_down: float = 0.85,        # σ contraction in drilling mode
        # ── Misc ───────────────────────────────────────────────────────
        log_slope_threshold: float = 1e-4,     # min log10(f) slope counted as improvement
        # ── h2h binomial crossover (always on) ────────────────────────
        # The droplet child is built as DE/current-to-best/1, then a binomial
        # crossover with the parent gates each coordinate (rate h2h_CR). This
        # preserves coordinate-aligned structure on separable multimodals
        # (F04 Büche-Rastrigin SR 77→100%, F17 Schaffer F7 47→73% at n=30).
        h2h_CR: float = 0.9,                   # h2h binomial crossover rate (DE/bin standard)
        # ── Rotation-aware close-contact (empirical covariance) ──────
        # Close-contact noise is drawn from N(0, σ_i²·C_pop) where C_pop is
        # the *instantaneous* empirical covariance of the population — no
        # history accumulation (cf. CMA-ES). Mean eigenvalue is normalized
        # to 1 so total step magnitude is preserved; floor prevents collapsed
        # axes. Closes the F11/F14 ill-conditioned gap to DE/CMA-ES (F11
        # mean 5.2e-8 → 0 at n=15, F14 SR_1e-7 80% → 87%).
        empirical_cov_floor: float = 0.01,     # min normalized eigenvalue (rugged/explore)
        # Adaptive anisotropy floor. The effective floor is interpolated (in log
        # space) between ``empirical_cov_floor`` (high → anisotropy capped ~14:1,
        # safe on rugged/multimodal) and ``cov_floor_low`` (low → anisotropy up to
        # ~1000:1, needed for ill-conditioned valleys) by a **smoothed natural
        # eigenvalue ratio** of the population covariance. Measured medians cleanly
        # separate the two regimes: ill-conditioned ≈ 1e5–1e7, rugged/multimodal
        # ≈ 3–600. When the smoothed ratio is ≥ ``cov_ratio_hi`` the floor relaxes
        # to ``cov_floor_low``; ≤ ``cov_ratio_lo`` it stays at the safe high floor.
        # The signal (a ratio of eigenvalues) is scale- and shift-invariant — no f
        # values. Set ``cov_floor_low = empirical_cov_floor`` to disable.
        cov_floor_low: float = 1e-3,
        cov_ratio_lo: float = 1e3,             # natural ratio at/below → high floor
        cov_ratio_hi: float = 3e4,             # natural ratio at/above → low floor
        cov_ratio_beta: float = 0.1,           # EMA rate (rejects rugged spikes)
        # ── Persistent close-contact covariance (rank-μ) ─────────────────────
        # C_pop is re-estimated every generation from a population that was drawn
        # from it, so the estimate→sample→estimate loop closes and the effective
        # rank collapses (measured 1.98 of 10 at dim 10, 5.74 of 20 at dim 20, on
        # every function — including F01-Sphere, whose true conditioning is 1.0
        # yet which is sampled at an effective conditioning of 4.2e3). Every
        # remedy that left this loop intact failed, and the measurements say the
        # binding constraint is the quality of the search directions, not the step
        # size. CMA-ES escapes the loop only because its distribution is a
        # PERSISTENT object with a learning rate.
        #
        # cc_learning_rate > 0 gives close-contact its own covariance C, started
        # at the identity and updated from the steps of infections that took hold:
        #     C ← (1−c)·C + c·mean(y yᵀ),  y = (x_child − x_parent)/σ_used
        # With a small c the identity component decays slowly, so C stays full
        # rank by construction while accumulating the valley geometry. Note this
        # is NOT the earlier cc_accum_rate attempt, which EMA-ed the population
        # *position* covariance and therefore inherited the collapse.
        #
        # C is reset to the identity on spillover / basin switch, so a basin
        # change re-adapts instantly — the property CMA-ES gives up by carrying
        # its covariance and evolution path across restarts. Cost is unchanged:
        # base already eigendecomposes a d×d matrix every generation.
        # 0.0 = off (pure C_pop close-contact), which is the pre-2026-08 behaviour.
        # The whole persistent-covariance apparatus is a HIGH-dimensional
        # remedy: at dim 2 the population's effective rank is 1.5 of 2, there is
        # no collapse to repair, and the raised close-contact share needed to
        # feed the learner costs the droplet/airborne escape that the tuned
        # dim-2 behaviour depends on (measured: SR@1e-10 93.54 → 87.92, with F23
        # 70→35, F18 100→60, F13 100→75). So it is switched on by dimension:
        # (dim/2 − 1) / (cc_dim_ref/2 − 1) clipped to [0, 1], which is exactly 0
        # at dim 2 — bit-identical — and 1 at cc_dim_ref and above.
        cc_dim_ref: int = 10,
        cc_learning_rate: float = 0.05,
        # Adoption rule for the rank-μ update (2026-09-29, arm only — default 0.0
        # keeps the historical rule bit-for-bit). The learned C is currently fed
        # only by close-contact children that BEAT THEIR OWN PARENT, which at
        # dim 20 leaves ~1 sample per generation (measured 0.1-0.3 successful
        # steps/generation) — far below what a rank-μ estimate of a dim×dim shape
        # needs, and the diagnosed cause of the ill-conditioned functions staying
        # at 0% (docs/history.md 2026-08-29). CMA-ES does not use a
        # beat-your-parent test at all: it takes the best μ of the λ offspring
        # unconditionally. > 0 replaces the parent test by exactly that — the
        # best `cc_mu_frac` share of THIS generation's placed close-contact
        # children, ranked by f — which raises the sample count instead of
        # squeezing more out of the same one or two samples (the four rejected
        # attempts, docs/history.md:865 / :979 / :1041, all did the latter).
        cc_mu_frac: float = 0.5,               # default since 2026-10-08 (was 0.0)
        # Generations after an ordinary spillover during which the learned C is
        # not updated (0 = off). On F12-BentCigar d10 the learned C reaches cond
        # 1e5-1e6 when spillovers are disabled, but with them (33 per run) it
        # oscillates at 1e2-1e4: the steps that succeed right after a uniform
        # reseed come from far-off, high-f hosts and say nothing about the basin.
        cc_spill_freeze_gens: int = 0,         # 0 since 2026-10-13 (v2_scan2d); 50 from 2026-10-08, 0 before
        # Keep the learned covariance across ordinary spillovers (reset only on a
        # full basin switch). See the note at the reset site.
        cc_keep_on_spillover: bool = True,
        # "Ill-conditioned" threshold for the learned covariance: the latch
        # (_update_cc_ill) fires once its log10 condition number reaches this,
        # and sigma_floor_ill_ratio / the containment hand-over read the latch.
        router_learned_cond: float = 3.0,
        # How much of the sampling shape comes from the persistent C rather than
        # the instantaneous C_pop. The two are complementary, not alternatives:
        # C_pop is right immediately wherever the population can align with the
        # landscape (F02-EllipsoidalSep and F05 are solved 100% by base and are
        # destroyed — 100% → 0% — if C_pop is simply replaced), while the learned
        # C keeps full rank where the population collapses (F06 0→100, F08 10→50,
        # F09 0→40). Blending keeps the instantaneous covariance that
        # distinguishes MC-ESO from CMA-ES and adds persistence only as a
        # supplement. 0 = pure C_pop (base), 1 = pure learned C.
        #
        # Blending the two MATRICES does not work: normalising C_pop to mean
        # eigenvalue 1 leaves its smallest eigenvalue near 1e-6 on an
        # ill-conditioned valley, and adding w × (a near-isotropic C) lifts it to
        # about w/dim, capping the achievable anisotropy at ~dim/w — 33x at
        # w = 0.3, where F02 needs 1e6. That is the same trap the rejected
        # cov_shrink fell into. Instead SPLIT the close-contact offspring: a
        # cc_persist_frac share is drawn from the learned C and the rest from
        # C_pop, and host competition decides which shape was right. No matrix is
        # blended, so neither shape loses its anisotropy, and no online
        # discriminator is needed — which is exactly what having several
        # transmission routes is for.
        cc_persist_frac: float = 0.5,
        cc_air_ratio: float = 0.0,             # airborne share once the gate is open; 0.10 until 2026-10-12
        # 0.25, not 0.20: the droplet channel's binomial crossover is what
        # protects coordinate-wise structure, and starving it costs
        # F02-EllipsoidalSep at dim 20 (100% → 15%, a12 = 1.00). 0.25 restores
        # F02 (median f 1.1e-9 → 4.3e-12, past the 1e-10 bar) while giving up
        # nothing at dim 10 (+12.08pt either way); 0.30 costs 3.3pt at dim 10.
        cc_h2h_ratio: float = 0.25,            # droplet share once the gate is open
        # Anisotropy floor for the LEARNED covariance. C_pop needs a floor
        # because rank deficiency gives it exact zero eigenvalues; the learned C
        # is built up from the identity and stays full rank (measured effective
        # rank 9.9 of 10), so it needs almost none — and cov_floor_low (1e-3)
        # caps its eigenvalue ratio at ~1e4 in variance, i.e. ~100:1 in σ, two
        # orders short of the 1e6 conditioning of F10/F11/F12/F14. Those four
        # functions are 0% for MC-ESO and 100% for every CMA-ES variant at dim
        # 5/10/20, and they account for most of the remaining gap.
        cc_cov_floor: float = 1e-11,
        # ── Informed restart (reservoir re-ignition + herd-immunity repulsion) ─
        # The spillover re-seed is *informed*, not a blind uniform draw: a
        # persistent niche-separated strain archive is harvested at every
        # spillover and a fraction `ir_archive_frac` of the re-seeded slots are
        # drawn as a tight Gaussian (σ = `ir_reignite_sigma_ratio` × span) around
        # a surviving reservoir host; the remaining uniform draws are rejection-
        # sampled (≤ `ir_repel_max_tries` tries) to avoid an `ir_repel_radius_ratio`
        # × span ball around every abandoned-basin centroid (herd immunity →
        # explore susceptible regions). The 2026-06 ablation showed the prior
        # blind uniform restart discarded all search structure except the single
        # best point; this closes that gap and is the genuine differentiation
        # from IPOP's blind restart. Verified non-harmful across BBOB dim2/dim3
        # and the CEC2022 dim10 hold-out (no significant regressions).
        ir_archive_frac: float = 0.5,
        # Re-ignition share from dimension 3 up (None = the same at every
        # dimension). A step, not a taper: tapered with the dimension gate it is
        # still 0.31 at 5D, and that alone erases the 5D Gallagher gain
        # (F21 stays 40, F22 70→60). 2026-10-12 sweep:
        # without re-ignition 5D F21 40→95 / F22 70→100, 10D F21 10→80 /
        # F22 35→65 (re-seeding around archived strains keeps returning the
        # search to the peaks it already knows), while 2D loses F23 −15.
        ir_archive_frac_hd: "float | None" = 0.0,  # default since 2026-10-13 (v2_scan2d); None until then
        # Reassortment sweep on a stall (off by default). When a spillover is
        # about to fire on a landscape the learned C does not call ill-
        # conditioned, and no sweep has run since the last meaningful
        # improvement, first swap single coordinates of the best host with the
        # coordinates of remembered optima (answer archive, strain reservoir,
        # the other hosts); keep the best improving swap and repeat until none
        # improves or reassort_budget_frac of the budget is spent. 2026-10-12
        # probe: on 5D F03 / F04 every best single swap removes exactly one
        # coordinate sitting in a neighbouring Rastrigin well (gap 2 → 1 → 0);
        # on the rotated F15 it does nothing.
        reassort_sweep: bool = True,           # default since 2026-10-13 (v2_scan2d)
        reassort_budget_frac: float = 0.1,
        # Additivity test (None = off): after the first pass of a sweep, combine
        # the two best improving swaps on different coordinates (one extra
        # evaluation). On a separable landscape their changes in f add up; if
        # |Δ_ij − (Δ_i + Δ_j)| exceeds this fraction of |Δ_i| + |Δ_j| (or fewer
        # than two coordinates improve), sweeps stop for the rest of the run.
        # Gallagher's coordinate swaps also improve, so improvement alone does
        # not tell separable landscapes apart (5D F21 90→65 without the test).
        reassort_additivity: "float | None" = 0.02,  # default since 2026-10-13 (v2_scan2d)
        # Number of pairs checked: the best coordinate paired with the next
        # reassort_pairs best ones (all must be additive). One pair let 10D
        # F22 (Gallagher 21 peaks) through by chance (75→55).
        reassort_pairs: int = 1,
        # Donors: "memory" = only the remembered optima (answer archive and
        # strain reservoir); "all" also uses the other hosts. With the hosts a
        # 10D pass costs 1000+ evaluations and every run pays it once
        # (10D Gallagher −10 / −20pt).
        reassort_donors: str = "top2",         # default since 2026-10-13 (v2_scan2d); "all" until then
        reassort_top_hosts: int = 10,       # donors="top": the best this many hosts
        # Smallest dimension that sweeps (2D lost F18 −15 with sweeps on).
        reassort_min_dim: int = 2,
        # When the additivity test fails, discard the sweep: the population and
        # the stagnation counters are restored, so the spillover it stood in
        # for still fires (an accepted Gallagher improvement delayed it: 10D
        # F22 75→45).
        reassort_discard: bool = False,
        # Separable decomposition (None = off): when the additivity test passes,
        # treat the landscape as separable and solve it as D one-dimensional
        # problems — a grid of this many points over each coordinate's range,
        # then a ternary refinement inside the best grid cell, coordinate by
        # coordinate (two passes at most). 2026-10-12 probe from v2_hlC's final
        # best (10 seeds): 5D F04 0→10/10, F03 8→10/10; 10D F03 4→10, F04 0→10,
        # at ~3100 / ~5700 evaluations. F20 (coupled consecutive coordinates) and
        # the rotated F15 do not benefit — the additivity test should refuse them.
        reassort_scan: "int | None" = 200,     # default since 2026-10-13 (v2_scan2d)
        # Smallest dimension that uses the recombination waves (3 = the
        # 2026-10-12 default; 2 also runs the stall wave at 2D, where the
        # 2026-10-11 test with the 1e-3 quality gate never fired on F24).
        wave_min_dim: int = 3,
        # Let a stall wave that is still beating the outbreak's best when its
        # share runs out continue on the rest of the budget (once per run; the
        # containment hand-over stays available). 2026-10-12 trace of
        # v2_scan2d on 10D F17: successful runs spent ~8000 evaluations of
        # close contact between the stall wave and the late hand-over.
        wave_follow: bool = False,
        # Run the stall wave regardless of `wave_quality` once this many
        # spillovers in a row have failed (None = off). 2026-10-13 trace of the
        # default on 10D F17 / F18 (instances 1-5): the runs that end at
        # f - f* ~ 1e-2 (8-9 of 20, the worst mean error of the panel) re-seed
        # ~10 times in a row with the wave gated off.
        wave_fail_gate: "int | None" = None,
        # Budget share of a wave released by wave_fail_gate (None = wave_budget_frac).
        wave_fail_budget_frac: "float | None" = None,
        # Early probe (contact tracing, 2026-10-13): once probe_at × budget is
        # spent (≥ wave_min_dim), run one short whitening-free CMA wave from the
        # best host (share probe_frac, λ = wave_handover_lambda rule, σ0 = σ) and,
        # if it gained more decades per evaluation than the outbreak did over the
        # same number of evaluations just before it, let it take the rest of the
        # budget (a hand-over). Otherwise its best rejoins the outbreak. Measured
        # at 20D: the outbreak is 2-6 decades behind IPOP-CMA-ES at 2000
        # evaluations on F08 / F12 and hands over only after ~20000.
        probe_at: "float | None" = None,
        probe_frac: float = 0.05,
        # Cap on the initial population (None = pop_init_mult · D uncapped).
        pop_init_cap: "int | None" = None,
        # Strain competition (≥ wave_min_dim, 2026-10-13): a CMA strain
        # (λ = 2 × the CMA-ES default) co-circulates with the outbreak from the
        # start. Each strain's reproduction rate is an EMA of the decades its own
        # best gained per evaluation; the CMA strain gets that share of the
        # evaluations (clipped to [strain_min, 1 − strain_min]). Its best host
        # immigrates into the outbreak, and a stopped strain re-emerges from
        # the global best with twice the population.
        strain_compete: bool = False,
        strain_min: float = 0.1,
        strain_max: "float | None" = None,   # cap on the CMA strain's share (None = 1 − strain_min)
        strain_immigrate: bool = True,       # the strain's best replaces the outbreak's worst host
        strain_stop_on_handover: bool = False,  # retire the strain once the hand-over has run
        # "own": a strain's rate is the decades its own best gained; "record":
        # only gains of the global best (who set the new record) count, so a
        # strain improving from a worse start gets no credit (2026-10-14: at 5D
        # the CMA strain took its full share on its own progress and starved
        # the outbreak on the multimodal functions).
        strain_credit: str = "own",
        # Scale the strain's share bounds by the covariance dimension gate
        # (_cc_dim_gate: 0 at 2D, 0.375 at 5D, 1 from cc_dim_ref = 10D up): the
        # covariance-learning strain gets budget where covariance learning pays.
        strain_dim_gate: bool = False,
        # Dimension at which the strain gate reaches 1 (None = cc_dim_ref).
        strain_dim_ref: "int | None" = None,
        # Smallest dimension that runs the strain / the fail-gated wave (staged
        # v3: the outbreak alone at low D, + fail-gated wave from 10D, + CMA
        # strain from 20D; 2026-10-14).
        strain_min_dim: int = 0,
        wave_fail_min_dim: int = 0,
        # Takeover: when the CMA strain set the latest record and the outbreak
        # has set none for a stagnation window, the strain gets the whole
        # budget (a variant replacing the circulating one). 2026-10-14: at 20D
        # F10 the 50% share delayed the outbreak's hand-over from ~9k to ~17k.
        strain_takeover: bool = False,
        # Initial hosts by Latin hypercube (each coordinate's range cut into
        # n_pop strata, one host per stratum) instead of i.i.d. uniform.
        init_lhs: bool = False,
        # When the outbreak latches ill-conditioning (the containment hand-over)
        # while a CMA strain is circulating, give the strain the rest of the
        # budget instead of starting a fresh hand-over wave: it has already
        # learned the shape (2026-10-14: at 20D F10 the shared budget delayed
        # the hand-over from ~9k to ~17k evaluations).
        strain_handover: bool = False,
        # Retire the strain (lower bound share) once the outbreak's learned C
        # latches ill-conditioning: the containment hand-over then runs on the
        # outbreak's own learned shape with the budget the strain would take.
        strain_retire_ill: bool = False,
        # Cross-immunity: when the CMA strain's covariance reaches the
        # ill-conditioning threshold first, the outbreak adopts that shape and
        # latches (the hand-over then whitens by it). Only the shape crosses,
        # not hosts.
        strain_share_shape: bool = False,
        strain_lam_mult: float = 2.0,        # CMA strain λ = this × the CMA-ES default
        # Budget share of the containment hand-over wave (1.0 = all that is
        # left). With < 1 the wave is followed (wave_follow-style) only while it
        # still beats the outbreak, else the outbreak resumes. 2026-10-14: on
        # CEC2022 hybrids (G06) the learned C latches ill at ~4700 evaluations
        # and the whole budget goes to a CMA run that stalls.
        handover_frac: float = 1.0,
        # Transmission chain (2026-10-14, CMA-ES's evolution-path idea inside the
        # outbreak): the learned C also gets a rank-one term along the cumulated
        # drift of the outbreak's centre (mean of its best quarter of hosts),
        # normalised by σ. A single steep direction (BentCigar) shows up in that
        # drift long before the rank-μ term resolves it: without the pycma
        # waves 10D F12 stalls at f ≈ 10 with the learned C at cond 10^3.7.
        cc_chain: bool = False,
        # Still adapting (2026-10-14): no spillover while the learned C's
        # log10 condition rose by more than this over the last stagnation
        # window — the outbreak is still adapting to the host's shape (None =
        # off). Without the pycma waves 10D F12 re-seeds every ~1500
        # evaluations while its C is still growing.
        spill_adapt_veto: "float | None" = None,
        # Contact tracing (2026-10-15): this fraction of the spillover's re-seed
        # is drawn around the source the past outbreaks point at — the
        # rank-weighted mean of the best abandoned-basin bests, spread by their
        # per-coordinate s.d. — instead of a repelled uniform draw (None = off).
        # On big-valley landscapes (rotated Rastrigin, Schaffers, Lunacek) the
        # local optima cluster around the global one, which a uniform re-seed
        # ignores.
        spill_trace: "float | None" = None,
        # New host population (2026-10-15): at a basin switch the population is
        # regrown to this fraction of its initial size and the size schedule
        # restarts over the remaining budget (None = off). A full restart late
        # in the run otherwise re-seeds only the shrunk population (10D F15:
        # ~10 basin switches, the later ones with 40 hosts against 160 at start).
        switch_regrow: "float | None" = None,
        # Own-outbreak pacing (2026-10-15): after a basin switch, stagnation is
        # measured on the new outbreak's own best (meaningful improvement as for
        # the global best) until it overtakes the global best. Otherwise a fresh
        # outbreak cannot "improve" until it beats the old best and is cut after
        # one window (10D F15: a basin switch every ~1660 evaluations).
        switch_pace: bool = False,
        # Pacing only while the outbreak can still overtake (2026-10-16): a
        # paced step resets the stall count only if, at the log-rate of that
        # step, the outbreak would reach the global best within this fraction
        # of the remaining budget (None = always). Gallagher's peaks otherwise
        # let a restart keep "improving" toward a local peak that can never win.
        pace_horizon: "float | None" = None,
        # Ring vaccination (2026-10-17): this fraction of an ordinary
        # spillover's re-seed is placed in rings around the best host, at radii
        # log-uniform between 10·σ and the restart σ (None = off). In 10D a
        # uniform re-seed lands nowhere useful: on F17 a run stuck at 0.016
        # fires 8 failed spillovers in a row while the neighbouring basins of
        # the Schaffer rings sit just outside the converged σ.
        spill_ring: "float | None" = None,
        # σ steps per evaluation, not per generation (2026-10-17, ≥3D): the
        # σ factor of a generation is raised to the power
        # max(1, evaluations of that generation / sigma_batch) (None = off).
        # One generation is 0.25·n_pop evaluations (80 at 20D with 16·D hosts),
        # so σ moves ×0.95–1.1 per 80 evaluations: on 20D F01 σ goes 5.2 → 4.3
        # in the first 2000 evaluations while CMA-ES is already at f 1e-4.
        sigma_batch: "float | None" = None,
        # Rank infectivity (2026-10-17, ≥3D): parent weights exp(−β·rank/(n−1))
        # with this β instead of the range-normalised f (None = off). A few
        # very bad hosts stretch f_max − f_min, which flattens the weights of
        # everyone else early in a run.
        select_rank: "float | None" = None,
        # Mirrored close contact (2026-10-17, ≥3D): close-contact children come
        # in pairs from the same parent with opposite noise.
        close_mirror: bool = False,
        # Variable virulence (2026-10-17): each droplet child draws its own F
        # from N(h2h_F, this) clipped to [0.05, 1.0] (None = the fixed h2h_F).
        h2h_dither: "float | None" = None,
        # Control, not a method (2026-10-17): every wave is a plain CMA-ES from
        # the best host — default λ, σ0 = 0.2·span as the baselines, no
        # whitening by the learned C — at the same switch points. Measures how
        # much of the waves' gain comes from MC-ESO's state.
        wave_plain: bool = False,
        # Growing outbreaks (2026-10-18, ≥3D): the run starts from
        # max(20, pop_init_hd·D) hosts instead of pop_init_mult·D (None = off),
        # and each basin switch regrows the population to switch_grow × the
        # size the previous outbreak started with, restarting the size schedule
        # over the remaining budget (None = off). Reverses the large-to-small
        # order of the default: at 20D IPOP-CMA-ES behind the same separability
        # scan beats MC-ESO on the ill-conditioned unimodal functions (F06-F12)
        # that a 320-host start wastes, and on multimodal ones grows λ later.
        pop_init_hd: "float | None" = None,
        # Quarantine (2026-10-20, ≥3D): an ordinary spillover keeps this
        # fraction of the best hosts as they are and re-seeds only the rest
        # (None = keep only the best). On 20D F09 / F07 a spillover fires every
        # ~3200 evaluations while the population is still creeping along the
        # curved valley, and re-seeding everyone but the best destroys it.
        spill_keep: "float | None" = None,
        # Still spreading (2026-10-20, ≥3D): no ordinary spillover while the
        # population's median f fell by more than this many decades over the
        # last stagnation window (None = off): the best has stalled but the
        # outbreak is still moving along the valley.
        spill_med_veto: "float | None" = None,
        # Forecast child (2026-10-20, ≥3D): each generation one extra child at
        # the minimiser of a separable quadratic f ≈ a + bᵀy + Σ c_i y_i² fitted
        # to the better half of the hosts, y = Vᵀ(x − x_best) in the learned
        # C's eigenbasis (coordinates with c_i ≤ 0 do not move; the step is
        # capped at the population's radius). Replaces the worst host if better.
        forecast: bool = False,
        # Droplet pull toward a random host of the best h2h_pbest fraction
        # instead of a niched strain (None = strains), as JADE's current-to-pbest.
        h2h_pbest: "float | None" = None,
        # Droplet difference x_a − x_b with x_b drawn from the population plus
        # the hosts displaced by children (capped at n_pop), as JADE's archive.
        h2h_archive: bool = False,
        # Recovery (2026-10-22, ≥3D, the R of SIR): hosts older than this many
        # generations — none of their children has displaced them as worst, so
        # they have stopped transmitting — are removed first, up to half of
        # each generation's kill quota, never the best fifth (None = SI only).
        recover_age: "int | None" = None,
        # Chain tracing (2026-10-22, ≥3D): before an ordinary spillover, a
        # golden-section line search from the best host along each of the
        # learned C's top-2 axes (≤ 2·40 evaluations); the spillover is skipped
        # if it improves the best (once per meaningful improvement).
        chain_trace: bool = False,
        # forecast fitted to the best 4·D points of the last 20·D evaluations
        # instead of the better half of the hosts.
        forecast_hist: bool = False,
        # forecast only on runs the channel router committed to DROPLET (the
        # ill-conditioned valleys it was measured to help), not on multimodal
        # keep-air runs (Gallagher, where it misleads).
        forecast_route: bool = False,
        switch_grow: "float | None" = None,
        # Airborne jumps from a Cauchy instead of a Gaussian (rare long-range
        # spread between distant basins, e.g. CEC2022 compositions).
        air_cauchy: bool = False,
        # While neither strain gains (both rates 0): "keep" the last share, or
        # "explore" = give the outbreak (the explorer) the budget back by
        # dropping to the lower bound (2026-10-14: with "keep", the CMA strain
        # held 90% after converging on a Gallagher peak; 10D F21 9/10 → 2/10).
        strain_idle: str = "keep",
        # Share before either strain has a rate (None = 0.5). A strain that
        # starts from a few cases (the lower bound) has to earn its share.
        strain_init: "float | None" = None,
        strain_ema: float = 0.3,
        # Allow the separable scan again at the next stall after a scan that did
        # not improve the best (True = the 2026-10-13 default behaviour).
        reassort_scan_retry: bool = True,
        ir_reignite_sigma_ratio: float = 0.05,
        ir_repel_radius_ratio: float = 0.1,
        ir_repel_max_tries: int = 20,
    ):
        super().__init__(benchmark, seed)
        # Dimension-aware population: fixed 20 underfills high-dim search.
        self.n_pop = (n_pop if n_pop is not None
                      else max(20, int(round(n_pop_dim_mult * self.dim))))
        self.pop_schedule = pop_schedule
        self.pop_init_mult = pop_init_mult
        self.pop_final_mult = pop_final_mult
        self.pop_min = pop_min
        self.pop_shrink_power = pop_shrink_power
        if pop_schedule == "linear" and n_pop is None:
            # An explicit n_pop is the initial size of the schedule.
            self.n_pop = max(20, int(round(pop_init_mult * self.dim)))
            if pop_init_cap is not None:
                self.n_pop = max(20, min(self.n_pop, int(pop_init_cap)))
            if pop_init_hd is not None and self.dim >= 3:
                self.n_pop = max(20, int(round(pop_init_hd * self.dim)))
        self._pop_final = max(pop_min, int(round(pop_final_mult * self.dim)))
        self.sigma_floor_ill_ratio = sigma_floor_ill_ratio
        self.hd_sigma_success = hd_sigma_success
        self._wave_lambda = hd_wave_lambda if self.dim >= wave_min_dim else None
        self.wave_sigma_ratio = wave_sigma_ratio
        self.wave_budget_frac = wave_budget_frac
        self.wave_quality = wave_quality
        self.wave_follow = wave_follow
        self.wave_fail_gate = wave_fail_gate
        self.wave_fail_budget_frac = wave_fail_budget_frac
        self.probe_at = probe_at
        self.strain_compete = strain_compete
        self.strain_min = strain_min
        self.strain_max = strain_max
        self.strain_immigrate = strain_immigrate
        self.strain_stop_on_handover = strain_stop_on_handover
        self.strain_credit = strain_credit
        self.strain_dim_gate = strain_dim_gate
        self.strain_dim_ref = strain_dim_ref
        self.strain_min_dim = strain_min_dim
        self.wave_fail_min_dim = wave_fail_min_dim
        self.strain_takeover = strain_takeover
        self.init_lhs = init_lhs
        self.strain_handover = strain_handover
        self.strain_retire_ill = strain_retire_ill
        self.strain_share_shape = strain_share_shape
        self.strain_lam_mult = strain_lam_mult
        self.handover_frac = handover_frac
        self.cc_chain = cc_chain
        self.spill_adapt_veto = spill_adapt_veto
        self.spill_trace = spill_trace
        self.switch_regrow = switch_regrow
        self.switch_pace = switch_pace
        self.pace_horizon = pace_horizon
        self.spill_ring = spill_ring
        self.sigma_batch = sigma_batch
        self.select_rank = select_rank
        self.close_mirror = close_mirror
        self.h2h_dither = h2h_dither
        self.wave_plain = wave_plain
        self.switch_grow = switch_grow
        self.spill_keep = spill_keep
        self.spill_med_veto = spill_med_veto
        self.forecast = forecast
        self.h2h_pbest = h2h_pbest
        self.h2h_archive = h2h_archive
        self.recover_age = recover_age
        self.chain_trace = chain_trace
        self.forecast_hist = forecast_hist
        self.forecast_route = forecast_route
        self.air_cauchy = air_cauchy
        self.strain_idle = strain_idle
        self.strain_init = strain_init
        self.strain_ema = strain_ema
        self.probe_frac = probe_frac
        self.reassort_scan_retry = reassort_scan_retry
        self.wave_use_C = wave_use_C
        self.wave_ill_handover = wave_ill_handover
        self.wave_handover_drill = wave_handover_drill
        self.wave_handover_lambda = wave_handover_lambda
        self.wave_handover_restart = wave_handover_restart
        self._n_pop0 = self.n_pop          # initial size; restored at each run start
        self.sigma = sigma
        self.air_ratio = air_ratio
        self.n_elite_max = n_elite_max
        self.niche_radius_ratio = niche_radius_ratio
        self.host_sigma_min_scale = host_sigma_min_scale
        self.air_sigma_amplifier = air_sigma_amplifier
        self.droplet_variant = droplet_variant
        self.channel_schedule = channel_schedule
        self.cond_droplet_thresh = cond_droplet_thresh
        self.align_close_thresh = align_close_thresh
        self.close_mgap_thresh = close_mgap_thresh
        self.cond_droplet_early = cond_droplet_early
        self.route_commit_gen = route_commit_gen
        self.log_slope_threshold = log_slope_threshold
        self.h2h_ratio = h2h_ratio
        self.h2h_F = h2h_F
        self.kill_fraction = kill_fraction
        self.restart_no_improve_threshold = restart_no_improve_threshold
        self.restart_window_dim_scale = restart_window_dim_scale
        self.softmax_beta = softmax_beta
        self.restart_sigma_ratio = restart_sigma_ratio
        self.restart_quality_rel_floor = restart_quality_rel_floor
        self.basin_switch_after_failed_spillovers = basin_switch_after_failed_spillovers
        self.basin_switch_quality_rel_floor = basin_switch_quality_rel_floor
        self.solution_archive_max = solution_archive_max
        self.exhausted_sigma_tol = exhausted_sigma_tol
        self.exhausted_no_improve_mult = exhausted_no_improve_mult
        self.hunt_no_improve_mult = hunt_no_improve_mult
        self.hunt_level_tol = hunt_level_tol
        self.sigma_up = sigma_up
        self.sigma_down = sigma_down
        self.sigma_floor_ratio = sigma_floor_ratio
        self.sigma_ceil_ratio = sigma_ceil_ratio
        self.precision_sigma_ratio = precision_sigma_ratio
        self.sigma_drill_down = sigma_drill_down
        self.h2h_CR = h2h_CR
        self.empirical_cov_floor = empirical_cov_floor
        self.cov_floor_low = cov_floor_low
        self.cov_ratio_lo = cov_ratio_lo
        self.cov_ratio_hi = cov_ratio_hi
        self.cov_ratio_beta = cov_ratio_beta
        self.cc_dim_ref = cc_dim_ref
        self.cc_learning_rate = cc_learning_rate
        self.cc_mu_frac = cc_mu_frac
        self.cc_spill_freeze_gens = cc_spill_freeze_gens
        self.cc_keep_on_spillover = cc_keep_on_spillover
        self.router_learned_cond = router_learned_cond
        self.cc_persist_frac = cc_persist_frac
        self.cc_air_ratio = cc_air_ratio
        self.cc_h2h_ratio = cc_h2h_ratio
        self.cc_cov_floor = cc_cov_floor
        self.ir_archive_frac = ir_archive_frac
        self.ir_archive_frac_hd = ir_archive_frac_hd
        self.reassort_sweep = reassort_sweep
        self.reassort_budget_frac = reassort_budget_frac
        self.reassort_additivity = reassort_additivity
        self.reassort_pairs = reassort_pairs
        self.reassort_donors = reassort_donors
        self.reassort_top_hosts = reassort_top_hosts
        self.reassort_min_dim = reassort_min_dim
        self.reassort_discard = reassort_discard
        self.reassort_scan = reassort_scan
        self.ir_reignite_sigma_ratio = ir_reignite_sigma_ratio
        self.ir_repel_radius_ratio = ir_repel_radius_ratio
        self.ir_repel_max_tries = ir_repel_max_tries

    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Hybrid boundary handler: reflect on large overshoots (preserves
        diversity), snap to boundary on small overshoots (within span × 1e-3).

        Without the snap-clause, boundary-optimum landscapes (F05 LinearSlope)
        can never reach exact zero: σ-step pushes a few epsilons over the
        boundary, reflection bounces those few epsilons back inside, and the
        candidate orbits the optimum at radius σ_floor forever.
        """
        span = hi - lo
        # Reflected version (existing logic — wraps periodically for large
        # excursions so the candidate cloud doesn't pile up at the bound).
        x_rel = (x - lo) % (2 * span)
        x_rel = np.where(x_rel > span, 2 * span - x_rel, x_rel)
        reflected = x_rel + lo
        # Snap to boundary when the overshoot is tiny — lets the algorithm
        # actually land on the bound (F05). Threshold is much smaller than any
        # σ that would carry meaningful diversification, so large reflects are
        # untouched.
        snap = span * 1e-3
        result = reflected
        result = np.where((x < lo) & (lo - x < snap), lo, result)
        result = np.where((x > hi) & (x - hi < snap), hi, result)
        return result

    def _niche_elites(self, pop_x: np.ndarray, pop_f: np.ndarray,
                      niche_radius: float) -> set:
        """Strain coexistence: pick up to n_elite_max spatially-separated hosts.

        Walk candidates in f-ascending order, keep one if it is at least
        ``niche_radius`` away from every strain already picked. The resulting
        pool is the donor side of the droplet channel — children pulled toward
        a randomly chosen strain rather than a single global best, preserving
        multi-basin coverage.
        """
        elite_idx: list[int] = []
        elite_pos: list[np.ndarray] = []
        for candidate in np.argsort(pop_f):
            if not elite_pos or np.all(
                np.linalg.norm(pop_x[candidate] - np.array(elite_pos), axis=1) > niche_radius
            ):
                elite_idx.append(int(candidate))
                elite_pos.append(pop_x[candidate])
            if len(elite_idx) >= self.n_elite_max:
                break
        return set(elite_idx)

    def _softmax_weights(self, f_vals: np.ndarray) -> np.ndarray:
        # Softmax over -f with unit temperature (fixed; ablation showed no
        # benefit from tuning T away from the canonical 1.0). Note this uses RAW
        # f differences, so the selection pressure depends on the problem's f
        # scale; softmax_beta > 0 switches to the range-normalised form, which
        # gives the same pressure whatever the f scale or the dimension.
        if self.select_rank is not None and self.dim >= 3 and len(f_vals) > 1:
            r = np.empty(len(f_vals))
            r[np.argsort(f_vals, kind="stable")] = np.arange(len(f_vals))
            w = np.exp(-self.select_rank * r / (len(f_vals) - 1))
            return w / w.sum()
        if self.softmax_beta > 0.0:
            f_min, f_max = float(f_vals.min()), float(f_vals.max())
            spread = f_max - f_min
            if spread <= 0.0:
                return np.full(len(f_vals), 1.0 / len(f_vals))
            w = np.exp(-self.softmax_beta * (f_vals - f_min) / spread)
            return w / w.sum()
        scores = f_vals.max() - f_vals
        scores -= scores.max()
        w = np.exp(scores)
        return w / w.sum()

    def _host_sigma_scale(self, pop_f: np.ndarray, pop_age: np.ndarray) -> np.ndarray:
        """Per-host σ scaling factor: high-quality / old hosts get a smaller
        fraction of σ_global so they probe finer. lq ∈ [0, 1] is the relative
        log-fitness, ar ∈ [0, 1] is age / 5 (5 = canonical lifespan), and the
        factor is ``host_sigma_min_scale ** (lq · (0.7 + 0.3 · ar))``."""
        lf = np.log10(pop_f + 1e-10)
        spread = float(lf.max() - lf.min()) + 1e-30
        lq = np.clip((lf.max() - lf) / spread, 0.0, 1.0)
        # Age normalizer fixed at 5 generations (canonical default).
        ar = np.minimum(pop_age.astype(float) / 5.0, 1.0)
        return self.host_sigma_min_scale ** (lq * (0.7 + 0.3 * ar))

    def _meaningful_improvement(self, f: float, log_best_ref: float, evals_since_reset: int) -> bool:
        if evals_since_reset == 0:
            return True
        # max() floors f to 1e-300 so FP-noise-induced negative values near the
        # optimum (e.g. -1e-15 from BBOB - f_opt cancellation) don't crash log10.
        slope = (log_best_ref - math.log10(max(f, 1e-300))) / evals_since_reset
        return slope >= self.log_slope_threshold

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        st = self._init_state(max_evals)
        compete = (self.strain_compete and self._wave_lambda is not None
                   and self.dim >= self.strain_min_dim)
        if compete:
            st.strain_start = len(st.history_f)
        while st.budget_left:
            if compete and self._strain_step(st):
                break
            # A spillover may consume the remaining budget; if so, stop before
            # spending a (now-impossible) generation, matching the original
            # mid-loop ``break``.
            if self._maybe_spillover(st):
                break
            b0_rec = st.best_so_far
            rec = self.strain_credit == "record"
            b0 = st.best_so_far if rec else float(np.min(st.pop_f))
            n0 = len(st.history_f)
            self._run_generation(st)
            self._record_generation(st)
            if compete and st.best_so_far < b0_rec:
                st.outbreak_record_eval = len(st.history_f)
            if compete:
                st.outbreak_rate = self._ema_rate(st.outbreak_rate, b0,
                                                  st.best_so_far if rec else float(np.min(st.pop_f)),
                                                  len(st.history_f) - n0)

        # Reported set: the surviving hosts, the strain reservoir, and the answer
        # archive of every basin drilled and abandoned.
        solutions = ([x.copy() for x in st.pop_x]
                     + [x.copy() for x in st.ir_archive_x]
                     + [x.copy() for x in st.sol_archive_x])
        result = self._make_result(st.history_x, st.history_f, st.history_pop,
                                   solutions=solutions or None)
        result.history_pop_sigma = st.history_pop_sigma
        result.history_sigma_global = st.history_sigma_global
        result.history_n_elite = st.history_n_elite
        result.history_no_improve = st.history_no_improve
        result.history_eval_count = st.history_eval_count
        result.history_sigma_eval = st.history_sigma_eval
        result.trace = self._trace_result(st)
        return result

    # ── diagnostics for the results UI (recording only) ────────────────────
    @staticmethod
    def _pad_eval_channel(st: _MCESOState, code: int) -> None:
        missing = len(st.history_f) - len(st.eval_channel)
        if missing > 0:
            st.eval_channel.extend([code] * missing)

    def _trace_result(self, st: _MCESOState) -> dict:
        self._pad_eval_channel(st, _CH_UNKNOWN)
        return {
            "gen": {k: list(v) for k, v in st.trace_gen.items()},
            "eval_channel": bytes(st.eval_channel[:len(st.history_f)]),
            "channels": TRACE_CHANNELS,
            "routes": TRACE_ROUTES,
            "events": list(st.trace_events),
        }

    # ── run lifecycle ───────────────────────────────────────────────────────
    def _init_state(self, max_evals: int) -> _MCESOState:
        """Seed the RNG, draw the initial host pool, and prime the history
        buffers / bookkeeping scalars for one run."""
        rng = np.random.default_rng(self.seed)
        self.n_pop = self._n_pop0
        lo, hi = self.bounds
        span = hi - lo
        sigma = self.sigma * span        # σ_init (scalar; per-dim bounds not supported)

        # Pre-allocate numpy arrays — avoids repeated list→array conversions per iteration
        if self.init_lhs:
            n = self.n_pop
            strata = np.stack([rng.permutation(n) for _ in range(self.dim)], axis=1)
            pop_x = lo + (hi - lo) * (strata + rng.uniform(0, 1, (n, self.dim))) / n
        else:
            pop_x = rng.uniform(lo, hi, (self.n_pop, self.dim))      # (n_pop, dim)
        pop_f = np.array([self.func(x) for x in pop_x])              # (n_pop,)
        # pop_age is the σ_i age-ratio normalizer (divided by 5, the canonical lifespan).
        # Death is f-based (μ+λ greedy), so age has no effect on survival.
        pop_age = np.zeros(self.n_pop, dtype=int)

        best_so_far = float(pop_f.min())
        st = _MCESOState(
            rng=rng, lo=lo, hi=hi, span=span, max_evals=max_evals,
            sigma=float(sigma), sigma_init=float(sigma),
            niche_radius=self.niche_radius_ratio * span,
            pop_x=pop_x, pop_f=pop_f, pop_age=pop_age,
            best_so_far=best_so_far,
            basin_best=best_so_far,
            # Reference scale for the relative quality-floor gates in spillover.
            # Anchored to the initial-population best so the floors adapt to the
            # problem's natural f range (multiplicatively scale-invariant).
            f_init_scale=max(abs(best_so_far), 1e-300),
            # log10(f) at last meaningful reset.
            log_best_ref=math.log10(max(best_so_far, 1e-300)),
        )
        st.history_x = [row.copy() for row in pop_x]
        st.history_f = pop_f.tolist()
        st.history_pop = [pop_x.copy()]
        st.history_pop_sigma = [float(sigma) * self._host_sigma_scale(pop_f, pop_age)]
        st.eval_channel = bytearray([_CH_INIT] * len(st.history_f))
        st.trace_gen = {k: [] for k in TRACE_GEN_KEYS}
        return st

    # ── informed restart (reservoir re-ignition + herd-immunity repulsion) ──
    # Called from _maybe_spillover; overridable so the diagnostic ablation can
    # pin the original blind uniform restart (see mceso_ablations.py).
    def _on_spillover_start(self, st: _MCESOState, basin_switch: bool) -> None:
        """Harvest the converging basin into the persistent strain archive and
        record its centroid for herd-immunity repulsion — before the reseed
        overwrites the population."""
        # Answer archive: keep the best point of the basin being abandoned.
        # Sequential niching is defined by holding found optima off-line
        # (Beasley+ 1993); without this the run re-finds optima and drops them.
        # Measured at 25k evals: on N06-Shubert2D a run touches 12.2 of the 18
        # optima but reports 6.0, and on N10-ModRastrigin 10.4 of 12 but reports
        # 6.4. Recording only — the search is untouched, so SR is unchanged.
        if self.solution_archive_max > 0 and len(st.pop_f):
            best_i = int(np.argmin(st.pop_f))
            st.sol_archive_x.append(st.pop_x[best_i].copy())
            st.sol_archive_f.append(float(st.pop_f[best_i]))
            if len(st.sol_archive_f) > self.solution_archive_max:
                keep = np.argsort(st.sol_archive_f)[:self.solution_archive_max]
                st.sol_archive_x = [st.sol_archive_x[i] for i in keep]
                st.sol_archive_f = [float(st.sol_archive_f[i]) for i in keep]
        # A new basin has its own geometry: reset the persistent close-contact
        # covariance so re-adaptation is instant (CMA-ES instead carries its
        # covariance and evolution path across restarts).
        if self._cc_dim_gate() > 0.0 and (basin_switch or not self.cc_keep_on_spillover):
            # Only a real basin switch invalidates the learned geometry. An
            # ordinary spillover keeps the population's best and re-seeds around
            # it, so the shape it has learned is still the right one — and
            # resetting it there is fatal on functions whose solution *is* an
            # extremely elongated C. Traced on F12-BentCigar at dim 10: the run
            # that succeeds lets the covariance collapse to effective rank 1.00
            # with condition number 2.2e6; the run that fails has it reset back
            # to rank 9.08 just as it was getting there, and cycles forever at
            # median f 37.8.
            st.cc_C = np.eye(self.dim)
        # Remember the basin we are about to abandon (its current best location).
        best_i = int(np.argmin(st.pop_f))
        st.ir_basin_centroids.append(st.pop_x[best_i].copy())
        # Fold the current population's niched elites into the archive, keeping
        # it niche-separated and capped at n_elite_max (best-f wins ties).
        elite_idx = self._niche_elites(st.pop_x, st.pop_f, st.niche_radius)
        cand_x = [st.pop_x[i].copy() for i in elite_idx] + list(st.ir_archive_x)
        cand_f = [float(st.pop_f[i]) for i in elite_idx] + list(st.ir_archive_f)
        kept_x: list[np.ndarray] = []
        kept_f: list[float] = []
        for j in np.argsort(cand_f):
            xj = cand_x[j]
            if not kept_x or np.all(
                np.linalg.norm(np.array(kept_x) - xj, axis=1) > st.niche_radius
            ):
                kept_x.append(xj)
                kept_f.append(cand_f[j])
            if len(kept_x) >= self.n_elite_max:
                break
        st.ir_archive_x = kept_x
        st.ir_archive_f = kept_f

    def _diversified_reseed(self, st: _MCESOState, x_best_snap) -> np.ndarray:
        """Return one informed re-seed candidate. When the basin is **exhausted**
        (sequential-niching restart), commit to a genuinely new region: a pure
        repelled-uniform draw with a *fine* repel radius and no reservoir
        re-ignition, so the search is not pulled back to an already-drilled basin
        and densely packed optima (Shubert ~0.88 apart) are not masked. Otherwise
        the usual informed restart: reservoir re-ignition (tight Gaussian around a
        surviving archived strain) with prob ir_archive_frac, else a basin-repelled
        uniform draw avoiding remembered basins."""
        rng, lo, hi, dim = st.rng, st.lo, st.hi, self.dim
        if self._basin_exhausted(st):
            repel_r = 0.02 * st.span
            cand = rng.uniform(lo, hi, dim)
            if st.ir_basin_centroids:
                centroids = np.array(st.ir_basin_centroids)
                for _ in range(self.ir_repel_max_tries):
                    if np.all(np.linalg.norm(centroids - cand, axis=1) > repel_r):
                        break
                    cand = rng.uniform(lo, hi, dim)
            return cand
        # Reservoir re-ignition.
        ir_frac = (self.ir_archive_frac_hd
                   if self.ir_archive_frac_hd is not None and self.dim >= 3
                   else self.ir_archive_frac)
        if st.ir_archive_x and rng.random() < ir_frac:
            k = rng.integers(0, len(st.ir_archive_x))
            sigma = self.ir_reignite_sigma_ratio * st.span
            return self._reflect(
                st.ir_archive_x[k] + sigma * rng.standard_normal(dim), lo, hi)
        if self.spill_trace is not None and rng.random() < self.spill_trace:
            traced = self._traced_source(st)
            if traced is not None:
                return traced
        # Herd-immunity repulsion — uniform draw avoiding remembered basins.
        repel_r = self.ir_repel_radius_ratio * st.span
        cand = rng.uniform(lo, hi, dim)
        if st.ir_basin_centroids:
            centroids = np.array(st.ir_basin_centroids)
            for _ in range(self.ir_repel_max_tries):
                if np.all(np.linalg.norm(centroids - cand, axis=1) > repel_r):
                    break
                cand = rng.uniform(lo, hi, dim)
        return cand

    def _traced_source(self, st: _MCESOState) -> "np.ndarray | None":
        """spill_trace: one draw around the source the past outbreaks point at.
        The best half (at least 3) of the abandoned-basin bests and the current
        best are averaged with log-rank weights; the draw is that centre plus a
        Gaussian with their per-coordinate s.d. (floored at 1% of the span)."""
        xs = list(st.sol_archive_x)
        fs = list(st.sol_archive_f)
        if len(st.pop_f):
            i = int(np.argmin(st.pop_f))
            xs.append(st.pop_x[i])
            fs.append(float(st.pop_f[i]))
        if len(xs) < 3:
            return None
        order = np.argsort(fs)[:max(3, len(xs) // 2)]
        P = np.array([xs[j] for j in order])
        w = np.log(len(order) + 0.5) - np.log(np.arange(1, len(order) + 1))
        w /= w.sum()
        centre = w @ P
        sd = np.maximum(P.std(axis=0), 0.01 * st.span)
        return self._reflect(centre + sd * st.rng.standard_normal(self.dim), st.lo, st.hi)

    def _droplet_strain_positions(self, st: _MCESOState, elite_arr: np.ndarray,
                                  n_h2h: int) -> np.ndarray:
        """Donor ('strain') positions for the droplet channel's current-to-best
        pull — one (dim,) row per child. Base: sample from the *live* niched-elite
        indices (original behaviour, RNG-identical). A subclass returns positions
        from a persistent strain archive so the pull keeps targeting coexisting
        basins even after the live population has collapsed onto one."""
        if self.h2h_pbest is not None:
            top = np.argsort(st.pop_f)[:max(2, int(round(self.h2h_pbest * len(st.pop_f))))]
            return st.pop_x[st.rng.choice(top, size=n_h2h)]
        return st.pop_x[st.rng.choice(elite_arr, size=n_h2h)]

    # ── spillover (stagnation re-seed) + sequential niching ─────────────────
    def _basin_exhausted(self, st: _MCESOState) -> bool:
        """Scale- and shift-invariant convergence detector: the step size σ has
        bottomed out at its floor (the search can no longer drill any finer) AND
        the run has stagnated. No reference to the (unknown, problem-dependent)
        optimum value — only σ relative to the domain span and the improvement
        history. When true the basin is at the algorithm's resolution limit, so
        leaving it for a fresh repelled basin cannot cost SR (base would make no
        further progress here either) while it can discover additional optima."""
        sigma_bottomed = st.sigma <= self.exhausted_sigma_tol * st.span * self.sigma_floor_ratio
        if st.has_exhausted and self.hunt_level_tol > 0.0:  # noqa: SIM102
            # Equal-height multi-global problems give the run a free landmark:
            # once a later hunt matches the depth already banked, that basin
            # holds another global optimum and there is nothing more to learn by
            # drilling it. A hunt that has *not* matched it keeps going, which is
            # what rugged landscapes need — on N06-Shubert2D, stopping hunts at a
            # fixed sigma costs peak ratio because a shallow hunt cannot tell a
            # global optimum from one of the ~760 local ones.
            sigma_bottomed = (sigma_bottomed
                              or st.basin_best <= self.hunt_level_tol * st.f_init_scale)
        mult = self.exhausted_no_improve_mult
        if st.has_exhausted and self.hunt_no_improve_mult > 0.0:
            # The first exhaustion has to be conservative — declaring it early on
            # a hard single-optimum function would restart a run that was still
            # drilling. Later hunts carry no such risk (depth is banked and
            # archived), and at 5000 evals the 3x window costs ~600 evals per
            # hunt spent sitting at the sigma floor with nothing left to do.
            mult = self.hunt_no_improve_mult
        stagnated = st.no_improve >= mult * self._stagnation_window()
        if sigma_bottomed and stagnated:
            if not st.has_exhausted:
                st.trace_events.append((len(st.history_f), "exhausted"))
            st.has_exhausted = True
        return sigma_bottomed and stagnated

    def _cc_dim_gate(self) -> float:
        """How strongly the persistent-covariance machinery applies at this
        dimension: 0 at dim 2 (so the tuned low-dim behaviour is bit-identical)
        rising to 1 at ``cc_dim_ref``."""
        if self.cc_learning_rate <= 0.0:
            return 0.0
        span = max(1e-9, self.cc_dim_ref / 2.0 - 1.0)
        return float(min(1.0, max(0.0, (self.dim / 2.0 - 1.0) / span)))

    def _cc_channel_ratios(self) -> tuple[float, float]:
        """Airborne / droplet shares, tapered toward the close-contact-dominant
        setting the learner needs as the dimension gate opens."""
        g = self._cc_dim_gate()
        if g <= 0.0:
            return self.air_ratio, self.h2h_ratio
        return (self.air_ratio + g * (self.cc_air_ratio - self.air_ratio),
                self.h2h_ratio + g * (self.cc_h2h_ratio - self.h2h_ratio))

    def _stagnation_window(self) -> float:
        """Stagnation window in evaluations, scaled for dimension.

        ``no_improve`` is an eval counter, but "the outbreak has stalled" is a
        generational statement and a generation consumes kill_fraction·n_pop
        evaluations, which grows with dim. Scaling by (dim/2)**dim_scale keeps
        the window meaningful in generations at high dim while leaving dim 2 —
        the reference the defaults were tuned at — exactly at
        ``restart_no_improve_threshold`` (the factor is 1.0 there for any
        exponent, so low-dim runs stay bit-identical).
        """
        if self.restart_window_dim_scale == 0.0:
            return float(self.restart_no_improve_threshold)
        return (self.restart_no_improve_threshold
                * (self.dim / 2.0) ** self.restart_window_dim_scale)

    def _spillover_should_fire(self, st: _MCESOState) -> bool:
        """Whether a spillover triggers this generation. Two regimes:

        • basin **exhausted** (drilled out) → fire on stagnation regardless of the
          precision gate, so the search leaves the solved basin to hunt others;
        • otherwise → the precision-gated condition (don't disturb an in-progress
          drilling), which protects SR@1e-10 on hard single-optimum functions.
        """
        window = self._stagnation_window()
        if self.spill_med_veto is not None and self.dim >= 3 and len(st.pop_f):
            now = len(st.history_f)
            st.med_hist.append((now, math.log10(max(float(np.median(st.pop_f)), 1e-300))))
            past = [m for e, m in st.med_hist if e <= now - window]
            if (past and past[-1] - st.med_hist[-1][1] > self.spill_med_veto
                    and not self._basin_exhausted(st)):
                return False
        if self.spill_adapt_veto is not None and len(st.cc_cond_hist) >= 2:
            now_e, now_c = st.cc_cond_hist[-1]
            past = [c for e, c in st.cc_cond_hist if e <= now_e - window]
            if past and now_c - past[-1] > self.spill_adapt_veto:
                return False                    # still adapting to the host
        if self._basin_exhausted(st):
            # Once an optimum is drilled out, no later basin can improve the
            # global best, so `no_improve` never resets and this branch chops
            # every hunt into exactly one window — 300 evals at dim 2, against
            # the ~350 a fresh basin needs to reach 1e-10. Pacing on the basin's
            # own progress instead lets each hunt finish before the next starts.
            return st.no_improve >= window
        return (st.no_improve >= window
                and st.best_so_far > self.restart_quality_rel_floor * st.f_init_scale)

    def _spillover_basin_switch(self, st: _MCESOState) -> bool:
        """Whether this spillover escalates to a full basin switch (wipe the
        population, reset σ to σ_init). An exhausted basin always switches so the
        search fully commits to a fresh repelled region; otherwise base escalates
        only after a streak of failed spillovers (suppressed near precision)."""
        if self._basin_exhausted(st):
            return True
        return (st.consecutive_failed_spillovers >= self.basin_switch_after_failed_spillovers
                and st.best_so_far > self.basin_switch_quality_rel_floor * st.f_init_scale)

    def _maybe_spillover(self, st: _MCESOState) -> bool:
        """Spillover event: when the outbreak stalls, the population spills over
        to a fresh host pool around the global best, giving the search another
        chance to align onto a productive direction (essential for ill-conditioned
        landscapes). Quality-gated to avoid disturbing already-converged runs.

        Returns ``True`` iff the eval budget was exhausted inside the event, in
        which case the caller breaks the generation loop.
        """
        if (self.probe_at is not None and self._wave_lambda is not None and not st.probe_done
                and len(st.history_f) >= self.probe_at * st.max_evals):
            st.probe_done = True
            st.trace_events.append((len(st.history_f), "probe"))
            self._recombination_wave(st, sigma0=st.sigma, budget_frac=self.probe_frac,
                                     lam_mult=self.wave_handover_lambda,
                                     follow=self._probe_outpaces)
            self._pad_eval_channel(st, _CH_LS)
            if st.follow_done:
                st.handover_done = True
            if not st.budget_left:
                return True
        if (self._wave_lambda is not None and self.wave_ill_handover and not st.handover_done
                and (not self.wave_handover_drill
                     or st.sigma < st.span * self.precision_sigma_ratio)
                and self._update_cc_ill(st)):
            st.handover_done = True
            st.trace_events.append((len(st.history_f), "handover"))
            if self.strain_handover and st.strain_es is not None:
                st.strain_owns = True
                return self._strain_step(st)
            self._recombination_wave(st, sigma0=st.sigma, budget_frac=self.handover_frac,
                                     follow=(self.handover_frac < 1.0),
                                     lam_mult=self.wave_handover_lambda,
                                     restart=self.wave_handover_restart)
            self._pad_eval_channel(st, _CH_LS)
            if not st.budget_left:
                return True
        if not self._spillover_should_fire(st):
            return False
        if (self._wave_lambda is not None and not st.wave_spent
                and not self._basin_exhausted(st)
                and (self.wave_quality is None
                     or st.best_so_far <= self.wave_quality * st.f_init_scale
                     or (self.wave_fail_gate is not None and self.dim >= self.wave_fail_min_dim
                         and st.consecutive_failed_spillovers >= self.wave_fail_gate))):
            st.wave_spent = True
            st.trace_events.append((len(st.history_f), "wave"))
            gated = not (self.wave_quality is None
                         or st.best_so_far <= self.wave_quality * st.f_init_scale)
            self._recombination_wave(
                st, follow=self.wave_follow,
                budget_frac=(self.wave_fail_budget_frac if gated else None))
            self._pad_eval_channel(st, _CH_LS)
            return not st.budget_left
        if (self.chain_trace and self.dim >= 3 and not st.trace_spent
                and not self._basin_exhausted(st) and st.cc_C is not None):
            st.trace_spent = True
            before = st.best_so_far
            self._chain_trace(st)
            self._pad_eval_channel(st, _CH_SWEEP)
            if not st.budget_left:
                return True
            if st.best_so_far < before:
                st.no_improve = 0
                return False
        if (self.reassort_sweep and self.dim >= self.reassort_min_dim
                and not st.sweep_spent and not st.sweep_dead
                and st.consecutive_failed_spillovers >= 1
                and not self._basin_exhausted(st) and not self._update_cc_ill(st)):
            st.sweep_spent = True
            st.trace_events.append((len(st.history_f), "reassort"))
            self._reassort_sweep(st)
            self._pad_eval_channel(st, _CH_SWEEP)
            return not st.budget_left

        rng, lo, hi = st.rng, st.lo, st.hi

        # Escalation policy based on consecutive_failed_spillovers:
        #   basin switch: wipe best, fully uniform, σ_init reset
        #   else:         fully uniform, best preserved, smaller σ
        basin_switch = self._spillover_basin_switch(st)
        st.trace_events.append((len(st.history_f), "basin_switch" if basin_switch else "spillover"))
        st.sweep_spent = False          # reassort_sweep: next stall sweeps again
        sigma_restart = (st.sigma_init if basin_switch
                         else st.sigma_init * self.restart_sigma_ratio)
        div_ratio = 1.0

        best_pre_spillover = st.best_so_far
        if (basin_switch and (self.switch_regrow is not None or self.switch_grow is not None)
                and self.pop_schedule == "linear"):
            if self.switch_grow is not None:
                n_new = max(self.n_pop, int(round((st.sched_n0 or self._n_pop0) * self.switch_grow)))
            else:
                n_new = max(self.n_pop, int(round(self._n_pop0 * self.switch_regrow)))
            left = st.max_evals - len(st.history_f)
            n_new = min(n_new, max(self.n_pop, left // 4))
            if n_new > self.n_pop:
                extra = n_new - self.n_pop
                st.pop_x = np.vstack([st.pop_x, np.repeat(st.pop_x[:1], extra, axis=0)])
                st.pop_f = np.concatenate([st.pop_f, np.full(extra, np.inf)])
                st.pop_age = np.concatenate([st.pop_age, np.zeros(extra, dtype=st.pop_age.dtype)])
                self.n_pop = n_new
            st.sched_start = len(st.history_f)
            st.sched_n0 = self.n_pop
        if basin_switch:
            # Wipe everything — including the current best — and re-seed all slots
            # uniformly. best_so_far is preserved as a tracker of the historical
            # best, but the population starts fresh.
            reseed_idx = list(range(self.n_pop))
            x_best_snap = None  # unused since div_ratio = 1.0
        else:
            best_idx_global = int(np.argmin(st.pop_f))
            x_best_snap = st.pop_x[best_idx_global].copy()
            reseed_idx = [i for i in range(self.n_pop) if i != best_idx_global]
            if self.spill_keep is not None and self.dim >= 3:
                keep = set(int(i) for i in np.argsort(st.pop_f)[:max(1, int(round(self.spill_keep * self.n_pop)))])
                reseed_idx = [i for i in reseed_idx if i not in keep]

        # Informed-restart hook: population still holds the converging basin
        # here, so a subclass can harvest a persistent archive / record the
        # abandoned basin centroid before the reseed overwrites it. No-op in base.
        self._on_spillover_start(st, basin_switch)

        n_div = int(round(div_ratio * len(reseed_idx)))
        # Shuffle so the assignment of "diversified" vs "local" is random
        rng.shuffle(reseed_idx)
        diversified = set(reseed_idx[:n_div])
        ring_lo = math.log(max(10.0 * st.sigma, 1e-12))
        ring_hi = math.log(max(sigma_restart, 10.0 * st.sigma, 1e-12))
        for i in reseed_idx:
            if (self.spill_ring is not None and x_best_snap is not None
                    and rng.random() < self.spill_ring):
                r = math.exp(rng.uniform(ring_lo, ring_hi))
                new_x = self._reflect(x_best_snap + r * rng.standard_normal(self.dim), lo, hi)
                sig_log = r
            elif i in diversified or x_best_snap is None:
                new_x = self._diversified_reseed(st, x_best_snap)
                sig_log = (float(np.linalg.norm(new_x - x_best_snap))
                           if x_best_snap is not None else float(sigma_restart))
            else:
                new_x = self._reflect(
                    x_best_snap + sigma_restart * rng.standard_normal(self.dim), lo, hi)
                sig_log = float(sigma_restart)
            f_new = float(self.func(new_x))
            st.pop_x[i] = new_x
            st.pop_f[i] = f_new
            st.pop_age[i] = 0
            st.history_x.append(new_x.copy())
            st.history_f.append(f_new)
            st.history_sigma_eval.append(sig_log)
            st.eval_channel.append(_CH_RESEED)
            if f_new < st.best_so_far:
                st.best_so_far = f_new
            if not st.budget_left:
                break
        st.sigma = sigma_restart
        st.no_improve = 0
        st.basin_best = float(np.min(st.pop_f))
        if self.switch_pace and basin_switch and not self._basin_exhausted(st):
            st.paced = st.basin_best > st.best_so_far
            st.pace_ref = math.log10(max(st.basin_best, 1e-300))
            st.pace_since = 0
        st.log_best_ref = math.log10(max(st.best_so_far, 1e-300))
        st.evals_since_reset = 0
        if not basin_switch:
            st.pop_age[best_idx_global] = 0
            if self.cc_spill_freeze_gens > 0:
                st.cc_freeze_until = (len(st.history_sigma_global)
                                      + self.cc_spill_freeze_gens)

        # Update streak based on whether this spillover improved best
        if st.best_so_far < best_pre_spillover - 1e-12:
            st.consecutive_failed_spillovers = 0
        else:
            st.consecutive_failed_spillovers += 1
        # Budget may have run out inside the re-seed loop above.
        return not st.budget_left

    # ── transmission channels (offspring generators) ────────────────────────
    def _adaptive_cov_floor(self, st: _MCESOState, eigvals_raw: np.ndarray) -> float:
        """Effective close-contact anisotropy floor for this generation.

        Uses an EMA of ``log10`` of the population covariance's natural eigenvalue
        ratio ``λ_max/λ_min``. Measured medians cleanly separate the regimes —
        ill-conditioned valleys sit at 1e5–1e7, rugged/multimodal at 3–600 — so a
        *sustained* huge ratio relaxes the floor toward ``cov_floor_low`` while a
        small ratio (with at most transient spikes, which the EMA rejects) keeps
        it at the safe ``empirical_cov_floor``. Interpolated geometrically between
        ``cov_ratio_lo`` and ``cov_ratio_hi``. Ratio-of-eigenvalues is scale- and
        shift-invariant — never references f values.
        """
        hi, lo = self.empirical_cov_floor, self.cov_floor_low
        if lo >= hi:
            return hi  # adaptation disabled
        ev = np.maximum(eigvals_raw, 1e-300)
        logr = math.log10(float(ev[-1]) / float(ev[0]))
        st.cc_logratio_ema = (logr if st.cc_logratio_ema is None else
                              (1.0 - self.cov_ratio_beta) * st.cc_logratio_ema
                              + self.cov_ratio_beta * logr)
        lo_l, hi_l = math.log10(self.cov_ratio_lo), math.log10(self.cov_ratio_hi)
        t = (st.cc_logratio_ema - lo_l) / (hi_l - lo_l)
        t = min(1.0, max(0.0, t))
        return float(hi * (lo / hi) ** t)

    def _close_contact_children(self, st: _MCESOState, n_local: int,
                                weights: np.ndarray, log_f_max: float,
                                log_f_spread: float) -> tuple[np.ndarray, np.ndarray]:
        """Close-contact transmission — local Gaussian, rotation-aware.

        σ_i adapts to host quality/age, and the noise is drawn from the
        *instantaneous* empirical covariance C_pop of the population (eigenvalues
        mean-normalized to 1, floored to prevent collapsed axes) — full
        anisotropic + rotated shape without CMA-ES-style history accumulation.
        Returns ``(new_local, sigma_i)`` (sigma_i logged as per-child σ).
        """
        if n_local <= 0:
            return np.empty((0, self.dim)), np.empty(0)
        rng = st.rng
        gi_arr = rng.choice(self.n_pop, size=n_local, p=weights)
        if self.close_mirror and self.dim >= 3 and n_local >= 2:
            gi_arr[1::2] = gi_arr[0:n_local - 1:2][:len(gi_arr[1::2])]
        lq = np.clip(
            (log_f_max - np.log10(st.pop_f[gi_arr] + 1e-10)) / (log_f_spread + 1e-30),
            0.0, 1.0)
        ar = np.minimum(st.pop_age[gi_arr].astype(float) / 5.0, 1.0)
        host_scale = self.host_sigma_min_scale ** (lq * (0.7 + 0.3 * ar))
        noise = rng.standard_normal((n_local, self.dim))
        if self.close_mirror and self.dim >= 3 and n_local >= 2:
            noise[1::2] = -noise[0:n_local - 1:2][:len(noise[1::2])]
        raw_noise = noise.copy()   # kept for the persistent-C stream below
        local_parent_x = st.pop_x[gi_arr].copy()
        sigma_i = st.sigma * host_scale
        if self._cc_dim_gate() > 0.0:
            st.gen_parent_x.append(local_parent_x.copy())
            st.gen_sigma_used.append(sigma_i.copy())
            st.gen_parent_f = st.pop_f[gi_arr].copy()
            st.gen_n_local = n_local
        if self.dim >= 2:
            cov = np.cov(st.pop_x, rowvar=False)
            if isinstance(cov, np.ndarray) and cov.shape == (self.dim, self.dim):
                eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
                floor_eff = self._adaptive_cov_floor(st, eigvals)
                if self._cc_dim_gate() > 0.0:
                    # The adaptive floor is calibrated against C_pop's natural
                    # eigenvalue ratios and would clamp a learned C exactly while
                    # it is learning the extreme anisotropy these functions need.
                    floor_eff = self.cc_cov_floor
                # Axis-alignment signal for the channel router (separability):
                # mean over eigenvectors of their max |component|. EMA-smoothed
                # (same beta as the conditioning EMA). Written only; base never
                # reads it, so base numerics are unchanged.
                align = float(np.mean(np.max(np.abs(eigvecs), axis=0)))
                st.cc_align_ema = (align if st.cc_align_ema is None else
                                   (1.0 - self.cov_ratio_beta) * st.cc_align_ema
                                   + self.cov_ratio_beta * align)
                # Max normalized coordinate gap (second separability signal).
                mgap = 0.0
                for d in range(self.dim):
                    col = np.sort(st.pop_x[:, d])
                    rng = float(col[-1] - col[0])
                    if rng > 1e-300:
                        mgap = max(mgap, float(np.max(np.diff(col)) / rng))
                st.cc_mgap_ema = (mgap if st.cc_mgap_ema is None else
                                  (1.0 - self.cov_ratio_beta) * st.cc_mgap_ema
                                  + self.cov_ratio_beta * mgap)
                mean_eig = float(eigvals.mean())
                if mean_eig > 1e-30:
                    eigvals = eigvals / mean_eig
                else:
                    eigvals = np.ones(self.dim)
                eigvals = np.maximum(eigvals, floor_eff)
                eigvals = eigvals / float(eigvals.mean())  # re-normalize
                # Transform: noise (n_local, dim) @ (eigvecs · √eigvals)ᵀ
                transform = eigvecs * np.sqrt(eigvals)[None, :]
                noise = noise @ transform.T
        gate = self._cc_dim_gate()
        if (gate > 0.0 and self.cc_persist_frac > 0.0
                and st.cc_C is not None and self.dim >= 2 and n_local > 0):
            # Second close-contact stream, shaped by the persistent C. It reuses
            # the SAME standard-normal rows (kept in raw_noise before the C_pop
            # transform), so no extra RNG is drawn and the split is exact.
            n_p = int(round(min(1.0, self.cc_persist_frac) * gate * n_local))
            if n_p > 0:
                ev2, evec2 = np.linalg.eigh(st.cc_C)
                m2 = float(ev2.mean())
                ev2 = ev2 / m2 if m2 > 1e-30 else np.ones(self.dim)
                ev2 = np.maximum(ev2, self.cc_cov_floor)
                ev2 = ev2 / float(ev2.mean())
                noise[-n_p:] = raw_noise[-n_p:] @ (evec2 * np.sqrt(ev2)[None, :]).T
        new_local = self._reflect(local_parent_x + noise * sigma_i[:, None], st.lo, st.hi)
        return new_local, sigma_i

    def _droplet_children(self, st: _MCESOState, n_h2h: int, weights: np.ndarray,
                          elite_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Droplet transmission — host-to-host DE/current-to-best/1:
            x_child = x_parent + F·(x_strain − x_parent) + F·(x_a − x_b)
        The diff term injects population-shape-aware drift (implicit anisotropy
        without a covariance matrix); the strain-pull term accelerates descent
        along narrow valleys. A DE-style binomial crossover then preserves
        per-dim inheritance from the parent (critical on separable problems).
        Returns ``(new_h2h, step_norms)`` where step_norms is the logged σ.
        """
        if n_h2h <= 0:
            return np.empty((0, self.dim)), np.empty(0)
        rng, n = st.rng, self.n_pop
        h2h_parents_gi = rng.choice(n, size=n_h2h, p=weights)
        h2h_a_li = rng.integers(0, n, size=n_h2h)
        h2h_b_li = rng.integers(0, n, size=n_h2h)
        if self.h2h_archive and st.dead_x:
            pool = np.vstack([st.pop_x, np.asarray(st.dead_x)])
            xb = pool[rng.integers(0, len(pool), size=n_h2h)]
            diff = st.pop_x[h2h_a_li] - xb
        else:
            diff = st.pop_x[h2h_a_li] - st.pop_x[h2h_b_li]
        parents_x = st.pop_x[h2h_parents_gi]
        # current-to-best/2 on the committed DROPLET route: a second difference
        # vector adds donor diversity that rescues runs stalled in a wrong ill-
        # conditioned basin. Applied ONLY on the droplet route, so multimodal
        # keep-air functions — which a global second vector regresses — stay
        # bit-identical to base (the extra RNG draw is skipped off-route).
        # droplet_variant="cur2best" pins the single-difference behaviour.
        if self.droplet_variant == "best2_droplet" and st.channel_route == "droplet":
            c = rng.integers(0, n, size=n_h2h)
            d = rng.integers(0, n, size=n_h2h)
            diff = diff + (st.pop_x[c] - st.pop_x[d])
        if len(elite_arr) > 0:
            strain_pos = self._droplet_strain_positions(st, elite_arr, n_h2h)
            best_pull = strain_pos - parents_x
            F = self.h2h_F
            if self.h2h_dither is not None:
                F = np.clip(self.h2h_F + self.h2h_dither * rng.standard_normal((n_h2h, 1)), 0.05, 1.0)
            h2h_step = F * (best_pull + diff)
        else:
            F = self.h2h_F
            if self.h2h_dither is not None:
                F = np.clip(self.h2h_F + self.h2h_dither * rng.standard_normal((n_h2h, 1)), 0.05, 1.0)
            h2h_step = F * diff
        h2h_trial = parents_x + h2h_step
        cr_mask = rng.random((n_h2h, self.dim)) < self.h2h_CR
        forced = rng.integers(0, self.dim, size=n_h2h)
        cr_mask[np.arange(n_h2h), forced] = True
        h2h_offspring = np.where(cr_mask, h2h_trial, st.pop_x[h2h_parents_gi])
        eff_step = h2h_offspring - st.pop_x[h2h_parents_gi]
        h2h_step_norms = np.linalg.norm(eff_step, axis=1)
        if self._cc_dim_gate() > 0.0:
            st.gen_parent_x.append(st.pop_x[h2h_parents_gi].copy())
            st.gen_sigma_used.append(h2h_step_norms.copy())
        new_h2h = self._reflect(h2h_offspring, st.lo, st.hi)
        return new_h2h, h2h_step_norms

    def _airborne_children(self, st: _MCESOState, n_air: int,
                           air_sigma_vec) -> np.ndarray:
        """Airborne transmission — population-independent long-range spread:
            x_child = x_random_host + N(0, σ_air I)
        σ_air inflates as the outbreak clusters (caller-supplied). Suppressed in
        drilling mode (caller sets n_air = 0)."""
        if n_air <= 0:
            return np.empty((0, self.dim))
        rng = st.rng
        air_parents_gi = rng.integers(0, self.n_pop, size=n_air)
        noise_air = (np.clip(rng.standard_cauchy((n_air, self.dim)), -20, 20) if self.air_cauchy
                     else rng.standard_normal((n_air, self.dim)))
        if self._cc_dim_gate() > 0.0:
            st.gen_parent_x.append(st.pop_x[air_parents_gi].copy())
            st.gen_sigma_used.append(np.full(n_air, float(np.max(air_sigma_vec))))
        return self._reflect(st.pop_x[air_parents_gi] + noise_air * air_sigma_vec,
                             st.lo, st.hi)

    # ── σ-regime channel schedule ───────────────────────────────────────────
    def _channel_ratios(self, st: _MCESOState) -> tuple[float, float]:
        """Effective (airborne, droplet) offspring shares for this generation.

        Base (``channel_schedule=False``) reproduces the original policy exactly:
        a flat ``air_ratio`` that is hard-switched to 0 once σ enters drilling
        mode (σ < span × precision_sigma_ratio), droplet flat at ``h2h_ratio``.

        With ``channel_schedule=True`` the **per-landscape channel router** runs
        base keep-air until the ``route_commit_gen`` checkpoint, then commits once
        (and locks for the run) to one of three routes, chosen from two
        f-independent, scale-invariant signals (both EMAs, updated during
        close-contact) evaluated at the checkpoint:

          • ``cond`` = ``cc_logratio_ema`` (log10 λmax/λmin of the population
            covariance) — conditioning;
          • ``algA`` = ``cc_align_ema`` (mean max|component| of the covariance
            eigenvectors) — axis-alignment / separability.

            cond > cond_droplet_thresh (≈2.5)  → DROPLET route: air tapers (σ-ramp)
                                                  and the freed budget → droplet
                                                  (F11–F14 ill-conditioned valleys).
            else algA > align_close_thresh(.98)→ CLOSE route:   air tapers, freed
                                                  budget → close-contact (F04/F16
                                                  separable / axis-aligned).
            otherwise                           → KEEP-AIR route: base ratios
                                                  (multimodal escape untouched).

        Why routing rather than a uniform schedule: all four uniform variants
        (2026-07) lost overall SR@1e-10 because reallocating the air budget the
        *same* way everywhere helps one landscape class and hurts another (droplet
        helps F11–14 but hurts separable F04; cutting air anywhere hurts multimodal
        escape). The diagnostic (docs/history.md) showed ``cond`` cleanly separates
        the droplet class and ``algA`` the close class, while everything else — the
        multimodal/escape functions — is left on the KEEP-AIR default = base
        (bit-identical), so the router only diverges from base where a landscape
        clearly signals it, bounding the risk. The taper reuses the validated
        σ-ramp so a routed function starts at base at full σ and only diverges as it
        drills. Deterministic and structure-driven — no reward bandit (cf. rejected
        MC-ESO-V2a). Signals are ``None`` on the first generation ⇒ KEEP-AIR (base).
        """
        span = st.span
        drilling = st.sigma < span * self.precision_sigma_ratio
        # Base ratios, tapered toward the close-contact-dominant setting the
        # persistent-covariance learner needs once the dimension gate opens.
        air_r, h2h_r = self._cc_channel_ratios()
        air_base = 0.0 if drilling else air_r
        if not self.channel_schedule:
            return air_base, h2h_r
        # Drilling always runs airborne-free (as in base); the route is moot.
        if drilling:
            return air_base, h2h_r
        # Commit the route once, after a warmup, from the stabilized EMAs — then
        # lock it. Conditioning/separability are run-invariant landscape
        # properties, so a single committed route removes the generation-to-
        # generation flip-flop that otherwise perturbs threshold-borderline
        # functions (F04/F14) and even leaks keep-air functions (F06) off base.
        if st.channel_route is None:
            if st.cc_logratio_ema is None or st.cc_align_ema is None:
                return air_base, h2h_r                  # KEEP-AIR (base) while warming
            if st.cc_logratio_ema > self.cond_droplet_early:
                st.channel_route = "droplet"             # early high-cond → ill-cond valley
            elif len(st.history_sigma_global) < self.route_commit_gen:
                return air_base, h2h_r                  # KEEP-AIR (base) until checkpoint
            # Checkpoint: commit from the stabilized EMA values, then lock.
            elif st.cc_logratio_ema > self.cond_droplet_thresh:
                st.channel_route = "droplet"             # ill-conditioned valley
            elif (st.cc_align_ema > self.align_close_thresh
                  and st.cc_mgap_ema is not None
                  and st.cc_mgap_ema > self.close_mgap_thresh):
                st.channel_route = "close"               # separable, wide-gap (F04/F16)
            else:
                st.channel_route = "keepair"             # multimodal / deceptive (base)
        if st.channel_route == "keepair":
            return air_r, h2h_r                          # base ratios (escape untouched)
        # σ-ramp taper (0 at the explore scale span×sigma → 1 at the drilling
        # threshold), reused from the validated variants so a routed function
        # starts at the base ratio at full σ and only diverges as it drills.
        s = math.log10(max(st.sigma, 1e-300) / span)
        s_hi, s_lo = math.log10(self.sigma), math.log10(self.precision_sigma_ratio)
        t = (s_hi - s) / (s_hi - s_lo) if s_hi > s_lo else 1.0
        t = min(1.0, max(0.0, t))
        air_tapered = air_r * (1.0 - t)
        if st.channel_route == "droplet":
            return air_tapered, h2h_r + (air_r - air_tapered)
        # CLOSE route: freed airborne budget → close-contact (implicit).
        return air_tapered, h2h_r

    # ── one generation (μ+λ greedy step) ────────────────────────────────────
    def _reassort_sweep(self, st: _MCESOState) -> None:
        """reassort_sweep: greedy single-coordinate swaps of the best host with
        remembered optima; every evaluation goes through the history."""
        cap = min(int(self.reassort_budget_frac * st.max_evals),
                  st.max_evals - len(st.history_f))
        start = len(st.history_f)
        bi = int(np.argmin(st.pop_f))
        best_x, best_f = st.pop_x[bi].copy(), float(st.pop_f[bi])
        donors = [x for x in st.sol_archive_x] + [x for x in st.ir_archive_x]
        if self.reassort_donors == "all":
            donors += [st.pop_x[i] for i in range(len(st.pop_f)) if i != bi]
        elif self.reassort_donors == "top":
            order = [int(i) for i in np.argsort(st.pop_f) if int(i) != bi]
            donors += [st.pop_x[i] for i in order[:self.reassort_top_hosts]]
        elif self.reassort_donors == "top2":
            # The best few remembered optima and the best few hosts: the answer
            # archive grows to 200 points and a 10D pass over it cost ~850
            # evaluations per sweep (10D F22 swept 4–5 times on small gains).
            k = self.reassort_top_hosts
            mem = sorted(zip(st.sol_archive_f, range(len(st.sol_archive_x))))[:k]
            donors = [st.sol_archive_x[j] for _, j in mem] + [x for x in st.ir_archive_x]
            order = [int(i) for i in np.argsort(st.pop_f) if int(i) != bi]
            donors += [st.pop_x[i] for i in order[:k]]
        if not donors:
            return
        D = np.asarray(donors, dtype=float)
        tol = 1e-9 * st.span
        improved = True
        first_pass = True
        saved = (st.no_improve, st.log_best_ref, st.evals_since_reset)
        while improved and st.budget_left and len(st.history_f) - start < cap:
            improved = False
            per_coord: dict = {}            # coordinate -> (Δf, value) of its best swap
            vals = {}
            for i in range(self.dim):
                for v in np.unique(D[:, i]):
                    if abs(v - best_x[i]) > tol:
                        vals.setdefault(i, []).append(float(v))
            trial_best_x, trial_best_f = None, best_f
            for i, vs in vals.items():
                for v in vs:
                    if not st.budget_left or len(st.history_f) - start >= cap:
                        break
                    x = best_x.copy()
                    x[i] = v
                    f = float(self.func(x))
                    self._record_eval(st, x, f, float("nan"))
                    if f < trial_best_f:
                        trial_best_x, trial_best_f = x, f
                    d = f - best_f
                    # The additivity test needs two coordinates, improving or
                    # not (a run one well away has a single improving one);
                    # with the scan off it keeps the original improving-only rule.
                    if ((d < 0 or self.reassort_scan)
                            and (i not in per_coord or d < per_coord[i][0])):
                        per_coord[i] = (d, v)
            if first_pass and self.reassort_additivity is not None:
                first_pass = False
                top = sorted(per_coord.items(), key=lambda kv: kv[1][0])[:1 + self.reassort_pairs]
                additive = len(top) == 1 + self.reassort_pairs
                if additive:
                    i, (di, vi) = top[0]
                    for j, (dj, vj) in top[1:]:
                        if not st.budget_left:
                            additive = False
                            break
                        x = best_x.copy()
                        x[i], x[j] = vi, vj
                        f = float(self.func(x))
                        self._record_eval(st, x, f, float("nan"))
                        dij = f - best_f
                        if f < trial_best_f:
                            trial_best_x, trial_best_f = x, f
                        if abs(dij - (di + dj)) > self.reassort_additivity * (abs(di) + abs(dj)):
                            additive = False
                            break
                if additive and self.reassort_scan:
                    if trial_best_x is not None and trial_best_f < best_f:
                        best_x, best_f = trial_best_x, trial_best_f
                    # The scan has its own budget (two full passes); the sweep's
                    # cap cut it off after ~4 of 5 coordinates at 5D.
                    scan_start = len(st.history_f)
                    scan_cap = 2 * self.dim * (int(self.reassort_scan) + 61)
                    f_before_scan = best_f
                    best_x, best_f = self._separable_scan(st, best_x, best_f, scan_start, scan_cap)
                    if not self.reassort_scan_retry and not (
                            best_f < f_before_scan and self._meaningful_improvement(
                                best_f, math.log10(max(f_before_scan, 1e-300)),
                                len(st.history_f) - scan_start)):
                        # A scan that gained less than the stagnation slope: the
                        # one-pair additivity test was a false positive (rotated
                        # CEC2022 hybrids pass it at 2% and re-scan ~5200
                        # evaluations at every stall for a 1e-3 gain).
                        st.sweep_dead = True
                    if best_f < float(st.pop_f[bi]):
                        st.pop_x[bi], st.pop_f[bi], st.pop_age[bi] = best_x, best_f, 0
                    return
                if not additive:
                    st.sweep_dead = True
                    if self.reassort_discard:
                        st.no_improve, st.log_best_ref, st.evals_since_reset = saved
                        return
                    if trial_best_x is not None and trial_best_f < float(st.pop_f[bi]):
                        st.pop_x[bi], st.pop_f[bi], st.pop_age[bi] = trial_best_x, trial_best_f, 0
                    return
            if trial_best_x is not None:
                best_x, best_f = trial_best_x, trial_best_f
                improved = True
        if not best_f < float(st.pop_f[bi]):
            st.sweep_dead = True            # reassortment does not spread here
        if best_f < float(st.pop_f[bi]):
            st.pop_x[bi] = best_x
            st.pop_f[bi] = best_f
            st.pop_age[bi] = 0

    def _separable_scan(self, st: _MCESOState, best_x: np.ndarray, best_f: float,
                        start: int, cap: int) -> tuple[np.ndarray, float]:
        """reassort_scan: coordinate-wise 1-D global search (grid + ternary
        refinement) on a landscape the additivity test called separable."""
        lo_v = np.broadcast_to(np.asarray(st.lo, dtype=float), (self.dim,))
        hi_v = np.broadcast_to(np.asarray(st.hi, dtype=float), (self.dim,))
        G = int(self.reassort_scan)

        def ev(x):
            f = float(self.func(x))
            self._record_eval(st, x, f, float("nan"))
            return f

        def room(n):
            return st.budget_left and len(st.history_f) - start + n <= cap

        for _ in range(2):
            changed = False
            for i in range(self.dim):
                if not room(G + 61):
                    return best_x, best_f
                grid = np.linspace(lo_v[i], hi_v[i], G)
                fs = np.empty(G)
                for k, v in enumerate(grid):
                    x = best_x.copy(); x[i] = v
                    fs[k] = ev(x)
                k = int(np.argmin(fs))
                a, b = grid[max(k - 1, 0)], grid[min(k + 1, G - 1)]
                for _ in range(30):
                    m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
                    x1 = best_x.copy(); x1[i] = m1
                    x2 = best_x.copy(); x2[i] = m2
                    if ev(x1) < ev(x2):
                        b = m2
                    else:
                        a = m1
                xi = best_x.copy(); xi[i] = 0.5 * (a + b)
                cand = [(fs[k], grid[k]), (ev(xi), xi[i])]
                fbest_i, v_i = min(cand, key=lambda t: t[0])
                if fbest_i < best_f:
                    best_x = best_x.copy(); best_x[i] = v_i
                    best_f = fbest_i
                    changed = True
            if not changed:
                break
        return best_x, best_f

    def _recombination_wave(self, st: _MCESOState, sigma0: "float | None" = None,
                            budget_frac: "float | None" = None,
                            lam_mult: "float | None" = None,
                            restart: bool = False, follow: bool = False) -> None:
        """hd_wave_lambda: a (μ/μ_w, λ) weighted-recombination wave (the cma
        package's CMA-ES) from the best host; every evaluation goes through the
        history. Its best replaces the worst host."""
        import cma
        frac = self.wave_budget_frac if budget_frac is None else budget_frac
        cap = min(int(frac * st.max_evals), st.max_evals - len(st.history_f))
        if lam_mult is None:
            # hd_wave_lambda > 0: λ = that × D; < 0: −k × the CMA-ES default
            # 4 + ⌊3 ln D⌋ (as wave_handover_lambda). 2026-10-13: λ = 4·D = 80 at
            # 20D leaves ~150 generations in the stall wave's 25% share and
            # does not get through Rosenbrock's curved valley (F08 error 5-12).
            lam = (max(4, int(round(self._wave_lambda * self.dim))) if self._wave_lambda > 0
                   else int(round(-self._wave_lambda * (4 + int(3 * math.log(self.dim))))))
        elif lam_mult == 0:
            lam = 4 + int(3 * math.log(self.dim))
        elif lam_mult < 0:
            lam = int(round(-lam_mult * (4 + int(3 * math.log(self.dim)))))
        else:
            lam = max(4, int(round(lam_mult * self.dim)))
        if self.wave_plain:
            lam = 4 + int(3 * math.log(self.dim))
        if cap < 2 * lam:
            return
        lo, hi = float(np.min(st.lo)), float(np.max(st.hi))
        x0 = st.pop_x[int(np.argmin(st.pop_f))].copy()
        s0 = self.wave_sigma_ratio * st.span if sigma0 is None else float(sigma0)
        L = None
        if (self.wave_use_C or sigma0 is not None) and st.cc_C is not None and self._update_cc_ill(st):
            ev, V = np.linalg.eigh(st.cc_C)
            ev = np.maximum(ev / max(float(ev.mean()), 1e-30), self.cc_cov_floor)
            L = V * np.sqrt(ev)[None, :]
        if self.wave_plain:
            L, s0 = None, 0.2 * float(np.mean(st.span))
        start = len(st.history_f)
        best_x, best_f = None, float("inf")
        best_before = st.best_so_far
        while True:
            opts = {"popsize": lam, "verbose": -9, "seed": int(st.rng.integers(1, 2 ** 31 - 1)),
                    "tolfun": 1e-15, "tolx": 1e-16, "tolfunhist": 0,
                    "maxfevals": cap - (len(st.history_f) - start)}
            if L is None:
                opts["bounds"] = [lo, hi]
                es = cma.CMAEvolutionStrategy(x0, s0, opts)
            else:
                es = cma.CMAEvolutionStrategy(np.zeros(self.dim), s0, opts)
            while not es.stop() and st.budget_left and len(st.history_f) - start + lam <= cap:
                X = es.ask()
                F = []
                for x in X:
                    x = np.asarray(x, dtype=float)
                    if L is not None:
                        x = x0 + L @ x
                    x = np.clip(x, st.lo, st.hi)
                    f = float(self.func(x))
                    self._record_eval(st, x, f, float(es.sigma))
                    F.append(f)
                    if f < best_f:
                        best_x, best_f = x.copy(), f
                    if not st.budget_left:
                        break
                if len(F) < len(X):
                    break
                es.tell(X, F)
                if (follow and len(st.history_f) - start + lam > cap
                        and best_f < best_before and not st.follow_done
                        and (not callable(follow) or follow(st, start, best_f))):
                    st.follow_done = True
                    st.trace_events.append((len(st.history_f), "follow"))
                    cap = st.max_evals - start
                    es.opts["maxfevals"] = cap
            lam *= 2
            if (not restart or best_x is None or not st.budget_left
                    or len(st.history_f) - start + 2 * lam > cap):
                break
            x0 = best_x.copy()
        if best_x is not None:
            w = int(np.argmax(st.pop_f))
            if best_f < st.pop_f[w]:
                st.pop_x[w] = best_x
                st.pop_f[w] = best_f
                st.pop_age[w] = 0

    def _ema_rate(self, old: float, before: float, after: float, n: int) -> float:
        lg = lambda v: math.log10(max(v, 1e-300))
        r = max(0.0, lg(before) - lg(after)) / max(n, 1)
        return (1 - self.strain_ema) * old + self.strain_ema * r

    def _strain_step(self, st: _MCESOState) -> bool:
        """strain_compete: run CMA-strain generations until its share of the
        evaluations since the start matches its reproduction-rate share.
        Returns True if the budget ran out."""
        import cma
        if self.strain_stop_on_handover and st.handover_done:
            return not st.budget_left
        rm, rc = st.outbreak_rate, st.strain_rate
        if (self.strain_idle == "explore" and st.strain_evals > 0
                and len(st.history_f) - st.record_eval > self._stagnation_window()):
            rm = rc = 0.0                       # no new record for a stagnation window
        if rm + rc <= 0:
            if st.strain_evals == 0:
                share = 0.5 if self.strain_init is None else self.strain_init
            elif self.strain_idle == "explore":
                share = 0.0                     # clipped up to the lower bound below
            else:
                share = st.strain_evals / max(len(st.history_f) - st.strain_start, 1)
        else:
            share = rc / (rm + rc)
        if not self.strain_dim_gate:
            g = 1.0
        elif self.strain_dim_ref is None:
            g = self._cc_dim_gate()
        else:
            g = float(min(1.0, max(0.0, (self.dim / 2.0 - 1.0) / max(1e-9, self.strain_dim_ref / 2.0 - 1.0))))
        hi = (1 - self.strain_min if self.strain_max is None else self.strain_max) * g
        share = min(hi, max(self.strain_min * g, share))
        if st.strain_owns:
            share = 1.0
        elif self.strain_retire_ill and st.cc_ill:
            share = 0.0
        if (self.strain_takeover and st.strain_record_eval > st.outbreak_record_eval
                and len(st.history_f) - st.outbreak_record_eval > self._stagnation_window()):
            share = 1.0                         # the strain has taken over
        used = len(st.history_f) - st.strain_start
        while st.budget_left and st.strain_evals < share * used:
            if st.strain_es is None or st.strain_es.stop():
                base = 4 + int(3 * math.log(self.dim))
                st.strain_lam = (max(4, int(round(self.strain_lam_mult * base))) if st.strain_es is None
                                 else 2 * st.strain_lam)
                bi = int(np.argmin(st.pop_f))
                s0 = 0.2 * st.span if st.strain_es is None else max(st.sigma, 1e-3 * st.span)
                st.strain_es = cma.CMAEvolutionStrategy(
                    st.pop_x[bi].copy(), s0,
                    {"popsize": st.strain_lam, "verbose": -9, "bounds": [float(np.min(st.lo)), float(np.max(st.hi))],
                     "seed": int(st.rng.integers(1, 2 ** 31 - 1)), "tolfun": 1e-15, "tolx": 1e-16,
                     "tolfunhist": 0})
                st.trace_events.append((len(st.history_f), "strain"))
            es = st.strain_es
            X = es.ask()
            F = []
            n0 = len(st.history_f)
            record_before = st.best_so_far
            best_before = es.best.f if es.best.f is not None else float("inf")
            for x in X:
                x = np.clip(np.asarray(x, dtype=float), st.lo, st.hi)
                f = float(self.func(x))
                self._record_eval(st, x, f, float(es.sigma))
                F.append(f)
                if not st.budget_left:
                    break
            self._pad_eval_channel(st, _CH_LS)
            st.strain_evals += len(st.history_f) - n0
            if st.best_so_far < record_before:
                st.strain_record_eval = len(st.history_f)
            used = len(st.history_f) - st.strain_start
            if len(F) < len(X):
                return True
            es.tell(X, F)
            gb = float(min(F))
            if self.strain_credit == "record":
                st.strain_rate = self._ema_rate(st.strain_rate, record_before, st.best_so_far, len(F))
            else:
                first = float(np.max(F)) if best_before == float("inf") else min(best_before, float(np.max(F)))
                st.strain_rate = self._ema_rate(st.strain_rate, first, min(gb, best_before), len(F))
            # Immigration: the strain's best replaces the outbreak's worst host.
            w = int(np.argmax(st.pop_f))
            j = int(np.argmin(F))
            if self.strain_immigrate and F[j] < st.pop_f[w]:
                st.pop_x[w] = np.clip(np.asarray(X[j], dtype=float), st.lo, st.hi)
                st.pop_f[w] = F[j]
                st.pop_age[w] = 0
        return not st.budget_left

    def _probe_outpaces(self, st: _MCESOState, start: int, probe_best: float) -> bool:
        """probe_at: did the probe gain more decades per evaluation than the
        outbreak over the same number of evaluations just before it?"""
        n = len(st.history_f) - start
        lg = lambda v: math.log10(max(v, 1e-300))
        at_start = float(np.min(st.history_f[:start]))
        before = float(np.min(st.history_f[:max(1, start - n)]))
        return (lg(at_start) - lg(probe_best)) / max(n, 1) > (lg(before) - lg(at_start)) / max(n, 1)

    def _update_cc_ill(self, st: _MCESOState) -> bool:
        """Latch: has the learned covariance reached 10^router_learned_cond?"""
        if not st.cc_ill and st.cc_C is not None:
            ev = np.linalg.eigvalsh(st.cc_C)
            st.cc_ill = bool(ev[0] > 0 and math.log10(ev[-1] / ev[0]) >= self.router_learned_cond)
        if (not st.cc_ill and self.strain_share_shape and st.strain_es is not None
                and st.cc_C is not None):
            C = np.asarray(st.strain_es.C, dtype=float)
            ev = np.linalg.eigvalsh(C)
            if ev[0] > 0 and math.log10(ev[-1] / ev[0]) >= self.router_learned_cond:
                st.cc_C = C / float(np.mean(ev))
                st.cc_ill = True
                st.trace_events.append((len(st.history_f), "shape"))
        return st.cc_ill

    def _shrink_population(self, st: _MCESOState) -> None:
        """Linear population-size reduction: drop the worst hosts down to the
        size scheduled for the evaluations spent so far."""
        frac = len(st.history_f) / max(st.max_evals, 1)
        if st.sched_n0:
            frac = ((len(st.history_f) - st.sched_start)
                    / max(st.max_evals - st.sched_start, 1))
            n0 = st.sched_n0
            target = int(round(self._pop_final
                               + (n0 - self._pop_final) * (1.0 - frac) ** self.pop_shrink_power))
            target = max(self._pop_final, min(self.n_pop, target))
            if target < self.n_pop:
                keep = np.sort(np.argsort(st.pop_f)[:target])
                st.pop_x, st.pop_f, st.pop_age = st.pop_x[keep], st.pop_f[keep], st.pop_age[keep]
                self.n_pop = target
            return
        if self.pop_shrink_power == 1.0:
            # Exactly the formula the canonical pop_lin* runs used.
            target = int(round(self._n_pop0 + (self._pop_final - self._n_pop0) * frac))
        else:
            target = int(round(self._pop_final
                               + (self._n_pop0 - self._pop_final)
                               * (1.0 - frac) ** self.pop_shrink_power))
        target = max(self._pop_final, min(self.n_pop, target))
        if target >= self.n_pop:
            return
        keep = np.sort(np.argsort(st.pop_f)[:target])
        st.pop_x = st.pop_x[keep]
        st.pop_f = st.pop_f[keep]
        st.pop_age = st.pop_age[keep]
        self.n_pop = target

    def _run_generation(self, st: _MCESOState) -> None:
        """One outbreak generation: select strains + the worst-K hosts to kill,
        spawn offspring across the three channels, evaluate them into the dead
        slots with greedy rollback, age survivors, and adapt σ_global."""
        if self.pop_schedule == "linear":
            self._shrink_population(st)
        span, n = st.span, self.n_pop
        gen_best_before = st.best_so_far  # snapshot for σ adaptation

        # Strain coexistence: niched-elite pool for the droplet channel's
        # current-to-best pull. Spatial diversity also feeds the airborne
        # channel's σ modulator (denser cluster → larger aerosol jumps).
        pop_diversity = np.mean(np.std(st.pop_x, axis=0) / span)
        diversity_ratio = float(np.clip(pop_diversity / 0.289, 0.0, 1.0))
        elite_global = self._niche_elites(st.pop_x, st.pop_f, st.niche_radius)
        elite_arr = (np.fromiter(elite_global, dtype=int)
                     if elite_global else np.empty(0, dtype=int))
        st.last_n_elite = len(elite_global)

        # Host competition (μ+λ greedy): each gen, the worst K = kill_fraction · n
        # hosts die (the global best always survives by not being in worst-K).
        # Children that fail to outcompete the host they replaced are rolled back.
        n_kill = max(1, min(n, int(round(self.kill_fraction * n))))
        dead_global = np.argsort(st.pop_f)[::-1][:n_kill]
        if self.recover_age is not None and self.dim >= 3 and n_kill >= 2:
            order = np.argsort(st.pop_f)
            protect = set(int(i) for i in order[:max(1, n // 5)])
            old = [int(i) for i in np.argsort(-st.pop_age)
                   if st.pop_age[i] >= self.recover_age and int(i) not in protect]
            old = old[:n_kill // 2]
            if old:
                rest = [int(i) for i in order[::-1] if int(i) not in old]
                dead_global = np.array(old + rest[:n_kill - len(old)])
        n_dead = len(dead_global)
        if n_dead > 0:
            dead_orig_x = st.pop_x[dead_global].copy()
            dead_orig_f = st.pop_f[dead_global].copy()
        else:
            dead_orig_x = None
            dead_orig_f = None

        if n_dead == 0:
            st.gen_counts = (0, 0, 0, 0, 0)
            st.pop_age += 1
            # No offspring this gen → no σ signal, so σ is held unchanged. When
            # all individuals are elite no births occur and history_f never grows
            # → still advance no_improve so the spillover fires.
            st.no_improve += 1
            return

        weights = self._softmax_weights(st.pop_f)

        # 3-channel split: close-contact (local Gaussian), droplet (h2h
        # DE/current-to-best), airborne (random spread). Airborne is pure noise —
        # the σ-regime schedule (_channel_ratios) tapers it off as σ contracts so
        # precision grinding isn't disrupted (reaching 0 in drilling mode).
        air_ratio_eff, h2h_ratio_eff = self._channel_ratios(st)
        n_air = max(0, int(round(air_ratio_eff * n_dead)))
        n_h2h = max(0, int(round(h2h_ratio_eff * n_dead))) if n >= 3 else 0
        # If rounding overflows, trim airborne first (preserves the
        # droplet/close-contact intent).
        if n_air + n_h2h > n_dead:
            n_air = max(0, n_dead - n_h2h)
        n_local = n_dead - n_air - n_h2h

        # The momentum / migratory slots (always 0) keep the trace layout.
        st.gen_counts = (n_local, n_h2h, n_air, 0, 0)

        # Log-scale quality anchored to the global (history-wide) best. When the
        # population converges to a local optimum, all f_i ≈ f_pop_max but
        # best_so_far may be far better → lq ≈ 0 → σ_i = σ_global (full exploration).
        pop_log_f = np.log10(st.pop_f + 1e-10)
        log_f_max = float(pop_log_f.max())
        log_f_best = float(np.log10(st.best_so_far + 1e-10))  # anchored to history best
        log_f_spread = log_f_max - log_f_best

        # Airborne σ: large when converged (need to escape), small when diverse.
        air_sigma_factor = 1.5 + self.air_sigma_amplifier * (1.0 - diversity_ratio)
        air_sigma_base = np.maximum(st.sigma, st.sigma_init * 0.3)
        air_sigma_vec = air_sigma_base * air_sigma_factor

        # Batch generate all children (channel order fixes the RNG draw sequence)
        if self._cc_dim_gate() > 0.0:
            st.gen_parent_x = []
            st.gen_sigma_used = []
            st.gen_n_local = 0
            if st.cc_C is None:
                st.cc_C = np.eye(self.dim)
        new_local, sigma_i = self._close_contact_children(
            st, n_local, weights, log_f_max, log_f_spread)
        new_h2h, h2h_step_norms = self._droplet_children(st, n_h2h, weights, elite_arr)
        new_air = self._airborne_children(st, n_air, air_sigma_vec)
        new_xs = np.concatenate([new_local, new_h2h, new_air], axis=0)

        # Per-child sigma: local→σ_i, h2h→|step|, air→air_sigma_vec
        _sc: list[np.ndarray] = []
        if n_local > 0:
            _sc.append(sigma_i)
        if n_h2h > 0:
            _sc.append(h2h_step_norms)
        if n_air > 0:
            _sc.append(np.full(n_air, float(air_sigma_vec)))
        _sigma_children = np.concatenate(_sc) if _sc else np.array([])

        # Evaluate offspring and resolve host competition (placement + rollback +
        # aging). Factored into a hook so niching variants can swap the *global*
        # competition rule for a local (crowding) one without touching channels.
        st.gen_eval_start = len(st.history_f)
        self._place_and_compete(
            st, new_xs, _sigma_children, n_dead,
            dead_global, dead_orig_x, dead_orig_f)
        # Channel of each child evaluated this generation (children are evaluated
        # in the concatenation order above; the budget may stop it early).
        codes = bytearray()
        for code, cnt in enumerate(st.gen_counts):
            codes.extend([code] * cnt)
        st.eval_channel.extend(codes[:len(st.history_f) - st.gen_eval_start])
        self._pad_eval_channel(st, _CH_UNKNOWN)
        if (self.forecast and self.dim >= 3 and st.budget_left
                and (not self.forecast_route or st.channel_route == "droplet")):
            self._forecast_step(st)

        self._adapt_sigma(st, gen_best_before)

    def _chain_trace(self, st: _MCESOState) -> None:
        """chain_trace: golden-section line search from the best host along the
        learned C's top-2 eigenvectors, bracket ±σ·√λ·4."""
        ev, V = np.linalg.eigh(st.cc_C)
        bi = int(np.argmin(st.pop_f))
        x0, f0 = st.pop_x[bi].copy(), float(st.pop_f[bi])
        phi = (math.sqrt(5) - 1) / 2
        for k in (-1, -2):
            d = V[:, k] * math.sqrt(max(float(ev[k]), 1e-12))
            h = 4.0 * st.sigma
            a, b = -h, h
            best_t, best_f = 0.0, f0
            for _ in range(20):
                if not st.budget_left:
                    return
                c, e = b - phi * (b - a), a + phi * (b - a)
                fs = []
                for t in (c, e):
                    x = self._reflect(x0 + t * d, st.lo, st.hi)
                    f = float(self.func(x))
                    self._record_eval(st, x, f, abs(t))
                    fs.append(f)
                    if f < best_f:
                        best_t, best_f = t, f
                if fs[0] < fs[1]:
                    b = e
                else:
                    a = c
            if best_f < f0:
                x0 = self._reflect(x0 + best_t * d, st.lo, st.hi)
                f0 = best_f
        if f0 < float(st.pop_f[bi]):
            st.pop_x[bi], st.pop_f[bi], st.pop_age[bi] = x0, f0, 0

    def _forecast_step(self, st: _MCESOState) -> None:
        """forecast: one child at the minimiser of a separable quadratic fitted
        to the better half of the hosts in the learned C's eigenbasis."""
        n, D = len(st.pop_f), self.dim
        m = max(2 * D + 2, n // 2)
        if n < 2 * D + 2:
            return
        if self.forecast_hist:
            w = min(len(st.history_f), 20 * D)
            HX = np.asarray(st.history_x[-w:]); HF = np.asarray(st.history_f[-w:])
            m = min(len(HF), 4 * D)
            if m < 2 * D + 2:
                return
            idx = np.argsort(HF)[:m]
            X, F = HX[idx], HF[idx]
        else:
            idx = np.argsort(st.pop_f)[:m]
            X, F = st.pop_x[idx], st.pop_f[idx]
        xb = X[0]
        if st.cc_C is not None:
            _, V = np.linalg.eigh(st.cc_C)
        else:
            V = np.eye(D)
        Y = (X - xb) @ V
        sc = np.maximum(Y.std(axis=0), 1e-300)
        Z = Y / sc
        A = np.hstack([np.ones((m, 1)), Z, Z ** 2])
        try:
            coef, *_ = np.linalg.lstsq(A, F, rcond=None)
        except np.linalg.LinAlgError:
            return
        b, c = coef[1:D + 1], coef[D + 1:]
        z = np.where(c > 0, -b / (2 * np.where(c > 0, c, 1.0)), 0.0)
        y = z * sc
        r = float(np.max(np.linalg.norm(Y, axis=1)))
        ny = float(np.linalg.norm(y))
        if not np.isfinite(ny) or ny <= 0:
            return
        if ny > r:
            y *= r / ny
        x = self._reflect(xb + V @ y, st.lo, st.hi)
        f = float(self.func(x))
        self._record_eval(st, x, f, float(np.linalg.norm(y)))
        st.eval_channel.append(_CH_LS)
        w = int(np.argmax(st.pop_f))
        if f < st.pop_f[w]:
            st.pop_x[w], st.pop_f[w], st.pop_age[w] = x, f, 0

    def _adapt_sigma(self, st: _MCESOState, gen_best_before: float) -> None:
        """σ_global adaptation (always on): improved → × sigma_up, else
        × sigma_down (or sigma_drill_down once drilling). Overridable so niching
        variants can use a monotone-contraction schedule that drills every
        occupied basin to FP precision instead of letting cross-basin
        improvements pump σ back up."""
        span = st.span
        in_drilling = st.sigma < span * self.precision_sigma_ratio
        sigma_floor_eff = span * self.sigma_floor_ratio
        if self.sigma_floor_ill_ratio is not None and self._update_cc_ill(st):
            sigma_floor_eff = span * self.sigma_floor_ill_ratio
        improved = st.best_so_far < gen_best_before
        if (self.hd_sigma_success is not None and self.dim >= 3
                and st.gen_parent_f is not None and st.gen_n_local > 0):
            f = np.asarray(st.history_f[st.gen_eval_start:st.gen_eval_start + st.gen_n_local])
            m = min(len(f), len(st.gen_parent_f))
            if m > 0:
                improved = float(np.mean(f[:m] < st.gen_parent_f[:m])) > self.hd_sigma_success
        k = 1.0
        if self.sigma_batch is not None and self.dim >= 3:
            k = max(1.0, (len(st.history_f) - st.gen_eval_start) / self.sigma_batch)
        if improved:
            st.sigma *= self.sigma_up ** k
        else:
            st.sigma *= (self.sigma_drill_down if in_drilling else self.sigma_down) ** k
        st.sigma = max(sigma_floor_eff, min(st.sigma, span * self.sigma_ceil_ratio))

    # ── host competition (overridable for niching variants) ──────────────────
    def _record_eval(self, st: _MCESOState, x: np.ndarray, f: float,
                     sigma_used: float) -> None:
        """Log one offspring evaluation and update best_so_far / no_improve.

        Shared by every competition policy so the stagnation/spillover signal is
        identical regardless of how offspring are placed into the population.
        """
        st.history_x.append(x.copy())
        st.history_f.append(f)
        st.history_sigma_eval.append(sigma_used)
        st.evals_since_reset += 1
        paced_step = False
        if st.paced:
            st.pace_since += 1
            if f < st.basin_best and self._meaningful_improvement(f, st.pace_ref, st.pace_since):
                paced_step = True
                if self.pace_horizon is not None:
                    lf = math.log10(max(f, 1e-300))
                    rate = (st.pace_ref - lf) / max(st.pace_since, 1)
                    gap = lf - math.log10(max(st.best_so_far, 1e-300))
                    left = st.max_evals - len(st.history_f)
                    paced_step = rate > 0 and gap <= rate * self.pace_horizon * left
                st.pace_ref = math.log10(max(f, 1e-300))
                st.pace_since = 0
        if f < st.basin_best:
            st.basin_best = f
        if f < st.best_so_far:
            st.paced = False
            st.best_so_far = f
            st.record_eval = len(st.history_f)
            if self._meaningful_improvement(f, st.log_best_ref, st.evals_since_reset):
                st.no_improve = 0
                st.wave_spent = False
                st.sweep_spent = False
                st.trace_spent = False
                st.log_best_ref = math.log10(max(f, 1e-300))
                st.evals_since_reset = 0
            else:
                st.no_improve += 1
        else:
            st.no_improve = 0 if paced_step else st.no_improve + 1

    def _place_and_compete(
        self, st: _MCESOState, new_xs: np.ndarray, sigma_children: np.ndarray,
        n_dead: int, dead_global: np.ndarray,
        dead_orig_x: np.ndarray | None, dead_orig_f: np.ndarray | None,
    ) -> None:
        """Global host competition (μ+λ greedy): place each child into its dead
        slot, then roll back any child that failed to beat the host it replaced
        so the outbreak monotonically improves toward the single best basin.

        Niching variants override this with a *local* (crowding) rule so
        spatially separated basins survive simultaneously.
        """
        replaced_slots: list[int] = []
        for k in range(min(n_dead, len(new_xs))):
            slot = int(dead_global[k])
            x = new_xs[k]
            f = float(self.func(x))
            st.pop_x[slot] = x
            st.pop_f[slot] = f
            st.pop_age[slot] = 0
            replaced_slots.append(slot)
            sigma_used_k = (float(sigma_children[k])
                            if k < len(sigma_children) else float(st.sigma))
            self._record_eval(st, x, f, sigma_used_k)
            if not st.budget_left:
                break

        # Host-competition rollback: children worse than the host they replaced.
        survived: list = []
        if dead_orig_x is not None:
            for k, slot in enumerate(replaced_slots):
                if dead_orig_f[k] < st.pop_f[slot]:
                    st.pop_x[slot] = dead_orig_x[k]
                    st.pop_f[slot] = dead_orig_f[k]
                    continue
                if self.h2h_archive:
                    st.dead_x.append(dead_orig_x[k].copy())
                    if len(st.dead_x) > self.n_pop:
                        st.dead_x.pop(int(st.rng.integers(0, len(st.dead_x))))
                if self._cc_dim_gate() > 0.0:
                    survived.append((k, st.pop_x[slot].copy(), st.pop_f[slot]))
        if (self._cc_dim_gate() > 0.0
                and len(st.history_sigma_global) >= st.cc_freeze_until):
            self._update_cc_cov(st, survived)

        # Age active survivors (per-generation).
        replaced_mask = np.zeros(self.n_pop, dtype=bool)
        if replaced_slots:
            replaced_mask[replaced_slots] = True
        st.pop_age[~replaced_mask] += 1

    def _update_cc_cov(self, st: _MCESOState, survived: list) -> None:
        """Rank-μ update of the persistent close-contact covariance.

        Only infections that took hold contribute, and each contributes the
        direction it travelled scaled by the σ it was drawn with, so the matrix
        learns shape and not scale:

            C ← (1−c)·C + c · mean( y yᵀ ),   y = (x_child − x_parent) / σ_used

        The identity C starts from decays at rate c, so with a small c the matrix
        keeps full rank while accumulating the valley geometry — the property the
        instantaneous population covariance cannot have. Renormalised to mean
        eigenvalue 1 so σ keeps its meaning.
        """
        if not survived or st.cc_C is None or not st.gen_parent_x:
            return
        parents = np.concatenate(st.gen_parent_x, axis=0)
        sigmas = np.concatenate(st.gen_sigma_used, axis=0)
        cand_y = []
        cand_f = []
        cand_beat = []
        pf = st.gen_parent_f
        for k, child_x, child_f in survived:
            # Only close-contact children are drawn from C; droplet and airborne
            # follow different distributions and would bias the estimate.
            if k >= st.gen_n_local or k >= len(parents) or k >= len(sigmas):
                continue
            sig = float(sigmas[k])
            if sig <= 1e-300:
                continue
            # Historical rule: the step must be a genuine success — better than the
            # host that spawned it — because the placement test only asks it to beat
            # a worst-quartile host, which is not a success criterion (measured
            # acceptance 0.47). cc_mu_frac > 0 replaces this test by a rank cut over
            # the whole generation (below); see the constructor note.
            beat = not (pf is not None and k < len(pf) and not (child_f < pf[k]))
            cand_y.append((child_x - parents[k]) / sig)
            cand_f.append(child_f)
            cand_beat.append(beat)
        if not cand_y:
            return
        if self.cc_mu_frac > 0.0:
            # Top-μ of this generation's placed close-contact children, by f.
            # Selection only, in the generation's own order so the estimate does
            # not depend on the sort being stable.
            mu = max(1, int(math.ceil(min(1.0, self.cc_mu_frac) * len(cand_f))))
            keep = set(int(i) for i in np.argsort(np.asarray(cand_f))[:mu])
            sel = [i for i in range(len(cand_f)) if i in keep]
        else:
            sel = [i for i in range(len(cand_f)) if cand_beat[i]]
        ys = [cand_y[i] for i in sel]
        if not ys:
            return
        Y = np.asarray(ys)
        rank_mu = (Y.T @ Y) / len(Y)
        c = min(1.0, self.cc_learning_rate)
        C = (1.0 - c) * st.cc_C + c * rank_mu
        if self.cc_chain:
            n = self.dim
            mu = max(1, len(st.pop_f) // 4)
            center = np.mean(st.pop_x[np.argsort(st.pop_f)[:mu]], axis=0)
            if st.cc_center is not None and st.sigma > 0:
                cc = 4.0 / (n + 4.0)                       # CMA-ES's c_c (μ_eff ≈ μ)
                c1 = 2.0 / ((n + 1.3) ** 2 + mu)           # CMA-ES's c_1
                drift = (center - st.cc_center) / st.sigma
                if st.cc_chain is None:
                    st.cc_chain = np.zeros(n)
                st.cc_chain = (1 - cc) * st.cc_chain + math.sqrt(cc * (2 - cc) * mu) * drift / math.sqrt(mu)
                C = (1.0 - c1) * C + c1 * np.outer(st.cc_chain, st.cc_chain)
            st.cc_center = center
        tr = float(np.trace(C))
        if tr > 1e-300 and np.all(np.isfinite(C)):
            st.cc_C = C * (self.dim / tr)      # mean eigenvalue 1
            if self.spill_adapt_veto is not None:
                ev = np.linalg.eigvalsh(st.cc_C)
                if ev[0] > 0:
                    st.cc_cond_hist.append((len(st.history_f), math.log10(ev[-1] / ev[0])))

    # ── per-generation dynamics recording ───────────────────────────────────
    def _record_generation(self, st: _MCESOState) -> None:
        """Append one row of generation-level dynamics (population always = n_pop)."""
        st.history_pop_sigma.append(
            float(st.sigma) * self._host_sigma_scale(st.pop_f, st.pop_age))
        st.history_pop.append(st.pop_x.copy())
        st.history_sigma_global.append(float(st.sigma))
        st.history_n_elite.append(st.last_n_elite)
        st.history_no_improve.append(int(st.no_improve))
        st.history_eval_count.append(len(st.history_f))
        self._record_trace(st)

    def _record_trace(self, st: _MCESOState) -> None:
        """One row of internals for the results UI. Reads state only."""
        t = st.trace_gen
        if not t:
            return
        cc_cond = float("nan")
        if st.cc_C is not None:
            ev = np.linalg.eigvalsh(st.cc_C)
            if ev[0] > 0:
                cc_cond = float(np.log10(ev[-1] / ev[0]))
        route = st.channel_route
        row = (len(st.history_f), int(self.n_pop), float(st.sigma), float(st.best_so_far),
               int(st.last_n_elite), int(st.no_improve),
               int(st.sigma < st.span * self.precision_sigma_ratio),
               TRACE_ROUTES.index(route) if route in TRACE_ROUTES else 0,
               float("nan") if st.cc_logratio_ema is None else float(st.cc_logratio_ema),
               float("nan") if st.cc_align_ema is None else float(st.cc_align_ema),
               float("nan") if st.cc_mgap_ema is None else float(st.cc_mgap_ema),
               *(int(c) for c in st.gen_counts), cc_cond)
        for k, v in zip(TRACE_GEN_KEYS, row):
            t[k].append(v)
