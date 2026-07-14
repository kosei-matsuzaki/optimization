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
    ir_basin_centroids: list[np.ndarray] = field(default_factory=list)  # abandoned-basin memory

    @property
    def budget_left(self) -> bool:
        return len(self.history_f) < self.max_evals


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
        # None → dimension-aware default max(20, 4·dim): 20 up to dim 5, then
        # scales up (e.g. 40 at dim 10). A fixed 20 underfills the population in
        # higher dimensions — at dim 10 it let niching restarts wander badly on
        # CEC2022 G06-Hybrid1 (best_f 2140 → 40 once n_pop reached 40). Low-dim
        # (BBOB dim 2/3) is unchanged. Pass an int to override.
        n_pop: "int | None" = None,
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
        # ── Migratory (vector-borne) channel: stuck-gated structured escape ──
        # A 4th transmission channel modelling a carrier moving the pathogen to a
        # distant region. It fires ONLY when a run is stuck — drilled in (σ below
        # the drilling threshold) yet stagnating (no_improve ≥ threshold) — i.e.
        # precision has already stalled, so diverting some offspring costs no
        # productive precision evals. Offspring are structured long jumps from the
        # current best: half along the population's principal covariance axis
        # (valley/ridge escape for F13/F14), half isotropic (multimodal escape for
        # F18/F23), at migratory_jump_ratio × span. Because offspring go through
        # μ+λ greedy rollback and best_f is the min over all evals, this channel
        # can only lower best_f or be ignored — it CANNOT reduce SR@1e-10. Off by
        # default (bit-identical to base when disabled — no offspring, no RNG draws).
        migratory_channel: bool = False,
        migratory_ratio: float = 0.34,         # share of a stuck gen's offspring
        # Stagnation gate. Kept BELOW the spillover threshold (300) so migratory
        # fires before the population is reset, but well above the transient
        # stagnation of a still-recovering run (e.g. F14's delayed breakthrough) so
        # those runs are not perturbed — the loose 100 regressed F14/F17/F19.
        migratory_no_improve_thresh: int = 200,
        migratory_jump_ratio: float = 0.2,     # jump magnitude, × span
        # ── Greedy (μ+λ) replacement ───────────────────────────────────
        kill_fraction: float = 0.25,           # fraction of active killed per gen (by f)
        # ── Restart on stagnation ──────────────────────────────────────
        restart_no_improve_threshold: int = 300,  # no_improve count that triggers restart
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
        # ── σ adaptation ──────────────────────────────────────────────
        # Multiplicative step-size adaptation. Once σ < span × precision_sigma_ratio
        # the contraction switches to sigma_drill_down to drill to FP precision.
        sigma_up: float = 1.1,                 # gentle expansion when improving
        sigma_down: float = 0.95,              # gentle contraction when not
        sigma_floor_ratio: float = 1e-6,       # σ_global absolute floor (× span)
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
        ir_reignite_sigma_ratio: float = 0.05,
        ir_repel_radius_ratio: float = 0.1,
        ir_repel_max_tries: int = 20,
    ):
        super().__init__(benchmark, seed)
        # Dimension-aware population: fixed 20 underfills high-dim search.
        self.n_pop = n_pop if n_pop is not None else max(20, 4 * self.dim)
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
        self.migratory_channel = migratory_channel
        self.migratory_ratio = migratory_ratio
        self.migratory_no_improve_thresh = migratory_no_improve_thresh
        self.migratory_jump_ratio = migratory_jump_ratio
        self.log_slope_threshold = log_slope_threshold
        self.h2h_ratio = h2h_ratio
        self.h2h_F = h2h_F
        self.kill_fraction = kill_fraction
        self.restart_no_improve_threshold = restart_no_improve_threshold
        self.restart_sigma_ratio = restart_sigma_ratio
        self.restart_quality_rel_floor = restart_quality_rel_floor
        self.basin_switch_after_failed_spillovers = basin_switch_after_failed_spillovers
        self.basin_switch_quality_rel_floor = basin_switch_quality_rel_floor
        self.exhausted_sigma_tol = exhausted_sigma_tol
        self.exhausted_no_improve_mult = exhausted_no_improve_mult
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
        self.ir_archive_frac = ir_archive_frac
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
        # benefit from tuning T away from the canonical 1.0).
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
        while st.budget_left:
            # A spillover may consume the remaining budget; if so, stop before
            # spending a (now-impossible) generation, matching the original
            # mid-loop ``break``.
            if self._maybe_spillover(st):
                break
            self._run_generation(st)
            self._record_generation(st)

        result = self._make_result(st.history_x, st.history_f, st.history_pop)
        result.history_pop_sigma = st.history_pop_sigma
        result.history_sigma_global = st.history_sigma_global
        result.history_n_elite = st.history_n_elite
        result.history_no_improve = st.history_no_improve
        result.history_eval_count = st.history_eval_count
        result.history_sigma_eval = st.history_sigma_eval
        return result

    # ── run lifecycle ───────────────────────────────────────────────────────
    def _init_state(self, max_evals: int) -> _MCESOState:
        """Seed the RNG, draw the initial host pool, and prime the history
        buffers / bookkeeping scalars for one run."""
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        span = hi - lo
        sigma = self.sigma * span        # σ_init (scalar; per-dim bounds not supported)

        # Pre-allocate numpy arrays — avoids repeated list→array conversions per iteration
        pop_x = rng.uniform(lo, hi, (self.n_pop, self.dim))          # (n_pop, dim)
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
        return st

    # ── informed restart (reservoir re-ignition + herd-immunity repulsion) ──
    # Called from _maybe_spillover; overridable so the diagnostic ablation can
    # pin the original blind uniform restart (see mceso_ablations.py).
    def _on_spillover_start(self, st: _MCESOState, basin_switch: bool) -> None:
        """Harvest the converging basin into the persistent strain archive and
        record its centroid for herd-immunity repulsion — before the reseed
        overwrites the population."""
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
        if st.ir_archive_x and rng.random() < self.ir_archive_frac:
            k = rng.integers(0, len(st.ir_archive_x))
            sigma = self.ir_reignite_sigma_ratio * st.span
            return self._reflect(
                st.ir_archive_x[k] + sigma * rng.standard_normal(dim), lo, hi)
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

    def _droplet_strain_positions(self, st: _MCESOState, elite_arr: np.ndarray,
                                  n_h2h: int) -> np.ndarray:
        """Donor ('strain') positions for the droplet channel's current-to-best
        pull — one (dim,) row per child. Base: sample from the *live* niched-elite
        indices (original behaviour, RNG-identical). A subclass returns positions
        from a persistent strain archive so the pull keeps targeting coexisting
        basins even after the live population has collapsed onto one."""
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
        stagnated = st.no_improve >= self.exhausted_no_improve_mult * self.restart_no_improve_threshold
        return sigma_bottomed and stagnated

    def _spillover_should_fire(self, st: _MCESOState) -> bool:
        """Whether a spillover triggers this generation. Two regimes:

        • basin **exhausted** (drilled out) → fire on stagnation regardless of the
          precision gate, so the search leaves the solved basin to hunt others;
        • otherwise → the precision-gated condition (don't disturb an in-progress
          drilling), which protects SR@1e-10 on hard single-optimum functions.
        """
        if self._basin_exhausted(st):
            return st.no_improve >= self.restart_no_improve_threshold
        return (st.no_improve >= self.restart_no_improve_threshold
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
        if not self._spillover_should_fire(st):
            return False

        rng, lo, hi = st.rng, st.lo, st.hi

        # Escalation policy based on consecutive_failed_spillovers:
        #   basin switch: wipe best, fully uniform, σ_init reset
        #   else:         fully uniform, best preserved, smaller σ
        basin_switch = self._spillover_basin_switch(st)
        sigma_restart = (st.sigma_init if basin_switch
                         else st.sigma_init * self.restart_sigma_ratio)
        div_ratio = 1.0

        best_pre_spillover = st.best_so_far
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

        # Informed-restart hook: population still holds the converging basin
        # here, so a subclass can harvest a persistent archive / record the
        # abandoned basin centroid before the reseed overwrites it. No-op in base.
        self._on_spillover_start(st, basin_switch)

        n_div = int(round(div_ratio * len(reseed_idx)))
        # Shuffle so the assignment of "diversified" vs "local" is random
        rng.shuffle(reseed_idx)
        diversified = set(reseed_idx[:n_div])
        for i in reseed_idx:
            if i in diversified or x_best_snap is None:
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
            if f_new < st.best_so_far:
                st.best_so_far = f_new
            if not st.budget_left:
                break
        st.sigma = sigma_restart
        st.no_improve = 0
        st.log_best_ref = math.log10(max(st.best_so_far, 1e-300))
        st.evals_since_reset = 0
        if not basin_switch:
            st.pop_age[best_idx_global] = 0

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
        lq = np.clip(
            (log_f_max - np.log10(st.pop_f[gi_arr] + 1e-10)) / (log_f_spread + 1e-30),
            0.0, 1.0)
        ar = np.minimum(st.pop_age[gi_arr].astype(float) / 5.0, 1.0)
        host_scale = self.host_sigma_min_scale ** (lq * (0.7 + 0.3 * ar))
        noise = rng.standard_normal((n_local, self.dim))
        local_parent_x = st.pop_x[gi_arr].copy()
        sigma_i = st.sigma * host_scale
        if self.dim >= 2:
            cov = np.cov(st.pop_x, rowvar=False)
            if isinstance(cov, np.ndarray) and cov.shape == (self.dim, self.dim):
                eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
                floor_eff = self._adaptive_cov_floor(st, eigvals)
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
            h2h_step = self.h2h_F * (best_pull + diff)
        else:
            h2h_step = self.h2h_F * diff
        h2h_trial = parents_x + h2h_step
        cr_mask = rng.random((n_h2h, self.dim)) < self.h2h_CR
        forced = rng.integers(0, self.dim, size=n_h2h)
        cr_mask[np.arange(n_h2h), forced] = True
        h2h_offspring = np.where(cr_mask, h2h_trial, st.pop_x[h2h_parents_gi])
        eff_step = h2h_offspring - st.pop_x[h2h_parents_gi]
        h2h_step_norms = np.linalg.norm(eff_step, axis=1)
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
        noise_air = rng.standard_normal((n_air, self.dim))
        return self._reflect(st.pop_x[air_parents_gi] + noise_air * air_sigma_vec,
                             st.lo, st.hi)

    def _migratory_children(self, st: _MCESOState, n_mig: int) -> np.ndarray:
        """Migratory (vector-borne) transmission — stuck-gated structured escape.

        Long jumps from the current best (magnitude ``migratory_jump_ratio`` ×
        span): half along the population's principal covariance axis (valley/ridge
        escape), half isotropic (multimodal escape). Only invoked when the run is
        stuck (caller gates on drilling + stagnation), so it never diverts
        productive precision. Offspring compete via μ+λ rollback, so this can only
        improve best_f, never worsen SR@1e-10."""
        if n_mig <= 0:
            return np.empty((0, self.dim))
        rng = st.rng
        x_best = st.pop_x[int(np.argmin(st.pop_f))]
        jump = self.migratory_jump_ratio * st.span
        v_max = None
        if self.dim >= 2:
            cov = np.cov(st.pop_x, rowvar=False)
            if isinstance(cov, np.ndarray) and cov.shape == (self.dim, self.dim):
                _, eigvecs = np.linalg.eigh(cov)
                v_max = eigvecs[:, -1]
        out = np.empty((n_mig, self.dim))
        for k in range(n_mig):
            if v_max is not None and rng.random() < 0.5:
                out[k] = x_best + rng.choice([-1.0, 1.0]) * jump * v_max   # along ridge/valley
            else:
                out[k] = x_best + jump * rng.standard_normal(self.dim)     # isotropic escape
        return self._reflect(out, st.lo, st.hi)

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
        air_base = 0.0 if drilling else self.air_ratio
        if not self.channel_schedule:
            return air_base, self.h2h_ratio
        # Drilling always runs airborne-free (as in base); the route is moot.
        if drilling:
            return air_base, self.h2h_ratio
        # Commit the route once, after a warmup, from the stabilized EMAs — then
        # lock it. Conditioning/separability are run-invariant landscape
        # properties, so a single committed route removes the generation-to-
        # generation flip-flop that otherwise perturbs threshold-borderline
        # functions (F04/F14) and even leaks keep-air functions (F06) off base.
        if st.channel_route is None:
            if st.cc_logratio_ema is None or st.cc_align_ema is None:
                return air_base, self.h2h_ratio          # KEEP-AIR (base) while warming
            if st.cc_logratio_ema > self.cond_droplet_early:
                st.channel_route = "droplet"             # early high-cond → ill-cond valley
            elif len(st.history_sigma_global) < self.route_commit_gen:
                return air_base, self.h2h_ratio          # KEEP-AIR (base) until checkpoint
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
            return self.air_ratio, self.h2h_ratio        # base ratios (escape untouched)
        # σ-ramp taper (0 at the explore scale span×sigma → 1 at the drilling
        # threshold), reused from the validated variants so a routed function
        # starts at the base ratio at full σ and only diverges as it drills.
        s = math.log10(max(st.sigma, 1e-300) / span)
        s_hi, s_lo = math.log10(self.sigma), math.log10(self.precision_sigma_ratio)
        t = (s_hi - s) / (s_hi - s_lo) if s_hi > s_lo else 1.0
        t = min(1.0, max(0.0, t))
        air_tapered = self.air_ratio * (1.0 - t)
        if st.channel_route == "droplet":
            return air_tapered, self.h2h_ratio + (self.air_ratio - air_tapered)
        # CLOSE route: freed airborne budget → close-contact (implicit).
        return air_tapered, self.h2h_ratio

    # ── one generation (μ+λ greedy step) ────────────────────────────────────
    def _run_generation(self, st: _MCESOState) -> None:
        """One outbreak generation: select strains + the worst-K hosts to kill,
        spawn offspring across the three channels, evaluate them into the dead
        slots with greedy rollback, age survivors, and adapt σ_global."""
        rng, span, n = st.rng, st.span, self.n_pop
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
        n_dead = len(dead_global)
        if n_dead > 0:
            dead_orig_x = st.pop_x[dead_global].copy()
            dead_orig_f = st.pop_f[dead_global].copy()
        else:
            dead_orig_x = None
            dead_orig_f = None

        if n_dead == 0:
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
        # Migratory (vector-borne) channel: only when stuck (drilled in AND
        # stagnating). It draws from the close-contact share (airborne is already 0
        # in drilling), giving stalled precision offspring a structured escape.
        in_drilling_now = st.sigma < span * self.precision_sigma_ratio
        migratory_active = (self.migratory_channel and in_drilling_now
                            and st.no_improve >= self.migratory_no_improve_thresh)
        n_mig = max(0, int(round(self.migratory_ratio * n_dead))) if migratory_active else 0
        n_local = n_dead - n_air - n_h2h - n_mig
        if n_local < 0:                     # migratory overflow → trim it
            n_mig = max(0, n_mig + n_local)
            n_local = n_dead - n_air - n_h2h - n_mig

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
        new_local, sigma_i = self._close_contact_children(
            st, n_local, weights, log_f_max, log_f_spread)
        new_h2h, h2h_step_norms = self._droplet_children(st, n_h2h, weights, elite_arr)
        new_air = self._airborne_children(st, n_air, air_sigma_vec)
        # Migratory is generated last so that when the channel is off (n_mig = 0)
        # no RNG is drawn and the run stays bit-identical to base.
        new_mig = self._migratory_children(st, n_mig)
        new_xs = np.concatenate([new_local, new_h2h, new_air, new_mig], axis=0)

        # Per-child sigma: local→σ_i, h2h→|step|, air→air_sigma_vec, mig→jump
        _sc: list[np.ndarray] = []
        if n_local > 0:
            _sc.append(sigma_i)
        if n_h2h > 0:
            _sc.append(h2h_step_norms)
        if n_air > 0:
            _sc.append(np.full(n_air, float(air_sigma_vec)))
        if n_mig > 0:
            _sc.append(np.full(n_mig, float(self.migratory_jump_ratio * span)))
        _sigma_children = np.concatenate(_sc) if _sc else np.array([])

        # Evaluate offspring and resolve host competition (placement + rollback +
        # aging). Factored into a hook so niching variants can swap the *global*
        # competition rule for a local (crowding) one without touching channels.
        self._place_and_compete(
            st, new_xs, _sigma_children, n_dead,
            dead_global, dead_orig_x, dead_orig_f)

        self._adapt_sigma(st, gen_best_before)

    def _adapt_sigma(self, st: _MCESOState, gen_best_before: float) -> None:
        """σ_global adaptation (always on): improved → × sigma_up, else
        × sigma_down (or sigma_drill_down once drilling). Overridable so niching
        variants can use a monotone-contraction schedule that drills every
        occupied basin to FP precision instead of letting cross-basin
        improvements pump σ back up."""
        span = st.span
        in_drilling = st.sigma < span * self.precision_sigma_ratio
        sigma_floor_eff = span * self.sigma_floor_ratio
        if st.best_so_far < gen_best_before:
            st.sigma *= self.sigma_up
        else:
            st.sigma *= self.sigma_drill_down if in_drilling else self.sigma_down
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
        if f < st.best_so_far:
            st.best_so_far = f
            if self._meaningful_improvement(f, st.log_best_ref, st.evals_since_reset):
                st.no_improve = 0
                st.log_best_ref = math.log10(max(f, 1e-300))
                st.evals_since_reset = 0
            else:
                st.no_improve += 1
        else:
            st.no_improve += 1

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
        if dead_orig_x is not None:
            for k, slot in enumerate(replaced_slots):
                if dead_orig_f[k] < st.pop_f[slot]:
                    st.pop_x[slot] = dead_orig_x[k]
                    st.pop_f[slot] = dead_orig_f[k]

        # Age active survivors (per-generation).
        replaced_mask = np.zeros(self.n_pop, dtype=bool)
        if replaced_slots:
            replaced_mask[replaced_slots] = True
        st.pop_age[~replaced_mask] += 1

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
