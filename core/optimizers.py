from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np
import cma

from .benchmarks import BenchmarkFunction


@dataclass
class OptimizeResult:
    best_x: np.ndarray
    best_f: float
    history_x: list[np.ndarray]  # all evaluated points (for trajectory)
    history_best: list[float]    # best_f at each evaluation (for convergence)
    history_f: list[float]       # raw f value at each evaluation
    history_pop: list[np.ndarray]  # population snapshot per generation (n, dim)
    n_evals: int
    # per-individual sigma per generation (n,) array; empty = not recorded
    history_pop_sigma: list[np.ndarray] = field(default_factory=list)
    # MC-ESO-specific dynamics (one entry per generation; empty for non-MC-ESO)
    history_sigma_global: list[float] = field(default_factory=list)
    history_n_elite: list[int] = field(default_factory=list)
    history_no_improve: list[int] = field(default_factory=list)
    history_eval_count: list[int] = field(default_factory=list)
    # sigma actually used to generate each offspring (one per eval after init pop;
    # nan = random reactivation with no parent sigma)
    history_sigma_eval: list[float] = field(default_factory=list)


class BaseOptimizer(ABC):
    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
    ):
        self.benchmark = benchmark
        self.func = benchmark.func
        self.bounds = benchmark.bounds
        self.dim = benchmark.dim
        self.seed = seed

    @abstractmethod
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        ...

    def _make_result(
        self,
        history_x: list[np.ndarray],
        history_f: list[float],
        history_pop: Optional[list[np.ndarray]] = None,
    ) -> OptimizeResult:
        best_idx = int(np.argmin(history_f))
        history_best: list[float] = []
        current_best = float("inf")
        for f in history_f:
            if f < current_best:
                current_best = f
            history_best.append(current_best)
        return OptimizeResult(
            best_x=history_x[best_idx],
            best_f=history_f[best_idx],
            history_x=history_x,
            history_best=history_best,
            history_f=history_f,
            history_pop=history_pop or [],
            n_evals=len(history_f),
        )


class CMAESOptimizer(BaseOptimizer):
    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        sigma0: float = 1.0,
        x0: Optional[np.ndarray] = None,
    ):
        super().__init__(benchmark, seed)
        self.sigma0 = sigma0
        self.x0 = x0

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds

        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        history_pop: list[np.ndarray] = []

        x0 = self.x0 if self.x0 is not None else rng.uniform(lo, hi, self.dim)
        sigma = self.sigma0
        restart_seed = self.seed

        # When CMA-ES converges, restart from best found (not random) with
        # tighter sigma to continue using the full eval budget locally.
        while len(history_f) < max_evals:
            remaining = max_evals - len(history_f)
            opts = cma.CMAOptions()
            opts["seed"] = restart_seed
            opts["bounds"] = [[lo] * self.dim, [hi] * self.dim]
            opts["maxfevals"] = remaining
            opts["verbose"] = -9
            es = cma.CMAEvolutionStrategy(x0, sigma, opts)
            restart_seed += 1

            while len(history_f) < max_evals and not es.stop():
                solutions = es.ask()
                fitnesses = [self.func(np.array(s)) for s in solutions]
                es.tell(solutions, fitnesses)
                history_pop.append(np.array(solutions))
                for s, f in zip(solutions, fitnesses):
                    history_x.append(np.array(s))
                    history_f.append(f)

            # Restart from best found with tighter sigma (no random jump)
            x0 = np.array(es.result.xbest)
            sigma = max(es.result.stds.mean() * 0.1, 1e-8)

        return self._make_result(history_x, history_f, history_pop)


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

      • **Spillover event with basin-avoidance memory** (stagnation-triggered
        re-seed): when the outbreak stalls (no improvement for
        ``restart_no_improve_threshold`` evals) and the current best is still
        loose, the population spills over to a fresh host pool around the
        best with σ = σ_init·``restart_sigma_ratio``. Each failed spillover
        appends its pre-spillover best position to a memory list; the next
        uniform reseed rejects samples within ``basin_radius_ratio`` × span
        of any remembered point, preventing re-capture of the same suboptimal
        basin (essential for F18 SchafferF7-ill).

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
        n_pop: int = 20,
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
        air_sigma_min: float = 1.5,
        air_sigma_max: float = 5.0,
        # ── Greedy (μ+λ) replacement ───────────────────────────────────
        kill_fraction: float = 0.25,           # fraction of active killed per gen (by f)
        # ── Restart on stagnation ──────────────────────────────────────
        restart_no_improve_threshold: int = 300,  # no_improve count that triggers restart
        restart_sigma_ratio: float = 0.3,      # σ after restart, relative to σ_init
        restart_quality_floor: float = 1e-8,   # skip restart if already below this f
        # Diversified spillover: on each restart, a fraction of the re-seeded
        # population is placed uniformly across the search domain rather than
        # around the best. This lets the search escape deceptive basins
        # (F17/F20) — pure local re-seed alone can't, because the same basin
        # recaptures the population.
        restart_diversify_ratio: float = 0.75, # share of re-seed assigned to Uniform(lo, hi)
        # Spillover escalation. When consecutive spillovers fail to improve
        # the best, ratchet up the disruption — first widen to fully uniform
        # re-seed (best preserved), then break the best entirely and reset σ.
        # Addresses deceptive double-funnel landscapes (F24) and rugged
        # separable functions (F04) where the algorithm gets locked into a
        # wrong basin and pure local spillover only re-explores it.
        escalate_after_failed_spillovers: int = 1,   # streak → diversify_ratio = 1.0
        basin_switch_after_failed_spillovers: int = 2,  # streak → wipe best & reset σ
        basin_switch_quality_floor: float = 1e-2,    # basin switch suppressed when
                                                     # best_so_far ≤ this — protects
                                                     # runs that are slowly grinding
                                                     # toward the optimum (F13 ridge,
                                                     # C01 deep precision) from being
                                                     # wiped by a premature switch
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
        lifespan: int = 5,                     # age normalizer for local σ_i scaling
        temperature: float = 1.0,              # softmax temp for parent-selection weighting
        stagnation_limit: int = 2000,          # absolute kill-switch on stalled runs
        log_slope_threshold: float = 1e-4,     # min log10(f) slope counted as improvement
        # ── h2h binomial crossover (always on) ────────────────────────
        # The droplet child is built as DE/current-to-best/1, then a binomial
        # crossover with the parent gates each coordinate (rate h2h_CR). This
        # preserves coordinate-aligned structure on separable multimodals
        # (F04 Büche-Rastrigin SR 77→100%, F17 Schaffer F7 47→73% at n=30).
        h2h_CR: float = 0.7,                   # h2h binomial crossover rate
        # ── Rotation-aware close-contact (empirical covariance) ──────
        # Close-contact noise is drawn from N(0, σ_i²·C_pop) where C_pop is
        # the *instantaneous* empirical covariance of the population — no
        # history accumulation (cf. CMA-ES). Mean eigenvalue is normalized
        # to 1 so total step magnitude is preserved; floor prevents collapsed
        # axes. Closes the F11/F14 ill-conditioned gap to DE/CMA-ES (F11
        # mean 5.2e-8 → 0 at n=15, F14 SR_1e-7 80% → 87%).
        empirical_cov_floor: float = 0.01,     # min normalized eigenvalue
        # ── Basin-avoidance memory (spillover anti-recapture) ─────────
        # Each failed spillover records its best-position; subsequent
        # uniform reseeds reject samples within ``basin_radius_ratio`` ×
        # span of any remembered point. Targets multimodal recapture
        # (F18 SchafferF7-ill SR_1e-10 33% → 67% at n=15).
        basin_radius_ratio: float = 0.05,      # avoidance radius / span
        basin_memory_size: int = 5,            # max remembered failed basins
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.lifespan = lifespan
        self.sigma = sigma
        self.air_ratio = air_ratio
        self.n_elite_max = n_elite_max
        self.temperature = temperature
        self.stagnation_limit = stagnation_limit
        self.niche_radius_ratio = niche_radius_ratio
        self.host_sigma_min_scale = host_sigma_min_scale
        self.air_sigma_min = air_sigma_min
        self.air_sigma_max = air_sigma_max
        self.log_slope_threshold = log_slope_threshold
        self.h2h_ratio = h2h_ratio
        self.h2h_F = h2h_F
        self.kill_fraction = kill_fraction
        self.restart_no_improve_threshold = restart_no_improve_threshold
        self.restart_sigma_ratio = restart_sigma_ratio
        self.restart_quality_floor = restart_quality_floor
        self.restart_diversify_ratio = restart_diversify_ratio
        self.escalate_after_failed_spillovers = escalate_after_failed_spillovers
        self.basin_switch_after_failed_spillovers = basin_switch_after_failed_spillovers
        self.basin_switch_quality_floor = basin_switch_quality_floor
        self.sigma_up = sigma_up
        self.sigma_down = sigma_down
        self.sigma_floor_ratio = sigma_floor_ratio
        self.sigma_ceil_ratio = sigma_ceil_ratio
        self.precision_sigma_ratio = precision_sigma_ratio
        self.sigma_drill_down = sigma_drill_down
        self.h2h_CR = h2h_CR
        self.empirical_cov_floor = empirical_cov_floor
        self.basin_radius_ratio = basin_radius_ratio
        self.basin_memory_size = basin_memory_size

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

    def _axis_sweep(self, x_best: np.ndarray, lo: float, hi: float
                    ) -> list[np.ndarray]:
        """Coordinate-axis line search around x_best. Per dimension, probe at
        a few axis-aligned step sizes both directions plus the two bounds.

        Built for **separable / boundary-optimal** landscapes:
          • F05 LinearSlope (optimum on a corner): the {lo, hi} probes land
            on the optimum exactly when the right sign is picked.
          • F04 BucheRastrigin (separable, ~1.0-wide local basins on a grid):
            step sizes of span × 0.1, 0.2, 0.4 jump between adjacent local
            basins, so an axis-aligned probe lands inside the global basin
            with much higher probability than isotropic Gaussian sampling.

        Total candidates ≈ dim × (6 + 2). For dim=2 this is ~16 evals per
        sweep, dwarfed by the spillover that follows it.
        """
        span = hi - lo
        cands: list[np.ndarray] = []
        for i in range(self.dim):
            for k in (1, 2, 4):
                step = k * span * 0.1
                for sign in (-1.0, 1.0):
                    cand = x_best.copy()
                    cand[i] = x_best[i] + sign * step
                    cands.append(self._reflect(cand, lo, hi))
            for bv in (lo, hi):
                cand = x_best.copy()
                cand[i] = bv
                cands.append(cand)
        return cands

    @staticmethod
    def _uniform_avoiding(rng, lo: float, hi: float, dim: int,
                          memory: list, radius: float) -> np.ndarray:
        """Rejection-sample a uniform point in [lo, hi]^dim at least ``radius``
        away from every memory point. Falls back to plain uniform after 20
        rejections so the search never stalls when memory blankets the box."""
        for _ in range(20):
            x = rng.uniform(lo, hi, dim)
            if not memory:
                return x
            if all(np.linalg.norm(x - m) >= radius for m in memory):
                return x
        return x

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
        f_max = f_vals.max()
        scores = (f_max - f_vals) / (self.temperature + 1e-30)
        scores -= scores.max()
        w = np.exp(scores)
        return w / w.sum()

    def _host_sigma_scale(self, pop_f: np.ndarray, pop_age: np.ndarray) -> np.ndarray:
        """Per-host σ scaling factor: high-quality / old hosts get a smaller
        fraction of σ_global so they probe finer. lq ∈ [0, 1] is the relative
        log-fitness, ar ∈ [0, 1] is age / lifespan, and the factor is
        ``host_sigma_min_scale ** (lq · (0.7 + 0.3 · ar))``."""
        lf = np.log10(pop_f + 1e-10)
        spread = float(lf.max() - lf.min()) + 1e-30
        lq = np.clip((lf.max() - lf) / spread, 0.0, 1.0)
        ar = np.minimum(pop_age.astype(float) / max(self.lifespan, 1), 1.0)
        return self.host_sigma_min_scale ** (lq * (0.7 + 0.3 * ar))

    def _meaningful_improvement(self, f: float, log_best_ref: float, evals_since_reset: int) -> bool:
        if evals_since_reset == 0:
            return True
        # max() floors f to 1e-300 so FP-noise-induced negative values near the
        # optimum (e.g. -1e-15 from BBOB - f_opt cancellation) don't crash log10.
        slope = (log_best_ref - math.log10(max(f, 1e-300))) / evals_since_reset
        return slope >= self.log_slope_threshold

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        span = hi - lo
        sigma = self.sigma * span                # σ_init (scalar; per-dim bounds not supported)
        sigma_init = float(sigma)
        niche_radius = self.niche_radius_ratio * span

        # Pre-allocate numpy arrays — avoids repeated list→array conversions per iteration
        pop_x = rng.uniform(lo, hi, (self.n_pop, self.dim))          # (n_pop, dim)
        pop_f = np.array([self.func(x) for x in pop_x])              # (n_pop,)
        # pop_age is the σ_i age-ratio normalizer divisor (scaled by self.lifespan).
        # Death is f-based (μ+λ greedy), so age has no effect on survival.
        pop_age = np.zeros(self.n_pop, dtype=int)

        history_x: list[np.ndarray] = [row.copy() for row in pop_x]
        history_f: list[float] = pop_f.tolist()
        history_pop: list[np.ndarray] = [pop_x.copy()]
        history_pop_sigma: list[np.ndarray] = [
            float(sigma) * self._host_sigma_scale(pop_f, pop_age)]
        history_sigma_global: list[float] = []
        history_n_elite: list[int] = []
        history_no_improve: list[int] = []
        history_eval_count: list[int] = []
        history_sigma_eval: list[float] = []

        best_so_far = float(pop_f.min())
        no_improve = 0
        log_best_ref = math.log10(max(best_so_far, 1e-300))  # log10(f) at last meaningful reset
        evals_since_reset = 0                             # evals elapsed since last meaningful reset
        # Step 6: track consecutive spillover events that failed to improve
        # best_so_far. Used to ratchet up disruption from "diversified local"
        # → "fully uniform" → "basin switch (wipe best & reset σ)".
        consecutive_failed_spillovers = 0
        # Basin-avoidance memory — best positions at each failed spillover.
        # Uniform reseeds reject samples within ``basin_radius_ratio`` × span
        # of any remembered point, so we don't spend another cycle re-converging
        # to a known-bad basin (essential for F18 SchafferF7-ill).
        bad_basin_memory: list[np.ndarray] = []

        while len(history_f) < max_evals:
            # Spillover event: when the outbreak stalls, the population spills
            # over to a fresh host pool around the global best, giving the search
            # another chance to align onto a productive direction (essential for
            # ill-conditioned landscapes). Quality-gated to avoid disturbing
            # already-converged runs.
            if (no_improve >= self.restart_no_improve_threshold
                    and best_so_far > self.restart_quality_floor):
                # Step 6: escalation policy based on consecutive_failed_spillovers.
                #   streak ≥ basin_switch_after: wipe best, fully uniform, σ_init reset
                #   streak ≥ escalate_after:    fully uniform, best preserved
                #   streak  = 0:                default mix (75% uniform / 25% local)
                if (consecutive_failed_spillovers
                        >= self.basin_switch_after_failed_spillovers
                        and best_so_far > self.basin_switch_quality_floor):
                    basin_switch = True
                    div_ratio = 1.0
                    sigma_restart = sigma_init   # fresh σ for new basin
                elif (consecutive_failed_spillovers
                        >= self.escalate_after_failed_spillovers):
                    basin_switch = False
                    div_ratio = 1.0
                    sigma_restart = sigma_init * self.restart_sigma_ratio
                else:
                    basin_switch = False
                    div_ratio = self.restart_diversify_ratio
                    sigma_restart = sigma_init * self.restart_sigma_ratio

                # Step 7: coordinate-axis sweep before escalated spillovers.
                # Targets separable / boundary-optimal landscapes that the
                # isotropic uniform re-seed alone fails on (F04 BucheRastrigin,
                # F05 LinearSlope). Only fires once a normal spillover has
                # already failed (streak ≥ 1) so the eval cost is paid only
                # when standard exploration is clearly stuck.
                if consecutive_failed_spillovers >= self.escalate_after_failed_spillovers:
                    x_best_for_sweep = pop_x[int(np.argmin(pop_f))].copy()
                    for sweep_cand in self._axis_sweep(x_best_for_sweep, lo, hi):
                        if len(history_f) >= max_evals:
                            break
                        f_sweep = float(self.func(sweep_cand))
                        history_x.append(sweep_cand.copy())
                        history_f.append(f_sweep)
                        history_sigma_eval.append(
                            float(np.linalg.norm(sweep_cand - x_best_for_sweep)))
                        if f_sweep < best_so_far:
                            best_so_far = f_sweep
                            # Inject the better candidate into the population so
                            # the upcoming spillover anchors on it (or, on basin
                            # switch, so it's part of the historical best).
                            worst_i = int(np.argmax(pop_f))
                            pop_x[worst_i] = sweep_cand.copy()
                            pop_f[worst_i] = f_sweep
                            pop_age[worst_i] = 0
                    if len(history_f) >= max_evals:
                        break

                best_pre_spillover = best_so_far
                # Snapshot best position now — appended to bad_basin_memory
                # below if this spillover fails to improve.
                x_best_pre_spillover = pop_x[int(np.argmin(pop_f))].copy()
                if basin_switch:
                    # Wipe everything — including the current best — and re-seed
                    # all slots uniformly. best_so_far is preserved as a tracker
                    # of the historical best, but the population starts fresh.
                    reseed_idx = list(range(self.n_pop))
                    x_best_snap = None  # unused since div_ratio = 1.0
                else:
                    best_idx_global = int(np.argmin(pop_f))
                    x_best_snap = pop_x[best_idx_global].copy()
                    reseed_idx = [i for i in range(self.n_pop) if i != best_idx_global]

                n_div = int(round(div_ratio * len(reseed_idx)))
                # Shuffle so the assignment of "diversified" vs "local" is random
                rng.shuffle(reseed_idx)
                diversified = set(reseed_idx[:n_div])
                for i in reseed_idx:
                    if i in diversified or x_best_snap is None:
                        if bad_basin_memory:
                            new_x = self._uniform_avoiding(
                                rng, lo, hi, self.dim, bad_basin_memory,
                                span * self.basin_radius_ratio)
                        else:
                            new_x = rng.uniform(lo, hi, self.dim)
                        sig_log = (float(np.linalg.norm(new_x - x_best_snap))
                                   if x_best_snap is not None else float(sigma_restart))
                    else:
                        new_x = self._reflect(
                            x_best_snap + sigma_restart * rng.standard_normal(self.dim), lo, hi)
                        sig_log = float(sigma_restart)
                    f_new = float(self.func(new_x))
                    pop_x[i] = new_x
                    pop_f[i] = f_new
                    pop_age[i] = 0
                    history_x.append(new_x.copy())
                    history_f.append(f_new)
                    history_sigma_eval.append(sig_log)
                    if f_new < best_so_far:
                        best_so_far = f_new
                    if len(history_f) >= max_evals:
                        break
                sigma = sigma_restart
                no_improve = 0
                log_best_ref = math.log10(max(best_so_far, 1e-300))
                evals_since_reset = 0
                if not basin_switch:
                    pop_age[best_idx_global] = 0

                # Update streak based on whether this spillover improved best
                if best_so_far < best_pre_spillover - 1e-12:
                    consecutive_failed_spillovers = 0
                else:
                    consecutive_failed_spillovers += 1
                    # This spillover failed → remember the basin it started
                    # from so the next uniform reseed avoids it.
                    bad_basin_memory.append(x_best_pre_spillover)
                    if len(bad_basin_memory) > self.basin_memory_size:
                        bad_basin_memory.pop(0)
                if len(history_f) >= max_evals:
                    break

            n = self.n_pop
            gen_best_before = best_so_far  # snapshot for optional σ adaptation

            # Strain coexistence: niched-elite pool for the droplet channel's
            # current-to-best pull. Spatial diversity is also fed to the airborne
            # channel's σ modulator (denser cluster → larger aerosol jumps).
            pop_diversity = np.mean(np.std(pop_x, axis=0) / span)
            diversity_ratio = float(np.clip(pop_diversity / 0.289, 0.0, 1.0))
            elite_global = self._niche_elites(pop_x, pop_f, niche_radius)
            elite_arr = np.fromiter(elite_global, dtype=int) if elite_global else np.empty(0, dtype=int)

            # Host competition (μ+λ greedy): each gen, the worst K = kill_fraction · n
            # hosts die (regardless of strain status — the global best always
            # survives by not being in worst-K). Children that fail to outcompete
            # the host they replaced are rolled back after evaluation.
            n_kill = max(1, min(n, int(round(self.kill_fraction * n))))
            dead_global = np.argsort(pop_f)[::-1][:n_kill]
            n_dead = len(dead_global)
            if n_dead > 0:
                dead_orig_x = pop_x[dead_global].copy()
                dead_orig_f = pop_f[dead_global].copy()
            else:
                dead_orig_x = None
                dead_orig_f = None

            if n_dead == 0:
                pop_age += 1
                # No offspring this gen → no σ signal, so σ is held unchanged.
                # When all individuals are elite no births occur and history_f
                # never grows → still advance no_improve so the spillover and
                # stagnation_limit eventually fire.
                no_improve += 1
                if no_improve >= self.stagnation_limit:
                    break
            else:
                weights = self._softmax_weights(pop_f)

                # 3-channel split: close-contact (local Gaussian), droplet (h2h
                # DE/current-to-best), airborne (random spread). Airborne is
                # pure noise — suppressed once drilling mode is entered so
                # precision grinding isn't disrupted.
                in_drilling_now = sigma < span * self.precision_sigma_ratio
                air_ratio_eff = 0.0 if in_drilling_now else self.air_ratio
                n_air = max(0, int(round(air_ratio_eff * n_dead)))
                n_h2h = max(0, int(round(self.h2h_ratio * n_dead))) if n >= 3 else 0
                # If rounding overflows, trim airborne first (preserves the
                # droplet/close-contact intent).
                if n_air + n_h2h > n_dead:
                    n_air = max(0, n_dead - n_h2h)
                n_local = n_dead - n_air - n_h2h

                # Log-scale quality anchored to global best (history-wide).
                # When population converges to a local optimum, all f_i ≈ f_pop_max
                # but best_so_far may be far better → lq ≈ 0 → σ_i = σ_global (full exploration).
                pop_log_f = np.log10(pop_f + 1e-10)
                log_f_max = float(pop_log_f.max())
                log_f_best = float(np.log10(best_so_far + 1e-10))  # anchored to history best
                log_f_spread = log_f_max - log_f_best

                # Air sigma: large when converged (need to escape), small when diverse
                # diversity_ratio already computed above before elite selection
                air_sigma_factor = self.air_sigma_max - (self.air_sigma_max - self.air_sigma_min) * diversity_ratio
                air_sigma_base = np.maximum(sigma, sigma_init * 0.3)
                air_sigma_vec = air_sigma_base * air_sigma_factor

                # Batch generate all children before evaluation loop
                if n_local > 0:
                    gi_arr = rng.choice(n, size=n_local, p=weights)
                    lq = np.clip(
                        (log_f_max - np.log10(pop_f[gi_arr] + 1e-10)) / (log_f_spread + 1e-30),
                        0.0, 1.0)
                    ar = np.minimum(
                        pop_age[gi_arr].astype(float) / max(self.lifespan, 1), 1.0)
                    host_scale = self.host_sigma_min_scale ** (lq * (0.7 + 0.3 * ar))
                    noise = rng.standard_normal((n_local, self.dim))
                    local_parent_x = pop_x[gi_arr].copy()
                    sigma_i = sigma * host_scale
                    # Rotation-aware close-contact: sample noise from
                    # N(0, C_pop) using the *current* empirical covariance —
                    # full anisotropic + rotated shape without CMA-ES-style
                    # history accumulation. Eigenvalues are normalized so
                    # mean=1 (preserves total step magnitude); floor prevents
                    # collapsed axes.
                    if self.dim >= 2:
                        cov = np.cov(pop_x, rowvar=False)
                        if isinstance(cov, np.ndarray) and cov.shape == (self.dim, self.dim):
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            mean_eig = float(eigvals.mean())
                            if mean_eig > 1e-30:
                                eigvals = eigvals / mean_eig
                            else:
                                eigvals = np.ones(self.dim)
                            eigvals = np.maximum(eigvals, self.empirical_cov_floor)
                            eigvals = eigvals / float(eigvals.mean())  # re-normalize
                            # Transform: noise (n_local, dim) @ (eigvecs · √eigvals)ᵀ
                            transform = eigvecs * np.sqrt(eigvals)[None, :]
                            noise = noise @ transform.T
                    new_local = self._reflect(local_parent_x + noise * sigma_i[:, None], lo, hi)
                else:
                    gi_arr = np.empty(0, dtype=int)
                    sigma_i = np.empty(0)
                    new_local = np.empty((0, self.dim))

                # Droplet channel (DE/current-to-best/1):
                #   x_child = x_parent + F·(x_strain - x_parent) + F·(x_a - x_b)
                # The diff term (x_a-x_b) injects population-shape-aware drift —
                # this gives MC-ESO implicit anisotropy without a covariance matrix.
                # The strain-pull term accelerates descent along narrow valleys.
                if n_h2h > 0:
                    h2h_parents_gi = rng.choice(n, size=n_h2h, p=weights)
                    h2h_a_li = rng.integers(0, n, size=n_h2h)
                    h2h_b_li = rng.integers(0, n, size=n_h2h)
                    diff = pop_x[h2h_a_li] - pop_x[h2h_b_li]
                    if len(elite_arr) > 0:
                        elite_pick = rng.choice(elite_arr, size=n_h2h)
                        best_pull = pop_x[elite_pick] - pop_x[h2h_parents_gi]
                        h2h_step = self.h2h_F * (best_pull + diff)
                    else:
                        h2h_step = self.h2h_F * diff
                    h2h_trial = pop_x[h2h_parents_gi] + h2h_step
                    # DE-style binomial crossover preserves per-dim inheritance
                    # from the parent — critical for separable problems where
                    # dimensions can be solved independently. CR=0.7 trades off
                    # F18/F19-style rotated landscapes (favored by CR=0.9) for
                    # large gains on F04/F17 separable multimodals.
                    cr_mask = rng.random((n_h2h, self.dim)) < self.h2h_CR
                    forced = rng.integers(0, self.dim, size=n_h2h)
                    cr_mask[np.arange(n_h2h), forced] = True
                    h2h_offspring = np.where(cr_mask, h2h_trial, pop_x[h2h_parents_gi])
                    eff_step = h2h_offspring - pop_x[h2h_parents_gi]
                    h2h_step_norms = np.linalg.norm(eff_step, axis=1)
                    new_h2h = self._reflect(h2h_offspring, lo, hi)
                else:
                    new_h2h = np.empty((0, self.dim))
                    h2h_step_norms = np.empty(0)

                if n_air > 0:
                    air_parents_gi = rng.integers(0, n, size=n_air)
                    noise_air = rng.standard_normal((n_air, self.dim))
                    new_air = self._reflect(pop_x[air_parents_gi] + noise_air * air_sigma_vec, lo, hi)
                else:
                    new_air = np.empty((0, self.dim))

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

                # Evaluate and place offspring into dead slots
                replaced_slots: list[int] = []
                for k in range(min(n_dead, len(new_xs))):
                    slot = int(dead_global[k])
                    x = new_xs[k]
                    f = float(self.func(x))
                    pop_x[slot] = x
                    pop_f[slot] = f
                    pop_age[slot] = 0
                    replaced_slots.append(slot)
                    history_x.append(x.copy())
                    history_f.append(f)
                    sigma_used_k = float(_sigma_children[k]) if k < len(_sigma_children) else float(sigma)
                    history_sigma_eval.append(sigma_used_k)
                    evals_since_reset += 1

                    if f < best_so_far:
                        best_so_far = f
                        if self._meaningful_improvement(f, log_best_ref, evals_since_reset):
                            no_improve = 0
                            log_best_ref = math.log10(max(f, 1e-300))
                            evals_since_reset = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                    if len(history_f) >= max_evals or no_improve >= self.stagnation_limit:
                        break

                # Host-competition rollback: children worse than the host they
                # replaced are reverted, so the outbreak monotonically improves.
                if dead_orig_x is not None:
                    for k, slot in enumerate(replaced_slots):
                        if dead_orig_f[k] < pop_f[slot]:
                            pop_x[slot] = dead_orig_x[k]
                            pop_f[slot] = dead_orig_f[k]

                if no_improve >= self.stagnation_limit:
                    break

                # Age active survivors (per-generation)
                replaced_mask = np.zeros(self.n_pop, dtype=bool)
                if replaced_slots:
                    replaced_mask[replaced_slots] = True
                pop_age[~replaced_mask] += 1

                # σ adaptation always on: improved → × sigma_up,
                # else → × sigma_down (or sigma_drill_down in drilling mode).
                in_drilling = sigma < span * self.precision_sigma_ratio
                sigma_floor_eff = span * self.sigma_floor_ratio
                if best_so_far < gen_best_before:
                    sigma *= self.sigma_up
                else:
                    sigma *= self.sigma_drill_down if in_drilling else self.sigma_down
                sigma = max(sigma_floor_eff,
                            min(sigma, span * self.sigma_ceil_ratio))

            # Per-generation dynamics recording (population always = n_pop)
            history_pop_sigma.append(
                float(sigma) * self._host_sigma_scale(pop_f, pop_age))
            history_pop.append(pop_x.copy())
            history_sigma_global.append(float(sigma))
            history_n_elite.append(len(elite_global))
            history_no_improve.append(int(no_improve))
            history_eval_count.append(len(history_f))

        result = self._make_result(history_x, history_f, history_pop)
        result.history_pop_sigma = history_pop_sigma
        result.history_sigma_global = history_sigma_global
        result.history_n_elite = history_n_elite
        result.history_no_improve = history_no_improve
        result.history_eval_count = history_eval_count
        result.history_sigma_eval = history_sigma_eval
        return result


class PSOOptimizer(BaseOptimizer):
    """Particle Swarm Optimization (Kennedy & Eberhart, 1995).

    Standard inertia-weight PSO with velocity clamping.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_particles: int = 30,
        w: float = 0.729,    # inertia weight
        c1: float = 1.494,   # cognitive coefficient
        c2: float = 1.494,   # social coefficient
    ):
        super().__init__(benchmark, seed)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        span = hi - lo
        v_max = 0.2 * span

        pos = rng.uniform(lo, hi, (self.n_particles, self.dim))
        vel = rng.uniform(-v_max, v_max, (self.n_particles, self.dim))
        fit = np.array([self.func(x) for x in pos])

        pbest_pos = pos.copy()
        pbest_fit = fit.copy()
        gbest_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[gbest_idx].copy()

        history_x: list[np.ndarray] = list(pos)
        history_f: list[float] = list(fit)
        history_pop: list[np.ndarray] = [pos.copy()]

        while len(history_f) < max_evals:
            r1 = rng.random((self.n_particles, self.dim))
            r2 = rng.random((self.n_particles, self.dim))
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos - pos))
            vel = np.clip(vel, -v_max, v_max)
            pos = np.clip(pos + vel, lo, hi)

            for i, x in enumerate(pos):
                if len(history_f) >= max_evals:
                    break
                f = self.func(x)
                history_x.append(x.copy())
                history_f.append(f)
                if f < pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest_pos[i] = x.copy()
                    if f < pbest_fit[gbest_idx]:
                        gbest_idx = i
                        gbest_pos = x.copy()

            history_pop.append(pos.copy())

        return self._make_result(history_x, history_f, history_pop)


class DEOptimizer(BaseOptimizer):
    """Differential Evolution / rand/1/bin (Storn & Price, 1997).

    For each target x_i in the current population, three distinct donors
    a, b, c are picked uniformly at random (all ≠ i) and a mutant is formed
    as ``v = x_a + F·(x_b - x_c)``. Binomial crossover with x_i (rate CR,
    plus one guaranteed inherited dimension) yields the trial u, which
    replaces x_i only if it improves on it. Updates are synchronous: all
    donors are drawn from the current generation, all replacements applied
    at gen end.

    DE/rand/1/bin is the canonical single-channel differential baseline
    against which MC-ESO's droplet channel (DE/current-to-best/1 with
    niched-elite pull) is contrasted.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_pop: int = 30,
        F: float = 0.5,        # mutation scale
        CR: float = 0.9,       # binomial crossover rate
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.F = F
        self.CR = CR

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds

        pop = rng.uniform(lo, hi, (self.n_pop, self.dim))
        fit = np.array([self.func(x) for x in pop])

        history_x: list[np.ndarray] = list(pop)
        history_f: list[float] = list(fit)
        history_pop: list[np.ndarray] = [pop.copy()]

        while len(history_f) < max_evals:
            new_pop = pop.copy()
            new_fit = fit.copy()
            for i in range(self.n_pop):
                if len(history_f) >= max_evals:
                    break
                # Three distinct donors ≠ i
                others = np.array([k for k in range(self.n_pop) if k != i])
                a, b, c = rng.choice(others, size=3, replace=False)
                v = pop[a] + self.F * (pop[b] - pop[c])
                v = np.clip(v, lo, hi)
                # Binomial crossover; force ≥1 dim inherited from v
                mask = rng.random(self.dim) < self.CR
                mask[rng.integers(0, self.dim)] = True
                u = np.where(mask, v, pop[i])
                f_u = float(self.func(u))
                history_x.append(u.copy())
                history_f.append(f_u)
                if f_u <= fit[i]:
                    new_pop[i] = u
                    new_fit[i] = f_u
            pop = new_pop
            fit = new_fit
            history_pop.append(pop.copy())

        return self._make_result(history_x, history_f, history_pop)


class SaVOAOptimizer(BaseOptimizer):
    """Self-Adaptive Virus Optimization Algorithm (approx. based on 2020 paper).

    Same structure as VOA but sigma adapts multiplicatively based on whether
    best fitness improved each generation — no manual sigma parameter.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_pop: int = 30,
        strong_ratio: float = 0.2,
        air_ratio: float = 0.2,
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.strong_ratio = strong_ratio
        self.air_ratio = air_ratio

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        span = hi - lo
        sigma = 0.2 * span
        sigma_max = 0.5 * span
        sigma_min = 1e-8

        pop = rng.uniform(lo, hi, (self.n_pop, self.dim))
        fit = np.array([self.func(x) for x in pop])

        history_x: list[np.ndarray] = list(pop)
        history_f: list[float] = list(fit)
        history_pop: list[np.ndarray] = [pop.copy()]

        best_f = float(np.min(fit))

        while len(history_f) < max_evals:
            order = np.argsort(fit)
            n_strong = max(1, int(self.n_pop * self.strong_ratio))
            strong_idx = order[:n_strong]
            common_idx = order[n_strong:]

            offspring_x: list[np.ndarray] = []
            offspring_f: list[float] = []

            for i in strong_idx:
                if len(history_f) + len(offspring_f) >= max_evals:
                    break
                x_new = np.clip(pop[i] + sigma * rng.standard_normal(self.dim), lo, hi)
                f_new = self.func(x_new)
                offspring_x.append(x_new)
                offspring_f.append(f_new)
                history_x.append(x_new.copy())
                history_f.append(f_new)

            for i in common_idx:
                if len(history_f) >= max_evals:
                    break
                if rng.random() > self.air_ratio:
                    j = strong_idx[rng.integers(0, n_strong)]
                    r = rng.random(self.dim)
                    x_new = np.clip(pop[j] + r * (pop[j] - pop[i]), lo, hi)
                else:
                    x_new = rng.uniform(lo, hi, self.dim)
                f_new = self.func(x_new)
                offspring_x.append(x_new)
                offspring_f.append(f_new)
                history_x.append(x_new.copy())
                history_f.append(f_new)

            if offspring_x:
                combined_x = np.vstack([pop, np.array(offspring_x)])
                combined_f = np.concatenate([fit, np.array(offspring_f)])
                top_idx = np.argsort(combined_f)[:self.n_pop]
                pop = combined_x[top_idx]
                fit = combined_f[top_idx]
                history_pop.append(pop.copy())

            new_best = float(np.min(fit))
            if new_best < best_f:
                best_f = new_best
                sigma = min(sigma * 1.2, sigma_max)
            else:
                sigma = max(sigma * 0.9, sigma_min)

        return self._make_result(history_x, history_f, history_pop)


