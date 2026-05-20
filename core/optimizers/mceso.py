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

      • **Spillover event** (stagnation-triggered re-seed): when the outbreak
        stalls (no improvement for ``restart_no_improve_threshold`` evals)
        and ``best_so_far`` still exceeds
        ``restart_quality_rel_floor × |f_init|`` (i.e. the run has not yet
        reduced the initial-population best by ~8 orders of magnitude), the
        population spills over to a fresh host pool around the best with
        σ = σ_init·``restart_sigma_ratio``. After a streak of failed
        spillovers the next event escalates to a full basin switch (best
        discarded, σ reset to σ_init). Quality floors are relative to the
        initial-population best so they remain meaningful under
        multiplicative rescaling of f.

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
        # Airborne σ inflates linearly with population convergence:
        #   factor = 1.5 + air_sigma_amplifier × (1 - diversity_ratio)
        # i.e. factor = 1.5 at full diversity, 1.5+amplifier at full convergence.
        # Default 3.5 reproduces the prior min=1.5, max=5.0 behaviour.
        air_sigma_amplifier: float = 3.5,
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
        # across the search domain and an axis-aligned boundary sweep is performed.
        # Best is preserved unless the streak of failed spillovers triggers a full
        # basin switch below.
        basin_switch_after_failed_spillovers: int = 2,  # streak → wipe best & reset σ
        basin_switch_quality_rel_floor: float = 1e-2,   # basin switch suppressed when
                                                        # best_so_far / |f_init| ≤ this
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
        empirical_cov_floor: float = 0.01,     # min normalized eigenvalue
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.sigma = sigma
        self.air_ratio = air_ratio
        self.n_elite_max = n_elite_max
        self.niche_radius_ratio = niche_radius_ratio
        self.host_sigma_min_scale = host_sigma_min_scale
        self.air_sigma_amplifier = air_sigma_amplifier
        self.log_slope_threshold = log_slope_threshold
        self.h2h_ratio = h2h_ratio
        self.h2h_F = h2h_F
        self.kill_fraction = kill_fraction
        self.restart_no_improve_threshold = restart_no_improve_threshold
        self.restart_sigma_ratio = restart_sigma_ratio
        self.restart_quality_rel_floor = restart_quality_rel_floor
        self.basin_switch_after_failed_spillovers = basin_switch_after_failed_spillovers
        self.basin_switch_quality_rel_floor = basin_switch_quality_rel_floor
        self.sigma_up = sigma_up
        self.sigma_down = sigma_down
        self.sigma_floor_ratio = sigma_floor_ratio
        self.sigma_ceil_ratio = sigma_ceil_ratio
        self.precision_sigma_ratio = precision_sigma_ratio
        self.sigma_drill_down = sigma_drill_down
        self.h2h_CR = h2h_CR
        self.empirical_cov_floor = empirical_cov_floor

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
        """Coordinate-axis boundary probes around x_best. Per dimension,
        probe both bounds.

        Built for **boundary-optimal** landscapes:
          • F05 LinearSlope (optimum on a corner): the {lo, hi} probes land
            on the optimum exactly when the right sign is picked.

        Total candidates = dim × 2. For dim=2 this is 4 evals per sweep.
        """
        cands: list[np.ndarray] = []
        for i in range(self.dim):
            for bv in (lo, hi):
                cand = x_best.copy()
                cand[i] = bv
                cands.append(cand)
        return cands

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

    # ── spillover (stagnation re-seed) ──────────────────────────────────────
    def _maybe_spillover(self, st: _MCESOState) -> bool:
        """Spillover event: when the outbreak stalls, the population spills over
        to a fresh host pool around the global best, giving the search another
        chance to align onto a productive direction (essential for ill-conditioned
        landscapes). Quality-gated to avoid disturbing already-converged runs.

        Returns ``True`` iff the eval budget was exhausted inside the event, in
        which case the caller breaks the generation loop.
        """
        if not (st.no_improve >= self.restart_no_improve_threshold
                and st.best_so_far > self.restart_quality_rel_floor * st.f_init_scale):
            return False

        rng, lo, hi = st.rng, st.lo, st.hi

        # Escalation policy based on consecutive_failed_spillovers:
        #   streak ≥ basin_switch_after: wipe best, fully uniform, σ_init reset
        #   else:                       fully uniform, best preserved
        if (st.consecutive_failed_spillovers >= self.basin_switch_after_failed_spillovers
                and st.best_so_far > self.basin_switch_quality_rel_floor * st.f_init_scale):
            basin_switch = True
            sigma_restart = st.sigma_init   # fresh σ for new basin
        else:
            basin_switch = False
            sigma_restart = st.sigma_init * self.restart_sigma_ratio
        div_ratio = 1.0

        # Coordinate-axis sweep before every spillover. Targets separable /
        # boundary-optimal landscapes that the isotropic uniform re-seed alone
        # fails on (F04 BucheRastrigin, F05 LinearSlope).
        x_best_for_sweep = st.pop_x[int(np.argmin(st.pop_f))].copy()
        for sweep_cand in self._axis_sweep(x_best_for_sweep, lo, hi):
            if not st.budget_left:
                break
            f_sweep = float(self.func(sweep_cand))
            st.history_x.append(sweep_cand.copy())
            st.history_f.append(f_sweep)
            st.history_sigma_eval.append(
                float(np.linalg.norm(sweep_cand - x_best_for_sweep)))
            if f_sweep < st.best_so_far:
                st.best_so_far = f_sweep
                # Inject the better candidate into the population so the upcoming
                # spillover anchors on it (or, on basin switch, so it's part of
                # the historical best).
                worst_i = int(np.argmax(st.pop_f))
                st.pop_x[worst_i] = sweep_cand.copy()
                st.pop_f[worst_i] = f_sweep
                st.pop_age[worst_i] = 0
        if not st.budget_left:
            return True

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

        n_div = int(round(div_ratio * len(reseed_idx)))
        # Shuffle so the assignment of "diversified" vs "local" is random
        rng.shuffle(reseed_idx)
        diversified = set(reseed_idx[:n_div])
        for i in reseed_idx:
            if i in diversified or x_best_snap is None:
                new_x = rng.uniform(lo, hi, self.dim)
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
        if len(elite_arr) > 0:
            elite_pick = rng.choice(elite_arr, size=n_h2h)
            best_pull = st.pop_x[elite_pick] - st.pop_x[h2h_parents_gi]
            h2h_step = self.h2h_F * (best_pull + diff)
        else:
            h2h_step = self.h2h_F * diff
        h2h_trial = st.pop_x[h2h_parents_gi] + h2h_step
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
        # suppressed once drilling mode is entered so precision grinding isn't
        # disrupted.
        in_drilling_now = st.sigma < span * self.precision_sigma_ratio
        air_ratio_eff = 0.0 if in_drilling_now else self.air_ratio
        n_air = max(0, int(round(air_ratio_eff * n_dead)))
        n_h2h = max(0, int(round(self.h2h_ratio * n_dead))) if n >= 3 else 0
        # If rounding overflows, trim airborne first (preserves the
        # droplet/close-contact intent).
        if n_air + n_h2h > n_dead:
            n_air = max(0, n_dead - n_h2h)
        n_local = n_dead - n_air - n_h2h

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
            st.pop_x[slot] = x
            st.pop_f[slot] = f
            st.pop_age[slot] = 0
            replaced_slots.append(slot)
            st.history_x.append(x.copy())
            st.history_f.append(f)
            sigma_used_k = (float(_sigma_children[k])
                            if k < len(_sigma_children) else float(st.sigma))
            st.history_sigma_eval.append(sigma_used_k)
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
            if not st.budget_left:
                break

        # Host-competition rollback: children worse than the host they replaced
        # are reverted, so the outbreak monotonically improves.
        if dead_orig_x is not None:
            for k, slot in enumerate(replaced_slots):
                if dead_orig_f[k] < st.pop_f[slot]:
                    st.pop_x[slot] = dead_orig_x[k]
                    st.pop_f[slot] = dead_orig_f[k]

        # Age active survivors (per-generation)
        replaced_mask = np.zeros(self.n_pop, dtype=bool)
        if replaced_slots:
            replaced_mask[replaced_slots] = True
        st.pop_age[~replaced_mask] += 1

        # σ adaptation always on: improved → × sigma_up, else → × sigma_down
        # (or sigma_drill_down in drilling mode).
        in_drilling = st.sigma < span * self.precision_sigma_ratio
        sigma_floor_eff = span * self.sigma_floor_ratio
        if st.best_so_far < gen_best_before:
            st.sigma *= self.sigma_up
        else:
            st.sigma *= self.sigma_drill_down if in_drilling else self.sigma_down
        st.sigma = max(sigma_floor_eff, min(st.sigma, span * self.sigma_ceil_ratio))

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
