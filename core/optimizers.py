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
    # VSO-specific dynamics (one entry per generation; empty for non-VSO)
    history_sigma_global: list[float] = field(default_factory=list)
    history_n_active: list[int] = field(default_factory=list)
    history_n_elite: list[int] = field(default_factory=list)
    history_no_improve: list[int] = field(default_factory=list)
    history_elite_cutoff: list[float] = field(default_factory=list)
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


class VirusOptimizer(BaseOptimizer):
    """Epidemic-inspired optimizer.

    Individuals (infected hosts) spread locally or via air transmission.
    Low f(x) → high infection probability. Elites are immortal.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_pop: int = 20,
        lifespan: int = 5,
        sigma: float = 0.2,
        sigma_decay: float = 0.99,
        air_ratio: float = 0.2,
        n_elite_max: int = 6,
        temperature: float = 1.0,
        stagnation_limit: int = 2000,
        niche_radius: float = 1.0,
        niche_radius_min: float = 0.05,
        elite_quality_factor: float = 1.0,
        sigma_min_ratio: float = 0.05,
        air_sigma_min: float = 1.5,
        air_sigma_max: float = 5.0,
        n_pop_min: int = 5,
        pop_change_by: int = 2,
        pop_grow_trigger: int = 200,
        pop_shrink_trigger: int = 20,
        pop_change_cooldown: int = 30,
        lifespan_range: int = 4,
        dormant_mode: str = "freeze",
        air_noise: str = "normal",
        adaptive_air_ratio: bool = False,
        log_slope_threshold: float = 1e-4,
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.lifespan = lifespan
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.air_ratio = air_ratio
        self.n_elite_max = n_elite_max
        self.temperature = temperature
        self.stagnation_limit = stagnation_limit
        self.niche_radius = niche_radius
        self.niche_radius_min = niche_radius_min
        self.elite_quality_factor = elite_quality_factor
        self.sigma_min_ratio = sigma_min_ratio
        self.air_sigma_min = air_sigma_min
        self.air_sigma_max = air_sigma_max
        self.n_pop_min = n_pop_min
        self.pop_change_by = pop_change_by
        self.pop_grow_trigger = pop_grow_trigger
        self.pop_shrink_trigger = pop_shrink_trigger
        self.pop_change_cooldown = pop_change_cooldown
        self.lifespan_range = lifespan_range
        self.dormant_mode = dormant_mode  # "freeze" | "aging" | "replace"
        self.air_noise = air_noise            # "uniform" | "normal"
        self.adaptive_air_ratio = adaptive_air_ratio
        self.log_slope_threshold = log_slope_threshold

    @staticmethod
    def _reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Reflect out-of-bounds values back into [lo, hi] instead of clamping."""
        span = hi - lo
        x_rel = (x - lo) % (2 * span)
        x_rel = np.where(x_rel > span, 2 * span - x_rel, x_rel)
        return x_rel + lo

    def _sample_lifespans(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.lifespan_range <= 0:
            return np.full(n, self.lifespan, dtype=int)
        lo_ls = max(1, self.lifespan - self.lifespan_range)
        hi_ls = self.lifespan + self.lifespan_range
        return rng.integers(lo_ls, hi_ls + 1, size=n)

    def _niche_elites(self, pop_x: np.ndarray, pop_f: np.ndarray,
                      niche_radius: float | None = None,
                      best_f_global: float | None = None,
                      eqf_override: float | None = None) -> set:
        """Dynamically select spatially diverse elites that meet a quality threshold."""
        if niche_radius is None:
            niche_radius = self.niche_radius
        n_elite_max = self.n_elite_max
        f_ref = best_f_global if best_f_global is not None else pop_f.min()
        eqf = eqf_override if eqf_override is not None else self.elite_quality_factor
        quality_cutoff = f_ref + eqf * max(f_ref, 1e-8)

        elite_idx: list[int] = []
        elite_pos: list[np.ndarray] = []
        for candidate in np.argsort(pop_f):
            if pop_f[candidate] > quality_cutoff:
                break
            if not elite_pos or np.all(
                np.linalg.norm(pop_x[candidate] - np.array(elite_pos), axis=1) > niche_radius
            ):
                elite_idx.append(int(candidate))
                elite_pos.append(pop_x[candidate])
            if len(elite_idx) >= n_elite_max:
                break
        return set(elite_idx)

    def _softmax_weights(self, f_vals: np.ndarray) -> np.ndarray:
        f_max = f_vals.max()
        scores = (f_max - f_vals) / (self.temperature + 1e-30)
        scores -= scores.max()
        w = np.exp(scores)
        return w / w.sum()

    def _meaningful_improvement(self, f: float, log_best_ref: float, evals_since_reset: int) -> bool:
        if evals_since_reset == 0:
            return True
        slope = (log_best_ref - math.log10(f + 1e-300)) / evals_since_reset
        return slope >= self.log_slope_threshold

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        span = hi - lo
        sigma = self.sigma * span
        sigma_init_mean = float(np.mean(sigma))

        adaptive = self.n_pop_min > 0 and self.n_pop_min < self.n_pop

        # Pre-allocate numpy arrays — avoids repeated list→array conversions per iteration
        pop_x = rng.uniform(lo, hi, (self.n_pop, self.dim))          # (n_pop, dim)
        pop_f = np.array([self.func(x) for x in pop_x])              # (n_pop,)
        pop_lifespan = self._sample_lifespans(rng, self.n_pop)        # (n_pop,) int
        # Stochastic lifespan: individual variation desynchronises deaths → age=0 OK.
        pop_age = np.zeros(self.n_pop, dtype=int)
        pop_active = np.ones(self.n_pop, dtype=bool)

        history_x: list[np.ndarray] = [row.copy() for row in pop_x]
        history_f: list[float] = pop_f.tolist()
        history_pop: list[np.ndarray] = [pop_x.copy()]
        _lf0 = np.log10(pop_f + 1e-10)
        _lq0 = np.clip((_lf0.max() - _lf0) / (float(_lf0.max() - _lf0.min()) + 1e-30), 0.0, 1.0)
        _a0  = np.minimum(pop_age.astype(float) / np.maximum(pop_lifespan.astype(float), 1), 1.0)
        _s0  = self.sigma_min_ratio ** (_lq0 * (0.7 + 0.3 * _a0))
        history_pop_sigma: list[np.ndarray] = [float(sigma) * _s0]
        history_sigma_global: list[float] = []
        history_n_active: list[int] = []
        history_n_elite: list[int] = []
        history_no_improve: list[int] = []
        history_elite_cutoff: list[float] = []
        history_eval_count: list[int] = []
        history_sigma_eval: list[float] = []

        best_so_far = float(pop_f.min())
        no_improve = self.pop_shrink_trigger  # neutral start
        pop_cooldown = self.pop_change_cooldown
        log_best_ref = math.log10(best_so_far + 1e-300)  # log10(f) at last meaningful reset
        evals_since_reset = 0                             # evals elapsed since last meaningful reset

        while len(history_f) < max_evals:
            active_idx = np.where(pop_active)[0]
            n = len(active_idx)
            pop_x_arr = pop_x[active_idx]
            pop_f_arr = pop_f[active_idx]

            sigma_mean = float(np.mean(sigma))
            sigma_ratio = sigma_mean / sigma_init_mean
            niche_radius_dyn = max(self.niche_radius_min, self.niche_radius * sigma_ratio)
            top5_x = pop_x_arr[np.argsort(pop_f_arr)[:min(5, n)]]
            top5_spread = float(np.mean(np.std(top5_x, axis=0))) / span
            # Unimodal detection: only collapse niches when near the global optimum (f≈0),
            # not when merely clustered at a local optimum (multimodal functions).
            unimodal_converged = top5_spread < 0.05 and best_so_far < 1e-3
            niche_radius_eff = self.niche_radius_min if unimodal_converged else niche_radius_dyn
            # Dynamic elite_quality_factor: narrow (strict) when dispersed, wide (lenient) when clustered
            pop_diversity = np.mean(np.std(pop_x_arr, axis=0) / span)
            diversity_ratio = float(np.clip(pop_diversity / 0.289, 0.0, 1.0))
            eqf_lo = self.elite_quality_factor * 0.3  # dispersed → strict cutoff
            eqf_hi = self.elite_quality_factor * 3.0  # clustered → lenient cutoff
            eqf_eff = eqf_hi - (eqf_hi - eqf_lo) * diversity_ratio
            # elite_local: indices into active_idx; elite_global: indices into full pop arrays
            elite_local = self._niche_elites(pop_x_arr, pop_f_arr, niche_radius_eff,
                                             best_f_global=best_so_far, eqf_override=eqf_eff)
            elite_global = {active_idx[i] for i in elite_local}
            elite_arr = np.fromiter(elite_global, dtype=int) if elite_global else np.empty(0, dtype=int)

            # Dead: aged-out active non-elites
            not_elite_mask = ~np.isin(active_idx, elite_arr)
            aged_out = pop_age[active_idx] > pop_lifespan[active_idx]
            dead_global = active_idx[aged_out & not_elite_mask]
            n_dead = len(dead_global)

            stagnation_ratio = min(no_improve / max(self.stagnation_limit, 1), 1.0)

            if n_dead == 0:
                pop_age[active_idx] += 1
                if self.dormant_mode == "aging":
                    pop_age[~pop_active] += 1
                sigma *= self.sigma_decay
                # When all active individuals are elite, no births occur and
                # history_f never grows → stagnation counter must still advance
                # so the pop_grow_trigger and stagnation_limit eventually fire.
                no_improve += 1
                if no_improve >= self.stagnation_limit:
                    break
            else:
                weights = self._softmax_weights(pop_f_arr)

                if self.adaptive_air_ratio:
                    air_ratio_lo = self.air_ratio * 0.5
                    air_ratio_hi = min(0.7, self.air_ratio * 3.0)
                    air_ratio_eff = air_ratio_lo + (air_ratio_hi - air_ratio_lo) * stagnation_ratio
                else:
                    air_ratio_eff = self.air_ratio + (0.5 - self.air_ratio) * stagnation_ratio ** 2
                n_air = max(0, round(air_ratio_eff * n_dead))
                n_local = n_dead - n_air

                # Log-scale quality anchored to global best (history-wide).
                # When population converges to a local optimum, all f_i ≈ f_pop_max
                # but best_so_far may be far better → lq ≈ 0 → σ_i = σ_global (full exploration).
                pop_log_f = np.log10(pop_f_arr + 1e-10)
                log_f_max = float(pop_log_f.max())
                log_f_best = float(np.log10(best_so_far + 1e-10))  # anchored to history best
                log_f_spread = log_f_max - log_f_best

                # Air sigma: large when converged (need to escape), small when diverse
                # diversity_ratio already computed above before elite selection
                air_sigma_factor = self.air_sigma_max - (self.air_sigma_max - self.air_sigma_min) * diversity_ratio
                air_sigma_base = np.maximum(sigma, sigma_init_mean * 0.3)
                air_sigma_vec = air_sigma_base * air_sigma_factor

                # Batch generate all children before evaluation loop
                if n_local > 0:
                    local_li = rng.choice(n, size=n_local, p=weights)
                    gi_arr = active_idx[local_li]
                    lq = np.clip(
                        (log_f_max - np.log10(pop_f[gi_arr] + 1e-10)) / (log_f_spread + 1e-30),
                        0.0, 1.0)
                    ar = np.minimum(
                        pop_age[gi_arr].astype(float) / np.maximum(pop_lifespan[gi_arr].astype(float), 1),
                        1.0)
                    sigma_i = sigma * (self.sigma_min_ratio ** (lq * (0.7 + 0.3 * ar)))
                    noise = rng.standard_normal((n_local, self.dim))
                    new_local = self._reflect(pop_x[gi_arr] + noise * sigma_i[:, None], lo, hi)
                else:
                    new_local = np.empty((0, self.dim))

                if n_air > 0:
                    air_li = rng.integers(0, n, size=n_air)
                    if self.air_noise == "uniform":
                        noise_air = rng.uniform(-1.0, 1.0, (n_air, self.dim))
                    else:
                        noise_air = rng.standard_normal((n_air, self.dim))
                    new_air = self._reflect(pop_x[active_idx[air_li]] + noise_air * air_sigma_vec, lo, hi)
                else:
                    new_air = np.empty((0, self.dim))

                new_xs = np.concatenate([new_local, new_air], axis=0)

                # Per-child sigma: local children use individual sigma_i, air children use air_sigma_vec
                _sc: list[np.ndarray] = []
                if n_local > 0:
                    _sc.append(sigma_i)
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
                    history_sigma_eval.append(float(_sigma_children[k]) if k < len(_sigma_children) else float(sigma))
                    evals_since_reset += 1
                    if f < best_so_far:
                        best_so_far = f
                        if self._meaningful_improvement(f, log_best_ref, evals_since_reset):
                            no_improve = 0
                            log_best_ref = math.log10(f + 1e-300)
                            evals_since_reset = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                    if len(history_f) >= max_evals or no_improve >= self.stagnation_limit:
                        break

                if replaced_slots:
                    pop_lifespan[replaced_slots] = self._sample_lifespans(rng, len(replaced_slots))

                if no_improve >= self.stagnation_limit:
                    break

                # Age active survivors (per-generation)
                replaced_mask = np.zeros(self.n_pop, dtype=bool)
                if replaced_slots:
                    replaced_mask[replaced_slots] = True
                pop_age[active_idx[~replaced_mask[active_idx]]] += 1
                if self.dormant_mode == "aging":
                    pop_age[~pop_active] += 1

                sigma *= self.sigma_decay

            # Virtual breathing: deactivate when improving, reactivate when stagnating
            if adaptive and pop_cooldown == 0:
                n_active = int(pop_active.sum())
                dormant_idx = np.where(~pop_active)[0]
                if no_improve >= self.pop_grow_trigger and n_active < self.n_pop:
                    # Reactivate best dormant individuals (2x change_by).
                    best_dormant = dormant_idx[np.argsort(pop_f[dormant_idx])]
                    n_react = min(self.pop_change_by * 2, self.n_pop - n_active, len(best_dormant))
                    if n_react > 0:
                        slots = best_dormant[:n_react]
                        if self.dormant_mode == "aging":
                            # Aged-out dormant individuals get fresh random positions on wakeup.
                            aged_out_slots = slots[pop_age[slots] > pop_lifespan[slots]]
                            if len(aged_out_slots) > 0:
                                new_xs = rng.uniform(lo, hi, (len(aged_out_slots), self.dim))
                                for i, slot in enumerate(aged_out_slots):
                                    if len(history_f) >= max_evals:
                                        break
                                    pop_x[slot] = new_xs[i]
                                    pop_f[slot] = float(self.func(pop_x[slot]))
                                    pop_lifespan[slot] = self._sample_lifespans(rng, 1)[0]
                                    history_x.append(pop_x[slot].copy())
                                    history_f.append(pop_f[slot])
                                    history_sigma_eval.append(float('nan'))  # random position, no parent sigma
                                    evals_since_reset += 1
                                    if pop_f[slot] < best_so_far:
                                        best_so_far = pop_f[slot]
                                        if self._meaningful_improvement(pop_f[slot], log_best_ref, evals_since_reset):
                                            no_improve = 0
                                            log_best_ref = math.log10(pop_f[slot] + 1e-300)
                                            evals_since_reset = 0
                                        else:
                                            no_improve += 1
                                    else:
                                        no_improve += 1
                        elif self.dormant_mode == "replace":
                            # Discard dormant position; generate offspring from current active parents.
                            wts_react = self._softmax_weights(pop_f[active_idx])
                            for slot in slots:
                                if len(history_f) >= max_evals:
                                    break
                                parent_local = rng.choice(n, p=wts_react)
                                new_x = self._reflect(
                                    pop_x[active_idx[parent_local]] + rng.standard_normal(self.dim) * sigma,
                                    lo, hi,
                                )
                                new_f = float(self.func(new_x))
                                pop_x[slot] = new_x
                                pop_f[slot] = new_f
                                pop_lifespan[slot] = self._sample_lifespans(rng, 1)[0]
                                history_x.append(new_x.copy())
                                history_f.append(new_f)
                                history_sigma_eval.append(float(sigma))  # replace mode uses global sigma
                                evals_since_reset += 1
                                if new_f < best_so_far:
                                    best_so_far = new_f
                                    if self._meaningful_improvement(new_f, log_best_ref, evals_since_reset):
                                        no_improve = 0
                                        log_best_ref = math.log10(new_f + 1e-300)
                                        evals_since_reset = 0
                                    else:
                                        no_improve += 1
                                else:
                                    no_improve += 1
                        pop_active[slots] = True
                        pop_age[slots] = 0
                        pop_cooldown = self.pop_change_cooldown
                elif no_improve < self.pop_shrink_trigger and n_active > self.n_pop_min:
                    # Deactivate worst non-elite active individuals (virtual shrink)
                    pool = active_idx[~np.isin(active_idx, elite_arr)]
                    deactivatable = pool[np.argsort(pop_f[pool])[::-1]]
                    n_deact = min(self.pop_change_by, n_active - self.n_pop_min, len(deactivatable))
                    if n_deact > 0:
                        pop_active[deactivatable[:n_deact]] = False
                        pop_cooldown = self.pop_change_cooldown
            else:
                pop_cooldown = max(0, pop_cooldown - 1)

            pf_end = pop_f[active_idx]
            pa_end = pop_age[active_idx].astype(float)
            lf_end = np.log10(pf_end + 1e-10)
            lq_e = np.clip((lf_end.max() - lf_end) / (float(lf_end.max() - lf_end.min()) + 1e-30), 0.0, 1.0)
            ar_e = np.minimum(pa_end / max(self.lifespan, 1), 1.0)
            scale_e = self.sigma_min_ratio ** (lq_e * (0.7 + 0.3 * ar_e))
            history_pop_sigma.append(float(sigma) * scale_e)
            history_pop.append(pop_x.copy())
            # --- VSO dynamics recording ---
            history_sigma_global.append(float(sigma))
            history_n_active.append(int(pop_active.sum()))
            history_n_elite.append(len(elite_global))
            history_no_improve.append(int(no_improve))
            history_elite_cutoff.append(
                best_so_far + eqf_eff * max(best_so_far, 1e-8)
            )
            history_eval_count.append(len(history_f))

        result = self._make_result(history_x, history_f, history_pop)
        result.history_pop_sigma = history_pop_sigma
        result.history_sigma_global = history_sigma_global
        result.history_n_active = history_n_active
        result.history_n_elite = history_n_elite
        result.history_no_improve = history_no_improve
        result.history_elite_cutoff = history_elite_cutoff
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


class GAOptimizer(BaseOptimizer):
    """Real-valued Genetic Algorithm with SBX crossover and polynomial mutation."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_pop: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        eta_c: float = 20.0,   # SBX distribution index
        eta_m: float = 20.0,   # polynomial mutation distribution index
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eta_c = eta_c
        self.eta_m = eta_m

    def _sbx(self, rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray,
             lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.dim):
            if rng.random() > 0.5:
                continue
            if abs(p1[i] - p2[i]) < 1e-14:
                continue
            y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            beta_l = 1 + 2 * (y1 - lo) / (y2 - y1)
            beta_r = 1 + 2 * (hi - y2) / (y2 - y1)
            alpha_l = 2 - beta_l ** (-(self.eta_c + 1))
            alpha_r = 2 - beta_r ** (-(self.eta_c + 1))
            u = rng.random()
            if u <= 1 / alpha_l:
                beta_q = (u * alpha_l) ** (1 / (self.eta_c + 1))
            else:
                beta_q = (1 / (2 - u * alpha_l)) ** (1 / (self.eta_c + 1))
            c1[i] = 0.5 * ((p1[i] + p2[i]) - beta_q * (y2 - y1))
            u = rng.random()
            if u <= 1 / alpha_r:
                beta_q = (u * alpha_r) ** (1 / (self.eta_c + 1))
            else:
                beta_q = (1 / (2 - u * alpha_r)) ** (1 / (self.eta_c + 1))
            c2[i] = 0.5 * ((p1[i] + p2[i]) + beta_q * (y2 - y1))
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _poly_mutate(self, rng: np.random.Generator, x: np.ndarray,
                     lo: float, hi: float) -> np.ndarray:
        x = x.copy()
        for i in range(self.dim):
            if rng.random() > self.mutation_rate:
                continue
            delta_l = (x[i] - lo) / (hi - lo)
            delta_r = (hi - x[i]) / (hi - lo)
            u = rng.random()
            if u < 0.5:
                delta_q = (2 * u + (1 - 2 * u) * (1 - delta_l) ** (self.eta_m + 1)) ** (1 / (self.eta_m + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta_r) ** (self.eta_m + 1)) ** (1 / (self.eta_m + 1))
            x[i] = np.clip(x[i] + delta_q * (hi - lo), lo, hi)
        return x

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds

        pop = rng.uniform(lo, hi, (self.n_pop, self.dim))
        fit = np.array([self.func(x) for x in pop])

        history_x: list[np.ndarray] = list(pop)
        history_f: list[float] = list(fit)
        history_pop: list[np.ndarray] = [pop.copy()]

        while len(history_f) < max_evals:
            # Tournament selection (k=2)
            def tournament(n: int) -> np.ndarray:
                a = rng.integers(0, self.n_pop, n)
                b = rng.integers(0, self.n_pop, n)
                return np.where(fit[a] <= fit[b], a, b)

            offspring = []
            parents = tournament(self.n_pop)
            rng.shuffle(parents)

            for k in range(0, self.n_pop - 1, 2):
                p1, p2 = pop[parents[k]], pop[parents[k + 1]]
                if rng.random() < self.crossover_rate:
                    c1, c2 = self._sbx(rng, p1, p2, lo, hi)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                offspring.append(self._poly_mutate(rng, c1, lo, hi))
                offspring.append(self._poly_mutate(rng, c2, lo, hi))

            # Evaluate offspring
            new_fit = []
            for x in offspring:
                if len(history_f) >= max_evals:
                    break
                f = self.func(x)
                history_x.append(x.copy())
                history_f.append(f)
                new_fit.append(f)

            # Elitist replacement: combine parents + offspring, keep best n_pop
            if new_fit:
                combined_x = np.vstack([pop, np.array(offspring[:len(new_fit)])])
                combined_f = np.concatenate([fit, np.array(new_fit)])
                best_idx = np.argsort(combined_f)[:self.n_pop]
                pop = combined_x[best_idx]
                fit = combined_f[best_idx]
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


