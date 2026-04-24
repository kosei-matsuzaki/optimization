from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
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
        pop_change_by: int = 5,
        pop_grow_trigger: int = 200,
        pop_shrink_trigger: int = 20,
        pop_change_cooldown: int = 30,
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

    @staticmethod
    def _reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Reflect out-of-bounds values back into [lo, hi] instead of clamping."""
        span = hi - lo
        x_rel = (x - lo) % (2 * span)
        x_rel = np.where(x_rel > span, 2 * span - x_rel, x_rel)
        return x_rel + lo

    def _niche_elites(self, pop_x: np.ndarray, pop_f: np.ndarray,
                      niche_radius: float | None = None) -> set:
        """Dynamically select spatially diverse elites that meet a quality threshold."""
        if niche_radius is None:
            niche_radius = self.niche_radius
        n_elite_max = self.n_elite_max
        f_best = pop_f.min()
        f_spread = np.percentile(pop_f, 75) - f_best
        quality_cutoff = f_best + self.elite_quality_factor * max(f_spread, 1e-30)

        elite_idx: set = set()
        for candidate in np.argsort(pop_f):
            if pop_f[candidate] > quality_cutoff:
                break
            if not elite_idx or all(
                np.linalg.norm(pop_x[candidate] - pop_x[e]) > niche_radius
                for e in elite_idx
            ):
                elite_idx.add(int(candidate))
            if len(elite_idx) >= n_elite_max:
                break
        return elite_idx

    def _softmax_weights(self, f_vals: np.ndarray) -> np.ndarray:
        f_max = f_vals.max()
        scores = (f_max - f_vals) / (self.temperature + 1e-30)
        scores -= scores.max()
        w = np.exp(scores)
        return w / w.sum()

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        span = hi - lo
        sigma = self.sigma * span
        sigma_init_mean = float(np.mean(sigma))

        # Virtual breathing: n_pop_min > 0 enables active-mask mode
        adaptive = self.n_pop_min > 0 and self.n_pop_min < self.n_pop

        # Internal storage as lists for variable-length population
        init_x = rng.uniform(lo, hi, (self.n_pop, self.dim))
        init_f = np.array([self.func(x) for x in init_x])
        init_age = rng.integers(0, self.lifespan, size=self.n_pop)

        pop_x: list[np.ndarray] = [init_x[i].copy() for i in range(self.n_pop)]
        pop_f: list[float] = list(init_f)
        pop_age: list[int] = list(init_age)
        pop_active: list[bool] = [True] * self.n_pop

        history_x: list[np.ndarray] = list(init_x)
        history_f: list[float] = list(init_f)
        history_pop: list[np.ndarray] = [init_x.copy()]
        _lf0 = np.log10(init_f + 1e-10)
        _lq0 = np.clip((_lf0.max() - _lf0) / (float(_lf0.max() - _lf0.min()) + 1e-30), 0.0, 1.0)
        _a0  = np.minimum(init_age / max(self.lifespan, 1), 1.0)
        _s0  = self.sigma_min_ratio ** (_lq0 * (0.7 + 0.3 * _a0))
        history_pop_sigma: list[np.ndarray] = [float(sigma) * _s0]

        best_so_far = float(np.min(init_f))
        no_improve = self.pop_shrink_trigger  # neutral start
        pop_cooldown = self.pop_change_cooldown  # warmup: no size change for first N iters

        while len(history_f) < max_evals:
            # Work only with active individuals
            active_idx = [i for i in range(len(pop_x)) if pop_active[i]]
            n = len(active_idx)
            pop_x_arr = np.array([pop_x[i] for i in active_idx])
            pop_f_arr = np.array([pop_f[i] for i in active_idx])

            sigma_mean = float(np.mean(sigma))
            sigma_ratio = sigma_mean / sigma_init_mean
            niche_radius_dyn = max(self.niche_radius_min, self.niche_radius * sigma_ratio)
            top5_x = pop_x_arr[np.argsort(pop_f_arr)[:min(5, n)]]
            top5_spread = float(np.mean(np.std(top5_x, axis=0))) / span
            # Unimodal detection: only collapse niches when near the global optimum (f≈0),
            # not when merely clustered at a local optimum (multimodal functions).
            unimodal_converged = top5_spread < 0.05 and best_so_far < 1e-3
            niche_radius_eff = self.niche_radius_min if unimodal_converged else niche_radius_dyn
            # elite_local: indices into active_idx; elite_global: indices into full pop lists
            elite_local = self._niche_elites(pop_x_arr, pop_f_arr, niche_radius_eff)
            elite_global = {active_idx[i] for i in elite_local}

            # Mark dead: aged-out active non-elites
            dead_global = [
                active_idx[i] for i in range(n)
                if pop_age[active_idx[i]] > self.lifespan and active_idx[i] not in elite_global
            ]
            n_dead = len(dead_global)

            if n_dead == 0:
                for i in active_idx:
                    pop_age[i] += 1
                sigma *= self.sigma_decay
            else:
                weights = self._softmax_weights(pop_f_arr)

                stagnation_ratio = min(no_improve / max(self.stagnation_limit, 1), 1.0)
                air_ratio_eff = self.air_ratio + (0.5 - self.air_ratio) * stagnation_ratio ** 2
                n_air = max(0, round(air_ratio_eff * n_dead))
                n_local = n_dead - n_air

                # Log-scale quality: distinguishes f=0.001 vs f=0.01 far better than linear.
                # Bad individuals (log_quality≈0) always get scale=1 (full exploration),
                # so they can escape local optima regardless of population ranking.
                pop_log_f = np.log10(pop_f_arr + 1e-10)
                log_f_max = float(pop_log_f.max())
                log_f_spread = float(log_f_max - pop_log_f.min())

                # Air sigma: large when converged (need to escape), small when diverse
                # Normalized diversity: uniform distribution gives std ≈ 0.289 * span
                pop_diversity = np.mean(np.std(pop_x_arr, axis=0) / span)
                diversity_ratio = np.clip(pop_diversity / 0.289, 0.0, 1.0)

                air_sigma_factor = self.air_sigma_max - (self.air_sigma_max - self.air_sigma_min) * diversity_ratio
                air_sigma_base = np.maximum(sigma, sigma_init_mean * 0.3)
                air_sigma_vec = air_sigma_base * air_sigma_factor

                new_xs: list[np.ndarray] = []
                if n_local > 0:
                    local_idx_local = rng.choice(n, size=n_local, p=weights)
                    for li in local_idx_local:
                        gi = active_idx[li]
                        log_quality = float(np.clip(
                            (log_f_max - np.log10(pop_f_arr[li] + 1e-10)) / (log_f_spread + 1e-30),
                            0.0, 1.0))
                        age_ratio = min(pop_age[gi] / max(self.lifespan, 1), 1.0)
                        combined = log_quality * (0.7 + 0.3 * age_ratio)
                        scale = self.sigma_min_ratio ** combined
                        sigma_i = sigma * scale
                        child = pop_x[gi] + rng.normal(0, sigma_i, self.dim)
                        child = self._reflect(child, lo, hi)
                        new_xs.append(child)
                for _ in range(n_air):
                    li = int(rng.integers(0, n))
                    gi = active_idx[li]
                    child = pop_x[gi] + rng.normal(0, air_sigma_vec, self.dim)
                    child = self._reflect(child, lo, hi)
                    new_xs.append(child)

                # Place offspring into dead slots
                replaced_global: set[int] = set()
                for slot, x in zip(dead_global, new_xs):
                    f = self.func(x)
                    pop_x[slot] = x
                    pop_f[slot] = f
                    pop_age[slot] = 0
                    replaced_global.add(slot)
                    history_x.append(x.copy())
                    history_f.append(f)

                    if f < best_so_far:
                        best_so_far = f
                        no_improve = 0
                    else:
                        no_improve += 1

                    if len(history_f) >= max_evals or no_improve >= self.stagnation_limit:
                        break

                if no_improve >= self.stagnation_limit:
                    break

                # Age active survivors
                for i in active_idx:
                    if i not in replaced_global:
                        pop_age[i] += 1

                sigma *= self.sigma_decay

            # Virtual breathing: deactivate when improving, reactivate when stagnating
            if adaptive and pop_cooldown == 0:
                n_active = sum(pop_active)
                dormant_idx = [i for i in range(len(pop_x)) if not pop_active[i]]
                if no_improve >= self.pop_grow_trigger and n_active < self.n_pop:
                    # Reactivate best dormant individuals (2x change_by).
                    best_dormant = sorted(dormant_idx, key=lambda i: pop_f[i])
                    n_react = min(self.pop_change_by * 2, self.n_pop - n_active, len(best_dormant))
                    if n_react > 0:
                        for i in best_dormant[:n_react]:
                            pop_active[i] = True
                            pop_age[i] = 0
                        # Evict top active stuck near a local optimum.
                        pool = [i for i in active_idx if i not in elite_global]
                        evict = sorted(pool, key=lambda i: pop_f[i])[:self.pop_change_by]
                        for i in evict:
                            pop_active[i] = False
                        # Do NOT reset no_improve — let it decay naturally as new individuals improve.
                        pop_cooldown = self.pop_change_cooldown
                elif no_improve < self.pop_shrink_trigger and n_active > self.n_pop_min:
                    # Deactivate worst non-elite active individuals (virtual shrink)
                    cur_elite = elite_global
                    deactivatable = sorted(
                        [i for i in active_idx if i not in cur_elite],
                        key=lambda i: pop_f[i], reverse=True
                    )
                    n_deact = min(self.pop_change_by, n_active - self.n_pop_min, len(deactivatable))
                    if n_deact > 0:
                        for i in deactivatable[:n_deact]:
                            pop_active[i] = False
                        pop_cooldown = self.pop_change_cooldown
            else:
                pop_cooldown = max(0, pop_cooldown - 1)

            pf_end   = np.array([pop_f[i]   for i in active_idx])
            pa_end   = np.array([pop_age[i] for i in active_idx], dtype=float)
            lf_end   = np.log10(pf_end + 1e-10)
            lq_e     = np.clip((lf_end.max() - lf_end) / (float(lf_end.max() - lf_end.min()) + 1e-30), 0.0, 1.0)
            ar_e     = np.minimum(pa_end / max(self.lifespan, 1), 1.0)
            combined_e = lq_e * (0.7 + 0.3 * ar_e)
            scale_e    = self.sigma_min_ratio ** combined_e
            history_pop_sigma.append(float(sigma) * scale_e)
            history_pop.append(np.array(pop_x).copy())

        result = self._make_result(history_x, history_f, history_pop)
        result.history_pop_sigma = history_pop_sigma
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


