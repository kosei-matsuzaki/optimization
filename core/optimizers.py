from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    Low f(x) → high infection probability.
    Population fluctuates within [n_pop, n_pop_max]; elites are immortal.
    Crowding control scatters excess individuals clustered near optima.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_pop: int = 20,
        n_pop_max: int = 40,
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
        bloom_size: int = 2,
        crowd_radius_frac: float = 1e-5,
        max_crowd_fraction: float = 0.10,
        sigma_min_ratio: float = 0.10,
        air_sigma_min: float = 1.5,
        air_sigma_max: float = 5.0,
    ):
        super().__init__(benchmark, seed)
        self.n_pop = n_pop
        self.n_pop_max = n_pop_max
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
        self.bloom_size = bloom_size
        self.crowd_radius_frac = crowd_radius_frac
        self.max_crowd_fraction = max_crowd_fraction
        self.sigma_min_ratio = sigma_min_ratio
        self.air_sigma_min = air_sigma_min
        self.air_sigma_max = air_sigma_max

    def _niche_elites(self, pop_x: np.ndarray, pop_f: np.ndarray,
                      niche_radius: float | None = None) -> set:
        """Dynamically select spatially diverse elites that meet a quality threshold."""
        if niche_radius is None:
            niche_radius = self.niche_radius
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
            if len(elite_idx) >= self.n_elite_max:
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

        # Internal storage as lists for variable-length population
        init_x = rng.uniform(lo, hi, (self.n_pop, self.dim))
        init_f = np.array([self.func(x) for x in init_x])
        init_age = rng.integers(0, self.lifespan, size=self.n_pop)

        pop_x: list[np.ndarray] = [init_x[i].copy() for i in range(self.n_pop)]
        pop_f: list[float] = list(init_f)
        pop_age: list[int] = list(init_age)

        history_x: list[np.ndarray] = list(init_x)
        history_f: list[float] = list(init_f)
        history_pop: list[np.ndarray] = [init_x.copy()]

        best_so_far = float(np.min(init_f))
        no_improve = 0
        force_air_slots: set[int] = set()
        sigma_d = np.ones(self.dim)

        while len(history_f) < max_evals:
            n = len(pop_x)
            pop_x_arr = np.array(pop_x)
            pop_f_arr = np.array(pop_f)

            k_top = max(2, n // 4)
            top_k_x = pop_x_arr[np.argsort(pop_f_arr)[:k_top]]
            dim_var = np.var(top_k_x, axis=0)
            mean_var = float(np.mean(dim_var)) + 1e-30
            sigma_d_target = np.clip(np.sqrt(dim_var / mean_var), 0.1, 10.0)
            sigma_d = sigma_d * 0.95 + sigma_d_target * 0.05
            sigma_d /= (float(sigma_d.mean()) + 1e-30)

            sigma_mean = float(np.mean(sigma))
            sigma_ratio = sigma_mean / sigma_init_mean
            niche_radius_dyn = max(self.niche_radius_min, self.niche_radius * sigma_ratio)
            crowd_radius = self.crowd_radius_frac * sigma_mean

            top5_x = pop_x_arr[np.argsort(pop_f_arr)[:min(5, n)]]
            top5_spread = float(np.mean(np.std(top5_x, axis=0))) / span
            niche_radius_eff = self.niche_radius_min if top5_spread < 0.05 else niche_radius_dyn
            elite_idx = self._niche_elites(pop_x_arr, pop_f_arr, niche_radius_eff)

            # Mark dead: age exceeded lifespan, excluding elites
            dead_idx = [
                i for i in range(n)
                if pop_age[i] > self.lifespan and i not in elite_idx
            ]
            n_dead = len(dead_idx)

            if n_dead == 0:
                for i in range(n):
                    pop_age[i] += 1
                sigma *= self.sigma_decay
                continue

            weights = self._softmax_weights(pop_f_arr)

            stagnation_ratio = min(no_improve / max(self.stagnation_limit, 1), 1.0)
            air_ratio_eff = self.air_ratio + (0.5 - self.air_ratio) * stagnation_ratio ** 2
            n_air = max(0, round(air_ratio_eff * n_dead))
            n_local = n_dead - n_air

            f_min_local = pop_f_arr.min()
            f_max_local = pop_f_arr.max()
            f_spread_local = f_max_local - f_min_local

            # Air sigma: large when converged (need to escape), small when diverse
            # Normalized diversity: uniform distribution gives std ≈ 0.289 * span
            pop_diversity = np.mean(np.std(pop_x_arr, axis=0) / span)
            diversity_ratio = np.clip(pop_diversity / 0.289, 0.0, 1.0)
            air_sigma_factor = self.air_sigma_max - (self.air_sigma_max - self.air_sigma_min) * diversity_ratio
            air_sigma_base = np.maximum(sigma, sigma_init_mean * 0.3)
            air_sigma_vec = air_sigma_base * air_sigma_factor * sigma_d

            new_xs: list[np.ndarray] = []
            if n_local > 0:
                parent_idx = rng.choice(n, size=n_local, p=weights)
                for pi in parent_idx:
                    quality = (f_max_local - pop_f_arr[pi]) / (f_spread_local + 1e-30)
                    age_ratio = min(pop_age[pi] / max(self.lifespan, 1), 1.0)
                    combined = 0.5 * quality + 0.5 * age_ratio
                    scale = self.sigma_min_ratio + (1.0 - combined) * (1.0 - self.sigma_min_ratio)
                    sigma_i = sigma * scale * sigma_d
                    child = pop_x[pi] + rng.normal(0, sigma_i, self.dim)
                    child = np.clip(child, lo, hi)
                    new_xs.append(child)
            for _ in range(n_air):
                pi = rng.integers(0, n)
                child = pop_x[pi] + rng.normal(0, air_sigma_vec, self.dim)
                child = np.clip(child, lo, hi)
                new_xs.append(child)

            # Place offspring into dead slots
            replaced: set[int] = set()
            improved_this_gen = False
            for slot, x in zip(dead_idx, new_xs):
                if slot in force_air_slots:
                    force_air_slots.discard(slot)
                    pi = rng.integers(0, n)
                    x = pop_x[pi] + rng.normal(0, air_sigma_vec, self.dim)
                    x = np.clip(x, lo, hi)
                f = self.func(x)
                pop_x[slot] = x
                pop_f[slot] = f
                pop_age[slot] = 0
                replaced.add(slot)
                history_x.append(x.copy())
                history_f.append(f)

                if f < best_so_far:
                    best_so_far = f
                    no_improve = 0
                    improved_this_gen = True
                else:
                    no_improve += 1

                if len(history_f) >= max_evals or no_improve >= self.stagnation_limit:
                    break

            if no_improve >= self.stagnation_limit:
                break

            # Bloom: add individuals on improvement
            if improved_this_gen and len(history_f) < max_evals:
                n_add = min(self.bloom_size, self.n_pop_max - len(pop_x))
                for _ in range(n_add):
                    if len(history_f) >= max_evals:
                        break
                    bx = pop_x[int(np.argmin(np.array(pop_f)))] + rng.normal(0, sigma, self.dim) * 2.0
                    bx = np.clip(bx, lo, hi)
                    bf = self.func(bx)
                    pop_x.append(bx)
                    pop_f.append(bf)
                    pop_age.append(0)
                    history_x.append(bx.copy())
                    history_f.append(bf)
                    if bf < best_so_far:
                        best_so_far = bf
                        no_improve = 0
                    else:
                        no_improve += 1

            # Cap: remove worst non-elite individuals if over n_pop_max
            while len(pop_x) > self.n_pop_max:
                px_arr = np.array(pop_x)
                pf_arr = np.array(pop_f)
                ei = self._niche_elites(px_arr, pf_arr, niche_radius_eff)
                candidates = [i for i in range(len(pop_x)) if i not in ei]
                if not candidates:
                    break
                worst = candidates[int(np.argmax(pf_arr[candidates]))]
                pop_x.pop(worst)
                pop_f.pop(worst)
                pop_age.pop(worst)

            # Crowding control: force-age excess individuals near each elite
            if crowd_radius > 0 and diversity_ratio > 0.05:
                px_arr = np.array(pop_x)
                pf_arr = np.array(pop_f)
                ei = self._niche_elites(px_arr, pf_arr, niche_radius_eff)
                max_near = max(1, int(self.max_crowd_fraction * len(pop_x)))
                for e in ei:
                    dists = np.linalg.norm(px_arr - px_arr[e], axis=1)
                    near = [i for i in np.where(dists < crowd_radius)[0]
                            if i not in ei]
                    if len(near) > max_near:
                        near_sorted = sorted(near, key=lambda i: pop_f[i], reverse=True)
                        for i in near_sorted[max_near:]:
                            pop_age[i] = self.lifespan + 1
                            force_air_slots.add(i)

            # Age survivors
            for i in range(len(pop_x)):
                if i not in replaced:
                    pop_age[i] += 1

            sigma *= self.sigma_decay
            history_pop.append(np.array(pop_x).copy())

        return self._make_result(history_x, history_f, history_pop)


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


class VOAOptimizer(BaseOptimizer):
    """Virus Optimization Algorithm (Liang & Juarez, 2016).

    Approximation based on published description: population split into
    "strong" (top fraction, local exploitation) and "common" (rest, guided
    or random exploration). sigma grows when no improvement, resets on improvement.
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
        sigma0 = 0.1 * span
        delta_sigma = 0.1 * span
        sigma_max = 0.5 * span
        sigma = sigma0

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
                sigma = sigma0
            else:
                sigma = min(sigma + delta_sigma, sigma_max)

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
