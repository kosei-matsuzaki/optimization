"""MAP-Elites — the quality-diversity baseline (Mouret & Clune, 2015)."""
from __future__ import annotations
import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


class MAPElitesOptimizer(BaseOptimizer):
    """MAP-Elites over a grid of behaviour descriptors.

    Quality-diversity keeps one elite per cell of a behaviour space the user
    defines, rather than one best solution overall. It is the dominant framing
    in game and design applications, so a study of what diverse solution sets
    are worth is incomplete without it — niching methods answer a different
    question (find every optimum) than QD does (fill a descriptor space).

    **Behaviour descriptor.** These benchmarks carry no descriptor, so this uses
    the convention QD papers use for continuous test functions: the first two
    coordinates, normalised to the box. At dim 2 the descriptor space *is* the
    search space, which is exactly the canonical illumination setup for 2-D toy
    problems; at higher dimensions the remaining coordinates are free within a
    cell. The choice is the method's defining parameter, and results move with
    it — that is a property of QD, not an artefact of this implementation, and
    it is why the descriptor is stated wherever these numbers are reported.

    Reported solutions are the archive's elites, which is what a QD run is for.
    """

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        grid: int = 20,           # cells per descriptor dimension
        n_init: int = 100,        # random solutions before the archive drives search
        sigma: float = 0.1,       # mutation width, fraction of the span
        n_bd: int = 2,            # descriptor dimensions
        bd: str = "coords",       # which descriptor: coords | rot45 | polar | random
    ):
        super().__init__(benchmark, seed)
        self.grid = grid
        self.n_init = n_init
        self.sigma = sigma
        self.n_bd = min(n_bd, benchmark.dim)
        self.bd = bd
        # A fixed random projection, drawn once per run, for bd="random". Seeded
        # off the run's seed so a descriptor choice is reproducible.
        self._proj = np.random.default_rng(seed + 991).standard_normal(
            (self.n_bd, benchmark.dim))
        self._proj /= np.linalg.norm(self._proj, axis=1, keepdims=True)

    def _descriptor(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Behaviour descriptor in [0, 1]^n_bd.

        Four choices, all defensible, none privileged — which is the point. QD
        results are a property of the descriptor as much as of the algorithm, so
        the same run is scored under several of them to see how far the answer
        moves.
        """
        u = (x - lo) / (hi - lo)                       # normalised coordinates
        if self.bd == "coords":                        # the usual convention
            return u[:self.n_bd]
        if self.bd == "rot45":                         # same information, rotated
            a, b = u[0], u[min(1, self.dim - 1)]
            return np.array([(a + b) / 2, (a - b + 1) / 2])[:self.n_bd]
        if self.bd == "polar":                         # radius and angle from centre
            c = u - 0.5
            r = float(np.linalg.norm(c)) / (0.5 * self.dim ** 0.5)
            ang = (np.arctan2(c[min(1, self.dim - 1)], c[0]) + np.pi) / (2 * np.pi)
            return np.array([r, ang])[:self.n_bd]
        if self.bd == "random":                        # a fixed random projection
            v = self._proj @ (u - 0.5)
            return np.clip(v / (self.dim ** 0.5) + 0.5, 0.0, 1.0)
        raise ValueError(f"unknown bd: {self.bd}")

    def _cell(self, x: np.ndarray, lo: float, hi: float) -> tuple[int, ...]:
        d = self._descriptor(np.asarray(x, dtype=float), lo, hi)
        idx = np.clip((d * self.grid).astype(int), 0, self.grid - 1)
        return tuple(int(i) for i in idx)

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        span = hi - lo

        archive: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
        history_x: list[np.ndarray] = []
        history_f: list[float] = []

        def submit(x: np.ndarray) -> None:
            f = float(self.func(x))
            history_x.append(x.copy())
            history_f.append(f)
            c = self._cell(x, lo, hi)
            cur = archive.get(c)
            if cur is None or f < cur[1]:
                archive[c] = (x.copy(), f)

        for _ in range(min(self.n_init, max_evals)):
            submit(rng.uniform(lo, hi, self.dim))

        while len(history_f) < max_evals:
            # Uniform choice among elites is the original selection rule: every
            # occupied cell is an equally good parent regardless of its fitness,
            # which is what keeps the archive spreading instead of collapsing.
            keys = list(archive)
            parent = archive[keys[rng.integers(len(keys))]][0]
            child = np.clip(parent + self.sigma * span * rng.standard_normal(self.dim),
                            lo, hi)
            submit(child)

        elites = [x for x, _ in archive.values()]
        return self._make_result(history_x, history_f, solutions=elites or None)
