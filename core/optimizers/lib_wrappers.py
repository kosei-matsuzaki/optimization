"""Library-backed baselines: published DE winners and nevergrad portfolios.

Each class wraps a third-party implementation under the BaseOptimizer
interface, following the pattern of ``lshade.py``: the objective is wrapped
so that every evaluation is recorded and the budget is enforced *exactly*
(the wrapper refuses the (max_evals+1)-th call), and the library is seeded
from ``self.seed``.

| class                   | algorithm (paper)                         | backend            |
|-------------------------|-------------------------------------------|--------------------|
| IMODEOptimizer          | IMODE, Sallam+ CEC 2020 winner            | mealpy (+LPSR here)|
| LSHADEcnEpSinOptimizer  | LSHADE-cnEpSin, Awad+ CEC 2017            | mealpy             |
| JSOOptimizer            | jSO, Brest+ CEC 2017 2nd                  | pyade / minionpy   |
| LSRTDEOptimizer         | L-SRTDE, Stanovov & Semenkin CEC 2024 win | minionpy           |
| NGOptOptimizer          | NGOpt algorithm selector                  | nevergrad          |
| NGPortfolioOptimizer    | Portfolio (passive portfolio)             | nevergrad          |

Libraries that are not in the base requirements (nevergrad, minionpy) are
imported inside ``optimize`` so this module always imports.

Global RNG: mealpy's L-SHADE family draws F/CR via ``scipy.stats.*.rvs``
(global ``np.random``), and pyade calls ``np.random.seed`` / ``random.seed``.
``_seeded_global_rng`` seeds both global generators for the run and restores
their previous state afterwards, so runs are reproducible and do not perturb
the caller's RNG.
"""
from __future__ import annotations

import contextlib
import logging
import random
import warnings

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult
from .lshade import _eval_recording_wrapper


@contextlib.contextmanager
def _seeded_global_rng(seed: int):
    np_state = np.random.get_state()
    py_state = random.getstate()
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def _require(module: str, pip_name: str):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            f"This optimizer needs the optional package '{pip_name}' "
            f"(pip install {pip_name})."
        ) from e


def _mealpy_problem(opt: BaseOptimizer, wrapped) -> dict:
    from mealpy import FloatVar
    lo, hi = opt.bounds
    return {
        "bounds": FloatVar(lb=[float(lo)] * opt.dim, ub=[float(hi)] * opt.dim, name="x"),
        "obj_func": wrapped,
        "minmax": "min",
        "log_to": None,
        "verbose": False,
    }


# ---------------------------------------------------------------------------
# IMODE (mealpy) ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _make_lpsr_imode(base_cls):
    """mealpy's OriginalIMODE keeps the population size fixed. Sallam et al.
    (2020) reduce it linearly from N_init to N_min over the evaluation budget;
    this subclass adds that reduction (by evaluations, keeping the best)."""

    class _LPSRIMODE(base_cls):
        def __init__(self, *args, n_min: int = 4, max_evals: int = 0,
                     nfe_ref=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._n_init = self.pop_size
            self._n_min = n_min
            self._max_evals = max_evals
            self._nfe_ref = nfe_ref  # list whose len() is the evaluation count

        def evolve(self, epoch):
            super().evolve(epoch)
            nfe = len(self._nfe_ref)
            target = int(round(self._n_init + (self._n_min - self._n_init)
                               * nfe / self._max_evals))
            target = max(self._n_min, target)
            if target < self.pop_size:
                self.pop = self.get_sorted_population(self.pop, self.problem.minmax)[:target]
                self.pop_size = target

    return _LPSRIMODE


class IMODEOptimizer(BaseOptimizer):
    """IMODE (Sallam et al., CEC 2020 winner) via mealpy.sota_based.IMODE.

    Defaults follow the paper: N_init = 6*D^2 reduced linearly to 4 (``lpsr``,
    added here — mealpy itself has a fixed population), memory size 5.
    The final SQP local search of the original is NOT implemented (mealpy has
    none). ``pop_size`` is clamped to [5, 10000] by mealpy's validator.
    """

    def __init__(self, benchmark: BenchmarkFunction, seed: int = 42,
                 pop_size: int | None = None, memory_size: int = 5,
                 archive_size: int = 20, lpsr: bool = True, pop_size_min: int = 4):
        super().__init__(benchmark, seed)
        n = 6 * benchmark.dim * benchmark.dim if pop_size is None else pop_size
        self.pop_size = int(min(max(n, 5), 10000))
        self.memory_size = memory_size
        self.archive_size = archive_size
        self.lpsr = lpsr
        self.pop_size_min = pop_size_min

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        from mealpy import Termination
        from mealpy.sota_based.IMODE import OriginalIMODE

        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        wrapped = _eval_recording_wrapper(self.func, history_x, history_f, max_evals)

        common = dict(pop_size=self.pop_size, memory_size=self.memory_size,
                      archive_size=self.archive_size)
        if self.lpsr:
            epoch = min(100000, max_evals // max(self.pop_size_min, 1) + 20)
            model = _make_lpsr_imode(OriginalIMODE)(
                epoch=epoch, n_min=self.pop_size_min, max_evals=max_evals,
                nfe_ref=history_f, **common)
        else:
            epoch = min(100000, int(np.ceil(max_evals / self.pop_size)) + 20)
            model = OriginalIMODE(epoch=epoch, **common)

        logging.getLogger("mealpy").setLevel(logging.WARNING)
        with _seeded_global_rng(self.seed):
            try:
                model.solve(_mealpy_problem(self, wrapped), mode="single",
                            termination=Termination(max_fe=max_evals), seed=self.seed)
            except wrapped._budget_exc:
                pass
        return self._make_result(history_x, history_f, history_pop=None)


# ---------------------------------------------------------------------------
# LSHADE-cnEpSin (mealpy) ---------------------------------------------------
# ---------------------------------------------------------------------------

def _cnepsin_epochs(max_evals: int, n_init: int, n_min: int) -> int:
    """Smallest epoch count E whose LPSR schedule (as mealpy computes it, by
    epoch/E) spends at least max_evals evaluations, so the schedule — and the
    'first half sinusoidal' switch — line up with the evaluation budget."""
    for E in range(1, 100001):
        total, n = n_init, n_init
        for e in range(1, E + 1):
            total += n
            n = max(n_min, int(n_min + (n_init - n_min) * (E - e) / E))
        if total >= max_evals:
            return E
    return 100000


class LSHADEcnEpSinOptimizer(BaseOptimizer):
    """LSHADE-cnEpSin (Awad et al., CEC 2017) via
    mealpy.sota_based.LSHADEcnEpSin.OriginalLSHADEcnEpSin.

    N_init = 18*D, N_min = 4, H = 5, freq = 0.5, ps = 0.5, pc = 0.4 (paper).
    mealpy schedules LPSR and the sinusoidal phase by *epoch*; the epoch count
    is chosen so that schedule ends exactly when the evaluation budget does.
    """

    def __init__(self, benchmark: BenchmarkFunction, seed: int = 42,
                 pop_size: int | None = None, pop_size_min: int = 4,
                 memory_size: int = 5, freq: float = 0.5, ps: float = 0.5,
                 pc: float = 0.4):
        super().__init__(benchmark, seed)
        self.pop_size = 18 * benchmark.dim if pop_size is None else pop_size
        self.pop_size = max(self.pop_size, 5)
        self.pop_size_min = pop_size_min
        self.memory_size = memory_size
        self.freq, self.ps, self.pc = freq, ps, pc

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        from mealpy import Termination
        from mealpy.sota_based.LSHADEcnEpSin import OriginalLSHADEcnEpSin

        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        wrapped = _eval_recording_wrapper(self.func, history_x, history_f, max_evals)

        epoch = _cnepsin_epochs(max_evals, self.pop_size, self.pop_size_min)
        model = OriginalLSHADEcnEpSin(
            epoch=epoch, pop_size=self.pop_size, pop_size_min=self.pop_size_min,
            memory_size=self.memory_size, freq=self.freq, ps=self.ps, pc=self.pc)

        logging.getLogger("mealpy").setLevel(logging.WARNING)
        with _seeded_global_rng(self.seed):
            try:
                model.solve(_mealpy_problem(self, wrapped), mode="single",
                            termination=Termination(max_fe=max_evals), seed=self.seed)
            except wrapped._budget_exc:
                pass
        return self._make_result(history_x, history_f, history_pop=None)


# ---------------------------------------------------------------------------
# minionpy (C++ implementations) --------------------------------------------
# ---------------------------------------------------------------------------

def _run_minion(opt: BaseOptimizer, algo_name: str, max_evals: int,
                options: dict) -> OptimizeResult:
    minionpy = _require("minionpy", "minionpy")
    history_x: list[np.ndarray] = []
    history_f: list[float] = []

    def batch(X):
        out = []
        for x in X:
            if len(history_f) >= max_evals:
                out.append(float("inf"))  # never reached: minion honours maxevals
                continue
            x = np.asarray(x, dtype=float)
            f = float(opt.func(x))
            history_x.append(x.copy())
            history_f.append(f)
            out.append(f)
        return out

    lo, hi = opt.bounds
    algo = getattr(minionpy, algo_name)(
        batch, [(float(lo), float(hi))] * opt.dim, maxevals=int(max_evals),
        seed=int(opt.seed), options=dict(options))
    algo.optimize()
    return opt._make_result(history_x, history_f, history_pop=None)


class LSRTDEOptimizer(BaseOptimizer):
    """L-SRTDE (Stanovov & Semenkin, CEC 2024 winner) via minionpy.LSRTDE.

    minionpy defaults (N = 20*D, memory 5, success rate 0.5). Its coordinate
    tolerance stop (x_tol=1e-8) is disabled by default so the run spends the
    whole budget like the original competition code; pass ``x_tol=1e-8`` to
    restore minionpy's default."""

    def __init__(self, benchmark: BenchmarkFunction, seed: int = 42,
                 x_tol: float = -1.0, **options):
        super().__init__(benchmark, seed)
        self.options = {"x_tol": x_tol, "f_tol": -1.0, **options}

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        return _run_minion(self, "LSRTDE", max_evals, self.options)


# ---------------------------------------------------------------------------
# jSO (pyade or minionpy) ---------------------------------------------------
# ---------------------------------------------------------------------------

class JSOOptimizer(BaseOptimizer):
    """jSO (Brest et al., CEC 2017 2nd place).

    backend="pyade" (default; pyade.jso, already a dependency) or
    backend="minionpy" (C++ jSO; needs minionpy). The pyade implementation
    deviates from the paper in several places (see module report); minionpy
    is the closer one. N_init = round(25*ln(D)*sqrt(D)) in both.
    """

    def __init__(self, benchmark: BenchmarkFunction, seed: int = 42,
                 backend: str = "pyade", pop_size: int | None = None,
                 memory_size: int = 5):
        super().__init__(benchmark, seed)
        if backend not in ("pyade", "minionpy"):
            raise ValueError(f"unknown jSO backend {backend!r}")
        self.backend = backend
        d = benchmark.dim
        self.pop_size = (int(round(25 * np.log(d) * np.sqrt(d)))
                         if pop_size is None else pop_size)
        self.pop_size = max(self.pop_size, 5)
        self.memory_size = memory_size

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        if self.backend == "minionpy":
            return _run_minion(self, "jSO", max_evals, {
                "population_size": self.pop_size, "memory_size": self.memory_size,
                "x_tol": -1.0, "f_tol": -1.0})

        import pyade.jso
        lo, hi = self.bounds
        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        wrapped = _eval_recording_wrapper(self.func, history_x, history_f, max_evals)
        with _seeded_global_rng(self.seed):  # pyade re-seeds the globals itself
            try:
                pyade.jso.apply(
                    population_size=self.pop_size, individual_size=self.dim,
                    bounds=np.array([[float(lo), float(hi)]] * self.dim),
                    func=wrapped, opts=None, memory_size=self.memory_size,
                    callback=None, max_evals=int(max_evals), seed=int(self.seed))
            except wrapped._budget_exc:
                pass
        return self._make_result(history_x, history_f, history_pop=None)


# ---------------------------------------------------------------------------
# nevergrad -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _run_nevergrad(opt: BaseOptimizer, optimizer_name: str,
                   max_evals: int) -> OptimizeResult:
    ng = _require("nevergrad", "nevergrad")
    lo, hi = opt.bounds
    param = ng.p.Array(shape=(opt.dim,), lower=float(lo), upper=float(hi))
    # Seed the parametrization's RNG *before* building the optimizer; every
    # nevergrad optimizer draws its randomness from it.
    param.random_state = np.random.RandomState(opt.seed % (2 ** 32))
    optimizer = ng.optimizers.registry[optimizer_name](
        parametrization=param, budget=int(max_evals), num_workers=1)

    history_x: list[np.ndarray] = []
    history_f: list[float] = []
    with _seeded_global_rng(opt.seed), warnings.catch_warnings():
        # some sub-optimizers touch np.random; cma/nevergrad warn chattily
        warnings.simplefilter("ignore")
        for _ in range(int(max_evals)):
            cand = optimizer.ask()
            x = np.asarray(cand.value, dtype=float)
            f = float(opt.func(x))
            history_x.append(x.copy())
            history_f.append(f)
            optimizer.tell(cand, f)
    return opt._make_result(history_x, history_f, history_pop=None)


class NGOptOptimizer(BaseOptimizer):
    """nevergrad NGOpt (algorithm selector by budget / dim). Needs nevergrad."""

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        return _run_nevergrad(self, "NGOpt", max_evals)


class NGPortfolioOptimizer(BaseOptimizer):
    """nevergrad Portfolio (budget split across CMA / DE / scrHammersley...).
    Needs nevergrad."""

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        return _run_nevergrad(self, "Portfolio", max_evals)
