from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import ioh


# ── Shape tags ───────────────────────────────────────────────────────────────
# The BBOB `category` (separable / moderate-cond / ill-cond / multimodal /
# weak-structure) follows Hansen et al.'s official 5 groups, but each group name
# only names ONE axis — so a function's actual landscape shape is not read off
# the group (e.g. F02 sits in "separable" yet its defining difficulty is
# ill-conditioning; F03 sits in "separable" yet its defining difficulty is
# multimodality). `SHAPE_TAGS` gives an orthogonal, multi-axis description of the
# shape itself, keyed by function name. Axes used (a function carries the tags
# that apply to it):
#   modality      : unimodal | multimodal | multi-global
#   separability  : separable | non-separable
#   conditioning  : well-conditioned | moderate-cond | ill-conditioned
#   structure     : global-structure | weak-structure   (multimodal only)
#   landscape     : smooth | linear | asymmetric | plateau | bent-valley |
#                   sharp-ridge | rugged | deceptive | boundary-optimum | needle
#   suite-shape   : hybrid | composition                (CEC2022 constructions)
# Keep in sync with the tables in docs/experiments.md.
SHAPE_TAGS: dict[str, list[str]] = {
    # BBOB — tags describe the true shape, independent of the official group.
    "F01-Sphere":             ["unimodal", "separable", "well-conditioned", "smooth"],
    "F02-EllipsoidalSep":     ["unimodal", "separable", "ill-conditioned"],
    "F03-RastriginSep":       ["multimodal", "separable", "global-structure"],
    "F04-BucheRastrigin":     ["multimodal", "separable", "global-structure", "asymmetric"],
    "F05-LinearSlope":        ["unimodal", "separable", "linear", "boundary-optimum"],
    "F06-AttractiveSector":   ["unimodal", "non-separable", "asymmetric", "moderate-cond"],
    "F07-StepEllipsoidal":    ["unimodal", "non-separable", "plateau", "moderate-cond"],
    "F08-Rosenbrock":         ["unimodal", "non-separable", "bent-valley"],
    "F09-RosenbrockRot":      ["unimodal", "non-separable", "bent-valley"],
    "F10-EllipsoidalRot":     ["unimodal", "non-separable", "ill-conditioned"],
    "F11-Discus":             ["unimodal", "non-separable", "ill-conditioned"],
    "F12-BentCigar":          ["unimodal", "non-separable", "ill-conditioned", "bent-valley"],
    "F13-SharpRidge":         ["unimodal", "non-separable", "ill-conditioned", "sharp-ridge"],
    "F14-DiffPowers":         ["unimodal", "non-separable", "ill-conditioned"],
    "F15-RastriginRot":       ["multimodal", "non-separable", "global-structure"],
    "F16-Weierstrass":        ["multimodal", "non-separable", "global-structure", "rugged"],
    "F17-SchafferF7":         ["multimodal", "non-separable", "global-structure", "moderate-cond"],
    "F18-SchafferF7ill":      ["multimodal", "non-separable", "global-structure", "ill-conditioned"],
    "F19-GriewankRosenbrock": ["multimodal", "non-separable", "global-structure", "bent-valley"],
    "F20-Schwefel":           ["multimodal", "separable", "weak-structure", "deceptive"],
    "F21-Gallagher101":       ["multimodal", "non-separable", "weak-structure"],
    "F22-Gallagher21":        ["multimodal", "non-separable", "weak-structure"],
    "F23-Katsuura":           ["multimodal", "non-separable", "weak-structure", "rugged"],
    "F24-LunacekRastrigin":   ["multimodal", "non-separable", "weak-structure", "deceptive"],
    # Custom 2-D — `multi-global` marks the multi-global-optima problems (PR metric).
    "C01-Himmelblau":         ["multi-global", "multimodal", "smooth"],
    "C02-SixHumpCamel":       ["multi-global", "multimodal", "smooth"],
    "C03-Shubert":            ["multi-global", "multimodal", "rugged"],
    "C04-FiveWell":           ["multimodal", "deceptive"],
    "C05-Eggholder":          ["multimodal", "rugged", "deceptive", "boundary-optimum"],
    "C06-Michalewicz":        ["multimodal", "plateau", "deceptive"],
    "C07-BukinN6":            ["multimodal", "sharp-ridge", "deceptive"],
    "C08-StyblinskiTang":     ["multimodal", "deceptive"],
    "C09-Easom":              ["unimodal", "plateau", "needle"],
    "C10-SchafferN2":         ["multimodal", "rugged"],
    "C11-DeJongF5":           ["multimodal", "plateau", "deceptive"],
    # CEC2022 hold-out (dim=10).
    "G01-Zakharov":           ["unimodal", "non-separable", "ill-conditioned"],
    "G02-Rosenbrock":         ["multimodal", "non-separable", "bent-valley"],
    "G03-SchafferF7":         ["multimodal", "non-separable", "rugged"],
    "G04-Rastrigin":          ["multimodal", "non-separable", "global-structure"],
    "G05-Levy":               ["multimodal", "non-separable", "global-structure"],
    "G06-Hybrid1":            ["multimodal", "non-separable", "hybrid"],
    "G07-Hybrid2":            ["multimodal", "non-separable", "hybrid"],
    "G08-Hybrid3":            ["multimodal", "non-separable", "hybrid"],
    "G09-Composition1":       ["multimodal", "non-separable", "weak-structure", "composition"],
    "G10-Composition2":       ["multimodal", "non-separable", "weak-structure", "composition"],
    "G11-Composition3":       ["multimodal", "non-separable", "weak-structure", "composition"],
    "G12-Composition4":       ["multimodal", "non-separable", "weak-structure", "composition"],
}

# Presentation grouping of shape tags by axis — used to order/label columns in
# reference views (the /benchmarks matrix). Mirrors the axes documented above and
# TAG_ORDER in web/static/result.js (JS cannot import this, so keep both in sync).
TAG_AXES: list[tuple[str, list[str]]] = [
    ("modality",     ["unimodal", "multimodal", "multi-global"]),
    ("separability", ["separable", "non-separable"]),
    ("conditioning", ["well-conditioned", "moderate-cond", "ill-conditioned"]),
    ("structure",    ["global-structure", "weak-structure"]),
    ("landscape",    ["smooth", "linear", "asymmetric", "plateau", "bent-valley",
                      "sharp-ridge", "rugged", "deceptive", "boundary-optimum", "needle"]),
    ("suite-shape",  ["hybrid", "composition"]),
]


@dataclass
class BenchmarkFunction:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]
    optimum: float
    category: str
    dim: int = 2
    optima_pos: list[list[float]] | None = None
    tags: list[str] = field(default_factory=list)
    # CEC2013-niching metadata; None for every other suite. niche_rho is the
    # radius the official peak counter uses to tell two reported solutions
    # apart, n_global_optima is K, suite_max_evals the competition budget.
    niche_rho: float | None = None
    n_global_optima: int | None = None
    suite_max_evals: int | None = None

    def __post_init__(self) -> None:
        # Single source of truth: derive shape tags from the name unless a caller
        # supplied them explicitly. Covers every construction path (BBOB, custom,
        # CEC2022, and the worker-side make_benchmark_by_name reconstruction).
        if not self.tags:
            self.tags = list(SHAPE_TAGS.get(self.name, []))


# ── Evaluation-noise models (quick --noise mode) ──────────────────────────
# BBOB-noisy-style multiplicative noise on the OBSERVED f: the optimizer only
# ever sees the noisy value; scoring re-evaluates visited points noise-free
# (see core/runner.run_experiment). Multiplicative forms keep the models
# scale-invariant across functions (f - f_opt >= 0 by construction). Following
# the BBOB-noisy convention, noise is suppressed once true f <= NOISE_FREE_LEVEL
# so the deepest precision targets stay measurable.
NOISE_FREE_LEVEL = 1e-8
NOISE_MODELS = ("gauss_mild", "gauss_sev", "cauchy")


def make_noisy_func(
    base_func: Callable[[np.ndarray], float],
    model: str,
    rng: np.random.Generator,
) -> Callable[[np.ndarray], float]:
    """Wrap ``base_func`` with one of NOISE_MODELS (its own RNG, independent
    of the optimizer's seed):

      gauss_mild   f * exp(0.01 * N(0,1))       BBOB moderate gaussian
      gauss_sev    f * exp(1.00 * N(0,1))       BBOB severe gaussian (~x e^±2)
      cauchy       f * (1 + |Cauchy|) w.p. 0.2  seldom-severe upward spikes
    """
    if model not in NOISE_MODELS:
        raise ValueError(f"unknown noise model {model!r}; choose from {NOISE_MODELS}")

    def noisy(x: np.ndarray) -> float:
        f = float(base_func(x))
        if f <= NOISE_FREE_LEVEL:
            return f
        if model == "gauss_mild":
            return f * float(np.exp(0.01 * rng.standard_normal()))
        if model == "gauss_sev":
            return f * float(np.exp(1.0 * rng.standard_normal()))
        # cauchy: upward-only spikes so min-tracking is stressed by outliers,
        # not poisoned by spuriously deflated observations.
        if rng.random() < 0.2:
            return f * (1.0 + abs(float(rng.standard_cauchy())))
        return f

    return noisy


# BBOB noiseless suite (Hansen et al., 2009)
# 24 functions covering 5 difficulty groups; instance=1 fixes the transformation.
# func(x) returns f(x) - f_opt so the global minimum is always 0.
_BBOB_SPECS: list[tuple[int, str, str]] = [
    # Group 1: Separable functions
    (1,  "F01-Sphere",             "separable"),
    (2,  "F02-EllipsoidalSep",     "separable"),
    (3,  "F03-RastriginSep",       "separable"),
    (4,  "F04-BucheRastrigin",     "separable"),
    (5,  "F05-LinearSlope",        "separable"),
    # Group 2: Low / moderate conditioning
    (6,  "F06-AttractiveSector",   "moderate-cond"),
    (7,  "F07-StepEllipsoidal",    "moderate-cond"),
    (8,  "F08-Rosenbrock",         "moderate-cond"),
    (9,  "F09-RosenbrockRot",      "moderate-cond"),
    # Group 3: High conditioning, unimodal
    (10, "F10-EllipsoidalRot",     "ill-cond"),
    (11, "F11-Discus",             "ill-cond"),
    (12, "F12-BentCigar",          "ill-cond"),
    (13, "F13-SharpRidge",         "ill-cond"),
    (14, "F14-DiffPowers",         "ill-cond"),
    # Group 4: Multi-modal, adequate global structure
    (15, "F15-RastriginRot",       "multimodal"),
    (16, "F16-Weierstrass",        "multimodal"),
    (17, "F17-SchafferF7",         "multimodal"),
    (18, "F18-SchafferF7ill",      "multimodal"),
    (19, "F19-GriewankRosenbrock", "multimodal"),
    # Group 5: Multi-modal, weak global structure
    (20, "F20-Schwefel",           "weak-structure"),
    (21, "F21-Gallagher101",       "weak-structure"),
    (22, "F22-Gallagher21",        "weak-structure"),
    (23, "F23-Katsuura",           "weak-structure"),
    (24, "F24-LunacekRastrigin",   "weak-structure"),
]


def _make_ioh_benchmark(problem_id, name: str, category: str, dim: int,
                        instance: int = 1, problem_class=None) -> BenchmarkFunction:
    """Wrap an ``ioh`` problem as a BenchmarkFunction shifted so f_opt = 0.

    ``problem_id`` is a BBOB function id (int, with ``problem_class``) or a
    CEC2022 problem name (str). Shared by both _make_bbob and _make_cec2022.
    """
    kwargs: dict = {"instance": instance, "dimension": dim}
    if problem_class is not None:
        kwargs["problem_class"] = problem_class
    prob = ioh.get_problem(problem_id, **kwargs)
    lo = float(prob.bounds.lb[0])
    hi = float(prob.bounds.ub[0])
    f_opt = float(prob.optimum.y)
    opt_x = [list(prob.optimum.x)]

    def func(x: np.ndarray) -> float:
        return float(prob(x.tolist())) - f_opt

    return BenchmarkFunction(
        name=name,
        func=func,
        bounds=(lo, hi),
        optimum=0.0,
        category=category,
        dim=dim,
        optima_pos=opt_x,
    )


def _make_bbob(fid: int, name: str, category: str, dim: int, instance: int = 1) -> BenchmarkFunction:
    return _make_ioh_benchmark(fid, name, category, dim, instance,
                               problem_class=ioh.ProblemClass.BBOB)


def _build(dim: int) -> list[BenchmarkFunction]:
    return [_make_bbob(fid, name, cat, dim) for fid, name, cat in _BBOB_SPECS]


# Custom multi-global-optima benchmarks (2-D only)
def _himmelblau() -> BenchmarkFunction:
    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        return (a**2 + b - 11)**2 + (a + b**2 - 7)**2

    return BenchmarkFunction(
        name="C01-Himmelblau",
        func=func,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="multi-optima",
        dim=2,
        optima_pos=[
            [3.0,       2.0      ],
            [-2.805118, 3.131312 ],
            [-3.779310, -3.283186],
            [3.584428,  -1.848126],
        ],
    )


def _six_hump_camel() -> BenchmarkFunction:
    _f_opt = -1.0316284534898774

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        return (4 - 2.1*a**2 + a**4/3)*a**2 + a*b + (-4 + 4*b**2)*b**2 - _f_opt

    return BenchmarkFunction(
        name="C02-SixHumpCamel",
        func=func,
        bounds=(-2.0, 2.0),
        optimum=0.0,
        category="multi-optima",
        dim=2,
        optima_pos=[
            [ 0.0898, -0.7126],
            [-0.0898,  0.7126],
        ],
    )


def _shubert() -> BenchmarkFunction:
    # f(x,y) = (Σⱼ j cos((j+1)x + j)) · (Σⱼ j cos((j+1)y + j))
    # 18 global minima at f ≈ -186.7309 on [-10, 10]²
    _f_opt = -186.7309088310240

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        u = sum(j * np.cos((j + 1) * a + j) for j in range(1, 6))
        v = sum(j * np.cos((j + 1) * b + j) for j in range(1, 6))
        return float(u * v) - _f_opt

    return BenchmarkFunction(
        name="C03-Shubert",
        func=func,
        bounds=(-10.0, 10.0),
        optimum=0.0,
        category="multi-optima",
        dim=2,
        optima_pos=[
            [-7.7083, -7.0835], [-7.7083, -0.8003], [-7.7083,  5.4829],
            [-7.0835, -7.7083], [-7.0835, -1.4251], [-7.0835,  4.8581],
            [-1.4251, -7.0835], [-1.4251, -0.8003], [-1.4251,  5.4829],
            [-0.8003, -7.7083], [-0.8003, -1.4251], [-0.8003,  4.8581],
            [ 4.8581, -7.0835], [ 4.8581, -0.8003], [ 4.8581,  5.4829],
            [ 5.4829, -7.7083], [ 5.4829, -1.4251], [ 5.4829,  4.8581],
        ],
    )


def _five_well() -> BenchmarkFunction:
    # Five-well potential (Tomitomi3 Qiita). 5 wells, only one global on [-20, 20]².
    # _f_opt refined numerically (L-BFGS-B from published x*) — the textbook
    # value -1.4616268944... was off by ~1e-5 and produced negative f post-shift.
    _f_opt = -1.4616377135103535

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        inner = (1.0
            - 1.0 / (1 + 0.05 * (a**2 + (b - 10)**2))
            - 1.0 / (1 + 0.05 * ((a - 10)**2 + b**2))
            - 1.5 / (1 + 0.03 * ((a + 10)**2 + b**2))
            - 2.0 / (1 + 0.05 * ((a - 5)**2 + (b + 10)**2))
            - 1.0 / (1 + 0.1  * ((a + 5)**2 + (b + 10)**2)))
        return float(inner * (1 + 0.0001 * (a**2 + b**2) ** 1.2)) - _f_opt

    return BenchmarkFunction(
        name="C04-FiveWell",
        func=func,
        bounds=(-20.0, 20.0),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[4.9213, -9.8873]],
    )


def _eggholder() -> BenchmarkFunction:
    _f_opt = -959.6406627208506

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        return (-(b + 47) * np.sin(np.sqrt(abs(b + a / 2 + 47)))
                - a * np.sin(np.sqrt(abs(a - (b + 47))))) - _f_opt

    return BenchmarkFunction(
        name="C05-Eggholder",
        func=func,
        bounds=(-512.0, 512.0),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[512.0, 404.2319]],
    )


def _michalewicz() -> BenchmarkFunction:
    # 2-D, m=10 (steepness). Global min ≈ -1.8013 at (2.20319, 1.57049) on [0, π]²
    _m = 10
    _f_opt = -1.8013034100985537

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        v = (np.sin(a) * np.sin(1 * a**2 / np.pi) ** (2 * _m)
             + np.sin(b) * np.sin(2 * b**2 / np.pi) ** (2 * _m))
        return float(-v) - _f_opt

    return BenchmarkFunction(
        name="C06-Michalewicz",
        func=func,
        bounds=(0.0, float(np.pi)),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[2.20319, 1.57049]],
    )


def _bukin_n6() -> BenchmarkFunction:
    # Standard domain is asymmetric (x∈[-15,-5], y∈[-3,3]); we use the symmetric
    # box [-15, 15]² to fit the single-range BenchmarkFunction.bounds — the global
    # min at (-10, 1) and the y = 0.01·x² ridge structure are unchanged.
    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        return float(100 * np.sqrt(abs(b - 0.01 * a**2)) + 0.01 * abs(a + 10))

    return BenchmarkFunction(
        name="C07-BukinN6",
        func=func,
        bounds=(-15.0, 15.0),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[-10.0, 1.0]],
    )


def _styblinski_tang() -> BenchmarkFunction:
    # 2-D global min at (-2.903534, -2.903534) with f ≈ -78.3323.
    # Three additional local minima of distinct depth at (±2.903534, ±2.7468).
    _f_opt = -78.33233140754285

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        return float(0.5 * (a**4 - 16 * a**2 + 5 * a
                            + b**4 - 16 * b**2 + 5 * b)) - _f_opt

    return BenchmarkFunction(
        name="C08-StyblinskiTang",
        func=func,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[-2.903534, -2.903534]],
    )


def _easom() -> BenchmarkFunction:
    # Needle-in-haystack: nearly flat outside a small region near (π, π).
    _f_opt = -1.0

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        return float(-np.cos(a) * np.cos(b)
                     * np.exp(-((a - np.pi)**2 + (b - np.pi)**2))) - _f_opt

    return BenchmarkFunction(
        name="C09-Easom",
        func=func,
        bounds=(-100.0, 100.0),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[float(np.pi), float(np.pi)]],
    )


def _schaffer_n2() -> BenchmarkFunction:
    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        num = np.sin(a**2 - b**2) ** 2 - 0.5
        den = (1 + 0.001 * (a**2 + b**2)) ** 2
        return float(0.5 + num / den)

    return BenchmarkFunction(
        name="C10-SchafferN2",
        func=func,
        bounds=(-100.0, 100.0),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[0.0, 0.0]],
    )


def _dejong_f5() -> BenchmarkFunction:
    # Shekel's foxholes: 25 wells on a 5×5 grid; global at the (-32, -32) well.
    _A1 = np.array([-32, -16, 0, 16, 32] * 5, dtype=float)
    _A2 = np.array([v for v in (-32, -16, 0, 16, 32) for _ in range(5)], dtype=float)
    _f_opt = 0.9980038378

    def func(x: np.ndarray) -> float:
        a, b = float(x[0]), float(x[1])
        s = 0.002
        for j in range(25):
            s += 1.0 / ((j + 1) + (a - _A1[j])**6 + (b - _A2[j])**6)
        return float(1.0 / s) - _f_opt

    return BenchmarkFunction(
        name="C11-DeJongF5",
        func=func,
        bounds=(-65.536, 65.536),
        optimum=0.0,
        category="deceptive-2d",
        dim=2,
        optima_pos=[[-32.0, -32.0]],
    )


CUSTOM_BENCHMARKS: list[BenchmarkFunction] = [
    _himmelblau(),
    _six_hump_camel(),
    _shubert(),
    _five_well(),
    _eggholder(),
    _michalewicz(),
    _bukin_n6(),
    _styblinski_tang(),
    _easom(),
    _schaffer_n2(),
    _dejong_f5(),
]

# Name -> factory for custom benchmarks, so workers can reconstruct a fresh
# instance from just the name. Keep in sync with CUSTOM_BENCHMARKS.
_CUSTOM_FACTORIES = {
    "C01-Himmelblau":     _himmelblau,
    "C02-SixHumpCamel":   _six_hump_camel,
    "C03-Shubert":        _shubert,
    "C04-FiveWell":       _five_well,
    "C05-Eggholder":      _eggholder,
    "C06-Michalewicz":    _michalewicz,
    "C07-BukinN6":        _bukin_n6,
    "C08-StyblinskiTang": _styblinski_tang,
    "C09-Easom":          _easom,
    "C10-SchafferN2":     _schaffer_n2,
    "C11-DeJongF5":       _dejong_f5,
}

BENCHMARKS    = _build(2)
BENCHMARKS_3D = _build(3)
BENCHMARKS_4D = _build(4)
# Higher-dimensional BBOB registries for the dimension-scaling snapshot
# (n = 2, 3, 5, 10, 20). BBOB is defined at any dimension via ioh, so these are
# just `_build(d)` at the requested d. Custom (C*) benchmarks are 2-D only and
# are intentionally absent here.
BENCHMARKS_5D  = _build(5)
BENCHMARKS_10D = _build(10)
BENCHMARKS_20D = _build(20)

BENCHMARKS_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS + CUSTOM_BENCHMARKS
}
BENCHMARKS_3D_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS_3D
}
BENCHMARKS_5D_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS_5D
}
BENCHMARKS_10D_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS_10D
}
BENCHMARKS_20D_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS_20D
}


# CEC2022 suite — held-out benchmark independent of BBOB.
# Used to test whether MC-ESO mechanisms generalize beyond the BBOB
# transformations they were developed against. HPs MUST NOT be re-tuned
# for CEC2022 evaluation.
_CEC2022_SPECS: list[tuple[str, str, str]] = [
    # (ioh problem name, display name, category)
    ("CEC2022Zakharov",            "G01-Zakharov",      "unimodal"),
    ("CEC2022Rosenbrock",          "G02-Rosenbrock",    "basic-multimodal"),
    ("CEC2022SchafferF7",          "G03-SchafferF7",    "basic-multimodal"),
    ("CEC2022Rastrigin",           "G04-Rastrigin",     "basic-multimodal"),
    ("CEC2022Levy",                "G05-Levy",          "basic-multimodal"),
    ("CEC2022HybridFunction1",     "G06-Hybrid1",       "hybrid"),
    ("CEC2022HybridFunction2",     "G07-Hybrid2",       "hybrid"),
    ("CEC2022HybridFunction3",     "G08-Hybrid3",       "hybrid"),
    ("CEC2022CompositionFunction1", "G09-Composition1", "composition"),
    ("CEC2022CompositionFunction2", "G10-Composition2", "composition"),
    ("CEC2022CompositionFunction3", "G11-Composition3", "composition"),
    ("CEC2022CompositionFunction4", "G12-Composition4", "composition"),
]


def _make_cec2022(ioh_name: str, display_name: str, category: str,
                  dim: int, instance: int = 1) -> BenchmarkFunction:
    return _make_ioh_benchmark(ioh_name, display_name, category, dim, instance)


def _build_cec2022(dim: int) -> list[BenchmarkFunction]:
    return [_make_cec2022(ioh_name, name, cat, dim)
            for ioh_name, name, cat in _CEC2022_SPECS]


# CEC2022 standard dim=10 (smallest supported by the full 12-function set)
BENCHMARKS_CEC2022_10D = _build_cec2022(10)
BENCHMARKS_CEC2022_10D_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS_CEC2022_10D
}


# ── CEC2013 niching suite (low-dimensional multi-global subset) ──────────
# Li, Engelbrecht & Epitropakis (2013), "Benchmark Functions for CEC'2013
# Special Session and Competition on Niching Methods for Multimodal Function
# Optimization". Formulas transcribed from the reference MATLAB implementation
# (github.com/mikeagn/CEC2013, matlab/niching_func.m); f_goptima, rho, the
# number of global optima and MaxFEs come from get_fgoptima / get_rho /
# get_no_goptima / get_maxfes in the same package.
#
# The suite is stated as MAXIMISATION. Each function is registered here as
# `f_goptima - f_raw(x)`, so it minimises to 0 like everything else in this
# project and the sr_1e-1 .. sr_1e-5 columns coincide exactly with the
# competition's accuracy levels epsilon.
#
# Only the 2-D/3-D subset (N04-N10) is registered. F1-F3 are 1-D and nothing
# here supports dim=1 (pycma needs N>=2, the visualisations assume 2-D/3-D);
# F11-F20 are composition functions that need the suite's shift/rotation data
# files. Both gaps are deliberate — see docs/experiments.md.
_NICHING_K = np.array([3.0, 4.0])          # modified Rastrigin peak counts (D=2)


def _n_himmelblau(x: np.ndarray) -> float:
    return 200.0 - (x[0] ** 2 + x[1] - 11.0) ** 2 - (x[0] + x[1] ** 2 - 7.0) ** 2


def _n_six_hump(x: np.ndarray) -> float:
    return -((4.0 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3.0) * x[0] ** 2
             + x[0] * x[1] + (4.0 * x[1] ** 2 - 4.0) * x[1] ** 2)


def _n_shubert(x: np.ndarray) -> float:
    j = np.arange(1.0, 6.0)
    prod = 1.0
    for xi in x:
        prod *= float(np.sum(j * np.cos((j + 1.0) * xi + j)))
    return -prod


def _n_vincent(x: np.ndarray) -> float:
    return float(np.mean(np.sin(10.0 * np.log(x))))


def _n_mod_rastrigin(x: np.ndarray) -> float:
    return -float(np.sum(10.0 + 9.0 * np.cos(2.0 * np.pi * _NICHING_K * x)))


# (name, raw maximisation function, bounds, f_goptima, dim, #global optima,
#  rho, suite MaxFEs, tags). Bounds are the official ones except N05 — see below.
_NICHING_SPECS: list[tuple] = [
    ("N04-Himmelblau",    _n_himmelblau,    (-6.0, 6.0),   200.0,
     2, 4,   0.01,  50_000, ["multi-global", "multimodal", "non-separable", "smooth"]),
    # Official box is x1 in [-1.9, 1.9], x2 in [-1.1, 1.1]. BenchmarkFunction
    # carries a single range for every axis, so x2 is widened to [-1.9, 1.9].
    # The two global optima are unchanged (f grows outside the official strip,
    # so the extra area adds no maximum) and distances are undistorted, which
    # keeps rho-based peak counting comparable — but the search volume is 1.7x
    # the official one, so PR here is not comparable to published F5 numbers.
    ("N05-SixHumpCamel",  _n_six_hump,      (-1.9, 1.9),   1.031628453489877,
     2, 2,   0.5,   50_000, ["multi-global", "multimodal", "non-separable", "smooth"]),
    ("N06-Shubert2D",     _n_shubert,       (-10.0, 10.0), 186.730908831024,
     2, 18,  0.5,  200_000, ["multi-global", "multimodal", "non-separable", "rugged"]),
    ("N07-Vincent2D",     _n_vincent,       (0.25, 10.0),  1.0,
     2, 36,  0.2,  200_000, ["multi-global", "multimodal", "separable", "rugged"]),
    ("N08-Shubert3D",     _n_shubert,       (-10.0, 10.0), 2709.093505572820,
     3, 81,  0.5,  400_000, ["multi-global", "multimodal", "non-separable", "rugged"]),
    ("N09-Vincent3D",     _n_vincent,       (0.25, 10.0),  1.0,
     3, 216, 0.2,  400_000, ["multi-global", "multimodal", "separable", "rugged"]),
    ("N10-ModRastrigin2D", _n_mod_rastrigin, (0.0, 1.0),   -2.0,
     2, 12,  0.01, 200_000, ["multi-global", "multimodal", "separable", "smooth"]),
]


def _make_niching(spec: tuple) -> BenchmarkFunction:
    name, raw, bounds, f_gopt, dim, n_opt, rho, maxfes, tags = spec

    def func(x: np.ndarray, _raw=raw, _f=f_gopt) -> float:
        return float(_f - _raw(np.asarray(x, dtype=float)))

    return BenchmarkFunction(
        name=name, func=func, bounds=bounds, optimum=0.0,
        category="multi-optima", dim=dim, tags=list(tags),
        niche_rho=rho, n_global_optima=n_opt, suite_max_evals=maxfes,
    )


NICHING_BENCHMARKS: list[BenchmarkFunction] = [_make_niching(s) for s in _NICHING_SPECS]
NICHING_BENCHMARKS_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in NICHING_BENCHMARKS
}


def make_benchmark_by_name(name: str, dim: int) -> BenchmarkFunction:
    """Reconstruct a fresh benchmark from its name (for use in worker processes).

    Covers custom benchmarks (C*), BBOB (F*), and CEC2022 (G*). Custom
    benchmarks are 2-D only; ``dim`` is ignored for them.
    """
    if name in _CUSTOM_FACTORIES:
        return _CUSTOM_FACTORIES[name]()
    spec = next((s for s in _BBOB_SPECS if s[1] == name), None)
    if spec is not None:
        return _make_bbob(spec[0], spec[1], spec[2], dim)
    spec = next((s for s in _CEC2022_SPECS if s[1] == name), None)
    if spec is not None:
        return _make_cec2022(spec[0], spec[1], spec[2], dim)
    spec = next((s for s in _NICHING_SPECS if s[0] == name), None)
    if spec is not None:
        return _make_niching(spec)
    raise ValueError(f"Unknown benchmark: {name}")
