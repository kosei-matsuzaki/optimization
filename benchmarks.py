from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import ioh


@dataclass
class BenchmarkFunction:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]
    optimum: float
    category: str
    dim: int = 2
    optima_pos: list[list[float]] | None = None


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


def _make_bbob(fid: int, name: str, category: str, dim: int, instance: int = 1) -> BenchmarkFunction:
    prob = ioh.get_problem(fid, instance=instance, dimension=dim, problem_class=ioh.ProblemClass.BBOB)
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


CUSTOM_BENCHMARKS: list[BenchmarkFunction] = [_himmelblau(), _six_hump_camel()]

BENCHMARKS    = _build(2)
BENCHMARKS_3D = _build(3)
BENCHMARKS_4D = _build(4)

BENCHMARKS_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS + CUSTOM_BENCHMARKS
}
