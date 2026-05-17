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

BENCHMARKS    = _build(2)
BENCHMARKS_3D = _build(3)
BENCHMARKS_4D = _build(4)

BENCHMARKS_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS + CUSTOM_BENCHMARKS
}
BENCHMARKS_3D_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS_3D
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
    prob = ioh.get_problem(ioh_name, instance=instance, dimension=dim)
    lo = float(prob.bounds.lb[0])
    hi = float(prob.bounds.ub[0])
    f_opt = float(prob.optimum.y)
    opt_x = [list(prob.optimum.x)]

    def func(x: np.ndarray) -> float:
        return float(prob(x.tolist())) - f_opt

    return BenchmarkFunction(
        name=display_name,
        func=func,
        bounds=(lo, hi),
        optimum=0.0,
        category=category,
        dim=dim,
        optima_pos=opt_x,
    )


def _build_cec2022(dim: int) -> list[BenchmarkFunction]:
    return [_make_cec2022(ioh_name, name, cat, dim)
            for ioh_name, name, cat in _CEC2022_SPECS]


# CEC2022 standard dim=10 (smallest supported by the full 12-function set)
BENCHMARKS_CEC2022_10D = _build_cec2022(10)
BENCHMARKS_CEC2022_10D_BY_NAME: dict[str, BenchmarkFunction] = {
    b.name: b for b in BENCHMARKS_CEC2022_10D
}
