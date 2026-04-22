from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class BenchmarkFunction:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: tuple[float, float]
    optimum: float
    category: str  # "unimodal" | "multimodal" | "multi-optima" | "deceptive"
    dim: int = 2
    optima_pos: list[list[float]] | None = None


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    return float(100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)


def rosenbrock_nd(x: np.ndarray) -> float:
    return float(sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)))


def rastrigin(x: np.ndarray) -> float:
    A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    return float(
        -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(c * x)) / n)
        + a + np.e
    )


def himmelblau(x: np.ndarray) -> float:
    return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)


def beale(x: np.ndarray) -> float:
    return float(
        (1.5 - x[0] + x[0] * x[1])**2
        + (2.25 - x[0] + x[0] * x[1]**2)**2
        + (2.625 - x[0] + x[0] * x[1]**3)**2
    )


def schwefel(x: np.ndarray) -> float:
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


_EGGHOLDER_SHIFT = 959.6407


def eggholder(x: np.ndarray) -> float:
    val = (
        -(x[1] + 47.0) * np.sin(np.sqrt(np.abs(x[0] / 2.0 + (x[1] + 47.0))))
        - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47.0))))
    )
    return float(val + _EGGHOLDER_SHIFT)


BENCHMARKS: list[BenchmarkFunction] = [
    BenchmarkFunction(
        name="Sphere",
        func=sphere,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="unimodal",
        dim=2,
        optima_pos=[[0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Rosenbrock",
        func=rosenbrock,
        bounds=(-2.0, 2.0),
        optimum=0.0,
        category="unimodal",
        dim=2,
        optima_pos=[[1.0, 1.0]],
    ),
    BenchmarkFunction(
        name="Rastrigin",
        func=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        category="multimodal",
        dim=2,
        optima_pos=[[0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Ackley",
        func=ackley,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="multimodal",
        dim=2,
        optima_pos=[[0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Himmelblau",
        func=himmelblau,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="multi-optima",
        dim=2,
        optima_pos=[
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ],
    ),
    BenchmarkFunction(
        name="Beale",
        func=beale,
        bounds=(-4.5, 4.5),
        optimum=0.0,
        category="multi-optima",
        dim=2,
        optima_pos=[[3.0, 0.5]],
    ),
    BenchmarkFunction(
        name="Schwefel",
        func=schwefel,
        bounds=(-500.0, 500.0),
        optimum=0.0,
        category="deceptive",
        dim=2,
        optima_pos=[[420.9687, 420.9687]],
    ),
    BenchmarkFunction(
        name="Eggholder",
        func=eggholder,
        bounds=(-512.0, 512.0),
        optimum=0.0,
        category="deceptive",
        dim=2,
        optima_pos=[[512.0, 404.2319]],
    ),
]

BENCHMARKS_3D: list[BenchmarkFunction] = [
    BenchmarkFunction(
        name="Sphere3D",
        func=sphere,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="unimodal",
        dim=3,
        optima_pos=[[0.0, 0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Rosenbrock3D",
        func=rosenbrock_nd,
        bounds=(-2.0, 2.0),
        optimum=0.0,
        category="unimodal",
        dim=3,
        optima_pos=[[1.0, 1.0, 1.0]],
    ),
    BenchmarkFunction(
        name="Rastrigin3D",
        func=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        category="multimodal",
        dim=3,
        optima_pos=[[0.0, 0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Ackley3D",
        func=ackley,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="multimodal",
        dim=3,
        optima_pos=[[0.0, 0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Schwefel3D",
        func=schwefel,
        bounds=(-500.0, 500.0),
        optimum=0.0,
        category="deceptive",
        dim=3,
        optima_pos=[[420.9687, 420.9687, 420.9687]],
    ),
]

BENCHMARKS_4D: list[BenchmarkFunction] = [
    BenchmarkFunction(
        name="Sphere4D",
        func=sphere,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="unimodal",
        dim=4,
        optima_pos=[[0.0, 0.0, 0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Rosenbrock4D",
        func=rosenbrock_nd,
        bounds=(-2.0, 2.0),
        optimum=0.0,
        category="unimodal",
        dim=4,
        optima_pos=[[1.0, 1.0, 1.0, 1.0]],
    ),
    BenchmarkFunction(
        name="Rastrigin4D",
        func=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        category="multimodal",
        dim=4,
        optima_pos=[[0.0, 0.0, 0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Ackley4D",
        func=ackley,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        category="multimodal",
        dim=4,
        optima_pos=[[0.0, 0.0, 0.0, 0.0]],
    ),
    BenchmarkFunction(
        name="Schwefel4D",
        func=schwefel,
        bounds=(-500.0, 500.0),
        optimum=0.0,
        category="deceptive",
        dim=4,
        optima_pos=[[420.9687, 420.9687, 420.9687, 420.9687]],
    ),
]

BENCHMARKS_BY_NAME: dict[str, BenchmarkFunction] = {b.name: b for b in BENCHMARKS}
