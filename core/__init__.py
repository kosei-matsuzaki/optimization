"""core — black-box optimization benchmarks, optimizers, and evaluation.

Convenience re-exports so callers can ``from core import MultiChannelEpidemicOptimizer``
instead of reaching into submodules. The existing submodule imports
(``from core.optimizers import ...`` etc.) keep working unchanged.

``core.visualize`` is intentionally *not* re-exported here: it pulls in
matplotlib eagerly, so it is imported explicitly only where plotting is needed.
"""
from __future__ import annotations

from .benchmarks import (
    BenchmarkFunction,
    BENCHMARKS, BENCHMARKS_3D, BENCHMARKS_4D, CUSTOM_BENCHMARKS,
    BENCHMARKS_BY_NAME, BENCHMARKS_3D_BY_NAME,
    BENCHMARKS_CEC2022_10D, BENCHMARKS_CEC2022_10D_BY_NAME,
)
from .optimizers import (
    OptimizeResult, BaseOptimizer,
    CMAESOptimizer, MultiChannelEpidemicOptimizer,
    PSOOptimizer, DEOptimizer, SaVOAOptimizer,
    LSHADEOptimizer, IPOPCMAESOptimizer, BIPOPCMAESOptimizer,
)
from .runner import (
    run_experiment, summarize, ecdf_auc, SR_THRESHOLDS,
    wilcoxon_vs_reference, vargha_delaney_a12,
)

__all__ = [
    # benchmarks
    "BenchmarkFunction",
    "BENCHMARKS", "BENCHMARKS_3D", "BENCHMARKS_4D", "CUSTOM_BENCHMARKS",
    "BENCHMARKS_BY_NAME", "BENCHMARKS_3D_BY_NAME",
    "BENCHMARKS_CEC2022_10D", "BENCHMARKS_CEC2022_10D_BY_NAME",
    # optimizers
    "OptimizeResult", "BaseOptimizer",
    "CMAESOptimizer", "MultiChannelEpidemicOptimizer",
    "PSOOptimizer", "DEOptimizer", "SaVOAOptimizer",
    "LSHADEOptimizer", "IPOPCMAESOptimizer", "BIPOPCMAESOptimizer",
    # runner / evaluation
    "run_experiment", "summarize", "ecdf_auc", "SR_THRESHOLDS",
    "wilcoxon_vs_reference", "vargha_delaney_a12",
]
