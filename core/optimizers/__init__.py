"""Optimizers — one file per method, re-exported here as a flat namespace.

``from core.optimizers import MultiChannelEpidemicOptimizer`` (and every other
class) keeps working exactly as before the package split; callers never need to
know which module a class lives in.

Classic / proposed baselines:
  base            OptimizeResult, BaseOptimizer
  cmaes           CMAESOptimizer
  mceso           MultiChannelEpidemicOptimizer  (the proposed method)
  pso / de / savoa  PSOOptimizer, DEOptimizer, SaVOAOptimizer
  nelder_mead     MultistartNelderMeadOptimizer  (multistart local-search floor)
  ncde            NCDEOptimizer                  (niching / multi-solution reference)
Stronger external-library baselines (formerly optimizers_modern.py):
  lshade          LSHADEOptimizer
  restart_cmaes   IPOPCMAESOptimizer, BIPOPCMAESOptimizer
"""
from __future__ import annotations

from .base import OptimizeResult, BaseOptimizer
from .cmaes import CMAESOptimizer
from .mceso import MultiChannelEpidemicOptimizer
from .mceso_ablations import MCESONoSpillover, MCESORandomRestart, MCESONoBoundarySnap
from .mceso_niching import MCESOEndemic
from .pso import PSOOptimizer
from .de import DEOptimizer
from .savoa import SaVOAOptimizer
from .nelder_mead import MultistartNelderMeadOptimizer
from .ncde import NCDEOptimizer
from .lshade import LSHADEOptimizer
from .restart_cmaes import IPOPCMAESOptimizer, BIPOPCMAESOptimizer

__all__ = [
    "OptimizeResult", "BaseOptimizer",
    "CMAESOptimizer", "MultiChannelEpidemicOptimizer",
    "MCESONoSpillover", "MCESORandomRestart", "MCESONoBoundarySnap", "MCESOEndemic",
    "PSOOptimizer", "DEOptimizer", "SaVOAOptimizer",
    "MultistartNelderMeadOptimizer", "NCDEOptimizer",
    "LSHADEOptimizer", "IPOPCMAESOptimizer", "BIPOPCMAESOptimizer",
]
