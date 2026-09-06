"""Did the c=1.0 arm actually fire on N18? (null = cancellation or never-fired?)

Reuses scripts.gate_power._ProbeMixin, which re-implements `_basin_exhausted`
with counters and returns the same value, so the search is unchanged. Counts
per run: hunts released by the level clause on its own, whether the run ever
exhausted, f_init_scale, and the tol the arm actually installed.
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, "/home/user/optimization")
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME
from scripts.gate_power import _GateProbe, _RelGateProbe

ap = argparse.ArgumentParser()
ap.add_argument("--func"); ap.add_argument("--arm"); ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()
b = NICHING_BENCHMARKS_BY_NAME[a.func]
for s in range(a.seeds):
    if a.arm == "base":
        o = _GateProbe(b, seed=s * 100)
    else:
        o = _RelGateProbe(b, seed=s * 100, rel_level=1e-5, fis_floor=1e-12)
    r = o.optimize(int(b.suite_max_evals))
    st = o._st
    print(f"{a.func:<13} {a.arm:<5} seed={s} fis={o.f_init_scale:.4g} "
          f"tol={o.hunt_level_tol:.4g} L_eff={o.hunt_level_tol*o.f_init_scale:.4g} "
          f"level_releases={o.n_level_release} ever_exhausted={st.has_exhausted} "
          f"f_at_exh={o.f_at_exh} best_f={r.best_f:.6g} n_rep={len(r.final_solutions)}",
          flush=True)
