"""3-second sign probe (entry 69's rule): before running an arm, count which way
c=1.0 moves the release level on THIS function.

base effective release level  L_base = hunt_level_tol(1e-6) * f_init_scale
candidate                     L_rel  = rel_level(1e-5)
If L_base > L_rel the candidate TIGHTENS (releases hunts deeper) -> can add depth.
If L_base < L_rel it LOOSENS -> structurally cannot add depth (entry 69, N09).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/user/optimization")))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME
from core.optimizers import MultiChannelEpidemicOptimizer

for name in ["N14-CF3-3D", "N16-CF3-5D", "N18-CF3-10D"]:
    b = NICHING_BENCHMARKS_BY_NAME[name]
    for s in (0, 100, 200):
        o = MultiChannelEpidemicOptimizer(b, seed=s)
        st = o._init_state(int(b.suite_max_evals))
        fis = float(st.f_init_scale)
        lbase = o.hunt_level_tol * fis
        print(f"{name:<14} seed={s:<4} f_init_scale={fis:<12.4g} "
              f"L_base={lbase:<12.4g} L_c1.0=1e-05 "
              f"ratio(L_base/1e-5)={lbase/1e-5:<10.3g} "
              f"{'TIGHTENS' if lbase > 1e-5 else 'LOOSENS'}")
