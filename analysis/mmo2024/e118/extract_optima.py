"""Dump the 16 problems' global-optima coordinates once (D=10, PIN01).

`make_mmo2024` reads a 6 MB table per problem, so this is called 16 times in
total and the result is cached as an .npz that the analysis reads.
ORACLE COLUMN WARNING (entry 115's lesson): `optima_pos` is the scorer's
information, not anything a method holds at run time. Everything downstream of
this file is a diagnostic about the suite, never a column handed to a method.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from core.benchmarks import make_mmo2024                       # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "optima_d10_pin01.npz"
D, PIN = 10, 1

d = {}
for pid in range(1, 17):
    t0 = time.time()
    b = make_mmo2024(pid, D, PIN)
    pos = np.asarray(b.optima_pos, dtype=float)
    d[f"M{pid:02d}"] = pos
    d[f"M{pid:02d}_bounds"] = np.array(b.bounds, dtype=float)
    print(f"M{pid:02d}: K={pos.shape[0]} dim={pos.shape[1]} "
          f"bounds={b.bounds} rho={b.niche_rho} ({time.time()-t0:.1f}s)", flush=True)
np.savez_compressed(OUT, **d)
print("wrote", OUT)
