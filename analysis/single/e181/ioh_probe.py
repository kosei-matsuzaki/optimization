"""Evaluate the 24 BBOB dim-2 functions (instance 1) at a fixed grid of points
and print a digest. Run under two ioh versions; identical digests mean the
BBOB implementation did not move, so it cannot explain a reproduction gap."""
import hashlib, sys
import numpy as np, ioh
rng = np.random.default_rng(12345)
pts = rng.uniform(-5, 5, size=(7, 2))
lines = []
for fid in range(1, 25):
    p = ioh.get_problem(fid, instance=1, dimension=2,
                        problem_class=ioh.ProblemClass.BBOB)
    vals = [float(p(list(map(float, x)))) for x in pts]
    opt = float(p.optimum.y)
    lines.append(f"F{fid:02d} opt={opt!r} " + " ".join(repr(v) for v in vals))
body = "\n".join(lines)
print("ioh", ioh.__version__ if hasattr(ioh, "__version__") else "?",
      "numpy", np.__version__)
print("digest", hashlib.sha256(body.encode()).hexdigest()[:24])
open(sys.argv[1], "w").write(body + "\n")
