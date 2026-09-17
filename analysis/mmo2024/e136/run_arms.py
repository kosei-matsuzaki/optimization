#!/usr/bin/env python3
"""e136: is the NMMSO shortfall at D=10 in the wrapper or in ``pynmmso``?

Entry 135 established that this harness's NMMSO misses the published CEC2013
peak ratios at D=10 by 0.23 / 0.41 / 0.18, and cleared budget accounting,
report truncation and the scorer. Three wrapper details were left as suspects
(queue item 2): the boundary ``np.clip``, the post-budget ``-inf`` answer, and
the global ``np.random.seed``. This script runs each of them off, one at a
time, on N19-CF4-10D.

``core/optimizers/nmmso.py`` is NOT modified -- the queue says not to touch it
until the cause is known, or entry 135's three runs lose their control. The
arms are built by subclassing the shipped optimizer and overriding
``optimize`` with the same body plus switches, so the ``base`` arm is a literal
copy and doubles as a fidelity gate against entry 135's 15 seeds.

It also counts what entry 135's probe did not: the number of *requests* made to
the fitness function. That probe reported ``len(history_f) / max_evals`` as
"real objective calls", but the wrapper stops appending at ``max_evals``, so
that ratio is 1.0000 by construction and says nothing about how often the
``-inf`` branch fired.

Usage:  run_arms.py <arm> <function> <seed> [<seed> ...]
Writes one CSV row per run to <arm>_<function>.csv.
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import niching_by_name                     # noqa: E402
from core.optimizers import NMMSOOptimizer                      # noqa: E402
from core.runner import niching_peak_metrics                    # noqa: E402

from pynmmso import Nmmso                                       # noqa: E402

ARMS = ("base", "noclip", "nocut", "noseed", "sw10d")
LEVELS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
HERE = Path(__file__).resolve().parent


class ArmNMMSO(NMMSOOptimizer):
    """The shipped wrapper with one detail switchable off. ``base`` is a copy."""

    def __init__(self, benchmark, seed=42, swarm_size=10, arm="base"):
        super().__init__(benchmark, seed, swarm_size)
        assert arm in ARMS, arm
        self.arm = arm
        if arm == "sw10d":
            # Fieldsend 2014 §V: "maximum swarm size n = 10D". The shipped
            # wrapper hard-codes 10 for every D; entry 135 checked that against
            # pynmmso's own default rather than against the paper.
            self.swarm_size = 10 * benchmark.dim
        self.stats: dict[str, int] = {}

    def optimize(self, max_evals: int = 5000):
        lo, hi = self.bounds
        dim = self.dim
        arm = self.arm
        history_x: list[np.ndarray] = []
        history_f: list[float] = []
        # `nocut`: raise the ceiling so the -inf branch cannot fire at all.
        hard_cap = int(1.1 * max_evals) if arm == "nocut" else max_evals
        counts = {"requests": 0, "inf_answers": 0, "clipped": 0}

        func = self.func

        class _Problem:
            @staticmethod
            def fitness(params) -> float:
                counts["requests"] += 1
                if len(history_f) >= hard_cap:
                    counts["inf_answers"] += 1
                    return float("-inf")
                raw = np.asarray(params, dtype=float)
                out = bool(np.any(raw < lo) or np.any(raw > hi))
                counts["clipped"] += int(out)
                x = raw if arm == "noclip" else np.clip(raw, lo, hi)
                f = float(func(x))
                history_x.append(x.copy())
                history_f.append(f)
                return -f                          # pynmmso maximises

            @staticmethod
            def get_bounds():
                return [lo] * dim, [hi] * dim

        if arm != "noseed":
            np.random.seed(int(self.seed) % (2 ** 32))
        engine = Nmmso(_Problem(), swarm_size=self.swarm_size)
        modes = engine.run(max_evals)

        solutions = [np.clip(np.asarray(m.location, dtype=float), lo, hi)
                     for m in modes if np.isfinite(m.value)]
        if not history_f:
            x0 = np.full(dim, (lo + hi) / 2.0)
            history_x, history_f = [x0], [float(func(x0))]
        self.stats = dict(counts, evals_recorded=len(history_f),
                          pynmmso_counter=int(engine.evaluations),
                          modes_returned=len(modes),
                          modes_finite=len(solutions),
                          swarms=len(engine.swarms))
        return self._make_result(history_x, history_f,
                                 solutions=solutions or None)


def history_reach(res, b) -> tuple[float, list[float]]:
    """Entry 135's two sensitivity statistics, unchanged: the peak ratio an
    ideal reporting rule could have got out of the whole history, and each
    optimum's distance to the nearest point ever evaluated."""
    hx = np.asarray(res.history_x, dtype=float)
    hf = np.asarray(res.history_f, dtype=float)
    opts = np.asarray(b.optima_pos, dtype=float)
    rho = float(b.niche_rho)
    d = np.linalg.norm(hx[:, None, :] - opts[None, :, :], axis=2)
    hit = 0
    mins = []
    for k in range(len(opts)):
        inside = d[:, k] <= rho
        mins.append(float(d[:, k].min()))
        if inside.any() and hf[inside].min() <= 1e-5:
            hit += 1
    return hit / b.n_global_optima, mins


def main() -> None:
    arm, fname = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]]
    b = niching_by_name(fname)
    budget = int(b.suite_max_evals)
    out = HERE / f"{arm}_{fname}.csv"
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["arm", "function", "dim", "K", "seed", "budget",
                        *[f"pr_{a:.0e}".replace("e-0", "e-") for a in LEVELS],
                        "n_reported", "hist_sup_pr_1e-5", "near_0p1",
                        "requests", "inf_answers", "clipped", "evals_recorded",
                        "pynmmso_counter", "modes_returned", "modes_finite",
                        "swarms", "seconds"])
        for seed in seeds:
            t0 = time.time()
            opt = ArmNMMSO(b, seed=seed, arm=arm)
            res = opt.optimize(budget)
            dt = time.time() - t0
            m = niching_peak_metrics([res], b, LEVELS)
            sup, mins = history_reach(res, b)
            near = sum(1 for v in mins if v <= 0.1)
            s = opt.stats
            w.writerow([arm, fname, b.dim, b.n_global_optima, seed, budget,
                        *[f"{m[f'cec_pr_{a:.0e}'.replace('e-0', 'e-')]:.4f}"
                          for a in LEVELS],
                        int(m["n_reported"]), f"{sup:.4f}", near,
                        s["requests"], s["inf_answers"], s["clipped"],
                        s["evals_recorded"], s["pynmmso_counter"],
                        s["modes_returned"], s["modes_finite"], s["swarms"],
                        f"{dt:.1f}"])
            fh.flush()
            print(f"{arm:>7} {fname} seed {seed:>4}  "
                  f"pr1e-5={m['cec_pr_1e-5']:.4f}  sup={sup:.4f}  near={near}  "
                  f"req={s['requests']} inf={s['inf_answers']} "
                  f"clip={s['clipped']}  ({dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
