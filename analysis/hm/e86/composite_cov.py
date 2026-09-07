"""Does the adoption-candidate composite arm move *coverage* on the CF3 pair?

Question 1 of the queue. The arithmetic in status.md closed the depth side of
case (A): within one run `PR@eps` is non-increasing in eps (the scorer counts
the same reported set at a stricter f threshold), so `PR@1e-1` is that run's
ceiling on `PR@1e-5`. MC-ESO's ceiling is below the published best on all seven
of F14-F20. The only way case (A) survives is if an arm raises the ceiling
itself, so this measures `PR@1e-1` / `PRtrue@1e-1`, not the judgement levels.

Arms (paired on the seed, 12 seeds, full suite budget 4e5):

  base   the shipped optimiser
  comp   steps (a) + (c) + (d) of the adoption procedure, i.e. every candidate
         that has passed the BBOB gate and is *not* function-dependent:
           (a) rel_level = 1e-5  (c = 1.0, entry 46's corner) with the adoption
               clamp fis_floor = 1e-12 (entry 44)
           (c) exhausted_sigma_tol = 1.0 (`_sig10`), sigma_floor_ratio = 1e-8
               (`_fl08`)
           (d) sol_trim_mode = "rho" (`soltrim_rho`)
         Step (f) (`commit_place`) is deliberately left out: entries 41/42
         closed it as coverage-limited-only and it has never passed the BBOB
         gate, so it is not part of the composite the queue names.

`core/` is untouched; the composite is the MRO of two shipped diagnostic
classes on disjoint hooks (`RelLevelMCESO._init_state`,
`SolArchiveTrimMCESO._on_spillover_start`), which is the same construction
`scripts/diagnose_niching.py` uses for `ct_soltrim_rel_fl08` minus the commit
layer. `--arm base` instantiates the shipped class itself, so the base column is
a real base run and not a disabled variant.

Scoring is done twice on the same run, following entry 77's driver:
  pr_*      the official scorer (`core.runner._niching_counts`, cap + rho-greedy)
  prtrue_*  the same reported set with each kept point attributed to its nearest
            true optimum, which is what entry 76 needs: on 10D the 1e-1 contour
            of a basin is wider than rho = 0.01, so the official PR@1e-1 counts
            points inside one basin as separate niches (N18 scored 6/6 where the
            true coverage was 4/6).

  H1  the composite raises true coverage above the published best
      (`PRtrue@1e-1` > 0.677 on N16, > 0.667 on N18).
  Refuted if it does not: case (A) is closed on all seven functions and no
      amount of depth or report tuning can reach the published best there.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import NICHING_BENCHMARKS_BY_NAME              # noqa: E402
from core.optimizers import MultiChannelEpidemicOptimizer           # noqa: E402
from core.optimizers.mceso_commit_reseed import CommitReseedMCESO   # noqa: E402
from core.optimizers.mceso_rel_level import RelLevelMCESO           # noqa: E402
from core.optimizers.mceso_sol_archive import SolArchiveTrimMCESO   # noqa: E402
from core.runner import _niching_counts                             # noqa: E402

EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)

#  Adoption steps (a) + (c) + (d). `sig10` / `fl08` are constructor arguments of
#  the shipped class, so only two override layers are needed.
COMP_KW = dict(rel_level=1e-5, fis_floor=1e-12,
               exhausted_sigma_tol=1.0, sigma_floor_ratio=1e-8,
               sol_trim_mode="rho")


#  Added after the pre-registered pair came back null (see the log entry): the
#  composite above holds the *placement* of the restarts fixed, and step (f)
#  `commit_place` is the only shipped dial that moves it (entries 62/63 raised
#  N09-Vincent3D's coverage ceiling with it). Without this arm the negative
#  would only say "no depth or report dial moves coverage", which was already
#  known; with it the negative covers every dial the repository has.
COMPF_KW = dict(COMP_KW, commit_sigma_mode="place", commit_sigma_ratio=0.1)


class CompositeMCESO(RelLevelMCESO, SolArchiveTrimMCESO):
    """rel-level over the niche-greedy archive trim, both on disjoint hooks."""


class CompositePlaceMCESO(RelLevelMCESO, CommitReseedMCESO,
                          SolArchiveTrimMCESO):
    """The same composite with the committed reseed placement underneath it.

    The MRO is the one `scripts/diagnose_niching.py` documents for
    `ct_soltrim_rel_fl08` (rel-level -> commit -> adaptive-repel -> archive trim
    -> shipped); only `commit_sigma_mode` differs, "place" being the adoption
    shape of step (f) (the run sigma stays at base).
    """


def _attribute(x, opt: np.ndarray) -> int:
    """Index of the true optimum nearest to x (entry 77's attribution)."""
    d = np.linalg.norm(opt - np.asarray(x, dtype=float)[None, :], axis=1)
    return int(np.argmin(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="N18-CF3-10D")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--arm", choices=("base", "comp", "compf"), default="comp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    b = NICHING_BENCHMARKS_BY_NAME[a.func]
    ev_budget = a.evals or int(b.suite_max_evals)
    opt = np.asarray(b.optima_pos, dtype=float)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pf = open(f"{out}_pr.csv", "w", newline="")
    pw = csv.writer(pf)
    pw.writerow(["func", "arm", "seed", "best_f", "n_rep", "wall_s"]
                + [f"pr_{e:g}" for e in EPS]
                + [f"prtrue_{e:g}" for e in EPS])

    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        t0 = time.time()
        if a.arm == "base":
            o = MultiChannelEpidemicOptimizer(b, seed=s * 100)
        elif a.arm == "comp":
            o = CompositeMCESO(b, seed=s * 100, **COMP_KW)
        else:
            o = CompositePlaceMCESO(b, seed=s * 100, **COMPF_KW)
        r = o.optimize(ev_budget)

        counts, n_rep = _niching_counts([r], b, EPS)
        pr = [counts[0, j] / b.n_global_optima for j in range(len(EPS))]

        X = np.asarray(r.final_solutions, dtype=float)
        F = np.array([float(b.func(x)) for x in X])
        cap = max(100, 2 * b.n_global_optima)
        if len(F) > cap:
            keep = np.argsort(F)[:cap]
            X, F = X[keep], F[keep]
        prt = [len({_attribute(X[k], opt) for k in range(len(X)) if F[k] <= e})
               / b.n_global_optima for e in EPS]

        pw.writerow([a.func, a.arm, s, f"{r.best_f:.6g}", n_rep[0],
                     f"{time.time() - t0:.1f}"]
                    + [f"{v:.6g}" for v in pr] + [f"{v:.6g}" for v in prt])
        pf.flush()
        print(f"{a.func} {a.arm} seed={s} best_f={r.best_f:.6g} "
              f"n_rep={n_rep[0]} PR@1e-1={pr[0]:.3f} PRtrue@1e-1={prt[0]:.3f} "
              f"PR@1e-3={pr[2]:.3f} PR@1e-5={pr[4]:.3f} "
              f"wall={time.time() - t0:.1f}s", flush=True)

    pf.close()


if __name__ == "__main__":
    main()
