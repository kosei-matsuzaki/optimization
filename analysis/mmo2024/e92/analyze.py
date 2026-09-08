"""Class ceiling on the GECCO'2024 suite, in the same unit as the published table.

Entry 91 drew the ceiling at eps_f = 1e-5 only and compared it against the
competition's MPR, which is the mean of PR over eps_f in {1e-1 .. 1e-5}.  That
comparison is type-mismatched: MPR >= PR@1e-5 for one and the same run, so the
1e-5 ceiling understates what the class can score.  Entry 91 flagged this as
the reason it could not say whether the ceiling clears the published best.

This script closes that gap without any new optimisation run: the descent dumps
already carry ``best_f`` per descent, so the same multinomial argument as entry
88 section 6 can be evaluated at each of the five accuracies and averaged.

  1. read the uniform-restart + isotropic-descent dump (hunt_coverage --null),
  2. for each eps in (1e-1 .. 1e-5): keep descents with best_f <= eps, count
     landings per optimum -> p_j(eps)  (failures keep their share of the mass),
  3. n = suite budget / mean evaluations per descent = restarts one run affords,
  4. PR_ceiling(eps) = sum_j [1 - (1 - p_j(eps))^n] / K,
  5. MPR_ceiling = mean over the five eps.  This is the number the published
     MPR is measured in.

Both numbers are upper bounds for the whole class: perfect reporting, no repel
bookkeeping, and no budget spent on anything except descent.  At the loose
accuracies the bound is generous in a second way -- a descent that stalls at
f <= 1e-1 is credited to its nearest global optimum whether or not it is in
that basin -- which only makes it a safer ceiling.

Uncertainty: the ceiling is a plug-in estimate from a finite number of draws,
and its bias is one-sided (an optimum never landed on gets p_j = 0, so the
ceiling is biased *down*).  Two things are reported:
  * a percentile bootstrap 95% CI over draws, and
  * the same estimate recomputed from the first 40 draws, paired per problem,
    which measures how much of entry 91's 40-draw screen was that downward bias.

Usage: python3 analysis/mmo2024/e92/analyze.py
"""
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.benchmarks import niching_by_name          # noqa: E402

EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
HERE = Path(__file__).resolve().parent
E91 = HERE.parent / "e91"
BOOT = 2000


def load(name: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Rows for `name`, preferring the 200-draw dump over the 40-draw screen."""
    for d in (HERE, E91):
        for stem in (f"{name}_desc200", f"{name}_desc"):
            p = d / f"{stem}.csv.gz"
            q = d / f"{stem}.csv"
            src = p if p.exists() else (q if q.exists() else None)
            if src is None:
                continue
            op = gzip.open if src.suffix == ".gz" else open
            with op(src, "rt") as fh:
                rows = list(csv.DictReader(fh))
            b = niching_by_name(name)
            best_f = np.array([float(r["best_f"]) for r in rows])
            land = np.array([int(r["land_opt"]) for r in rows])
            evals = np.array([float(r["evals"]) for r in rows])
            return best_f, land, evals, b
    raise FileNotFoundError(name)


def ceiling(best_f, land, evals, K, budget, idx=None) -> tuple[float, list]:
    """PR ceiling at each eps and their mean, over the draws selected by `idx`."""
    if idx is None:
        idx = np.arange(len(best_f))
    bf, ld, ev = best_f[idx], land[idx], evals[idx]
    n = budget / ev.mean()
    per_eps = []
    for eps in EPS:
        p = np.zeros(K)
        for j in ld[bf <= eps]:
            p[j] += 1
        p /= len(idx)                   # failures keep their share of the mass
        per_eps.append(float(np.sum(1.0 - (1.0 - p) ** n)) / K)
    return float(np.mean(per_eps)), per_eps


def main() -> None:
    names = sorted({p.name.split("_desc")[0]
                    for d in (HERE, E91) for p in d.glob("*_desc*.csv*")})
    rng = np.random.default_rng(0)
    out = []
    for nm in names:
        best_f, land, evals, b = load(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        mpr, per_eps = ceiling(best_f, land, evals, K, budget)
        m = len(best_f)
        boot = np.array([ceiling(best_f, land, evals, K, budget,
                                 rng.integers(0, m, m))[0] for _ in range(BOOT)])
        mpr40, _ = ceiling(best_f, land, evals, K, budget, np.arange(min(40, m)))
        out.append({
            "func": nm, "K": K, "draws": m,
            "hit_1e-5": float((best_f <= 1e-5).mean()),
            "reached_1e-5": int(len(set(land[best_f <= 1e-5].tolist()))),
            "restarts": budget / evals.mean(),
            "pr_1e-1": per_eps[0], "pr_1e-3": per_eps[2], "pr_1e-5": per_eps[4],
            "mpr": mpr,
            "mpr_lo": float(np.percentile(boot, 2.5)),
            "mpr_hi": float(np.percentile(boot, 97.5)),
            "mpr_first40": mpr40,
        })

    hdr = ("func", "K", "draws", "hit_1e-5", "reached_1e-5", "restarts",
           "pr_1e-1", "pr_1e-3", "pr_1e-5", "mpr", "mpr_lo", "mpr_hi",
           "mpr_first40")
    print(" ".join(f"{h:>13}" for h in hdr))
    for r in out:
        print(" ".join(f"{r[h]:>13.4g}" if isinstance(r[h], float)
                       else f"{str(r[h]):>13}" for h in hdr))
    with open(HERE / "ceiling_mpr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)

    mpr = np.array([r["mpr"] for r in out])
    print(f"\nmean MPR ceiling over {len(out)} problems: {mpr.mean():.4f}")
    print(f"published best D=10 (RR-CMA-ES, 16-problem mean MPR): 0.651")
    over = [r for r in out if r["mpr_lo"] > 0.651]
    print(f"problems whose MPR ceiling's 95% CI is entirely above 0.651: "
          f"{[r['func'] for r in over] or 'none'}")

    # paired 40-vs-full drift: how much of the 40-draw screen was downward bias
    big = [r for r in out if r["draws"] >= 200]
    if big:
        d = np.array([r["mpr"] - r["mpr_first40"] for r in big])
        print(f"\npaired drift on {len(big)} problems re-drawn to "
              f"{big[0]['draws']} (full minus first-40 MPR): "
              f"median {np.median(d):+.4f}, range {d.min():+.4f}..{d.max():+.4f}, "
              f"{int((d > 0).sum())}/{len(d)} positive")


if __name__ == "__main__":
    main()
