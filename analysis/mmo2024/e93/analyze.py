"""Class ceiling on the GECCO'2024 suite at D=20 (entry 93).

Entry 92 drew the ceiling at D=10 and closed the type mismatch with the
published table (MPR = mean of PR over eps_f in 1e-1 .. 1e-5).  It left D=20
undrawn, and D=20 is where the published best is *lowest* (0.476 against 0.651
at D=10), so it is where the class ceiling has the most room to clear it --
i.e. the dimension most likely to break the premise behind option (C).

Nothing new is estimated here: the three ceilings (fixed-cost `mpr`, the
infinite-restart bound `mpr_sup`, and the early-stop ceiling) come from entry
92's estimator, imported rather than reimplemented.  What is new is the data
(D=20 dumps from `hunt_coverage.py --null`) and the D=20 published value.

Pre-registered rejection condition, in the same form entry 92 used: if the
16-problem mean ceiling -- with either cost model -- exceeds 0.476, the premise
"the class cannot clear the published best" is broken at D=20 and that goes to
status.md as a decision for the user.

Two-stage draws: every problem is drawn 40 times (which is all 16 problems
inside one cycle), the highest-ceiling ones are then re-drawn to 200 (draws
0-39 of which reproduce the 40-draw run bit for bit -- the draw seed is
1_000_000 + k, so deepening a problem is an extension, not a redraw).  The
40-draw estimate is biased *down* (an optimum never landed on gets p_j = 0),
so the paired 40-vs-200 drift measured here is the D=20 counterpart of entry
92's +0.0741 median at D=10, and is reported next to the mean.

Usage: python3 analysis/mmo2024/e93/analyze.py
"""
import csv
import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name          # noqa: E402

# entry 92's estimator, imported so the two dimensions are scored by the same
# code path (the file is not a package, hence the explicit spec load).
_spec = importlib.util.spec_from_file_location(
    "e92_analyze", HERE.parent / "e92" / "analyze.py")
_e92 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e92)
ceiling, early_stop_ceiling, EPS = _e92.ceiling, _e92.early_stop_ceiling, _e92.EPS

PUBLISHED_D20 = 0.476        # competition best MPR at D=20 (external/mmo2024/docs)
BOOT = 2000


def chao1_sup(best_f, land, K) -> float:
    """Infinite-restart bound with the unseen optima put back in (Chao1).

    `mpr_sup` counts the optima the draws actually landed on, so it is biased
    down by exactly the optima a finite sample missed -- which is the one
    objection this whole ceiling is open to.  Chao1's bias-corrected form
      S_est = S_obs + f1 (f1 - 1) / (2 (f2 + 1))
    (f1 = optima landed on exactly once, f2 = exactly twice) estimates the
    richness a sample of this size cannot see, and is a *lower* bound on true
    richness in expectation -- so using it here makes the ceiling generous,
    which is the safe direction for a claim of the form "the class does not
    reach the published value".  Capped at K, averaged over the five accuracies.
    """
    per_eps = []
    for eps in EPS:
        counts = np.bincount(land[best_f <= eps], minlength=K)
        s_obs = int((counts > 0).sum())
        f1, f2 = int((counts == 1).sum()), int((counts == 2).sum())
        s_est = s_obs + f1 * (f1 - 1) / (2 * (f2 + 1))
        per_eps.append(min(s_est, K) / K)
    return float(np.mean(per_eps))


def load(name: str):
    """Rows for `name`, preferring the deeper dump."""
    for stem in (f"{name}_desc200", f"{name}_desc40", f"{name}_desc"):
        for suf in (".csv.gz", ".csv"):
            src = HERE / f"{stem}{suf}"
            if not src.exists():
                continue
            op = gzip.open if suf == ".csv.gz" else open
            with op(src, "rt") as fh:
                rows = list(csv.DictReader(fh))
            b = niching_by_name(name)
            draw = np.array([int(r["draw"]) for r in rows])
            best_f = np.array([float(r["best_f"]) for r in rows])
            land = np.array([int(r["land_opt"]) for r in rows])
            evals = np.array([float(r["evals"]) for r in rows])
            cross = None
            if f"ev_{EPS[0]:g}" in rows[0]:
                cross = {e: (np.array([float(r[f"ev_{e:g}"]) for r in rows]),
                             np.array([int(r[f"opt_{e:g}"]) for r in rows]))
                         for e in EPS}
            return best_f, land, evals, b, cross, draw
    raise FileNotFoundError(name)


def main() -> None:
    names = sorted({p.name.split("_desc")[0] for p in HERE.glob("*_desc*.csv*")})
    rng = np.random.default_rng(0)
    out = []
    for nm in names:
        best_f, land, evals, b, cross, draw = load(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        mpr, per_eps, mpr_sup = ceiling(best_f, land, evals, K, budget)
        m = len(best_f)
        boot = np.array([ceiling(best_f, land, evals, K, budget,
                                 rng.integers(0, m, m))[0] for _ in range(BOOT)])
        out.append({
            "func": nm, "K": K, "draws": m,
            "hit_1e-5": float((best_f <= 1e-5).mean()),
            "reached_1e-5": int(len(set(land[best_f <= 1e-5].tolist()))),
            "restarts": budget / evals.mean(),
            "pr_1e-1": per_eps[0], "pr_1e-3": per_eps[2], "pr_1e-5": per_eps[4],
            "mpr": mpr,
            "mpr_lo": float(np.percentile(boot, 2.5)),
            "mpr_hi": float(np.percentile(boot, 97.5)),
            # the 40-draw estimate, selected by draw *id* rather than by row
            # order: the pool returns rows in completion order, so "first 40
            # rows" is not the same set as draws 0-39 (it differs on M12).
            # Draws 0-39 of a 200-draw dump are bit-identical to a separate
            # 40-draw run of the same problem (checked on all three), which is
            # why those separate dumps are not kept.
            "mpr_first40": ceiling(best_f, land, evals, K, budget,
                                   np.flatnonzero(draw < 40))[0],
            "mpr_sup": mpr_sup,
            "mpr_sup_chao1": chao1_sup(best_f, land, K),
            "mpr_earlystop": (early_stop_ceiling(cross, evals, K, budget)[0]
                              if cross else float("nan")),
        })

    hdr = ("func", "K", "draws", "hit_1e-5", "reached_1e-5", "restarts",
           "pr_1e-1", "pr_1e-3", "pr_1e-5", "mpr", "mpr_lo", "mpr_hi",
           "mpr_first40", "mpr_sup", "mpr_sup_chao1", "mpr_earlystop")
    print(" ".join(f"{h:>13}" for h in hdr))
    for r in out:
        print(" ".join(f"{r[h]:>13.4g}" if isinstance(r[h], float)
                       else f"{str(r[h]):>13}" for h in hdr))
    with open(HERE / "ceiling_mpr_d20.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)

    mpr = np.array([r["mpr"] for r in out])
    sup = np.array([r["mpr_sup"] for r in out])
    es = np.array([r["mpr_earlystop"] for r in out])
    ch = np.array([r["mpr_sup_chao1"] for r in out])
    print(f"\nmean MPR ceiling over {len(out)} problems: {mpr.mean():.4f}"
          f"   (infinite-restart bound: {sup.mean():.4f}"
          f"; early-stop: {np.nanmean(es):.4f}"
          f"; infinite-restart + Chao1 unseen-optima correction: {ch.mean():.4f})")
    print(f"published best D=20 (16-problem mean MPR): {PUBLISHED_D20}")
    print("REJECTION CONDITION (pre-registered): mean ceiling > "
          f"{PUBLISHED_D20} breaks the premise.  "
          f"{'BROKEN' if max(mpr.mean(), sup.mean(), ch.mean()) > PUBLISHED_D20 else 'not broken'}")
    print("problems whose infinite-restart bound is above the published mean: "
          f"{[r['func'] for r in out if r['mpr_sup'] > PUBLISHED_D20] or 'none'}")

    big = [r for r in out if r["draws"] > 40]
    if big:
        d = np.array([r["mpr"] - r["mpr_first40"] for r in big])
        print(f"\npaired 40-vs-{big[0]['draws']} drift on {len(big)} problems "
              f"(full minus first-40 MPR): median {np.median(d):+.4f}, "
              f"range {d.min():+.4f}..{d.max():+.4f}, "
              f"{int((d > 0).sum())}/{len(d)} positive")
        print(f"  mean ceiling with that median drift added to every problem: "
              f"{mpr.mean() + np.median(d):.4f}")


if __name__ == "__main__":
    main()
