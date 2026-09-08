"""D=10 class ceiling with group B (K=10) drawn 200 deep -- the last open
sampling objection to the D=10 number.

Entry 92 drew group A (M01-M08, K=20) 200 times and found the 40-draw screen
biased *down* by a paired median of +0.0741 MPR.  It left group B (M09-M16,
K=10) at 40 draws, on the strength of one problem (M13, +0.008) -- so the
16-problem mean 0.486 that entry 92 compared against the published 0.651 still
rests on eight problems whose estimate is known-biased in an unknown amount.
The bias runs in the direction that *weakens* the premise of route (C) ("the
class cannot reach the published best"), so it has to be measured, not assumed.

This entry re-draws the seven remaining group-B problems (M13 was already 200)
to 200 and rescores all 16 with the same four ceilings entries 92 and 93 used:

  mpr            fixed-cost multinomial ceiling      (entry 92)
  mpr_earlystop  descent cut the moment it is in eps (entry 92)
  mpr_sup        infinite restarts, support / K      (entry 92)
  mpr_sup_chao1  the same with unseen optima put back (entry 93)

Refutation condition, written before the runs: if the 16-problem mean of ANY of
the four exceeds the published best at D=10 (RR-CMA-ES, MPR 0.651), the premise
of route (C) breaks at D=10.  Four readings, the most generous of which grants
infinite restarts AND credits optima never landed on -- so a "does not reach"
here is a statement about the class, not about the sample size.

The estimators come from entries 92 and 93 by importlib, so all three
dimensions/entries are scored by one code path.

Usage: python3 analysis/mmo2024/e94/analyze.py
"""
import csv
import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name          # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_e92 = _load_module("e92_analyze", HERE.parent / "e92" / "analyze.py")
_e93 = _load_module("e93_analyze", HERE.parent / "e93" / "analyze.py")
ceiling, early_stop_ceiling, EPS = _e92.ceiling, _e92.early_stop_ceiling, _e92.EPS
chao1_sup = _e93.chao1_sup

PUBLISHED_D10 = 0.651        # competition best MPR at D=10 (external/mmo2024/docs)
BOOT = 2000
# deepest dump first: entry 94 (group B, 200), entry 92 (group A, 200),
# entry 91 (M01/M13 at 200, the rest of group B at 40).
DIRS = (HERE, HERE.parent / "e92", HERE.parent / "e91")


def load(name: str):
    for d in DIRS:
        for stem in (f"{name}_desc200", f"{name}_desc40", f"{name}_desc"):
            for suf in (".csv.gz", ".csv"):
                src = d / f"{stem}{suf}"
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
                return best_f, land, evals, b, cross, draw, src
    raise FileNotFoundError(name)


def main() -> None:
    names = [f"M{p:02d}-D10-PIN01" for p in range(1, 17)]
    rng = np.random.default_rng(0)
    out = []
    for nm in names:
        best_f, land, evals, b, cross, draw, src = load(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        mpr, per_eps, mpr_sup = ceiling(best_f, land, evals, K, budget)
        m = len(best_f)
        boot = np.array([ceiling(best_f, land, evals, K, budget,
                                 rng.integers(0, m, m))[0] for _ in range(BOOT)])
        # the 40-draw estimate is selected by draw *id*: --null collects with
        # imap_unordered, so the first 40 rows are not draws 0-39 (entry 93).
        first40 = np.flatnonzero(draw < 40)
        out.append({
            "func": nm, "K": K, "draws": m,
            "hit_1e-5": float((best_f <= 1e-5).mean()),
            "reached_1e-5": int(len(set(land[best_f <= 1e-5].tolist()))),
            "restarts": budget / evals.mean(),
            "pr_1e-1": per_eps[0], "pr_1e-3": per_eps[2], "pr_1e-5": per_eps[4],
            "mpr": mpr,
            "mpr_lo": float(np.percentile(boot, 2.5)),
            "mpr_hi": float(np.percentile(boot, 97.5)),
            "mpr_first40": ceiling(best_f, land, evals, K, budget, first40)[0],
            "mpr_sup": mpr_sup,
            "mpr_sup_chao1": chao1_sup(best_f, land, K),
            "mpr_earlystop": (early_stop_ceiling(cross, evals, K, budget)[0]
                              if cross else float("nan")),
            "src": src.parent.name,
        })

    hdr = ("func", "K", "draws", "hit_1e-5", "reached_1e-5", "restarts",
           "pr_1e-1", "pr_1e-3", "pr_1e-5", "mpr", "mpr_lo", "mpr_hi",
           "mpr_first40", "mpr_sup", "mpr_sup_chao1", "mpr_earlystop", "src")
    print(" ".join(f"{h:>13}" for h in hdr))
    for r in out:
        print(" ".join(f"{r[h]:>13.4g}" if isinstance(r[h], float)
                       else f"{str(r[h]):>13}" for h in hdr))
    with open(HERE / "ceiling_mpr_d10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)

    mpr = np.array([r["mpr"] for r in out])
    sup = np.array([r["mpr_sup"] for r in out])
    ch = np.array([r["mpr_sup_chao1"] for r in out])
    es = np.array([r["mpr_earlystop"] for r in out])
    esm = es[~np.isnan(es)]
    print(f"\n16-problem mean ceiling, D=10:"
          f"  fixed-cost {mpr.mean():.4f}"
          f"  early-stop {esm.mean():.4f} (on {len(esm)} instrumented)"
          f"  infinite-restart {sup.mean():.4f}"
          f"  +Chao1 {ch.mean():.4f}")
    print(f"published best D=10 (RR-CMA-ES, 16-problem mean MPR): {PUBLISHED_D10}")
    worst = max(mpr.mean(), sup.mean(), ch.mean())
    print(f"refutation condition -- any 16-problem mean above {PUBLISHED_D10} "
          f"breaks the premise.  "
          f"{'BROKEN' if worst > PUBLISHED_D10 else 'not broken'} "
          f"(most generous reading {worst:.4f}, short by {PUBLISHED_D10 - worst:.4f})")
    print(f"problems whose infinite-restart bound is above {PUBLISHED_D10}: "
          f"{[r['func'] for r in out if r['mpr_sup'] > PUBLISHED_D10] or 'none'}"
          f"  (per-problem published values do not exist -- entry 92 section 6)")

    # paired 40-vs-200 drift, split by group: entry 91's operating rule said
    # group B was stable at 40 draws, on the evidence of M13 alone.
    for label, sel in (("group A (K=20)", lambda r: r["K"] == 20),
                       ("group B (K=10)", lambda r: r["K"] == 10)):
        big = [r for r in out if r["draws"] >= 200 and sel(r)]
        if not big:
            continue
        d = np.array([r["mpr"] - r["mpr_first40"] for r in big])
        w = (stats.wilcoxon(d)[1] if len(d) > 5 and np.any(d != 0) else float("nan"))
        print(f"\npaired 40->200 drift, {label}, {len(big)} problems "
              f"(200-draw minus draws 0-39): median {np.median(d):+.4f}, "
              f"range {d.min():+.4f}..{d.max():+.4f}, "
              f"{int((d > 0).sum())}/{len(d)} positive, Wilcoxon p = {w:.4g}")
        print("   per problem: " + ", ".join(
            f"{r['func'][:3]} {r['mpr_first40']:.3f}->{r['mpr']:.3f}" for r in big))

    # what the 16-problem mean would have been on the old (40-draw group B) mix
    old = np.array([r["mpr_first40"] if r["K"] == 10 else r["mpr"] for r in out])
    print(f"\n16-problem fixed-cost mean with group B at 40 draws: {old.mean():.4f}"
          f"   with group B at 200: {mpr.mean():.4f}"
          f"   (entry 92 reported 0.445 for the former)")

    # the gap to the published best closed between entry 92 and here for two
    # separable reasons: group B went 40 -> 200 draws, and entry 93's Chao1
    # correction did not exist when D=10 was last scored.  Split them by
    # recomputing every bound on the draws-0-39 subset of group B.
    rows40 = []
    for r, nm in zip(out, names):
        if r["K"] == 20:
            rows40.append((r["mpr_sup"], r["mpr_sup_chao1"]))
            continue
        best_f, land, evals, b, cross, draw, _ = load(nm)
        sel = np.flatnonzero(draw < 40)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        rows40.append((ceiling(best_f, land, evals, K, budget, sel)[2],
                       chao1_sup(best_f[sel], land[sel], K)))
    s40 = np.array([a for a, _ in rows40])
    c40 = np.array([c for _, c in rows40])
    print(f"\ngap to published {PUBLISHED_D10}, decomposed:"
          f"\n  infinite-restart, group B at 40 (entry 92's number): {s40.mean():.4f}"
          f"  -> gap {PUBLISHED_D10 - s40.mean():.4f}"
          f"\n  infinite-restart, group B at 200:                    {sup.mean():.4f}"
          f"  -> gap {PUBLISHED_D10 - sup.mean():.4f}"
          f"\n  + Chao1 on top of that:                              {ch.mean():.4f}"
          f"  -> gap {PUBLISHED_D10 - ch.mean():.4f}"
          f"\n  (Chao1 alone, group B at 40: {c40.mean():.4f})")


if __name__ == "__main__":
    main()
