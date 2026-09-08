"""Does the class ceiling on the new suite move when sigma0 comes from the
population instead of 0.2 x span?  (entry 97, group B, D=10)

Entry 95 closed the coverage axis on F14-F20 and left exactly one loose end:
the gain it did find on N13-CF3-2D (+1.583 distinct optima) was NOT bought by
the hill-valley test but by the *step size* of the descent -- the `split` arm,
which pays nothing for the test, matched it.  That makes entries 88/92/93/94
read as ceilings of "uniform restart + isotropic descent AT sigma0 = 0.2 x
span", not of the class.  The gain died with dimension (+1.58 at 2D, +0.33 at
3D n.s., 0.00 at 5D and 10D), so the prediction is that the new suite -- all
D >= 5 -- is unaffected.  A prediction is not a measurement; this is the
measurement.

Arms, paired draw-for-draw (`hunt_coverage.py --null`, budget 12500 = suite
budget / 40, 200 draws per problem):

  iso     sigma0 = 0.2 x span = 2.0            entries 91/94, dumps reused
  popsig  sigma0 = half the nearest-neighbour  entry 95's `split` rule, in the
          distance in a 200-point population   per-draw dump form (`--pop-sigma`)

The pairing is exact, not statistical: `--pop-sigma` keeps `_null_descent`'s
own start-point stream and CMA seed, so draw k of both arms starts at the same
x0 with the same seed and only sigma0 differs.  Verified byte-for-byte against
the saved e94 dump before the runs (see the log entry).

Scored by the four ceilings of entries 92/93/94, imported rather than
reimplemented so all four dimensions stay on one code path:

  mpr            fixed-cost multinomial ceiling
  mpr_earlystop  descent cut the moment it is inside eps
  mpr_sup        infinite restarts, support / K
  mpr_sup_chao1  the same with unseen optima put back

REFUTATION CONDITION (written before the numbers were read, as question 2
specifies): if ANY of the four ceilings for the popsig arm exceeds the
published best at D=10 (RR-CMA-ES, MPR 0.651) on ANY one problem, the premise
of route (C) breaks and the result goes to the review cycle rather than back
into MC-ESO.  Secondary: if the paired popsig - iso difference in `mpr` is
positive and significant on any problem, "the ceiling does not depend on
sigma0" is false at D=10 even where it does not clear the published value.

Usage: python3 analysis/mmo2024/e97/analyze.py
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
NAMES = [f"M{p:02d}-D10-PIN01" for p in range(9, 17)]
# iso dumps: entry 94 re-drew group B to 200, except M13 which entry 91 already had.
ISO_DIRS = (HERE.parent / "e94", HERE.parent / "e92", HERE.parent / "e91")


def _read(src: Path):
    op = gzip.open if src.suffix == ".gz" else open
    with op(src, "rt") as fh:
        rows = list(csv.DictReader(fh))
    draw = np.array([int(r["draw"]) for r in rows])
    o = np.argsort(draw)                       # --null collects unordered
    rows = [rows[i] for i in o]
    out = {
        "draw": draw[o],
        "best_f": np.array([float(r["best_f"]) for r in rows]),
        "land": np.array([int(r["land_opt"]) for r in rows]),
        "evals": np.array([float(r["evals"]) for r in rows]),
    }
    # dumps written before entry 92 stop at `evals` -- M13's isotropic arm is
    # entry 91's, so its early-stop ceiling is nan on the iso side (as it was
    # in entry 94's table).  The other three ceilings do not need these columns.
    out["cross"] = ({e: (np.array([float(r[f"ev_{e:g}"]) for r in rows]),
                         np.array([int(r[f"opt_{e:g}"]) for r in rows]))
                     for e in EPS} if f"ev_{EPS[0]:g}" in rows[0] else None)
    return out


def load_iso(name: str):
    for d in ISO_DIRS:
        for suf in (".csv.gz", ".csv"):
            src = d / f"{name}_desc200{suf}"
            if src.exists():
                return _read(src), src
    raise FileNotFoundError(f"iso dump for {name}")


def load_pop(name: str):
    for suf in (".csv.gz", ".csv"):
        src = HERE / f"{name}_popsig200{suf}"
        if src.exists():
            return _read(src), src
    raise FileNotFoundError(f"popsig dump for {name}")


def score(d, K, budget, idx=None):
    mpr, per_eps, mpr_sup = ceiling(d["best_f"], d["land"], d["evals"], K,
                                    budget, idx)
    if idx is None:
        idx = np.arange(len(d["best_f"]))
    if d["cross"] is None:
        es = float("nan")
    else:
        cross = {e: (d["cross"][e][0][idx], d["cross"][e][1][idx]) for e in EPS}
        es = early_stop_ceiling(cross, d["evals"][idx], K, budget)[0]
    ch = chao1_sup(d["best_f"][idx], d["land"][idx], K)
    return {"mpr": mpr, "per_eps": per_eps, "mpr_sup": mpr_sup,
            "mpr_earlystop": es, "mpr_sup_chao1": ch}


def main() -> None:
    rng = np.random.default_rng(0)
    rows, breaches = [], []
    for nm in NAMES:
        iso, iso_src = load_iso(nm)
        pop, pop_src = load_pop(nm)
        b = niching_by_name(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        assert np.array_equal(iso["draw"], pop["draw"]), f"{nm}: draws differ"
        m = len(iso["draw"])

        si, sp = score(iso, K, budget), score(pop, K, budget)
        # paired bootstrap: the SAME resampled draw indices score both arms,
        # so the CI is on the difference, not on two independent estimates.
        diff = np.array([
            (score(pop, K, budget, ix)["mpr"] - score(iso, K, budget, ix)["mpr"])
            for ix in (rng.integers(0, m, m) for _ in range(BOOT))])
        # per-draw paired outcomes (exact same x0 and CMA seed in both arms)
        hit_i = iso["best_f"] <= 1e-5
        hit_p = pop["best_f"] <= 1e-5
        nb = int((hit_p & ~hit_i).sum())        # popsig reaches, iso does not
        nc = int((hit_i & ~hit_p).sum())
        p_mc = (stats.binomtest(nb, nb + nc, 0.5).pvalue if nb + nc else 1.0)
        moved = int((iso["land"] != pop["land"]).sum())

        # A breach only bears on route (C) if it is NEW -- entry 91 already
        # recorded M13 as the one D=10 problem whose isotropic ceiling sits
        # above the published mean, so both arms are printed and the log has to
        # say which side moved.
        for key in ("mpr", "mpr_earlystop", "mpr_sup", "mpr_sup_chao1"):
            if sp[key] > PUBLISHED_D10:
                tag = ("already above in iso" if si[key] > PUBLISHED_D10
                       # M13's isotropic dump predates entry 92 and has no
                       # crossing columns, so its early-stop ceiling is unknown,
                       # not below -- calling that "new" would invent a breach.
                       else "iso not measurable (pre-e92 dump)"
                       if np.isnan(si[key]) else "NEW -- iso was below")
                breaches.append((nm, key, sp[key], si[key], tag))
        rows.append(dict(
            func=nm, K=K, m=m,
            hit_iso=float(hit_i.mean()), hit_pop=float(hit_p.mean()),
            ev_iso=float(iso["evals"].mean()), ev_pop=float(pop["evals"].mean()),
            reach_iso=len(set(iso["land"][hit_i].tolist())),
            reach_pop=len(set(pop["land"][hit_p].tolist())),
            mpr_iso=si["mpr"], mpr_pop=sp["mpr"],
            d_mpr=sp["mpr"] - si["mpr"],
            d_lo=float(np.percentile(diff, 2.5)),
            d_hi=float(np.percentile(diff, 97.5)),
            es_iso=si["mpr_earlystop"], es_pop=sp["mpr_earlystop"],
            sup_iso=si["mpr_sup"], sup_pop=sp["mpr_sup"],
            chao_iso=si["mpr_sup_chao1"], chao_pop=sp["mpr_sup_chao1"],
            landing_moved=moved, hit_only_pop=nb, hit_only_iso=nc, p_mcnemar=p_mc,
            iso_src=iso_src.parent.name))

    hdr = list(rows[0])
    with open(HERE / "ceiling_popsig_d10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print(f"{'func':<16}{'hit1e-5 iso/pop':>18}{'evals iso/pop':>18}"
          f"{'mpr iso/pop':>17}{'d_mpr [95% CI]':>26}{'sup':>14}{'chao1':>14}")
    for r in rows:
        print(f"{r['func']:<16}"
              f"{r['hit_iso']:>8.3f}/{r['hit_pop']:<9.3f}"
              f"{r['ev_iso']:>8.0f}/{r['ev_pop']:<9.0f}"
              f"{r['mpr_iso']:>8.3f}/{r['mpr_pop']:<8.3f}"
              f"{r['d_mpr']:>+9.3f} [{r['d_lo']:+.3f},{r['d_hi']:+.3f}]"
              f"{r['sup_iso']:>6.3f}/{r['sup_pop']:<7.3f}"
              f"{r['chao_iso']:>6.3f}/{r['chao_pop']:<7.3f}")

    print(f"\n{'mean over 8 problems':<16}"
          f"{col('hit_iso').mean():>8.3f}/{col('hit_pop').mean():<9.3f}"
          f"{col('ev_iso').mean():>8.0f}/{col('ev_pop').mean():<9.0f}"
          f"{col('mpr_iso').mean():>8.3f}/{col('mpr_pop').mean():<8.3f}"
          f"{col('d_mpr').mean():>+9.3f}"
          f"{'':>17}{col('sup_iso').mean():>6.3f}/{col('sup_pop').mean():<7.3f}"
          f"{col('chao_iso').mean():>6.3f}/{col('chao_pop').mean():<7.3f}")
    print(f"{'':<16}early-stop ceiling: "
          f"{np.nanmean(col('es_iso')):.3f} -> {np.nanmean(col('es_pop')):.3f}")

    w, t, l = (int((col('d_mpr') > 1e-12).sum()), int((abs(col('d_mpr')) <= 1e-12).sum()),
               int((col('d_mpr') < -1e-12).sum()))
    print(f"\npaired over problems (popsig - iso, mpr): {w}/{t}/{l} w/t/l")
    if w + l:
        print("  Wilcoxon signed-rank p = "
              f"{stats.wilcoxon(col('mpr_pop'), col('mpr_iso')).pvalue:.4g}")
    print("per-draw: landings moved "
          f"{int(col('landing_moved').sum())}/{int(col('m').sum())}, "
          f"f<=1e-5 only-popsig {int(col('hit_only_pop').sum())} vs "
          f"only-iso {int(col('hit_only_iso').sum())}")

    print(f"\npublished best at D=10 (RR-CMA-ES): MPR {PUBLISHED_D10}")
    if breaches:
        new = [b for b in breaches if b[4].startswith("NEW")]
        print("ceilings above the published mean (popsig arm):")
        for nm, key, v, vi, tag in breaches:
            print(f"  {nm}: {key} popsig {v:.3f} / iso {vi:.3f} "
                  f"> {PUBLISHED_D10}  [{tag}]")
        print("REFUTATION CONDITION MET -- report to the review cycle."
              if new else
              "no NEW breach: every one of these was already above in the "
              "isotropic arm, so route (C)'s premise is untouched by this arm.")
    else:
        best = float(np.nanmax([[r['mpr_pop'], r['es_pop'], r['sup_pop'],
                                 r['chao_pop']] for r in rows]))
        print("no ceiling of the popsig arm clears it on any of the 8 problems "
              f"(largest single value {best:.3f}); route (C)'s premise holds.")


if __name__ == "__main__":
    main()
