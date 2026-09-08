"""Does the *descent model* move the F14-F20 coverage ceiling? (entry 89, queue item 2 (ii))

Entries 87/88 closed the coverage axis on F14-F20 from the side of the *class*:
"uniform restart + isotropic local descent" reaches fewer distinct optima per
run than the published best PR@1e-5 x K on all seven functions, and on the CF3
family the same two optima ({2, 3}) get zero landings at 3D, 5D and 10D.  The
queue's conclusion was that the remaining lever is the descent rule, not the
restart rule, and it named two candidates: (i) hill-valley-guided descent, (ii)
an anisotropic initial covariance.  It asked for (ii) first, because it is the
cheap one: flipping CMA-ES's covariance adaptation on inside the *null* answers
"can this descent model reach {2, 3} at all" with zero optimizer runs, and a
descent model that cannot move the ceiling is not worth implementing as an arm.

Design.  ``scripts/hunt_coverage.py --null --full-cov`` runs exactly the null of
e87/e88 with ``CMA_on`` left at its default instead of 0.  Draw k starts from
the same x0 and uses the same CMA seed in both modes, so every comparison below
is **paired draw by draw** against the stored isotropic CSVs (e88 for six
functions, e87 for N18).  Two budget allocations of the same 400000-evaluation
suite budget, as in e88: short descents (1499 evals = MC-ESO's own hunt length)
and long descents (CMA-ES stops itself); the class ceiling is the better of the
two.

Rejection condition, written before any number was read (queue item 2): if the
``--full-cov`` descent null still puts 0 draws on CF3's {2, 3}, anisotropy does
not reach them and route (i) -- hill-valley -- is the only one left.  If it does
reach them, the setting becomes an arm and gets measured on N16 / N18 with
``PRtrue@1e-1``.  The ceiling comparison (per-run distinct at f <= 1e-5 vs
published best x K) is reported for all seven functions on the same run, since
the question "does the ceiling move at all" is the general form of the same ask.

Accuracy discipline from e88 section 2: a landing is only counted for optimum j
if the endpoint is actually at the accuracy (``best_f <= eps``).  Nearest-
attribution alone reverses the sign on the high-dimensional functions.

Usage: python3 analysis/hm/e89/analyze.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
E87 = HERE.parent / "e87"
E88 = HERE.parent / "e88"
RNG = np.random.default_rng(20260908)
NSIM = 20000
SUITE_BUDGET = 400_000

# function -> (K, published best PR@1e-5, the optima the isotropic null never
# reaches at 3D/5D/10D -- e88 section 4, carried in only for the CF3 headline)
FUNCS = {
    "N14-CF3-3D": (6, 0.793),
    "N15-CF4-3D": (8, 0.750),
    "N16-CF3-5D": (6, 0.677),
    "N17-CF4-5D": (8, 0.745),
    "N18-CF3-10D": (6, 0.667),
    "N19-CF4-10D": (8, 0.512),
    "N20-CF4-20D": (8, 0.465),
}
CF3 = ["N14-CF3-3D", "N16-CF3-5D", "N18-CF3-10D"]
CF3_MISSING = [2, 3]          # e88 section 4


def _iso_path(func: str, budget: int) -> Path:
    """The stored isotropic CSV for this function and allocation.

    N18 was run by e87 under its own file names, and its long allocation used
    15000 evaluations where e88 used 14990 -- a 0.07% difference in the cap that
    neither run reaches (CMA-ES stops itself well before it), recorded here
    because the pairing is otherwise exact.
    """
    if func == "N18-CF3-10D":
        stem = "null_iso_b1499" if budget == 1499 else "null_iso_b15000"
        return E87 / f"{stem}.csv.gz"
    return E88 / f"null_iso_b{budget}_{func}.csv.gz"


def _cov_path(func: str, budget: int) -> Path:
    """This entry's own CSV, plain or gzipped (they are gzipped on commit)."""
    stem = HERE / f"null_cov_b{budget}_{func}.csv"
    return stem if stem.exists() else stem.with_suffix(".csv.gz")


def _read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def per_run_distinct(p: np.ndarray, n: int) -> tuple[float, float, float]:
    """Distinct optima covered by n independent landings drawn from p.

    p may sum to less than 1; the deficit is "this descent ended at no optimum
    at the accuracy" and is carried as a category contributing no coverage.
    """
    q = np.append(p, max(0.0, 1.0 - p.sum()))
    draws = RNG.multinomial(n, q, size=NSIM)[:, :len(p)]
    d = (draws > 0).sum(axis=1).astype(float)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def landing_probs(t: pd.DataFrame, K: int, eps: float | None) -> np.ndarray:
    sel = t if eps is None else t[t.best_f <= eps]
    c = np.zeros(K, dtype=np.int64)
    if len(sel):
        for j, n in sel.land_opt.value_counts().items():
            c[int(j)] = int(n)
    return c / len(t)


def reach_at(t: pd.DataFrame, eps: float) -> set[int]:
    sub = t[t.best_f <= eps]
    return set(int(j) for j in sub.land_opt.unique()) if len(sub) else set()


def one(func: str, budget: int) -> dict:
    K, pub = FUNCS[func]
    cov = _read(_cov_path(func, budget))
    iso = _read(_iso_path(func, budget))
    # pair on the draw index: same x0, same CMA seed, different descent model
    m = cov.merge(iso, on="draw", suffixes=("_cov", "_iso"))
    assert len(m) == min(len(cov), len(iso)), (len(m), len(cov), len(iso))

    out = {"func": func, "K": K, "pub": pub, "thr": pub * K, "budget": budget,
           "n_draws": len(cov), "n_paired": len(m)}
    for tag, t in (("cov", cov), ("iso", iso)):
        n_hunts = max(1, int(SUITE_BUDGET / t.evals.mean()))
        pr5 = per_run_distinct(landing_probs(t, K, 1e-5), n_hunts)
        pr1 = per_run_distinct(landing_probs(t, K, 1e-1), n_hunts)
        out[f"{tag}_n"] = n_hunts
        out[f"{tag}_evals_med"] = float(t.evals.median())
        out[f"{tag}_pr5"] = pr5[0]
        out[f"{tag}_pr5_lo"], out[f"{tag}_pr5_hi"] = pr5[1], pr5[2]
        out[f"{tag}_pr1"] = pr1[0]
        out[f"{tag}_frac1e1"] = float((t.best_f <= 0.1).mean())
        out[f"{tag}_frac1e5"] = float((t.best_f <= 1e-5).mean())
        out[f"{tag}_reach5"] = sorted(reach_at(t, 1e-5))
        out[f"{tag}_union"] = sorted(set(int(j) for j in t.land_opt.unique()))

    # paired: does the anisotropic descent from the same start end deeper?
    a, b = m.best_f_cov.to_numpy(), m.best_f_iso.to_numpy()
    diff = a - b
    nz = diff != 0
    if nz.sum():
        w = stats.wilcoxon(a[nz], b[nz])
        # matched-pair rank-biserial: positive = cov ends *lower* f = deeper
        out["w_p"] = float(w.pvalue)
        out["w_better"] = int((diff < 0).sum())
        out["w_worse"] = int((diff > 0).sum())
        out["w_tie"] = int((~nz).sum())
    else:
        out["w_p"], out["w_better"], out["w_worse"] = float("nan"), 0, 0
        out["w_tie"] = int(len(m))
    # paired: same start, different landing?
    out["moved"] = int((m.land_opt_cov != m.land_opt_iso).sum())
    # paired McNemar on "this draw is a scoring landing (f <= 1e-5)"
    hc = (m.best_f_cov <= 1e-5).to_numpy()
    hi = (m.best_f_iso <= 1e-5).to_numpy()
    b01, b10 = int((hc & ~hi).sum()), int((~hc & hi).sum())
    out["mc_cov_only"], out["mc_iso_only"] = b01, b10
    out["mc_p"] = (float(stats.binomtest(b01, b01 + b10, 0.5).pvalue)
                   if b01 + b10 else float("nan"))
    return out


def show(r: dict) -> None:
    f, K = r["func"], r["K"]
    print(f"\n{'=' * 78}\n== {f}  (budget {r['budget']}, {r['n_draws']} descents, K = {K})")
    for tag, label in (("iso", "isotropic (e87/e88)"), ("cov", "full covariance")):
        print(f"  {label:<22} median evals {r[tag + '_evals_med']:>6.0f}  "
              f"n/run {r[tag + '_n']:>4}  "
              f"frac f<=1e-1 {r[tag + '_frac1e1']:.3f}  "
              f"frac f<=1e-5 {r[tag + '_frac1e5']:.3f}")
        print(f"  {'':<22} reach@1e-5 {r[tag + '_reach5']}   "
              f"union(nearest) {r[tag + '_union']}")
        print(f"  {'':<22} PER-RUN distinct @1e-5 {r[tag + '_pr5']:.2f} "
              f"[{r[tag + '_pr5_lo']:.0f}, {r[tag + '_pr5_hi']:.0f}]   @1e-1 "
              f"{r[tag + '_pr1']:.2f}")
    print(f"  threshold pub x K        : {r['pub']:.3f} x {K} = {r['thr']:.2f}")
    print(f"  ceiling moves?           : iso {r['iso_pr5']:.2f} -> cov "
          f"{r['cov_pr5']:.2f}  (delta {r['cov_pr5'] - r['iso_pr5']:+.2f})")
    print(f"  VERDICT                  : "
          f"{'REJECT (headroom under anisotropy)' if r['cov_pr5'] > r['thr'] else 'no headroom'}")
    print(f"  paired, same start       : landing changed in {r['moved']}/{r['n_paired']} draws; "
          f"endpoint deeper/shallower/tied {r['w_better']}/{r['w_worse']}/{r['w_tie']}, "
          f"Wilcoxon p = {r['w_p']:.3g}")
    print(f"  paired scoring landings  : cov-only {r['mc_cov_only']}, iso-only "
          f"{r['mc_iso_only']}, McNemar p = {r['mc_p']:.3g}")


def cf3_headline(rows: list[dict]) -> None:
    print(f"\n{'=' * 78}\n== THE PRE-REGISTERED QUESTION: does an anisotropic descent reach "
          f"CF3's {CF3_MISSING}?")
    print(f"{'func':<13}{'budget':>7} | {'iso: draws on {2,3}':>22} | "
          f"{'cov: draws on {2,3}':>22}")
    print("-" * 72)
    hit = 0
    for r in rows:
        if r["func"] not in CF3:
            continue
        f, budget = r["func"], r["budget"]
        cov = _read(_cov_path(f, budget))
        iso = _read(_iso_path(f, budget))
        cells = []
        for t in (iso, cov):
            near = int(t.land_opt.isin(CF3_MISSING).sum())
            at5 = int((t.land_opt.isin(CF3_MISSING) & (t.best_f <= 1e-5)).sum())
            cells.append(f"{near} nearest / {at5} @1e-5")
        # the same comparison paired on the draw index, so "3 of 2000 vs 0 of
        # 2000" is read against the noise of the pairing rather than against 0.
        m = cov.merge(iso, on="draw", suffixes=("_cov", "_iso"))
        hc = (m.land_opt_cov.isin(CF3_MISSING) & (m.best_f_cov <= 1e-5)).to_numpy()
        hi = (m.land_opt_iso.isin(CF3_MISSING) & (m.best_f_iso <= 1e-5)).to_numpy()
        b01, b10 = int((hc & ~hi).sum()), int((~hc & hi).sum())
        p = (stats.binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10
             else float("nan"))
        hit += int(int(hc.sum()) > 0)
        print(f"{f:<13}{budget:>7} | {cells[0]:>22} | {cells[1]:>22} | "
              f"paired cov-only {b01}, iso-only {b10}, exact p = {p:.3g}")
    print(f"\n  rejection condition (0 draws on {CF3_MISSING} at f<=1e-5 in every CF3 "
          f"function and both allocations):")
    print(f"  -> {'NOT MET: anisotropy reaches them; implement as an arm' if hit else 'MET: anisotropy does not reach them; only route (i) hill-valley is left'}")


def main() -> None:
    rows = [one(f, b) for b in (1499, 14990) for f in FUNCS]
    for r in rows:
        show(r)

    print(f"\n{'=' * 78}\n== SUMMARY: distinct optima one run covers at f <= 1e-5, "
          f"isotropic vs anisotropic descent,\n   under both allocations of the same "
          f"{SUITE_BUDGET} evaluations. thr = published best PR@1e-5 x K.")
    print(f"\n{'func':<13}{'K':>3}{'thr':>7} |"
          f"{'iso':>7}{'cov':>7}{'d':>7} |{'iso':>7}{'cov':>7}{'d':>7} |"
          f"{'best cov':>9}{'verdict':>9}")
    print(f"{'':<23} |{'--- short descents ---':^23} |"
          f"{'--- long descents ---':^23} |")
    print("-" * 88)
    by = {(r["func"], r["budget"]): r for r in rows}
    for f in FUNCS:
        s, l = by[(f, 1499)], by[(f, 14990)]
        best = max(s["cov_pr5"], l["cov_pr5"])
        print(f"{f:<13}{s['K']:>3}{s['thr']:>7.2f} |"
              f"{s['iso_pr5']:>7.2f}{s['cov_pr5']:>7.2f}{s['cov_pr5'] - s['iso_pr5']:>+7.2f} |"
              f"{l['iso_pr5']:>7.2f}{l['cov_pr5']:>7.2f}{l['cov_pr5'] - l['iso_pr5']:>+7.2f} |"
              f"{best:>9.2f}{('REJECT' if best > s['thr'] else 'no'):>9}")

    cf3_headline(rows)

    pd.DataFrame([{k: v for k, v in r.items()
                   if not isinstance(v, list)} for r in rows]).to_csv(
        HERE / "summary.csv", index=False)
    print(f"\nsummary written to {HERE / 'summary.csv'}")


if __name__ == "__main__":
    main()
