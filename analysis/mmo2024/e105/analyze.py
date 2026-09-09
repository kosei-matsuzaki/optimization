"""Does group A's sweep maximum sit at an interior sigma0, and does the swept
16-problem ceiling clear the published best?  (entry 105, D=10)

Entry 98 swept sigma0 on group B (M09-M16, K=10) over {0.05, 0.1, 0.2, 0.31} and
found the maximum is INTERIOR at 0.1.  Entry 99 added group A (M01-M08, K=20) at
0.1 only, so group A's best point was the smallest sigma ever run on it -- an
edge, not a bracket.  This cycle runs group A at 0.05 and closes that bracket.

Arms, paired draw-for-draw (`--null`, 200 draws, budget 12500 = suite budget /
40; `--sigma-ratio` moves sigma0 and nothing else, so draw k of every arm starts
at the same x0 with the same CMA seed):

  sig050   sigma0 = 0.05 x span = 0.5   group A: NEW here; group B: entry 98
  sig100   sigma0 = 0.10 x span = 1.0   group A: entry 99;  group B: entry 98
  iso      sigma0 = 0.20 x span = 2.0   entries 92 / 91 (A), 94 / 91 (B)

Everything is recomputed from the dumps, never from imported constants, so the
iso and sig100 columns reproduce entries 94/98/99 as an identity check.

`mpr_sup_chao1` is printed as an instrument only and is NOT compared with the
published best -- entry 104 §6 settled that it falls while `S_obs` rises.

REFUTATION CONDITION (prereg.md): if the 16-problem mean of `mpr` or `mpr_sup`
at the per-problem sweep maximum exceeds the published best at D=10 (RR-CMA-ES,
MPR 0.651), route (C)'s premise depends on sigma0 and the result goes to the
review cycle -- not back into MC-ESO (standing instruction, item 3).

STOPPING RULE (prereg.md): one further sigma point (0.025) is bought only if
group A's mean mpr(0.05) - mpr(0.1) >= +0.035 AND the paired Wilcoxon over the 8
problems has p < 0.05.  Otherwise the sigma sweep closes here and sigma = 0.4 is
never measured (entry 97 already has the large side falling 7/8 at 0.31, and a
point that can only lower the maximum cannot change a statement that takes it).

Usage: python3 analysis/mmo2024/e105/analyze.py
"""
import csv
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


# one scoring path for the whole theme: e99 -> e92/e93 (entry 104's invariant)
_e99 = _load_module("e99_analyze", HERE.parent / "e99" / "analyze.py")
load, score, KEYS = _e99.load, _e99.score, _e99.KEYS

PUBLISHED_D10 = 0.651        # competition best MPR at D=10 (external/mmo2024/docs)
BOOT = 2000
GROUP_A = [f"M{p:02d}-D10-PIN01" for p in range(1, 9)]     # K=20
GROUP_B = [f"M{p:02d}-D10-PIN01" for p in range(9, 17)]    # K=10
# arm label -> (suffix, directories searched in order)
ARMS = {
    "sig050": ("_sig050200", (HERE, HERE.parent / "e98")),
    "sig100": ("_sig100200", (HERE.parent / "e99", HERE.parent / "e98")),
    "iso":    ("_desc200",   (HERE.parent / "e92", HERE.parent / "e94",
                              HERE.parent / "e91")),
}
SIGMA = {"sig050": 0.05, "sig100": 0.10, "iso": 0.20}
SWEPT = ("sig050", "sig100", "iso")      # the set the maximum is taken over


def col(rs, k):
    return np.array([r[k] for r in rs], dtype=float)


def main() -> None:
    rng = np.random.default_rng(0)
    rows, missing = [], []
    for nm in GROUP_A + GROUP_B:
        b = niching_by_name(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        d = {a: load(dirs, nm, suf) for a, (suf, dirs) in ARMS.items()}
        assert d["iso"] is not None, f"no isotropic dump for {nm}"
        row = dict(func=nm, K=K, group="A" if nm in GROUP_A else "B",
                   m=len(d["iso"]["draw"]))
        for a in SWEPT:
            row[f"have_{a}"] = d[a] is not None
            if d[a] is None:
                missing.append(f"{nm[:3]}/{a}")
                continue
            # every arm must pair draw-for-draw with the isotropic dump
            assert np.array_equal(d[a]["draw"], d["iso"]["draw"]), f"{nm} {a} draws"
            s = score(d[a], K, budget)
            hit = d[a]["best_f"] <= 1e-5
            row[f"hit_{a}"] = float(hit.mean())
            row[f"ev_{a}"] = float(d[a]["evals"].mean())
            row[f"reach_{a}"] = len(set(d[a]["land"][hit].tolist()))
            for k in KEYS:
                row[f"{k}_{a}"] = s[k]
        # per-problem sweep maximum over the arms actually present
        for k in KEYS:
            vals = {a: row.get(f"{k}_{a}", float("nan")) for a in SWEPT}
            fin = {a: v for a, v in vals.items() if np.isfinite(v)}
            row[f"{k}_max"] = max(fin.values()) if fin else float("nan")
            row[f"{k}_argmax"] = (max(fin, key=fin.get) if fin else "-")
        # paired diagnostics: 0.05 against 0.1 (the bracket test)
        lo, hi = d["sig050"], d["sig100"]
        if lo is not None and hi is not None:
            m = len(hi["draw"])
            diff = np.array([
                (score(lo, K, budget, ix)["mpr"] - score(hi, K, budget, ix)["mpr"])
                for ix in (rng.integers(0, m, m) for _ in range(BOOT))])
            row["d_mpr"] = row["mpr_sig050"] - row["mpr_sig100"]
            row["d_lo"] = float(np.percentile(diff, 2.5))
            row["d_hi"] = float(np.percentile(diff, 97.5))
            hl, hh = lo["best_f"] <= 1e-5, hi["best_f"] <= 1e-5
            nb, nc = int((hl & ~hh).sum()), int((hh & ~hl).sum())
            row["moved"] = int((lo["land"] != hi["land"]).sum())
            row["only_050"], row["only_100"] = nb, nc
            row["p_mcnemar"] = (stats.binomtest(nb, nb + nc, 0.5).pvalue
                                if nb + nc else 1.0)
        else:
            for c in ("d_mpr", "d_lo", "d_hi", "moved", "only_050", "only_100",
                      "p_mcnemar"):
                row[c] = float("nan")
        rows.append(row)

    keys = sorted({k for r in rows for k in r})
    with open(HERE / "sweep_sigma_d10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

    A = [r for r in rows if r["group"] == "A" and r["have_sig050"]]
    print("entry 105 -- group A at sigma0 = 0.05 x span, 200 draws, budget 12500")
    print(f"  measured here: {[r['func'][:3] for r in A] or 'none'}")
    print(f"  arms missing:  {missing or 'none'}")

    print("\nper-problem `mpr` across the swept sigma0 (0.05 / 0.1 / 0.2):")
    print(f"{'func':<7}{'K':>3}{'gp':>4}{'mpr .05':>9}{'mpr .10':>9}{'mpr .20':>9}"
          f"{'argmax':>9}{'hit .05':>9}{'hit .10':>9}{'hit .20':>9}")
    for r in rows:
        def g(k):
            v = r.get(k, float("nan"))
            return f"{v:>9.3f}" if np.isfinite(v) else f"{'-':>9}"
        print(f"{r['func'][:3]:<7}{r['K']:>3}{r['group']:>4}"
              f"{g('mpr_sig050')}{g('mpr_sig100')}{g('mpr_iso')}"
              f"{r['mpr_argmax']:>9}{g('hit_sig050')}{g('hit_sig100')}{g('hit_iso')}")

    if A:
        print("\ngroup A, sigma 0.05 vs 0.1 (paired on the same draws):")
        print(f"{'func':<7}{'d_mpr':>9}{'95% CI':>22}{'moved':>7}"
              f"{'only.05/only.10':>18}{'reach .05/.10':>15}")
        for r in A:
            print(f"{r['func'][:3]:<7}{r['d_mpr']:>+9.3f}"
                  f"   [{r['d_lo']:+.3f},{r['d_hi']:+.3f}]{int(r['moved']):>10d}"
                  f"{int(r['only_050']):>10d}/{int(r['only_100']):<7d}"
                  f"{int(r['reach_sig050']):>6d}/{int(r['reach_sig100']):<8d}")
        dd = col(A, "d_mpr")
        w_, t_, l_ = (int((dd > 1e-12).sum()), int((abs(dd) <= 1e-12).sum()),
                      int((dd < -1e-12).sum()))
        p_w = (stats.wilcoxon(col(A, "mpr_sig050"), col(A, "mpr_sig100")).pvalue
               if w_ + l_ else 1.0)
        print(f"\n  mean d_mpr {dd.mean():+.4f}/problem (w/t/l = {w_}/{t_}/{l_}), "
              f"Wilcoxon p = {p_w:.4g}")
        print(f"  CI excludes 0: up on "
              f"{[r['func'][:3] for r in A if r['d_lo'] > 0] or 'none'}, down on "
              f"{[r['func'][:3] for r in A if r['d_hi'] < 0] or 'none'}")
        print(f"  landings moved {int(col(A,'moved').sum())}/{int(col(A,'m').sum())}, "
              f"f<=1e-5 only-0.05 {int(col(A,'only_050').sum())} vs only-0.1 "
              f"{int(col(A,'only_100').sum())}  <- firing check (entry 86 form 3)")
        print(f"  hit@1e-5 {col(A,'hit_sig100').mean():.3f} -> "
              f"{col(A,'hit_sig050').mean():.3f}; distinct optima at 1e-5 "
              f"{int(col(A,'reach_sig100').sum())} -> {int(col(A,'reach_sig050').sum())}"
              f" of {int(col(A,'K').sum())}")
        print(f"  mean evals/descent {col(A,'ev_sig100').mean():.0f} -> "
              f"{col(A,'ev_sig050').mean():.0f}")
        fires = (dd.mean() >= 0.035) and (p_w < 0.05)
        print(f"\n  STOPPING RULE: mean d_mpr {dd.mean():+.4f} (>= +0.035? "
              f"{dd.mean() >= 0.035}) and p {p_w:.4g} (< 0.05? {p_w < 0.05}) -> "
              + ("BUY the sigma = 0.025 point" if fires else
                 "SIGMA SWEEP CLOSED (sigma = 0.4 is never measured)"))

    # Two aggregations, both needed.  (i) the HEADLINE is the mean over all 16
    # problems of the per-problem sweep maximum -- M05-M08 have no 0.05 arm, so
    # they enter at max(0.1, 0.2), which is the conservative direction (a point
    # not yet measured can only raise a maximum, never lower it).  (ii) the
    # fixed-sigma columns are only comparable on the subset where all three arms
    # exist, so they are printed on that subset with its own n (entry 99's fix:
    # a nan on one side silently compares different problem sets).
    print("\n16-problem mean of the per-problem sweep maximum over sigma0 "
          "(the only unit with a published counterpart -- entry 92 §6):")
    print(f"{'ceiling':<18}{'sweep max':>11}{'published':>11}{'margin':>10}{'n':>4}")
    breach = []
    for k in KEYS:
        mx = col(rows, f"{k}_max")
        fin = np.isfinite(mx)
        vmax = float(mx[fin].mean())
        note = "  (instrument only)" if k == "mpr_sup_chao1" else ""
        print(f"{k:<18}{vmax:>11.4f}{PUBLISHED_D10:>11.3f}"
              f"{PUBLISHED_D10 - vmax:>+10.4f}{int(fin.sum()):>4}{note}")
        if k in ("mpr", "mpr_sup") and vmax > PUBLISHED_D10:
            breach.append((k, vmax, int(fin.sum())))

    print("\nfixed-sigma columns, on the subset where all three arms exist:")
    print(f"{'ceiling':<18}{'s=0.05':>10}{'s=0.10':>10}{'s=0.20':>10}"
          f"{'sweep max':>11}{'n':>4}")
    for k in KEYS:
        cols = {a: np.array([r.get(f"{k}_{a}", np.nan) for r in rows], float)
                for a in SWEPT}
        mx = col(rows, f"{k}_max")
        both = np.all([np.isfinite(v) for v in cols.values()], axis=0) & np.isfinite(mx)
        vals = {a: float(cols[a][both].mean()) for a in SWEPT}
        print(f"{k:<18}{vals['sig050']:>10.4f}{vals['sig100']:>10.4f}"
              f"{vals['iso']:>10.4f}{float(mx[both].mean()):>11.4f}"
              f"{int(both.sum()):>4}")
    if breach:
        print("\nREFUTATION CONDITION MET -- sweep-max 16-problem mean above the "
              "published best on: "
              + ", ".join(f"{k} = {v:.4f} (n={n})" for k, v, n in breach))
        print("Report to the review cycle; do NOT route this back into MC-ESO "
              "(standing instruction, item 3).")
    else:
        print("\nneither `mpr` nor `mpr_sup` clears 0.651 at the per-problem "
              "sweep maximum; route (C)'s premise survives the sigma sweep.")


if __name__ == "__main__":
    main()
