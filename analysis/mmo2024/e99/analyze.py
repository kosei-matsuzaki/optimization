"""Does the class ceiling clear the published best once group A is swept too?
(entry 99, group A, D=10, K=20, sigma0 = 0.1 x span)

Entry 98 swept sigma0 on group B (M09-M16, K=10) and found the ceiling rises on
the small side; the margin of the most permissive ceiling in the only comparable
unit -- the 16-problem mean, since per-problem published values do not exist
(entry 92 §6) -- shrank from 0.070 to 0.017.  That number still carried group A
(M01-M08, K=20) at its isotropic values, and entry 98 wrote the extrapolation
explicitly: group A rising by the same +0.106/problem puts the 16-problem mean at
0.687, above the published 0.651.  This script measures group A instead.

Arms, paired draw-for-draw (`--null`, 200 draws, budget 12500 = suite budget /
40; `--sigma-ratio` moves sigma0 and nothing else, so draw k of every arm starts
at the same x0 with the same CMA seed):

  sig100   sigma0 = 0.10 x span = 1.0   group A: new here; group B: entry 98
  iso      sigma0 = 0.20 x span = 2.0   entries 92 / 91 (A), 94 / 91 (B), reused

The 16-problem mean is computed from dumps on both sides, not from imported
constants, so the isotropic column here reproduces entry 94's numbers as a check.
Group A problems whose sig100 dump is missing (the cycle was split M01-M04 /
M05-M08 on cost) are held at their isotropic value and named in the output; that
is the conservative direction for the refutation condition below.

REFUTATION CONDITION (pre-registered in prereg.md): if the 16-problem mean of any
of the four ceilings exceeds the published best at D=10 (RR-CMA-ES, MPR 0.651),
route (C)'s premise depends on the choice of sigma0 and the result goes to the
review cycle rather than back into MC-ESO (standing instruction, item 3).

Usage: python3 analysis/mmo2024/e99/analyze.py
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
KEYS = ("mpr", "mpr_earlystop", "mpr_sup", "mpr_sup_chao1")
GROUP_A = [f"M{p:02d}-D10-PIN01" for p in range(1, 9)]     # K=20, swept here
GROUP_B = [f"M{p:02d}-D10-PIN01" for p in range(9, 17)]    # K=10, entry 98
# sig100 dumps live here for group A and in e98 for group B; the isotropic
# baseline is entry 92 (A: M02-M08), 94 (B: M09-M12, M14-M16), 91 (M01, M13).
SIG_DIRS = (HERE, HERE.parent / "e98")
ISO_DIRS = (HERE.parent / "e92", HERE.parent / "e94", HERE.parent / "e91")


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
    # Dumps written before entry 92 stop at `evals` (M01's and M13's isotropic
    # arms are entry 91's), so their early-stop ceiling is nan -- unknown, not
    # below.
    out["cross"] = ({e: (np.array([float(r[f"ev_{e:g}"]) for r in rows]),
                         np.array([int(r[f"opt_{e:g}"]) for r in rows]))
                     for e in EPS} if f"ev_{EPS[0]:g}" in rows[0] else None)
    return out


def load(dirs, name, suffix):
    for d in dirs:
        for ext in (".csv.gz", ".csv"):
            src = d / f"{name}{suffix}{ext}"
            if src.exists():
                return _read(src)
    return None


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
    rows, missing = [], []
    for nm in GROUP_A + GROUP_B:
        b = niching_by_name(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        iso = load(ISO_DIRS, nm, "_desc200")
        sig = load(SIG_DIRS, nm, "_sig100200")
        assert iso is not None, f"no isotropic dump for {nm}"
        row = dict(func=nm, K=K, group="A" if nm in GROUP_A else "B",
                   m=len(iso["draw"]), swept=sig is not None)
        if sig is None:
            missing.append(nm)
        else:
            assert np.array_equal(sig["draw"], iso["draw"]), f"{nm} draws"
        s_iso = score(iso, K, budget)
        s_sig = score(sig, K, budget) if sig is not None else s_iso
        for lab, d, sc in (("iso", iso, s_iso), ("sig100", sig or iso, s_sig)):
            hit = d["best_f"] <= 1e-5
            row[f"hit_{lab}"] = float(hit.mean())
            row[f"ev_{lab}"] = float(d["evals"].mean())
            row[f"reach_{lab}"] = len(set(d["land"][hit].tolist()))
            for k in KEYS:
                row[f"{k}_{lab}"] = sc[k]
        if sig is not None:
            # paired bootstrap: the SAME resampled draw indices score both arms,
            # so the CI is on the difference, not on two independent estimates.
            m = len(iso["draw"])
            diff = np.array([
                (score(sig, K, budget, ix)["mpr"] - score(iso, K, budget, ix)["mpr"])
                for ix in (rng.integers(0, m, m) for _ in range(BOOT))])
            row["d_mpr"] = s_sig["mpr"] - s_iso["mpr"]
            row["d_lo"] = float(np.percentile(diff, 2.5))
            row["d_hi"] = float(np.percentile(diff, 97.5))
            hb, hi_ = sig["best_f"] <= 1e-5, iso["best_f"] <= 1e-5
            nb, nc = int((hb & ~hi_).sum()), int((hi_ & ~hb).sum())
            row["moved"] = int((sig["land"] != iso["land"]).sum())
            row["only_sig"], row["only_iso"] = nb, nc
            row["p_mcnemar"] = (stats.binomtest(nb, nb + nc, 0.5).pvalue
                                if nb + nc else 1.0)
        else:
            for c in ("d_mpr", "d_lo", "d_hi", "moved", "only_sig", "only_iso",
                      "p_mcnemar"):
                row[c] = float("nan")
        rows.append(row)

    with open(HERE / "ceiling_groupA_d10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    A = [r for r in rows if r["group"] == "A"]
    Asw = [r for r in A if r["swept"]]

    def col(rs, k):
        return np.array([r[k] for r in rs], dtype=float)

    print(f"group A (K=20), sigma0 = 0.1 x span vs 0.2, 200 draws, budget 12500")
    print(f"  swept this cycle: {[r['func'][:3] for r in Asw] or 'none'}; "
          f"held at isotropic: {[n[:3] for n in missing] or 'none'}")
    print(f"\n{'func':<16}{'K':>4}{'hit iso':>9}{'hit .1':>9}"
          f"{'mpr iso':>9}{'mpr .1':>9}{'d_mpr [95% CI]':>26}"
          f"{'moved':>7}{'reach iso/.1':>14}")
    for r in A:
        ci = (f"{r['d_mpr']:>+9.3f} [{r['d_lo']:+.3f},{r['d_hi']:+.3f}]"
              if r["swept"] else f"{'(not swept)':>26}")
        mv = f"{int(r['moved']):>7d}" if r["swept"] else f"{'-':>7}"
        print(f"{r['func']:<16}{r['K']:>4}{r['hit_iso']:>9.3f}{r['hit_sig100']:>9.3f}"
              f"{r['mpr_iso']:>9.3f}{r['mpr_sig100']:>9.3f}{ci}{mv}"
              f"{int(r['reach_iso']):>7d}/{int(r['reach_sig100']):<6d}")
    if Asw:
        d = col(Asw, "d_mpr")
        w_, t_, l_ = (int((d > 1e-12).sum()), int((abs(d) <= 1e-12).sum()),
                      int((d < -1e-12).sum()))
        print(f"\n  swept group A: mean d_mpr {d.mean():+.4f} per problem "
              f"(w/t/l = {w_}/{t_}/{l_})")
        if w_ + l_ and len(Asw) > 1:
            print("  Wilcoxon signed-rank p = "
                  f"{stats.wilcoxon(col(Asw,'mpr_sig100'), col(Asw,'mpr_iso')).pvalue:.4g}")
        up = [r["func"][:3] for r in Asw if r["d_lo"] > 0]
        dn = [r["func"][:3] for r in Asw if r["d_hi"] < 0]
        print(f"  CI excludes 0: up on {up or 'none'}, down on {dn or 'none'}")
        print(f"  landings moved {int(col(Asw,'moved').sum())}/{int(col(Asw,'m').sum())}, "
              f"f<=1e-5 only-sig100 {int(col(Asw,'only_sig').sum())} vs only-iso "
              f"{int(col(Asw,'only_iso').sum())}")
        print(f"  hit@1e-5 {col(Asw,'hit_iso').mean():.3f} -> "
              f"{col(Asw,'hit_sig100').mean():.3f}; distinct optima reached "
              f"{int(col(Asw,'reach_iso').sum())} -> {int(col(Asw,'reach_sig100').sum())} "
              f"of {int(col(Asw,'K').sum())}")
        print(f"  mean evals/descent {col(Asw,'ev_iso').mean():.0f} -> "
              f"{col(Asw,'ev_sig100').mean():.0f}")
        print(f"\n  entry 98's extrapolation was group B's +0.106/problem; "
              f"measured here: {d.mean():+.4f}/problem")

    print("\n16-problem mean of each ceiling (the only unit with a published "
          "counterpart -- entry 92 §6):")
    print(f"{'ceiling':<18}{'isotropic':>12}{'sigma=0.1':>12}{'published':>12}"
          f"{'margin':>10}{'n':>4}")
    breach = []
    for k in KEYS:
        # both arms averaged over the SAME problems: M01's and M13's isotropic
        # dumps predate the crossing columns, so their `mpr_earlystop` is nan on
        # one side only and a plain nanmean would compare different problem sets.
        both = np.isfinite(col(rows, f"{k}_iso")) & np.isfinite(col(rows, f"{k}_sig100"))
        v_iso = float(col(rows, f"{k}_iso")[both].mean())
        v_sig = float(col(rows, f"{k}_sig100")[both].mean())
        print(f"{k:<18}{v_iso:>12.4f}{v_sig:>12.4f}{PUBLISHED_D10:>12.3f}"
              f"{PUBLISHED_D10 - v_sig:>+10.4f}{int(both.sum()):>4}")
        if v_sig > PUBLISHED_D10:
            breach.append((k, v_sig, int(both.sum())))
    print(f"  group A held at isotropic for "
          f"{[n[:3] for n in missing] or 'nothing'}; n < 16 means that ceiling "
          f"is undefined on some problem in BOTH arms' average")
    if breach:
        print("\nREFUTATION CONDITION MET -- 16-problem mean above the published "
              "best on: " + ", ".join(f"{k} = {v:.4f} (n={n})" for k, v, n in breach))
        print("Report to the review cycle; do NOT route this back into MC-ESO "
              "(standing instruction, item 3).")
    else:
        print("\nno ceiling's 16-problem mean clears 0.651; route (C)'s premise "
              "survives the sweep at sigma = 0.1 on both groups.")


if __name__ == "__main__":
    main()
