"""Does the class ceiling on the new suite clear the published best at ANY
step size?  (entry 98, group B, D=10, sigma0 sweep -- small side)

Entry 97 showed the ceiling of entries 88/92/93/94 is not the ceiling of the
class but the ceiling *at* `sigma0 = 0.2 x span`: moving sigma0 up to the
population spread (3.108) changed 510 of 1600 landings and lowered `mpr` on 7
of 8 problems.  Question 2's remaining step is the sweep, so that the sentence
"the class ceiling stays under the published best" can drop the qualifier.

Arms, all paired draw-for-draw (`--null`, 200 draws, budget 12500 = suite
budget / 40; `--sigma-ratio` moves sigma0 and nothing else, so draw k of every
arm starts at the same x0 with the same CMA seed):

  sig050   sigma0 = 0.05 x span = 0.5    new here
  sig100   sigma0 = 0.10 x span = 1.0    new here
  iso      sigma0 = 0.20 x span = 2.0    entries 94 / 91, dumps reused
  popsig   sigma0 = population spread    entry 97, dump reused (~0.31 x span)

0.4 is deferred to the next cycle: entry 97 already measured that direction
(sigma0 up => ceiling down on 7/8), so the small side is where a rise could
still be hiding.

REFUTATION CONDITION (pre-registered in prereg.md): if the per-problem maximum
over the sweep exceeds the published best at D=10 (RR-CMA-ES, MPR 0.651) on any
problem that was not already above it in the isotropic arm -- M13 is, entry 91
recorded it -- route (C)'s premise depends on sigma0 and the result goes to the
review cycle rather than back into MC-ESO.  Secondary: a positive paired `mpr`
difference with a bootstrap CI excluding zero on any problem means the sweep has
to be pushed below 0.05 before "class ceiling" is used unqualified.

Usage: python3 analysis/mmo2024/e98/analyze.py
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
NAMES = [f"M{p:02d}-D10-PIN01" for p in range(9, 17)]
# (arm label, directory, filename suffix) -- iso and popsig are reused dumps.
ARMS = [
    ("sig050", (HERE,), "_sig050200"),
    ("sig100", (HERE,), "_sig100200"),
    ("iso", (HERE.parent / "e94", HERE.parent / "e92", HERE.parent / "e91"),
     "_desc200"),
    ("popsig", (HERE.parent / "e97",), "_popsig200"),
]
NEW_ARMS = ("sig050", "sig100")
BASE = "iso"


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
    # Dumps written before entry 92 stop at `evals` (M13's isotropic arm is
    # entry 91's), so its early-stop ceiling is nan -- unknown, not below.
    out["cross"] = ({e: (np.array([float(r[f"ev_{e:g}"]) for r in rows]),
                         np.array([int(r[f"opt_{e:g}"]) for r in rows]))
                     for e in EPS} if f"ev_{EPS[0]:g}" in rows[0] else None)
    return out


def load(dirs, name, suffix):
    for d in dirs:
        for ext in (".csv.gz", ".csv"):
            src = d / f"{name}{suffix}{ext}"
            if src.exists():
                return _read(src), src
    raise FileNotFoundError(f"{name}{suffix} in {[str(d) for d in dirs]}")


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
        b = niching_by_name(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        dumps, srcs = {}, {}
        for label, dirs, suf in ARMS:
            dumps[label], srcs[label] = load(dirs, nm, suf)
        base = dumps[BASE]
        m = len(base["draw"])
        for label, d in dumps.items():
            assert np.array_equal(d["draw"], base["draw"]), f"{nm}/{label} draws"

        sc = {label: score(d, K, budget) for label, d in dumps.items()}
        row = dict(func=nm, K=K, m=m)
        for label, d in dumps.items():
            hit = d["best_f"] <= 1e-5
            row[f"hit_{label}"] = float(hit.mean())
            row[f"ev_{label}"] = float(d["evals"].mean())
            row[f"reach_{label}"] = len(set(d["land"][hit].tolist()))
            for k in KEYS:
                row[f"{k}_{label}"] = sc[label][k]
        # paired bootstrap of each new arm against the 0.2 baseline: the SAME
        # resampled draw indices score both arms, so the CI is on the difference.
        for label in NEW_ARMS:
            diff = np.array([
                (score(dumps[label], K, budget, ix)["mpr"]
                 - score(base, K, budget, ix)["mpr"])
                for ix in (rng.integers(0, m, m) for _ in range(BOOT))])
            row[f"d_mpr_{label}"] = sc[label]["mpr"] - sc[BASE]["mpr"]
            row[f"d_lo_{label}"] = float(np.percentile(diff, 2.5))
            row[f"d_hi_{label}"] = float(np.percentile(diff, 97.5))
            hb = dumps[label]["best_f"] <= 1e-5
            hi_ = base["best_f"] <= 1e-5
            nb, nc = int((hb & ~hi_).sum()), int((hi_ & ~hb).sum())
            row[f"moved_{label}"] = int((dumps[label]["land"] != base["land"]).sum())
            row[f"only_{label}"] = nb
            row[f"only_iso_vs_{label}"] = nc
            row[f"p_mcnemar_{label}"] = (stats.binomtest(nb, nb + nc, 0.5).pvalue
                                         if nb + nc else 1.0)
        # the headline: best over the sweep, per problem, per ceiling
        for k in KEYS:
            vals = {label: sc[label][k] for label, _, _ in ARMS}
            best_label = max(vals, key=lambda s: (-np.inf if np.isnan(vals[s])
                                                  else vals[s]))
            row[f"max_{k}"] = vals[best_label]
            row[f"argmax_{k}"] = best_label
            if vals[best_label] > PUBLISHED_D10:
                iso_v = vals[BASE]
                tag = ("already above in iso" if iso_v > PUBLISHED_D10 else
                       "iso not measurable (pre-e92 dump)" if np.isnan(iso_v)
                       else "NEW -- iso was below")
                breaches.append((nm, k, best_label, vals[best_label], iso_v, tag))
        rows.append(row)

    with open(HERE / "ceiling_sweep_d10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    labels = [a[0] for a in ARMS]
    print("sigma0 sweep, group B, D=10, 200 draws, budget 12500")
    print(f"{'func':<16}" + "".join(f"{'hit ' + s:>12}" for s in labels)
          + "".join(f"{'mpr ' + s:>12}" for s in labels)
          + f"{'max mpr':>10}{'arm':>9}")
    for r in rows:
        print(f"{r['func']:<16}"
              + "".join(f"{r['hit_' + s]:>12.3f}" for s in labels)
              + "".join(f"{r['mpr_' + s]:>12.3f}" for s in labels)
              + f"{r['max_mpr']:>10.3f}{r['argmax_mpr']:>9}")
    print(f"{'mean':<16}"
          + "".join(f"{col('hit_' + s).mean():>12.3f}" for s in labels)
          + "".join(f"{col('mpr_' + s).mean():>12.3f}" for s in labels)
          + f"{col('max_mpr').mean():>10.3f}")

    print("\ndistinct optima reached at f<=1e-5 (support of the arm):")
    print(f"{'func':<16}{'K':>4}" + "".join(f"{s:>10}" for s in labels))
    for r in rows:
        print(f"{r['func']:<16}{r['K']:>4}"
              + "".join(f"{r['reach_' + s]:>10d}" for s in labels))
    print(f"{'sum':<16}{int(col('K').sum()):>4}"
          + "".join(f"{int(col('reach_' + s).sum()):>10d}" for s in labels))

    print("\nmean evaluations per descent (budget 12500):")
    print(f"{'':<16}" + "".join(f"{s:>10}" for s in labels))
    print(f"{'mean':<16}" + "".join(f"{col('ev_' + s).mean():>10.0f}"
                                    for s in labels))

    for label in NEW_ARMS:
        print(f"\npaired vs iso (sigma 0.2) -- arm {label}")
        print(f"{'func':<16}{'d_mpr [95% CI]':>26}{'moved':>8}"
              f"{'only_new/only_iso':>20}{'p_mcnemar':>12}")
        for r in rows:
            print(f"{r['func']:<16}"
                  f"{r['d_mpr_' + label]:>+9.3f} "
                  f"[{r['d_lo_' + label]:+.3f},{r['d_hi_' + label]:+.3f}]"
                  f"{r['moved_' + label]:>8d}"
                  f"{r['only_' + label]:>10d}/{r['only_iso_vs_' + label]:<9d}"
                  f"{r['p_mcnemar_' + label]:>12.3g}")
        d = col("d_mpr_" + label)
        w_, t_, l_ = (int((d > 1e-12).sum()), int((abs(d) <= 1e-12).sum()),
                      int((d < -1e-12).sum()))
        print(f"  problems w/t/l = {w_}/{t_}/{l_}, mean d_mpr {d.mean():+.3f}")
        if w_ + l_:
            print("  Wilcoxon signed-rank p = "
                  f"{stats.wilcoxon(col('mpr_' + label), col('mpr_iso')).pvalue:.4g}")
        up = [r['func'] for r in rows if r[f'd_lo_{label}'] > 0]
        dn = [r['func'] for r in rows if r[f'd_hi_{label}'] < 0]
        print(f"  CI excludes 0: up on {up or 'none'}, down on {dn or 'none'}")
        print(f"  landings moved {int(col('moved_' + label).sum())}"
              f"/{int(col('m').sum())}, f<=1e-5 only-{label} "
              f"{int(col('only_' + label).sum())} vs only-iso "
              f"{int(col('only_iso_vs_' + label).sum())}")

    print("\nper-problem maximum over the sweep, all four ceilings:")
    print(f"{'func':<16}" + "".join(f"{k:>16}" for k in KEYS))
    for r in rows:
        print(f"{r['func']:<16}"
              + "".join(f"{r['max_' + k]:>9.3f} {r['argmax_' + k]:<6}"
                        for k in KEYS))
    print(f"{'mean':<16}" + "".join(f"{np.nanmean(col('max_' + k)):>9.3f} "
                                    f"{'':<6}" for k in KEYS))

    print(f"\npublished best at D=10 (RR-CMA-ES): MPR {PUBLISHED_D10}")
    if breaches:
        new = [x for x in breaches if x[5].startswith("NEW")]
        print("sweep maxima above the published mean:")
        for nm, k, lab, v, iso_v, tag in breaches:
            print(f"  {nm}: {k} = {v:.3f} at {lab} (iso {iso_v:.3f}) "
                  f"> {PUBLISHED_D10}  [{tag}]")
        print("REFUTATION CONDITION MET -- report to the review cycle."
              if new else
              "no NEW breach: every one was already above in the isotropic arm, "
              "so route (C)'s premise survives the sweep measured so far.")
    else:
        best = float(np.nanmax([[r[f'max_{k}'] for k in KEYS] for r in rows]))
        print("no ceiling at any swept sigma0 clears it on any of the 8 problems "
              f"(largest single value {best:.3f}); route (C)'s premise holds.")


if __name__ == "__main__":
    main()
