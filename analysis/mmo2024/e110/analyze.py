"""entry 110 -- score the restart-lander null as a run and take the estimate apart.

Reads:
  by_problem/*.csv   the harness's own scoring of the runs (rule = current)
  descents/*.csv     one row per descent: landing optimum, best_f, evals, stop
  ../e106/mceso_vs_null_d10.csv   per-problem MC-ESO and entry 103's estimate
  ../e109/nmmso_d10.csv           per-problem NMMSO

Writes lander_d10.csv (per problem) and prints the report the log entry quotes.

The four candidate over-estimation sources from the pre-registration are all
computed off the same runs:
  cap        harness MPR (reported set capped at max(100,2K)) vs the same
             reported set uncapped
  restarts   descents the run actually ran vs n' = budget / mean descent cost
  iid        entry 103's multinomial formula recomputed on this run's own
             landings, vs the same formula on the 500-descent offline sample
  stop       what ends a descent when tolfun = tolfunhist = tolx = 0

Usage: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e110/analyze.py
"""
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name          # noqa: E402

EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
NAMES = [f"M{p:02d}-D10-PIN01" for p in range(1, 17)]
BUDGET = 500_000
PUBLISHED = 0.651


def wilcoxon(a, b, label):
    """Two-sided signed-rank on the paired difference, with rank-biserial."""
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return f"{label}: all 16 differences are zero"
    W, p = stats.wilcoxon(d)
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return (f"{label}: mean diff {d.mean():+.4f}, "
            f"{int((d > 0).sum())}/{int((d < 0).sum())}/{int((d == 0).sum())} "
            f"(win/loss/tie), W={W:g}, p={p:.4g}, rank-biserial={rb:+.3f}")


def multinomial_mpr(best_f, land, evals, K, budget):
    """Entry 103's fixed-cost ceiling, on whatever descents it is handed."""
    n = budget / np.mean(evals)
    per = []
    for e in EPS:
        p = np.zeros(K)
        for j in land[best_f <= e]:
            p[j] += 1
        p /= len(best_f)
        per.append(float(np.sum(1.0 - (1.0 - p) ** n)) / K)
    return float(np.mean(per)), per, float(n)


def support_mpr(best_f, land, K, sel=None):
    """|support at eps| / K, averaged over the five accuracies."""
    bf, ld = (best_f, land) if sel is None else (best_f[sel], land[sel])
    return float(np.mean([len(set(ld[bf <= e].tolist())) / K for e in EPS]))


def main() -> None:
    ref = {r["problem"]: r for r in
           csv.DictReader(open(ROOT / "analysis/mmo2024/e106/mceso_vs_null_d10.csv"))}
    nm = {r["problem"]: r for r in
          csv.DictReader(open(ROOT / "analysis/mmo2024/e109/nmmso_d10.csv"))}
    stops: Counter = Counter()
    out = []
    for name in NAMES:
        run_csv = HERE / "by_problem" / f"{name}.csv"
        if not run_csv.exists():
            print(f"!! {name}: no run csv -- skipped")
            continue
        rows = [r for r in csv.DictReader(open(run_csv))
                if r["method"] == "Restart-Lander" and r["rule"] == "current"]
        if not rows:
            print(f"!! {name}: run csv has no scored rows -- skipped")
            continue
        b = niching_by_name(name)
        K = int(b.n_global_optima)
        per_seed = np.array([[float(r[f"pr_{e:.0e}".replace("e-0", "e-")])
                              for e in EPS] for r in rows])
        mpr = float(per_seed.mean(axis=1).mean())
        n_rep = float(np.mean([float(r["n_reported"]) for r in rows]))

        # per-descent dump, seed by seed
        caps, uncaps, mults, nds, costs, boot = [], [], [], [], [], []
        for r in rows:
            seed = int(r["seed"]) * 100
            d = HERE / "descents" / f"{name}_seed{seed}.csv"
            gz = d.with_suffix(".csv.gz")          # dumps are gzipped on commit
            if gz.exists():
                dr = list(csv.DictReader(gzip.open(gz, "rt")))
            elif d.exists():
                dr = list(csv.DictReader(open(d)))
            else:
                continue
            bf = np.array([float(x["best_f"]) for x in dr])
            ld = np.array([int(x["land_opt"]) for x in dr])
            ev = np.array([float(x["evals"]) for x in dr])
            stops.update(x["stop"] for x in dr)
            keep = np.argsort(bf)[:max(100, 2 * K)]
            caps.append(support_mpr(bf, ld, K, keep))
            uncaps.append(support_mpr(bf, ld, K))
            mults.append(multinomial_mpr(bf, ld, ev, K, BUDGET)[0])
            nds.append(len(dr))
            costs.append(ev.mean())
            # One seed this cycle, so the seed-to-seed spread is not measured.
            # A run is however an average over its own 40-150 independent
            # descents, and resampling those (cap re-applied each time) bounds
            # how much of this problem's MPR is which descents happened to be
            # drawn.  It is not a substitute for a second seed -- it holds the
            # descent count fixed -- but it costs zero evaluations and says
            # whether the run-to-run number can be read at all.
            rs = np.random.default_rng(0)
            bs = []
            for _ in range(1000):
                idx = rs.integers(0, len(bf), len(bf))
                bb, lb = bf[idx], ld[idx]
                kk = np.argsort(bb)[:max(100, 2 * K)]
                bs.append(support_mpr(bb, lb, K, kk))
            boot.append((float(np.percentile(bs, 2.5)),
                         float(np.percentile(bs, 97.5))))
        out.append({
            "problem": name, "K": K, "seeds": len(rows),
            "mpr_run": mpr, "n_reported": n_rep,
            "mpr_run_capped_chk": float(np.mean(caps)) if caps else float("nan"),
            "mpr_run_uncapped": float(np.mean(uncaps)) if uncaps else float("nan"),
            "mpr_multinom_on_run": float(np.mean(mults)) if mults else float("nan"),
            "mpr_boot_lo": float(np.mean([b[0] for b in boot])) if boot else float("nan"),
            "mpr_boot_hi": float(np.mean([b[1] for b in boot])) if boot else float("nan"),
            "descents": float(np.mean(nds)) if nds else float("nan"),
            "descent_cost": float(np.mean(costs)) if costs else float("nan"),
            "n_prime_e103": BUDGET / np.mean(costs) if costs else float("nan"),
            "mpr_est_e103": float(ref[name]["null_mpr"]),
            "sup_est_e103": float(ref[name]["null_sup"]),
            "mpr_mceso": float(ref[name]["mpr_mean"]),
            "mpr_nmmso": float(nm[name]["nmmso_mpr"]),
        })

    hdr = list(out[0])
    with open(HERE / "lander_d10.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)

    print(f"\n{'problem':<16}{'K':>3}{'|rep|':>7}{'desc':>6}{'cost':>8}"
          f"{'MPR run':>9}{'95% boot':>16}{'uncap':>8}{'multinom':>10}"
          f"{'e103 est':>10}{'MC-ESO':>8}{'NMMSO':>8}")
    print("-" * 109)
    for r in out:
        ci = f"[{r['mpr_boot_lo']:.3f},{r['mpr_boot_hi']:.3f}]"
        print(f"{r['problem']:<16}{r['K']:>3}{r['n_reported']:>7.0f}"
              f"{r['descents']:>6.0f}{r['descent_cost']:>8.0f}"
              f"{r['mpr_run']:>9.4f}{ci:>16}{r['mpr_run_uncapped']:>8.4f}"
              f"{r['mpr_multinom_on_run']:>10.4f}{r['mpr_est_e103']:>10.4f}"
              f"{r['mpr_mceso']:>8.4f}{r['mpr_nmmso']:>8.4f}")

    col = {k: np.array([r[k] for r in out], float) for k in hdr if k != "problem"}
    n = len(out)
    print(f"\n== {n}-problem means")
    for k, lbl in (("mpr_run", "MPR, run (harness, capped)"),
                   ("mpr_run_capped_chk", "  same from the dump (check)"),
                   ("mpr_run_uncapped", "MPR, run reported set uncapped"),
                   ("mpr_multinom_on_run", "entry 103's formula on this run's landings"),
                   ("mpr_est_e103", "entry 103's estimate (500 offline descents)"),
                   ("sup_est_e103", "entry 103's support bound"),
                   ("mpr_mceso", "MC-ESO (entry 106)"),
                   ("mpr_nmmso", "NMMSO (entry 109)")):
        print(f"   {lbl:<46}{col[k].mean():.4f}")
    print(f"   {'published best RR-CMA-ES':<46}{PUBLISHED:.4f}")
    print(f"\n   descents per run: mean {col['descents'].mean():.1f}, "
          f"range {col['descents'].min():.0f}-{col['descents'].max():.0f}"
          f"   |  n' = budget/mean cost: mean {col['n_prime_e103'].mean():.1f}, "
          f"range {col['n_prime_e103'].min():.1f}-{col['n_prime_e103'].max():.1f}")

    m = col["mpr_run"].mean()
    branch = ("1 (estimate confirmed by a run)" if m >= 0.50 else
              "2 (estimate was overstated)" if m < 0.40 else
              "3 (dead band: separate the sources before closing)")
    print(f"\n== pre-registered branch: mean MPR {m:.4f} -> branch {branch}")

    print("\n== paired tests (unit = problem, n = %d)" % n)
    print("   " + wilcoxon(col["mpr_run"], col["mpr_est_e103"],
                           "run vs entry 103's estimate"))
    print("   " + wilcoxon(col["mpr_run"], col["mpr_mceso"], "run vs MC-ESO   "))
    print("   " + wilcoxon(col["mpr_run"], col["mpr_nmmso"], "run vs NMMSO    "))
    print("   " + wilcoxon(col["mpr_run_uncapped"], col["mpr_run"],
                           "uncapped vs capped (the report cap's price)"))
    print("   " + wilcoxon(col["mpr_multinom_on_run"], col["mpr_run"],
                           "formula-on-run vs run (the formula's price)  "))

    print("\n== what ends a descent (tolfun = tolfunhist = tolx = 0)")
    tot = sum(stops.values())
    for s, c in stops.most_common():
        print(f"   {s:<44}{c:>6}  ({100 * c / tot:5.1f}%)")


if __name__ == "__main__":
    main()
