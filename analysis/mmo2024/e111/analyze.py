"""entry 111 -- the pre-registered second seed of the Restart-Lander null.

Reads entry 110's seed-0 outputs and this entry's seed-1 outputs, keeps them
separable, and answers three things:

  branch   does seed 1 alone hold the 16-problem mean at >= 0.50 (entry 110's
           branch 1), or was branch 1 a one-seed hit
  table    the 2-seed per-problem mean that a paper table would carry, with the
           seed-to-seed gap next to it
  bootstrap entry 110 bootstrapped over a run's own descents and said in its own
           comment that this is "not a substitute for a second seed".  With two
           seeds that is checkable: how often does seed 1 land inside seed 0's
           95% descent-bootstrap interval, and how do the widths compare.

Usage: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e111/analyze.py
"""
import csv
import gzip
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
E110 = HERE.parent / "e110"
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
        return f"{label}: all {len(d)} differences are zero"
    W, p = stats.wilcoxon(d)
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return (f"{label}: mean diff {d.mean():+.4f}, "
            f"{int((d > 0).sum())}/{int((d < 0).sum())}/{int((d == 0).sum())} "
            f"(win/loss/tie), W={W:g}, p={p:.4g}, rank-biserial={rb:+.3f}")


def support_mpr(best_f, land, K, sel=None):
    """|support at eps| / K, averaged over the five accuracies."""
    bf, ld = (best_f, land) if sel is None else (best_f[sel], land[sel])
    return float(np.mean([len(set(ld[bf <= e].tolist())) / K for e in EPS]))


def read_run(base, name):
    """Harness MPR + reported-set size for the single Restart-Lander row."""
    p = base / "by_problem" / f"{name}.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.DictReader(open(p))
            if r["method"] == "Restart-Lander" and r["rule"] == "current"]
    if not rows:
        return None
    r = rows[0]
    per = [float(r[f"pr_{e:.0e}".replace("e-0", "e-")]) for e in EPS]
    return {"mpr": float(np.mean(per)), "per": per,
            "n_rep": float(r["n_reported"]), "seed_idx": int(r["seed"])}


def read_dump(base, name, seed_idx):
    """One row per descent; entry 110 gzips these on commit."""
    d = base / "descents" / f"{name}_seed{seed_idx * 100}.csv"
    gz = d.with_suffix(".csv.gz")
    if gz.exists():
        dr = list(csv.DictReader(gzip.open(gz, "rt")))
    elif d.exists():
        dr = list(csv.DictReader(open(d)))
    else:
        return None
    return {"best_f": np.array([float(x["best_f"]) for x in dr]),
            "land": np.array([int(x["land_opt"]) for x in dr]),
            "evals": np.array([float(x["evals"]) for x in dr]),
            "stop": [x["stop"] for x in dr]}


def boot_ci(bf, ld, K, n=1000, seed=0):
    """Entry 110's descent bootstrap, cap re-applied inside each resample."""
    rs = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        idx = rs.integers(0, len(bf), len(bf))
        bb, lb = bf[idx], ld[idx]
        kk = np.argsort(bb)[:max(100, 2 * K)]
        bs.append(support_mpr(bb, lb, K, kk))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ref = {r["problem"]: r for r in
           csv.DictReader(open(ROOT / "analysis/mmo2024/e106/mceso_vs_null_d10.csv"))}
    nm = {r["problem"]: r for r in
          csv.DictReader(open(ROOT / "analysis/mmo2024/e109/nmmso_d10.csv"))}
    stops: Counter = Counter()
    out = []
    for name in NAMES:
        r0, r1 = read_run(E110, name), read_run(HERE, name)
        if r0 is None or r1 is None:
            print(f"!! {name}: missing seed {'0' if r0 is None else '1'} -- skipped")
            continue
        K = int(niching_by_name(name).n_global_optima)
        d0 = read_dump(E110, name, r0["seed_idx"])
        d1 = read_dump(HERE, name, r1["seed_idx"])
        if d1 is not None:
            stops.update(d1["stop"])
        lo, hi = boot_ci(d0["best_f"], d0["land"], K) if d0 else (float("nan"),) * 2
        out.append({
            "problem": name, "K": K,
            "mpr_seed0": r0["mpr"], "mpr_seed1": r1["mpr"],
            "mpr_2seed": 0.5 * (r0["mpr"] + r1["mpr"]),
            "gap": abs(r0["mpr"] - r1["mpr"]),
            "boot_lo_seed0": lo, "boot_hi_seed0": hi,
            "seed1_in_boot0": int(lo - 1e-12 <= r1["mpr"] <= hi + 1e-12),
            "n_rep_seed0": r0["n_rep"], "n_rep_seed1": r1["n_rep"],
            "descents_seed0": float(len(d0["best_f"])) if d0 else float("nan"),
            "descents_seed1": float(len(d1["best_f"])) if d1 else float("nan"),
            "cost_seed1": float(d1["evals"].mean()) if d1 else float("nan"),
            "mpr_est_e103": float(ref[name]["null_mpr"]),
            "mpr_mceso": float(ref[name]["mpr_mean"]),
            "mpr_nmmso": float(nm[name]["nmmso_mpr"]),
        })

    hdr = list(out[0])
    with open(HERE / "lander_d10_2seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)

    print(f"\n{'problem':<16}{'K':>3}{'seed0':>8}{'seed1':>8}{'2-seed':>8}"
          f"{'|gap|':>7}{'seed0 95% boot':>18}{'in':>4}"
          f"{'desc0':>7}{'desc1':>7}{'e103':>8}{'MC-ESO':>8}{'NMMSO':>8}")
    print("-" * 118)
    for r in out:
        ci = f"[{r['boot_lo_seed0']:.3f},{r['boot_hi_seed0']:.3f}]"
        print(f"{r['problem']:<16}{r['K']:>3}{r['mpr_seed0']:>8.4f}"
              f"{r['mpr_seed1']:>8.4f}{r['mpr_2seed']:>8.4f}{r['gap']:>7.4f}"
              f"{ci:>18}{'y' if r['seed1_in_boot0'] else 'N':>4}"
              f"{r['descents_seed0']:>7.0f}{r['descents_seed1']:>7.0f}"
              f"{r['mpr_est_e103']:>8.4f}{r['mpr_mceso']:>8.4f}{r['mpr_nmmso']:>8.4f}")

    col = {k: np.array([r[k] for r in out], float) for k in hdr if k != "problem"}
    n = len(out)
    print(f"\n== {n}-problem means")
    for k, lbl in (("mpr_seed0", "MPR, seed 0 (entry 110)"),
                   ("mpr_seed1", "MPR, seed 1 (this entry)"),
                   ("mpr_2seed", "MPR, 2-seed mean"),
                   ("mpr_est_e103", "entry 103's estimate (500 offline descents)"),
                   ("mpr_mceso", "MC-ESO (entry 106)"),
                   ("mpr_nmmso", "NMMSO (entry 109)")):
        print(f"   {lbl:<46}{col[k].mean():.4f}")
    print(f"   {'published best RR-CMA-ES':<46}{PUBLISHED:.4f}")
    print(f"\n   descents per run: seed 0 mean {col['descents_seed0'].mean():.1f}"
          f" (range {col['descents_seed0'].min():.0f}-{col['descents_seed0'].max():.0f}), "
          f"seed 1 mean {col['descents_seed1'].mean():.1f}"
          f" (range {col['descents_seed1'].min():.0f}-{col['descents_seed1'].max():.0f})")

    m1 = col["mpr_seed1"].mean()
    print(f"\n== pre-registered branch: seed 1's mean MPR {m1:.4f} -> branch "
          + ("A (entry 110's branch 1 holds on a second seed)" if m1 >= 0.50
             else "B (branch 1 was a one-seed hit -- back to entry 110's branch 3)"))

    print("\n== paired tests (unit = problem, n = %d)" % n)
    print("   " + wilcoxon(col["mpr_seed1"], col["mpr_seed0"],
                           "seed 1 vs seed 0 (is the seed carrying signal?)"))
    print("   " + wilcoxon(col["mpr_seed1"], col["mpr_est_e103"],
                           "seed 1 vs entry 103's estimate               "))
    print("   " + wilcoxon(col["mpr_2seed"], col["mpr_est_e103"],
                           "2-seed mean vs entry 103's estimate          "))
    print("   " + wilcoxon(col["mpr_2seed"], col["mpr_mceso"],
                           "2-seed mean vs MC-ESO                        "))
    print("   " + wilcoxon(col["mpr_2seed"], col["mpr_nmmso"],
                           "2-seed mean vs NMMSO                         "))

    print("\n== is entry 110's descent bootstrap a stand-in for a second seed?")
    cov = int(col["seed1_in_boot0"].sum())
    width = col["boot_hi_seed0"] - col["boot_lo_seed0"]
    print(f"   seed 1 inside seed 0's 95% descent-bootstrap interval: {cov}/{n}"
          f"  (an honest 95% interval would cover about 15/16)")
    print(f"   mean interval width {width.mean():.4f}   "
          f"mean |seed0 - seed1| gap {col['gap'].mean():.4f}   "
          f"max gap {col['gap'].max():.4f} ({out[int(np.argmax(col['gap']))]['problem']})")
    print("   n = 2 seeds cannot estimate a variance; this is a description, "
          "not a test.")

    print("\n== what ends a descent, seed 1 (tolfun = tolfunhist = tolx = 0)")
    tot = sum(stops.values())
    for s, c in stops.most_common():
        print(f"   {s:<44}{c:>6}  ({100 * c / tot:5.1f}%)")


if __name__ == "__main__":
    main()
