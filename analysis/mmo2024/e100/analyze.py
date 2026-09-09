"""Is entry 99's excess over the published best a function of the draw count?
(entry 100, both groups, D=10, sigma0 = 0.1 x span, 200 -> 400 draws)

Entry 99 put the 16-problem mean of the most permissive ceiling
(`mpr_sup_chao1`) at 0.6549 against the published best at D=10 (RR-CMA-ES, MPR
0.651): an excess of +0.0039, the first time route (C)'s pre-registered
refutation condition fired on coverage that was really landed on.  But
`mpr_sup = S_obs / K` is a richness statistic and richness is monotone
non-decreasing in the number of draws, and entry 94 recorded the bare support
still climbing at 200 draws.  So 0.6549 may be the coverage *of 200 draws*.

This script extends every sigma0 = 0.1 dump to 400 draws by concatenating the
saved 200-draw dumps (entry 99 for group A, entry 98 for group B) with draws
200-399 computed this cycle (`--descent-start`, identity-checked in prereg.md),
and reads the four ceilings as 16-problem means along n = 50 .. 400.

The estimators are imported from entries 92/93, not reimplemented, so every
cycle of this theme is scored by one code path.

Usage: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e100/analyze.py
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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_e92 = _load_module("e92_analyze", HERE.parent / "e92" / "analyze.py")
_e93 = _load_module("e93_analyze", HERE.parent / "e93" / "analyze.py")
ceiling, early_stop_ceiling, EPS = _e92.ceiling, _e92.early_stop_ceiling, _e92.EPS
chao1_sup = _e93.chao1_sup

PUBLISHED_D10 = 0.651            # competition best MPR at D=10
NS = (50, 100, 200, 300, 400)    # rarefaction points
KEYS = ("mpr", "mpr_earlystop", "mpr_sup", "mpr_sup_chao1")
PROBS = [f"M{p:02d}-D10-PIN01" for p in range(1, 17)]
GROUP_A = set(PROBS[:8])         # K=20, entry 99's 200-draw dumps
# where the saved 200-draw sigma0=0.1 dumps live
OLD_DIRS = (HERE.parent / "e99", HERE.parent / "e98")


def _rows(src: Path):
    op = gzip.open if src.suffix == ".gz" else open
    with op(src, "rt") as fh:
        return list(csv.DictReader(fh))


def _find_old(name):
    for d in OLD_DIRS:
        for suf in (".csv.gz", ".csv"):
            p = d / f"{name}_sig100200{suf}"
            if p.exists():
                return p
    return None


def _find_new(name):
    for suf in (".csv.gz", ".csv"):
        p = HERE / f"{name}_sig100_draws200to399{suf}"
        if p.exists():
            return p
    return None


def load(name):
    """Draws 0..N-1 of the sigma0 = 0.1 arm, saved dump first then this cycle's.

    Returns (arrays, n_old, n_new).  Rows are sorted by `draw` because `--null`
    collects them unordered, so the prefix of the returned arrays at any n is
    exactly draws 0..n-1 -- which is what makes the rarefaction curve a
    rarefaction curve and not a resampling.
    """
    old_p, new_p = _find_old(name), _find_new(name)
    if old_p is None:
        raise FileNotFoundError(f"no saved 200-draw dump for {name}")
    rows = _rows(old_p)
    n_old = len(rows)
    n_new = 0
    if new_p is not None:
        add = _rows(new_p)
        n_new = len(add)
        rows = rows + add
    draw = np.array([int(r["draw"]) for r in rows])
    o = np.argsort(draw)
    rows = [rows[i] for i in o]
    d = {
        "draw": draw[o],
        "best_f": np.array([float(r["best_f"]) for r in rows]),
        "land": np.array([int(r["land_opt"]) for r in rows]),
        "evals": np.array([float(r["evals"]) for r in rows]),
    }
    d["cross"] = ({e: (np.array([float(r[f"ev_{e:g}"]) for r in rows]),
                       np.array([int(r[f"opt_{e:g}"]) for r in rows]))
                   for e in EPS} if f"ev_{EPS[0]:g}" in rows[0] else None)
    return d, n_old, n_new


def ceilings_at(d, K, budget, n):
    """The four ceilings on the first n draws.  nan where the dump cannot say."""
    if n > len(d["draw"]):
        return {k: float("nan") for k in KEYS}
    idx = np.arange(n)
    mpr, _, sup = ceiling(d["best_f"], d["land"], d["evals"], K, budget, idx=idx)
    out = {"mpr": mpr, "mpr_sup": sup,
           "mpr_sup_chao1": chao1_sup(d["best_f"][idx], d["land"][idx], K)}
    if d["cross"] is None:
        out["mpr_earlystop"] = float("nan")
    else:
        cr = {e: (d["cross"][e][0][idx], d["cross"][e][1][idx]) for e in EPS}
        out["mpr_earlystop"] = early_stop_ceiling(cr, d["evals"][idx], K, budget)[0]
    return out


def chao1_parts(d, K, n):
    """S_obs/K and the Chao1 correction/K, averaged over the five accuracies.

    Entry 99 split the ceiling this way to show that its rise was real coverage
    rather than the correction term; the same split says here whether more
    draws buy optima or buy correction.
    """
    idx = np.arange(n)
    bf, ld = d["best_f"][idx], d["land"][idx]
    s_parts, c_parts = [], []
    for eps in EPS:
        counts = np.bincount(ld[bf <= eps], minlength=K)
        s_obs = int((counts > 0).sum())
        f1, f2 = int((counts == 1).sum()), int((counts == 2).sum())
        corr = f1 * (f1 - 1) / (2 * (f2 + 1))
        s_parts.append(s_obs / K)
        c_parts.append((min(s_obs + corr, K) - s_obs) / K)
    return float(np.mean(s_parts)), float(np.mean(c_parts))


def distinct_at_1e5(d, n):
    """Optima reached at f <= 1e-5 within the first n draws (firing evidence)."""
    idx = np.arange(n)
    return set(int(j) for j in d["land"][idx][d["best_f"][idx] <= 1e-5])


def main():
    per, missing, partial = {}, [], []
    for nm in PROBS:
        try:
            d, n_old, n_new = load(nm)
        except FileNotFoundError:
            missing.append(nm)
            continue
        b = niching_by_name(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        have = len(d["draw"])
        if have < 400:
            partial.append((nm, have))
        per[nm] = {"K": K, "budget": budget, "have": have, "d": d,
                   "n_old": n_old, "n_new": n_new,
                   "cur": {n: ceilings_at(d, K, budget, n) for n in NS if n <= have}}

    print("=" * 78)
    print("entry 100 -- sigma0 = 0.1 x span, new suite D=10, 200 -> 400 draws")
    print("=" * 78)
    if missing:
        print(f"!! no saved dump at all for: {', '.join(missing)}")
    if partial:
        print("!! fewer than 400 draws (held at their deepest available n in the "
              "16-problem mean, the conservative direction -- prereg §Scale-down):")
        for nm, have in partial:
            print(f"     {nm}: {have} draws")

    # ── identity check 3: the 400-draw prefix must reproduce entry 99/98 ──────
    print("\n== identity check: ceilings on the first 200 draws of the "
          "concatenated dump")
    print("   (must reproduce the 200-draw table these dumps were scored as)")
    m200 = {k: [] for k in KEYS}
    for nm, r in per.items():
        for k in KEYS:
            v = r["cur"].get(200, {}).get(k, float("nan"))
            if not np.isnan(v):
                m200[k].append(v)
    print("   16-problem mean at n=200: " + "  ".join(
        f"{k} {np.mean(m200[k]):.4f} (n={len(m200[k])})" for k in KEYS))

    # ── per-problem: 200 vs 400 ──────────────────────────────────────────────
    print("\n== per problem, sigma0 = 0.1: 200 -> 400 draws")
    hdr = (f"{'problem':<16}{'K':>3} {'mpr 200->400':>22} "
           f"{'sup 200->400':>20} {'chao1 200->400':>20} "
           f"{'opt@1e-5 200->400':>19} {'new-only draws':>15}")
    print(hdr)
    for nm in PROBS:
        if nm not in per:
            continue
        r = per[nm]
        c2, c4 = r["cur"].get(200), r["cur"].get(400)
        if c4 is None:
            print(f"{nm:<16}{r['K']:>3}  (only {r['have']} draws)")
            continue
        s2, s4 = distinct_at_1e5(r["d"], 200), distinct_at_1e5(r["d"], 400)
        newonly = len([1 for i in range(200, r["have"])
                       if r["d"]["best_f"][i] <= 1e-5
                       and int(r["d"]["land"][i]) not in s2])
        print(f"{nm:<16}{r['K']:>3} "
              f"{c2['mpr']:>9.4f}->{c4['mpr']:<7.4f}{c4['mpr']-c2['mpr']:>+6.4f} "
              f"{c2['mpr_sup']:>8.4f}->{c4['mpr_sup']:<7.4f}{c4['mpr_sup']-c2['mpr_sup']:>+5.4f} "
              f"{c2['mpr_sup_chao1']:>8.4f}->{c4['mpr_sup_chao1']:<7.4f}"
              f"{c4['mpr_sup_chao1']-c2['mpr_sup_chao1']:>+5.4f} "
              f"{len(s2):>8d}->{len(s4):<8d} {newonly:>13d}")

    # ── the headline: 16-problem means along n ───────────────────────────────
    print("\n== 16-problem mean of each ceiling along the draw count "
          f"(published best at D=10 = {PUBLISHED_D10})")
    print(f"{'n':>5} " + "".join(f"{k:>17}" for k in KEYS)
          + f"{'chao1 - 0.651':>16}")
    curve = {}
    for n in NS:
        row = {}
        for k in KEYS:
            vals = []
            for nm, r in per.items():
                # a problem short of n is held at its deepest available value
                nn = n if n <= r["have"] else max(x for x in NS if x <= r["have"])
                v = r["cur"][nn][k]
                if not np.isnan(v):
                    vals.append(v)
            row[k] = (float(np.mean(vals)), len(vals))
        curve[n] = row
        print(f"{n:>5} " + "".join(f"{row[k][0]:>11.4f} (n={row[k][1]:>2})"
                                   for k in KEYS)
              + f"{row['mpr_sup_chao1'][0] - PUBLISHED_D10:>+16.4f}")

    # ── Chao1 split: does a draw buy an optimum or buy correction? ───────────
    print("\n== `mpr_sup_chao1` split into landed coverage and Chao1 correction "
          "(16-problem mean)")
    print(f"{'n':>5}{'S_obs/K':>12}{'correction/K':>15}{'total':>10}")
    for n in NS:
        so, co = [], []
        for nm, r in per.items():
            nn = n if n <= r["have"] else max(x for x in NS if x <= r["have"])
            a, c = chao1_parts(r["d"], r["K"], nn)
            so.append(a)
            co.append(c)
        print(f"{n:>5}{np.mean(so):>12.4f}{np.mean(co):>15.4f}"
              f"{np.mean(so) + np.mean(co):>10.4f}")

    # ── firing check (prereg §3) ─────────────────────────────────────────────
    tot_new = sum(len(distinct_at_1e5(r["d"], r["have"])
                      - distinct_at_1e5(r["d"], 200)) for r in per.values())
    tot_K = sum(r["K"] for r in per.values())
    obs200 = sum(len(distinct_at_1e5(r["d"], 200)) for r in per.values())
    obs400 = sum(len(distinct_at_1e5(r["d"], r["have"])) for r in per.values())
    print(f"\n== firing check: distinct optima at f <= 1e-5 over all problems: "
          f"{obs200} -> {obs400} of {tot_K} "
          f"({tot_new} first reached in draws 200-399)")

    with open(HERE / "ceiling_vs_draws.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_draws", "n_problems_scored"] + list(KEYS)
                   + ["chao1_minus_published"])
        for n in NS:
            w.writerow([n, curve[n]["mpr"][1]]
                       + [f"{curve[n][k][0]:.6f}" for k in KEYS]
                       + [f"{curve[n]['mpr_sup_chao1'][0] - PUBLISHED_D10:+.6f}"])
        w.writerow([])
        w.writerow(["problem", "K", "draws", "ceiling", "at200", "at400", "delta"])
        for nm in PROBS:
            if nm not in per or 400 not in per[nm]["cur"]:
                continue
            r = per[nm]
            for k in KEYS:
                a, b_ = r["cur"][200][k], r["cur"][400][k]
                w.writerow([nm, r["K"], r["have"], k,
                            f"{a:.6f}", f"{b_:.6f}", f"{b_ - a:+.6f}"])
    print(f"\nwritten: {HERE / 'ceiling_vs_draws.csv'}")


if __name__ == "__main__":
    main()
