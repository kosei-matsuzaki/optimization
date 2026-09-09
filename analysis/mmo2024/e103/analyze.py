"""Does the bare support `S_obs/K` decelerate past 400 draws?
(entry 103, all 16 problems, D=10, sigma0 = 0.1 x span, 400 -> 500(+) draws)

Entry 101 §3 measured the 16-problem mean of `S_obs/K` rising by +0.0499
(100->200), +0.0263 (200->300), +0.0263 (300->400) per 100 draws -- the last two
equal, no deceleration.  While that holds, no draw-independent number may be
quoted as "the class ceiling": only "the coverage observed at N draws".  Queue
item 2 step (2) asks whether the increment finally shrinks.

This script concatenates, per problem, the saved 200-draw dump (entry 99 for
group A, entry 98 for group B), entry 100's draws 200-399, and every entry 103
chunk present, then reads the rarefaction curve at 100-draw spacing.  The four
ceiling estimators are imported from entries 92/93, not reimplemented, so the
whole theme is scored by one code path.

Draw k is a closed form of k alone (start point `default_rng(1_000_000 + k)`,
CMA seed `k + 1`), so chunks concatenate exactly; `check_contiguous` asserts the
draw indices are 0..n-1 with no gap and no repeat, and the identity check
reproduces the n=200 and n=400 tables the record already states.

Usage: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e103/analyze.py
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

PUBLISHED_D10 = 0.651                       # competition best MPR at D=10
NS = (50, 100, 200, 300, 400, 500, 600)     # rarefaction points
KEYS = ("mpr", "mpr_earlystop", "mpr_sup", "mpr_sup_chao1")
PROBS = [f"M{p:02d}-D10-PIN01" for p in range(1, 17)]
OLD_DIRS = (HERE.parent / "e99", HERE.parent / "e98")
# every chunk that extends the saved 200-draw dump, in draw order
CHUNKS = ((HERE.parent / "e100", "draws200to399"),
          (HERE, "draws400to499"),
          (HERE, "draws500to599"),
          (HERE, "draws600to699"))
# entry 101 §3, 16-problem mean of `S_obs/K`, per 100 draws
PRIOR_INC = {200: 0.0499, 300: 0.0263, 400: 0.0263}
NO_DECEL, DECEL = 0.0263, 0.0132            # prereg refutation branches


def _rows(src: Path):
    op = gzip.open if src.suffix == ".gz" else open
    with op(src, "rt") as fh:
        return list(csv.DictReader(fh))


def _find(d: Path, stem: str):
    for suf in (".csv.gz", ".csv"):
        p = d / f"{stem}{suf}"
        if p.exists():
            return p
    return None


def load(name):
    """Draws 0..N-1 of the sigma0 = 0.1 arm: saved dump, then chunks in order."""
    old_p = None
    for d in OLD_DIRS:
        old_p = old_p or _find(d, f"{name}_sig100200")
    if old_p is None:
        raise FileNotFoundError(f"no saved 200-draw dump for {name}")
    rows = _rows(old_p)
    parts = [("saved", len(rows))]
    for d, tag in CHUNKS:
        p = _find(d, f"{name}_sig100_{tag}")
        if p is None:
            break                    # chunks must be contiguous; stop at first gap
        add = _rows(p)
        rows += add
        parts.append((tag, len(add)))
    draw = np.array([int(r["draw"]) for r in rows])
    o = np.argsort(draw)
    rows = [rows[i] for i in o]
    d_ = {
        "draw": draw[o],
        "best_f": np.array([float(r["best_f"]) for r in rows]),
        "land": np.array([int(r["land_opt"]) for r in rows]),
        "evals": np.array([float(r["evals"]) for r in rows]),
    }
    d_["cross"] = ({e: (np.array([float(r[f"ev_{e:g}"]) for r in rows]),
                        np.array([int(r[f"opt_{e:g}"]) for r in rows]))
                    for e in EPS} if f"ev_{EPS[0]:g}" in rows[0] else None)
    return d_, parts


def check_contiguous(name, d):
    """Draw indices must be exactly 0..n-1: no gap, no repeat, no overlap."""
    want = np.arange(len(d["draw"]))
    if not np.array_equal(d["draw"], want):
        bad = int(np.argmax(d["draw"] != want))
        raise SystemExit(
            f"!! {name}: draw indices are not 0..n-1 (first mismatch at "
            f"position {bad}: got {d['draw'][bad]}).  Chunks do not concatenate; "
            f"no number from this cycle is reportable.")


def ceilings_at(d, K, budget, n):
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
    """S_obs/K and the Chao1 correction/K, averaged over the five accuracies."""
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
    idx = np.arange(n)
    return set(int(j) for j in d["land"][idx][d["best_f"][idx] <= 1e-5])


def main():
    per, missing = {}, []
    for nm in PROBS:
        try:
            d, parts = load(nm)
        except FileNotFoundError:
            missing.append(nm)
            continue
        check_contiguous(nm, d)
        b = niching_by_name(nm)
        K, budget = int(b.n_global_optima), int(b.suite_max_evals)
        have = len(d["draw"])
        per[nm] = {"K": K, "budget": budget, "have": have, "d": d, "parts": parts,
                   "cur": {n: ceilings_at(d, K, budget, n)
                           for n in NS if n <= have}}

    print("=" * 78)
    print("entry 103 -- sigma0 = 0.1 x span, new suite D=10, past 400 draws")
    print("=" * 78)
    if missing:
        print(f"!! no saved dump at all for: {', '.join(missing)}")
    depths = sorted({r["have"] for r in per.values()})
    print(f"problems loaded: {len(per)}   draw depths present: {depths}")
    # The homogeneous depth: the deepest n every loaded problem reaches.  Every
    # mean below is reported at this n only -- entry 101 §1 fixed the defect of
    # averaging problems held at different draw counts, and this cycle must not
    # reintroduce it.
    full_n = max([n for n in NS if all(r["have"] >= n for r in per.values())])
    print(f"homogeneous depth (every problem reaches it): n = {full_n}")
    for nm in PROBS:
        if nm in per and per[nm]["have"] != full_n:
            print(f"   deeper than the homogeneous depth: {nm} has "
                  f"{per[nm]['have']} draws (NOT used in any mean)")

    # ── identity checks: the record's existing tables must come back ─────────
    print("\n== identity check (must reproduce the tables already in the record)")
    for n, ref in ((200, "entry 99/98"), (400, "entry 101")):
        if n > full_n:
            continue
        row = {k: np.mean([per[nm]["cur"][n][k] for nm in per
                           if not np.isnan(per[nm]["cur"][n][k])]) for k in KEYS}
        print(f"   n={n:<4} " + "  ".join(f"{k} {row[k]:.4f}" for k in KEYS)
              + f"   [{ref}]")
    print("   entry 101 recorded at n=400: mpr 0.5509  mpr_earlystop 0.6093  "
          "mpr_sup 0.6813  mpr_sup_chao1 0.7276")

    # ── THE TEST: per-100-draw increment of the bare support ────────────────
    print("\n== PRIMARY (prereg): 16-problem mean of `S_obs/K` and its "
          "per-100-draw increment")
    print(f"{'n':>6}{'S_obs/K':>11}{'increment':>12}{'prior (e101)':>15}"
          f"{'optima at f<=1e-5':>20}")
    sobs, corr = {}, {}
    for n in [x for x in NS if x <= full_n]:
        vals = [chao1_parts(per[nm]["d"], per[nm]["K"], n) for nm in per]
        sobs[n] = float(np.mean([v[0] for v in vals]))
        corr[n] = float(np.mean([v[1] for v in vals]))
    incs = {}
    prev = None
    for n in [x for x in NS if x <= full_n]:
        tot = sum(len(distinct_at_1e5(per[nm]["d"], n)) for nm in per)
        inc = ""
        if prev is not None and n - prev == 100:
            incs[n] = sobs[n] - sobs[prev]
            inc = f"{incs[n]:+.4f}"
        elif prev is not None:
            incs[n] = (sobs[n] - sobs[prev]) * 100.0 / (n - prev)
            inc = f"{incs[n]:+.4f}*"
        pr = f"{PRIOR_INC[n]:+.4f}" if n in PRIOR_INC else ""
        print(f"{n:>6}{sobs[n]:>11.4f}{inc:>12}{pr:>15}{tot:>20}")
        prev = n
    print("   (* = normalised to 100 draws from a wider window)")

    tested = [n for n in incs if n > 400]
    for n in tested:
        v = incs[n]
        if v >= NO_DECEL:
            verdict = (f"NO DECELERATION (>= {NO_DECEL}): the support does not "
                       "saturate here; only 'coverage at N draws' is sayable")
        elif v <= DECEL:
            verdict = (f"DECELERATION (<= {DECEL}): a saturation scale exists; "
                       "a level may be quoted with it")
        else:
            verdict = (f"DEAD BAND ({DECEL} < x < {NO_DECEL}): neither branch "
                       "fires; the next chunk is needed")
        print(f"\n   >>> increment {400}->{n}: {v:+.4f}  -> {verdict}")

    # ── firing check (prereg §1) ────────────────────────────────────────────
    if tested:
        n1 = max(tested)
        print(f"\n== firing check: which problems bought optima in draws "
              f"400-{n1 - 1}")
        print(f"{'problem':<16}{'K':>3}{'opt@1e-5 400':>14}{'-> at ' + str(n1):>12}"
              f"{'new-only draws':>16}{'any f<=1e-5':>13}")
        bought = 0
        for nm in PROBS:
            if nm not in per:
                continue
            r = per[nm]
            s4, s5 = distinct_at_1e5(r["d"], 400), distinct_at_1e5(r["d"], n1)
            newonly = len([1 for i in range(400, n1)
                           if r["d"]["best_f"][i] <= 1e-5
                           and int(r["d"]["land"][i]) not in s4])
            ever = int((r["d"]["best_f"][:n1] <= 1e-5).sum())
            bought += len(s5) - len(s4)
            print(f"{nm:<16}{r['K']:>3}{len(s4):>14}{len(s5):>12}"
                  f"{newonly:>16}{ever:>13}")
        tot_K = sum(r["K"] for r in per.values())
        print(f"   optima bought by the {n1 - 400} new draws: {bought} of "
              f"{tot_K} (a problem buying 0 while 'any f<=1e-5' > 0 is a "
              f"saturated support, not a dead arm)")

    # ── the four ceilings along n, at the homogeneous depth only ────────────
    print(f"\n== 16-problem mean of each ceiling along the draw count "
          f"(published best at D=10 = {PUBLISHED_D10})")
    print(f"{'n':>6} " + "".join(f"{k:>17}" for k in KEYS)
          + f"{'chao1 - 0.651':>16}")
    curve = {}
    for n in [x for x in NS if x <= full_n]:
        row = {}
        for k in KEYS:
            vals = [per[nm]["cur"][n][k] for nm in per
                    if not np.isnan(per[nm]["cur"][n][k])]
            row[k] = (float(np.mean(vals)), len(vals))
        curve[n] = row
        print(f"{n:>6} " + "".join(f"{row[k][0]:>11.4f} (n={row[k][1]:>2})"
                                   for k in KEYS)
              + f"{row['mpr_sup_chao1'][0] - PUBLISHED_D10:>+16.4f}")

    # ── Chao1 split, extending entry 101's U-curve by one point ─────────────
    print("\n== `mpr_sup_chao1` split (16-problem mean).  entry 101: the "
          "correction/K is a U in n")
    print(f"{'n':>6}{'S_obs/K':>12}{'correction/K':>15}{'total':>10}")
    for n in [x for x in NS if x <= full_n]:
        print(f"{n:>6}{sobs[n]:>12.4f}{corr[n]:>15.4f}{sobs[n] + corr[n]:>10.4f}")

    # ── per-problem, 400 -> homogeneous depth ───────────────────────────────
    if tested:
        n1 = max(tested)
        print(f"\n== per problem, 400 -> {n1} draws")
        print(f"{'problem':<16}{'K':>3}{'mpr':>18}{'sup':>18}{'chao1':>18}")
        for nm in PROBS:
            if nm not in per:
                continue
            r = per[nm]
            c4, c5 = r["cur"][400], r["cur"][n1]
            print(f"{nm:<16}{r['K']:>3}"
                  f"{c4['mpr']:>8.4f}->{c5['mpr']:<9.4f}"
                  f"{c4['mpr_sup']:>8.4f}->{c5['mpr_sup']:<9.4f}"
                  f"{c4['mpr_sup_chao1']:>8.4f}->{c5['mpr_sup_chao1']:<9.4f}")

    with open(HERE / "support_vs_draws.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_draws", "n_problems", "S_obs_over_K",
                    "increment_per_100", "correction_over_K"] + list(KEYS)
                   + ["chao1_minus_published"])
        for n in [x for x in NS if x <= full_n]:
            w.writerow([n, len(per), f"{sobs[n]:.6f}",
                        f"{incs[n]:+.6f}" if n in incs else "",
                        f"{corr[n]:.6f}"]
                       + [f"{curve[n][k][0]:.6f}" for k in KEYS]
                       + [f"{curve[n]['mpr_sup_chao1'][0] - PUBLISHED_D10:+.6f}"])
        w.writerow([])
        w.writerow(["problem", "K", "draws_held", "opt_1e5_at_400",
                    "opt_1e5_at_depth", "mpr_at_depth", "sup_at_depth",
                    "chao1_at_depth"])
        for nm in PROBS:
            if nm not in per:
                continue
            r = per[nm]
            c = r["cur"][full_n]
            w.writerow([nm, r["K"], r["have"],
                        len(distinct_at_1e5(r["d"], 400)),
                        len(distinct_at_1e5(r["d"], full_n)),
                        f"{c['mpr']:.6f}", f"{c['mpr_sup']:.6f}",
                        f"{c['mpr_sup_chao1']:.6f}"])
    print(f"\nwrote {HERE / 'support_vs_draws.csv'}")


if __name__ == "__main__":
    main()
