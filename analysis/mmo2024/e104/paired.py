"""Does the bare support `S_obs/K` decelerate, tested on a 200-draw window?
(entry 104, all 16 problems, D=10, sigma0 = 0.1 x span, 200 / 400 / 600 draws)

Entry 103 measured the 100-draw window and could not test it: of 240 optima the
100 new draws bought 5, **8 of 16 problems were tied in both windows**, and the
signed-rank test returned p = 0.83 on a mean that had visibly dropped.  The
queue's next step is to double the window and pair the increments per problem:

    a_i = S_i(400) - S_i(200)        b_i = S_i(600) - S_i(400)

Pre-registered branches (`analysis/mmo2024/e104/prereg.md`), on the 16-problem
mean of `S_obs/K`, against the reference Delta(200->400) = +0.0525:

    >= +0.0526  no deceleration      <= +0.0263  deceleration
    in between  DEAD BAND -> the draw-count sweep closes and escalates

The dead band is a stopping rule, not an invitation to add a point: a count
sweep can always afford one more, so the retreat condition sits on the draw
count rather than on the answer (entry 103's lesson).

Everything is read through entry 103's `analyze.py` -- `load`, `check_contiguous`
and `chao1_parts` -- so the theme keeps ONE scoring code path and this script
adds only the pairing and the test.

Usage: PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e104/paired.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from core.benchmarks import niching_by_name          # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_e103 = _load_module("e103_analyze", HERE.parent / "e103" / "analyze.py")
load, check_contiguous = _e103.load, _e103.check_contiguous
chao1_parts, distinct_at_1e5 = _e103.chao1_parts, _e103.distinct_at_1e5
ceilings_at, PROBS, KEYS = _e103.ceilings_at, _e103.PROBS, _e103.KEYS

N_LO, N_MID, N_HI = 200, 400, 600
REF_INC = 0.0525                 # Delta(200->400) of the 16-problem mean
NO_DECEL, DECEL = 0.0526, 0.0263  # pre-registered branches
# tables the record already states; the identity check must reproduce them
IDENTITY = {200: (0.5393, 0.5864, 0.6287, 0.6549),
            400: (0.5509, 0.6093, 0.6813, 0.7276),
            500: (0.5529, 0.6133, 0.7006, 0.7357)}


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
        per[nm] = {"K": K, "budget": budget, "have": len(d["draw"]), "d": d}

    print("=" * 78)
    print("entry 104 -- paired 200-draw windows, new suite D=10, sigma0 = 0.1")
    print("=" * 78)
    if missing:
        print(f"!! no saved dump at all for: {', '.join(missing)}")
    have = {nm: r["have"] for nm, r in per.items()}
    print(f"problems loaded: {len(per)}   draw depths: {sorted(set(have.values()))}")

    short = {nm: n for nm, n in have.items() if n < N_HI}
    if short:
        # whole-set-or-nothing (entry 101 §1): never average across two depths
        print(f"\n!! {len(short)} problem(s) short of {N_HI} draws:")
        for nm, n in sorted(short.items()):
            print(f"     {nm}  {n} draws")
        print("   The 600-draw mean is NOT reported.  Per entry 101 §1 the "
              "16-problem mean must sit at one draw count.")

    # ── identity check: the record's existing tables must come back ─────────
    print("\n== identity check (must reproduce the tables already in the record)")
    ok = True
    for n, ref in IDENTITY.items():
        if any(r["have"] < n for r in per.values()):
            continue
        row = [np.mean([ceilings_at(per[nm]["d"], per[nm]["K"],
                                    per[nm]["budget"], n)[k] for nm in per])
               for k in KEYS]
        good = all(abs(a - b) < 5e-4 for a, b in zip(row, ref))
        ok &= good
        print(f"   n={n:<4} " + "  ".join(f"{k} {v:.4f}" for k, v in zip(KEYS, row))
              + f"   {'OK' if good else '!! MISMATCH, expected ' + str(ref)}")
    if not ok:
        raise SystemExit("!! identity check failed: the concatenation is wrong, "
                         "no number from this cycle is reportable.")

    depth = N_HI if not short else N_MID + 100 * ((min(have.values()) - N_MID) // 100)
    if depth < N_HI:
        print(f"\n== stopping at the homogeneous depth n = {depth}; the paired "
              f"400->600 test needs all 16 problems at 600.")
        return

    # ── per-problem support and the two paired increments ───────────────────
    rows = []
    for nm in PROBS:
        if nm not in per:
            continue
        d, K = per[nm]["d"], per[nm]["K"]
        s = {n: chao1_parts(d, K, n)[0] for n in (N_LO, N_MID, N_HI)}
        o = {n: distinct_at_1e5(d, n) for n in (N_LO, N_MID, N_HI)}
        rows.append({
            "problem": nm, "K": K,
            "s200": s[N_LO], "s400": s[N_MID], "s600": s[N_HI],
            "a": s[N_MID] - s[N_LO], "b": s[N_HI] - s[N_MID],
            "opt400": len(o[N_MID]), "opt600": len(o[N_HI]),
            "new": len(o[N_HI] - o[N_MID]),
            "reached": int((d["best_f"][:N_HI] <= 1e-5).sum()),
        })

    a = np.array([r["a"] for r in rows])
    b = np.array([r["b"] for r in rows])
    m200 = float(np.mean([r["s200"] for r in rows]))
    m400 = float(np.mean([r["s400"] for r in rows]))
    m600 = float(np.mean([r["s600"] for r in rows]))
    inc_a, inc_b = m400 - m200, m600 - m400

    print(f"\n== 16-problem mean of S_obs/K (the pre-registered branch quantity)")
    print(f"   n=200 {m200:.4f}    n=400 {m400:.4f}    n=600 {m600:.4f}")
    print(f"   Delta(200->400) = {inc_a:+.4f}   (record: {REF_INC:+.4f})")
    print(f"   Delta(400->600) = {inc_b:+.4f}   <-- the pre-registered quantity")

    if inc_b >= NO_DECEL:
        branch = (f"NO DECELERATION ({inc_b:+.4f} >= {NO_DECEL:+.4f}): the "
                  f"support does not saturate in this range; only 'the coverage "
                  f"observed at N draws' may be said.")
    elif inc_b <= DECEL:
        branch = (f"DECELERATION ({inc_b:+.4f} <= {DECEL:+.4f}): a saturation "
                  f"scale exists and a level may finally be quoted.")
    else:
        branch = (f"DEAD BAND ({DECEL:+.4f} < {inc_b:+.4f} < {NO_DECEL:+.4f}): "
                  f"neither branch fires.  Per the pre-registered stopping rule "
                  f"the draw-count sweep CLOSES and escalates to the review "
                  f"cycle -- it does not buy another point.")
    print(f"\n   BRANCH: {branch}")

    # ── the paired test: is the drop systematic or a few problems' luck? ────
    diff = a - b
    win = int((diff > 0).sum())     # 400->600 slower than 200->400
    los = int((diff < 0).sum())
    tie = int((diff == 0).sum())
    print(f"\n== paired test, per problem: a_i = S(400)-S(200), b_i = S(600)-S(400)")
    print(f"   slower / faster / tied = {win} / {los} / {tie}   (of {len(rows)})")
    if tie >= 8:
        print(f"   !! ties still dominate ({tie}/16): doubling the window did "
              f"NOT restore resolution; the sweep cannot be tested here.")
    if win + los == 0:
        print("   every problem tied: no test is possible.")
    else:
        w = wilcoxon(a, b)
        # rank-biserial from the two rank sums explicitly: scipy's `statistic` is
        # min(W+, W-), so deriving the effect size from it loses the sign.
        nz = diff[diff != 0]
        r = rankdata(np.abs(nz))
        w_pos, w_neg = r[nz > 0].sum(), r[nz < 0].sum()
        n_eff = len(nz)
        rb = (w_pos - w_neg) / (n_eff * (n_eff + 1) / 2)
        print(f"   W+ (400->600 slower) = {w_pos:.1f}, W- (faster) = {w_neg:.1f}, "
              f"n_nonzero = {n_eff}")
        print(f"   Wilcoxon signed-rank W={w.statistic:.1f}, p={w.pvalue:.4f}, "
              f"rank-biserial={rb:+.3f}, alpha=0.05 two-sided")
        print(f"   medians: a {np.median(a):+.4f}   b {np.median(b):+.4f}")
        print(f"   -> the mean drop is "
              f"{'SYSTEMATIC' if w.pvalue < 0.05 else 'NOT distinguishable from noise'}")

    # ── firing check: did the 200 new draws buy any optimum at all? ─────────
    tot_new = sum(r["new"] for r in rows)
    tot_opt = sum(r["K"] for r in rows)
    zero = [r["problem"] for r in rows if r["new"] == 0]
    print(f"\n== firing check (f <= 1e-5), 400 -> 600 draws")
    print(f"   optima bought by the 200 new draws: {tot_new} of {tot_opt}")
    print(f"   problems adding none: {len(zero)}/{len(rows)}")
    print(f"   (all of these still reach f <= 1e-5, i.e. support saturation, "
          f"not a dead measurement -- 'reached' column below)")

    print(f"\n{'problem':<16}{'K':>3}{'S200':>8}{'S400':>8}{'S600':>8}"
          f"{'a':>9}{'b':>9}{'opt400':>8}{'opt600':>8}{'new':>5}{'reached':>9}")
    for r in rows:
        print(f"{r['problem']:<16}{r['K']:>3}{r['s200']:>8.3f}{r['s400']:>8.3f}"
              f"{r['s600']:>8.3f}{r['a']:>+9.4f}{r['b']:>+9.4f}"
              f"{r['opt400']:>8}{r['opt600']:>8}{r['new']:>5}{r['reached']:>9}")

    # ── instruments only: the four ceilings and the Chao1 correction ────────
    print(f"\n== instruments (NOT the quantity under test).  published best = 0.651")
    print(f"{'n':>6}" + "".join(f"{k:>18}" for k in KEYS) + f"{'corr/K':>10}")
    for n in (200, 300, 400, 500, 600):
        if any(r["have"] < n for r in per.values()):
            continue
        row = {k: np.mean([ceilings_at(per[nm]["d"], per[nm]["K"],
                                       per[nm]["budget"], n)[k] for nm in per])
               for k in KEYS}
        corr = np.mean([chao1_parts(per[nm]["d"], per[nm]["K"], n)[1] for nm in per])
        print(f"{n:>6}" + "".join(f"{row[k]:>18.4f}" for k in KEYS) + f"{corr:>10.4f}")

    out = HERE / "paired_support.csv"
    with out.open("w") as fh:
        fh.write("problem,K,s200,s400,s600,inc_200_400,inc_400_600,"
                 "opt_1e5_at_400,opt_1e5_at_600,new_optima,draws_reaching_1e5\n")
        for r in rows:
            fh.write(f"{r['problem']},{r['K']},{r['s200']:.6f},{r['s400']:.6f},"
                     f"{r['s600']:.6f},{r['a']:.6f},{r['b']:.6f},{r['opt400']},"
                     f"{r['opt600']},{r['new']},{r['reached']}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
