#!/usr/bin/env python3
"""Entry 117: score the composite adoption arm against base, paired by seed.

Reads the shards written by the `diagnose_niching.py --variant {base,comp4}`
runs of this cycle and reports, per function and accuracy level:

  PR      reported peak ratio (`reported / K`) -- the CEC2013 official measure,
          scored off the reported set only (core.runner.niching_peak_metrics).
  paired  win / tie / loss over the shared seeds, Wilcoxon signed-rank
          (method="exact", per entry 111's lesson about the normal-approximation
          branch) and the rank-biserial effect size.

The judgement level is eps <= 1e-3 (entry 28's rule); 1e-5 is the primary.

Usage:
  python3 analysis/hm/e117/analyze.py --func N06-Shubert2D
"""
from __future__ import annotations
import argparse
import csv
import glob
from collections import defaultdict

from scipy.stats import wilcoxon


def load(pattern: str) -> dict:
    """(function, eps, seed) -> row."""
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for r in csv.DictReader(f):
                out[(r["function"], r["eps"], int(r["seed"]))] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="n06")
    ap.add_argument("--base", default="base")
    ap.add_argument("--arm", default="comp4")
    args = ap.parse_args()

    # Shards (…_s0.csv) before the merge, one merged file after it. The
    # patterns are anchored on purpose: a bare `{arm}*` also matches
    # `comp4off.csv` (the identity check), which silently overwrites the arm's
    # own rows for seeds 0-3 and moves the numbers.
    A = load(f"analysis/hm/e117/{args.tag}_{args.base}.csv") | \
        load(f"analysis/hm/e117/{args.tag}_{args.base}_s*.csv")
    B = load(f"analysis/hm/e117/{args.tag}_{args.arm}.csv") | \
        load(f"analysis/hm/e117/{args.tag}_{args.arm}_s*.csv")
    keys = sorted(set(A) & set(B), key=lambda k: (k[0], k[1], k[2]))
    if not keys:
        raise SystemExit("no paired cells")

    groups = defaultdict(list)
    for k in keys:
        groups[(k[0], k[1])].append(k)

    for (fn, eps), ks in sorted(groups.items()):
        K = float(A[ks[0]]["K"])
        pa = [float(A[k]["reported"]) / K for k in ks]
        pb = [float(B[k]["reported"]) / K for k in ks]
        win = sum(1 for x, y in zip(pa, pb) if y > x)
        loss = sum(1 for x, y in zip(pa, pb) if y < x)
        tie = len(ks) - win - loss
        d = [y - x for x, y in zip(pa, pb)]
        if any(v != 0 for v in d):
            try:
                p = wilcoxon(pb, pa, method="exact").pvalue
            except (ValueError, TypeError):
                p = wilcoxon(pb, pa).pvalue
            npos = sum(1 for v in d if v > 0)
            nneg = sum(1 for v in d if v < 0)
            rb = (npos - nneg) / (npos + nneg)
        else:
            p, rb = 1.0, 0.0
        # The identity columns: same trajectory means these must match exactly.
        ident = all(A[k]["best_f"] == B[k]["best_f"] for k in ks)
        hident = all(A[k]["hunts"] == B[k]["hunts"] for k in ks)
        print(f"{fn}  eps={eps}  n={len(ks)}  K={K:.0f}")
        print(f"   PR   base {sum(pa)/len(pa):.4f}   {args.arm} {sum(pb)/len(pb):.4f}   "
              f"delta {sum(pb)/len(pb) - sum(pa)/len(pa):+.4f}")
        print(f"   paired {win}/{tie}/{loss} (win/tie/loss)  p={p:.4g}  rank-biserial {rb:+.3f}")
        print(f"   visited base {sum(float(A[k]['visited']) for k in ks)/len(ks):.2f}  "
              f"{args.arm} {sum(float(B[k]['visited']) for k in ks)/len(ks):.2f}   "
              f"|rep| base {sum(float(A[k]['n_reported_pts']) for k in ks)/len(ks):.1f}  "
              f"{args.arm} {sum(float(B[k]['n_reported_pts']) for k in ks)/len(ks):.1f}")
        print(f"   best_f identical to base: {ident}   hunts identical: {hident}")


if __name__ == "__main__":
    main()
