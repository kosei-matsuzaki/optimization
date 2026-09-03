#!/usr/bin/env python3
"""Winning most scenarios and lowering expected regret are different verdicts.

The audit ranked methods by A12: how often a method's reported set beats
reporting the K best local minima. The seed-agreement column next to it asks
something else entirely -- whether the method's *mean* regret is lower. Where
the two disagree, the method wins often and loses heavily, and reporting only
the first number overstates what a diverse set buys.

This prints both verdicts per method and the size of the losing tail. Excess
regret is given in units of that function's own baseline 75th-percentile
regret. The far tail of that quantity is not interpretable -- the baseline is
near-optimal on many scenarios, so the denominator approaches zero and the
ratio is unbounded for numerical rather than substantive reasons -- so the tail
is summarised by how often the method loses by more than one such unit, not by
how large the worst case gets.

Usage:
  python3 scripts/verdict_split.py analysis/model_constraint15.csv
"""
from __future__ import annotations
import argparse
import collections
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path)
    ap.add_argument("--baseline", default="quality")
    args = ap.parse_args()

    d: dict[tuple[str, str], dict[tuple[int, int], float]] = collections.defaultdict(dict)
    with open(args.rows, newline="") as fh:
        for r in csv.DictReader(fh):
            d[(r["function"], r["method"])][(int(r["seed"]), int(r["scenario"]))] = \
                float(r["regret"])

    methods = sorted({k[1] for k in d} - {args.baseline})
    funcs = sorted({k[0] for k in d})

    scale = {}
    for fn in funcs:
        b = np.array(list(d[(fn, args.baseline)].values()))
        s = float(np.percentile(b, 75))
        scale[fn] = s if s > 0 else (float(b.max()) or 1.0)

    print(f"paired against '{args.baseline}'   functions={len(funcs)}")
    print("rank-win = functions where A12 > 0.5 (wins more scenarios)")
    print("mean-win = functions where mean regret is lower (cheaper on average)")
    print("loses>1  = share of scenarios losing by more than one baseline "
          "75th-pct regret\n")
    print(f"{'method':<14}{'rank-win':>10}{'mean-win':>10}{'disagree':>10}"
          f"{'med excess':>12}{'loses>1':>9}")
    print("-" * 65)

    dis_tot = cells = 0
    for m in methods:
        rank_w = mean_w = dis = 0
        exc = []
        for fn in funcs:
            a_d, b_d = d.get((fn, m), {}), d.get((fn, args.baseline), {})
            ks = sorted(set(a_d) & set(b_d))
            if len(ks) < 10:
                continue
            a = np.array([a_d[k] for k in ks])
            b = np.array([b_d[k] for k in ks])
            rw = ((np.sum(a < b) + 0.5 * np.sum(a == b)) / len(ks)) > 0.5
            mw = a.mean() < b.mean()
            rank_w += rw
            mean_w += mw
            dis += rw != mw
            cells += 1
            exc.append((a - b) / scale[fn])
        dis_tot += dis
        e = np.concatenate(exc)
        print(f"{m:<14}{rank_w:>7}/{len(funcs):<2}{mean_w:>7}/{len(funcs):<2}"
              f"{dis:>10}{np.median(e):>12.2f}{100 * np.mean(e > 1):>8.0f}%")

    print(f"\nthe two verdicts disagree on {dis_tot} of {cells} cells "
          f"({100 * dis_tot / cells:.0f}%)")
    print("A method that is rank-win but not mean-win reports a better set than")
    print("the baseline on most scenarios and a far worse one on the rest.")


if __name__ == "__main__":
    main()
