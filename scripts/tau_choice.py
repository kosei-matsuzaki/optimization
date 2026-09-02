#!/usr/bin/env python3
"""Is there a tolerance that is safe without knowing which unknown you face?

Reporting the K furthest-apart solutions within a tolerance of the best lowers
expected regret where the unknown deletes solutions, and costs quality where it
merely rescores them. The tolerance that is best therefore reverses with the
scenario model -- and which model you face is a property of the unknown, not of
the landscape, so no amount of looking at the problem reveals it.

That leaves one usable question: does some tolerance avoid harm under every
model at once? This lays the same tolerances side by side across the scenario
files and reports, per model, how many functions the rule wins and loses, and
how far the worst model is below break-even.

Refuted -- meaning the rule cannot be used blind -- if every tolerance that
helps anywhere loses functions somewhere.

Usage:
  python3 scripts/tau_choice.py analysis/tau_constraint.csv analysis/tau_tilt.csv ...
"""
from __future__ import annotations
import argparse
import collections
import csv
import re
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:                                   # pragma: no cover
    wilcoxon = None

BASE = "quality"


def _load(path: Path):
    d: dict[tuple[str, str], dict[tuple[int, int], float]] = collections.defaultdict(dict)
    # Width the adaptive rule chose, one value per (function, method, seed).
    eff: dict[str, dict[tuple[str, int], float]] = collections.defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d[(r["function"], r["method"])][(int(r["seed"]), int(r["scenario"]))] = \
                float(r["regret"])
            if r.get("tau_eff"):
                eff[r["method"]][(r["function"], int(r["seed"]))] = float(r["tau_eff"])
    return d, eff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path, nargs="+")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    per_model: dict[str, dict[str, tuple[int, int, int]]] = {}
    per_model_eff: dict[str, dict[str, dict]] = {}
    taus: list[str] = []
    for path in args.rows:
        d, eff = _load(path)
        per_model_eff[re.sub(r"^tau_|\.csv$", "", path.name)] = eff
        model = re.sub(r"^tau_|\.csv$", "", path.name)
        funcs = sorted({k[0] for k in d})
        # 'spread@T' is the fixed-width rule; 'spreadN@N' and 'spreadNc{cap}@N'
        # are the adaptive ones, which set the number of candidates in the band
        # instead of its width. Group by family, order by the level inside it.
        rules = sorted({k[1] for k in d if re.match(r"^spread\w*@", k[1])},
                       key=lambda m: (m.split("@")[0], float(m.split("@")[1])))
        taus = taus or rules
        out = {}
        for m in rules:
            better = worse = mean_w = 0
            misfire = cells = 0
            for fn in funcs:
                a_d, b_d = d[(fn, m)], d[(fn, BASE)]
                ks = sorted(set(a_d) & set(b_d))
                a = np.array([a_d[k] for k in ks])
                b = np.array([b_d[k] for k in ks])
                # A (function, seed) cell where the tolerance band holds no more
                # than K candidates: the rule falls back to the top K, so it is
                # the base rule and the comparison is a tie by construction, not
                # a finding. Counting these separates "the rule does not help"
                # from "the rule never ran".
                for sd in sorted({k[0] for k in ks}):
                    kk = [k for k in ks if k[0] == sd]
                    cells += 1
                    misfire += all(a_d[k] == b_d[k] for k in kk)
                mean_w += a.mean() < b.mean()
                diff = a - b
                if wilcoxon is None or not np.any(diff != 0):
                    continue
                p = wilcoxon(a, b).pvalue
                if p < args.alpha:
                    better += a.mean() < b.mean()
                    worse += a.mean() > b.mean()
            out[m] = (better, worse, mean_w, misfire, cells)
        per_model[model] = out

    models = list(per_model)
    print("the rule against reporting the K best, per scenario model")
    print("better/worse = functions where paired Wilcoxon is significant at "
          f"alpha={args.alpha}")
    print("(a tolerance usable without knowing the model must not lose functions "
          "under any of them)\n")
    head = f"{'tolerance':<18}"
    for mo in models:
        head += f"{mo:>18}"
    print(head + f"{'worst':>8}")
    print("-" * len(head + f"{'worst':>8}"))
    for m in taus:
        line = f"{m:<18}"
        worst = 10 ** 9
        for mo in models:
            b, w, _, _, _ = per_model[mo][m]
            line += f"{f'{b}-{w}':>18}"
            worst = min(worst, b - w)
        print(line + f"{worst:>8}")
    print("\nworst = the smallest (better - worse) across models: the rule's value "
          "when the unknown is chosen adversarially.")

    print("\ncells where the band held no more than K candidates, so the rule "
          "returned the top K and the comparison is a tie by construction:")
    head = f"{'tolerance':<18}"
    for mo in models:
        head += f"{mo:>18}"
    print(head)
    for m in taus:
        line = f"{m:<18}"
        for mo in models:
            _, _, _, mf, cl = per_model[mo][m]
            line += f"{f'{mf}/{cl}':>18}"
        print(line)
    print("a tolerance that misfires on most cells is not being tested; give the "
          "pool more starts (--pool-starts) before reading its row above.")

    adaptive = [m for m in taus if not m.startswith("spread@")
                and any(per_model_eff[mo].get(m) for mo in models)]
    if adaptive:
        print("\nwidth the adaptive rule actually chose (median over function x "
              "seed, and the 10th/90th percentile):")
        print(f"{'rule':<18}{'median':>10}{'p10':>10}{'p90':>10}{'inf':>8}")
        for m in adaptive:
            vals = [v for mo in models for v in per_model_eff[mo].get(m, {}).values()]
            fin = np.array([v for v in vals if np.isfinite(v)])
            n_inf = len(vals) - len(fin)
            if len(fin) == 0:
                print(f"{m:<18}{'-':>10}{'-':>10}{'-':>10}{n_inf:>8}")
                continue
            print(f"{m:<18}{np.median(fin):>10.3f}{np.percentile(fin, 10):>10.3f}"
                  f"{np.percentile(fin, 90):>10.3f}{n_inf:>8}")
        print("'inf' counts cells where the pool held fewer candidates than the "
              "rule asks for, so the band became the whole pool.")


if __name__ == "__main__":
    main()
