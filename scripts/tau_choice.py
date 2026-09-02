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
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d[(r["function"], r["method"])][(int(r["seed"]), int(r["scenario"]))] = \
                float(r["regret"])
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path, nargs="+")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    per_model: dict[str, dict[str, tuple[int, int, int]]] = {}
    taus: list[str] = []
    for path in args.rows:
        d = _load(path)
        model = re.sub(r"^tau_|\.csv$", "", path.name)
        funcs = sorted({k[0] for k in d})
        rules = sorted({k[1] for k in d if k[1].startswith("spread@")},
                       key=lambda m: float(m.split("@")[1]))
        taus = taus or rules
        out = {}
        for m in rules:
            better = worse = mean_w = 0
            for fn in funcs:
                a_d, b_d = d[(fn, m)], d[(fn, BASE)]
                ks = sorted(set(a_d) & set(b_d))
                a = np.array([a_d[k] for k in ks])
                b = np.array([b_d[k] for k in ks])
                mean_w += a.mean() < b.mean()
                diff = a - b
                if wilcoxon is None or not np.any(diff != 0):
                    continue
                p = wilcoxon(a, b).pvalue
                if p < args.alpha:
                    better += a.mean() < b.mean()
                    worse += a.mean() > b.mean()
            out[m] = (better, worse, mean_w)
        per_model[model] = out

    models = list(per_model)
    print("the rule against reporting the K best, per scenario model")
    print("better/worse = functions where paired Wilcoxon is significant at "
          f"alpha={args.alpha}")
    print("(a tolerance usable without knowing the model must not lose functions "
          "under any of them)\n")
    head = f"{'tolerance':<12}"
    for mo in models:
        head += f"{mo:>18}"
    print(head + f"{'worst':>8}")
    print("-" * len(head + f"{'worst':>8}"))
    for m in taus:
        line = f"{m:<12}"
        worst = 10 ** 9
        for mo in models:
            b, w, _ = per_model[mo][m]
            line += f"{f'{b}-{w}':>18}"
            worst = min(worst, b - w)
        print(line + f"{worst:>8}")
    print("\nworst = the smallest (better - worse) across models: the rule's value "
          "when the unknown is chosen adversarially.")


if __name__ == "__main__":
    main()
