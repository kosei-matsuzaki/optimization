#!/usr/bin/env python3
"""Why does the count of distinct solutions matter under instance shifts only?

Reported-set properties predict performance differently under each scenario
model: the number of distinct solutions decides it under instance shifts
(rho 0.83) and is worthless under a forbidden region (0.15), where spatial
spread takes over instead. One explanation is the shape of the acceptance set.
Where it breaks into many separate components, having a representative in each
is what pays, and that is a count. Where it is one blob, a forbidden region eats
into it continuously and what pays is distance, not count.

If that is right, the gap between the two models should shrink once functions
are split by how many components their acceptance set has. Refuted if the gap
is the same inside each stratum as it is overall.

Usage:
  python3 scripts/stratify_components.py [--quantile 0.1]
"""
from __future__ import annotations
import argparse
import collections
import csv
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:                                   # pragma: no cover
    spearmanr = None

_MODELS = {
    "tilt": "analysis/audit_t1.0.csv",
    "instance": "analysis/model_instance.csv",
    "constraint": "analysis/model_constraint.csv",
}
BASE = "quality"


def _a12(path: Path) -> dict[tuple[str, str], float]:
    d: dict[tuple[str, str], dict] = collections.defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d[(r["function"], r["method"])][(int(r["seed"]), int(r["scenario"]))] = \
                float(r["regret"])
    out = {}
    for (fn, m), v in d.items():
        if m == BASE:
            continue
        b = d.get((fn, BASE), {})
        ks = sorted(set(b) & set(v))
        if len(ks) < 10:
            continue
        a = np.array([v[k] for k in ks])
        q = np.array([b[k] for k in ks])
        out[(fn, m)] = float((np.sum(a < q) + 0.5 * np.sum(a == q)) / len(ks))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantile", default="0.1")
    ap.add_argument("--props", type=Path, default=Path("analysis/set_properties.csv"))
    ap.add_argument("--components", type=Path,
                    default=Path("analysis/components_bbob.csv"))
    args = ap.parse_args()

    k_big = {}
    with open(args.components, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["quantile"] == args.quantile:
                k_big[r["function"]] = int(r["k_big"])

    acc: dict[tuple[str, str], list] = collections.defaultdict(list)
    with open(args.props, newline="") as fh:
        for r in csv.DictReader(fh):
            acc[(r["function"], r["method"])].append(
                [float(r["spread"]), float(r["distinct"])])
    prop = {k: np.mean(np.array(v), axis=0) for k, v in acc.items()}

    strata = {
        "single (k=1)": lambda k: k <= 1,
        "few (2-4)": lambda k: 2 <= k <= 4,
        "many (5+)": lambda k: k >= 5,
    }
    print(f"acceptance-set components at quantile {args.quantile}; "
          "rho taken within each function, then averaged")
    print("(functions with k_big = 0 have no acceptance set at this quantile "
          "and are dropped)\n")
    print(f"{'model':<12}{'stratum':<15}{'fn':>4}{'rho(spread)':>13}"
          f"{'rho(distinct)':>15}{'gap':>8}")
    print("-" * 67)

    for model, path in _MODELS.items():
        p = Path(path)
        if not p.exists() or spearmanr is None:
            continue
        a12 = _a12(p)
        for label, test in strata.items():
            by_fn: dict[str, list] = collections.defaultdict(list)
            for k in a12:
                if k in prop and test(k_big.get(k[0], 0)) and k_big.get(k[0], 0) > 0:
                    by_fn[k[0]].append(k)
            rows = []
            for j in (0, 1):
                rs = []
                for fn, ks in by_fn.items():
                    if len(ks) < 5:
                        continue
                    x = np.array([prop[k][j] for k in ks])
                    y = np.array([a12[k] for k in ks])
                    if np.ptp(x) == 0 or np.ptp(y) == 0:
                        continue
                    r = spearmanr(x, y).statistic
                    if not np.isnan(r):
                        rs.append(float(r))
                rows.append(float(np.mean(rs)) if rs else float("nan"))
            n = len({fn for fn in by_fn if len(by_fn[fn]) >= 5})
            gap = rows[1] - rows[0]
            print(f"{model:<12}{label:<15}{n:>4}{rows[0]:>13.2f}{rows[1]:>15.2f}"
                  f"{gap:>8.2f}")
        print()

    print("gap = rho(distinct) - rho(spread). The claim is that it should be")
    print("positive where the acceptance set is fragmented and negative where it")
    print("is one blob, in every model -- not a fixed property of the model.")


if __name__ == "__main__":
    main()
