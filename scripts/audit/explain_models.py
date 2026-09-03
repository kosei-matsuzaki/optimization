#!/usr/bin/env python3
"""Which property of a reported set decides which unmodelled criterion it survives?

The audit's winners change completely with the shape of the unknown. This joins
the per-method set properties (scripts/set_properties.py) against the per-method
performance under each scenario model, and reports rank correlations.

Hypothesis: spatial spread predicts performance where the unknown relocates the
optimum (instances, forbidden regions); nominal quality predicts it where the
unknown only reorders nearby solutions (a linear bias). Refuted if neither
property correlates with either model.

Usage:
  python3 scripts/explain_models.py [--props analysis/audit/set_properties.csv]
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
_BASELINE = "quality"


def _a12_by_method(path: Path) -> dict[tuple[str, str], float]:
    """A12 of each method against the baseline, per function, from paired rows."""
    data: dict[tuple[str, str], dict[tuple[int, int], float]] = collections.defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            data[(r["function"], r["method"])][(int(r["seed"]), int(r["scenario"]))] = \
                float(r["regret"])
    out: dict[tuple[str, str], float] = {}
    for (fn, m), vals in data.items():
        if m == _BASELINE:
            continue
        base = data.get((fn, _BASELINE), {})
        keys = sorted(set(base) & set(vals))
        if len(keys) < 10:
            continue
        a = np.array([vals[k] for k in keys])
        b = np.array([base[k] for k in keys])
        out[(fn, m)] = float((np.sum(a < b) + 0.5 * np.sum(a == b)) / len(keys))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--props", type=Path, default=Path("analysis/audit/set_properties.csv"))
    args = ap.parse_args()

    props: dict[tuple[str, str], list[list[float]]] = collections.defaultdict(list)
    with open(args.props, newline="") as fh:
        for r in csv.DictReader(fh):
            props[(r["function"], r["method"])].append(
                [float(r["spread"]), float(r["distinct"]), float(r["quality"])])
    prop = {k: np.mean(np.array(v), axis=0) for k, v in props.items()}

    print("set property vs performance, per scenario model")
    print("spread = mean pairwise distance of the reported top K (span-relative)")
    print("quality = mean f of the top K, normalised by the spread of local optima")
    print("(quality is a cost, so a negative correlation means better sets score higher)")
    print("")
    print("Correlated WITHIN each function, then averaged over functions. Pooling")
    print("functions instead would mostly measure how the properties differ")
    print("between functions, whose scales differ by orders of magnitude; the")
    print("pooled value is printed alongside so that gap stays visible.")
    print("'agree' counts functions whose rho has the sign of the mean.")
    print("")
    print(f"{'model':<11}{'fn':>4}{'spread':>19}{'distinct':>19}{'quality':>19}")
    print(f"{'':<11}{'':>4}{'mean (agree)  pool':>19}{'mean (agree)  pool':>19}"
          f"{'mean (agree)  pool':>19}")
    print("-" * 72)

    per_model: dict[str, dict[str, float]] = {}
    for model, path in _MODELS.items():
        mp = Path(path)
        if not mp.exists():
            print(f"{model:<11}{'(missing ' + path + ')':>61}")
            continue
        a12 = _a12_by_method(mp)
        keys = [k for k in a12 if k in prop]
        if len(keys) < 10 or spearmanr is None:
            print(f"{model:<11}{len(keys):>4}   (too few paired rows)")
            continue

        by_fn: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for k in keys:
            by_fn[k[0]].append(k)

        cells, means = [], {}
        for j, prop_name in enumerate(("spread", "distinct", "quality")):
            rs = []
            for fn, ks in by_fn.items():
                if len(ks) < 5:
                    continue
                xv = np.array([prop[k][j] for k in ks])
                yv = np.array([a12[k] for k in ks])
                if np.ptp(xv) == 0 or np.ptp(yv) == 0:
                    continue
                r = float(spearmanr(xv, yv).statistic)
                if not np.isnan(r):
                    rs.append(r)
            pooled = float(spearmanr(np.array([prop[k][j] for k in keys]),
                                     np.array([a12[k] for k in keys])).statistic)
            if rs:
                m = float(np.mean(rs))
                agree = sum(1 for r in rs if (r > 0) == (m > 0))
                means[prop_name] = m
                cells.append(f"{m:>6.2f} ({agree:>2}/{len(rs):<2}){pooled:>6.2f}")
            else:
                cells.append("-")
        per_model[model] = means
        print(f"{model:<11}{len(by_fn):>4}" + "".join(f"{c:>19}" for c in cells))


    # Method-level view: average A12 per method under each model, next to the
    # method's average set properties. Correlations pool functions, which can
    # hide a method-level story, so both are printed.
    print("\nby method (A12 averaged over functions)")
    header = f"{'method':<14}{'spread':>8}{'distinct':>9}{'quality':>9}"
    a12s = {}
    for model, path in _MODELS.items():
        if Path(path).exists():
            a12s[model] = _a12_by_method(Path(path))
            header += f"{model:>11}"
    print(header)
    print("-" * len(header))
    methods = sorted({m for _, m in prop})
    for m in methods:
        pv = np.mean(np.array([prop[k] for k in prop if k[1] == m]), axis=0)
        line = f"{m:<14}{pv[0]:>8.3f}{pv[1]:>9.1f}{pv[2]:>9.3f}"
        for model in a12s:
            vals = [v for k, v in a12s[model].items() if k[1] == m]
            line += f"{np.mean(vals):>11.2f}" if vals else f"{'-':>11}"
        print(line)


if __name__ == "__main__":
    main()
