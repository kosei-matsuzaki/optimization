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
  python3 scripts/explain_models.py [--props analysis/set_properties.csv]
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
    ap.add_argument("--props", type=Path, default=Path("analysis/set_properties.csv"))
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
    print("(quality is a cost, so a negative correlation means better sets score higher)\n")
    print(f"{'model':<12}{'n':>5}{'rho(spread)':>13}{'rho(distinct)':>15}"
          f"{'rho(quality)':>14}")
    print("-" * 59)

    per_model: dict[str, dict[str, float]] = {}
    for model, path in _MODELS.items():
        p = Path(path)
        if not p.exists():
            print(f"{model:<12}{'(missing ' + path + ')':>48}")
            continue
        a12 = _a12_by_method(p)
        keys = [k for k in a12 if k in prop]
        if len(keys) < 10 or spearmanr is None:
            print(f"{model:<12}{len(keys):>5}   (too few paired rows)")
            continue
        y = np.array([a12[k] for k in keys])
        X = np.array([prop[k] for k in keys])
        rhos = [float(spearmanr(X[:, j], y).statistic) for j in range(3)]
        per_model[model] = dict(zip(("spread", "distinct", "quality"), rhos))
        print(f"{model:<12}{len(keys):>5}{rhos[0]:>13.2f}{rhos[1]:>15.2f}"
              f"{rhos[2]:>14.2f}")

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
