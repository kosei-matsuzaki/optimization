"""Rank MC-ESO and baselines on the CEC2022 hold-out suite.

For each of the 12 CEC2022 functions at dim=10, rank the 5 methods by
mean best_f (lower is better). Report:
  - per-function rank matrix
  - average rank per method
  - Wilcoxon paired test of MC-ESO vs each baseline

The point is to test whether MC-ESO's mechanisms generalize beyond BBOB
without any HP re-tuning. A competitive ranking here (better than at
least one baseline) is evidence against the "ad-hoc BBOB overfit" critique.

Usage:
    python -m analysis.cec2022_holdout
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
METHODS = ["CMA-ES", "PSO", "DE", "SaVOA", "MC-ESO"]


def _load_means(run_dir: Path) -> dict[str, dict[str, float]]:
    """{function: {method: mean_best_f}}."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with open(run_dir / "dim10" / "summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["function"]][row["method"]] = float(row["mean_best_f"])
    return out


def _ranks(per_method_mean: dict[str, float]) -> dict[str, float]:
    """Rank methods (1 = best, lower mean f). Ties get average rank."""
    methods = list(per_method_mean)
    vals = np.array([per_method_mean[m] for m in methods])
    ranks = stats.rankdata(vals, method="average")
    return dict(zip(methods, ranks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None,
                        help="Specific results dir (default: latest cec2022_holdout)")
    args = parser.parse_args()

    if args.dir:
        run_dir = Path(args.dir)
    else:
        candidates = sorted(RESULTS_ROOT.glob("*cec2022_holdout*"))
        if not candidates:
            raise SystemExit("No cec2022_holdout results found")
        run_dir = candidates[-1]

    print(f"Source: {run_dir}\n")
    means = _load_means(run_dir)
    funcs = sorted(means)

    # ── Per-function rank table ──────────────────────────────────────────
    print("## Per-function mean best_f and rank\n")
    print(f"| Function | {' | '.join(METHODS)} |")
    print("|" + "---|" * (len(METHODS) + 1))
    rank_rows: dict[str, list[float]] = {m: [] for m in METHODS}
    for func in funcs:
        ranks = _ranks(means[func])
        cells = [func]
        for m in METHODS:
            mean = means[func][m]
            r = ranks[m]
            rank_rows[m].append(r)
            cells.append(f"{mean:.2e} (#{int(r) if r == int(r) else r:.1f})")
        print("| " + " | ".join(cells) + " |")

    # ── Average rank per method ──────────────────────────────────────────
    print("\n## Aggregate ranking (lower is better)\n")
    print(f"| Method | avg rank | wins (rank=1) | top-2 | last |")
    print("|---|---:|---:|---:|---:|")
    avg_ranks = {m: float(np.mean(rs)) for m, rs in rank_rows.items()}
    for m in sorted(METHODS, key=lambda mm: avg_ranks[mm]):
        rs = np.array(rank_rows[m])
        wins = int((rs == 1).sum())
        top2 = int((rs <= 2).sum())
        last = int((rs == max(rank_rows[m])).sum() if rs.size else 0)
        last = int((rs >= len(METHODS) - 0.5).sum())
        print(f"| {m} | {avg_ranks[m]:.2f} | {wins} | {top2} | {last} |")

    # ── Wilcoxon MC-ESO vs each baseline (on means) ──────────────────────
    print("\n## Wilcoxon signed-rank: MC-ESO vs baseline on per-function mean f\n")
    print("`p_better` = one-sided p that MC-ESO has lower mean than baseline.\n")
    print("| Baseline | MC-ESO < baseline | tie | MC-ESO > baseline | p_two_sided | p_mceso_better |")
    print("|---|---:|---:|---:|---:|---:|")
    mceso_vals = np.array([means[f]["MC-ESO"] for f in funcs])
    for baseline in [m for m in METHODS if m != "MC-ESO"]:
        base_vals = np.array([means[f][baseline] for f in funcs])
        better = int((mceso_vals < base_vals).sum())
        worse = int((mceso_vals > base_vals).sum())
        tie = int((mceso_vals == base_vals).sum())
        # Wilcoxon two-sided
        diffs = mceso_vals - base_vals
        nz = diffs[diffs != 0]
        if len(nz) >= 1:
            try:
                _, p2 = stats.wilcoxon(nz)
                # One-sided: MC-ESO is "less" (better)
                _, p_less = stats.wilcoxon(nz, alternative="less")
            except ValueError:
                p2, p_less = float("nan"), float("nan")
        else:
            p2, p_less = float("nan"), float("nan")
        print(f"| {baseline} | {better} | {tie} | {worse} | {p2:.3f} | {p_less:.3f} |")


if __name__ == "__main__":
    main()
