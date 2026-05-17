"""Per-class effect aggregation across MC-ESO ablations.

Goal: for each ablation X, show whether removing/replacing mechanism X
affects an entire problem CLASS (separable / ill-cond / multimodal /
weak-structure / multi-optima / deceptive-2d), not just a single function.

Output: a (mechanism × class) table of mean Δsr_1e-7 with Wilcoxon
signed-rank p-values, plus a verdict ("class-level" vs "single-function")
based on whether the mean effect exceeds a noise threshold AND the
Wilcoxon p < 0.1.

Usage:
    python -m analysis.class_effects [--metric sr_1e-7] [--threshold 0.05]
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


# (ablation label, results dir glob, "vs" reference dir glob, accepted?)
ABLATIONS: list[tuple[str, str, str, bool]] = [
    ("A: niche_radius_ratio (× span)",   "20260516_000628_A_niche_relativized_quick",
                                          "20260515_172500_ベースライン_quick",       True),
    ("B: drop _meaningful_improvement",   "20260516_011156_B_drop_log_slope_quick",
                                          "20260516_000628_A_niche_relativized_quick", False),
    ("C: σ adapt always-on",              "20260516_023956_C_drop_sigma_gate_quick",
                                          "20260516_000628_A_niche_relativized_quick", True),
    ("D: σ-based drilling threshold",     "20260516_113353_D_precision_sigma_quick",
                                          "20260516_023956_C_drop_sigma_gate_quick",   True),
    ("E: σ_floor spillover gate",         "20260517_001256_E_sigma_gate_quick",
                                          "20260516_113353_D_precision_sigma_quick",   False),
    ("F: drop basin_switch",              "20260517_010047_F_no_basin_switch_quick",
                                          "20260516_113353_D_precision_sigma_quick",   False),
    ("G: drop escalate (+ diversify)",    "20260517_023446_G_no_escalate_quick",
                                          "20260516_113353_D_precision_sigma_quick",   True),
    ("I: axis sweep boundary-only",       "20260517_035303_I_axis_boundary_quick",
                                          "20260517_023446_G_no_escalate_quick",       True),
]


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _load_mceso_sr(run_dir_name: str, metric: str) -> dict[str, tuple[float, str]]:
    """Return {function: (sr_value_in_[0,1], category)} for MC-ESO."""
    csv_path = RESULTS_ROOT / run_dir_name / "dim2" / "summary.csv"
    out: dict[str, tuple[float, str]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] != "MC-ESO":
                continue
            raw = row[metric].rstrip("%")
            out[row["function"]] = (float(raw) / 100.0, row["category"])
    return out


def _aggregate_by_class(deltas: dict[str, float],
                        categories: dict[str, str]
                        ) -> dict[str, list[float]]:
    """Group per-function deltas into a {class: [delta, ...]} dict."""
    by_class: dict[str, list[float]] = defaultdict(list)
    for func, d in deltas.items():
        by_class[categories[func]].append(d)
    return by_class


def _class_stats(per_class: dict[str, list[float]]
                 ) -> dict[str, dict[str, float]]:
    """Compute mean / median / Wilcoxon p-value per class."""
    out: dict[str, dict[str, float]] = {}
    for cls, vals in per_class.items():
        arr = np.array(vals)
        # Wilcoxon vs zero: tests "is the mean delta different from zero?"
        nonzero = arr[arr != 0]
        if len(nonzero) >= 1:
            try:
                stat, p = stats.wilcoxon(nonzero) if len(nonzero) > 1 \
                          else (float("nan"), 1.0)
            except ValueError:
                stat, p = float("nan"), 1.0
        else:
            stat, p = float("nan"), 1.0
        out[cls] = {
            "n": float(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "n_pos": float((arr > 0).sum()),
            "n_neg": float((arr < 0).sum()),
            "p_wilcoxon": float(p),
        }
    return out


CLASS_ORDER = [
    "separable", "ill-cond", "multimodal", "weak-structure",
    "multi-optima", "deceptive-2d",
]


def _row_verdict(stats: dict[str, float], threshold: float) -> str:
    """One-character verdict: + class-level positive, - class-level negative,
    ~ mixed-but-significant, . no significant effect."""
    if stats["p_wilcoxon"] > 0.20 or abs(stats["mean"]) < threshold:
        return "."
    if stats["mean"] > 0:
        return "+"
    return "-"


def _print_table(rows: list[tuple[str, bool, dict[str, dict[str, float]]]],
                 metric: str, threshold: float) -> None:
    print(f"\n## Class-level ablation effects on {metric}\n")
    print(f"Cell format: `mean Δ (n_pos/n_neg, p)` — Δ in percentage points (e.g. -0.13 = -13%).")
    print(f"Verdict mark: `+/-` class-level effect (|Δ| ≥ {threshold:.2f} AND Wilcoxon p ≤ 0.20),")
    print("`.` no detectable class-level effect.\n")

    # Header
    head = ["Ablation", "✓/✗"] + CLASS_ORDER + ["overall"]
    widths = [max(len(h), 6) for h in head]
    widths[0] = 36  # ablation label
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    print("| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |")
    print(sep)
    for label, accepted, per_class_stats in rows:
        mark = "✓" if accepted else "✗"
        cells = [label.ljust(widths[0]), mark.center(widths[1])]
        # per class
        all_deltas: list[float] = []
        for cls, w in zip(CLASS_ORDER, widths[2:-1]):
            s = per_class_stats.get(cls)
            if s is None:
                cells.append("—".center(w))
            else:
                verdict = _row_verdict(s, threshold)
                mean = s["mean"]
                p = s["p_wilcoxon"]
                cell = f"{verdict}{mean:+.2f} p={p:.2f}"
                cells.append(cell.ljust(w))
                all_deltas.extend([s["mean"]] * int(s["n"]))
        # overall (weighted by counts)
        if all_deltas:
            overall_mean = float(np.mean(all_deltas))
            cells.append(f"{overall_mean:+.3f}".ljust(widths[-1]))
        else:
            cells.append("—".ljust(widths[-1]))
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="sr_1e-7",
                        help="Metric column (sr_1e-2, sr_1e-4, sr_1e-7, sr_1e-10)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Mean |Δ| threshold for class-level verdict (default 0.05 = 5%)")
    args = parser.parse_args()

    rows: list[tuple[str, bool, dict[str, dict[str, float]]]] = []
    for label, run_dir, ref_dir, accepted in ABLATIONS:
        try:
            cur = _load_mceso_sr(run_dir, args.metric)
            ref = _load_mceso_sr(ref_dir, args.metric)
        except FileNotFoundError as e:
            print(f"!! skipping {label}: {e}")
            continue
        funcs = sorted(set(cur) & set(ref))
        if not funcs:
            print(f"!! no overlapping functions for {label}")
            continue
        deltas = {f: cur[f][0] - ref[f][0] for f in funcs}
        categories = {f: cur[f][1] for f in funcs}
        per_class = _aggregate_by_class(deltas, categories)
        per_class_stats = _class_stats(per_class)
        rows.append((label, accepted, per_class_stats))

    _print_table(rows, args.metric, args.threshold)


if __name__ == "__main__":
    main()
