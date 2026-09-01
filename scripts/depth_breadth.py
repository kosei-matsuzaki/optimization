#!/usr/bin/env python3
"""Depth vs breadth: is any method good at both, or is the corner empty?

The niching literature scores one axis (how many optima, at eps >= 1e-5) and the
BBOB side scores the other (how deep one solution goes). This script puts both on
the same plane for every method in a quick run:

  depth    mean SR@1e-10 over the suite's functions — did the run bank at least
           one solution at full precision?
  breadth  mean cec_pr_mean — the CEC2013 peak ratio over the reported set

and marks the Pareto front. An empty top-right corner is the claim the
precision-portfolio formulation rests on: existing methods sit at one extreme or
the other, so a user who needs both has nothing to reach for.

Also prints, per function, which methods fail the depth requirement outright —
the "existing SOTA is disqualified" check.

Usage:
  python3 scripts/depth_breadth.py results/<run>_quick [more runs ...] [--dim 2]
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def _pct(s: str) -> float:
    s = (s or "").strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _rows(run_dir: Path, dim: int) -> list[dict]:
    path = run_dir / f"dim{dim}" / "summary.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _budget(run_dir: Path) -> str:
    import json
    meta = run_dir / "result.json"
    if meta.exists():
        try:
            return str(json.load(open(meta)).get("max_evals", "?"))
        except (ValueError, OSError):
            pass
    return "?"


def _pareto(points: dict[str, tuple[float, float]]) -> set[str]:
    """Methods not dominated on (depth, breadth), both higher-is-better."""
    front = set()
    for m, (d, b) in points.items():
        if not any(od >= d and ob >= b and (od > d or ob > b)
                   for om, (od, ob) in points.items() if om != m):
            front.add(m)
    return front


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--dim", type=int, default=2)
    args = ap.parse_args()

    for run in args.runs:
        rows = _rows(run, args.dim)
        if not rows:
            print(f"(no dim{args.dim} summary in {run.name})")
            continue
        by_method: dict[str, list[dict]] = {}
        for r in rows:
            by_method.setdefault(r["method"], []).append(r)

        pts: dict[str, tuple[float, float]] = {}
        for m, rs in by_method.items():
            depth = sum(_pct(r["sr_1e-10"]) for r in rs) / len(rs)
            breadth = sum(float(r["cec_pr_mean"]) for r in rs) / len(rs)
            pts[m] = (depth, breadth)
        front = _pareto(pts)

        print("=" * 72)
        print(f"{run.name}   dim{args.dim}   budget={_budget(run)}   "
              f"functions={len(by_method[next(iter(by_method))])}")
        print("=" * 72)
        print(f"  {'method':<14}{'depth SR@1e-10':>16}{'breadth PRmean':>16}   front")
        for m, (d, b) in sorted(pts.items(), key=lambda kv: -kv[1][1]):
            print(f"  {m:<14}{d:>15.0%}{b:>16.2f}   {'*' if m in front else ''}")

        # Scatter: breadth on x, depth on y.
        print()
        cols, rows_n = 46, 12
        grid = [[" "] * cols for _ in range(rows_n)]
        # One distinct letter per method; first letters collide (NCDE / NMMSO /
        # NM-Restart) and an unreadable key defeats the point of the plot.
        letters = {m: chr(ord("A") + i) for i, m in enumerate(sorted(pts))}
        for m, (d, b) in pts.items():
            x = min(cols - 1, int(b * (cols - 1)))
            y = min(rows_n - 1, int((1 - d) * (rows_n - 1)))
            grid[y][x] = letters[m] if grid[y][x] == " " else "+"
        for i, line in enumerate(grid):
            label = "1.0" if i == 0 else ("0.0" if i == rows_n - 1 else "   ")
            print(f"  depth {label} |" + "".join(line) + "|")
        print("  " + " " * 12 + "+" + "-" * cols + "+")
        print("  " + " " * 13 + "breadth 0.0" + " " * (cols - 22) + "1.0")
        print("  key: " + ", ".join(f"{letters[m]}={m}" for m in sorted(pts)))

        # Depth-requirement failures, per function.
        print("\n  methods failing the depth requirement (SR@1e-10 < 100%):")
        funcs = sorted({r["function"] for r in rows})
        for fn in funcs:
            fails = [(r["method"], _pct(r["sr_1e-10"])) for r in rows
                     if r["function"] == fn and _pct(r["sr_1e-10"]) < 1.0]
            if fails:
                txt = ", ".join(f"{m} {v:.0%}" for m, v in sorted(fails, key=lambda t: t[1]))
                print(f"    {fn:<22}{txt}")
        print()


if __name__ == "__main__":
    main()
