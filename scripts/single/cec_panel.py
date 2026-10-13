"""CEC-style rank of MC-ESO arms against a fixed panel of comparison methods.

The CEC mean-error rank is relative, so an arm only has a rank next to other
methods. Re-running the comparison methods for every arm is wasteful: the
panel (current default + comparison methods, same instances / seeds / budget)
is run once, and each arm is ranked as if it replaced the default in that
panel. Errors are core.run_data.cec_mean_errors (per-run f - f*, < 1e-8 as 0).

    python scripts/single/cec_panel.py --dim 10 \\
        --panel results/<panel_d10 run> --arms results/<arms_d10 run> [--base MC-ESO]

Prints, per arm: mean rank in the panel with the arm in place of the base
(lower is better) next to the base's own, the number of functions where the
arm's mean error is below / above the base's, and the functions whose rank
moved. Arms and panel must use the same instances, seeds and budget.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.run_data import cec_mean_errors, cec_ranks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--panel", type=Path, nargs="+", required=True,
                    help="one or more runs; their methods are pooled (the base must agree)")
    ap.add_argument("--arms", type=Path, nargs="+", required=True)
    ap.add_argument("--base", default="MC-ESO")
    ap.add_argument("--drop", default="", help="comma list of panel methods to leave out")
    args = ap.parse_args()

    panel: dict[str, dict[str, float]] = {}
    for d in args.panel:
        for f, by_m in cec_mean_errors(d / f"dim{args.dim}").items():
            for m, v in by_m.items():
                old = panel.setdefault(f, {}).get(m)
                if old is not None and old != v:
                    raise SystemExit(f"{m} differs between panel runs on {f}")
                panel[f][m] = v
    for f in panel:
        for m in filter(None, args.drop.split(",")):
            panel[f].pop(m, None)
    arms: dict[str, dict[str, float]] = {}
    for d in args.arms:
        for f, by_m in cec_mean_errors(d / f"dim{args.dim}").items():
            for m, v in by_m.items():
                if m != args.base:
                    arms.setdefault(m, {})[f] = v
    funcs = sorted(f for f in panel if all(f in a for a in arms.values()))
    others = sorted({m for f in funcs for m in panel[f]} - {args.base})

    def ranks_with(name: str, errs: dict[str, float]) -> dict[str, float]:
        table = {f: {**{m: panel[f][m] for m in others}, name: errs[f]} for f in funcs}
        rk = cec_ranks(table, others + [name])
        return {f: rk[f][name] for f in funcs}

    base_err = {f: panel[f][args.base] for f in funcs}
    base_rk = ranks_with(args.base, base_err)
    n = len(funcs)
    print(f"dim {args.dim}: {n} functions, panel = {args.base} + {', '.join(others)}")
    print(f"  {args.base:<22} mean rank {sum(base_rk.values()) / n:5.2f}")
    for m in sorted(arms):
        rk = ranks_with(m, arms[m])
        better = [f for f in funcs if arms[m][f] < base_err[f]]
        worse = [f for f in funcs if arms[m][f] > base_err[f]]
        moved = [(f[:3], base_rk[f], rk[f]) for f in funcs if rk[f] != base_rk[f]]
        print(f"  {m:<22} mean rank {sum(rk.values()) / n:5.2f}   "
              f"mean error below base on {len(better)}, above on {len(worse)}")
        if moved:
            print("      rank moved: " + ", ".join(f"{f} {a:g}->{b:g}" for f, a, b in moved))


if __name__ == "__main__":
    main()
