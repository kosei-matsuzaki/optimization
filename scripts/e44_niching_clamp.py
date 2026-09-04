#!/usr/bin/env python3
"""Entry 44, niching side: which clamp shapes leave the variant intact?

Entry 43's stored pairs (`analysis/hm/e43_*.csv`) are the reference: `base` and
the unclamped eps-relative arm `level_rel_c10`, 30 seeds, eps 1e-1/1e-3/1e-5.
This checks each clamped arm row for row against BOTH references, because the
two possible answers are the whole question:

  a clamp that only stops the endpoint  -> must reproduce `level_rel_c10`
  the default clamp entry 37 wrote down -> reproduces `base` on any function
                                           with f_init_scale < 1, i.e. it
                                           deletes the intervention there

`reported` is the peak count the PR is computed from, so equality on the
(function, seed, eps) key is equality of PR, seed by seed -- strictly stronger
than equality of the mean.

Usage:
  python3 scripts/e44_niching_clamp.py
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HM = ROOT / "analysis" / "hm"
E44 = HM / "e44"
KEY = ("function", "seed", "eps")
COLS = ("reported", "visited", "distinct", "hunts", "landed", "best_f")


def _index(paths) -> dict:
    out = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for r in csv.DictReader(open(p)):
            out[tuple(r[k] for k in KEY)] = r
    return out


def _cmp(name: str, arm: dict, ref: dict, ref_name: str) -> None:
    common = sorted(set(arm) & set(ref))
    if not common:
        print(f"  {name} vs {ref_name}: no overlapping cells")
        return
    diff = {c: 0 for c in COLS}
    rows_diff = 0
    for k in common:
        d = [c for c in COLS if arm[k][c] != ref[k][c]]
        for c in d:
            diff[c] += 1
        rows_diff += bool(d)
    funcs = sorted({k[0] for k in common})
    tag = "IDENTICAL" if rows_diff == 0 else f"{rows_diff} rows differ"
    print(f"  {name:<22} vs {ref_name:<6}: {len(common):>4} cells "
          f"[{','.join(f.split('-')[0] for f in funcs)}]  -> {tag}")
    if rows_diff:
        print("      per-column rows differing: "
              + ", ".join(f"{c}={n}" for c, n in diff.items() if n))


def main() -> None:
    # N06-Shubert2D is not in entry 43's set (entry 36 measured it): base at 30
    # seeds comes from entry 40's decomposition, the unclamped arm from entry
    # 36 at 15 seeds, so only seeds 0-14 overlap there.
    base = _index([HM / "e43_2d_base.csv", HM / "e43_n09_base.csv",
                   HM / "n06_decomp_base.csv"])
    rel = _index([HM / "e43_2d_rel.csv", HM / "e43_n09_rel.csv",
                  HM / "n06_rel_c10.csv"])
    print(f"reference cells: base {len(base)}, unclamped rel {len(rel)}\n")
    for arm in ("level_rel_c10_fis", "level_rel_c10_cap", "level_rel_c10_dflt"):
        a = _index([E44 / f"nich_{arm}_2d.csv", E44 / f"nich_{arm}_3d.csv"])
        if not a:
            continue
        print(f"=== {arm}  ({len(a)} cells run)")
        _cmp(arm, a, rel, "rel")
        _cmp(arm, a, base, "base")
        print()


if __name__ == "__main__":
    main()
