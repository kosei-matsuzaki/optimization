#!/usr/bin/env python3
"""その175 — per-problem の報告集合ダンプを次元ごとに 1 本の `.csv.gz` に畳む。

その128〜その131・その137・その139〜その141 と同じ手順（`problem` / `seed` 列を足して 1 本に
まとめ、**畳む前後で `analyze.py` の出力が 1 文字も変わらないことを確認してから消す**）。
**値は 1 つも変えない。**

  python3 analysis/mmo2024/e175/fold.py D05 [D20 ...]
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def fold(dim):
    d = os.path.join(HERE, "dumps", dim)
    files = sorted(f for f in os.listdir(d) if f.endswith((".csv", ".csv.gz")))
    rows, ncol = [], 0
    for fn in files:
        full = fn.split("_")[0]
        seed = fn.split("seed")[-1].split(".")[0]
        op = gzip.open if fn.endswith(".gz") else open
        with op(os.path.join(d, fn), "rt") as fh:
            rs = list(csv.DictReader(fh))
        ncol = max(ncol, sum(1 for k in rs[0] if k.startswith("x") and k[1:].isdigit()))
        for r in rs:
            r["problem"], r["seed"] = full, seed
            rows.append(r)
    out = os.path.join(HERE, f"report_sets_{dim}.csv.gz")
    cols = ["problem", "seed", "f"] + [f"x{i}" for i in range(ncol)]
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{dim}: {len(files)} ファイル / {len(rows)} 行 -> {out} "
          f"({os.path.getsize(out)} B)")


if __name__ == "__main__":
    for dim in (sys.argv[1:] or ["D05"]):
        fold(dim)
