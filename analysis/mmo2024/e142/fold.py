#!/usr/bin/env python3
"""その142 — 48 本の per-run 報告集合ダンプを 1 本の `report_sets.csv.gz` に畳む。

`problem` / `method` / `seed` 列を足すだけで、採点に渡る中身（`f` と座標）は 1 文字も変えない。
**畳む前後で `analyze.py` の出力が同一であることを確認してから per-run ダンプを消す**
（その128〜その131・その137・その139〜その141 と同じ手順）。**この script は `rm` を実行しない**
（その131 の事故 —— 畳む script と `rm` を同じコマンド行に並べない）。

使い方: python3 analysis/mmo2024/e142/fold.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "dumps")
OUT = os.path.join(HERE, "report_sets.csv.gz")


def main() -> int:
    if not os.path.isdir(DUMPS):
        print(f"no dumps dir: {DUMPS}")
        return 1
    files = sorted(fn for fn in os.listdir(DUMPS) if fn.endswith(".csv.gz"))
    if not files:
        print("no dumps")
        return 1
    rows, header, dim = [], None, None
    for fn in files:
        stem = fn[: -len(".csv.gz")]
        prob, rest = stem.split("_", 1)
        meth, sd = rest.rsplit("_seed", 1)
        with gzip.open(os.path.join(DUMPS, fn), "rt") as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames or []
            d = sum(1 for k in cols if k.startswith("x") and k[1:].isdigit())
            if dim is None:
                dim = d
            elif d != dim:
                print(f"** dim mismatch ** {fn}: {d} != {dim}")
                return 1
            for r in rd:
                rows.append([prob, meth, sd, r["f"]]
                            + [r[f"x{i}"] for i in range(dim)])
    header = ["problem", "method", "seed", "f"] + [f"x{i}" for i in range(dim)]
    with gzip.open(OUT, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"folded {len(files)} dumps / {len(rows)} rows -> {OUT}")
    print("次: analyze.py の出力が畳む前と同一か確認してから dumps/ を消すこと")
    return 0


if __name__ == "__main__":
    sys.exit(main())
