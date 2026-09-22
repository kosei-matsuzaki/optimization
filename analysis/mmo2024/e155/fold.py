#!/usr/bin/env python3
"""その155 — 48 本の per-problem ダンプを 2 本の `.csv.gz` に畳む。

その128〜その131・その137・その139〜その141・その153 と同じ手順:
`problem` 列（と `cov` 列）を足して 1 本にまとめ、**畳む前後で `analyze.py` の出力が
1 文字も変わらないことを確認してから** per-problem を消す。
**畳む script と `rm` は同じコマンド行に並べない**（その131 の事故）。

使い方: python3 analysis/mmo2024/e155/fold.py
"""
from __future__ import annotations

import csv
import glob
import gzip
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"(M\d\d-D05-PIN01)_cov([\d.]+)_seed0\.csv$")


def fold(subdir, outname):
    paths = sorted(glob.glob(os.path.join(HERE, subdir, "*.csv")))
    if not paths:
        raise SystemExit(f"入力が無い: {subdir}/")
    out = os.path.join(HERE, outname)
    header = None
    n = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        for p in paths:
            m = PAT.search(os.path.basename(p))
            if not m:
                raise SystemExit(f"名前が規則外: {p}")
            prob, cov = m.group(1), m.group(2)
            with open(p) as g:
                rd = csv.reader(g)
                head = next(rd)
                if header is None:
                    header = ["problem", "cov"] + head
                    w.writerow(header)
                elif ["problem", "cov"] + head != header:
                    raise SystemExit(f"列がそろわない: {p}")
                for row in rd:
                    w.writerow([prob, cov] + row)
                    n += 1
    print(f"{subdir}/ {len(paths)} 本 -> {outname}  {n} 行")


if __name__ == "__main__":
    fold("dumps", "dumps_rrcma_cov.csv.gz")
    fold("restarts", "restarts_cov.csv.gz")
