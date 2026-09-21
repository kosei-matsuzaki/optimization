#!/usr/bin/env python3
"""その152 の片付け — RR-CMA-ES の報告集合ダンプ 16 本を 1 本の `.csv.gz` に畳む
（その128〜その151 と同じ手順、`e151/fold.py` の写し。この回は driver CSV が無いので dumps だけ）。

**畳むだけ。削除はしない**（`rm` は別コマンドで打つ ＝ その131 の事故の再発防止）。
畳んだ後に `analyze.py` の出力が 1 文字も変わらないことを確認してから消すこと。

使い方: python3 analysis/mmo2024/e152/fold.py
"""
from __future__ import annotations

import csv
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def fold_dir(src, out, strip):
    d = os.path.join(HERE, src)
    rows, head = [], None
    for fn in sorted(os.listdir(d)):
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        op = gzip.open if fn.endswith(".gz") else open
        with op(os.path.join(d, fn), "rt") as fh:
            rd = csv.DictReader(fh)
            head = head or ["problem"] + list(rd.fieldnames)
            for r in rd:
                rows.append([fn.split(strip)[0]] + [r[k] for k in head[1:]])
    path = os.path.join(HERE, out)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    print(f"  {src}: {len(rows)} 行 -> {out}")


if __name__ == "__main__":
    fold_dir("dumps", "dumps_rrcma.csv.gz", "_rrcma")
