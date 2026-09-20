#!/usr/bin/env python3
"""その146 の片付け — 32 本の降下ダンプを 1 本の `.csv.gz` に、
32 本の driver CSV を 1 本の `driver_summary.csv` に畳む（その128〜その145 と同じ手順）。

**畳むだけ。削除はしない**（`rm` は別コマンドで打つ ＝ その131 の事故の再発防止）。
畳んだ後に `analyze.py` の出力が 1 文字も変わらないことを確認してから消すこと。

使い方: python3 analysis/mmo2024/e146/fold.py
"""
from __future__ import annotations

import csv
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def fold_descents():
    out = os.path.join(HERE, "descents.csv.gz")
    rows, head = [], None
    for arm in (3000, 6000):
        d = os.path.join(HERE, f"descents_{arm}")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith((".csv", ".csv.gz")):
                continue
            op = gzip.open if fn.endswith(".gz") else open
            with op(os.path.join(d, fn), "rt") as fh:
                rd = csv.DictReader(fh)
                head = head or ["arm", "problem"] + list(rd.fieldnames)
                for r in rd:
                    rows.append([arm, fn.split("_")[0]] + [r[k] for k in head[2:]])
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    print(f"  降下ダンプ {len(rows)} 行 -> {os.path.relpath(out, HERE)}")


def fold_driver():
    out = os.path.join(HERE, "driver_summary.csv")
    d = os.path.join(HERE, "by_problem")
    rows, head = [], None
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(d, fn)) as fh:
            rd = csv.DictReader(fh)
            head = head or list(rd.fieldnames)
            rows.extend([[r.get(k, "") for k in head] for r in rd])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    print(f"  driver CSV {len(rows)} 行 -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    fold_descents()
    fold_driver()
