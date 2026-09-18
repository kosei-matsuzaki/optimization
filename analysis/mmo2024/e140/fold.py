#!/usr/bin/env python3
"""その140 — per-problem のダンプを 1 本ずつに畳む（規約: 16 本以上出す回は同じ回で畳む）。

  * `dumps/*.csv.gz`（NMMSO の報告集合、12 本）   -> `report_sets.csv.gz`（`problem` / `seed` 列つき）
  * `descents/*.csv.gz`（null の降下、12 本）      -> `descents.csv.gz`（同上）
  * `by_problem*/ *.csv`（集計、24 本）            -> `runs.csv`（1 本）

**畳むだけ。削除はしない**（その131 の事故: 畳む script と `rm` を 1 コマンド行に並べない）。
削除は `analyze.py` の出力が 1 文字も変わらないことを確かめてから別のコマンドで行う。

使い方: python3 analysis/mmo2024/e140/fold.py
"""
from __future__ import annotations

import csv
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def fold_dir(dirname, out_name):
    dd = os.path.join(HERE, dirname)
    if not os.path.isdir(dd):
        return 0
    rows, header = [], None
    for fn in sorted(os.listdir(dd)):
        if not fn.endswith((".csv", ".csv.gz")):
            continue
        prob = fn.split("_")[0]
        sd = int(fn.rsplit("seed", 1)[1].split(".")[0])
        # 降下ダンプのファイル名は `seed_index x 100`（その123 の罠）。index に揃えて畳む。
        if sd >= 100:
            sd //= 100
        op = gzip.open if fn.endswith(".gz") else open
        with op(os.path.join(dd, fn), "rt") as fh:
            rd = csv.DictReader(fh)
            if header is None:
                header = ["problem", "seed"] + list(rd.fieldnames)
            for r in rd:
                rows.append([prob, sd] + [r[k] for k in header[2:]])
    if header is None:
        return 0
    with gzip.open(os.path.join(HERE, out_name), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def fold_agg(out_name="runs.csv"):
    rows, header = [], None
    for dirname in ("by_problem", "by_problem_null"):
        dd = os.path.join(HERE, dirname)
        if not os.path.isdir(dd):
            continue
        for fn in sorted(os.listdir(dd)):
            if not fn.endswith(".csv"):
                continue
            with open(os.path.join(dd, fn)) as fh:
                rd = csv.DictReader(fh)
                if header is None:
                    header = ["src"] + list(rd.fieldnames)
                for r in rd:
                    rows.append([fn[:-4]] + [r.get(k, "") for k in header[1:]])
    if header is None:
        return 0
    with open(os.path.join(HERE, out_name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    print("report_sets.csv.gz :", fold_dir("dumps", "report_sets.csv.gz"), "行")
    print("descents.csv.gz    :", fold_dir("descents", "descents.csv.gz"), "行")
    print("runs.csv           :", fold_agg(), "行")
