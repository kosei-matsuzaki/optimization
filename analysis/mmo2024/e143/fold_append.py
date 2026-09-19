#!/usr/bin/env python3
"""その143 — 13 本の per-run 報告集合ダンプを `e142/report_sets.csv.gz` に**追記**して畳む。

キュー 1 の指示（「13 本のダンプは同じサイクルで `e142/report_sets.csv.gz` に追記する形で畳む」）
どおりの形。`problem` / `method` / `seed` 列を足すだけで、採点に渡る中身（`f` と座標）の
**文字列表現を 1 文字も変えない**。既存行も 1 行も触らない。

検査（**これが通るまで per-run ダンプを消さない**）:
  1. 追記前に存在した行が、追記後も**同じ順序で同じ文字列**であること。
  2. 追記した 13 問の行が、**元のダンプと 1 文字違わず**一致すること。

この script は `rm` を実行しない（その131 の事故 —— 畳む script と `rm` を同じ行に並べない）。

使い方: python3 analysis/mmo2024/e143/fold_append.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
DUMPS = os.path.join(HERE, "dumps")
TARGET = os.path.join(MMO, "e142", "report_sets.csv.gz")


def read_gz(path):
    with gzip.open(path, "rt") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        return header, [tuple(r) for r in rd]


def dump_rows(path):
    """1 本のダンプを (f, x0..xd) の文字列タプル列で返す（値は触らない）。"""
    stem = os.path.basename(path)[: -len(".csv.gz")]
    prob, rest = stem.split("_", 1)
    meth, sd = rest.rsplit("_seed", 1)
    with gzip.open(path, "rt") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        dim = sum(1 for k in cols if k.startswith("x") and k[1:].isdigit())
        out = [(prob, meth, sd, r["f"]) + tuple(r[f"x{i}"] for i in range(dim))
               for r in rd]
    return prob, meth, dim, out


def main() -> int:
    if not os.path.isdir(DUMPS):
        print(f"no dumps dir: {DUMPS}")
        return 1
    files = sorted(fn for fn in os.listdir(DUMPS) if fn.endswith(".csv.gz"))
    if not files:
        print("no dumps")
        return 1
    header, before = read_gz(TARGET)
    base_dim = sum(1 for k in header if k.startswith("x") and k[1:].isdigit())
    existing = {(r[0], r[1], r[2]) for r in before}

    added, new_rows = {}, []
    for fn in files:
        prob, meth, dim, rows = dump_rows(os.path.join(DUMPS, fn))
        if dim != base_dim:
            print(f"** dim mismatch ** {fn}: {dim} != {base_dim}")
            return 1
        key = (prob, meth, rows[0][2] if rows else "0")
        if key in existing:
            print(f"** already folded ** {key} — 追記しない（二重計上を防ぐ）")
            return 1
        added[(prob, meth)] = rows
        new_rows.extend(rows)

    with gzip.open(TARGET, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(before)
        w.writerows(new_rows)

    # ---- 検査 1: 既存行が 1 行も動いていない
    header2, after = read_gz(TARGET)
    if header2 != header or after[:len(before)] != before:
        print("** 検査 1 失敗 ** 既存行が変わった。per-run ダンプは消さないこと。")
        return 1
    # ---- 検査 2: 追記行が元のダンプと 1 文字違わない
    if after[len(before):] != new_rows:
        print("** 検査 2 失敗 ** 追記行が元のダンプと一致しない。")
        return 1
    print(f"検査 1 通過: 既存 {len(before)} 行は順序も文字列も不変")
    print(f"検査 2 通過: 追記 {len(new_rows)} 行は元のダンプと 1 文字違わない")
    for (prob, meth), rows in sorted(added.items()):
        print(f"  + {prob} {meth}: {len(rows)} 行")
    print(f"-> {TARGET}  ({len(before)} -> {len(after)} 行)")
    print("次: e142/analyze.py を回し、その142 で取れていた 3 問の Score が 4 桁一致することを"
          "確認してから dumps/ を消すこと")
    return 0


if __name__ == "__main__":
    sys.exit(main())
