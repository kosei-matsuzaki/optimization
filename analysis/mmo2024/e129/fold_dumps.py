#!/usr/bin/env python3
"""その129 の片付け —— 32 本のダンプを `problem` 列つきの 2 本の `.csv.gz` に畳む（e128 と同じ形）。

16 ファイルに散らすとファイル数だけが増えて中身は同じなので、`analysis/` の目安 400 を
越えないように 1 手法 1 本にまとめる。**畳む前後で `analyze.py` の出力が 1 文字も
変わらないことを確認してから per-problem を消すこと**（この script は畳むだけで消さない）。

使い方: python3 analysis/mmo2024/e129/fold_dumps.py
"""
from __future__ import annotations

import csv
import glob
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)

JOBS = [
    (os.path.join(MMO, "e127", "dumps", "*_rrcma_seed200.csv*"),
     os.path.join(HERE, "dumps_rrcma_seed200.csv.gz"), "_rrcma_seed200"),
    (os.path.join(HERE, "descents", "*_seed200.csv*"),
     os.path.join(HERE, "descents_seed200.csv.gz"), "_seed200"),
]


def main():
    for pat, out, suffix in JOBS:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"  (skip) no files for {pat}")
            continue
        rows, header = [], None
        for path in files:
            prob = os.path.basename(path).split(suffix)[0]
            op = gzip.open if path.endswith(".gz") else open
            with op(path, "rt") as fh:
                rd = csv.reader(fh)
                head = next(rd)
                if header is None:
                    header = ["problem"] + head
                for r in rd:
                    rows.append([prob] + r)
        with gzip.open(out, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        print(f"  {len(files)} files -> {os.path.basename(out)}  ({len(rows)} rows, "
              f"{os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
