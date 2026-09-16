#!/usr/bin/env python3
"""その131 の片付け —— 今回埋めた 9 本を、その129 の既存の combined `.csv.gz` に**足し直す**。

その129 の `fold_dumps.py` は per-problem のグロブだけを見て combined を書き直すので、
そのまま流すと**既に畳んである 12 問 / 11 問が消える**。この script は
**既存 combined を読んでから、まだ入っていない problem だけを追記する**（既存行は 1 バイトも触らない）。

- `analysis/mmo2024/e127/dumps/*_rrcma_seed200.csv*` -> `analysis/mmo2024/e129/dumps_rrcma_seed200.csv.gz`
- `analysis/mmo2024/e129/descents/*_seed200.csv*`    -> `analysis/mmo2024/e129/descents_seed200.csv.gz`

**畳む前後で `e129/analyze.py` の出力が 1 文字も変わらないことを確認してから per-problem を消すこと**
（この script は畳むだけで消さない）。

使い方: python3 analysis/mmo2024/e131/fold_into_e129.py
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
     os.path.join(MMO, "e129", "dumps_rrcma_seed200.csv.gz"), "_rrcma_seed200"),
    (os.path.join(MMO, "e129", "descents", "*_seed200.csv*"),
     os.path.join(MMO, "e129", "descents_seed200.csv.gz"), "_seed200"),
]


def read_combined(path):
    if not os.path.exists(path):
        return None, []
    with gzip.open(path, "rt") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        return header, [r for r in rd]


def main():
    for pat, out, suffix in JOBS:
        header, rows = read_combined(out)
        have = {r[0] for r in rows}
        files = sorted(glob.glob(pat))
        added = 0
        for path in files:
            prob = os.path.basename(path).split(suffix)[0]
            if prob in have:
                print(f"  (skip) {prob} は既に combined にある")
                continue
            op = gzip.open if path.endswith(".gz") else open
            with op(path, "rt") as fh:
                rd = csv.reader(fh)
                head = next(rd)
                if header is None:
                    header = ["problem"] + head
                elif header[1:] != head:
                    raise SystemExit(f"列が一致しない: {path}")
                for r in rd:
                    rows.append([prob] + r)
            added += 1
        rows.sort(key=lambda r: r[0])
        with gzip.open(out, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        probs = sorted({r[0] for r in rows})
        print(f"  +{added} files -> {os.path.basename(out)}  "
              f"({len(rows)} rows, {len(probs)} problems, {os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
