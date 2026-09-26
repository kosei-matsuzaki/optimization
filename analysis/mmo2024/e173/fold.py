#!/usr/bin/env python3
"""e173: 6 本の per-chunk null ダンプ（3 問 x 2 チャンク）を 1 本へ畳む（`function` 列を足す）。

CLAUDE.md の規約（行単位の生 CSV は `.csv.gz`、per-problem を残さない）と
その128〜その131・その170〜その172 と同じ手順。**畳む前後で `analyze.py` の出力が 1 文字も
変わらないことを確認してから元 6 本を消すこと。`rm` はこの script に含めない**
（その131 の事故 —— 畳む script と `rm` を 1 つのコマンド行に並べない）。

使い方: python3 analysis/mmo2024/e173/fold.py
"""
from __future__ import annotations
import csv, gzip, re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    srcs = sorted((HERE / "null").glob("*_s*_rl.csv.gz"))
    if not srcs:
        print("ダンプなし。skip")
        return
    rows, fields = [], None
    for s in srcs:
        func = re.sub(r"_s\d+_rl\.csv\.gz$", "", s.name)
        with gzip.open(s, "rt", newline="") as fh:
            rd = csv.DictReader(fh)
            for r in rd:
                r["function"] = func
                rows.append(r)
            if fields is None:
                fields = ["function"] + list(rd.fieldnames or [])
    rows.sort(key=lambda r: (r["function"], int(r["draw"])))
    out = HERE / "null_topup.csv.gz"
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(srcs)} 本 / {len(rows)} 行 -> {out.name} ({out.stat().st_size} B)")


if __name__ == "__main__":
    main()
