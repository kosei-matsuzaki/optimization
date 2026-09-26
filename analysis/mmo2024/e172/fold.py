#!/usr/bin/env python3
"""e172: 13 本の per-function null ダンプを設定ごとに 1 本へ畳む（`function` 列を足す）。

CLAUDE.md の規約（行単位の生 CSV は `.csv.gz`、per-problem を残さない）と
その128〜その131・その170・その171 と同じ手順。**畳む前後で `analyze.py` の出力が 1 文字も
変わらないことを確認してから元 13 本を消すこと。`rm` はこの script に含めない**
（その131 の事故 —— 畳む script と `rm` を 1 つのコマンド行に並べない）。

使い方: python3 analysis/mmo2024/e172/fold.py
"""
from __future__ import annotations
import csv, gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAGS = ("rl",)


def main() -> None:
    for tag in TAGS:
        srcs = sorted((HERE / "null").glob(f"*_{tag}.csv.gz"))
        if not srcs:
            print(f"{tag}: ダンプなし。skip")
            continue
        rows, fields = [], None
        for s in srcs:
            func = s.name[: -len(f"_{tag}.csv.gz")]
            with gzip.open(s, "rt", newline="") as fh:
                rd = csv.DictReader(fh)
                for r in rd:
                    r["function"] = func
                    rows.append(r)
                if fields is None:
                    fields = ["function"] + list(rd.fieldnames or [])
        out = HERE / f"null_{tag}.csv.gz"
        with gzip.open(out, "wt", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"{tag}: {len(srcs)} 本 / {len(rows)} 行 -> {out.name} "
              f"({out.stat().st_size} B)")


if __name__ == "__main__":
    main()
