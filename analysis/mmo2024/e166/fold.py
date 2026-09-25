#!/usr/bin/env python3
"""その166 の降下ダンプ 21 本を 1 本の `descents.csv.gz` に畳む（`problem` / `seed` 列を足す）。

その128〜その131・その137・その139〜その141・その164 と同じ手順。
**畳む前後で `analyze.py` の出力が md5 まで一致することを確認してから per-problem を消す**
（消すのはこの script ではなく、確認したあとに別のコマンドで行う。その131 の事故の教訓 ——
畳む script と `rm` を 1 つのコマンド行に並べない）。
"""
from __future__ import annotations

import csv
import glob
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "descents")
OUT = os.path.join(HERE, "descents.csv.gz")


def main() -> None:
    paths = sorted(glob.glob(os.path.join(SRC, "*.csv")))
    if not paths:
        raise SystemExit(f"入力が無い: {SRC}/*.csv（削除済みなら畳む必要も無い）")
    # この回は 3D〜20D が混ざるので x 列の本数が問題ごとに違う。
    # 共通列 ＋ x0..x{max D-1} の和集合を取り、足りない x は空文字で埋める
    # （D=3 の行に x3.. は無い ＝ 読む側は `dim` 列で切ればよい）。
    FIX = ["descent", "evals", "best_f", "land_opt", "dist", "stop"]
    per = []
    for p in paths:
        base = os.path.basename(p)[: -len(".csv")]
        name, _, sd = base.rpartition("_seed")
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            cols = list(rd.fieldnames or [])
            if cols[: len(FIX)] != FIX:
                raise SystemExit(f"共通列が揃っていない: {p}")
            per.append((name, str(int(sd) // 100),
                        [c for c in cols if c.startswith("x")], list(rd)))
    dmax = max(len(xs) for _, _, xs, _ in per)
    xcols = [f"x{i}" for i in range(dmax)]
    header = ["problem", "seed", "dim"] + FIX + xcols
    rows = []
    for name, sd, xs, rr in per:
        for r in rr:
            rows.append([name, sd, str(len(xs))] + [r[c] for c in FIX]
                        + [r.get(c, "") for c in xcols])
    with gzip.open(OUT, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{len(paths)} 本 / {len(rows)} 行 -> {OUT} "
          f"({os.path.getsize(OUT)} バイト)")


if __name__ == "__main__":
    main()
