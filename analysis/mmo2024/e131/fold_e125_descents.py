#!/usr/bin/env python3
"""その131 の片付け —— その125 の降下ダンプ 32 本を腕ごとに 1 本へ畳む（その128-130 と同じ手順）。

**その125 の路線は生きている**（キュー 3 の始点が `e125/arm.py` と `e125/rho_degeneracy.csv`）ので
**消さずに畳むだけ**。32 ファイルに散らしていても中身は同じで、`analysis/` のファイル数だけを食う。

  * `analysis/mmo2024/e125/descents/*_reseed_seed0.csv.gz` -> `e125/descents_reseed_seed0.csv.gz`
  * `analysis/mmo2024/e125/descents/*_bhop_seed0.csv.gz`   -> `e125/descents_bhop_seed0.csv.gz`

**畳む前後で `e125/analyze.py` の出力が 1 文字も変わらないことを確認してから per-problem を消すこと**
（この script は畳むだけで消さない）。`e125/analyze.py` には combined を読む退避路を足してある。

使い方: python3 analysis/mmo2024/e131/fold_e125_descents.py
"""
from __future__ import annotations

import csv
import glob
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MMO = os.path.dirname(HERE)
E125 = os.path.join(MMO, "e125")

JOBS = [(os.path.join(E125, "descents", f"*_{arm}_seed0.csv*"),
         os.path.join(E125, f"descents_{arm}_seed0.csv.gz"), f"_{arm}_seed0")
        for arm in ("reseed", "bhop")]


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
                elif header[1:] != head:
                    raise SystemExit(f"列が一致しない: {path}")
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
