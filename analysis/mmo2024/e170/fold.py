#!/usr/bin/env python3
"""e170: 18 本の per-run ダンプを 2 本の `.csv.gz` に畳む（`function` / `seed` 列つき）。

保持規則（`CLAUDE.md` / research_loop.md）: **16 本以上のダンプを出すサイクルは、
出したぶんを同じサイクル内で畳んで per-run を残さない。**
畳む前後で `analyze.py` の出力が 1 文字も変わらないことを確認してから元を消す
（その128〜その141 と同じ手順。**`rm` はこの script に入れない** ＝ その131 の事故の再発防止）。
"""
from __future__ import annotations
import csv, glob, gzip, re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fold(pattern: str, out: str, key: re.Pattern) -> int:
    paths = sorted(glob.glob(str(HERE / pattern)))
    if not paths:
        print(f"{pattern}: 入力なし（既に畳んである）")
        return 0
    # 次元がちがうので `x*` 列の本数もちがう（2D は x0,x1 / 5D は x0..x4）。
    # 列は全ファイルの**和**を取り、無い列は空欄で埋める（`restval=""`）。
    rows, cols = [], ["function", "seed"]
    for p in paths:
        m = key.search(Path(p).name)
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", newline="") as fh:
            rd = csv.DictReader(fh)
            for c in (rd.fieldnames or []):
                if c not in cols:
                    cols.append(c)
            for r in rd:
                rows.append({"function": m.group(1), "seed": m.group(2), **r})
    with gzip.open(HERE / out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{pattern}: {len(paths)} 本 / {len(rows)} 行 → {out}")
    return len(paths)


def main() -> None:
    n = fold("rl_descents/*/*.csv", "rl_descents.csv.gz",
             re.compile(r"^(N\d\d-CF\d-\d+D)_seed(\d+)\.csv$"))
    n += fold("rl_reports/*/*.csv.gz", "rl_reports.csv.gz",
              re.compile(r"^(N\d\d-CF\d-\d+D)_Restart-Lander_seed(\d+)\.csv\.gz$"))
    if n:
        print(f"\n畳んだ元 {n} 本は別コマンドで消す:\n"
              f"  rm -r analysis/mmo2024/e170/rl_descents "
              f"analysis/mmo2024/e170/rl_reports")


if __name__ == "__main__":
    main()
