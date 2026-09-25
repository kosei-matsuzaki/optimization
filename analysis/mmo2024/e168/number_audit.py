#!/usr/bin/env python3
"""その168（統合の回）— 畳む／消す前の機械照合。

2 つを見る。
  **(1) 畳むログの小数が docs のどこかにあるか**（その162 と同じ手順）。
      `research_loop.md` の「作業ログ」から畳む範囲の小数を全部取り、
      **畳んだ後も残る文書**（同ファイルの残り ＋ `docs/*.md` ＋ `docs/*.html`）と照合する。
  **(2) 消す保存物の集計に出る小数が docs にあるか**。
      無いものは「消す前に文書へ移す」対象（保持規則）。

**この script は読むだけで、何も書かない**（出力は stdout）。

使い方: python3 analysis/mmo2024/e168/number_audit.py [畳む開始行] [畳む終了行]
"""
from __future__ import annotations
import glob
import re
import sys

NUM = re.compile(r"\d+\.\d+(?:e[+-]?\d+)?")
RL = "docs/research_loop.md"


def docs_corpus(exclude_rl_lines: tuple[int, int] | None) -> str:
    out = []
    for f in sorted(glob.glob("docs/*.md") + glob.glob("docs/*.html")):
        t = open(f, encoding="utf-8", errors="ignore").read()
        if f == RL and exclude_rl_lines:
            a, b = exclude_rl_lines
            ls = t.split("\n")
            t = "\n".join(ls[: a - 1] + ls[b:])
        out.append(t)
    return "".join(out)


def check(label: str, text: str, corpus: str) -> list[str]:
    uniq = sorted(set(NUM.findall(text)))
    miss = [n for n in uniq if n not in corpus]
    print(f"{label}: 小数 {len(NUM.findall(text))} 個 / 相異なり {len(uniq)} / docs に無い {len(miss)}")
    for n in miss:
        print(f"    {n}")
    return miss


def main() -> None:
    a = int(sys.argv[1]) if len(sys.argv) > 2 else 1882
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2185
    rl = open(RL, encoding="utf-8").read().split("\n")
    fold = "\n".join(rl[a - 1 : b])
    print(f"## (1) 畳むログ（{RL} の {a}-{b} 行、その159〜その164 の 6 件）\n")
    check("畳む範囲", fold, docs_corpus((a, b)))
    print("\n## (2) 消す保存物の集計（消したあとも数値が残っているか）\n")
    corpus = docs_corpus(None)
    for f in ("analysis/mmo2024/e165/scored.txt",
              "analysis/mmo2024/e115/scored.txt",
              "analysis/mmo2024/e115/s1/by_problem_s1.csv"):
        try:
            check(f, open(f, encoding="utf-8").read(), corpus)
        except FileNotFoundError:
            print(f"{f}: 無い（既に消えている）")
    print("\n注: (2) で「docs に無い」と出た小数は、**消していないファイル**（`scored.txt` /"
          "\n    `driver_summary.csv` / `by_problem_s1.csv`）に残っている値である。"
          "\n    その168 が消したのは行単位の降下ダンプだけで、集計は 1 つも消していない。")


if __name__ == "__main__":
    main()
