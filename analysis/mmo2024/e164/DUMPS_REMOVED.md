# `e164/descents.csv.gz` は その166 が削除した（2026-09-25）

**消したもの**: `e164/descents.csv.gz`（`descent_budget=50000` の D=20 × 16 run の降下ダンプ、163,424 バイト）。

**理由**: **降下長の軸は その163〜その165 の 4 水準で閉じた**（2026-09-24 の俯瞰の キュー 1 が
「問い 1 が閉じたら `e163` と `e164` の `descents.csv.gz` は消してよい」と名指しした）。

**数値の行き先**: [acceptance_topology.md](../../../docs/acceptance_topology.md) の
**「この軸の数値を消す前に移した表」**（その165 の節の末尾）。
**消す前に `e164/analyze.py` を実走して `scored.txt` がバイト一致することを確認し、
`scored.txt` の小数 176 個を docs と機械照合して、docs に無かった 9 個を先に docs へ移した。**

**残してあるもの**: `e151/descents.csv.gz`（12500、D=20 の標準対照）と
`e165/descents.csv.gz`（100000）。**この 2 本は消していない。**

**`e164/analyze.py` は「理由を印字して exit 1」に直した**（その162 が 4 本に当てた形）。
