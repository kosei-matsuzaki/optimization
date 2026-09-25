# `e165/descents.csv.gz` は その168 が削除した（2026-09-25、統合の回）

**消したもの**: `e165/descents.csv.gz`（`descent_budget=100000` の D=20 × 16 run の降下ダンプ、155,536 バイト）。

**理由（その166 の削除と違い、こちらは「読み手がもう居ない」）**:
**`e165/analyze.py` は 4 水準（12500 / 25000 / 50000 / 100000）を全部要求する**ので、
**その166 が `e163` / `e164` の `descents.csv.gz` を消した時点でこの script は走れなくなった**
（実測: `PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e165/analyze.py` は
`入力が無い: .../e163/descents.csv.gz` で exit 1）。
**この 100000 のダンプを読む経路は他に 1 本も無い**ので、その166 の commit の瞬間に孤児になっていた。
**その168 の走査 E（`os.path.join` の途中の要素が変数であるパス）で判明した** ——
`e165/analyze.py:105` は `os.path.join(MMO, entry, "descents.csv.gz")` の形なので、
`scripts/scan_silent_null.py` の走査 A / C / D はどれも掛からない。

**数値の行き先**:
- 4 水準の内訳（初出率・重複・未到達・降下本数・群別の 4 点・対差と p）は
  [acceptance_topology.md](../../../docs/acceptance_topology.md) の
  **その165 の節「この軸の数値を消す前に移した表」**に全部ある（その166 が移した）。
- この回の集計は同じディレクトリの **`scored.txt`（読める本文）と `driver_summary.csv`（16 行）** に残してある。
- 消す前に `scored.txt` の小数 323 個を docs と機械照合した（`analysis/mmo2024/e168/number_audit.txt`）。

**残してあるもの**: `e151/descents.csv.gz`（12500、**D=20 の標準対照**。生きた読み手が 6 本ある）。

**`e165/analyze.py` の 2 つのガードの文言を直した** ——
1 つ目は「この回のダンプも読み手を失ったので その168 が消した」を足し、
2 つ目は **その166 の削除だと書いてあった帰属を その168 に直した**（その166 は `e165` のダンプを消していない）。
