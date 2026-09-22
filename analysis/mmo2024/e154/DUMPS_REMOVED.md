# その156 の統合で削除したもの（2026-09-22）

- **`caps_by_problem.csv.gz`（720 行 ＝ 5 セル × 9 水準 × 16 問の問題別内訳）を削除した。** 報告規則の路線は その154 で閉じている。

**消す前に再生成で検算した**: `python3 analysis/mmo2024/e154/analyze.py` を叩いて出た 721 行が、削除する版と**全行完全一致**（不一致 0）。
**＝ 入力 4 セル（`e115/descents/` `e151/descents.csv.gz` `e152/dumps_rrcma.csv.gz` `e153/*.csv.gz`）が生きているかぎり、約 90 秒で戻る。**
**入力が欠けたときは `analyze.py` が理由を印字して exit 1 する**（黙って `nan` の表は出ない）。

**数値の正本**は [../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その154 の節（判定 (i) の 5 セル × 7 水準の表、判定 (ii) の当落表、§4 の天井）。
