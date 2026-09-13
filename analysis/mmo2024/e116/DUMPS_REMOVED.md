# `e116/dumps/` は その120（統合の回）で削除した（報告軸が閉じたため）

MC-ESO / NMMSO の**報告集合ダンプ 32 本**（`f,x0..x9`、計 6692 行）を削除した。

- **路線は閉じている**: その115・その116 が **3 手法とも合法な規則 = oracle 上限**（0/0/16、p=1）を出し、
  その116 自身が「この軸にはもう 1 サイクルも使わない」と書いている。
  俯瞰も旧・キュー 1（報告軸）を削除済み。
- **数値は残っている**: 集計は `ranking_d10.csv` と `by_problem.csv`、
  結論は [acceptance_topology.md の その116 の節](../../../docs/acceptance_topology.md)。
- **採点の手続きも残っている**: `analyze.py` / `run.sh` / `prereg.md`（キュー 2 が `analyze.py` を名指し）。
- **実体が要るとき**: `git show <この commit の親>:analysis/mmo2024/e116/dumps/<file>`、
  または `run.sh` で 2 手法を回し直す（16 問 × 2 手法 ＝ 約 43 分）。
