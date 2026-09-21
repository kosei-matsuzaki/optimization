# この路線のデータは 2026-09-21 その150（統合の回）で畳んだ

**路線**: `descent_budget` / `sigma_ratio` ＝ 「勝っている側（記憶なし多スタート null）の 2 つのつまみの監査」。
**その146・その147・その148 で 5 水準 ＋ 3 水準を振り終わり、[status.md](../../../docs/status.md) が
「この軸にはもう 1 サイクルも使わない」と確定させた** ＝ 畳んだ路線。

**消したもの**（その150）:

| 消した物 | 中身 | どこに残っているか |
|---|---|---|
| `e145/` 丸ごと（8 ファイル・462 KB） | `descent_budget=1540` の腕 ＋ MC-ESO / null の対照 | **per-problem 16 行は [acceptance_topology.md](../../../docs/acceptance_topology.md) の その145 の節の表**。集計は本ディレクトリの `by_problem_e145_arms.csv`（48 行） |
| `e146/` 丸ごと（9 ファイル・355 KB） | `descent_budget` 3000 / 6000 | **`by_problem.csv`（80 行）が 0 行の欠けも値のずれもなく e146 の 64 行を含む**（その150 が機械照合）。per-problem はその146 の節の表 |
| `e147/` 丸ごと（9 ファイル・230 KB） | `sigma_ratio` 0.05 / 0.2 | **per-problem 16 行はその147 の節の表**。集計は本ディレクトリの `by_problem_e147_sigma.csv`（48 行） |
| `e148/descents.csv.gz`（288 KB） | この回の行単位の降下ダンプ | **数値はその148 の節の表（5 水準 × 16 問）と `by_problem.csv`** |
| `e148/analyze.py` / `fold.py` | 採点器と畳み器 | **git 履歴**（`git show <この commit>^:analysis/mmo2024/e148/analyze.py`） |

**【重要】この路線の「再現は `analyze.py`」はもう成り立たない。** 採点器ごと消してあるので、
**中途半端に動いて NaN の表を出すことはない**（＝ その150 が `e115/analyze.py` で踏んだ罠をここには残していない）。
**再測定するなら `run.sh` と `prereg.md` から組み直すこと。**
