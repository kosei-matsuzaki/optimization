# その153 の生データについて

**行単位のダンプ 32 本は 2 本の `.csv.gz`（`problem` 列つき）に畳んである。**

* `descents.csv.gz` —— `Restart-Lander` の降下ダンプ 16 本（1360 行）
* `dumps_rrcma.csv.gz` —— RR-CMA-ES の報告集合ダンプ 16 本（420 行）

畳む前後で `analyze.py` の出力（`scored.txt`）が **1 文字も変わらないことを確認**してから
元の 32 本を消した（その128〜その152 と同じ手順。`fold.py` と `rm` は別コマンドで打った ＝ その131 の事故の再発防止）。
**`by_problem/` の driver CSV 16 本とログ 16 本も消した**（集計は `by_problem_d5.csv` にある）。

**再現**: `PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e153/analyze.py`
（この回の 2 本の `.csv.gz` と、`e115/descents/`（D=10 RL）・`e151/descents.csv.gz`・`e152/dumps_rrcma.csv.gz` を読む）。
**数値はすべて [../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その153 の節に表としてある。**
