# その153 の生データについて

**行単位のダンプ 32 本は 2 本の `.csv.gz`（`problem` 列つき）に畳んである。**

* `descents.csv.gz` —— `Restart-Lander` の降下ダンプ 16 本（1360 行）
* `dumps_rrcma.csv.gz` —— RR-CMA-ES の報告集合ダンプ 16 本（420 行）

畳む前後で `analyze.py` の出力（`scored.txt`）が **1 文字も変わらないことを確認**してから
元の 32 本を消した（その128〜その152 と同じ手順。`fold.py` と `rm` は別コマンドで打った ＝ その131 の事故の再発防止）。
**`by_problem/` の driver CSV 16 本とログ 16 本も消した**（集計は `by_problem_d5.csv` にある）。

**【2026-09-24 その162 の統合で追加削除】`dumps_rrcma.csv.gz`（20,889 バイト）を消した。**
理由: **測り方 5 軸が その160 で全部閉じた**ので、RR-CMA-ES の報告集合ダンプを読む生きた問いはもう無い。
消す前に `analyze.py` を実走して `scored.txt` がバイト一致で再生成されることを確認し、
**`scored.txt` の小数 178 個が全部 docs にあることを機械照合した**（漏れゼロ）。
**同じ統合で `e152/dumps_rrcma.csv.gz` も消えている**ので、**`analyze.py` は関門 2（D=20）でも入力を失っている。**
`_load_folded` は黙って空を返す形から **理由を印字して exit 1** する形に直した。

**再現**: `PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e153/analyze.py`
（`descents.csv.gz` と `e115/descents/`（D=10 RL）・`e151/descents.csv.gz` を読む。
**RR 側 2 本は削除済みなので、いまこのまま走らせると exit 1 する** —— 再測定は `run.sh`）。
**数値はすべて [../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その153 の節に表としてある。**
