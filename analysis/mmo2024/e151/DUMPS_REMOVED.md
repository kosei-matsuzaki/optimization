# その151 の生データについて

**行単位の降下ダンプ 16 本（`descents/M??-D20-PIN01_seed0.csv`、1883 行）は
`descents.csv.gz`（`problem` 列つき 1 本）に畳んである。** 畳む前後で `analyze.py` の
出力が **1 文字も変わらないことを確認**してから元の 16 本を消した（その128〜その149 と同じ手順）。

**`by_problem/` の driver CSV 16 本とログ 16 本も消した。**
集計 16 行は `driver_summary.csv` にある。

**再現**: `PYTHONPATH=/tmp/pystub python3 analysis/mmo2024/e151/analyze.py`
（`descents.csv.gz` と `e115/descents/`（D=10 の対照、保存物）を読む）。
**数値はすべて [../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その151 の節に表としてある。**
