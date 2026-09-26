# その121 の生データについて

**【2026-09-26 その170 で削除】行単位ダンプ 16 本（36,438 バイト）を消した。**

* `by_problem/*.csv.gz` —— 事前登録の掃引 8 本（1320 降下、25,082 バイト）
* `by_problem_far/*.csv.gz` —— 事後の遠方掃引 8 本（440 降下、11,356 バイト）

**理由**: **路線（その121 の盆地半径）は その125 で閉じている**（中心も半径も自己情報に replace すると
上限の符号すら再現しない）。**読み手は `e121` 自身の 4 本だけ**で、その168 が 3 本を実走して
バイト一致を確認したうえで「次に畳める路線」として名指ししていた（[../../../docs/research_loop.md](../../../docs/research_loop.md)）。

**消す前に `scored.txt` / `scored_far.txt` / `scored_scaled.txt` の小数を `e168/number_audit.py` で
機械照合し、docs に無かった 31 個（15 / 14 / 2）を
[../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その121 の節に写した。**
＝ **集計 3 本は残してあるが、数値はもう docs 側にもある。**

**したがって `analyze.py` / `analyze_far.py` / `analyze_scaled.py` は入力を失っている**
（黙って空を返さず、**理由を印字して exit 1** する形に直した。その162・その169 と同じ扱い）。
**再測定は `basin_radius.py --pid M01`**（`--radii` / `--conds` で掃引を替えられる。
seed 20260913 固定なので、ダンプを作り直せば上の数値はバイト再現する）。
