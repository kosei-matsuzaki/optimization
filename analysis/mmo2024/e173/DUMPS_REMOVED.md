# その173 の生データについて

**【2026-09-27 その174 の統合で削除】行単位ダンプ 1 本（40,579 バイト）を消した。**

* `null_topup.csv.gz` —— 3 問（M01 / M13 / M16）の draw 200-599 の上積み（400 本 × 3 問）

**理由**: **この回自身が軸を閉じた** —— **`mpr_sup` は上界ではなく台の個数の plug-in 推定 ＝ 真の台の下界**で、
**600 抽選でも飽和しない**（3 問で 200 → 600 本にすると平均 +0.113）。
**＝ 抽選を増やせば必ず上がるだけなので、どの値が出ても「一族は届かない」を支えない**
（[../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その173 / その174 の節。
キューの「旧 1 番（残り 10 問を 600 抽選に上積みする）は削除した。やり直さないこと」も同じ理由）。

**消していない**: `ceiling600.csv`、`support.csv`、`support.txt`、`scored.txt`、`null_runs.log`、`prereg.md`、`run.sh`。
**3 問の 200 / 400 / 600 本の値と水準別の台の検査は
[../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その173 の節に全部ある。**

**`analyze.py` と `support.py` は入力（自分の `null_topup.csv.gz` と `e172/null_rl.csv.gz`）を失っている**
（理由を印字して exit 1 する形に直した）。
**再測定は `e172/run.sh`（200 抽選）＋ `e173/run.sh`（400 本の上積み）。種は `1_000_000 + k` 固定。**
