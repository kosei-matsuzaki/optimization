# その172 の生データについて

**【2026-09-27 その174 の統合で削除】行単位ダンプ 1 本（87,164 バイト）を消した。**

* `null_rl.csv.gz` —— 新 suite D=10 の 13 問 × 200 抽選（σ₀ = 0.1·span ＝ `Restart-Lander` の既定、降下上限 12500 評価）

**理由**: その171 と同じ —— **上限表を引き直す軸は その171〜その173 で閉じた**
（測っているのは台の個数の下界推定なので、設定をどう揃えても「届かない」を支えられない。
[../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その173 / その174 の節）。

**消していない**: `ceiling_sigma_rl.csv`（13 問 × 3 推定量）、`scored.txt`、`base_repro_M06.log`、`null_runs.log`、`prereg.md`、`run.sh`。
**13 問の値は [../../../docs/acceptance_topology.md](../../../docs/acceptance_topology.md) の その172 の節に全部ある。**

**読み手は 3 本あった**: `e172/analyze.py`、`e173/analyze.py`、`e173/support.py`。
**3 本とも理由を印字して exit 1 する形に直した。**
**再測定は `run.sh`（抽選の種は `1_000_000 + k` 固定なのでビット再現する）。**
