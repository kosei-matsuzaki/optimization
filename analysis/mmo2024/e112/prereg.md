# その112 事前登録 — キュー 1「既存 run に F1 と Score を計算して併記する」

**claim: 2026-09-11 12:29 UTC。追加評価ゼロ（保存済み run の再採点のみ）。**
**この文書は計算を 1 つも走らせる前に書き、以後書き換えない。**

## 1. 指標の定義（出典に忠実に）

`external/mmo2024/docs/competition_setup_TR2024001.txt` §4 の原文:

> This score is the average of two indicators. The first one is the well-known peak ratio (PR) ...
> The second indicator is the F1 score: F1 = 2 × Precision × Recall / (Precision + Recall).
> ... recall ... is mathematically equal to PR. **Precision is the ratio of the detected global minima
> to the number of solutions reported/provided by the algorithm.**
> ... Five values for ϵf are considered: 1e-1 ... 1e-5. PR and static F1 are calculated for each problem
> with respect to each value of ϵf. The overall score of a method is the average of all these 2×5×960 numbers.

したがって run 単位・水準 ε ごとに:

- `recall(ε) = PR(ε) = detected(ε) / K`
- `precision(ε) = detected(ε) / n_reported`（**分子は PR の分子と同一**。§4 の "the detected global minima"）
- `F1(ε) = 2·P·R/(P+R)`、P=R=0 のときは 0
- `Score(ε) = (PR(ε) + F1(ε)) / 2`

**Score の平均は PR の平均と F1 の平均の平均に等しい**（Score は同じ 2×5×… 個の数の平均だから）。
この恒等式で**公表資料の D=10 Score を資料の MPR と mean-F1 から作れる**（下記 3 節）。

**分子の帰属規則**: `core/runner.py:_niching_counts` は報告集合を `max(100, 2K)` 点に f で trim したうえで
`count_goptima_nn`（最近傍帰属）で数える。**保存 CSV の `n_reported` は trim 後の実数**で、
**`pr_*` の分子はその同じ集合を数えたもの** ＝ **precision の分母と分子は同じ集合から来る。**
`detected(ε) = round(pr_ε × K)` で復元する（K は `n_optima` 列）。

## 2. 対象（追加 run ゼロ、すべて保存済み）

| 手法 | 出所 | seed |
|---|---|---|
| MC-ESO | `e106/baseline_d10_runs.csv`（rule=current） | 3 |
| NMMSO | `e109/nmmso_runs_d10.csv`（rule=current） | 2 |
| `Restart-Lander` | `e110/by_problem/*.csv` ＋ `e111/by_problem/*.csv` | 2 |

**集約の順序**: run → (seed 平均で) 問題 → 16 問平均。**水準は 5 つとも等重み。**

## 3. 突き合わせる公表値（D=10、`gecco2024_results_deck.txt` 68-80 行）

| | MPR | mean-F1 | **Score（= (MPR+F1)/2、この回で導出）** |
|---|---|---|---|
| RR-CMA-ES | 0.651 | 0.495 | **0.5730** |
| TRDE-LR（優勝） | 0.538 | 0.678 | **0.6080** |
| Niching CMA-ES | 0.078 | 0.020 | **0.0490** |

**資料の総合 Score（0.653 / 0.703 / 0.057）は全 D の平均なので、D=10 の列と同じ表に置かない。**
**公表値は 15 instance 平均、こちらは PIN01 のみ** ＝ 同一視しない（その92 以来の但し書き）。

## 4. 事前登録した棄却条件（キュー 1 の 2 枝を数値化する）

現状の MPR 側の位置づけ: **Δ_MPR = 0.651（MPR の公表最良）− 0.5478（`Restart-Lander` 2 seed）= 0.1032。**

- **枝 A（指標の取り違えは無かった）が発火する条件 —— 下の 2 つが両方成り立つとき:**
  1. **3 手法の Score 順位が MPR 順位と同じ**（`Restart-Lander` > NMMSO > MC-ESO）、かつ
  2. **Δ_Score = 0.6080（Score の公表最良 TRDE-LR）− `Restart-Lander` の Score ≤ 0.1532**
     （＝ Δ_MPR 0.1032 から **0.05 を超えては悪化しない**）。
  → status.md の主張 1 の弱点 (c) を消し、**2 番へ進む**。報告点数の軸は PR でも F1 でも空のまま。
- **枝 B（報告点数が効いている）**: 上のどちらかが崩れたとき。
  → **「null は公表最良に肉薄」は撤回**し、**報告集合を上位 K 点に切る腕を 2 番より先に置く。**

**この測定が何を反証しうるか**: 「20 サイクルの判定は公式指標の半分の上に載っていた ＝ 位置づけを取り違えている」
という status.md の懸念は、**枝 A が発火すれば反証される**（指標を公式のものに替えても位置は動かない）。
**逆に枝 B は、`mpr` 単独で書かれた過去の位置づけの記述を 1 行だけ書き直させる**（方針欄 09-11 (1)）。

## 5. 副次（判定には使わない。必ず併記する）

- **内訳を必ず出す**（方針欄 09-11 (1)「MPR と mean-F1 は内訳として必ず併記する」）: 3 手法 × (MPR, mean-F1, Score)。
- **対検定**: 16 問を対にした両側 Wilcoxon（`method="exact"` と既定の両方を書く。その111 の教訓）＋ rank-biserial。
  `Restart-Lander` 対 MC-ESO、`Restart-Lander` 対 NMMSO、MC-ESO 対 NMMSO を **Score の上で**。
- **n_reported の実数**を手法ごとに出す（上限値ではなく run ごとの実数。キュー 1 の注意 3 点目）。
- **水準別**（1e-1 … 1e-5）の 3 手法 PR / F1 を出す ——
  precision は水準に依らず分母が同じなので、**F1 の落ち方は水準で違うはず**。

## 6. 恒等検査（これが落ちたら副次の計算は捨てる）

`Restart-Lander` は `e110/descents/*.csv.gz` / `e111/descents/*.csv.gz` に
**降下ごとの `best_f` と `land_opt`（着地した最適の番号）**が残っている。
**`detected(ε) = |{land_opt : best_f ≤ ε}|`（trim 後）が `by_problem/*.csv` の `pr_ε × K` と
32 run × 5 水準すべてで一致するか**を確かめる。
**一致すれば**、報告集合を切る腕（枝 B の次の一手）が**追加評価ゼロで同じサイクル内に計算できる。**
**一致しなければ、ダンプは計器として使わずここで止める**（主結果は保存 CSV だけで出る）。

## 7. コスト

**最適化 run ゼロ。** CSV の再採点のみ。想定壁時計 5 分未満。
