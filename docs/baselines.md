# 比較手法（ベースライン）

MC-ESO と比較する既存最適化手法の一覧と実装詳細。提案手法 MC-ESO は [mceso.md](mceso.md)、ベンチマーク・評価基準は [experiments.md](experiments.md) を参照。

実装は `core/optimizers/`（1 ファイル 1 手法、`BaseOptimizer` 継承、`__init__.py` で再エクスポート）。

---

## 手法一覧

| 手法 | 分類 | 本実験での位置づけ |
|---|---|---|
| **MC-ESO** (Multi-Channel Epidemic Spread Optimizer) | 群知能・独自提案 | **提案手法**（[mceso.md](mceso.md)）|
| CMA-ES | 進化戦略 | ベースライン（強力な標準手法） |
| IPOP-CMA-ES | 進化戦略 + restart | ベースライン（Auger & Hansen 2005、λ 倍化リスタート） |
| BIPOP-CMA-ES | 進化戦略 + restart | ベースライン（Hansen 2009、大小 2 regime 交互リスタート） |
| PSO | 群知能 | ベースライン |
| DE | 進化的アルゴリズム | 直接比較対象（MC-ESO の飛沫チャネルが借用する差分変異の本家・単一機構版） |
| L-SHADE | 適応的 DE | ベースライン（Tanabe & Fukunaga 2014、CEC2014 チャンピオン） |
| SaVOA | ウイルス模倣・既存 | 直接比較対象（同じ生物模倣着想だが単一再生メカニズム） |
| NM-Restart | multistart 局所探索 | **下限ベースライン**（低次元 BBOB では restart 局所探索が非常に強い＝メタヒューリスティクスの意義を示すための基準線） |
| NCDE | niching DE | **多解比較対象**（Qu+ 2012。PR / MMOsr で MC-ESO の逐次 niching と比較する専門手法） |
| **Crowding-DE** | crowding DE | 多解比較対象（Thomsen 2004。NCDE から近傍変異だけを外した対照）|
| **r3pso** | ring-topology lbest PSO | 多解比較対象（Li 2010。**niche 半径を持たない** niching の古典）|
| **NMMSO** | 多スウォーム niching | 多解比較対象（Fieldsend 2014、`pynmmso` 経由。**公式実装で動く競技上位級**）|
| **Repel-CMA-ES** | 斥力付き restart ES | 多解比較対象（de Nobel+ 2024 の近似実装。MC-ESO の情報化リスタートの先行例）|

---

## CMA-ES

共分散行列適応進化戦略。現在の探索分布の「形」を共分散行列として学習し、楕円形の探索が可能。収束を検出したら最良点からタイトな sigma で再スタートするマルチスタートを実装済み。実装は `core/optimizers/cmaes.py`。

| パラメータ | 値 | 意味 |
|---|---|---|
| `sigma0` | `0.2 × (hi - lo)` | 初期探索範囲（`main.py` が問題ごとに付与。クラス既定値は 1.0）|
| マルチスタート | 有効 | 収束後、最良点から再起動 |

---

## PSO

慣性重み付き PSO（Kennedy & Eberhart, 1995）。各粒子が自身の最良点と群の最良点に引き寄せられながら速度を更新する。実装は `core/optimizers/pso.py`。

| パラメータ | 値 |
|---|---|
| `n_particles` | 30 |
| `w`（慣性重み） | 0.729 |
| `c1`, `c2`（認知・社会係数） | 1.494 |

---

## DE（直接比較対象）

差分進化 / `DE/rand/1/bin`（Storn & Price, 1997）の古典版。各世代、集団内の target ごとに 3 つの異なる donor `a, b, c` を一様乱数で選び、変異ベクトル `v = x_a + F·(x_b − x_c)` を生成。二項交叉（rate `CR`、少なくとも 1 次元は `v` から継承）で trial `u` を作り、`f(u) ≤ f(target)` なら置換、というシンプルな単一機構。MC-ESO の飛沫チャネルと差分変異を共有するため「差分変異単独でどこまで行けるか」のベースラインとして直接置く（[mceso.md の DE との関係](mceso.md#de-との関係個別)参照）。実装は `core/optimizers/de.py`。

| パラメータ | 値 | 意味 |
|---|---|---|
| `n_pop` | 30 | 集団サイズ（PSO と揃える） |
| `F` | 0.5 | 差分スケール |
| `CR` | 0.9 | 二項交叉率 |

---

## SaVOA（既存ウイルス手法・直接比較対象）

VOA の自己適応版（Liang & Juarez, 2020 近似実装）。sigma を世代ごとに乗法的に適応（改善 → σ×1.2、停滞 → σ×0.9）することで、手動チューニング不要にしたもの。同じ生物模倣着想だが再生メカニズムは単一。実装は `core/optimizers/savoa.py`。

---

## NM-Restart（multistart 局所探索・下限ベースライン）

restart 付き Nelder-Mead simplex（Nelder & Mead, 1965）。一様ランダム初期点から scipy の bounded Nelder-Mead を tight な収束条件（`xatol=1e-12` / `fatol=1e-14`、SR@1e-10 閾値より十分深い）まで走らせ、予算が尽きるまで再スタートを繰り返す。restart 間の情報引き継ぎは一切なし。

**位置づけ**: 2〜3 次元の BBOB では multistart 局所探索が極めて強い（COCO の公開データでも既知）ため、「凝った手法を使わずとも解ける問題設定ではないか」という査読上の問いに答える下限ベースライン。提案手法の機構が意味を持つには、最低限この基準線を超える必要がある。盲目 restart が自然に複数 basin を拾うため、多解指標（PR）でも強い比較相手になる。実装は `core/optimizers/nelder_mead.py`。

| パラメータ | 値 | 意味 |
|---|---|---|
| `xatol` / `fatol` | 1e-12 / 1e-14 | 1 restart あたりの収束判定（1e-10 到達を妨げない深さ） |
| 予算管理 | ラッパーで厳密 | 評価カウンタが `max_evals` 到達で即停止（超過なし） |

---

## NCDE（niching DE・多解比較対象）

Neighborhood-based Crowding DE（Qu, Suganthan & Liang, 2012）。DE ベースラインと同一の trial 生成（rand/1/bin, 同じ `n_pop`/`F`/`CR`）に対し、niching のための 2 変更を加える:

1. **近傍変異** — donor `a, b, c` を全集団からではなく target の最近傍 `m` 個体から選ぶ。差分ベクトルが basin 内に収まり、niche ごとの局所収束が可能になる。
2. **crowding 置換** — trial は親ではなく**最近傍個体**と競合（即時置換・steady-state）。trial が自分の近傍しか置き換えられないため、別 basin の部分集団が共存する。

**位置づけ**: MC-ESO の逐次 niching（σ-exhaustion）に対する多解探索（PR / MMOsr）の専門比較手法。DE 系統を共有するため「並列 crowding niche vs 逐次 niching」という機構差が切り分けやすい。実装は `core/optimizers/ncde.py`。

| パラメータ | 値 | 意味 |
|---|---|---|
| `n_pop` / `F` / `CR` | 30 / 0.5 / 0.9 | DE ベースラインと同一 |
| `m` | 6 | 近傍変異の近傍サイズ。`m ≥ n_pop−1` で素の crowding DE（Thomsen 2004）に戻るが、素の crowding は donor が basin をまたぎ deep 精度が出ない（Himmelblau PR@1e-4 が 0% vs m=6 で 80%+）ため近傍変異版を採用 |

---

## Crowding-DE / r3pso / NMMSO / Repel-CMA-ES（多峰スイート用）

多峰スイート用に追加した 4 手法。うち r3pso / NMMSO / Repel-CMA-ES が `--suite niching` の既定に入り、Crowding-DE は NCDE の ablation なので `--methods` で明示したときだけ回る。選定理由と見送った手法（RS-CMSA / HillVallEA / MOMMOP 等）は [related_work.md](related_work.md) を参照。

**既定の 7 手法**は MC-ESO / NM-Restart / IPOP-CMA-ES / Repel-CMA-ES / NCDE / r3pso / NMMSO。1 行 = 答える問い 1 つで選んであり、より高次元の単一解 black-box 向け手法（CMA-ES 単体・PSO・DE・L-SHADE・SaVOA）は既知の理由で多解に弱いので回さない。BIPOP-CMA-ES も restart ES の枠が IPOP と二重になるため既定から外した（Repel-CMA-ES の対照は IPOP）。浮いた計算は予算軸（低予算 × 多解）に回す。

| 手法 | 実装 | 主要パラメータ | 位置づけ |
|---|---|---|---|
| **Crowding-DE** | `ncde.py`（`m = n_pop`）| n_pop 30 / F 0.5 / CR 0.9 | 素の crowding DE。NCDE との差 = 近傍変異の寄与。ドナーが basin をまたぐため深精度が落ちる（実測: N04 で PR@1e-4 0.00 vs NCDE 0.25, 2000 評価）|
| **r3pso** | `r3pso.py` | n_particles 30 / w 0.729 / c1=c2 1.494 / ring 3 | 慣性・加速係数を PSO ベースラインと完全に揃えてあるので、PSO との差は**近傍トポロジのみ**。MC-ESO の系統共存（半径依存）に対する「半径なし niching」の対照 |
| **NMMSO** | `nmmso.py`（`pynmmso` ラッパ）| swarm_size 10 | スウォームの分裂・併合でニッチ数を自分で決める。再実装でないので「ベースラインの実装が悪い」という反論を封じられる |
| **Repel-CMA-ES** | `restart_cmaes.py:RepellingCMAESOptimizer` | repel_coverage 0.2 / repel_gamma 0.9 | restart の best を taboo 点にし、その球内に落ちた候補を引き直す。半径は「taboo 集合が箱の `repel_coverage` を塞ぐ」体積条件から決まり、restart が増えるほど自動で縮む |

実装上の注意:

- **NMMSO は最大化**なので符号を反転して渡す。`Nmmso.run` は反復の切れ目でしか予算を見ずオーバーランするため、`max_evals` に達した後の `fitness` は**関数を呼ばずに `-inf` を返す**。評価回数は厳密に一致し、偽の点がモードとして報告されることもない。
- **Repel-CMA-ES は de Nobel+ 2024 の近似**。棄却判定を Euclid 距離で行っている（原論文は現在の CMA 計量での Mahalanobis 距離 / σ）。`repel_coverage` も本プロジェクトの選択で、斥力の強さを決める唯一のパラメータなので、これに依存する主張をする前に感度を測ること。
- 多解指標は `final_solutions` だけを見る（[experiments.md](experiments.md#多解報告cec2013-ルール-niching-スイート)）。報告するのは r3pso が全粒子の pbest、NMMSO がモード集合、Repel-CMA-ES が各 restart の best ＋最終集団、Crowding-DE / NCDE が最終集団。
- 低予算では集団サイズが効く。NCDE / Crowding-DE / r3pso の既定 `n_pop=30` は 2D・5000 評価で 166 世代しか回らないので、負けが手法のせいか設定のせいかは予算を変えて確かめる必要がある。

---

## L-SHADE / IPOP-CMA-ES / BIPOP-CMA-ES（外部ライブラリ baseline）

より強力な近代手法を外部ライブラリ経由で `BaseOptimizer` インターフェースに合わせて組み込む。

- **L-SHADE**（`core/optimizers/lshade.py`, mealpy ラッパー）— SHADE に線形集団縮小を加えた適応的 DE（Tanabe & Fukunaga 2014、CEC2014 優勝）。初期集団 `N_init = 18 × d`、`miu_f = miu_cr = 0.5`。
- **IPOP-CMA-ES**（`core/optimizers/restart_cmaes.py`, pycma ラッパー）— 収束ごとに集団サイズ λ を倍化して再起動（Auger & Hansen 2005）。
- **BIPOP-CMA-ES**（同上）— 大規模・小規模 2 つの λ regime を予算が釣り合うよう交互に再起動（Hansen 2009）。
- IPOP/BIPOP も CMA-ES と同じく `sigma0 = 0.2 × span` を `main.py` が付与する。

> **注意（再現性）**: pycma 系（CMA-ES seed0 / IPOP / BIPOP）は同一 seed でも run ごとに結果が変動しうる。Wilcoxon 等の seed-paired 比較ではこの非決定性を念頭に置く。
