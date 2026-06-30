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

## L-SHADE / IPOP-CMA-ES / BIPOP-CMA-ES（外部ライブラリ baseline）

より強力な近代手法を外部ライブラリ経由で `BaseOptimizer` インターフェースに合わせて組み込む。

- **L-SHADE**（`core/optimizers/lshade.py`, mealpy ラッパー）— SHADE に線形集団縮小を加えた適応的 DE（Tanabe & Fukunaga 2014、CEC2014 優勝）。初期集団 `N_init = 18 × d`、`miu_f = miu_cr = 0.5`。
- **IPOP-CMA-ES**（`core/optimizers/restart_cmaes.py`, pycma ラッパー）— 収束ごとに集団サイズ λ を倍化して再起動（Auger & Hansen 2005）。
- **BIPOP-CMA-ES**（同上）— 大規模・小規模 2 つの λ regime を予算が釣り合うよう交互に再起動（Hansen 2009）。
- IPOP/BIPOP も CMA-ES と同じく `sigma0 = 0.2 × span` を `main.py` が付与する。

> **注意（再現性）**: pycma 系（CMA-ES seed0 / IPOP / BIPOP）は同一 seed でも run ごとに結果が変動しうる。Wilcoxon 等の seed-paired 比較ではこの非決定性を念頭に置く。
