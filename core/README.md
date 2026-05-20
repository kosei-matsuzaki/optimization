# core — 最適化手法とベンチマーク

提案手法 **MC-ESO** と既存手法、および評価に用いるベンチマーク関数を実装するパッケージ。手法は `core/optimizers/`（1ファイル1手法）、関数は `core/benchmarks.py` にある。プロジェクト全体の概要・実行方法・結果の見方は [../README.md](../README.md) を参照。

**読む順序**: [ディレクトリ構成](#ディレクトリ構成) → [ベンチマーク関数](#ベンチマーク関数) → [手法一覧](#手法一覧) → [提案手法 MC-ESO](#提案手法mc-eso) → [既存手法との差別化](#mc-eso-の新規性既存手法との差別化) → [ベースライン手法](#ベースライン手法) → [評価方法論](#評価方法論) → [開発メモ](#開発メモablation-記録)。

---

## ディレクトリ構成

```
core/
├── __init__.py             # 主要クラス/関数の公開API再エクスポート（visualize除く）
├── benchmarks.py           # BBOB 24関数 + カスタム + CEC2022（ioh 経由、2D/3D/4D/10D）
├── optimizers/             # 手法ごと1ファイル（__init__.py で全クラス再エクスポート）
│   ├── base.py             # OptimizeResult, BaseOptimizer
│   ├── cmaes.py            # CMA-ES（best-anchored restart）
│   ├── mceso.py            # MC-ESO（提案手法）+ _MCESOState
│   ├── pso.py / de.py / savoa.py  # PSO・DE・SaVOA baseline
│   ├── lshade.py           # L-SHADE（mealpy wrapper）
│   └── restart_cmaes.py    # IPOP/BIPOP-CMA-ES（pycma wrapper）
├── runner.py               # 複数run の実験実行・統計サマリー
└── visualize.py            # 関数地形図・収束曲線・各種 GIF の生成
```

---

## ベンチマーク関数

### BBOB 24 関数（主スイート）

**BBOB（Black-Box Optimization Benchmarking）ノイズなし版全24関数**を使用する。
BBOB は Hansen et al. (2009) が提案した連続最適化の標準ベンチマークスイートであり、GECCO の COCO ワークショップで毎年使用されている。関数は `ioh` ライブラリ（instance=1）経由で取得し、`f(x) − f_opt` に正規化することでグローバル最小値を常に0とする。探索範囲はすべての関数で **[-5, 5]^d**。

> **なぜ BBOB か**
> - 手作りの個別関数ではなく、査読済みスイートによる客観的な比較が可能
> - 5つの難易度グループが問題の特性を体系的にカバー（分離可能・条件数・多峰性・弱構造）
> - インスタンス変換（シフト・回転）が適用されており、座標軸や原点への過適合を防ぐ
> - 既発表の CMA-ES, PSO, GA 等の結果と直接比較できる

| FID | 関数名 | グループ | 主な難しさ |
|---|---|---|---|
| F01 | Sphere | separable | 最も単純。アルゴリズムの健全性確認 |
| F02 | Ellipsoidal (sep.) | separable | 軸方向に強い条件数 |
| F03 | Rastrigin (sep.) | separable | 分離可能な多峰性 |
| F04 | Büche-Rastrigin | separable | 非対称な多峰性 |
| F05 | Linear Slope | separable | 最適解が境界上 |
| F06 | Attractive Sector | moderate-cond | 非対称な単峰性 |
| F07 | Step Ellipsoidal | moderate-cond | 段差状の不連続性 |
| F08 | Rosenbrock | moderate-cond | バナナ型の曲がった谷 |
| F09 | Rosenbrock (rot.) | moderate-cond | Rosenbrock に回転を適用 |
| F10 | Ellipsoidal (rot.) | ill-cond | 高条件数、軸非整合 |
| F11 | Discus | ill-cond | 1次元のみ強く伸びた形状 |
| F12 | Bent Cigar | ill-cond | 曲がった葉巻型 |
| F13 | Sharp Ridge | ill-cond | 鋭い稜線 |
| F14 | Different Powers | ill-cond | 次元ごとに異なるべき乗 |
| F15 | Rastrigin (rot.) | multimodal | 局所解が密、回転あり |
| F16 | Weierstrass | multimodal | 高度に多峰・不規則 |
| F17 | Schaffer F7 | multimodal | 中程度の多峰性 |
| F18 | Schaffer F7 (ill) | multimodal | F17 に高条件数を追加 |
| F19 | Griewank-Rosenbrock | multimodal | 複合的な地形 |
| F20 | Schwefel | weak-structure | 大域構造が弱い多峰性 |
| F21 | Gallagher 101 peaks | weak-structure | 101 個のガウス峰が散在 |
| F22 | Gallagher 21 peaks | weak-structure | F21 より峰が少なく深い |
| F23 | Katsuura | weak-structure | フラクタル的な地形 |
| F24 | Lunacek bi-Rastrigin | weak-structure | 大域最適解が欺瞞的な位置 |

### Custom 関数 (2-D)

BBOB がカバーしない **多大域最適解**・**deceptive 2-D 多峰** 系の古典的テスト関数を補完。MC-ESO の「ニッチ系統共存」と「広域 spillover」の挙動を BBOB の回転・シフトに依らない素のランドスケープで検証する目的。各関数は `f(x) − f_opt` で正規化し最小値を 0 とする。

| ID | 関数名 | 探索域 | カテゴリ | 主な難しさ |
|---|---|---|---|---|
| C01 | Himmelblau | [-5, 5]² | multi-optima | 大域最適解が **4 箇所**（ニッチ性能の直接評価） |
| C02 | Six-hump Camel | [-2, 2]² | multi-optima | 大域最適解が **2 箇所** |
| C03 | Shubert | [-10, 10]² | multi-optima | 大域最適解が **18 箇所**（積形式・約760 局所解） |
| C04 | Five-well Potential | [-20, 20]² | deceptive-2d | 5 つの井戸（うち1つが大域）|
| C05 | Eggholder | [-512, 512]² | deceptive-2d | 極めて鋭い多峰・大域は境界近傍 |
| C06 | Michalewicz (m=10) | [0, π]² | deceptive-2d | 平坦域に細い谷、急峻 |
| C07 | Bukin N.6 | [-15, 15]² | deceptive-2d | y = 0.01x² の極細谷、gradient 不連続 |
| C08 | Styblinski-Tang | [-5, 5]² | deceptive-2d | 4 局所解、3 つが大域に近い深さ |
| C09 | Easom | [-100, 100]² | deceptive-2d | 広大な平坦域中の鋭い単一峰（needle-in-haystack）|
| C10 | Schaffer N.2 | [-100, 100]² | deceptive-2d | 同心円状の多峰、原点中心 |
| C11 | De Jong F5 (Shekel's foxholes) | [-65.536, 65.536]² | deceptive-2d | 5×5 格子の25局所解 |

### CEC2022（hold-out）

BBOB とは独立した CEC2022 12 関数（`ioh` 経由、dim=10）を hold-out スイートとして用意。BBOB の変換に対して開発された MC-ESO の機構が汎化するかの検証に使い、**CEC2022 用にハイパーパラメータを再調整しない**。

出典: BBOB は Hansen et al. (2009)。Custom 関数は Surjanovic & Bingham のテスト関数集および Tomitomi3 (Qiita) の整理を参照。

---

## 手法一覧

| 手法 | 分類 | 本実験での位置づけ |
|---|---|---|
| **MC-ESO** (Multi-Channel Epidemic Spread Optimizer) | 群知能・独自提案 | **提案手法** |
| CMA-ES | 進化戦略 | ベースライン（強力な標準手法） |
| IPOP-CMA-ES | 進化戦略 + restart | ベースライン（Auger & Hansen 2005、λ 倍化リスタート） |
| BIPOP-CMA-ES | 進化戦略 + restart | ベースライン（Hansen 2009、大小 2 regime 交互リスタート） |
| PSO | 群知能 | ベースライン |
| DE | 進化的アルゴリズム | 直接比較対象（MC-ESO の飛沫チャネルが借用する差分変異の本家・単一機構版） |
| L-SHADE | 適応的 DE | ベースライン（Tanabe & Fukunaga 2014、CEC2014 チャンピオン） |
| SaVOA | ウイルス模倣・既存 | 直接比較対象（同じ生物模倣着想だが単一再生メカニズム） |

提案手法 MC-ESO の詳細は次節、各ベースラインの実装詳細は[ベースライン手法](#ベースライン手法)を参照。

---

## 提案手法：MC-ESO

**Multi-Channel Epidemic Spread Optimizer** — 感染症の流行を「感染宿主が空間上に分布し、複数の伝染経路から新規宿主を生む」過程としてモデル化した群知能最適化手法。

### 着想

「**f(x) が低い領域 = 感染宿主が密集する場所**」とみなす。感染症は宿主密度の高い場所ほど広まりやすいため、f 値が低い個体ほど高い確率で次の親（感染源）に選ばれる。現実の感染症が単一経路ではなく複数経路（接触・飛沫・空気）で同時に拡散することを、複数の探索オペレータの並行適用として最適化に転写する。

### 探索オペレータ：3 つの伝染チャネル

毎世代、空きスロットを 3 チャネルに配分して子個体を生成する（割合は `air_ratio` / `h2h_ratio` と残り）。

| チャネル | 疫学アナロジー | 数学的形 | 役割 |
|---|---|---|---|
| **接触感染** (Close-contact) | 親密接触による局所感染 | `x_parent + σ_i · L_pop · N(0, I)`（`L_pop L_pop^T = C_pop`、集団経験共分散の固有分解）| 親近傍の精密探索。σ_i は親の品質と年齢で適応、`C_pop` で **瞬間共分散** を獲得（履歴累積なし、CMA-ES と差別化） |
| **飛沫感染** (Droplet) | 飛沫を介した宿主間感染 | `x_parent + F·(x_strain − x_parent) + F·(x_a − x_b)` ＋ 親との二項交叉 (CR=0.9, DE 標準値) | 系統 (niched elite) からの引力 ＋ 集団内差分ベクトルで集団形状を補強しつつ、二項交叉で座標方向の親情報を保護 (DE/current-to-best/1/bin) |
| **空気感染** (Airborne) | エアロゾルによる広域感染 | `x_random_host + N(0, σ_air I)`（drilling 中は停止）| 集団に依存しない遠方探索。局所最適脱出。`σ < span × 1e-3` の drilling mode では雑音化するため停止 |

### 集団レベルの 3 機構

| 機構 | 疫学アナロジー | 役割 |
|---|---|---|
| **系統共存** (Strain coexistence) | 空間的に離れた感染拠点の同時存続 | ニッチ半径で離れた最大 6 系統を保護、飛沫チャネルの引力対象 pool |
| **宿主競合** (Host competition) | 新感染が既存宿主に勝てないと排除される | 毎世代 25% kill、子が親より悪ければ rollback → 集団は単調改善 |
| **スピルオーバー** (Spillover) | 既存系統の絶滅後、新宿主集団へ感染が飛び火 | 300 評価改善なし AND f_best / \|f_init\| > 1e-8 で best を保持し残りを全域 Uniform 再播種。連続失敗 2 回で basin switch (best 破棄＋σ_init リセット) に escalate |

> **なぜ系統共存が多峰問題に効くか**: 単純な top-k 選択では最初に見つかった最適解周辺に個体が集中し、Himmelblau（最適解4箇所）のような多最適解問題で致命的になる。系統選択は (1) f 値の良い順に走査し、(2) 既保護系統との距離が全て `niche_radius_ratio × span` を超える候補のみ追加、(3) `n_elite_max` 個で終了する。これにより空間的に離れた複数の最適解周辺に独立した感染系統が自然形成され、飛沫感染の引力対象となる。

### 1 世代の動作フロー

```
1. 系統共存: ニッチエリート抽出（飛沫感染の引力 pool）
   └─ f 値の良い順に走査し、既存系統から niche_radius_ratio × span (=0.1×span) 以上離れた個体だけを
      最大 n_elite_max (=6) 個保護

2. スピルオーバー判定（停滞時の集団再播種、basin 回避版）
   └─ no_improve ≥ restart_no_improve_threshold (=300) かつ
      f_best / |f_init| > restart_quality_rel_floor (=1e-8) のとき発動（相対 8 桁進捗未満なら spillover、それ以下なら precision とみなし保護）。
      連続失敗回数 (consecutive_failed_spillovers) で動作切替:
         • streak < 2: 100% Uniform(lo, hi)（best は保持）+ 軸 sweep + σ ← σ_init×0.3
         • streak ≥ 2 かつ f_best > 1e-2: **ベイスン乗換え**
              — best も破棄、全 n_pop を Uniform 再生成、σ を σ_init にリセット
      ─ ベイスン乗換えで F24 双漏斗 / F04 rugged separable から脱出
        f_best ≤ 1e-2 のとき乗換え抑制 → F13 ridge / C01 deep precision を保護

3. 宿主競合: 死亡判定（μ+λ greedy）
   └─ 集団の f 値降順で下位 kill_fraction (=25%) を排除。最良宿主は自動生存

4. 親（感染源）の選択（softmax）
   └─ w_i ∝ exp(f_max − f_i)、f が低い個体ほど高い感染力（softmax 温度は 1.0 固定）

5. 子個体の 3 チャネル生成（空きスロット数だけ）
   ├─ 接触感染 [残り]
   │   ├─ σ_i = σ × host_sigma_min_scale^(log_quality × (0.7 + 0.3 × age_ratio))
   │   ├─ C_pop = (1/(n-1)) Σ (x_i − x̄)(x_i − x̄)^T  ← 集団経験共分散
   │   ├─ 固有分解 V Λ V^T = C_pop、Λ を平均 1 に正規化（floor=0.01）
   │   └─ child = 親 + σ_i × V √Λ × Gauss(0, I)
   │       → 瞬間共分散による回転・異方性追従。F11/F14 で ill-cond 楕円体に整列
   ├─ 飛沫感染 [h2h_ratio = 0.4]
   │   └─ trial = 親 + h2h_F × (x_strain − 親) + h2h_F × (x_a − x_b)
   │       child = 各次元で確率 h2h_CR (=0.9, DE 標準値) で trial を採用、残りは親をそのまま継承
   │       DE/current-to-best/1/bin と同型: 差分ベクトル ＋ 系統引力で集団形状を反映、
   │       二項交叉が separable 多峰の座標方向情報を保護
   └─ 空気感染 [air_ratio_eff = 0.3 if σ ≥ span × 1e-3 else 0]
       └─ ランダム宿主 + Normal(0, σ_air I), σ_air は集団分散に応じて 1.5×〜5× σ
          drilling mode (σ < span × precision_sigma_ratio) では停止 — 雑音による精度妨害を排除

6. 宿主競合: Rollback
   └─ 各空きスロットの子が「元そこにいた親」より悪ければ親を復元 → 集団は単調改善

7. σ 適応（always-on + drilling mode）
   └─ 改善: σ × sigma_up (=1.1)
   └─ 非改善 (通常): σ × sigma_down (=0.95)
   └─ 非改善 (drilling): σ < span × precision_sigma_ratio (=1e-3) のとき
                       σ × sigma_drill_down (=0.85) — 浮動小数限界へ追込み
```

### 適応機構と設計判断

MC-ESO は明示的なフェーズ切替パラメータを持たず、**σ の大きさだけで探索 ↔ 精密化を自動で切り替える**。主要な 3 つの適応機構（いずれも常時 ON）と、ablation を経た採用理由:

- **σ 適応（always-on）** — 改善時 `× sigma_up (1.1)`、非改善時 `× sigma_down (0.95)`（SaVOA 流の乗法適応を毎世代適用）。以前は `no_improve` ゲートで停滞中の減衰を緩める fallback を持っていたが、HP 削減（`sigma_adapt_stagnation_gate`, `sigma_decay`）と引き換えに除去した。
- **Drilling mode** — `σ < span × precision_sigma_ratio (1e-3)` に入ると「σ が basin スケールまで収縮済み」と判定し、非改善時の縮小を `× sigma_drill_down (0.85)` に強化して浮動小数限界まで追い込む。同時に空気感染チャネルを停止（`air_ratio_eff = 0`）し広域ランダム雑音による精度劣化を防ぐ。σ ベース閾値なので shift / scaling されたベンチマークでも調整不要。「正しい basin 到達」を `best_so_far` で判定するため、deceptive landscape での誤発動リスクは小さい。
- **Spillover / basin switch** — `no_improve ≥ 300` かつ `f_best / |f_init| > 1e-8` で停滞と判定し、best を保持して残りを全域 Uniform 再播種する（＋境界軸 sweep: 各次元で `lo`/`hi` を試す probe、dim=2 で 4 評価）。連続失敗が `basin_switch_after_failed_spillovers (2)` 回に達し、かつ `f_best / |f_init| > basin_switch_quality_rel_floor (1e-2)` なら **basin switch**（best も破棄し σ を σ_init にリセット）に escalate。quality gate により高精度到達済みの run（F13 ridge, C01 deep precision）での暴発を防ぐ。IPOP-CMA-ES の "restart with larger population" の MC-ESO 版にあたる。

### パラメータ一覧

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `n_pop` | 20 | 集団個体数 |
| `sigma` | 0.2 | 初期探索半径（探索範囲に対する比率） |
| `host_sigma_min_scale` | 0.05 | 接触感染チャネルにおける per-host σ_i スケーリング下限（高品質・高齢の宿主は σ_i = σ × 0.05 まで縮小して精密探索）|
| `empirical_cov_floor` | 0.01 | 接触感染チャネルの集団経験共分散 `C_pop` の固有値下限（平均 1 正規化後、軸の縮退を防ぐ）|
| `air_ratio` | 0.3 | 空気感染チャネルの割合 |
| `air_sigma_amplifier` | 3.5 | 空気感染 σ 倍率の振幅（factor = 1.5 + amp × (1 - diversity)、集団分散時 1.5、収束時 1.5+amp） |
| `h2h_ratio` | 0.4 | 飛沫感染チャネルの割合 |
| `h2h_F` | 0.5 | 飛沫感染の差分ベクトルスケール係数 |
| `h2h_CR` | 0.9 | 飛沫感染後の二項交叉率（DE/bin 標準値、座標方向の親情報を確率 1-CR で継承）|
| `kill_fraction` | 0.25 | 宿主競合で毎世代排除する割合 |
| `restart_no_improve_threshold` | 300 | スピルオーバー発動の no_improve 閾値 |
| `restart_sigma_ratio` | 0.3 | スピルオーバー後の σ（σ_init に対する比率） |
| `restart_quality_rel_floor` | 1e-8 | スピルオーバー skip 閾値（best_so_far / \|f_init\| ≤ this で skip）。乗法スケール不変 |
| `basin_switch_after_failed_spillovers` | 2 | この連続失敗回数で best 破棄＋σ_init リセットの完全ベイスン乗換え |
| `basin_switch_quality_rel_floor` | 1e-2 | best_so_far / \|f_init\| ≤ this でベイスン乗換えを抑制（相対 2 桁以上進捗で grinding 中とみなし保護）|
| `n_elite_max` | 6 | 系統共存の最大数（飛沫感染の引力対象） |
| `niche_radius_ratio` | 0.1 | 系統間の最小距離（span に対する比率、スケール不変。BBOB span=10 で実効 1.0、絶対値版と数学的に同一） |
| `log_slope_threshold` | 1e-4 | 「意味ある改善」の log10(f) 減少スロープ閾値 |
| `sigma_up` | 1.1 | σ adapt 改善時の乗数 |
| `sigma_down` | 0.95 | σ adapt 改善なし時の乗数 |
| `sigma_floor_ratio` | 1e-6 | σ_global の絶対下限（× span）|
| `sigma_ceil_ratio` | 1.0 | σ_global の絶対上限（× span）|
| `precision_sigma_ratio` | 1e-3 | drilling mode 発動の σ 閾値（σ < span × 1e-3 で発動。σ ベースなので問題スケール不変）|
| `sigma_drill_down` | 0.85 | drilling mode 中の σ 縮小乗数（通常の sigma_down より積極的）|

---

## MC-ESO の新規性（既存手法との差別化）

### CMA-ES との比較

| 観点 | CMA-ES | MC-ESO |
|---|---|---|
| 探索形状の学習 | **共分散行列を適応学習**（楕円形探索が可能） | 飛沫チャネルの差分ベクトルが **暗黙的に集団形状を反映**（共分散行列を学習しない） |
| 多峰対応 | 単一中心からの楕円分布（マルチスタートで多峰に対応） | **系統共存** — ニッチ分離されたエリート pool が多峰を保持 |
| 計算コスト/世代 | O(λ·d²)（行列演算あり） | O(pop·d)（行列演算なし） |
| 強みの関数タイプ | 連続・単峰・高 cond. | 多峰・弱構造を含む全クラスで安定 |

### PSO / DE との比較

| 観点 | PSO | DE | MC-ESO |
|---|---|---|---|
| 再生メカニズム数 | 1（速度ベクトル）| 1（差分変異 + 二項交叉）| **3 チャネル**（接触・飛沫・空気感染）|
| 個体の記憶 | 個体最良・群最良 | なし（target との 1:1 競合） | なし（感染確率で代替）|
| 集団選択 | 連続更新（置換なし） | greedy 1:1 置換 | **宿主競合** — μ+λ greedy + rollback で単調改善 |
| 多峰対応 | なし | なし（同じ basin に収束しがち） | **系統共存** ＋ **スピルオーバー** restart |

### DE との関係（個別）

MC-ESO の飛沫チャネルは `x_parent + F·(x_strain − x_parent) + F·(x_a − x_b)` という DE/current-to-best/1 と同型の更新式を使う。素の DE と比較することで:

- **差分変異単独 vs 3チャネル混合**: 同じ差分ベクトル機構を持つ DE が単独で何処まで到達するか。MC-ESO の overall 性能の何割が「差分ベクトルの効果」か、何割が「接触ガウス＋空気ランダム＋系統共存＋宿主競合＋スピルオーバー」の上乗せか、を切り分けられる
- **DE/rand/1/bin vs DE/current-to-best/1+ニッチ系統**: MC-ESO は引力対象を単一 best ではなくニッチ済みエリート pool から抽選するので、多峰関数（F15/F17/F21）での挙動差が直接観測できる

---

## ベースライン手法

### CMA-ES

共分散行列適応進化戦略。現在の探索分布の「形」を共分散行列として学習し、楕円形の探索が可能。収束を検出したら最良点からタイトな sigma で再スタートするマルチスタートを実装済み。

| パラメータ | 値 | 意味 |
|---|---|---|
| `sigma0` | `0.2 × (hi - lo)` | 初期探索範囲（`main.py` が問題ごとに付与。クラス既定値は 1.0）|
| マルチスタート | 有効 | 収束後、最良点から再起動 |

### PSO

慣性重み付きPSO（Kennedy & Eberhart, 1995）。各粒子が自身の最良点と群の最良点に引き寄せられながら速度を更新する。

| パラメータ | 値 |
|---|---|
| `n_particles` | 30 |
| `w`（慣性重み） | 0.729 |
| `c1`, `c2`（認知・社会係数） | 1.494 |

### DE（直接比較対象）

差分進化 / `DE/rand/1/bin`（Storn & Price, 1997）の古典版。各世代、集団内の target ごとに 3 つの異なる donor `a, b, c` を一様乱数で選び、変異ベクトル `v = x_a + F·(x_b − x_c)` を生成。二項交叉（rate `CR`、少なくとも 1 次元は `v` から継承）で trial `u` を作り、`f(u) ≤ f(target)` なら置換、というシンプルな単一機構。MC-ESO の飛沫チャネルと差分変異を共有するため「差分変異単独でどこまで行けるか」のベースラインとして直接置く（[DE との関係](#de-との関係個別)参照）。

| パラメータ | 値 | 意味 |
|---|---|---|
| `n_pop` | 30 | 集団サイズ（PSO と揃える） |
| `F` | 0.5 | 差分スケール |
| `CR` | 0.9 | 二項交叉率 |

### SaVOA（既存ウイルス手法・直接比較対象）

VOA の自己適応版（Liang & Juarez, 2020 近似実装）。sigma を世代ごとに乗法的に適応（改善 → σ×1.2、停滞 → σ×0.9）することで、手動チューニング不要にしたもの。同じ生物模倣着想だが再生メカニズムは単一。

### L-SHADE / IPOP-CMA-ES / BIPOP-CMA-ES（外部ライブラリ baseline）

より強力な近代手法を外部ライブラリ経由で `BaseOptimizer` インターフェースに合わせて組み込む。

- **L-SHADE**（`lshade.py`, mealpy ラッパー）— SHADE に線形集団縮小を加えた適応的 DE（Tanabe & Fukunaga 2014、CEC2014 優勝）。初期集団 `N_init = 18 × d`、`miu_f = miu_cr = 0.5`。
- **IPOP-CMA-ES**（`restart_cmaes.py`, pycma ラッパー）— 収束ごとに集団サイズ λ を倍化して再起動（Auger & Hansen 2005）。
- **BIPOP-CMA-ES**（同上）— 大規模・小規模 2 つの λ regime を予算が釣り合うよう交互に再起動（Hansen 2009）。
- IPOP/BIPOP も CMA-ES と同じく `sigma0 = 0.2 × span` を `main.py` が付与する。

> **注意**: pycma 系（CMA-ES seed0 / IPOP / BIPOP）は同一 seed でも run ごとに結果が変動しうる（再現性上の留意点）。

---

## 評価方法論

**多段 SR 報告**: BBOB 標準の ECDF 表示に倣い、各関数で `SR@10^k` (k = -1, -2, -3, -4, -5, -7, -10) を併記。`results/<run>/dim2/summary.csv` の `sr_1e-1, sr_1e-2, ..., sr_1e-10` 列で参照可能。

**Wilcoxon 符号順位検定**: MC-ESO vs 各既存手法を seed-paired で比較。`results/<run>/dim2/wilcoxon.csv` に関数 × 比較対手の p 値を保存。

- `p_value_two_sided`: 二側 p 値（差があるか）
- `p_value_ref_better`: 片側 p 値（MC-ESO が比較対手より優れているか）

---

## 開発メモ（ablation 記録）

### 検証フロー

`quick_check.py` は `_OPTIMIZERS` に MC-ESO 本体と既存手法を並べて 26 関数 × n_runs で ablation する。改良案を検証する場合は MC-ESO のサブクラスや別 entry を追加して quick で overall 改善を測り、確認できれば `main.py` の `_BASE_OPTIMIZERS` に統合する流れ。

```
./run.sh quick --funcs F08-Rosenbrock,F09-RosenbrockRot,F10-EllipsoidalRot,F12-BentCigar --max-evals 10000
```

`--funcs` 引数で任意関数に絞り込んだ集中検証ができる。

### ベースに統合された機構（MC-ESO 本体に常時 ON）

以下はすべて MC-ESO 本体に常時 ON で組込まれ、ablation で overall 改善を実証済み。

| 機構 | MC-ESO での位置付け |
|---|---|
| **飛沫感染チャネル**（h2h, DE/current-to-best/1） | 差分変異が集団形状から異方情報を獲得。F08/F09/F10/F12 の主因 |
| **h2h binomial crossover** (`h2h_CR=0.9`, DE/bin 標準値) | 飛沫の trial vector を親と座標毎に交叉し、separable 多峰の座標方向情報を保護 (DE/current-to-best/1/bin)。初期は F04/F17 用に 0.7 へ調整していたが、後続 ablation 完了後の hold-out 検証で標準値 0.9 のほうが overall で優ると判明し復帰 |
| **宿主競合**（μ+λ greedy + rollback） | 最良宿主の長期保持で F10/F12 の SR を改善 |
| **スピルオーバー＋basin switch** | quality-gated restart で全 pop を Uniform 再播種、連続失敗 2 回で best 破棄＋σ_init リセット。ill-cond の整列失敗と F24 双漏斗を救済 |
| **Drilling mode**（σ_drill_down=0.85） | σ < span × 1e-3 で σ 縮小を強化し浮動小数限界まで追込む |
| **接触感染の経験共分散** (`empirical_cov_floor=0.01`) | 集団経験共分散 `C_pop` の固有分解で接触感染ノイズを瞬間異方化。CMA-ES の rank-μ 学習と異なり履歴累積なし、basin 切替に即応 |
| **Drilling 中の空気感染停止** | `σ < span × precision_sigma_ratio` で `air_ratio_eff = 0`。drilling 中の広域ランダム雑音を排除し精度劣化を防止 |

### 検証され不採用となった variant

以下は `quick_check.py` で MC-ESO 統合候補として走らせたが overall 改善を実証できず、コードからも削除された。

| variant | 追加した挙動 | 不採用の理由 |
|---|---|---|
| MC-ESO-A1（per-dim σ close-contact） | 接触感染ノイズを集団 per-dim std で軸別スケール（軸整列の異方化） | F08/F17 では改善するが F14-DiffPowers の BBOB 回転と整合せず致命的劣化。後継の A2（経験共分散版）に置換 |
| MC-ESO-ABD | h2h_CR=0.9 ＋ σ-adapt 停滞ゲートを drilling 中バイパス ＋ 初回 spillover で座標軸 sweep | A_mild ベースと比べて CR=0.9 のため F18/F19 で勝つが F04 で回帰。Wilcoxon でも B/D 単独の有意寄与なし、結局 CR トレードオフに収束 |
| MC-ESO-A_mild_BD | 統合済み MC-ESO ＋ 同上の B/D | F09/F11/F18 での改善と F04/F14 での悪化が相殺しほぼ同等。B/D の overall 寄与なし |
| 旧 A〜N（`use_evolution_path` / `use_pop_covariance` / `use_lifespan_reset` / `use_adaptive_air` / `use_adaptive_h2h_F` / `use_aggressive_niche` / `use_h2h_archive` / `use_local_pair_h2h` ほか） | MC-ESO 初期開発で試した 8 案 | 各案とも単一関数の改善はあるものの 12 関数 SR 合計で baseline 以下、あるいは安全装置を要する構造欠陥（E）で overall を毀損し全削除 |

検証ログ: `results/20260515_150803_ベースライン_quick/dim2/{summary,wilcoxon}.csv`
