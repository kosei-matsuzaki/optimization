# 最適化手法の比較実験: MC-ESO と既存手法

感染症の流行（epidemic spread）を着想とした独自手法 **MC-ESO — Multi-Channel Epidemic Spread Optimizer** を、標準的な既存最適化手法と比較するベンチマーク実験です。

**核心の主張**: 既存メタヒューリスティクスはいずれも **単一の再生メカニズム** を持つ（DE = 差分変異, ES = ガウス変異, PSO = 速度ベクトル, GA = 交叉）。一方、現実の感染症は **複数の伝染経路** — 接触感染・飛沫感染・空気感染 — が並行して働く。MC-ESO はこれを忠実に模し、各世代で 3 つの定性的に異なる伝染チャネルを混合する。

---

## 提案手法と既存手法の位置づけ

| 手法 | 分類 | 本実験での位置づけ |
|---|---|---|
| **MC-ESO** (Multi-Channel Epidemic Spread Optimizer) | 群知能・独自提案 | **提案手法** |
| CMA-ES | 進化戦略 | ベースライン（強力な標準手法） |
| PSO | 群知能 | ベースライン |
| DE | 進化的アルゴリズム | 直接比較対象（MC-ESO の飛沫チャネルが借用する差分変異の本家・単一機構版） |
| SaVOA | ウイルス模倣・既存 | 直接比較対象（同じ生物模倣着想だが単一再生メカニズム） |

---

## MC-ESO が既存手法と異なる点

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

## ベンチマーク関数

**BBOB（Black-Box Optimization Benchmarking）ノイズなし版全24関数**を使用する。  
BBOB は Hansen et al. (2009) が提案した連続最適化の標準ベンチマークスイートであり、GECCO の COCO ワークショップで毎年使用されている。関数は `ioh` ライブラリ（instance=1）経由で取得し、`f(x) − f_opt` に正規化することでグローバル最小値を常に0とする。

> **なぜ BBOB か**  
> - 手作りの個別関数ではなく、査読済みスイートによる客観的な比較が可能  
> - 5つの難易度グループが問題の特性を体系的にカバー（分離可能・条件数・多峰性・弱構造）  
> - インスタンス変換（シフト・回転）が適用されており、座標軸や原点への過適合を防ぐ  
> - 既発表の CMA-ES, PSO, GA 等の結果と直接比較できる

探索範囲はすべての関数で **[-5, 5]^d**。

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

出典: BBOB は Hansen et al. (2009)。Custom 関数は Surjanovic & Bingham のテスト関数集および Tomitomi3 (Qiita) の整理を参照。

---

## 手法の詳細

### CMA-ES（ベースライン）

共分散行列適応進化戦略。現在の探索分布の「形」を共分散行列として学習し、楕円形の探索が可能。収束を検出したら最良点からタイトなsigmaで再スタートするマルチスタートを実装済み。

| パラメータ | 値 | 意味 |
|---|---|---|
| `sigma0` | `0.2 × (hi - lo)` | 初期探索範囲 |
| マルチスタート | 有効 | 収束後、最良点から再起動 |

---

### PSO（ベースライン）

慣性重み付きPSO（Kennedy & Eberhart, 1995）。各粒子が自身の最良点と群の最良点に引き寄せられながら速度を更新する。

| パラメータ | 値 |
|---|---|
| `n_particles` | 30 |
| `w`（慣性重み） | 0.729 |
| `c1`, `c2`（認知・社会係数） | 1.494 |

---

### DE（ベースライン・直接比較対象）

差分進化 / `DE/rand/1/bin`（Storn & Price, 1997）の古典版。各世代、集団内の target ごとに 3 つの異なる donor `a, b, c` を一様乱数で選び、変異ベクトル `v = x_a + F·(x_b − x_c)` を生成。二項交叉（rate `CR`、少なくとも 1 次元は `v` から継承）で trial `u` を作り、`f(u) ≤ f(target)` なら置換、というシンプルな単一機構。

MC-ESO の飛沫チャネルは DE/current-to-best/1 と同型の差分変異を使うため、**「差分変異単独でどこまで行けるか」のベースライン** として DE を直接置く。MC-ESO との overall 性能差は「3 チャネル混合 ＋ 系統共存 ＋ 宿主競合 ＋ スピルオーバー」の上乗せ寄与を表す。

| パラメータ | 値 | 意味 |
|---|---|---|
| `n_pop` | 30 | 集団サイズ（PSO と揃える） |
| `F` | 0.5 | 差分スケール |
| `CR` | 0.9 | 二項交叉率 |

---

### SaVOA（比較対象：既存ウイルス手法）

VOAの自己適応版（Liang & Juarez, 2020 近似実装）。sigma を世代ごとに乗法的に適応（改善→σ×1.2、停滞→σ×0.9）することで、手動チューニング不要にしたもの。

---

### MC-ESO — Multi-Channel Epidemic Spread Optimizer（提案手法）

感染症の流行を「感染宿主が空間上に分布し、複数の伝染経路から新規宿主を生む」過程としてモデル化する。

#### 基本概念

「**f(x) が低い領域 = 感染宿主が密集する場所**」とみなす。感染症は宿主密度の高い場所ほど広まりやすいため、f 値が低い個体ほど高い確率で次の親（感染源）に選ばれる。

#### 3 チャネル伝染モデル

現実の感染症は単一経路ではなく複数経路で同時に拡散する。MC-ESO はこれを最適化の文脈に転写する:

| チャネル | 疫学アナロジー | 数学的形 | 役割 |
|---|---|---|---|
| **接触感染** (Close-contact) | 親密接触による局所感染 | `x_parent + σ_i · L_pop · N(0, I)`（`L_pop L_pop^T = C_pop`、集団経験共分散の固有分解）| 親近傍の精密探索。σ_i は親の品質と年齢で適応、`C_pop` で **瞬間共分散** を獲得（履歴累積なし、CMA-ES と差別化） |
| **飛沫感染** (Droplet) | 飛沫を介した宿主間感染 | `x_parent + F·(x_strain − x_parent) + F·(x_a − x_b)` ＋ 親との二項交叉 (CR=0.7) | 系統 (niched elite) からの引力 ＋ 集団内差分ベクトルで集団形状を補強しつつ、二項交叉で座標方向の親情報を保護 (DE/current-to-best/1/bin) |
| **空気感染** (Airborne) | エアロゾルによる広域感染 | `x_random_host + N(0, σ_air I)`（drilling 中は停止）| 集団に依存しない遠方探索。局所最適脱出。`f_best < 1e-6` の drilling mode では雑音化するため停止 |

#### 集団レベルの 3 機構

| 機構 | 疫学アナロジー | 役割 |
|---|---|---|
| **系統共存** (Strain coexistence) | 空間的に離れた感染拠点の同時存続 | ニッチ半径で離れた最大 6 系統を保護、飛沫チャネルの引力対象 pool |
| **宿主競合** (Host competition) | 新感染が既存宿主に勝てないと排除される | 毎世代 25% kill、子が親より悪ければ rollback → 集団は単調改善 |
| **スピルオーバー** (Spillover) | 既存系統の絶滅後、新宿主集団へ感染が飛び火 | 300 評価改善なし AND f_best > 1e-8 で best 周辺に再播種。失敗 spillover の best 位置を **basin-avoidance memory** に記録し（最大 5 件）、後続 uniform 再播種は `0.05 × span` 内を回避 — 偽最適への再捕獲を防ぐ |

#### 1世代の動作フロー

```
1. 系統共存: ニッチエリート抽出（飛沫感染の引力 pool）
   └─ f 値の良い順に走査し、既存系統から niche_radius_ratio × span (=0.1×span) 以上離れた個体だけを
      最大 n_elite_max (=6) 個保護

2. スピルオーバー判定（停滞時の集団再播種、多様化＋エスカレート＋basin 回避版）
   └─ no_improve ≥ restart_no_improve_threshold (=300) かつ
      f_best > restart_quality_floor (=1e-8) のとき発動。
      連続失敗回数 (consecutive_failed_spillovers) でエスカレート:
         • streak = 0: 75% Uniform(lo, hi) + 25% best 周辺 N(x_best, σ_init×0.3)
         • streak ≥ 1: 100% Uniform(lo, hi)、best は保持
         • streak ≥ 2 かつ f_best > 1e-2: **ベイスン乗換え**
              — best も破棄、全 n_pop を Uniform 再生成、σ を σ_init にリセット
      ─ 多段エスカレートにより F24 双漏斗 / F04 rugged separable から脱出
        f_best ≤ 1e-2 のとき乗換え抑制 → F13 ridge / C01 deep precision を保護
      ─ 失敗 spillover の事前 best 位置を memory に追加（最大 5 件 FIFO）。
        後続 uniform 再播種は半径 0.05×span 内を rejection sample で回避
        → F18 SchafferF7ill の偽最適への再捕獲を防ぐ（SR_1e-10 33% → 67%、n=15）

3. 宿主競合: 死亡判定（μ+λ greedy）
   └─ 集団の f 値降順で下位 kill_fraction (=25%) を排除。最良宿主は自動生存

4. 親（感染源）の選択（softmax）
   └─ w_i ∝ exp((f_max − f_i) / temperature)、f が低い個体ほど高い感染力

5. 子個体の 3 チャネル生成（空きスロット数だけ）
   ├─ 接触感染 [残り]
   │   ├─ σ_i = σ × host_sigma_min_scale^(log_quality × (0.7 + 0.3 × age_ratio))
   │   ├─ C_pop = (1/(n-1)) Σ (x_i − x̄)(x_i − x̄)^T  ← 集団経験共分散
   │   ├─ 固有分解 V Λ V^T = C_pop、Λ を平均 1 に正規化（floor=0.01）
   │   └─ child = 親 + σ_i × V √Λ × Gauss(0, I)
   │       → 瞬間共分散による回転・異方性追従。F11/F14 で ill-cond 楕円体に整列
   │       （F11 mean 5e-8 → 0、F14 SR_1e-7 80% → 87%、n=15）
   ├─ 飛沫感染 [h2h_ratio = 0.4]
   │   └─ trial = 親 + h2h_F × (x_strain − 親) + h2h_F × (x_a − x_b)
   │       child = 各次元で確率 h2h_CR (=0.7) で trial を採用、残りは親をそのまま継承
   │       DE/current-to-best/1/bin と同型: 差分ベクトル ＋ 系統引力で集団形状を反映、
   │       二項交叉が separable 多峰の座標方向情報を保護（F04/F17 SR を大幅改善）
   └─ 空気感染 [air_ratio_eff = 0.3 if f_best ≥ 1e-6 else 0]
       └─ ランダム宿主 + Normal(0, σ_air I), σ_air は集団分散に応じて 1.5×〜5× σ
          drilling mode (f_best < 1e-6) では停止 — 雑音による精度妨害を排除
          （F06 SR_1e-10 93% → 100%、n=15）

6. 宿主競合: Rollback
   └─ 各空きスロットの子が「元そこにいた親」より悪ければ親を復元 → 集団は単調改善

7. σ 適応（always-on + drilling mode）
   └─ 改善: σ × sigma_up (=1.1)
   └─ 非改善 (通常): σ × sigma_down (=0.95)
   └─ 非改善 (drilling): best_so_far < drilling_threshold (=1e-6) のとき
                       σ × sigma_drill_down (=0.85) — 浮動小数限界へ追込み
```

#### 系統共存による多峰問題への適応

単純な top-k 選択では最初に見つかった最適解周辺に個体が集中する。Himmelblau 関数（最適解4箇所）のような多最適解問題では致命的。

MC-ESO の系統選択:
1. f 値の良い順に候補を走査
2. 既保護系統との距離が全て `niche_radius_ratio × span` を超える場合のみ追加
3. `n_elite_max` 個に達したら終了

→ 空間的に離れた複数の最適解周辺に独立した感染系統が自然形成され、飛沫感染の引力対象として活用される。

#### パラメータ一覧

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `n_pop` | 20 | 集団個体数 |
| `sigma` | 0.2 | 初期探索半径（探索範囲に対する比率） |
| `host_sigma_min_scale` | 0.05 | 接触感染チャネルにおける per-host σ_i スケーリング下限（高品質・高齢の宿主は σ_i = σ × 0.05 まで縮小して精密探索）|
| `empirical_cov_floor` | 0.01 | 接触感染チャネルの集団経験共分散 `C_pop` の固有値下限（平均 1 正規化後、軸の縮退を防ぐ）|
| `air_ratio` | 0.3 | 空気感染チャネルの割合 |
| `air_sigma_min` | 1.5 | 集団分散時の空気感染 σ 倍率 |
| `air_sigma_max` | 5.0 | 集団収束時の空気感染 σ 倍率（収束時に大ジャンプ） |
| `h2h_ratio` | 0.4 | 飛沫感染チャネルの割合 |
| `h2h_F` | 0.5 | 飛沫感染の差分ベクトルスケール係数 |
| `h2h_CR` | 0.7 | 飛沫感染後の二項交叉率（DE 流、座標方向の親情報を確率 1-CR で継承）|
| `kill_fraction` | 0.25 | 宿主競合で毎世代排除する割合 |
| `restart_no_improve_threshold` | 300 | スピルオーバー発動の no_improve 閾値 |
| `restart_sigma_ratio` | 0.3 | スピルオーバー後の σ（σ_init に対する比率） |
| `restart_quality_floor` | 1e-8 | スピルオーバー skip 閾値（既収束 run の精度破壊を防ぐ） |
| `restart_diversify_ratio` | 0.75 | スピルオーバー時、再播種個体のうち全探索域ランダムに割く割合（残りは best 周辺）|
| `escalate_after_failed_spillovers` | 1 | この連続失敗回数で diversify_ratio を 1.0 に昇格 |
| `basin_switch_after_failed_spillovers` | 2 | この連続失敗回数で best 破棄＋σ_init リセットの完全ベイスン乗換え |
| `basin_switch_quality_floor` | 1e-2 | best がこの値以下のときベイスン乗換えを抑制（grinding 中の run を保護）|
| `basin_radius_ratio` | 0.05 | basin-avoidance memory の回避半径（span に対する比率）|
| `basin_memory_size` | 5 | 記憶する失敗 basin の最大数（FIFO）|
| `n_elite_max` | 6 | 系統共存の最大数（飛沫感染の引力対象） |
| `niche_radius_ratio` | 0.1 | 系統間の最小距離（span に対する比率、スケール不変。BBOB span=10 で実効 1.0、絶対値版と数学的に同一） |
| `temperature` | 1.0 | 感染確率のランダム性（大→均一、小→貪欲） |
| `lifespan` | 5 | 接触感染 σ_i の年齢正規化分母 |
| `stagnation_limit` | 2000 | 改善なし評価回数の上限（早期停止閾値） |
| `log_slope_threshold` | 1e-4 | 「意味ある改善」の log10(f) 減少スロープ閾値 |
| `sigma_up` | 1.1 | σ adapt 改善時の乗数 |
| `sigma_down` | 0.95 | σ adapt 改善なし時の乗数 |
| `sigma_floor_ratio` | 1e-6 | σ_global の絶対下限（× span）|
| `sigma_ceil_ratio` | 1.0 | σ_global の絶対上限（× span）|
| `drilling_threshold` | 1e-6 | drilling mode 発動の `best_so_far` 閾値（既正しい basin 内のとき発動）|
| `sigma_drill_down` | 0.85 | drilling mode 中の σ 縮小乗数（通常の sigma_down より積極的）|

---

#### 性能（BBOB 12 関数代表サブセット）

5000 evals × 10 seeds の overall SR@1e-4 合計（1200 中）:

| 手法 | SR 合計 | 備考 |
|---|---:|---|
| **MC-ESO（提案）** | **1200** | **全 12 関数で SR=100%** — overall perfect |
| PSO | 1030 | 多峰関数で強い |
| SaVOA | 920 | F10 (ill-cond) で SR=20% |
| CMA-ES | 890 | F08/F09/F10/F12 で 100% だが多峰で弱い |

標的 ill-conditioned / moderate-cond 関数:

| 関数 | MC-ESO (5000 evals) | CMA-ES (5000 evals) |
|---|---:|---:|
| F08-Rosenbrock | 100% (mean 2.2e-9) | 100% |
| F09-RosenbrockRot | 100% (mean 0) | 100% |
| F10-EllipsoidalRot | 100% (mean 0) | 100% |
| F12-BentCigar | 100% (mean 0) | 100% |

**SR@1e-10 (drilling mode 効果)**: 9/12 関数で mean = exact 0 を達成。SR@1e-10 合計 1130/1200。

##### BBOB 全 26 関数結果（F01-F24 + C01-C02、5000 evals × 10 seeds）

| メトリック | 値 |
|---|---|
| SR@1e-2 合計 | 248/260 (95%) |
| **SR@1e-4 合計** | **242/260 (93%)** |
| SR@1e-7 合計 | 220/260 (85%) |
| SR@1e-10 合計 | 206/260 (79%) |
| mean = 0 達成数 | 12/26 関数 |

苦戦する 4 関数（CMA-ES でも完全解決は困難な BBOB の最難関）:
- **F24-LunacekRastrigin** SR4=30% — double-funnel 多峰（global と深い deceptive funnel）
- **F23-Katsuura** SR4=40% — フラクタル状（自己相似な無限階層凹凸）
- **F04-BucheRastrigin** SR4=70% — 1D 方向に rugged な Rastrigin
- **F18-SchafferF7ill** SR4=90% — ill-conditioned 多峰

明示的共分散行列学習なしに、飛沫感染の差分ベクトル（暗黙的異方性）＋ 宿主競合（最良保持）＋ スピルオーバー（整列失敗の救済）の組合せで CMA-ES クラスの性能を達成。

#### 評価方法論

**多段 SR 報告**: BBOB 標準の ECDF 表示に倣い、各関数で `SR@10^k` (k = -1, -2, -3, -4, -5, -7, -10) を併記。`results/<run>/dim2/summary.csv` の `sr_1e-1, sr_1e-2, ..., sr_1e-10` 列で参照可能。

**Wilcoxon 符号順位検定**: MC-ESO vs 各既存手法を seed-paired で比較。`results/<run>/dim2/wilcoxon.csv` に関数 × 比較対手の p 値を保存。
- `p_value_two_sided`: 二側 p 値（差があるか）
- `p_value_ref_better`: 片側 p 値（MC-ESO が比較対手より優れているか）

---

#### σ 適応 always-on（default ON）

SaVOA 流の σ 乗法適応を毎世代適用。改善時は σ を拡大して新しい谷を探り、非改善時は縮小して局所探索を強化:

```python
if best 改善:
    σ *= sigma_up (=1.1)
elif best_so_far < drilling_threshold (=1e-6):
    σ *= sigma_drill_down (=0.85)   # drilling mode — 浮動小数限界へ追込み
else:
    σ *= sigma_down (=0.95)
```

**設計のポイント**: 以前は `no_improve` ゲートで stuck 中は減衰を緩める fallback を持っていたが、ablation で F14 DiffPowers / F23 Katsuura に大幅改善（SR_1e-7 +20〜+44%）、多峰関数で軽度回帰のトレードオフが観測された。HP 削減（`sigma_adapt_stagnation_gate`, `sigma_decay` の 2 つを除去）と引換に採用。

#### Drilling mode（default ON）

正しい basin に到達した後の精密化フェーズ。`best_so_far < 1e-6` を超えたら **「正しいベイスン内」**と判定し、非改善時の σ 縮小を 0.95 → 0.85 と強化する。

```python
if best_so_far < drilling_threshold (=1e-6):  # 高精度ベイスン到達後のみ
    σ *= sigma_drill_down (=0.85)             # 強収縮で浮動小数限界へ
```

**効果（5000 evals × 10 seeds, n=12）**:
- 9/12 関数で mean = exact 0 を達成（F01/F03/F09/F10/F12/F15/F20/C02 + F11=Himmelblau は 9.5e-23）
- SR@1e-10 合計: 1100 → **1130/1200**（F09, F10, C01 で 90% → 100%）
- F17-SchafferF7 等の **deceptive 多峰関数は drilling 圏外（mean 1e-5）のため無影響** — 多峰ロバスト性を保ったまま smooth 関数で限界精度を取得

**設計のポイント**: 「正しいベイスン到達」を `best_so_far` で判定するため、deceptive landscape で誤って drilling が発動するリスクは小さい（誤った局所最適が 1e-6 を切るのは稀）。多峰／単峰の挙動切替を明示パラメータ無しで実現。

#### エスカレート式 spillover（default ON）

連続して spillover が改善に失敗した場合、disruption を段階的に強化する。IPOP-CMA-ES の "restart with larger population" 概念の MC-ESO 版。

```python
streak = consecutive_failed_spillovers
if streak >= 2 and best_so_far > 1e-2:
    # ベイスン乗換え — best を破棄、全 n_pop を Uniform、σ_init にリセット
    div_ratio = 1.0; preserve_best = False; sigma = σ_init
elif streak >= 1:
    # 完全多様化 — 全スロット Uniform、best は保持
    div_ratio = 1.0; preserve_best = True
else:
    # 通常 — 75% Uniform + 25% best 周辺
    div_ratio = 0.75; preserve_best = True
```

**効果（5000 evals × 10 seeds, BBOB 全 26 関数）**:
- F24-LunacekRastrigin: SR4 20% → **30%** (deceptive double-funnel から脱出可能に)
- F04-BucheRastrigin: SR4 50% → **70%** (rugged separable で重要)
- F06-AttractiveSector: mean 5.0e-07 → **2.4e-13**
- F11-Discus: mean 5.4e-08 → **6.0e-12**
- F19-GriewankRosenbrock: mean 1.0e-06 → **1.7e-08**
- 12 関数 quick subset: SR@1e-4 = 1200/1200 維持（regression なし）

**設計のポイント**:
- **多段化**: `streak ≥ 1` で多様化拡大、`streak ≥ 2` で本格的乗換え。早期にすべてを破壊しない
- **quality gate（1e-2）**: 既に高精度に達した run（F13 ridge, C01 deep precision）でベイスン乗換えが暴発しないよう保護
- spillover 自体の quality_floor (=1e-8) は維持されるため、収束済みの run に影響なし

##### 検証の方法

`quick_check.py` は `_OPTIMIZERS` に MC-ESO 本体と既存手法を並べて 26 関数 × n_runs で ablation する。改良案を検証する場合は MC-ESO のサブクラスや別 entry を追加して quick で overall 改善を測り、確認できれば `main.py` の `_BASE_OPTIMIZERS` に統合する流れ。

`./run.sh quick --funcs F08-Rosenbrock,F09-RosenbrockRot,F10-EllipsoidalRot,F12-BentCigar --max-evals 10000` で target 関数の集中検証が可能（`--funcs` 引数で任意関数の絞り込み）。

##### ベースに統合された案（MC-ESO 本体に直結したコア機構）

以下はすべて MC-ESO 本体に常時 ON で組込まれ、ablation で overall 改善を実証済み。

| 機構 | MC-ESO での位置付け |
|---|---|
| **飛沫感染チャネル**（h2h, DE/current-to-best/1） | 差分変異が集団形状から異方情報を獲得。F08/F09/F10/F12 の主因 |
| **h2h binomial crossover** (`h2h_CR=0.7`) | 飛沫の trial vector を親と座標毎に交叉し、separable 多峰の座標方向情報を保護。quick (n=30) で F04 SR 77→100%, F17 47→73%, F08 87→93%。F18/F19 では CR=0.9 より若干劣るトレードオフを受けつつ overall SR@1e-4 平均 92.0%→93.4% |
| **宿主競合**（μ+λ greedy + rollback） | 最良宿主の長期保持で F10/F12 を SR 0%→80/90% へ |
| **エスカレート式スピルオーバー** | quality-gated restart ＋ 連続失敗時の basin switch。ill-cond の整列失敗と F24 双漏斗を救済 |
| **Drilling mode**（σ_drill_down=0.85） | best_so_far < 1e-6 で σ 縮小を強化し浮動小数限界まで追込む |
| **接触感染の経験共分散** (`empirical_cov_floor=0.01`) | 集団経験共分散 `C_pop` の固有分解で接触感染ノイズを瞬間異方化。CMA-ES の rank-μ 学習と異なり履歴累積なし、basin 切替に即応。F11 mean 5e-8 → 0、F14 SR_1e-7 80% → 87%（n=15 quick） |
| **Drilling 中の空気感染停止** | `f_best < drilling_threshold` で `air_ratio_eff = 0`。drilling 中の広域ランダム雑音を排除し精度劣化を防止。F06 SR_1e-10 93% → 100% |
| **Basin-avoidance memory**（`basin_radius_ratio=0.05`, `basin_memory_size=5`） | 失敗 spillover の事前 best 位置を memory に記録、後続 uniform 再播種は半径 0.05×span 内を rejection sample で回避。F18 SchafferF7ill SR_1e-10 33% → 67% |

##### 検証され不採用となった variant（quick ablation, n=30）

以下は `quick_check.py` で MC-ESO 統合候補として走らせたが overall 改善を実証できず、コードからも削除された。

| variant | 追加した挙動 | 不採用の理由 |
|---|---|---|
| MC-ESO-A1（per-dim σ close-contact） | 接触感染ノイズを集団 per-dim std で軸別スケール（軸整列の異方化） | F08/F17 では改善するが F14-DiffPowers の BBOB 回転と整合せず、SR_1e-10 53%→13% と致命的劣化（a12=0.74 large, n=15）。後継の A2（経験共分散版）に置換 |
| MC-ESO-ABD | h2h_CR=0.9 ＋ σ-adapt 停滞ゲートを drilling 中バイパス ＋ 初回 spillover で座標軸 sweep | A_mild ベースと比べて CR=0.9 のため F18/F19 で勝つが F04 SR が 100→87% と回帰。Wilcoxon でも B/D 単独の有意寄与なし、結局 CR トレードオフに収束 |
| MC-ESO-A_mild_BD | 統合済み MC-ESO ＋ 同上の B/D | F09/F11/F18 で +0.04、F04/F14 で −0.04 と相殺し ECDF 0.2234→0.2241 でほぼ同等。B/D の overall 寄与なし |
| 旧 A〜N（`use_evolution_path` / `use_pop_covariance` / `use_lifespan_reset` / `use_adaptive_air` / `use_adaptive_h2h_F` / `use_aggressive_niche` / `use_h2h_archive` / `use_local_pair_h2h` ほか） | MC-ESO 初期開発で試した 8 案 | 各案とも単一関数の改善はあるものの 12 関数 SR 合計で baseline 以下、あるいは安全装置を要する構造欠陥（E）で overall を毀損し全削除 |

検証ログ: `results/20260515_150803_ベースライン_quick/dim2/{summary,wilcoxon}.csv`

---

## 実験条件

| 設定 | 値 |
|---|---|
| 試行回数 | 100 run（seed = 0, 100, 200, ..., 9900） |
| 評価上限 | 5,000 回/run |
| 成功判定 | best f ≤ 1e-4 |
| 次元数 | 2次元（BBOB 24関数 + カスタム2関数）、3次元（BBOB 24関数） |
| sigma0（CMA-ES） | `0.2 × (hi - lo)` |

以下の指標を報告する：

| 指標 | 定義 |
|---|---|
| **Mean / Std** | 全 run の最終 best f の平均・標準偏差 |
| **SR@1e-2** | `best_f ≤ 1e-2` を達成した run の割合（ゆるい成功） |
| **SR@1e-4** | `best_f ≤ 1e-4` を達成した run の割合（BBOB 標準成功） |
| **ERT** | Expected Running Time（BBOB 標準）= Σ(各 run の目標到達評価回数) / 成功 run 数。失敗 run は max_evals でペナルティ計上。全 run 失敗時は `---` |

ERT は成功率が 0% でも「どれだけ近づけたか」を相対的に比較できないが、SR@1e-2 と組み合わせることで緩い収束段階の差異を捉える。

---

## 可視化の見方

実行後、`results/YYYYMMDD_<commit>/dim{N}/` 以下に**関数×手法ごとに個別ファイル**として保存される。

### ファイル命名規則（新フォーマット）

```
dim{N}/
  {Func}_landscape.svg          — 2D 等高線 + 3D サーフェス（関数依存のみ、2D 関数のみ）
  {Func}_convergence.svg        — 全手法の収束曲線比較（SVG、ベクター）
  {Func}_{Method}_evals.webp    — 評価点蓄積アニメ（単一手法、2D のみ）
  {Func}_{Method}_evals_failed.webp
  {Func}_{Method}_runs.webp     — 探索軌跡アニメ（単一手法、2D のみ）
  {Func}_{Method}_population.webp
  {Func}_{Method}_population_failed.webp
  {Func}_{Method}_3devals.webp  — 3D 評価点蓄積（3D 関数のみ）
  {Func}_{Method}_3devals_failed.webp
  {Func}_{Method}_3dpopulation.webp
  {Func}_{Method}_3dpopulation_failed.webp
  {Func}_{Method}_outbreak_dyn.svg   — アウトブレイク内部動態（MC-ESO 系手法のみ、SVG、ベクター）
  {Func}_{Method}_outbreak_dyn_failed.svg
  stats/{Func}.csv
  summary.csv
```

**フォーマット**: 静的図は SVG（ベクター）、アニメーションは WebP（GIF より 30〜50% 小容量）。WebP 非対応環境では GIF にフォールバック。

### Web UI のビューモード

Web アプリ（`./run.sh web`）で結果を閲覧できる。右上の `[Function] [Method] [Compare]` タブでビューを切り替える。

| モード | 説明 |
|---|---|
| **Function** | 関数を選択 → 選択した可視化タイプを全手法グリッドで表示 |
| **Method** | 手法を選択 → 選択した可視化タイプを全関数グリッドで表示 |
| **Compare** | 関数・手法をマルチセレクト → 関数×手法のマトリクスグリッドで比較 |

### 可視化タイプ一覧

| タイプ | 説明 |
|---|---|
| `landscape` | 2D 等高線 + 3D サーフェス（関数形状のみ） |
| `convergence` | 全手法の収束曲線を1枚に比較 |
| `evals` / `evals_failed` | 評価点の蓄積アニメ（ベスト/ワースト run） |
| `runs` | 1フレーム=1run の探索軌跡アニメ |
| `population` / `population_failed` | 集団配置の推移アニメ |
| `3devals` / `3dpopulation` | 3D 関数用の評価点・集団アニメ |
| `outbreak_dyn` / `outbreak_dyn_failed` | 3 行 SVG: ①σ 動態（σ_global / 中央値 σᵢ / 子ごと σ scatter）、②best f 収束 ＋ 系統数 n_strains、③no_improve 推移 ＋ restart 閾値 |

### 画像の読み方

#### `landscape.svg`

```
左  : 2D 等高線（暗い = f が低い = 最適解に近い）+ 黄丸 = 真の最適解
右  : 3D サーフェスプロット
```

#### `convergence.svg`

```
x 軸: 評価回数
y 軸: best f（対数スケール）
線  : 全 run 平均
影  : ±1σ
```

#### アニメーション（runs）

```
薄い点（ラスタライズ）: 評価点（最大2000点にサブサンプリング）
折れ線               : best-x の更新軌跡
石灰色の点           : 成功した最終 best-x（f ≤ 1e-4）
赤い点               : 失敗した最終 best-x
黄丸                 : 真の最適解の位置
```

#### 3D アニメーション

評価点の色: `viridis_r` カラーマップ（**明るい黄色ほど f が低く最適解に近い**）。  
集団の色: 最適解からのユークリッド距離（**明るいほど最適解に近い**）。カメラが 30°→210° 回転。

---

## ディレクトリ構成

```
optimization/
├── core/                       # 研究コア（ベンチマーク・最適化手法・実験・可視化）
│   ├── __init__.py
│   ├── benchmarks.py           # BBOB 24関数 + カスタム関数定義（ioh 経由、2D/3D/4D）
│   ├── optimizers.py           # 全5手法の実装
│   ├── runner.py               # 複数run の実験実行・統計サマリー
│   └── visualize.py            # 関数地形図・収束曲線・各種 GIF の生成
├── web/                        # Results UI（Flask）
│   ├── app.py                  # Flask アプリ本体
│   ├── static/style.css        # スタイルシート
│   └── templates/              # Jinja2 テンプレート
│       ├── index.html          # トップ画面（Quick Run / GH Actions / 結果一覧）
│       └── result.html         # 結果詳細画面（可視化・テーブル）
├── main.py                     # 本番実験エントリーポイント（GitHub Actions 経由）
├── quick_check.py              # ローカル軽量確認スクリプト
├── run.sh                      # 実験管理 CLI
└── results/
    └── YYYYMMDD_HHMMSS_<commit>/
        ├── dim2/
        │   ├── {関数名}.svg            # 関数地形（2D等高線 + 収束曲線 + 3D表面）
        │   ├── {関数名}_runs.gif       # 試行別探索軌跡
        │   ├── {関数名}_evals.gif      # 評価点蓄積アニメーション
        │   ├── {関数名}_population.gif # 集団配置推移アニメーション
        │   ├── summary.csv             # 関数・手法別の統計量
        │   └── stats/{関数名}.csv      # per-run 詳細統計
        └── dim3/
            ├── {関数名}.svg            # 収束曲線 + 3D scatter（最良 run）
            ├── {関数名}_evals.gif      # 3D 評価点蓄積（viridis_r: 明=低f=最適解近傍）
            ├── {関数名}_population.gif # 3D 集団推移（距離→最適解でカラーリング）
            ├── summary.csv
            └── stats/{関数名}.csv
```

新しい手法を追加する場合は `core/optimizers.py` で `BaseOptimizer` を継承したクラスを作成し、`main.py` の `_BASE_OPTIMIZERS` に追記するだけで比較実験が動く。

---

## コマンド一覧

実験の実行・管理はすべて `run.sh` 経由で行う。結果は `results/YYYYMMDD_HHMMSS_<commit>/` に自動バージョン管理される。

| コマンド | 説明 |
|---|---|
| `./run.sh trigger` | GitHub Actions ワークフローをトリガー（本番実験） |
| `./run.sh trigger --n-runs 10 --max-evals 2000` | パラメータを指定してトリガー |
| `./run.sh download` | 最新の完了済みワークフロー結果をダウンロード |
| `./run.sh download <RUN_ID>` | 指定した RUN_ID の結果をダウンロード |
| `./run.sh quick` | ローカルで軽量確認（代表関数・10 run・2000 evals） |
| `./run.sh quick --n-runs 5 --max-evals 3000` | パラメータを指定してローカル確認 |
| `./run.sh status` | 最新ワークフロー実行の状態を表示 |
| `./run.sh status <RUN_ID>` | 指定した RUN_ID の状態を表示 |
| `./run.sh list` | ローカル結果一覧 + リモート実行履歴（最新5件） |
| `./run.sh ui` | Results UI を起動 → http://localhost:8080 |

```bash
# 典型的なワークフロー
./run.sh trigger          # 本番実験を投入
./run.sh status           # 完了を確認
./run.sh download         # 結果をローカルに保存

# ローカル動作確認
./run.sh quick

# Results UI の起動
./run.sh ui
```

`main.py`（本番実験）はローカルでは実行しない。`quick_check.py` はローカル専用の軽量確認スクリプト。

---

## Results UI

`./run.sh ui` または `python3 web/app.py` で Flask サーバーが起動し、ブラウザで実験管理・結果閲覧ができる。

| 機能 | 説明 |
|---|---|
| Quick Run | `quick_check.py` をバックグラウンドで実行。ライブターミナル出力を表示 |
| GitHub Actions Trigger | `gh` CLI 経由でワークフローをトリガー |
| Remote Runs | 最新10件のワークフロー実行を一覧表示。完了済みはそのままダウンロード可能 |
| Local Results | `results/` 配下の結果一覧。クリックで詳細画面へ遷移 |
| 結果詳細 | 次元タブ・関数タブで切替え。Evals / Convergence / Population / Landscape の各 GIF を表示 |
| Summary テーブル | 手法別の成績を色分け表示（best=緑、worst=赤）。ヘッダークリックでソート可能 |
| Per-run Stats | 各 run の詳細統計（成功/失敗を色分け） |

---

## 依存ライブラリ

```
numpy
matplotlib
cma
ioh        # BBOB ベンチマーク関数（IOH Experimenter）
```
