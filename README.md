# 最適化手法の比較実験: VSO と既存手法

ウイルス蔓延を模倣した独自手法 **VSO (Virus Spread Optimizer)** を、標準的な既存最適化手法と比較するベンチマーク実験です。

---

## 提案手法と既存手法の位置づけ

| 手法 | 分類 | 本実験での位置づけ |
|---|---|---|
| **VSO** (Virus Spread Optimizer) | 群知能・独自提案 | **提案手法** |
| CMA-ES | 進化戦略 | ベースライン（強力な標準手法） |
| PSO | 群知能 | ベースライン |
| GA | 進化的アルゴリズム | ベースライン |
| SaVOA | ウイルス模倣・既存 | 直接比較対象（同じウイルス着想） |

---

## VSO が既存手法と異なる点

### CMA-ES との比較

| 観点 | CMA-ES | VSO |
|---|---|---|
| 探索形状の学習 | **共分散行列を適応学習**（楕円形探索が可能） | 等方的ガウスノイズ（共分散非学習） |
| 個体多様性 | 単一分布から生成（多峰に弱い） | ニッチ選択で複数拠点を保持 |
| 計算コスト/世代 | O(λ·n²)（行列演算あり） | O(pop)（行列演算なし） |
| 強みの関数タイプ | 連続・単峰・滑らか | 多峰・複数最適解あり |

### PSO / GA との比較

| 観点 | PSO | GA | VSO |
|---|---|---|---|
| 個体の記憶 | 個体最良・群最良 | なし（選択・交叉） | なし（感染確率で代替） |
| 探索メカニズム | 速度ベクトル更新 | SBX交叉 + 多項式突然変異 | 局所感染（ガウス） + 空気感染（ランダム） |
| ニッチング | なし | なし | ニッチ半径で複数コロニーを保護 |

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

### GA（ベースライン）

実数値遺伝アルゴリズム。SBX（Simulated Binary Crossover）と多項式突然変異を使用。エリート選択で親・子世代の上位個体を次世代に引き継ぐ。

| パラメータ | 値 |
|---|---|
| `n_pop` | 50 |
| `crossover_rate` | 0.9 |
| `mutation_rate` | 0.1 |

---

### SaVOA（比較対象：既存ウイルス手法）

VOAの自己適応版（Liang & Juarez, 2020 近似実装）。sigma を世代ごとに乗法的に適応（改善→σ×1.2、停滞→σ×0.9）することで、手動チューニング不要にしたもの。

---

### VSO — Virus Spread Optimizer（提案手法）

ウイルスの蔓延を「感染者（個体）が空間上に分布し、感染力の高い場所から子孫を生む」過程としてモデル化する。

#### 基本概念

「**f(x) が低い領域 = 感染者が密集する場所**」とみなす。ウイルスは人の多い場所ほど広まりやすいため、f 値が低い個体ほど高い確率で親に選ばれる（感染力が高い）。

#### 次元別 sigma 適応（ill-conditioned 関数対応）

上位 n/4 個体の各次元の分散を毎世代観測し、次元別スケール `σ_d[i]` を EMA で更新する。

```
dim_var[i]  = var(top_k_x[:, i])        ← i 次元の上位個体のばらつき
σ_d_target  = sqrt(dim_var / mean(dim_var))
σ_d         = 0.95 · σ_d + 0.05 · σ_d_target    ← 遅い EMA (leak=0.95)
σ_d        /= mean(σ_d)                  ← 平均を 1 に正規化

局所感染: σ_i = σ × scale × σ_d   ← 次元ベクトルとして独立ガウスノイズ
空気感染: σ_vec = σ_base × air_sigma_factor × σ_d
```

**なぜこれで機能するか（例: BentCigar = x₀² + 10⁶·Σxᵢ²）**  
上位個体はx[1:]≈0（急峻な次元はすぐ選択で排除）→ dim_var[1:]が小さい → σ_d[1:]が縮小（精密探索）。  
x[0]はまだ探索中 → dim_var[0]が大きい → σ_d[0]が拡大（広域探索）。  
多峰性関数（Himmelblau等）では上位個体が複数盆地に分散 → 全次元の分散が揃う → σ_d≈1（影響なし）。

#### 1世代の動作フロー

```
1. ニッチ選択でエリートを特定
   └─ f値が良く かつ互いに niche_radius 以上離れた個体を最大 n_elite_max 個保護
   └─ ※上位5個体が探索空間の5%以内に集中（見かけ単峰）の場合は niche_radius_min を使用
      → 単一最適解への収束を妨げない

2. 死亡判定
   └─ age > lifespan の個体を削除（エリートは不死）

3. 親の感染確率を計算
   └─ softmax(-(f - f_max) / temperature) → f が低いほど高確率

4. 子個体の生成（死亡数だけ生成）
   ├─ 近傍感染（1 - air_ratio の割合）
   │   └─ 選ばれた親の位置 + Gauss(0, σ × scale)
   │       combined = 0.5 × quality + 0.5 × age_ratio
   │       scale    = sigma_min_ratio ^ combined   ← 指数減衰（combined=0 → 1, combined=1 → sigma_min_ratio）
   │       quality  = (f_max - f_親) / (f_max - f_min)  ← 集団内相対品質 [0,1]
   │       age_ratio = 親のage / lifespan               ← 年齢の相対値 [0,1]
   └─ 空気感染（air_ratio_eff の割合）
       └─ ランダムに選んだ親の位置 + Gauss(0, σ_air)
           air_ratio_eff = air_ratio + (0.5 - air_ratio) × stagnation_ratio²  ← 停滞が深いほど増加
           σ_air = max(σ, σ_init × 0.3) × air_sigma_factor           ← σ 下限でフロア保証
           air_sigma_factor = air_sigma_max − (air_sigma_max − air_sigma_min) × diversity_ratio
           diversity_ratio  = clip(集団の空間的分散 / 0.289, 0, 1)  ← 0=収束, 1=一様分布

5. 死亡スロットに新個体を配置（age = 0 でリセット）

6. Bloom（改善があった世代のみ）
   └─ 現集団の最良個体の近傍（σ × 2.0）に bloom_size 個を追加（上限 n_pop_max）
   └─ ※単峰性関数では有望領域を集中探索; 多峰性でも現最良が各盆地を巡るため多様性を維持

7. 過密制御（diversity_ratio > 0.05 のときのみ実行）
   └─ n_pop_max を超えたら最悪非エリート個体を除去
   └─ diversity_ratio ≤ 0.05（ほぼ収束）のときはスキップ → 単一最適解への収束を邪魔しない
   └─ 各エリート周辺の crowd_radius（= crowd_radius_frac × σ現在）内に過剰集中した個体を強制老化
   └─ 強制老化した個体の後継は空気感染扱い（force_air_slots）→ elite周辺への再集中を防ぐ
```

#### fitness・age に応じた探索半径（近傍感染）

```
quality   = (f_max - f_親) / (f_max - f_min)   # 0=最悪, 1=最良
age_ratio = 親のage / lifespan                  # 0=若い, 1=老い
combined  = 0.5 × quality + 0.5 × age_ratio
σ_i       = σ × sigma_min_ratio ^ combined   ← 指数減衰
```

| 状態 | combined | σ_i |
|------|----------|-----|
| 良い + 老い（精密探索） | 高 | σ × sigma_min_ratio |
| 良い + 若い / 悪い + 老い | 中 | 中程度 |
| 悪い + 若い（広域探索） | 低 | σ × 1.0 |

`σ` 自体は `sigma_decay` でグローバルに縮小するため、探索初期は全員が広く探索し、後期に両軸が効いてくる。

#### 空間的多様性に応じた空気感染半径

```
diversity_ratio = clip(mean(std(pop) / span) / 0.289, 0, 1)
air_sigma_factor = air_sigma_max − (air_sigma_max − air_sigma_min) × diversity_ratio
σ_air = σ × air_sigma_factor
```

- 集団が収束（diversity → 0）: `σ_air` → `σ_base × air_sigma_max`（大きく飛ぶ）
- 集団が分散（diversity → 1）: `σ_air` → `σ_base × air_sigma_min`（近くに留まる）

既に探索空間を広く覆っているときは遠くに飛んでも意味がなく、収束したときこそ大きく離れた場所を試す必要があるという状況を自動的に反映する。  
また `σ_base = max(σ, σ_init × 0.3)` のフロアにより、後期に `σ` が崩壊しても空気感染が有効な探索幅を維持する。停滞が長引くほど `air_ratio_eff` も増大し、局所解脱出圧力が自動的に強まる。

#### ニッチ選択による複数コロニー維持

単純な top-k 選択では、最初に見つかった最適解周辺に個体が集中する。Himmelblau 関数（最適解4箇所）のような多最適解問題では致命的。

VSO のエリート選択:
1. f 値の良い順に候補を走査
2. 既保護個体との距離が全て `niche_radius_dyn` を超える場合のみ追加
3. `n_elite_max` 個に達したら終了
4. 品質閾値（現集団のf値スプレッドに基づく）を超えた候補は除外

`niche_radius_dyn` は global sigma の減衰に比例して縮小する：

```
niche_radius_dyn = max(niche_radius_min, niche_radius × (σ_現在 / σ_初期))
```

- 探索初期：`niche_radius` が大きい → コロニーを広く分散させる
- 探索後期：`niche_radius` が縮小 → 近接した複数最適解も個別に保護できる

→ 空間的に離れた複数の最適解周辺に独立したコロニーが自然形成され、最適解間距離が小さい問題でも対応できる。

#### パラメータ一覧

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `n_pop` | 20 | 初期個体数 |
| `n_pop_max` | 40 | 個体数の上限 |
| `lifespan` | 5 | 個体の寿命（世代数） |
| `sigma` | 0.2 | 初期探索半径（探索範囲に対する比率） |
| `sigma_decay` | 0.99 | 世代ごとの探索半径縮小率 |
| `air_ratio` | 0.2 | 空気感染の割合 |
| `n_elite_max` | 6 | 保護するコロニー中心の最大数 |
| `niche_radius` | 1.0 | コロニー間の最小距離（初期値）|
| `niche_radius_min` | 0.05 | niche_radius の下限（探索後期に縮小） |
| `temperature` | 1.0 | 感染確率のランダム性（大→均一、小→貪欲） |
| `stagnation_limit` | 2000 | 改善なし評価回数の上限（早期停止閾値） |
| `elite_quality_factor` | 1.0 | エリート候補の品質閾値係数 |
| `bloom_size` | 2 | 改善時に追加する個体数 |
| `crowd_radius_frac` | 1e-5 | 過密制御の半径（現在の sigma の何倍か）。sigma の減衰に追従して縮小 |
| `max_crowd_fraction` | 0.10 | 各エリート周辺で許容する個体の割合 |
| `sigma_min_ratio` | 0.05 | sigma スケールの下限（`sigma_min_ratio^combined` の底; 小さいほど収束時の円が小さくなる） |
| `air_sigma_min` | 1.5 | 集団分散時の空気感染 sigma 倍率 |
| `air_sigma_max` | 5.0 | 集団収束時の空気感染 sigma 倍率 |

## 実験条件

| 設定 | 値 |
|---|---|
| 試行回数 | 30 run（seed = 0, 100, 200, ..., 2900） |
| 評価上限 | 5,000 回/run |
| 成功判定 | best f ≤ 1e-4 |
| 次元数 | 2次元（BBOB 24関数 + カスタム2関数）、3次元（BBOB 24関数） |
| sigma0（CMA-ES） | `0.2 × (hi - lo)` |

統計量として Mean / Std / Median / Success Rate を報告する。  
Success は `best_f ≤ 1e-4`（BBOB 標準精度ターゲット）で判定する。

---

## 可視化の見方

実行後、`results/YYYYMMDD_<commit>/dim{N}/` 以下に各関数ごとの図が保存される。

### 2次元関数

#### `{関数名}.svg` — 関数地形 + 収束曲線

3パネル構成（横18インチ）。

```
左  : 2D 等高線（暗い = f が低い = 最適解に近い）+ 黄丸 = 真の最適解
中  : 収束曲線（x 軸 = 評価回数、y 軸 = best f 対数スケール、線 = 全 run 平均、影 = ±1σ）
右  : 3D サーフェスプロット
```

#### `{関数名}_runs.gif` — 探索軌跡（試行回数別）

1フレーム = 1 run。各フレームで全手法の探索軌跡・評価点・最終収束点を同時表示。

```
薄い点（ラスタライズ）: 評価点（最大2000点にサブサンプリング）
折れ線               : best-x の更新軌跡
石灰色の点           : 成功した最終 best-x（f ≤ 1e-4）
赤い点               : 失敗した最終 best-x
黄丸                 : 真の最適解の位置
```

#### `{関数名}_evals.gif` — 評価点の蓄積アニメーション

評価点が順に積み上がる様子を eval=100 単位で描画。各手法の探索密度の変化を確認できる。

#### `{関数名}_population.gif` — 集団配置の推移アニメーション

各世代の現在集団の配置を描画。個体群がどのように空間を移動・収束するかを確認できる。

### 3次元関数

#### `{関数名}.svg` — 収束曲線 + 3D scatter

```
左  : 収束曲線（全手法の平均 ± 1σ）
右  : 最良 run の全評価点を 3D scatter で表示（viridis_r: 明るい黄=低f=最適解近傍）
```

#### `{関数名}_evals.gif` — 3D 評価点蓄積アニメーション

評価点が順に積み上がる様子を 3D 空間で描画。色は `viridis_r` カラーマップで f 値を表現（**明るい黄色ほど f が低く最適解に近い**）。

#### `{関数名}_population.gif` — 3D 集団配置推移アニメーション

各世代の集団配置を 3D 空間で描画。色は最適解位置からのユークリッド距離（**明るいほど最適解に近い**）。アニメーションに合わせてカメラが 30°→210° 回転するため、空間的な探索過程を多角的に確認できる。

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
| `./run.sh quick` | ローカルで軽量確認（代表関数・3 run・2000 evals） |
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
