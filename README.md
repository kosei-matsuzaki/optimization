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
| VOA | ウイルス模倣・既存 | 直接比較対象（同じウイルス着想） |
| SaVOA | ウイルス模倣・既存 | 直接比較対象（VOA の自己適応版） |

VOA・SaVOA は既発表のウイルス着想アルゴリズムです。**VSO はこれらと同じ着想を持ちつつ、異なるモデル化とメカニズムで設計した提案手法**です。

---

## VSO が既存手法と異なる点

### VOA / SaVOA との比較（最重要）

VOA/SaVOA と VSO はどちらも「ウイルスの感染」をモデル化していますが、個体の更新モデルが根本的に異なります。

| 観点 | VOA / SaVOA | VSO（提案手法） |
|---|---|---|
| 個体の更新 | 毎世代、**全個体**を置換する | **寿命制**：死んだ個体のみ置換。エリートは不死 |
| 探索範囲の制御 | sigma を世代単位でリセット or 乗算 | 親の**年齢**に応じて sigma をスケーリング（若い→広域、老いた→精密） |
| エリート保護 | top-k 個体を選択（空間非考慮） | **ニッチ選択**：空間的に離れた複数コロニーを同時保護 |
| 個体数 | 固定 | 改善時に一時増加（**bloom**）、上限 `n_pop_max` で制御 |
| 複数最適解への対応 | 最良解に収束しやすい | 空間分散エリートにより複数コロニーを自然に維持 |

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

### VOA（比較対象：既存ウイルス手法）

Liang & Juarez (2016) によるウイルス最適化アルゴリズム。個体群を「強力個体（上位 strong_ratio）」と「普通個体（残り）」に分割し、強力個体は局所探索、普通個体は強力個体への誘導またはランダム探索を行う。改善時にsigmaをリセット、停滞時に増大。

---

### SaVOA（比較対象：既存ウイルス手法）

VOAの自己適応版。sigmaを世代ごとに乗法的に適応（改善→σ×1.2、停滞→σ×0.9）することで、手動チューニング不要にしたもの。

---

### VSO — Virus Spread Optimizer（提案手法）

ウイルスの蔓延を「感染者（個体）が空間上に分布し、感染力の高い場所から子孫を生む」過程としてモデル化する。

#### 基本概念

「**f(x) が低い領域 = 感染者が密集する場所**」とみなす。ウイルスは人の多い場所ほど広まりやすいため、f 値が低い個体ほど高い確率で親に選ばれる（感染力が高い）。

#### 1世代の動作フロー

```
1. ニッチ選択でエリートを特定
   └─ f値が良く かつ互いに niche_radius 以上離れた個体を最大 n_elite_max 個保護

2. 死亡判定
   └─ age > lifespan の個体を削除（エリートは不死）

3. 親の感染確率を計算
   └─ softmax(-(f - f_max) / temperature) → f が低いほど高確率

4. 子個体の生成（死亡数だけ生成）
   ├─ 近傍感染（1 - air_ratio の割合）
   │   └─ 選ばれた親の位置 + Gauss(0, σ × σ_decay^{親のage})
   └─ 空気感染（air_ratio の割合）
       └─ 探索空間全体の一様ランダム点

5. 死亡スロットに新個体を配置（age = 0 でリセット）

6. Bloom（改善があった世代のみ）
   └─ bloom_size 個の個体を追加探索（上限 n_pop_max）

7. 過密制御
   └─ n_pop_max を超えたら最悪非エリート個体を除去
   └─ 各エリート周辺の crowd_radius 内に過剰集中した個体を強制老化
```

#### 親の年齢に応じた探索半径

```
σ_i = σ × σ_decay^(親のage)
```

若い親（age 小）→ 広めの探索 → 周辺の大まかなサンプリング  
老いた親（age 大）→ 狭い探索 → 有望領域の精密な掘り下げ

この仕組みにより、個体の「ライフサイクル」そのものが探索スケールの自動制御として機能する。

#### ニッチ選択による複数コロニー維持

単純な top-k 選択では、最初に見つかった最適解周辺に個体が集中する。Himmelblau 関数（最適解4箇所）のような多最適解問題では致命的。

VSO のエリート選択:
1. f 値の良い順に候補を走査
2. 既保護個体との距離が全て `niche_radius` を超える場合のみ追加
3. `n_elite_max` 個に達したら終了
4. 品質閾値（現集団のf値スプレッドに基づく）を超えた候補は除外

→ 空間的に離れた複数の最適解周辺に独立したコロニーが自然形成される。

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
| `niche_radius` | 1.0 | コロニー間の最小距離 |
| `temperature` | 1.0 | 感染確率のランダム性（大→均一、小→貪欲） |
| `stagnation_limit` | 2000 | 改善なし評価回数の上限（早期停止閾値） |
| `elite_quality_factor` | 1.0 | エリート候補の品質閾値係数 |
| `bloom_size` | 2 | 改善時に追加する個体数 |
| `crowd_radius_frac` | 1e-5 | 過密制御の半径（探索範囲の比率） |
| `max_crowd_fraction` | 0.10 | 各エリート周辺で許容する個体の割合 |

---

## 実験条件

| 設定 | 値 |
|---|---|
| 試行回数 | 30 run（seed = 0, 100, 200, ..., 2900） |
| 評価上限 | 5,000 回/run |
| 成功判定 | best f ≤ 1e-4 |
| 次元数 | 2次元（3次元・4次元は別途） |
| sigma0（CMA-ES） | `0.2 × (hi - lo)` |

統計量として Mean / Std / Median / Success Rate を報告する。  
Success は `best_f ≤ 1e-4`（BBOB 標準精度ターゲット）で判定する。

---

## 可視化の見方

実行後、`results/dim2/{関数名}/` 以下に図が保存される。

### trajectory.png — 探索軌跡

```
薄い点   : すべての評価点（どこを探索したか）
折れ線   : best f が更新されたときの位置の時系列軌跡
★        : 真の最適解の位置
背景色   : 等高線（暗い = f が低い = 最適解に近い）
```

### convergence.png — 収束曲線

```
x 軸 : 関数評価回数
y 軸 : その時点の best f（対数スケール）
線   : 30 run の中央値
影   : 四分位範囲（25%〜75%）
```

---

## ファイル構成

```
optimization/
├── benchmarks.py   # BBOB 24関数の定義（ioh 経由、2D/3D/4D）
├── optimizers.py   # 全6手法の実装（CMAESOptimizer, PSOOptimizer, GAOptimizer,
│                   #   VOAOptimizer, SaVOAOptimizer, VirusOptimizer）
├── runner.py       # 複数run の実験実行・統計サマリー
├── visualize.py    # 等高線+軌跡図・収束曲線の生成
├── main.py         # エントリーポイント
└── results/
    ├── dim2/
    │   └── {関数名}/
    │       ├── trajectory.png
    │       ├── convergence.png
    │       └── stats.json
    ├── dim3/   # run_dimension(BENCHMARKS_3D, "dim3") を有効化した場合
    └── dim4/   # run_dimension(BENCHMARKS_4D, "dim4") を有効化した場合
```

新しい手法を追加する場合は `optimizers.py` で `BaseOptimizer` を継承したクラスを作成し、`main.py` の `_BASE_OPTIMIZERS` に追記するだけで比較実験が動く。

---

## コマンド一覧

実験の実行・管理はすべて `run.sh` 経由で行う。結果は `results/YYYYMMDD_HHMMSS_<commit>/` に自動バージョン管理される。

| コマンド | 説明 |
|---|---|
| `./run.sh trigger` | GitHub Actions ワークフローをトリガー（本番実験） |
| `./run.sh trigger --n-runs 10 --max-evals 2000` | パラメータを指定してトリガー |
| `./run.sh download` | 最新の完了済みワークフロー結果をダウンロード |
| `./run.sh download <RUN_ID>` | 指定した RUN_ID の結果をダウンロード |
| `./run.sh quick` | ローカルで軽量確認（代表4関数・3 run・2000 evals） |
| `./run.sh quick --n-runs 5 --max-evals 3000` | パラメータを指定してローカル確認 |
| `./run.sh status` | 最新ワークフロー実行の状態を表示 |
| `./run.sh status <RUN_ID>` | 指定した RUN_ID の状態を表示 |
| `./run.sh list` | ローカル結果一覧 + リモート実行履歴（最新5件） |

```bash
# 典型的なワークフロー
./run.sh trigger          # 本番実験を投入
./run.sh status           # 完了を確認
./run.sh download         # 結果をローカルに保存

# ローカル動作確認
./run.sh quick
```

`main.py`（本番実験）はローカルでは実行しない。`quick_check.py` はローカル専用の軽量確認スクリプト。

---

## 依存ライブラリ

```
numpy
matplotlib
cma
ioh        # BBOB ベンチマーク関数（IOH Experimenter）
```
