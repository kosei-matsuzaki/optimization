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

#### 1世代の動作フロー

```
1. ニッチ選択でエリートを特定
   └─ f値が良く かつ互いに niche_radius_dyn 以上離れた個体を最大 n_elite_max 個保護（不死）
   └─ niche_radius_dyn = max(niche_radius_min, niche_radius × (σ現在 / σ初期))
      探索初期は広く分散、後期は縮小して近接最適解も個別保護

2. 死亡判定
   └─ age > lifespan の個体を削除（エリートは不死）

3. 親の感染確率を計算
   └─ softmax((f_max − f) / temperature) → f が低いほど高確率（active 個体のみ対象）

4. 子個体の生成（死亡スロット数だけ生成）
   ├─ 局所感染（1 − air_ratio の割合）
   │   └─ 親の位置 + Gauss(0, σ_i)
   │       log_quality = clip((log10(f_max) − log10(f_親)) / log10_spread, 0, 1)
   │       combined    = log_quality × (0.7 + 0.3 × age_ratio)
   │       σ_i         = σ × sigma_min_ratio ^ combined   ← 指数減衰
   │       悪い個体(log_quality≈0) → combined≈0 → σ_i=σ（全力探索）
   │       良くて高齢(log_quality≈1, age_ratio≈1) → σ_i=σ×sigma_min_ratio（精密探索）
   └─ 空気感染（air_ratio の割合）
       └─ ランダム親位置 + Uniform(−σ_air, +σ_air)
           σ_air = σ × air_sigma_factor
           air_sigma_factor = air_sigma_max − (air_sigma_max − air_sigma_min) × diversity_ratio
           diversity_ratio  = clip(集団の空間的分散 / 0.289, 0, 1)  ← 収束時↑、分散時↓
       子は反射境界条件（reflective clipping）で探索域内に収める

5. σ 減衰
   └─ σ_global × sigma_decay（=0.99）で縮小

6. 仮想呼吸（Virtual Breathing）— n_pop_min > 0 のとき有効
   ├─ 縮小（改善が続く場合: no_improve < pop_shrink_trigger）
   │   └─ 非エリート active の下位 pop_change_by 個を dormant 化 → 次世代では親候補・子配置対象から除外
   └─ 拡大（停滞が深い場合: no_improve ≥ pop_grow_trigger）
       └─ f値が良い dormant を pop_change_by×2 個復活（旧位置をそのまま復元）
          非エリート active の上位 pop_change_by 個を dormant 化（局所解付近から解放）
       個体数は n_pop_min〜n_pop の範囲で動的に変化
```

#### ニッチ選択による複数コロニー維持

単純な top-k 選択では、最初に見つかった最適解周辺に個体が集中する。Himmelblau 関数（最適解4箇所）のような多最適解問題では致命的。

VSO のエリート選択:
1. f 値の良い順に候補を走査
2. 既保護個体との距離が全て `niche_radius_dyn` を超える場合のみ追加
3. `n_elite_max` 個に達したら終了
4. 品質閾値（現集団のf値スプレッドに基づく）を超えた候補は除外

→ 空間的に離れた複数の最適解周辺に独立したコロニーが自然形成される。

#### パラメータ一覧

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `n_pop` | 20 | 個体数（active 上限） |
| `n_pop_min` | 5 | 仮想呼吸の下限個体数（0 で無効） |
| `lifespan` | 5 | 個体の寿命（世代数） |
| `sigma` | 0.2 | 初期探索半径（探索範囲に対する比率） |
| `sigma_decay` | 0.99 | 世代ごとの探索半径縮小率 |
| `air_ratio` | 0.2 | 空気感染の基準割合（進行中は×0.5、停滞時は×3 まで no_improve に応じて線形変化） |
| `n_elite_max` | 6 | 保護するコロニー中心の最大数 |
| `niche_radius` | 1.0 | コロニー間の最小距離（初期値）|
| `niche_radius_min` | 0.05 | niche_radius の下限 |
| `temperature` | 1.0 | 感染確率のランダム性（大→均一、小→貪欲） |
| `stagnation_limit` | 2000 | 改善なし評価回数の上限（早期停止閾値） |
| `elite_quality_factor` | 1.0 | エリート候補の品質閾値係数 |
| `sigma_min_ratio` | 0.05 | 局所感染 σ スケールの下限（良い高齢個体の精密探索幅） |
| `air_sigma_min` | 1.5 | 集団分散時の空気感染 σ 倍率 |
| `air_sigma_max` | 5.0 | 集団収束時の空気感染 σ 倍率 |
| `pop_shrink_trigger` | 20 | 縮小発動閾値（no_improve がこれ未満で縮小） |
| `pop_grow_trigger` | 200 | 拡大発動閾値（no_improve がこれ以上で拡大） |
| `pop_change_by` | 2 | 1回の縮小/拡大で変化させる個体数 |
| `pop_change_cooldown` | 30 | 縮小/拡大後のクールダウン世代数 |

---

## 実験条件

| 設定 | 値 |
|---|---|
| 試行回数 | 30 run（seed = 0, 100, 200, ..., 2900） |
| 評価上限 | 15,000 回/run |
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
  {Func}_{Method}_vso_dyn.svg   — VSO 内部動態（VSO 系手法のみ、SVG、ベクター）
  {Func}_{Method}_vso_dyn_failed.svg
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
| `vso_dyn` / `vso_dyn_failed` | σ 動態・エリート水位・仮想呼吸の3行SVG |

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
