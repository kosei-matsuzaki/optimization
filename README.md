# 最適化手法の比較実験: MC-ESO と既存手法

感染症の流行（epidemic spread）を着想とした独自手法 **MC-ESO — Multi-Channel Epidemic Spread Optimizer** を、標準的な既存最適化手法と比較するベンチマーク実験です。

**核心の主張**: 既存メタヒューリスティクスはいずれも **単一の再生メカニズム** を持つ（DE = 差分変異, ES = ガウス変異, PSO = 速度ベクトル, GA = 交叉）。一方、現実の感染症は **複数の伝染経路** — 接触感染・飛沫感染・空気感染 — が並行して働く。MC-ESO はこれを忠実に模し、各世代で 3 つの定性的に異なる伝染チャネルを混合する。

---

## 手法・ベンチマークの詳細

提案手法 **MC-ESO** と既存手法（CMA-ES / IPOP・BIPOP-CMA-ES / PSO / DE / L-SHADE / SaVOA）の位置づけ・差別化、および評価に用いるベンチマーク関数（BBOB 全24関数 + Custom 2-D 11関数）の詳細は **[core/README.md](core/README.md)** を参照。

---

## ディレクトリ構成

```
optimization/
├── core/                       # 研究コア（ベンチマーク・最適化手法・実験・可視化）→ 構成は core/README.md
├── web/                        # Results UI（Flask）→ 構成・API は web/README.md
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

新しい手法を追加する場合は `core/optimizers/` に新しいファイル（例 `myopt.py`）を作って `BaseOptimizer` を継承したクラスを定義し、`core/optimizers/__init__.py` で再エクスポートしたうえで、`main.py` の `_BASE_OPTIMIZERS` に追記するだけで比較実験が動く。

---

## 依存ライブラリ

```
numpy
matplotlib
cma
ioh        # BBOB ベンチマーク関数（IOH Experimenter）
```

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

`./run.sh ui`（または `python3 web/app.py`）で Flask サーバーが起動し、ブラウザで
実験管理・結果閲覧ができる。機能・ディレクトリ構成・アーキテクチャ・API 一覧の詳細は
**[web/README.md](web/README.md)** を参照。

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
