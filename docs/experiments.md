# 実験ガイド — ディレクトリ構造・実行・条件・評価基準

ディレクトリ構成、実行コマンド、実験条件、ベンチマーク関数、評価方法論、結果の見方をまとめる。提案手法は [mceso.md](mceso.md)、比較手法は [baselines.md](baselines.md)、工夫・フラグの履歴は [history.md](history.md)、Results UI は [web.md](web.md) を参照。

---

## ディレクトリ構成

```
optimization/
├── core/                       # 研究コア（ベンチマーク・最適化手法・実験・可視化）
│   ├── __init__.py             # 主要クラス/関数の公開API再エクスポート（visualize除く）
│   ├── benchmarks.py           # BBOB 24関数（ioh 経由、dim 2/3/5/10/20）+ カスタム(2D) + CEC2022(10D)
│   ├── optimizers/             # 手法ごと1ファイル（__init__.py で全クラス再エクスポート）
│   │   ├── base.py             # OptimizeResult, BaseOptimizer
│   │   ├── cmaes.py            # CMA-ES（best-anchored restart）
│   │   ├── mceso.py            # MC-ESO（提案手法）+ _MCESOState
│   │   ├── pso.py / de.py / savoa.py  # PSO・DE・SaVOA baseline
│   │   ├── lshade.py           # L-SHADE（mealpy wrapper）
│   │   └── restart_cmaes.py    # IPOP/BIPOP-CMA-ES（pycma wrapper）
│   ├── runner.py               # 複数run の実験実行・統計サマリー
│   └── visualize.py            # 関数地形図・収束曲線・各種 GIF の生成
├── web/                        # Results UI（Flask）→ docs/web.md
├── main.py                     # 本番実験エントリーポイント（GitHub Actions 経由）
├── quick_check.py              # ローカル軽量確認スクリプト
├── run.sh                      # 実験管理 CLI
├── docs/                       # ドキュメント（本ディレクトリ）
└── results/
    └── YYYYMMDD_HHMMSS_<commit>/
        ├── dim2/
        │   ├── {Func}_landscape.svg   # 関数地形（2D等高線 + 3D表面）
        │   ├── {Func}_convergence.svg # 全手法の収束曲線比較
        │   ├── {Func}_{Method}_*.webp # 探索軌跡・評価点蓄積・集団推移アニメ
        │   ├── summary.csv            # 関数・手法別の統計量
        │   ├── wilcoxon.csv           # MC-ESO vs 各手法の検定結果
        │   └── stats/{Func}.csv       # per-run 詳細統計
        └── dim3/                      # 3D 版（3D scatter / 3D アニメ）
```

結果はすべて `results/YYYYMMDD_HHMMSS_<commit>/` に自動バージョン管理される。

### 新しい手法を追加する

`core/optimizers/` に新しいファイル（例 `myopt.py`）を作り `BaseOptimizer` を継承したクラスを定義し、`core/optimizers/__init__.py` で再エクスポートしたうえで `main.py` の `_BASE_OPTIMIZERS` に追記すれば比較実験に組み込まれる。

---

## 依存ライブラリ

```
numpy
matplotlib
cma        # pycma（CMA-ES / IPOP / BIPOP）
mealpy     # L-SHADE
ioh        # BBOB / CEC2022 ベンチマーク関数（IOH Experimenter）
```

---

## コマンド一覧（run.sh）

実験の実行・管理はすべて `run.sh` 経由で行う。**手法の検証・評価は quick（n_runs=20 / max_evals=5000 / `--all`）で統一する**。`trigger`（GitHub Actions, n=100）は裏で補助的に回す実験で、手法評価の根拠には使わない。

| コマンド | 説明 |
|---|---|
| `./run.sh quick --all` | **手法評価の標準コマンド**。**2 次元 BBOB-24（F01-F24）のみ**を n_runs=20 / max_evals=5000 で評価 |
| `./run.sh quick --all --custom` | BBOB-24 に Custom 11（C01-C11, 2D 限定）を追加。多峰・多解など**特定目的の参照時のみ**使う |
| `./run.sh quick --funcs C01-Himmelblau,C02-SixHumpCamel` | Custom 単独に絞り込んだ集中確認 |
| `./run.sh quick --funcs F08-Rosenbrock,F10-EllipsoidalRot` | 任意関数に絞り込んだ集中検証（デバッグ用） |
| `./run.sh quick --suite niching --n-runs 20` | **低次元多峰の評価**。CEC2013 niching の 2D/3D サブセット（N04-N10）を各関数の次元で回す（`--dim` は無視）|
| `./run.sh quick --suite niching --suite-budget` | 同上を**競技の公式予算**（MaxFEs 5e4 / 2e5 / 4e5）で回す。文献値と比べたいときだけ使う |
| `./run.sh quick --all --dim {2\|3\|5\|10\|20} --max-evals <2500×d>` | **次元スケーリング計測**（BBOB-24 を各次元で）。現状把握のスナップショット用。採否判定は 2D が主対象 |
| `./run.sh quick --n-runs 5 --max-evals 3000` | パラメータを上書きしてローカル確認 |
| `./run.sh quick --all --noise gauss_sev` | **ノイズ評価モード**（診断用）。noisy f をアルゴリズムに見せ、指標は真値で再採点（下記） |
| `./run.sh ui` | Results UI を起動 → http://localhost:8080 |
| `./run.sh trigger` | GitHub Actions ワークフローをトリガー（**補助実験**, n=100。評価には使わない） |
| `./run.sh download` / `./run.sh download <RUN_ID>` | 完了済みワークフロー結果をダウンロード |
| `./run.sh status` / `./run.sh status <RUN_ID>` | ワークフロー実行の状態を表示 |
| `./run.sh list` | ローカル結果一覧 + リモート実行履歴（最新5件） |

```bash
# 手法の検証・評価（標準）
./run.sh quick --all             # n_runs=20 / max_evals=5000 / 2D BBOB-24 のみ
./run.sh quick --all --custom    # 多峰・多解など特定目的の参照時のみ Custom を追加
./run.sh ui                      # 結果を Results UI で確認

# 補助実験（裏で回す。評価には参照しない）
./run.sh trigger          # n=100 ワークフローを投入
./run.sh download         # 完了後にダウンロード
```

> **実行ルール**: `quick_check.py` はローカル専用スクリプトで、`python3 quick_check.py` を直接呼ばず `./run.sh quick` を使う。手法評価は**原則 2 次元 BBOB-24（`--all`）・n_runs=20・max_evals=5000** で実施し、quick-12 サブセットでの判定は行わない。Custom（C01-C11）は多峰・多解など特定目的を確認したいときにのみ `--custom` で追加参照する（採否判定は原則 BBOB-24 で決める）。`main.py`（補助実験）はローカルでは実行せず GitHub Actions 経由のみ。

### ノイズ評価モード（`--noise`, 診断用）

`./run.sh quick --noise {gauss_mild|gauss_sev|cauchy}` で BBOB-noisy 流の評価ノイズを掛けて頑健性を測る（`core/benchmarks.py: NOISE_MODELS / make_noisy_func`、採点は `core/runner.run_experiment`）。

- **アルゴリズムには noisy f のみを見せ、報告される全指標（SR / evals_succ / Wilcoxon / summary.csv）は訪問点の真値で再採点する**（COCO-noisy 慣行。観測 min で採点するとノイズ下裾を拾った見かけ成功が混ざるため）。
- ノイズは乗法型でスケール不変: `gauss_mild` = f·exp(0.01·N)、`gauss_sev` = f·exp(1.0·N)、`cauchy` = 確率 0.2 で f×(1+|Cauchy|) の上方スパイク。**真値 f ≤ 1e-8 はノイズ無効**（最終ターゲットが測定可能のまま残る、BBOB-noisy 慣行）。
- ノイズ RNG は (関数, モデル, run) の CRC32 seed で optimizer seed と独立・再現可能。
- **位置づけは診断**: noiseless の標準評価（SR@1e-10 死守ルール等）とは独立で、採否判定には使わない。`result.json` に `"noise"` フィールドが記録される。

### 評価の分析・自動化

| ツール | 用途 |
|---|---|
| `scripts/analyze_quick.py <run_dir> [--baseline <dir>] [--baseline-method <name>] [--dim N]` | quick 結果（`summary.csv`/`wilcoxon.csv`）を規定の **3 指標**（SR / `evals_succ_mean` / Wilcoxon）に集計し、SR@1e-10 主指標・関数別の改善/悪化・SR@1e-10 非回帰チェック・判定を出力。SR@1e-10 非回帰チェックは2モード: `--baseline <旧run_dir>`（cross-run, 同名 MC-ESO を run 間で差分） / `--baseline-method <名前>`（within-run, 同一 run 内の元版と差分。**改変 MC-ESO と元 MC-ESO の 2 手法のみ**を `--methods "MC-ESO,<元名>"` で回したとき用）。CSV を手で読む代わりにこれで報告する |
| サブエージェント `experimenter`（`.claude/agents/`） | 「比較手法設定 → `./run.sh quick` 実行 → monitor → `analyze_quick.py` で分析 → 判定を返す」一連を独立コンテキストで完結。手法ブラッシュアップ中に本会話を汚さず評価を回すためのもの（評価専任・コードは変更しない） |

---

## 実験条件（評価の標準）

手法の検証・評価は以下の条件（quick デフォルト）で統一する。

| 設定 | 値 |
|---|---|
| 試行回数 | **20 run**（seed = 0, 100, 200, ..., 1900） |
| 評価上限 | **5,000 回/run** |
| 成功判定 | best f ≤ 1e-4 |
| 次元数 | **2次元 BBOB-24（F01-F24）が判定の主対象**。Custom（C01-C11）は `--custom` で追加する特定目的の参照用。3次元（BBOB 24関数）・CEC2022 hold-out（dim10）は汎化確認用。**次元スケーリング計測用に BBOB は dim 2/3/5/10/20 を `--dim` で選択可**（`_build(d)` を各次元でレジストリ化。eval 予算は次元比例 `2500×d` を目安＝ d2=5000 / d3=7500 / d5=12500 / d10=25000 / d20=50000。採否判定は従来どおり 2D を主対象とし、多次元は現状把握のスナップショット用） |
| sigma0（CMA-ES 系） | `0.2 × (hi - lo)` |

> 補助的に GitHub Actions で n=100 の実験（`./run.sh trigger`）も回しているが、手法の検証・評価では参照しない。評価の根拠は常に上記 quick n=20 の結果とする。

---

## ベンチマーク関数

### BBOB 24 関数（主スイート）

**BBOB（Black-Box Optimization Benchmarking）ノイズなし版全 24 関数**を使用する。BBOB は Hansen et al. (2009) が提案した連続最適化の標準ベンチマークスイートであり、GECCO の COCO ワークショップで毎年使用されている。関数は `ioh` ライブラリ（instance=1）経由で取得し、`f(x) − f_opt` に正規化することでグローバル最小値を常に 0 とする。探索範囲はすべての関数で **[-5, 5]^d**。

> **なぜ BBOB か**
> - 手作りの個別関数ではなく、査読済みスイートによる客観的な比較が可能
> - 5 つの難易度グループが問題の特性を体系的にカバー（分離可能・条件数・多峰性・弱構造）
> - インスタンス変換（シフト・回転）が適用されており、座標軸や原点への過適合を防ぐ
> - 既発表の CMA-ES, PSO, GA 等の結果と直接比較できる

#### 分類の 2 軸: 公式グループ（category）と形状タグ（tags）

各関数には **2 通りの分類**を付与している（`core/benchmarks.py`）。

- **`category`（BBOB 公式 5 グループ）**: Hansen et al. の原著グループ（separable / moderate-cond / ill-cond / multimodal / weak-structure）をそのまま採用。**既発表結果との比較可能性**を保つための単一軸ラベルであり変更しない。
- **`tags`（形状タグ）**: 公式グループ名は 1 軸しか表さないため、関数の**実際のランドスケープ形状**が読み取れない（例: F02 は "separable" グループだが本質的難しさは悪条件、F03 は "separable" グループだが本質は多峰）。これを補うため、形状を**直交する複数軸**で記述するタグを別途付与する。`SHAPE_TAGS` が唯一の定義元で、`summary.csv` の `tags` 列（`|` 区切り）に出力され、Results UI の「形状タグ別」内訳と各関数のツールチップに表示される。

形状タグの軸（各関数は該当するタグを複数持つ）:

| 軸 | 値 |
|---|---|
| modality（峰性） | `unimodal` / `multimodal` / `multi-global`（大域最適解が複数） |
| separability（分離性） | `separable` / `non-separable` |
| conditioning（条件数） | `well-conditioned` / `moderate-cond` / `ill-conditioned` |
| structure（大域構造・多峰時のみ） | `global-structure` / `weak-structure` |
| landscape（局所形状） | `smooth` / `linear` / `asymmetric` / `plateau` / `bent-valley` / `sharp-ridge` / `rugged` / `deceptive` / `boundary-optimum` / `needle` |
| suite-shape（CEC2022 構成） | `hybrid` / `composition` |

| FID | 関数名 | グループ | 形状タグ | 主な難しさ |
|---|---|---|---|---|
| F01 | Sphere | separable | unimodal, separable, well-conditioned, smooth | 最も単純。アルゴリズムの健全性確認 |
| F02 | Ellipsoidal (sep.) | separable | unimodal, separable, **ill-conditioned** | 軸方向に強い条件数（形状は悪条件単峰） |
| F03 | Rastrigin (sep.) | separable | **multimodal**, separable, global-structure | 分離可能な多峰性（形状は多峰） |
| F04 | Büche-Rastrigin | separable | **multimodal**, separable, global-structure, asymmetric | 非対称な多峰性 |
| F05 | Linear Slope | separable | unimodal, separable, linear, boundary-optimum | 最適解が境界上 |
| F06 | Attractive Sector | moderate-cond | unimodal, non-separable, asymmetric, moderate-cond | 非対称な単峰性 |
| F07 | Step Ellipsoidal | moderate-cond | unimodal, non-separable, plateau, moderate-cond | 段差状の不連続性（階段プラトー） |
| F08 | Rosenbrock | moderate-cond | unimodal, non-separable, bent-valley | バナナ型の曲がった谷 |
| F09 | Rosenbrock (rot.) | moderate-cond | unimodal, non-separable, bent-valley | Rosenbrock に回転を適用 |
| F10 | Ellipsoidal (rot.) | ill-cond | unimodal, non-separable, ill-conditioned | 高条件数、軸非整合 |
| F11 | Discus | ill-cond | unimodal, non-separable, ill-conditioned | 1次元のみ強く伸びた形状 |
| F12 | Bent Cigar | ill-cond | unimodal, non-separable, ill-conditioned, bent-valley | 曲がった葉巻型の谷 |
| F13 | Sharp Ridge | ill-cond | unimodal, non-separable, ill-conditioned, sharp-ridge | 鋭い稜線（非平滑） |
| F14 | Different Powers | ill-cond | unimodal, non-separable, ill-conditioned | 次元ごとに異なるべき乗 |
| F15 | Rastrigin (rot.) | multimodal | multimodal, non-separable, global-structure | 局所解が密、回転あり |
| F16 | Weierstrass | multimodal | multimodal, non-separable, global-structure, rugged | 高度に多峰・不規則（rugged） |
| F17 | Schaffer F7 | multimodal | multimodal, non-separable, global-structure, moderate-cond | 中程度の多峰性 |
| F18 | Schaffer F7 (ill) | multimodal | multimodal, non-separable, global-structure, ill-conditioned | F17 に高条件数を追加 |
| F19 | Griewank-Rosenbrock | multimodal | multimodal, non-separable, global-structure, bent-valley | Rosenbrock 谷を含む複合地形 |
| F20 | Schwefel | weak-structure | multimodal, **separable**, weak-structure, deceptive | 大域構造が弱く欺瞞的（大域は境界寄り。座標独立で分離可能だが公式グループは weak-structure） |
| F21 | Gallagher 101 peaks | weak-structure | multimodal, non-separable, weak-structure | 101 個のガウス峰が散在 |
| F22 | Gallagher 21 peaks | weak-structure | multimodal, non-separable, weak-structure | F21 より峰が少なく深い |
| F23 | Katsuura | weak-structure | multimodal, non-separable, weak-structure, rugged | フラクタル的な地形 |
| F24 | Lunacek bi-Rastrigin | weak-structure | multimodal, non-separable, weak-structure, deceptive | 二重ファネルで大域が欺瞞的 |

**太字**は公式グループ名からは読めない、形状タグで補正される特性（F02=悪条件、F03/F04=多峰）。

### Custom 関数（2-D）

BBOB がカバーしない **多大域最適解**・**deceptive 2-D 多峰** 系の古典的テスト関数を補完。MC-ESO の「ニッチ系統共存」と「広域 spillover」の挙動を BBOB の回転・シフトに依らない素のランドスケープで検証する目的。各関数は `f(x) − f_opt` で正規化し最小値を 0 とする。

> **位置づけ**: Custom は評価の**主対象ではない**。標準の手法評価は 2 次元 BBOB-24（F01-F24）で行い、Custom は多峰・多解性能など**特定の目的を重視して確認したいときにのみ** `./run.sh quick --all --custom`（または `--funcs C01,...` で単独）で参照する。採否の判定は原則 BBOB-24 で決める。

| ID | 関数名 | 探索域 | カテゴリ | 形状タグ | 主な難しさ |
|---|---|---|---|---|---|
| C01 | Himmelblau | [-5, 5]² | multi-optima | multi-global, multimodal, smooth | 大域最適解が **4 箇所**（ニッチ性能の直接評価） |
| C02 | Six-hump Camel | [-2, 2]² | multi-optima | multi-global, multimodal, smooth | 大域最適解が **2 箇所** |
| C03 | Shubert | [-10, 10]² | multi-optima | multi-global, multimodal, rugged | 大域最適解が **18 箇所**（積形式・約760 局所解） |
| C04 | Five-well Potential | [-20, 20]² | deceptive-2d | multimodal, deceptive | 5 つの井戸（うち1つが大域）|
| C05 | Eggholder | [-512, 512]² | deceptive-2d | multimodal, rugged, deceptive, boundary-optimum | 極めて鋭い多峰・大域は境界近傍 |
| C06 | Michalewicz (m=10) | [0, π]² | deceptive-2d | multimodal, plateau, deceptive | 平坦域に細い谷、急峻 |
| C07 | Bukin N.6 | [-15, 15]² | deceptive-2d | multimodal, sharp-ridge, deceptive | y = 0.01x² の極細谷、gradient 不連続 |
| C08 | Styblinski-Tang | [-5, 5]² | deceptive-2d | multimodal, deceptive | 4 局所解、3 つが大域に近い深さ |
| C09 | Easom | [-100, 100]² | deceptive-2d | unimodal, plateau, needle | 広大な平坦域中の鋭い単一峰（needle-in-haystack）|
| C10 | Schaffer N.2 | [-100, 100]² | deceptive-2d | multimodal, rugged | 同心円状の多峰、原点中心 |
| C11 | De Jong F5 (Shekel's foxholes) | [-65.536, 65.536]² | deceptive-2d | multimodal, plateau, deceptive | 5×5 格子の25局所解 |

### CEC2013 niching（低次元多峰, N04-N10）

多解探索を分野の土俵で測るためのスイート。Li, Engelbrecht & Epitropakis (2013) の CEC'2013 niching competition ベンチマークのうち、**2D/3D の 7 関数**を実装した（`core/benchmarks.py:_NICHING_SPECS`）。式は参照実装 `github.com/mikeagn/CEC2013` の `matlab/niching_func.m`、f_goptima / rho / 大域解数 / MaxFEs は同梱の `get_fgoptima` / `get_rho` / `get_no_goptima` / `get_maxfes` から取った。

公式スイートは**最大化**問題。ここでは `f_goptima - f_raw(x)` として登録するので他スイートと同じく 0 へ最小化し、**`sr_1e-1` 〜 `sr_1e-5` 列がそのまま競技の精度水準 ε に一致する**。

| 名前 | dim | 大域解数 K | rho | 公式 MaxFEs | 探索域 |
|---|---|---|---|---|---|
| N04-Himmelblau | 2 | 4 | 0.01 | 5e4 | [-6, 6]² |
| N05-SixHumpCamel | 2 | 2 | 0.5 | 5e4 | [-1.9, 1.9]²（下記） |
| N06-Shubert2D | 2 | 18 | 0.5 | 2e5 | [-10, 10]² |
| N07-Vincent2D | 2 | 36 | 0.2 | 2e5 | [0.25, 10]² |
| N08-Shubert3D | 3 | 81 | 0.5 | 4e5 | [-10, 10]³ |
| N09-Vincent3D | 3 | 216 | 0.2 | 4e5 | [0.25, 10]³ |
| N10-ModRastrigin2D | 2 | 12 | 0.01 | 2e5 | [0, 1]² |

公式スイートからの逸脱は 3 つ。いずれも意図的で、論文に書くときはそのまま明示する。

- **F1-F3（1 次元）は未実装**。この repo は dim=1 を通せない（pycma は N≥2、可視化も 2D/3D 前提）。
- **F11-F20（合成関数）は未実装**。スイートの shift / rotation データファイルが要る。低次元 2D の難問（F11-F13）が抜けるので、必要になったら次に足すのはここ。
- **N05 の探索域**は公式が x₁∈[-1.9, 1.9] / x₂∈[-1.1, 1.1] の非対称ボックス。`BenchmarkFunction.bounds` が全軸共通の 1 レンジしか持たないため x₂ を [-1.9, 1.9] に広げた。大域解 2 個は変わらず（公式帯の外では f が増えるので最大値は増えない）、距離も歪まないので rho ベースの計数は保たれるが、**探索体積が公式の 1.7 倍**なので公式 F5 の公表値とは直接比較できない。

### CEC2022（hold-out）

BBOB とは独立した CEC2022 12 関数（`ioh` 経由、dim=10）を hold-out スイートとして用意。BBOB の変換に対して開発された MC-ESO の機構が汎化するかの検証に使い、**CEC2022 用にハイパーパラメータを再調整しない**。

出典: BBOB は Hansen et al. (2009)。Custom 関数は Surjanovic & Bingham のテスト関数集および Tomitomi3 (Qiita) の整理を参照。

---

## 評価方法論

> **評価の鉄則**: 手法を比較・評価する際は必ず以下の **3 指標**（SR / 評価回数 / 統計検定）を揃えて報告する。単一指標（SR のみ等）での判定は不可。評価は**原則 2 次元 BBOB-24（F01-F24）全関数**で実施し、関数別の改善・悪化を列挙して regression を見落とさない。Custom（C01-C11）は多峰・多解など特定目的の参照用で、採否判定の主対象にはしない。

### 1. SR（Success Rate, 多段報告）

BBOB は `f − f_opt` 正規化により最適値が 0。**最高精度 SR@1e-10 を主指標**として 0 への到達率で評価し、補助的に SR@1e-2 / 1e-4 / 1e-7 も併記して精度階層全体の挙動を確認する。BBOB 標準の ECDF 表示に倣い、各関数で `SR@10^k` (k = -1, -2, -3, -4, -5, -7, -10) を `summary.csv` の `sr_1e-1, ..., sr_1e-10` 列で参照可能。

### 2. 平均評価回数（Evals to target）

成功 run のみを対象とした目標到達評価回数の**平均** `evals_succ_mean`（失敗 run は集計から除外、ERT のような max_evals ペナルティ補正は行わない。全 run 失敗時は `---`）。成功 run のみを対象とするため評価回数のばらつきは小さく外れ値は生じにくいので、中央値ではなく**平均で評価する**（ランキングもこの平均値で算出）。SR が同等なら少ない評価回数が優位。**必ず SR と併せて読む**: 成功率が低い手法では少数の成功 run だけで平均が決まる点に注意。`summary.csv` には参考用に中央値 `evals_succ_med` と、旧来の `ert`（失敗 run を max_evals でペナルティ計上した BBOB 標準値）も併記。

### 3. 統計的優位性（Wilcoxon 符号順位検定）

MC-ESO（V1）を reference とし、各手法を seed-paired で比較。`wilcoxon.csv` に関数 × 比較対手の p 値を保存。

- `p_value_two_sided`: 二側 p 値（差があるか）
- `p_value_ref_better`: 片側 p 値（MC-ESO が比較対手より優れているか）
- 有意水準 α=0.05。p < 0.05 で「有意差あり」と判定し、A12 で効果量（negligible/small/medium/large）も併記。
- 判定は **quick n=20** で行う。n=20 は標本数が限られ signal-to-noise が高くないため、僅差・単一関数の有意差は効果量（A12）と関数別の方向性も併せて慎重に解釈する。

### 多解（MMO）報告

大域最適解が複数ある関数（C01 Himmelblau=4 / C02 Six-hump=2 / C03 Shubert=18）では、SR（= 1 つでも到達したか）に加え、MC-ESO の「並行的な多解探索」能力を直接計測する。指標は走行後に `history_x`（全評価点）と `benchmark.optima_pos`（既知の全大域最適解座標）から後付け計算され、**最適化器は一切変更しない**（＝ SR は定義上不変）。`core/runner.py:optima_found_mask` / `peak_metrics` が正準実装。

- **Peak Ratio (`pr_1e-2`, `pr_1e-4`)** — 既知 K 個の大域最適解のうち、`f ≤ tol` かつ最近傍割当で半径内に評価点が落ちた解の割合（run 平均）。各評価点は最近傍の最適解 1 つにのみ帰属させ、近接最適解（Shubert は最小間隔 ~0.88）での二重カウントを防ぐ。
- **MMO Success Rate (`mmo_sr_1e-2`, `mmo_sr_1e-4`)** — K 個**すべて**を見つけた run の割合。
- `n_optima` 列に K を記録。`summary.csv` の `mean_optima_found` / `mean_optima_rate` は従来からの `tol=1e-4` 単一値（後方互換）。
- 注意: 発見は**時間的**（走行中にいずれかの世代で訪れた）で、厳密な「同時保持」ではない。

### 多解報告（CEC2013 ルール, niching スイート）

上の Custom 向け PR は**全評価点** (`history_x`) から後付けで数えるため、密にサンプルするだけの手法（ランダム探索・多点 restart）を過大評価する。niching スイートでは競技の規則どおり、**run が「解」として報告した集合だけ**を採点する（`core/runner.py:niching_peak_metrics`）。

- **報告集合** = 最終集団 ＋ restart 系手法の各 restart の best（`OptimizeResult.final_solutions`）。MC-ESO は生存ホスト＋永続系統アーカイブ、NM-Restart / IPOP / BIPOP / CMA-ES は restart best ＋最終集団、集団を持たない場合は best 1 点。f の良い順に `max(100, 2K)` 点で打ち切る。
- **計数** (`count_goptima`) は公式 `how_many_goptima` と同じ順序: 報告集合を f 順に並べ、既採用点から rho より遠い点だけを seed として拾い、**その後で**精度 ε 内かを判定する。良いが不正確な点がニッチを塞ぐ挙動まで含めて再現する（冗長な報告を罰するのがこの指標の要点）。
- 列は `cec_pr_{ε}` / `cec_sr_{ε}`（ε = 1e-1 … 1e-5）、その平均 `cec_pr_mean` / `cec_sr_mean`、報告点数 `n_reported`、`cec_k`。`scripts/analyze_quick.py` が `[1b]` 節で集計する。

> **SR@1e-10 は絶対死守**: 多解探索の改善でも SR@1e-10 を下げる構成は採用しない。最低 1 解は深精度到達を保証する。

---

## 結果の見方（可視化）

実行後、`results/YYYYMMDD_<commit>/dim{N}/` 以下に**関数 × 手法ごとの個別ファイル**として保存される。静的図は SVG（ベクター）、アニメーションは WebP（GIF より 30〜50% 小容量、非対応環境では GIF フォールバック）。

### ファイル命名規則

```
dim{N}/
  {Func}_landscape.svg          — 2D 等高線 + 3D サーフェス（関数依存のみ、2D 関数のみ）
  {Func}_convergence.svg        — 全手法の収束曲線比較
  {Func}_{Method}_evals.webp    — 評価点蓄積アニメ（単一手法、2D のみ）／ _evals_failed.webp
  {Func}_{Method}_runs.webp     — 探索軌跡アニメ（単一手法、2D のみ）
  {Func}_{Method}_population.webp / _population_failed.webp
  {Func}_{Method}_3devals.webp  — 3D 評価点蓄積（3D 関数のみ）／ _3dpopulation.webp
  {Func}_{Method}_outbreak_dyn.svg  — アウトブレイク内部動態（MC-ESO 系手法のみ）／ _outbreak_dyn_failed.svg
  stats/{Func}.csv
  summary.csv
  wilcoxon.csv
```

### 可視化タイプ一覧

| タイプ | 説明 |
|---|---|
| `landscape` | 2D 等高線 + 3D サーフェス（関数形状のみ） |
| `convergence` | 全手法の収束曲線を 1 枚に比較 |
| `evals` / `evals_failed` | 評価点の蓄積アニメ（ベスト/ワースト run） |
| `runs` | 1 フレーム=1run の探索軌跡アニメ |
| `population` / `population_failed` | 集団配置の推移アニメ |
| `3devals` / `3dpopulation` | 3D 関数用の評価点・集団アニメ |
| `outbreak_dyn` / `outbreak_dyn_failed` | 3 行 SVG: ①σ 動態（σ_global / 中央値 σᵢ / 子ごと σ scatter）、②best f 収束 ＋ 系統数 n_strains、③no_improve 推移 ＋ restart 閾値 |

### 画像の読み方

- **`landscape.svg`** — 左: 2D 等高線（暗い = f が低い = 最適解に近い）+ 黄丸 = 真の最適解。右: 3D サーフェスプロット。
- **`convergence.svg`** — x 軸: 評価回数、y 軸: best f（対数スケール）、線: 全 run 平均、影: ±1σ。
- **アニメーション（runs）** — 薄い点（ラスタライズ）: 評価点（最大 2000 点にサブサンプリング）、折れ線: best-x の更新軌跡、石灰色の点: 成功した最終 best-x（f ≤ 1e-4）、赤い点: 失敗した最終 best-x、黄丸: 真の最適解の位置。
- **3D アニメーション** — 評価点の色は `viridis_r` カラーマップ（**明るい黄色ほど f が低く最適解に近い**）。集団の色は最適解からのユークリッド距離（**明るいほど最適解に近い**）。カメラが 30°→210° 回転。
