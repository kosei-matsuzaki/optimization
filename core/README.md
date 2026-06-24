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
| **スピルオーバー** (Spillover, 情報化リスタート) | 既存系統の絶滅後、生存リザーバから未感染域へ飛び火 | 300 評価改善なし AND f_best / \|f_init\| > 1e-8 で発動。**情報化再播種**: 一部 (`ir_archive_frac`) を永続系統アーカイブまわりで再着火、残りは放棄 basin 重心を斥力で避けた Uniform（集団免疫）。連続失敗 2 回で basin switch (best 破棄＋σ_init リセット) に escalate。盲目 Uniform restart（IPOP 流）との差別化点 |

> **なぜ系統共存が多峰問題に効くか**: 単純な top-k 選択では最初に見つかった最適解周辺に個体が集中し、Himmelblau（最適解4箇所）のような多最適解問題で致命的になる。系統選択は (1) f 値の良い順に走査し、(2) 既保護系統との距離が全て `niche_radius_ratio × span` を超える候補のみ追加、(3) `n_elite_max` 個で終了する。これにより空間的に離れた複数の最適解周辺に独立した感染系統が自然形成され、飛沫感染の引力対象となる。

### 1 世代の動作フロー

```
1. 系統共存: ニッチエリート抽出（飛沫感染の引力 pool）
   └─ f 値の良い順に走査し、既存系統から niche_radius_ratio × span (=0.1×span) 以上離れた個体だけを
      最大 n_elite_max (=6) 個保護

2. スピルオーバー判定（停滞時の情報化リスタート、basin 回避版）
   └─ no_improve ≥ restart_no_improve_threshold (=300) かつ
      f_best / |f_init| > restart_quality_rel_floor (=1e-8) のとき発動（相対 8 桁進捗未満なら spillover、それ以下なら precision とみなし保護）。
      ① 発動時に集団の niched elite を永続アーカイブへ harvest、放棄 basin の重心を記憶（集団免疫メモリ）。
      ② **情報化再播種**（盲目 Uniform でなく探索結果を活用）:
         • 確率 ir_archive_frac (=0.5): 生存アーカイブ系統まわりの σ=ir_reignite_sigma_ratio×span (=0.05) ガウスで再着火
         • 残り: 記憶 basin の ir_repel_radius_ratio×span (=0.1) 内を rejection で避けた Uniform（未感染=未踏域へ）
      ③ 連続失敗回数 (consecutive_failed_spillovers) で動作切替:
         • streak < 2: 上記情報化再播種（best は保持）+ 軸 sweep + σ ← σ_init×0.3
         • streak ≥ 2 かつ f_best > 1e-2: **ベイスン乗換え**（best も破棄、全 n_pop 再播種、σ を σ_init にリセット）
      ─ ベイスン乗換えで F24 双漏斗 / F04 rugged separable から脱出。f_best ≤ 1e-2 のとき乗換え抑制 → F13 ridge / C01 deep precision を保護

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
- **Spillover / basin switch（情報化リスタート）** — `no_improve ≥ 300` かつ `f_best / |f_init| > 1e-8` で停滞と判定し、best を保持して残りを再播種する（＋境界軸 sweep: 各次元で `lo`/`hi` を試す probe、dim=2 で 4 評価）。再播種は**盲目 Uniform でなく情報化**: 発動時に集団の niched elite を永続アーカイブへ harvest し放棄 basin の重心を記憶、再播種スロットの一部 (`ir_archive_frac`) を生存アーカイブまわりで再着火、残りは記憶 basin を斥力で避けた Uniform（集団免疫＝未踏域探索）。連続失敗が `basin_switch_after_failed_spillovers (2)` 回に達し、かつ `f_best / |f_init| > basin_switch_quality_rel_floor (1e-2)` なら **basin switch**（best も破棄し σ を σ_init にリセット）に escalate。quality gate により高精度到達済みの run（F13 ridge, C01 deep precision）での暴発を防ぐ。IPOP-CMA-ES の "restart with larger population" の MC-ESO 版だが、**探索結果（リザーバ・basin メモリ）を再利用する点で盲目 restart と差別化**（2026-06 ablation で盲目 Uniform restart が探索構造を捨てていたことが判明し情報化、BBOB dim2/dim3・CEC2022 dim10 hold-out で有意な regression なしを確認のうえ本体に統合）。

### パラメータ一覧

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `n_pop` | 20 | 集団個体数 |
| `sigma` | 0.2 | 初期探索半径（探索範囲に対する比率） |
| `host_sigma_min_scale` | 0.05 | 接触感染チャネルにおける per-host σ_i スケーリング下限（高品質・高齢の宿主は σ_i = σ × 0.05 まで縮小して精密探索）|
| `empirical_cov_floor` | 0.01 | 接触感染チャネルの異方性 floor の**高い側**（rugged/多峰で安全。固有値比を約 14:1 にクランプ）|
| `cov_floor_low` | 1e-3 | 異方性 floor の**低い側**（悪条件の谷で異方性比 ~1000:1 まで許容）。`cov_floor_low = empirical_cov_floor` で適応を無効化し固定 floor |
| `cov_ratio_lo` / `cov_ratio_hi` | 1e3 / 3e4 | **適応 floor の切替閾値**。集団共分散の素の固有値比（平滑化）がこの範囲で `empirical_cov_floor`⇄`cov_floor_low` を log 補間。実測中央値が ill-cond ≈1e5–1e7・rugged ≈3–600 と桁違いに分離するのを利用 |
| `cov_ratio_beta` | 0.1 | 固有値比 EMA の更新率（rugged の瞬間スパイクを除去） |
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
| `ir_archive_frac` | 0.5 | 情報化再播種で生存アーカイブから再着火するスロット割合（残りは basin 忌避 Uniform）|
| `ir_reignite_sigma_ratio` | 0.05 | アーカイブ系統まわり再着火ガウスの σ（× span）|
| `ir_repel_radius_ratio` | 0.1 | basin 忌避（集団免疫）の斥力球半径（× span、記憶 basin 内の Uniform を rejection）|
| `ir_repel_max_tries` | 20 | basin 忌避の rejection 上限回数（超えたら plain Uniform にフォールバック）|
| `n_elite_max` | 6 | 系統共存の最大数（飛沫感染の引力対象、情報化再播種のアーカイブ容量も兼ねる） |
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

**多解（multi-modal optimization, MMO）報告**: 大域最適解が複数ある関数（C01 Himmelblau=4 / C02 Six-hump=2 / C03 Shubert=18）では、SR（= 1 つでも到達したか）に加え、MC-ESO の「並行的な多解探索」能力を直接計測する。SR は単一目的指標で**いくつ別個の最適解を見つけたか**を見ないため、この強みは SR には現れない。指標は走行後に `history_x`（全評価点）と `benchmark.optima_pos`（既知の全大域最適解座標）から後付け計算され、**最適化器は一切変更しない**（＝ SR は定義上不変）。`core/runner.py:optima_found_mask` / `peak_metrics` が正準実装。

- **Peak Ratio (`pr_1e-2`, `pr_1e-4`)** — 既知 K 個の大域最適解のうち、`f ≤ tol` かつ最近傍割当で半径内に評価点が落ちた解の割合（run 平均）。各評価点は**最近傍の最適解1つ**にのみ帰属させ、近接最適解（Shubert は最小間隔 ~0.88）での二重カウントを防ぐ。
- **MMO Success Rate (`mmo_sr_1e-2`, `mmo_sr_1e-4`)** — K 個**すべて**を見つけた run の割合。
- `n_optima` 列に K を記録。`summary.csv` の `mean_optima_found` / `mean_optima_rate` は従来からの `tol=1e-4` 単一値（後方互換）。
- 注意: 発見は**時間的**（走行中にいずれかの世代で訪れた）で、厳密な「同時保持」ではない。同時並行保持の計測には別途 niching mode が必要（未実装）。

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
| **スピルオーバー＋basin switch** | quality-gated restart、連続失敗 2 回で best 破棄＋σ_init リセット。ill-cond の整列失敗と F24 双漏斗を救済 |
| **情報化リスタート** (`ir_archive_frac=0.5`, 2026-06 統合) | spillover 再播種を盲目 Uniform から**リザーバ再着火＋basin 忌避**へ。診断 ablation で旧 restart が探索構造を捨てていたと判明し情報化。dim2 +1.7pt、dim3/CEC2022 hold-out で有意 regression なし。IPOP 盲目 restart との差別化点 |
| **Drilling mode**（σ_drill_down=0.85） | σ < span × 1e-3 で σ 縮小を強化し浮動小数限界まで追込む |
| **接触感染の経験共分散** (`empirical_cov_floor=0.01`) | 集団経験共分散 `C_pop` の固有分解で接触感染ノイズを瞬間異方化。CMA-ES の rank-μ 学習と異なり履歴累積なし、basin 切替に即応 |
| **適応異方性 floor** (`cov_floor_low=1e-3`, 2026-06) | 異方性の頭打ち（floor）を**集団共分散の素の固有値比**で自動調整。悪条件の谷（比 1e5–1e7）では floor を下げ異方性を解放、rugged/多峰（比 3–600）では高く保ちノイズの偽異方性をクランプ。スケール・シフト不変（f 値非依存）。全35関数で固定 0.01 比 **+2.6pt（85.4→88.0）・回帰ゼロ**、固定 1e-3（F17/C11/C05 で回帰）をも上回る。F02/F10→100%、F18 60→80% 等 ill-cond を改善しつつ rugged を保護 |
| **Drilling 中の空気感染停止** | `σ < span × precision_sigma_ratio` で `air_ratio_eff = 0`。drilling 中の広域ランダム雑音を排除し精度劣化を防止 |

### 検証され不採用となった variant

以下は `quick_check.py` で MC-ESO 統合候補として走らせたが overall 改善を実証できず、コードからも削除された。

| variant | 追加した挙動 | 不採用の理由 |
|---|---|---|
| MC-ESO-A1（per-dim σ close-contact） | 接触感染ノイズを集団 per-dim std で軸別スケール（軸整列の異方化） | F08/F17 では改善するが F14-DiffPowers の BBOB 回転と整合せず致命的劣化。後継の A2（経験共分散版）に置換 |
| MC-ESO-ABD | h2h_CR=0.9 ＋ σ-adapt 停滞ゲートを drilling 中バイパス ＋ 初回 spillover で座標軸 sweep | A_mild ベースと比べて CR=0.9 のため F18/F19 で勝つが F04 で回帰。Wilcoxon でも B/D 単独の有意寄与なし、結局 CR トレードオフに収束 |
| MC-ESO-A_mild_BD | 統合済み MC-ESO ＋ 同上の B/D | F09/F11/F18 での改善と F04/F14 での悪化が相殺しほぼ同等。B/D の overall 寄与なし |
| 旧 A〜N（`use_evolution_path` / `use_pop_covariance` / `use_lifespan_reset` / `use_adaptive_air` / `use_adaptive_h2h_F` / `use_aggressive_niche` / `use_h2h_archive` / `use_local_pair_h2h` ほか） | MC-ESO 初期開発で試した 8 案 | 各案とも単一関数の改善はあるものの 12 関数 SR 合計で baseline 以下、あるいは安全装置を要する構造欠陥（E）で overall を毀損し全削除 |
| **MC-ESO-V2a** (UCB-AOS on 3 channels, 2026-06) | 接触・飛沫・空気の比率を世代毎に UCB ベース AOS で自動調整（credit = 世代内中央値で正規化した Δf）。drilling 中の air 抑制は V1 から踏襲 | BBOB+Custom 35 関数の主指標 SR@1e-10 が V1 24.40 → V2a 22.10（−2.30）。Wilcoxon (n=10, α=0.05) で有意な勝ち 1（F17）に対し有意な負け 3（F06 / F20 / C07）。F23-Katsuura では 0%→70% の劇的改善が出たが overall regression を覆せず削除 |
| **MC-ESO-V2b** (V2a + 4 新チャネル, 2026-06) | V2a に Lévy 超拡散 / 重心組換 (μ-recombination) / 系統間クロスオーバ / 反対称跳躍 (`2·centroid − x_p`) を追加し、UCB の arms を 3→7 に拡張 | SR@1e-10 が V1 24.40 → V2b 19.80（−4.60）。Wilcoxon で有意な勝ち 0、有意な負け 5（F02 / F11 / F18 / C06 / C11）。Lévy 等の大ジャンプが ill-conditioned 関数の precision grinding を妨害。V2a より明確に劣り削除 |

検証ログ: `results/20260515_150803_ベースライン_quick/dim2/{summary,wilcoxon}.csv`、V2 系は `results/20260605_190353_v2_compare_all_quick/dim2/{summary,wilcoxon}.csv`

### 診断 ablation（チャネル vs リスタートの寄与分解, 2026-06）

「性能はチャネル/系統共存でなく頻繁なランダムリスタート由来では」という疑義を検証するため、`mceso_ablations.py` に 2 つの**診断用** variant を追加（改善候補ではなく寄与の切り分け用、`quick_check.py` の `_OPTIMIZERS` 常設）。

- **MC-ESO-NoSpill** — チャネル ON / spillover 完全停止（`_maybe_spillover` が常に False）。チャネル単独の到達力を測る。
- **MC-ESO-RandRestart** — spillover・σ適応・drilling・μ+λ greedy は維持し、3 チャネル＋系統共存を**等方ガウス局所探索 1 本**（`x_parent + σ_global·N(0,I)`）に置換。リスタート＋バニラ局所だけで何処まで行くかを測る。

**結果（BBOB24+Custom11, n=10, max_evals=5000, dim=2、平均 SR@1e-10）**: MC-ESO **83.7%** / NoSpill **68.6%** / RandRestart **48.9%**。

- **主動力はチャネル機構**: RandRestart で 83.7→48.9% に激減（Rosenbrock/ill-cond/F02 は 100→0%）。Wilcoxon で MC-ESO が **26/35 関数で有意に優位（負け 0、全 large）**。「リスタートのくじ運で発見」説は棄却。
- **spillover は二次的・限定的**: NoSpill でも 68.6% を維持。MC-ESO が NoSpill に有意優位なのは **7/35（F03/F04/F15/F20/F24/C05/C11 ＝ 多峰・deceptive）**。spillover 発火回数も大半の関数で 0〜1 回（F20=5.6, F24=11.2 のみ「頻繁」）。
- **ただし系統共存は不活性**: 平均 n_elite は大半の関数で ~1.0–1.2（n_elite=1 の世代が 92–99%）。多 basin 保持は F20(1.56)/F24(3.85) でしか発火せず、宣伝機構が 30/35 関数で no-op = **novelty gap**（性能の出所が DE×経験共分散＋IPOP 風 restart で、epidemic 固有の新規性と不一致）。改善は「系統共存の実活性化（永続アーカイブ / crowding）＋ restart の情報化（basin 忌避）」に的を絞る。

検証ログ: `results/20260605_200551_diag_restart_ablation_quick/dim2/{summary,wilcoxon}.csv`

### 情報化リスタートの統合 / 系統共存活性化の不採用（2026-06）

診断 ablation を起点に2方向の改善を検証し、片方を本体統合・片方を不採用とした。検証はすべて `_on_spillover_start` / `_diversified_reseed` / `_droplet_strain_positions` の拡張フック（既定で RNG 順不変）経由でサブクラス化し quick で測定。

**① 情報化リスタート（IR）→ 本体統合（採用）**
- 動機: 診断で「リスタートは実寄与あるが**無情報**（best 以外を全域 Uniform 再播種）」と判明。**リザーバ再着火**（spillover 時に niched elite を永続アーカイブへ harvest し一部スロットを系統まわりで再生成, `ir_archive_frac`/`ir_reignite_sigma_ratio`）＋**集団免疫忌避**（放棄 basin 重心を記憶し残り Uniform を斥力 rejection, `ir_repel_radius_ratio`/`ir_repel_max_tries`）で情報化。
- 結果（平均 SR@1e-10）: dim2 83.7→**85.4（+1.7pt）**（改善 C09+40/F23+20/F20+10/C11+10、悪化 F10/F19 各−10）、dim3 flat・有意差0、CEC2022 dim10 hold-out は medf≈同点で composition 系に有意 best_f 改善5・回帰0。**全次元・hold-out で有意 regression なし**を確認し本体に統合（`MultiChannelEpidemicOptimizer` 既定挙動）。診断 `MC-ESO-RandRestart` は旧盲目 Uniform restart を pin して比較基準を維持。

**② 系統共存の実活性化（SC）/ IR+SC 併用（IRSC）→ 不採用**
- 動機: 診断のもう一つの的「系統共存が不活性（live n_elite≈1）」。永続アーカイブ（品質ゲート `sc_quality_band` で自己調整）から飛沫 donor を抽選し多 basin 引力を常時化。
- 結果: SC 単独は dim2 net-neutral（−0.3pt、F04/F23/C09 改善と F17/F24/F19 悪化が相殺、低速化）。IRSC は dim2 raw 最良（86.0, +2.3pt; F13/C05/F19 で超加法）だが **F17/F24 有意回帰**を持込み、drilling-mode 抑制でも overall は上がらず（85.7）。決定打は**汎化失敗**: dim3 で有意回帰3（F08 Rosenbrock−30 含）、CEC2022 dim10 で medf 2624 vs 1650 と大崩れ。
- 含意: **「系統共存（epidemic 固有の宣伝機構）を活性化しても性能に結びつかない」が全次元・hold-out で確定**（novelty gap）。SC/IRSC 関連コード（`mceso_sc.py`/`mceso_combo.py`）は削除。

検証ログ: `results/20260605_200551_diag_restart_ablation_quick/`、`results/20260610_103749_ir_verify_quick/`、`results/20260610_113659_sc_verify_quick/`、`results/20260610_*_irsc_verify_quick/`、`results/20260612_*_{irsc_drill,gen_dim3,gen_cec}_quick/{dim2,dim3,dim10}/{summary,wilcoxon}.csv`。**n=10 は低信号 → IR の本判定は n=100 本実験で行う。**

### MC-ESO-Endemic（多解 niching variant, 2026-06）

「ウイルス模倣で複数最適解を探索する」という当初の主張が peak-ratio 実測で**実体を持っていない**（MC-ESO は SR@1e-10=100% でも PR@1e-4 が Himmelblau 0.28 / Shubert 0.06）と確定したのを受けた逐次 niching。**2026-06、base `MultiChannelEpidemicOptimizer` 本体に統合**（`_basin_exhausted` で「掘り切った」を検知し restart）。`mceso_niching.py:MCESOEndemic` は後方互換エイリアス（base と同一）。**デフォルト MC-ESO 自体が多解探索を行う**ため、本実験では単一手法 `MC-ESO` として評価。`exhausted_no_improve_mult` を巨大値にすれば niching 無効化（純粋単一 basin）も可能。統合後の全35関数 SR@1e-10 ≈ 87.7%（純粋単一 basin 比 F13 −10 のみ）、多解は C01 PR@1e-2 0.60 / C02 MMOsr 90% / C03 0.19。

**設計＝逐次 niching（並行 crowding ではない）＋ 精度ゲート**。当初 crowding／per-host σ を試作したが、**1集団で複数 basin を同時に深精度化できず SR@1e-10 が崩壊**した。SR は本研究の主指標で犠牲不可。そこで BIPOP-CMA-ES が PR と SR を両立する構造、すなわち「**1 basin を掘る→記憶→既発見 basin から斥力で離れて restart→次の basin を掘る**」を採る。各 basin は base MC-ESO の単一σ drilling をそのまま使う。

**SR 死守の鍵＝2レジーム化（`_basin_exhausted` で切替, スケール不変）**。SR を一切落とさないため、**basin を掘り切る前は base と完全に同一挙動**（`_spillover_should_fire` / `_spillover_basin_switch` / `_diversified_reseed` すべて `super()` に委譲）＝掘りかけの best basin を絶対に破壊しない。掘り切った後に**初めて**多解探索を起動：(1) 精度 quality-gate を外し停滞ごとに restart、(2) basin-switch（集団を捨て σ_init で新領域に全コミット）、(3) reseed を reignition OFF＋細斥力（0.02×span）に切替え新領域へ。

**「掘り切った」の検知はスケール・シフト不変**（最適値非依存）。`_basin_exhausted` = **σ がフロア（`σ ≤ exhausted_sigma_tol × span × sigma_floor_ratio`）に到達**（これ以上細かく掘れない、探索域 span 相対の判定で f 値を一切見ない）**かつ** `no_improve ≥ exhausted_no_improve_mult × restart_no_improve_threshold`（フロア到達後の停滞）。この時 basin はアルゴリズムの分解能限界＝base が続けても同じ深さで詰まるので、離脱しても SR を失わない。
> 初期版は secure 判定に **絶対 floor 1e-11** を使ったが、これは「最適値 0」という BBOB 正規化依存で一般関数に通用しない（指摘により撤回）。σ ベース検知は f_opt も「optimum=0」も仮定しない。停滞許容 `exhausted_no_improve_mult=3` は F14(DiffPowers) 等の平坦 basin で base の遅延 breakthrough を取りこぼさないための粘りマージン（小さすぎると F14 で SR 低下）。

**結果（quick n=10, max_evals=5000, dim2, 全35関数, MC-ESO → MC-ESO-Endemic）**:
- **SR@1e-10: 35関数すべてで Endemic ≥ base（回帰ゼロ）。平均 85.4% → 86.3%（F02/F11/F19 で +10、exhausted basin からの restart が失敗 run に再挑戦の機会を与える）**。
- 多解（多大域）関数の改善:

| 関数 | K | SR@1e-10 | PR@1e-2 | PR@1e-4 | MMOsr@1e-4 |
|---|---|---|---|---|---|
| C01 Himmelblau | 4 | 100% → **100%** | 0.28 → **0.62** | 0.28 → **0.53** | 0% → 0% |
| C02 Six-hump | 2 | 100% → **100%** | 0.75 → **1.00** | 0.60 → **0.95** | 20% → **90%** |
| C03 Shubert | 18 | 100% → **100%** | 0.06 → **0.17** | 0.06 → **0.16** | 0%（18 global が ~760 local に埋もれ hard だが約3倍, BIPOP 0.11 超）|

**SR を全35関数で一切犠牲にせず（むしろ +0.9pt）**、全多大域関数で多解探索を改善（C01 約2倍・C02 ほぼ完璧・Shubert 初改善）。「多解探索」が SR 無犠牲で実体化。

検証ログ: `results/20260616_141021_peakratio_baseline_quick/`（baseline 全手法）、`results/20260624_113046_endemic_sigexh_quick/`（**σ-exhaustion 確定版・全35関数 SR 回帰ゼロ**）。**n=10 は低信号 → 本判定は n=100 本実験で。** 撤回した試作: crowding=`20260616_145202_endemic_v3`、SR を落とした always-restart=`20260623_145326_1ec0bf0`、絶対floor版=`20260624_103753_endemic_secured`。
