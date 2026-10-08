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
| **Restart-Lander** | 記憶なし多スタート | **null（下限ではなく「協調なしで届く線」）**。一様再起動 ＋ 等方降下だけ。`scripts/niching/niching_baseline.py` から `--methods Restart-Lander` で回す（下記）|

---

## CMA-ES

共分散行列適応進化戦略。現在の探索分布の「形」を共分散行列として学習し、楕円形の探索が可能。収束を検出したら最良点からタイトな sigma で再スタートするマルチスタートを実装済み。実装は `core/optimizers/cmaes.py`。

| パラメータ | 値 | 意味 |
|---|---|---|
| `sigma0` | `0.2 × (hi - lo)` | 初期探索範囲（`main.py` / `quick_check.py` が問題ごとに付与。クラス既定値は 1.0）|
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
| **NMMSO** | `nmmso.py`（`pynmmso` ラッパ）| swarm_size `10·D`（**公表設定。2026-09-17 までは直書き 10 だった。下記の注意を読むこと**）| スウォームの分裂・併合でニッチ数を自分で決める。再実装でないので「ベースラインの実装が悪い」という反論を封じられる |
| **Repel-CMA-ES** | `restart_cmaes.py:RepellingCMAESOptimizer` | repel_coverage 0.2 / repel_gamma 0.9 | restart の best を taboo 点にし、その球内に落ちた候補を引き直す。半径は「taboo 集合が箱の `repel_coverage` を塞ぐ」体積条件から決まり、restart が増えるほど自動で縮む |

実装上の注意:

- **【2026-09-17・その137 で修正済み】`swarm_size` の既定は `10 * benchmark.dim`。** 原論文 §V は
  *"the maximum swarm size n = 10D (where D is the number of design parameters)"* と書いており、
  **D=10 では 100、D=20 では 200 が公表設定**（`pynmmso` のライブラリ既定 `4+floor(3·ln D)` ともまた別）。
  **その137 より前は D に依らず 10 を直書きしていた** ＝ **公表設定の 1/D 倍**で、
  **この 1 個のずれで CEC2013 の公表 PR に届かなくなっていた**（その136 の実測）——
  **N19-CF4-10D の PR@1e-5 は `swarm_size=10` で 0.0917（15 run）、`10·D` で 0.3500（5 run）、公表値 0.443。**
  低次元（N04-N10、D=2/3）で公表値と合っていたのは**問題が易しくこのずれに鈍かったから**で、配線が正しかったからではない。
  **`swarm_size` 引数は残してあるので、旧設定を再現したい腕は明示的に `swarm_size=10` を渡す。**
  **【重要】その137 より前に記録された NMMSO の数値はすべて旧既定（`swarm_size=10`）のもの**である
  —— 内訳と関門の結果は [archive/multisolution.md](archive/multisolution.md)（全文は git タグ `archive/multisolution-2026-09-29` の その137 の節）。
- **NMMSO は最大化**なので符号を反転して渡す。`Nmmso.run` は反復の切れ目でしか予算を見ずオーバーランするため、`max_evals` に達した後の `fitness` は**関数を呼ばずに `-inf` を返す**。評価回数は厳密に一致し、偽の点がモードとして報告されることもない。
- **Repel-CMA-ES は de Nobel+ 2024 の近似**。棄却判定を Euclid 距離で行っている（原論文は現在の CMA 計量での Mahalanobis 距離 / σ）。`repel_coverage` も本プロジェクトの選択で、斥力の強さを決める唯一のパラメータなので、これに依存する主張をする前に感度を測ること。
- 多解指標は `final_solutions` だけを見る（[experiments.md](experiments.md#多解報告cec2013-ルール-niching-スイート)）。報告するのは r3pso が全粒子の pbest、NMMSO がモード集合、Repel-CMA-ES が各 restart の best ＋最終集団、Crowding-DE / NCDE が最終集団。
- 低予算では集団サイズが効く。NCDE / Crowding-DE / r3pso の既定 `n_pop=30` は 2D・5000 評価で 166 世代しか回らないので、負けが手法のせいか設定のせいかは予算を変えて確かめる必要がある。
- **【2026-09-18 その141】`r3pso` の `n_particles=30` は公表設定から外れている（既定は未変更）。** Li 2010 §VI-B は母集団を**定数ではなく帯**で与えており、易しい関数（1e4 評価）で 20-50、Shubert 2-D で 200-500、**D=8〜20 では 300-800**（本文は `analysis/mmo2024/e141/refs/r3pso_li2010.txt.gz`、タグ `archive/multisolution-2026-09-29` 内）。実測でも CEC2013 の 5 関数 10 seed で **300 にすると PR@1e-5 が +0.1233（p=1.757e-7、5 関数すべてで正）**。**慣性・加速係数は論文と一致**（χ=0.7298 / φ=2.05 の構成係数形）＝ 外れているのは母集団 1 個だけ。**既定を変えるかは未決**（変えると記録済みの `r3pso` の値が全部「旧既定のもの」になるため、[research_loop.md](research_loop.md) の方針欄 2026-09-11 (3) の代償の行が要る）。
- **【2026-09-18 その141】`NCDE` の `n_pop=30` は主水準では欠陥ではない。** 同じ測定で 300（`m` は出荷の 1/5 という割合を保って 60）にしても **PR@1e-5 は +0.0150 / p=0.348**。**ただし内訳は打ち消し**で、N15 +0.1000・N17 +0.0875 が有意な一方、**N20-CF4-20D は −0.1250（0/9/1、p=0.0039）で、落ちるのは深さだけ**（PR@1e-3 までは同値、1e-5 で 0.0000。40 万評価 ÷ 300 個体 ≈ 1,333 世代）。**NCDE の原論文（Qu+ 2012）の母集団設定は未取得**（open-access PDF が無い）。

### 記録値がどの配線で取られたか（但し書き。2026-09-21 その150 が追加）

**外部 3 手法の記録値は配線が揃っていない。** 数値を引くときは必ずこの 3 行を併読すること
（**新しい測定はしていない**。出典は git タグ `archive/multisolution-2026-09-29` の `docs/acceptance_topology.md`、その137・その141・その143 の節）。

| 手法 | 記録値の配線 | 公表設定との差が実測でどれだけ効くか | 判定への影響 |
|---|---|---|---|
| **`r3pso`** | **すべて `n_particles=30`**（`r3pso.py:36` の直書き） | Li 2010 §VI-B の公表帯（D=8〜20 で 300-800）の**下端未満**。CEC2013 の N15/N17/N18/N19/N20 で **300 は 30 より PR@1e-5 が平均 +0.1233**（その141）、新 suite D=10 で **400 は 30 より Score が +0.1873**（その143） | **無し。** 記憶なし多スタートへの **16/0/0 は両方の配線で変わらない**（その143。補正は差 −0.4816 の 39% しか埋めない） |
| **NMMSO** | **その137（2026-09-17）を境に 2 つある。** 以後は公表設定 `swarm_size=10·D`、**それ以前の記録値はすべて旧既定 `swarm_size=10`** | 新 suite D=10 で **Score 0.1548 → 0.4144**（その139）＝ 3 本の中で**唯一、判定を動かした** | **有り。** 「niching 手法が 16 問全敗」は崩れ、**16 問中 4 問で NMMSO が null を上回る**（その139・その140）。**測り直したのは新 suite 16 問と CEC2013 の 5 関数だけで、それ以外の記録値は旧既定のまま** |
| **`NCDE`** | **すべて出荷既定 `n_pop=30`** | CEC2013 5 関数の主水準では帰無（**+0.0150、p=0.348**）。**ただし内訳は打ち消し**（N15 +0.1000 / N17 +0.0875 対 N20 −0.1250）。**新 suite では一度も振っていない** | **不明（未測定）。** 16/0/0 は 30 の配線の上でしか確かめていない |

**読み方 1 行**: **3 本のうち公表設定への修正が判定を動かしたのは NMMSO だけ**で、
**`r3pso` は「欠陥は実在して大きいが判定を 1 問も動かさない」、`NCDE` は「新 suite では未検証」**である。

---

## L-SHADE / IPOP-CMA-ES / BIPOP-CMA-ES（外部ライブラリ baseline）

より強力な近代手法を外部ライブラリ経由で `BaseOptimizer` インターフェースに合わせて組み込む。

- **L-SHADE**（`core/optimizers/lshade_port.py`, 論文どおりの numpy 移植。2026-10-06 から登録名 `L-SHADE` はこちら）— SHADE に線形集団縮小を加えた適応的 DE（Tanabe & Fukunaga 2014、CEC2014 優勝）。既定は `n_init_factor=18`（初期集団 `18 × d`）/ `n_min=4` / `memory_size=6` / `p_best=0.11` / `arc_rate=2.6`。旧 mealpy ラッパー（`core/optimizers/lshade.py`）は `L-SHADE-mealpy` として参照用に残る（下の「この作業で見つかった既存の比較手法の問題」）。
- **IPOP-CMA-ES**（`core/optimizers/restart_cmaes.py`, pycma ラッパー）— 収束ごとに集団サイズ λ を倍化して再起動（Auger & Hansen 2005）。
- **BIPOP-CMA-ES**（同上）— 大規模・小規模 2 つの λ regime を予算が釣り合うよう交互に再起動（Hansen 2009）。
- IPOP/BIPOP も CMA-ES と同じく `sigma0 = 0.2 × span` を `main.py` / `quick_check.py` が付与する（クラス側の既定 `sigma0=1.0` は使われない）。

> **注意（再現性）**: pycma 系（CMA-ES / IPOP / BIPOP）が同一 seed でも run ごとに変動していたのは、run 0 のシード 0 を pycma が「時刻から乱数」と解釈していたこちらの包みの不具合で、2026-10-06 に修正済み（下の「この作業で見つかった既存の比較手法の問題」）。修正前の記録値には run 0 の非決定性が残っている。

---

## Restart-Lander（記憶なし多スタート null）

`core/optimizers/restart_lander.py`。**手法の提案ではなく null** ——
niching 手法が「協調している」ことの値打ちを測るための、協調を全部外した対照。

1 run は降下の直列鎖: **箱から一様に 1 点引く → そこから等方 CMA-ES で降下
（`CMA_on=0`、`tolfun = tolfunhist = tolx = 0`、1 降下あたり上限 `descent_budget` 評価）
→ その降下の best 点を報告集合に積む → 予算を使い切るまで繰り返す**。
**記憶・反発・draw の選抜のいずれも持たない。**

| パラメータ | 既定 | 意味 |
|---|---|---|
| `sigma_ratio` | 0.1 | `sigma0 = sigma_ratio × span` |
| `descent_budget` | 12500 | 1 降下の上限評価回数（新 suite の予算 5e5 の 1/40） |
| `iso` | True | 共分散を止めてステップ幅だけ（`CMA_on=0`）|

既定値は `scripts/niching/hunt_coverage.py --null` の offline 降下（`_null_descent`）と同一で、
**降下 1 本は offline 版と厳密に一致する**（`analysis/mmo2024/e110/identity_check.py`、タグ `archive/multisolution-2026-09-29` 内 が
保存ダンプに対して evals / best_f / 着地最適の一致を確認）。
違うのは**鎖にしたこと**だけ ＝ 再起動本数が「予算 ÷ 平均降下コスト」ではなく実際に買えた本数になり、
報告集合が `max(100, 2K)` の上限を通る。

環境変数 `RESTART_LANDER_DUMP` にディレクトリを指定すると、
1 降下 1 行の CSV を書き出す。**列は 2 種類に分かれている**（その115 の教訓）:

- **アルゴリズム自身が実行時に握っている列** —— `descent` / `evals` / `best_f` / `stop` /
  **`x0..x{D-1}`（降下の best 点の座標）**。**報告規則の腕はこの列だけで定義できる**
  （＝ 競技規則 §5 の下で合法）。
- **オラクル列**（真の最適の位置が要る） —— `land_opt`（最近傍の最適の index）/ `dist`（その距離）。
  **採点と診断にだけ使う。規則の定義に使ってはいけない。**

**ダンプの有無・列の増減は探索経路を変えない**（`_dump` は run 終了時に 1 回書き出すだけ）。
座標列を足す前後で `best_f` 列が 32/32 run 完全一致することを その115 が確認している。

## ハイブリッド・最新手法の比較候補（2026-10-06 実装。まだ比較手法のリストには入れていない）

`quick_check.py` の `_OPTIMIZERS` に名前で登録してあり、`--methods` で指定すれば回せる。既定の比較リストには入っていない。どれも評価回数を上限ちょうどで止め、同じシードで同じ結果になることを確認した（BBOB 2D F02、1000 評価）。調査の経緯は [related_work.md](related_work.md) の「比較手法の候補の洗い出し」。

| 名前 | ファイル | 出典 | 原典との突き合わせ | 推測・注意 |
|---|---|---|---|---|
| `EA4eig` / `EA4eig-jSO-IDEbd` / `EA4eig-Simpl` | `ea4eig.py` | Bujok ら CEC 2022 優勝、Biedrzycki（*Evol. Comput.*）の解析 | Biedrzycki の補足資料の MATLAB / C++ を行番号つきで転写。CEC2022 10D・200k 評価で公表値（30 run）と 5 run の揺らぎの範囲で一致（F9 229、F12 164 など） | 原典の不具合 3 件は既定で修正（フラグで再現可）。CMA-ES の条件数 1e14 で全体が止まる挙動は既定で無効（`cma_cond_stop`）。`EA4eig-Simpl` は IDEbd 単体 |
| `LSHADE-SPACMA` | `lshade_spacma.py` | Mohamed ら CEC 2017（3 位） | 元コードは非公開。mLSHADE-SPACMA リポジトリ内に残る元の行に従った。公表値との比較なし | CMA-ES の破綻判定は `eigh` の非正固有値で代用 |
| `AMALGAM-SO` / `AMALGAM-SO-DE` | `amalgam_so.py` | Vrugt, Robinson & Hyman, IEEE TEC 2009 | 自作のずらした CEC2005 相当（10D）で論文と同等以上（Sphere 1e-6 まで 1236 評価、論文 1756） | 単目的版のコードは非公開で、PDF の数式は画像から読んだ。CMA の更新（共有集団の上位から学ぶ）と境界処理は推測で、論文より速い原因かもしれない |
| `HSES` | `hses.py` | Zhang & Shi CEC 2018 優勝 | 作者の HSES.m の MATLAB 移植から | 第 1 段階だけで 20,200 評価を使うので、2D / 5000 では第 1 段階しか動かない（`stage1_frac` は論文外の選択肢） |
| `ICMAES-ILS` | `icmaes_ils.py` | Liao & Stützle CEC 2013 優勝 | Liao の博士論文（Alg. 3–4、Table 4.1） | 内部の CMA-ES は pycma |
| `MOS` | `mos.py` | LaTorre ら BBOB-2010 | 論文の式 1–3 | 成績の測り方に推測があり、Sphere でも配分が DE に寄る |
| `UMOEA-II` | `umoea.py` | Elsayed ら CEC 2016（2 位） | 論文の全文から | 局所探索（scipy `trust-constr` で代用）は予算の最後の 25% のみ。25k 評価では F10 が 1e-8 に届かない |
| `EBOwithCMAR` | `ebowithcmar.py` | Kumar ら CEC 2017 優勝 | 作者の公開 MATLAB を行単位で移植 | 範囲外の点は評価前に修復、SQP は SciPy SLSQP。CMA の重みが f の生の値に比例する癖を残した（`cma_weights="uniform"` で変更可） |
| `HMHH` / `HMHH-random` | `hmhh.py` | Grobler ら CEC 2010 / *Inf. Sci.* 2015 | 論文の表と擬似コードから | 推測が多い（docstring に列挙）。10D F15 ではタブー割当（既定）がランダム割当より悪い |
| `PS-CMA-ES` | `ps_cmaes.py` | Müller ら CEC 2009 | 論文の Alg. 1–2 | 情報交換は 200 世代ごとなので、10D / 25k では一度も起きない（実質 15 個の独立 CMA-ES） |
| `DEPSO` | `depso.py` | Zhang & Xie, IEEE SMC 2003 | 全文が入手できず、要旨と二次資料から | 推測が多い |
| `jSO` | `jso.py` | Brest ら CEC 2017（2 位） | 論文どおりの numpy 移植 | pyade の jSO は論文から外れる（記憶サイズ、初期値ほか）ので使わない |
| `L-SRTDE` | `lsrtde.py` | Stanovov & Semenkin CEC 2024 優勝 | 作者の C++ を行単位で移植。公式 CEC2022 10D の全 12 関数で C++ 実行と平均誤差が一致 | — |
| `jSO-minionpy` / `L-SRTDE-minionpy` | `lib_wrappers.py` | minionpy 1.9.1 | — | 参照用。比較には移植版を使う |
| `IMODE` | `imode.py` | Sallam ら CEC 2020 優勝 | 作者の競技提出コード（`P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark`）を行番号つきで移植。各子の成績は実際に作った演算子に付く（mealpy 版の不具合を解消）。公表値との比較はできていない（CEC2020 10D の予算は 1e6、CEC2022 の IMODE の表は入手できず） | 配分は元コードどおり改善率のみ（`allocation="paper"` で論文の「質と多様性」を推測で再現、未検証）。SQP は SciPy SLSQP で勾配評価も予算に数える。元コードの終了規則（残り < 4·N で停止）を再現するので、予算を少し残して止まる。初期集団 6·D² が大きく、10D / 25k の F15 は mealpy 版より悪い |
| `ELSHADE-SPACMA` | `elshade_spacma.py` | Hadi ら CEC 2018（3 位） | 主催者が公開する作者の MATLAB（`P-N-Suganthan/CEC2018`）を行単位で移植 | LSHADE-SPACMA の 1 世代と EADE の 1 世代を交互に回す。公表値との比較なし |
| `APGSK-IMODE` | `apgsk_imode.py` | Mohamed ら CEC 2021（3 位） | 主催者が公開する作者の MATLAB（`P-N-Suganthan/2021-SO-BCO`）を行単位で移植。作者同梱の CEC2022 結果と定性的に同じ範囲（予算が違う可能性） | 元コードの不具合（IMODE のアーカイブが常に空、など）を意図的に再現（docstring に列挙）。集団 30·D が大きく、10D / 25k の F10 / F15 は弱い |
| `SPS-L-SHADE-EIG` | `sps_lshade_eig.py` | Guo ら CEC 2015 優勝 | 共著者のコード（`ChinChangYang/RobustOptimizer`）の移植 | 名前の「自己最適化」は競技の関数ごとの事前調整なので再現せず、作者の未調整の既定値を使う。元コードの癖 2 件は既定で残す（`fixed_cauchy_table`、`archive_parent`） |
| `CoBiDE` | `cobide.py` | Wang ら *Appl. Soft Comput.* 2014 | 論文と公式 `CoBiDE.m` | EA4eig の中の CoBiDE とは 5 点違う（境界処理、共分散の安全策、F の引き直し、交叉、集団の縮小）。単体版は元論文に従う |
| `IMODE-mealpy` | `lib_wrappers.py` | mealpy 3.0.3 | — | mealpy の実装に不具合（評価後に演算子の割当を引き直し、成績が別の演算子に付く）と SQP 欠落。10D F10 で 10〜67。比較に使うなら移植が要る |
| `LSHADE-cnEpSin` | `lib_wrappers.py` | mealpy 3.0.3 | — | 集団を論文の 18·D → 4 に直した。**mealpy 3.0.3 は numpy ≤ 1.26 を要求し、numpy 2.4.6 の環境には入らない**（入る 3.0.2 には `mealpy.sota_based` が無い。2026-10-09 確認） |
| `NGOpt` / `NG-Portfolio` | `lib_wrappers.py` | nevergrad 1.0.12 | — | NGOpt は遅い（10D で 1 run 40〜240 秒）。Portfolio は Sphere でも弱い（ライブラリ側の性質）。**numpy 2.4.6 の環境では NGOpt が内部の代理モデルで落ちる**（`metamodel.py` の `float(model.predict(...))`、2026-10-09 確認）。NG-Portfolio は動く |

**この作業で見つかった既存の比較手法の問題（2026-10-06 に修正。修正前の記録値は測り直し待ち）**
- `cmaes.py` / `restart_cmaes.py`（CMA-ES / IPOP / BIPOP）→ **修正済み**（`pycma_seed` で 0 だけを固定の非ゼロ値に置き換え、最後の世代は残りの予算分だけ評価して止める。seed 100・200 の run は予算内の best が修正前と完全一致することを 3 手法 × 3 関数で確認）。修正前の内容: (1) run 0 のシードが 0 になり、pycma が 0 を「時刻から乱数」と解釈するので run 0 だけ再現しない（[findings.md](findings.md) の「cma ライブラリは run 間で決定的でない」の少なくとも一部はこれ）。(2) 最後の世代を丸ごと評価するので、評価回数の上限を最大で集団サイズ − 1 回超える。
- `lshade.py`（L-SHADE, mealpy）→ **`lshade_port.py`（論文どおりの numpy 移植）に差し替えた**（`quick_check.py` の `L-SHADE`。旧版は `L-SHADE-mealpy`）。移植は 10D の F08 9.7e-10、F10 3.0e-10、F12 3.3e-3、F15 5.6（5 seed の median、jSO の移植と同水準）。旧版の説明: mealpy は F / CR を NumPy の共通乱数から引くが、実行器はそれを初期化しないので、シードから再現しない可能性がある。10D の F10 で誤差 2.0e3 と、論文どおりの jSO / L-SRTDE の移植（1e-10 以下）に比べて大幅に弱い。
- 手元の CEC2022（ioh）は、公式の C コードと比べて F3（Schaffer F7）・F5（Levy）が最適解から離れた点で値が違い、F9 もわずかに違う（f* は同じ）。
