# 関連研究 — 低次元多峰（multimodal / niching）

調査日 2026-08-30。目標を「低次元多峰性」に設定するにあたり、多解探索（niching）分野の標準ベンチマーク・指標・SOTA と、MC-ESO の各機構に対する先行研究を洗い出した。MC-ESO 側の現状は [mceso.md](mceso.md)、採否の履歴は [history.md](history.md) を参照。

---

## 分野の構図

多峰最適化（MMO）は「1 回の run で複数の大域解を得る」問題設定で、niching はそのための多様性維持機構の総称。系統は 5 つ:

| 系統 | 代表 | 機構 |
|---|---|---|
| 古典 niching | fitness sharing (Goldberg & Richardson 1987), crowding (De Jong 1975 / Thomsen 2004), clearing (Pétrowski 1996), speciation | niche 半径で適応度を割る・近傍と置換する |
| 位相ベース | [HillVallEA](https://arxiv.org/pdf/1810.07085) (Maree+ 2018), NBC/NEA2 (Preuss) | 2 点間の線分をサンプリングして「間に山があるか」で basin を判定（hill-valley 検定）。半径パラメータが要らない |
| 斥力・タブー | [RS-CMSA-ES](https://direct.mit.edu/evco/article/25/3/439/1047/Multimodal-Optimization-by-Covariance-Matrix-Self) (Ahrari+ 2017), RS-CMSA-ESII (2021) | 発見済み解を taboo 点として部分集団を反発させる |
| 逐次 niching | Beasley+ 1993, sequential niching memetic (2011) | 1 解を見つけたら landscape を derating して再探索 |
| 多目的変換 / 代理モデル | MOMMOP, SAKT-MMEA (2023), [BSP-SMs-NEA](https://www.sciencedirect.com/science/article/abs/pii/S2210650225000641) (2025), [APDMMO](https://arxiv.org/pdf/2503.18066) (GECCO 2025) | 多解性を第 2 目的に変換 / 高価な評価を代理モデルで置換 |

DE 系の最近の総説は [Advancements in Multimodal Differential Evolution](https://arxiv.org/abs/2504.00717)（Chauhan+ 2025、2017-2024 を対象）。挙がっている未解決課題は多様性と収束の両立・スケーラビリティ・**niching パラメータ依存の削減**。

---

## 標準ベンチマークと指標 — 本プロジェクトの設定と食い違う

| 項目 | 分野の標準 | 本プロジェクト |
|---|---|---|
| ベンチ | [CEC2013 niching](https://titan.csit.rmit.edu.au/~e46507/cec13-niching/competition/cec2013-niching-benchmark-tech-report.pdf) 20 問（1D-20D、大域解 1〜216 個）。GECCO 2024/2025 は Ahrari+ の[新 tunable suite](https://dl.acm.org/doi/10.1145/3638529.3654016)（16 問、scalable composite）に移行 | BBOB-24（大域解 1 個）＋ Custom C01-C11（2D、多解は C01/C02/C03 のみ）|
| 予算 | 2D の易しい問題で 5e4 評価、難問は 2e5〜4e5 | 2D で **5e3 評価**（1/10）|
| 精度 | 精度水準 ε は 1e-1〜1e-5。**ε ≥ 1e-5 なら結果は ε に依存しない**とされ、それより深い精度は測らない | 主指標 **SR@1e-10** |
| 指標 | PR（発見した大域解の割合）、SR（全解発見率）、static F1（報告解のうち真の大域解の割合との調和平均）、**dynamic F1 = F1 を予算で積分した anytime 指標**（GECCO 2018 以降） | SR@各精度 / evals_succ_mean / Wilcoxon ＋ PR・MMOsr（後付け計算）|

指標側の批判も出ている。[Zhang & Wang 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11505590/)（Biomimetics）は「PR は出力の冗長性に無関心」として F 値（precision/recall の調和平均）を推し、さらに**解の抽出そのものに niche 半径やピーク高さの事前知識が要る**点を問題にして hill-valley ベースの peak identification（HVPI）を提案している。Robust Peak Ratio（半径設定を不要にする非二値の成功判定）も提案済み。

**帰結**: 「低次元多峰」を主軸に据えるなら、CEC2013 niching（低次元サブセット）の導入は実装以前に必要。いまの Custom 3 問（C01/C02/C03）では分野の土俵に乗らない。→ 2026-08-30 に 2D/3D サブセット N04-N10 を導入済み（[experiments.md](experiments.md#cec2013-niching低次元多峰-n04-n10)）。一方、予算と精度の設定は分野と真逆で、そこは**あえてずらした設定として主張できる**（下記「空き地」）。

---

## 到達水準

GECCO'17 niching 競技（CEC2013 ベース）の上位は RS-CMSA が平均 PR **0.856**、RLSIS 0.822、NEA2+ 0.810。[HillVallEA](https://arxiv.org/pdf/1907.10988)（AMaLGaM-Univariate 核）は 0.847 で僅差 2 位相当。その後 RS-CMSA-ESII が過年度優勝手法を上回ると報告している。つまり **CEC2013 は上位手法が PR 0.85 前後で飽和**しており、新 suite（GECCO 2024）はこの飽和を解消するために作られた。

参考として、[Cano+ 2022 "Out of the Niche"](https://www.mdpi.com/2227-7390/10/9/1494)（Mathematics）は multistart 直接探索で複数大域解を狙う路線を示している。本プロジェクトの NM-Restart 下限ベースライン（[baselines.md](baselines.md)）と同じ発想で、低次元では強い比較相手になる。

### CEC2013 の合成関数 F11-F20（対象集合を広げる案 (A) の実額。その70 で実測）

**登録済みの N04-N10 は suite の易しい側**で、正規予算では NMMSO が実質解く（最小 0.984、その50）。
文献が headroom を名指ししているのは**合成関数 F11-F20**、とくに **F18/F19（10D）と F20（20D）**
（[RLEMMO, arXiv 2404.08242](https://arxiv.org/pdf/2404.08242)、[arXiv 2511.17571](https://arxiv.org/pdf/2511.17571)）。
suite の仕様は下表（**公式表 = 参照実装 `mikeagn/CEC2013` の `get_*`、ioh = このリポジトリが既に依存している
`ioh` パッケージ。その70 で 1 セルずつ照合し、K / rho / MaxFEs / 箱 / f_goptima は全 10 関数で完全一致**）:

| | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 合成核 | CF1 | CF2 | CF3 | CF3 | CF4 | CF3 | CF4 | CF3 | CF4 | CF4 |
| 次元 D | 2 | 2 | 2 | 3 | 3 | 5 | 5 | **10** | **10** | **20** |
| 大域解 K | 6 | 8 | 6 | 6 | 8 | 6 | 8 | 6 | 8 | 8 |
| rho | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| MaxFEs | 2e5 | 2e5 | 2e5 | 4e5 | 4e5 | 4e5 | 4e5 | 4e5 | 4e5 | 4e5 |

**K は 6-8 しかない。** 登録済みの難所（N08 K=81 / N09 K=216）とは難しさの軸が違い、
**F11-F20 は「K が大きい」ではなく「D が大きい」問題**。箱は全関数 [-5,5]^D、f_goptima は全関数 0。
**ゴール文（「K の大きい関数」）とはずれるので、対象集合を替えるならゴールの言い回しも替わる** ——
これは実行役ではなく俯瞰の判断。

**公表されている F11-F20 の PR（手法別の実額）は、この回では取れなかった。**
このサンドボックスの egress は `arxiv.org` / `pmc.ncbi.nlm.nih.gov` / `semanticscholar.org` /
`dl.acm.org` を全部塞いでおり（CONNECT が 403）、通るのは `raw.githubusercontent.com` と PyPI だけ。
WebSearch は通るが、要約は「F11-F20 の表は取り出せない」と返した。
**手に入っている集計値は上の GECCO'17 の 20 関数平均（RS-CMSA 0.856 ほか）だけで、
これは F11-F20 単独の額ではない。** 取りに行く経路は 2 つあり、どちらも次の回で実行できる:
(1) **GitHub Actions は egress 制限が違う**ので、`run.yml` に 1 ステップ足して PDF を取り、
成果物として持ち帰る（`mode: collect` と同じ経路、17 秒）。
(2) 手法側の公開実装リポジトリ（`raw.githubusercontent.com` は通る）に結果 CSV があればそこから読む。

---

## MC-ESO の各機構に対する先行研究

| MC-ESO の機構 | 先行 | 判定 |
|---|---|---|
| 系統共存（`niche_radius_ratio` で離れたエリートを保護）| fitness sharing / clearing の半径系（1987-1996）| 新規性なし。しかも**半径依存という既知の弱点側**の設計。分野は半径不要化（hill-valley, ring topology PSO）へ動いている |
| 逐次 niching（掘り切り → 記憶 → 斥力 → 次の basin）| Beasley+ 1993 の系譜。既知の欠点は半径依存と derating による偽の局所解生成 | 発想は既存。ただし MC-ESO は derating せず斥力再播種型なので、実体は下の斥力系に近い |
| 情報化リスタート（放棄 basin の重心を記憶し斥力で避ける）| **[de Nobel+ 2024, PPSN XVIII](https://arxiv.org/abs/2405.01226)** — CMA-ES に repelling 機構を付けて basin 再訪（Coupon Collector 問題）を防ぐ。RS-CMSA の taboo 点も同型 | **直接の先行例あり**。差別化を主張するなら機構でなく測定（どの予算帯でどれだけ再訪が減るか）で示すしかない |
| σ-exhaustion による「掘り切った」検知 | CMA-ES の停止条件（TolX/TolFun）、HillVallEA の restart 判定 | 相当物あり。f 値非依存・span 相対という実装上の性質は残る |
| 深精度ゲート（掘り切る前は多解探索を起動せず SR@1e-10 を死守）| **相当物を確認できず**。MMO 側が ε ≥ 1e-5 で打ち切るため、そもそもこの要求が立たない | 現時点で唯一の「空き」。ただし「誰も測っていない」は「価値がある」の証明ではない |
| 多チャネル（疫学メタファ）| CVOA / EOSA / CVO / COVO / H5N1 / COVIDOA | いずれも**niching 手法ではない**。多峰関数を「ベンチマークの一種」として使うだけで PR / CEC2013 での評価はしていない。メタファ路線は [Sörensen 2015](https://onlinelibrary.wiley.com/doi/abs/10.1111/itor.12001)、[Aranha+ 2022](https://link.springer.com/content/pdf/10.1007/s11721-021-00202-9.pdf)、Camacho-Villalón+ 2023 で強く批判されており、メタファの新しさを新規性として出すのは不可 |

---

## QD の欠点と、QD 以外の多様解定式化（2026-08-31 調査）

「多様解」と言うとき QD（MAP-Elites 系）が想起されがちだが、定式化は 5 種類あり QD はその 1 つでしかない。

| 定式化 | 何を求めるか | 多様性の定義 | 代表 |
|---|---|---|---|
| niching / MMO | f の等価な大域解を全部 | 探索空間の距離（rho）| NMMSO, RS-CMSA, HillVallEA, NCDE |
| **QD** | 行動空間の各セルで最良 | **人が設計した行動記述子 (BD)** | MAP-Elites, CMA-ME, CMA-MAE |
| **EDO**（進化的多様性最適化）| f ≥ 閾値を満たす解を k 個、互いに最も離して | 明示的な多様性指標 | Neumann / Bossek 系（理論解析あり）|
| k-diverse near-optimal | 近最適解の k 個集合 | Hamming 距離等 | MAXDIVERSEKSET, DiversiTree, MGA |
| 多峰多目的 (MMOP) | 同じ Pareto front に写る複数の Pareto set | 決定空間の距離 | MO_Ring_PSO_SCD 等 |

### QD の欠点（文献で確認できるもの）と本プロジェクトでの検証状況

| 欠点 | 検証状況 |
|---|---|
| **細胞内の精度が出ない**（セル内最良はその basin の最適解ではない）| **確認済**（2026-09-01）。素の MAP-Elites は niching スイートで PR@1e-4 = 0.00、SR@1e-10 = 0%、best_f ≈ 1e-2 |
| **サンプル効率が悪い**（QD 論文の予算は 10⁵〜10⁷ 評価）| **確認済（間接）**。5000 評価では上記のとおり成立しない |
| **BD 設計に全面依存**。事前知識が要り、変えれば結果も変わる | **未測定**（次に測る）。[Vector Quantized-Elites](https://arxiv.org/html/2504.08057v1) 2025 が無教師 BD で対処を試みている |
| BD 次元の呪い（グリッドが指数的、CVT で緩和するが実用 5〜10 次元）| 未測定 |
| ノイズ下でエリートが楽観バイアス（まぐれ評価が居座る）| 未測定（`--noise` は実装済み）|
| QD-score がアーカイブ解像度・境界に依存し論文間比較が難しい | 未測定 |
| 多様性が BD 空間の話で、パラメータ空間の多様性を保証しない | 未測定 |

### QD より筋が良い場面

- **EDO** — 「品質基準を満たす解を k 個、互いにできるだけ違う形で」。BD 設計が不要で多様性が明示的目的。**ゲームバランスの実務要求に最も近い**（勝率 48〜52% を満たす配置を複数）
- **niching** — 精度が出る、BD 不要。ただし等高の大域解を仮定
- **k-diverse near-optimal（厳密解法）** — 離散・組合せなら QD より確実
- **後処理の多様選択（DPP / 劣モジュラ）** — 探索は普通にやり、最後に k 個を多様に選ぶ。安上がりで、比較の下限ベースラインとして必ず置くべき
- **高価評価向けの BO 系** — 予算が数百〜数千評価ならこちらの土俵

**注記**: ゲーム分野では QD が事実上の標準なので、EDO 形で立てる場合でも QD は必ず比較相手に置く必要がある。

---

## 多峰の比較手法 — 候補と選定（2026-08-30）

低次元多峰を主軸にする以上、比較相手も多峰用に組み直す必要がある。現有は [baselines.md](baselines.md) の 9 手法で、そのうち多峰専用は NCDE だけ。候補を入手性と工数で並べたのが下表。

| 手法 | 系統 | 入手性 | 工数 | 採否 |
|---|---|---|---|---|
| NM-Restart（現有）| multistart 局所探索 | 実装済み | — | **採用**。低次元では強敵で、[Cano+ 2022](https://www.mdpi.com/2227-7390/10/9/1494) の路線そのもの。下限であって噛ませ犬ではない |
| IPOP / BIPOP-CMA-ES（現有）| restart ES | 実装済み | — | **採用**。peak-ratio 診断で MC-ESO より多解を拾った実績（Himmelblau BIPOP 0.78 vs 0.28）|
| NCDE（現有）| 近傍変異 + crowding DE | 実装済み | — | **採用**。並列 crowding の代表 |
| Crowding DE (Thomsen 2004) | crowding DE | NCDE の `m` を `n_pop` にするだけ | ほぼ 0 | **採用**。素の crowding と近傍変異版の差が出る |
| r3pso / ring-topology PSO ([Li 2010](https://ieeexplore.ieee.org/document/5352335/)) | lbest PSO | 論文のみ、式は単純 | 小（既存 PSO に近傍を足す）| **採用**。「niching パラメータ不要」路線の古典で、ほぼ全ての MMO 論文が比較に置く |
| NMMSO ([Fieldsend 2014](https://github.com/fieldsend/ieee_cec_2014_nmmso)) | 多スウォーム + 分裂/併合 | **[pynmmso](https://github.com/EPCCed/pynmmso) が pip で入る** | 小（pycma / mealpy と同じラッパ）| **採用**。競技上位級を再実装なしで 1 つ確保できる |
| 斥力付き restart CMA-ES ([de Nobel+ 2024](https://arxiv.org/abs/2405.01226)) | taboo 斥力 restart | modCMA ベース、コードは Zenodo | 中（既存 restart_cmaes に taboo を足す）| **採用**。MC-ESO の情報化リスタートの直接の先行例なので、これが無いと「de Nobel 2024 と何が違うのか」に測定で答えられない |
| HillVallEA ([Maree+ 2018](https://github.com/scmaree/HillVallEA)) | hill-valley クラスタリング + core search | C++（make）| 大（ビルド + subprocess 駆動、または簡易版の自作）| **保留**。GECCO'18/'19 優勝で PR 0.847。天井を示すには要るが、まず上の 6 つを揃えてから |
| RS-CMSA-ES / ESII (Ahrari+ 2017/2021) | taboo + CMSA 部分集団 | MATLAB（ResearchGate / COIN-Lab）| 大（MATLAB 依存 or 移植）| **見送り**。GECCO'17 優勝 PR 0.856 だが MATLAB 依存が重い。文献値の引用に留める |
| MOMMOP / LIPS / dADE / SDE / NEA2 / WGraD | 各種 niching | 論文のみ、保守されたコードなし | 各 中 | **見送り**。再実装のたびに実装差の疑義が付く。査読で「DE 系 SOTA が無い」と言われたら 1 つだけ足す |
| CVOA / EOSA / H5N1 等の疫学メタファ手法 | メタファ | 論文のみ | — | **見送り**。niching 手法ではなく PR での評価実績もない。同着想の対照は SaVOA 1 つで足りる |

**既定は 7 手法**: MC-ESO / NM-Restart / IPOP-CMA-ES / Repel-CMA-ES / NCDE / r3pso / NMMSO。1 行 = 答える問い 1 つで選び、系統は「多点 restart・restart ES・斥力 restart・crowding・ring PSO・多スウォーム」を覆う。

上表で採用としながら既定から外したものが 2 つ:

- **BIPOP-CMA-ES** — restart ES の枠が IPOP と二重になる。Repel-CMA-ES は IPOP に斥力を足したものなので、対照として要るのは IPOP。ただし peak-ratio 診断（2026-06）で BIPOP は MC-ESO より多解を拾った実績があるので、「niching 手法でなく強い restart で足りるのでは」という問いを立てるときは戻す。
- **Crowding-DE** — 競合手法ではなく NCDE の ablation（近傍変異の寄与）。機構分析のときだけ回す。

**より高次元の単一解 black-box 向け手法（CMA-ES 単体 / PSO / DE / L-SHADE / SaVOA）は多峰スイートでは回さない**。多解で負けるのは既知で、示しても情報が増えない。手法数を削って浮いた計算は、同じ関数を複数予算で回す方（空き地 1）に充てる: 7 手法 × 予算 3 点は 9 手法 × 予算 1 点とほぼ同コストで、言えることが増える。

### 実装するときの制約

- **報告集合を全手法で揃える**。多解指標は `OptimizeResult.final_solutions` だけを見る（[experiments.md](experiments.md#多解報告cec2013-ルール-niching-スイート)）。新規手法は「最終集団」か「各ニッチ/スウォームの代表」を報告する。NMMSO はスウォーム代表を出すのが自然で、これは本来の設計どおり。
- **評価回数を厳密に止める**。外部ライブラリは自前の予算管理を持つので、NM-Restart と同じくラッパ側で `max_evals` 到達時に打ち切る。
- **低予算では集団サイズが効く**。NCDE / r3pso の既定 `n_pop=30` は 2D・5000 評価だと 166 世代しか回らない。既定値のまま回した結果と、予算に合わせて縮めた結果の両方を出さないと、負けが手法のせいか設定のせいか分からない。

## 空き地の候補

いずれも「誰もやっていない」ではなく「測れば意味のある差が出そう」という基準で選んだ。

1. **低予算 × 多解** — CEC2013 は 2D で 5e4 評価。予算を 1 桁下げたときに上位 niching 手法の順位が保たれるかは、少なくとも競技の公開結果からは読めない。dynamic F1 が予算依存を測る指標として既にあるので、道具は揃っている。本プロジェクトの ECDF/`ecdf_auc` 実装（`core/runner.py`）がそのまま使える。
2. **深精度 × 多解のトレードオフの定量化** — 「全解を ε=1e-4 で拾う」と「1 解を 1e-10 まで掘り、残りを粗く拾う」は別の要求で、後者は文献の指標では測れない。MC-ESO の 2 レジーム設計（掘り切るまで base と同一挙動）はこの要求への答えになっており、既存 niching 手法が深精度を要求されたときどう壊れるかは測定として成立する。
3. **半径パラメータの除去** — `niche_radius_ratio` と `ir_repel_radius_ratio` は半径依存のまま。hill-valley 検定に置換すれば半径を 2 つ落とせるが、検定は線分上に追加評価を要するので低予算設定と正面から衝突する。**この衝突自体が 1 の測定対象**になる。

---

## 未読・未確認

- CEC2013 tech report 原本（PDF のテキスト抽出に失敗）。関数の式・f_goptima・rho・大域解数・MaxFEs は参照実装 `github.com/mikeagn/CEC2013` の MATLAB ソースから取って実装済みなので、残るのは本文の記述（問題の設計意図・推奨実験手順）のみ
- GECCO 2024/2025 suite の仕様書と Python 実装（[配布ページ](https://sites.google.com/view/evopt/projects/gecco2024-mmo) の Google Drive 内）
- RS-CMSA-ESII 本文（taboo 距離の適応則）、HillVallEA の core search 選択則
- Robust Peak Ratio の定義
