# e159 先行調査 — 「既に見つけた解の近傍に再び降りることを抑える」

**2026-09-23 / キュー 1 / 最適化 run ゼロ・追加評価ゼロ・`core/` 不変・腕は作っていない**

調べた手続き（prereg で 1 文に固定）:
**一様多スタート ＋ 局所降下の再起動点の選び方を、既に見つけた解の近傍を避ける向きに変える。**

経路: ローカル egress は arxiv.org / en.wikipedia.org / proceedings.mlr.press とも CONNECT 403（`WebFetch` も同じ proxy）。
**使えたのは (1) 手元の `analysis/**/refs/` 9 本の本文、(2) 検索（要約のみ、逐語は引けない）、(3) CI の `fetch_refs`。**
**逐語で引けたものだけを「逐語」と明記する。**

---

## (i) 進化計算・niching

### RS-CMSA-ES（Ahrari, Deb & Preuss, Evolutionary Computation 25(3):439-471, 2017）— **逐語一致。撤退の根拠**

**手元の本文から逐語で引ける**（Maree+ の Hill-Valley Clustering 論文 arXiv 1810.07085、
`analysis/hm/e71/hvc_1810.07085.txt.gz` の 138-153 行。**第三者による手続きの記述**）:

> Winner of the GECCO'17 niching competition is the repelling-subpopulations (RS-CMSA) algorithm [1, 2].
> In RS-CMSA, instances of the core search algorithm CMSA are randomly initialized and maintain a minimum
> distance to each other, using rejection sampling. When all core search algorithms are terminated, the
> method is restarted, and **the core search algorithms are furthermore kept away from previously located
> global optima.** In this way, easy optima are found first, and the search is later pushed to unexplored
> regions of the search space. **Rejection regions from located optima, referred to as taboo regions, are
> considered to be (hyper-)spheres around the located optimum. The radius of the taboo region grows
> adaptively if the corresponding optimum is located multiple times**, and shrinks again if rejection
> sampling fails too often.

**一致する点（3 つとも一致する ＝ 組み合わせの一致）**:
1. **目的が多解**（複数の大域最適を見つける）。RR-CMA-ES と違ってここが揃う。
2. **手続きが「既に見つけた<u>大域</u>最適の近傍を避けて再起動する」**。腕の置き場そのもの。
3. **適応の形まで一致する** —— **同じ最適が複数回見つかったら半径を広げる**。
   これは その158 が動機づける量（既出に落ちた回数）をそのまま半径に使う設計で、
   こちらが「測定から設計する」と思っていた部分が先に書かれている。

**一致しない点**: 無い。**この 3 点が揃っている時点で、方針欄 2026-09-08 の追記 (1) の撤退基準
（組み合わせが一致し、かつ概念ではなく具体的手続きが一致する）を満たす。**

### RS-CMSA-ESII（Ahrari+ 2021, IEEE TEVC 9555836）— 一致（検索要約のみ、逐語は未取得）

検索の要約によれば、**critical taboo region に入った候補解は<u>評価せずに棄却</u>**され、
**taboo 領域は archive 済みの解を中心に、部分集団の共分散行列で形を与えられる**（＝ 非球形の棄却領域）。
ESII は **archive 済み解の正規化 taboo 距離の適応則**を改良したものと記述されている。
**逐語は引けていないので「確認済み」とは書かない。** ただし ES 本体（2017）だけで撤退判定は足りる。

### RR-CMA-ES（de Nobel+ 2024, arXiv 2405.01226）— **一致は半分。そして<u>残り半分を自分で "future work" と書いている</u>**

**本文は手元にある**（`analysis/mmo2024/e92/refs/rrcmaes_denobel2024.txt.gz`）。逐語:

- **目的が違う**（§2.5）: "The overall goal is to make the CMA-ES more sample-efficient for the goal of
  finding **a single best solution** (i.e., the best local optimum approximation, given a budget B of function
  evaluations), **rather than multiple solutions** representing different local optima (i.e., multiple niches)."
- **手続き**（§4.1, Algorithm 2, eq. 8-11）: tabu 点は **各再起動の収束した重心 m**（＝ 局所最適。大域に限らない）。
  棄却は **CMA-ES の世代内で λ 個を引くたび**に Mahalanobis 距離 `d_m(x,x_T,C^-1)/σ < γ^{n_rej} δ(T)` で行う。
  半径 δ(T) は**大域的な体積予算 c** から決まり（`V(T) = n_T·S/(c·σ_0·R)`）、n_T（同じ tabu 点に収束した回数）で按分する。
- **＝ 一致するのは「既出の盆地を避ける」という向きだけで、対象（局所最適 対 大域最適）も
  作用点（世代内の標本棄却 対 再起動点の選択）も違う。**

**そして §6（Discussion）が、腕のもう半分を<u>やっていない</u>と逐語で書いている**:

> In addition to more effectively preventing a restart from converging to a known basin of attraction, one
> could also utilize the information from prior restarts to more intelligently select the new location and
> initial parameterization of new restarts. **In comparison to, e.g., MLSL, starting points could be selected
> based on which regions of the domain have already been sampled**, which would also cause the repelling
> mechanism to trigger less often during the initial phase of the restarted run.

**＝ 「再起動点の選び方そのものを既訪領域から決める」形は 2405.01226 では未実装で、
同論文はその参照先として MLSL を名指ししている。** 下の (ii) へ続く。

### TRDE-LR（Y. Wang+ 2024）— **題名で一致。GECCO'2024 competition の<u>優勝手法</u>**

題名は **"Tabu Restart-Based Niching Differential Evolution Algorithm with Local Refinement"**
（`docs/acceptance_topology.md` の その92 の節に 2026-09-09 から記録されている）。
**本文は arXiv に無い**（正式題名の API 検索が 0 件、その92 で確認済み）ので手続きの照合はできないが、
**題名の "Tabu Restart" が腕と同語である。**
**＝ この suite の公表 1 位（TRDE-LR）と 2 位（RR-CMA-ES）は、<u>どちらも tabu 再起動を持っている。</u>**

### HillVallEA（Maree+ 2018/2019）— **不一致（時間方向の記憶が無い）**

hill-valley 検定で**初期標本を niche にクラスタリングし、niche ごとに核探索を 1 本だけ立てる**。
「同じ盆地で 2 本走らせない」は満たすが、**既に見つけた最適の記憶を持って後続の再起動を曲げる部分が無い**。
本文は手元にある（`analysis/hm/e71/hvc_1810.07085.txt.gz`）。

### S-CARD-CMSA（Chauhan 2026, arXiv 2607.13764）— **本文取得済み。腕そのものではないが<u>2 つ効く</u>**

IEEE **CEC 2026** niching competition の投稿で、**RS-CMSA-ESII をそのまま基盤にしている。** 逐語:

> Rather than modifying the core search dynamics of RS-CMSA-ESII, S-CARD-CMSA **preserves its sampling,
> covariance adaptation, taboo-region update, restart, and termination mechanisms.**

> RS-CMSA-ESII: **Use covariance-adaptive local search together with repelling taboo regions around
> archived optima.** Selected as the base optimizer because it can adapt to basin shape and supports
> archive-based repulsion.

**(1) 占有の現在形**: **2026 年の競技投稿が、まだ RS-CMSA-ESII の taboo 領域を基盤として使っている。**
＝ 腕は「昔の手法が持っていた」のではなく **いまの最前線が持っている。**
**(2) 報告規則の軸に効く（腕とは別件。俯瞰へ）**: この論文の本体は
**"score-aware density-filtered reporting rule ... balancing robust peak ratio and precision-driven F1-score"**、
すなわち **PR と F1 のトレードオフを<u>報告側で</u>取りに行く規則**である。
**その154 が「報告規則の項は空（現行が Score 最大）」と結論した軸に、外から 1 本新しい公表例が立った。**

---

## (ii) 大域最適化の多スタート理論

### Multi-Level Single Linkage（Rinnooy Kan & Timmer, Math. Prog. 39:27-56 / 57-78, 1987）— **逐語一致。しかも<u>その158 が測った失敗に名前と定理がついている</u>**

**本文が取れた**（DEFT-FUNNEL, arXiv 1912.12637 §2 が MLSL を書き下している。
`analysis/mmo2024/e159/refs/deft_funnel_mlsl_1912.12637.txt.gz` の 123-186 行）。逐語:

> MLSL aims at avoiding unnecessary and costly local searches that culminate in the same local minima.
> To achieve this goal, sample points are drawn from an uniform distribution in the global phase and then
> **a local search procedure is applied to each of them except if there is another sample point or a
> previously detected local minimum within a critical distance with smaller objective function value.**

Algorithm 2.1 の 6 行目（＝ 開始規則そのもの）:

> `if ∄x : ‖x − x_i‖ ≤ r_k and f(x) < f(x_i) then  L* = L* ∪ LocalSearch(x_i)`

臨界距離 `r_k = π^{-1/2} ( Γ(1 + n/2) · m(S) · σ · log(kN) / (kN) )^{1/n}`（式 3。`m(S)` は S の Lebesgue 測度）。

**そして その158 が測った量に、この論文は名前と定理を与えている**:

> • **Error 1.** The same local minimum x\* has been found after applying local search to two or more points
>   belonging to the same region of attraction of x\*.
> • Error 2. The region of attraction of a local minimum x\* contains at least one sampled point, but local
>   search has never been applied to points in this region.
> • **Property 1.** (Theorem 8 in [26] and Theorem 1 in [27]) If σ > 4 in (3), then, even if the sampling
>   continues forever, **the total number of local searches ever started by MLSL is finite with probability 1.**
> • **Property 2.** (Theorem 12 in [26] and Theorem 2 in [27]) If r_k tends to 0 with increasing k, then any
>   local minimum x\* will be found within a finite number of iterations with probability 1.

**＝ その158 の「既出に落ちる降下」は MLSL の <u>Error 1</u> で、その抑制は 1987 年に
「σ > 4 なら開始回数は確率 1 で有限」という定理つきで解かれている。**

**この harness の `Restart-Lander` との関係が、この撤退でいちばん重い 1 行**:
**`Restart-Lander`（一様多スタート ＋ 局所降下、記憶なし）は <u>MLSL から開始規則（Algorithm 2.1 の 6 行目）を
外したもの</u>である。** ＝ **腕を作るとは「MLSL の filter を入れ直す」ことであり、
1987 年の手続きへ戻る操作にほかならない。**

**一致しない点**: MLSL の規則は **`f(x) < f(x_i)` という<u>より良い</u>点の存在**で止める形で、
**「大域最適に届いた降下」に条件づけていない**（大域と局所を区別しない）。
**その158 の測定はこの条件づけの上にあるが、<u>手続きの側はこの差で新しくならない</u>** ——
MLSL の規則に「既発見の大域最適だけを中心にする」を足した形は、そのまま上の RS-CMSA である。

### Boender & Rinnooy Kan 1987（Bayesian stopping rules）— **既に 4 件目の撤退として記録済み**

（`research_loop.md` の「占有されている領域」、2026-09-05）。未発見数の推定側であって、
再起動点を曲げる側ではないが、**同じ一族・同じ著者の 1987 年の仕事**である。

---

## (iii) QD / novelty

- **novelty search の archive** は「どこを訪れたかの記憶」で、**cycling（同じ領域に戻る）を防ぐために存在する**
  （検索要約: "a memory of which points in the feature space have been visited... Having only a population
  would lead to cycling"）。
- **MAP-Elites** は behaviour space を bin に割り、bin ごとに 1 体だけ残すことで**再発見を構造的に潰す**。
- **CMA-ME の emitter** は **archive が 1 つも更新されなくなったら再起動する**（＝ 既出にしか落ちなくなったら打ち切る）。
  **ME-MAP-Elites** は emitter 種別の予算をバンディットで配る。

**一致する点**: 「既訪の記憶で再探索を抑える」という**枠組み**。
**一致しない点**: **対象が behaviour space の bin** であって決定空間の盆地ではなく、
**作用が novelty スコア / bin の占有**であって再起動点の棄却ではない。
**＝ 手続きは一致しない。ただし枠組みが教科書事項である以上、「記憶で再探索を抑える」という概念自体は
新規性として主張できない**（方針欄 2026-09-08 の追記 (1) の「部品の既存性」に当たる）。

---

## (iv) ベイズ最適化・能動学習

- **local penalization（González, Dai, Hennig & Lawrence, AISTATS 2016）** —— 獲得関数を
  **既に<u>そのバッチに入っている</u>点の周りで乗法的に減衰させる**（Lipschitz 定数から排除域を作る）。
  **本文取得済み**（`analysis/mmo2024/e159/refs/local_penalization_gonzalez2016.txt.gz`）。
  逐語で確認した点: 排除域は **Lipschitz 定数 L から決まる球 `B_r(x_i)`**（Figure 1 の "Exclusion cones" /
  "The exclusion zones for the maximum of f determined by the balls B_r(x_i) are shown"）で、
  目的は **"the convergence to the maximum"**（単一最適）。
  **一致しない点**: 対象が**バッチ内の多様性**で、**時間方向の既発見<u>最適</u>ではない**（中心は評価済みの点）。
  目的も単一最適。**撤退理由にならない。**
- **arXiv 2210.06635（A Bayesian Optimization Framework for Finding Local Optima in Expensive Multi-Modal
  Functions）** —— **目的は一致**（多解を取りに行く）が、**手続きが違う**（目的関数と 1 階微分の同時分布を
  解析的に出して獲得関数に使う）。排除域でも再起動点の選択でもない。
- **level set estimation** は水準集合の同定であって、既発見解の回避ではない（その118 の splitting 路線と同じ位置）。

---

## 判定

**反証条件 (a)（4 分野のどれかが空振り）は不発** —— 4 分野とも引けた（(i) 6 本、(ii) 2 本、(iii) 4 本、(iv) 3 本）。

**判定は撤退。しかも<u>2 本から独立に</u>落ちる。**
1. **RS-CMSA-ES（Ahrari, Deb & Preuss 2017）が、目的（多解）・手続き（既発見の<u>大域</u>最適を中心とする
   棄却領域）・適応則（同じ最適を複数回見つけたら半径を広げる）の 3 点で一致する。**
2. **MLSL（Rinnooy Kan & Timmer 1987）が、抑制すべき失敗（Error 1）に名前を与え、開始規則を書き下し、
   「開始回数は確率 1 で有限」という定理まで持っている。** `Restart-Lander` はこの開始規則を外したもの。

方針欄 2026-09-08 の追記 (1) の緩い基準（部品の既存性は撤退理由にならない／一致するのは組み合わせと
具体的手続きのときだけ）でも、**これは撤退側に落ちる。**
**さらに悪い（＝ 判定が固い）方向の事実が 2 つ**:
**(a) この suite の公表 1 位（TRDE-LR "Tabu Restart-Based…"）と 2 位（RR-CMA-ES）が<u>どちらも</u>持っている。**
**(b) RS-CMSA を作った Ahrari は、この研究が測っている GECCO'2024/'2025 suite の設計者本人である。**

**腕は作らない。次のサイクルで設計に入ってはいけない。**

## 残っているもの（次の回のために）

1. **測定の側は空いたまま。** 今回引いた 15 本のどれも、**大域最適に届いた降下だけを分母にした重複率**も、
   **それが飽和の帰結でないことを示す集中度**も報告していない。
   2405.01226 の redundancy factor は**全再起動の評価回数に占める冗長分**で、分母が違う量である。
   ＝ **その158 の測定は潰れていない。潰れたのは「その測定から腕を作る」道だけ。**
2. **逐語が取れなかったのは 2 本だけ**: **RS-CMSA-ES / ESII の原論文**（EvCo 25(3) / IEEE TEVC 9555836。
   どちらも CI からも経路が無い）と **TRDE-LR**（arXiv に無い、その92 で確認済み）。
   **撤退の根拠にしたのは第三者の逐語記述**（HVC 論文の RS-CMSA 記述、S-CARD-CMSA の RS-CMSA-ESII 記述）**と
   MLSL の逐語**なので、**判定は逐語で支えられている。** 原論文の逐語は次に論文を書く回が取ること。
   **CI で取得できた 4 本**（MLSL 経由 / S-CARD / MAP-Elites / local penalization）は
   `analysis/mmo2024/e159/refs/` にある。
3. **この撤退は、手元の記録が<u>既に持っていた</u>事実で決まった** ——
   `docs/related_work.md` の表は 2026-08-30 から
   「斥力・タブー ｜ RS-CMSA-ES (Ahrari+ 2017), RS-CMSA-ESII (2021) ｜ **発見済み解を taboo 点として部分集団を反発させる**」
   という 1 行を持っている。**2026-09-09（RR-CMA-ES）・2026-09-12（Cano+）に続いて 3 度目の同型の見落とし。**
