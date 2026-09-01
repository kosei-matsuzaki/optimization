# MC-ESO — コンセプトと最新アーキテクチャ

提案手法 **MC-ESO（Multi-Channel Epidemic Spread Optimizer）** のコンセプト・探索機構・最新アーキテクチャ・パラメータ・既存手法との差別化をまとめる。比較対象の既存手法は [baselines.md](baselines.md)、ベンチマーク・評価基準は [experiments.md](experiments.md)、これまで試した工夫・フラグの履歴は [history.md](history.md) を参照。

実装は `core/optimizers/mceso.py`（`MultiChannelEpidemicOptimizer` + `_MCESOState`）。

---

## 核心の主張

既存メタヒューリスティクスはいずれも **単一の再生メカニズム** を持つ（DE = 差分変異, ES = ガウス変異, PSO = 速度ベクトル, GA = 交叉）。一方、現実の感染症は **複数の伝染経路** — 接触感染・飛沫感染・空気感染 — が並行して働く。MC-ESO はこれを忠実に模し、各世代で 3 つの定性的に異なる伝染チャネルを混合する。

---

## 着想

「**f(x) が低い領域 = 感染宿主が密集する場所**」とみなす。感染症は宿主密度の高い場所ほど広まりやすいため、f 値が低い個体ほど高い確率で次の親（感染源）に選ばれる。現実の感染症が単一経路ではなく複数経路（接触・飛沫・空気）で同時に拡散することを、複数の探索オペレータの並行適用として最適化に転写する。

---

## 探索オペレータ：3 つの伝染チャネル

毎世代、空きスロットを 3 チャネルに配分して子個体を生成する（割合は `air_ratio` / `h2h_ratio` と残り）。この配分は固定でなく、**per-landscape チャネルルーター**（後述）が集団共分散シグナルから landscape を検知して空気感染予算を最適チャネルへ振り分ける。

| チャネル | 疫学アナロジー | 数学的形 | 役割 |
|---|---|---|---|
| **接触感染** (Close-contact) | 親密接触による局所感染 | `x_parent + σ_i · L · N(0, I)`。形状 `L Lᵀ` は **2 系統**: ①集団経験共分散 `C_pop`（瞬時）②成功ステップから学習する持続共分散 `C`（rank-μ、高次元でのみ有効化）| 親近傍の精密探索。σ_i は親の品質と年齢で適応。2 系統の子を同時に生成し宿主競合に選ばせる（後述）|
| **飛沫感染** (Droplet) | 飛沫を介した宿主間感染 | `x_parent + F·(x_strain − x_parent) + F·(x_a − x_b)`（droplet ルート確定時は第 2 差分 `+ F·(x_c − x_d)` を追加＝current-to-best/**2**）＋ 親との二項交叉 (CR=0.9, DE 標準値) | 系統 (niched elite) からの引力 ＋ 集団内差分ベクトルで集団形状を補強しつつ、二項交叉で座標方向の親情報を保護。**悪条件ルート (droplet) では第 2 差分ベクトルでドナー多様性を増し、誤 basin で停滞した run を救出**（F13/F14、高次元でより顕著）|
| **空気感染** (Airborne) | エアロゾルによる広域感染 | `x_random_host + N(0, σ_air I)`（drilling 中は停止）| 集団に依存しない遠方探索。局所最適脱出。`σ < span × 1e-3` の drilling mode では雑音化するため停止 |

---

## 接触感染の 2 系統化（高次元、`cc_learning_rate > 0`）

`C_pop` は毎世代**その `C_pop` から生成した集団**から推定し直すため、推定→生成→推定が閉ループになり高次元で低ランクへ縮退する（実測: 実効ランク dim10 で 1.98/10、dim20 で 5.74/20。**真の条件数 1.0 の F01-Sphere でも実効条件数 4.2e3** で標本抽出＝landscape でなくアルゴリズムが異方性を自作している）。これは EMNA 型（集団から分布を推定する EDA）に共通の既知の弱点である。

**対策**: 接触感染の子を 2 系統に分け、**同じ標準正規乱数**を 2 通りの形状変換にかけて宿主競合に選ばせる。

| 系統 | 形状 | 得意 |
|---|---|---|
| 瞬時系統 | `C_pop`（従来どおり）| 集団が即座に整列できる関数（F02/F05）|
| 持続系統 | 学習 `C`：`C ← (1−c)·C + c·mean(y yᵀ)`, `y = (親に勝った接触感染の子の変位)/σ`。単位行列から開始し、**basin 乗換えのときだけ**リセット（`cc_keep_on_spillover`）| 集団が縮退する悪条件・回転系（F06/F08〜F14）|

**行列を混ぜず、判別器も持たない**のが要点。加法混合は最小固有値を `w/dim` に持ち上げて達成可能な異方性を `~dim/w` に制限し、F02（cond 1e6）を失う。完全置換も F02 を失う。2 系統並走なら両方の異方性が保たれ、どちらが正しいかの判定も不要になる。

**次元ゲート**: `gate = clip((dim/2 − 1)/(cc_dim_ref/2 − 1), 0, 1)` が **dim2 で厳密に 0**（低次元は bit-identical）、`cc_dim_ref`(=10) 以上で 1。ゲートに応じてチャネル配分も close 寄りにテーパー（air 0.30→`cc_air_ratio`, h2h 0.40→`cc_h2h_ratio`）し、学習器のサンプル不足を防ぐ。

> **既存手法との関係（正直な位置づけ）**: 学習 `C` の更新則は CMA-ES の rank-μ そのもの（移植）。spillover での C リセットも IPOP/BIPOP が各リスタートで行うことと同じ。「異なる分布から子を生成し選択に委ねる」構造も DE/EDA (Sun et al. 2005) と同型。本手法に固有なのは**瞬時共分散（EMNA 型）と累積共分散（CMA 型）を同一世代で並走させる**組み合わせと、低次元の調整済み挙動を厳密に保存する次元ゲート。詳細は [history.md](history.md)。

---

## per-landscape チャネルルーター（`channel_schedule=True`, デフォルト）

3 チャネルの割合を landscape に応じて動的に振り分ける機構（`core/optimizers/mceso.py:_channel_ratios`）。**空気感染(air)予算の最適な行き先は関数タイプで割れる**（ill-cond 谷は droplet、separable は close、多峰は air 温存）ため、一律配分でなく **run ごとに 1 ルートを検知して確定**する。

**検知シグナル**（3 つ、いずれも f 非依存・スケール不変・EMA 平滑、接触感染の固有分解から算出）:
- `cond` = log10(λmax/λmin) 集団共分散の固有値比 — 悪条件度
- `algA` = 固有ベクトルの平均 max|成分| — 軸整列（separability）
- `mgap` = 座標方向の最大正規化間隙 — separable の第 2 指標（regular separable vs deceptive を分離）

**ルート確定**（run 内で一度決めて固定 = flip-flop 回避）:
```
・cond EMA > cond_droplet_early (4.0) に達したら即 DROPLET 確定（早期 latch）
・gen route_commit_gen (120) で未確定なら:
    cond > cond_droplet_thresh (3.0)                          → DROPLET
    elif algA > align_close_thresh (0.965) かつ mgap > close_mgap_thresh (0.36) → CLOSE
    else                                                     → KEEP-AIR
・確定前は base keep-air で走る
```

**各ルートの動作**（air 予算の行き先。σ-ramp で連続的に）:

| route | 動作 | 対象 landscape（例）|
|---|---|---|
| **DROPLET** | air を減らし飛沫感染へ | ill-cond 谷/ridge（F11/F12/F13/F14/F08）|
| **CLOSE** | air を減らし接触感染へ | separable/軸整列（F04/F16）|
| **KEEP-AIR** | base 比のまま（air 温存）| 多峰/deceptive（F17/F19/F20/C11）＋ 未検知の全関数 |

**設計判断**: (1) **デフォルト=keep-air=base** なので検知が立たない関数は base 完全一致で無傷（bounded risk）。(2) **決定論的**（報酬バンディットなし — reactive UCB 版 V2a は overall regression で却下）。(3) **run 内 commit で固定**（per-gen 分類は閾値付近で flip-flop し回帰する）。(4) 一律再配分（air を σ や cond で一律に削る 4 案）は全て overall SR@1e-10 を落として却下 — close-contact が深精度の load-bearing チャネルで、一律削減は必ずそれを毀損するため。統合経緯・シグナル診断は [history.md](history.md) 参照。

---

## 集団レベルの 3 機構

| 機構 | 疫学アナロジー | 役割 |
|---|---|---|
| **系統共存** (Strain coexistence) | 空間的に離れた感染拠点の同時存続 | ニッチ半径で離れた最大 6 系統を保護、飛沫チャネルの引力対象 pool |
| **宿主競合** (Host competition) | 新感染が既存宿主に勝てないと排除される | 毎世代 25% kill、子が親より悪ければ rollback → 集団は単調改善 |
| **スピルオーバー** (Spillover, 情報化リスタート) | 既存系統の絶滅後、生存リザーバから未感染域へ飛び火 | 停滞窓（dim2 で 300 評価、次元比例）改善なし AND f_best / \|f_init\| > 1e-8 で発動。**情報化再播種**: 一部 (`ir_archive_frac`) を永続系統アーカイブまわりで再着火、残りは放棄 basin 重心を斥力で避けた Uniform（集団免疫）。連続失敗 2 回で basin switch (best 破棄＋σ_init リセット) に escalate。盲目 Uniform restart（IPOP 流）との差別化点 |

> **なぜ系統共存が多峰問題に効くか**: 単純な top-k 選択では最初に見つかった最適解周辺に個体が集中し、Himmelblau（最適解4箇所）のような多最適解問題で致命的になる。系統選択は (1) f 値の良い順に走査し、(2) 既保護系統との距離が全て `niche_radius_ratio × span` を超える候補のみ追加、(3) `n_elite_max` 個で終了する。これにより空間的に離れた複数の最適解周辺に独立した感染系統が自然形成され、飛沫感染の引力対象となる。

---

## 1 世代の動作フロー

```
1. 系統共存: ニッチエリート抽出（飛沫感染の引力 pool）
   └─ f 値の良い順に走査し、既存系統から niche_radius_ratio × span (=0.1×span) 以上離れた個体だけを
      最大 n_elite_max (=6) 個保護

2. スピルオーバー判定（停滞時の情報化リスタート、basin 回避版）
   └─ no_improve ≥ 停滞窓 = restart_no_improve_threshold (=300) × (dim/2)^restart_window_dim_scale (=1.0) かつ
      f_best / |f_init| > restart_quality_rel_floor (=1e-8) のとき発動（相対 8 桁進捗未満なら spillover、それ以下なら precision とみなし保護）。
      ① 発動時に集団の niched elite を永続アーカイブへ harvest、放棄 basin の重心を記憶（集団免疫メモリ）。
      ② **情報化再播種**（盲目 Uniform でなく探索結果を活用）:
         • 確率 ir_archive_frac (=0.5): 生存アーカイブ系統まわりの σ=ir_reignite_sigma_ratio×span (=0.05) ガウスで再着火
         • 残り: 記憶 basin の ir_repel_radius_ratio×span (=0.1) 内を rejection で避けた Uniform（未感染=未踏域へ）
      ③ 連続失敗回数 (consecutive_failed_spillovers) で動作切替:
         • streak < 2: 上記情報化再播種（best は保持）+ σ ← σ_init×0.3
         • streak ≥ 2 かつ f_best > 1e-2: **ベイスン乗換え**（best も破棄、全 n_pop 再播種、σ を σ_init にリセット）
      ─ ベイスン乗換えで F24 双漏斗 / F04 rugged separable から脱出。f_best ≤ 1e-2 のとき乗換え抑制 → F13 ridge / C01 deep precision を保護

3. 宿主競合: 死亡判定（μ+λ greedy）
   └─ 集団の f 値降順で下位 kill_fraction (=25%) を排除。最良宿主は自動生存

4. 親（感染源）の選択（softmax、スケール不変）
   └─ w_i ∝ exp(−softmax_beta × (f_i − f_min)/(f_max − f_min))、f が低い個体ほど高い感染力
      集団内の f 範囲で正規化するため、問題の f スケール・次元によらず同じ選択圧になる
      （旧式 exp(f_max − f_i) は f の絶対差依存で、収束して f 差が ≪1 になると重みが
        平坦化し実効親数が n_pop と一致＝一様選択に退化していた。[history.md](history.md) 参照）

5. 子個体の 3 チャネル生成（空きスロット数だけ）
   ├─ 接触感染 [残り]
   │   ├─ σ_i = σ × host_sigma_min_scale^(log_quality × (0.7 + 0.3 × age_ratio))
   │   ├─ C_pop = (1/(n-1)) Σ (x_i − x̄)(x_i − x̄)^T  ← 集団経験共分散
   │   ├─ 固有分解 V Λ V^T = C_pop、Λ を平均 1 に正規化（floor=0.01）
   │   └─ child = 親 + σ_i × V √Λ × Gauss(0, I)
   │       → 瞬間共分散による回転・異方性追従。F11/F14 で ill-cond 楕円体に整列
   ├─ 飛沫感染 [h2h_ratio = 0.4]
   │   ├─ trial = 親 + h2h_F × (x_strain − 親) + h2h_F × (x_a − x_b)
   │   │   （**channel_route == "droplet" の run のみ** 第 2 差分 + h2h_F × (x_c − x_d) を加算
   │   │    ＝ current-to-best/**2**。off-route では追加 RNG を引かず base と bit-identical）
   │   └─ child = 各次元で確率 h2h_CR (=0.9, DE 標準値) で trial を採用、残りは親をそのまま継承
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

---

## 適応機構と設計判断

MC-ESO は明示的なフェーズ切替パラメータを持たず、**σ の大きさだけで探索 ↔ 精密化を自動で切り替える**。主要な 3 つの適応機構（いずれも常時 ON）と、ablation を経た採用理由:

- **σ 適応（always-on）** — 改善時 `× sigma_up (1.1)`、非改善時 `× sigma_down (0.95)`（SaVOA 流の乗法適応を毎世代適用）。以前は `no_improve` ゲートで停滞中の減衰を緩める fallback を持っていたが、HP 削減（`sigma_adapt_stagnation_gate`, `sigma_decay`）と引き換えに除去した。
- **Drilling mode** — `σ < span × precision_sigma_ratio (1e-3)` に入ると「σ が basin スケールまで収縮済み」と判定し、非改善時の縮小を `× sigma_drill_down (0.85)` に強化して浮動小数限界まで追い込む。同時に空気感染チャネルを停止（`air_ratio_eff = 0`）し広域ランダム雑音による精度劣化を防ぐ。σ ベース閾値なので shift / scaling されたベンチマークでも調整不要。「正しい basin 到達」を `best_so_far` で判定するため、deceptive landscape での誤発動リスクは小さい。
- **Spillover / basin switch（情報化リスタート）** — `no_improve ≥ 停滞窓`（dim2 で 300、`restart_window_dim_scale` により次元比例）かつ `f_best / |f_init| > 1e-8` で停滞と判定し、best を保持して残りを再播種する。再播種は**盲目 Uniform でなく情報化**: 発動時に集団の niched elite を永続アーカイブへ harvest し放棄 basin の重心を記憶、再播種スロットの一部 (`ir_archive_frac`) を生存アーカイブまわりで再着火、残りは記憶 basin を斥力で避けた Uniform（集団免疫＝未踏域探索）。連続失敗が `basin_switch_after_failed_spillovers (2)` 回に達し、かつ `f_best / |f_init| > basin_switch_quality_rel_floor (1e-2)` なら **basin switch**（best も破棄し σ を σ_init にリセット）に escalate。quality gate により高精度到達済みの run（F13 ridge, C01 deep precision）での暴発を防ぐ。IPOP-CMA-ES の "restart with larger population" の MC-ESO 版だが、**探索結果（リザーバ・basin メモリ）を再利用する点で盲目 restart と差別化**（統合経緯は [history.md](history.md) 参照）。

---

## 多解探索（逐次 niching, base 統合済み）

「ウイルス模倣で複数最適解を探索する」という当初の主張は、peak-ratio 実測で当初**実体を持っていなかった**（MC-ESO は SR@1e-10=100% でも PR@1e-4 が Himmelblau 0.28 / Shubert 0.06）。これを受けて**逐次 niching を base 本体に統合**した（`_basin_exhausted` で「掘り切った」を検知して restart）。デフォルト MC-ESO 自体が多解探索を行うため、本実験では単一手法 `MC-ESO` として評価する。`mceso_niching.py:MCESOEndemic` は後方互換エイリアス（base と同一）。

**設計 = 逐次 niching（並行 crowding ではない）＋ 精度ゲート**。1 集団で複数 basin を同時に深精度化すると SR@1e-10 が崩壊するため、BIPOP-CMA-ES 流の「**1 basin を掘る → 記憶 → 既発見 basin から斥力で離れて restart → 次の basin を掘る**」を採る。各 basin は base MC-ESO の単一σ drilling をそのまま使う。

**SR 死守の鍵 = 2 レジーム化**（`_basin_exhausted` で切替、スケール不変）。掘り切る前は base と完全に同一挙動（掘りかけの best basin を絶対に破壊しない）。掘り切った後に**初めて**多解探索を起動する。「掘り切った」の検知は **σ がフロアに到達**（`σ ≤ exhausted_sigma_tol × span × sigma_floor_ratio`、span 相対で f 値非依存）**かつ** フロア到達後の停滞（`no_improve ≥ exhausted_no_improve_mult × restart_no_improve_threshold`）で、最適値・「optimum=0」を一切仮定しない。`exhausted_no_improve_mult` を巨大値にすれば niching 無効化（純粋単一 basin）も可能。

統合後の効果（多大域関数）: C01 Himmelblau PR@1e-2 0.28→0.62、C02 Six-hump MMOsr 20%→90%、C03 Shubert PR@1e-2 0.06→0.17。SR@1e-10 は全 35 関数で回帰ゼロ（むしろ +0.9pt）。詳細な検証ログは [history.md](history.md)。

---

## hunt の刻み（2 回目以降の basin 探索）

掘り切りを検知したあとの MC-ESO は「別の basin を探しては掘る」を繰り返す。ここのコストが多解性能を直接決める。

初期実装では 2 回目以降の hunt も毎回 σ をフロアまで歩かせ直していたため、**5000 評価で hunt が 4 回しか回らなかった**（間隔きっかり 920 評価、発火時の σ は毎回フロア）。深精度は最初の basin で確保・アーカイブ済みで、SR は履歴の min なので後から失われようがない以上、同じフルコストを払う理由がない。

現在の規則は 2 つ:

- **深さで終了** — hunt の basin best が `hunt_level_tol × |f_init|` に達したら終了。等高の多大域問題では、既知の最良水準への到達が「ここも大域解」の手掛かりになる。**到達しない hunt は従来どおり σ フロアまで掘る**ので、大域解 18 に対し局所解 ~760 を持つ N06-Shubert2D のような landscape でも判別能力を落とさない（固定 σ で打ち切る案はここで有意に悪化して不採用 — [history.md](history.md)）。
- **停滞窓を半分に** — 後続 hunt のみ `hunt_no_improve_mult`。最初の掘り切り判定は 3.0 のまま据え置く（緩めると掘っている最中の単峰関数でリスタートが起き SR を壊す）。

効果（niching dim2, n=20, 5000 評価）: PRmean 0.45 → 0.52、ピーク数検定で 2 関数有意勝ち・有意な負けゼロ。**SR@1e-10 と evals は不変、BBOB-24 dim2 は全関数完全一致**。

---

## 解アーカイブ（報告解集合）

多解の指標は run が**報告した解集合**だけを見る（[experiments.md](experiments.md#多解報告cec2013-ルール-niching-スイート)）。MC-ESO が報告するのは 3 つ:

| 中身 | 何のためにあるか | 容量 |
|---|---|---|
| 生存ホスト（`pop_x`）| 探索の現在地 | `n_pop` |
| 系統リザーバ（`ir_archive_x`）| spillover 時の再着火元 | `n_elite_max` |
| **解アーカイブ（`sol_archive_x`）** | **掘って放棄した basin の best を残す** | `solution_archive_max` |

3 つ目が無いと、逐次 niching は解を見つけては捨てる。実測（25000 評価, 5 seed）で N06-Shubert2D は 18 解中 12.2 個に触れながら報告は 6.0 個、N10-ModRastrigin は 10.4 個に触れて 6.4 個だった。1 run で 25 回 basin を乗り換えても、最終集団と容量 6 のリザーバには 6 個分しか残らないため。`_on_spillover_start`（集団がまだ放棄前の basin を保持している時点）で best を追記することで PRmean 0.61 → 0.75、N06 は平均 6.31 → 13.09 ピーク。**SR@1e-10 と評価回数は不変**（記録のみの変更）。効くのは中〜高予算帯で、5000 評価では spillover 回数が少なく効果はほぼ無い。

---

## パラメータ一覧

| パラメータ | デフォルト | 意味 |
|---|---|---|
| `n_pop` | `max(20, 4·dim)` | 集団個体数。**次元適応**（None 指定時）：dim≤5 で 20、高次元でスケール（dim=10→40）。固定 20 は高次元で過小（dim=10 で niching restart が CEC2022 G06-Hybrid1 を彷徨 best_f 2140→40 once n_pop=40）。BBOB dim2/3 は 20 で無変更。int 明示で上書き可 |
| `sigma` | 0.2 | 初期探索半径（探索範囲に対する比率） |
| `host_sigma_min_scale` | 0.05 | 接触感染チャネルにおける per-host σ_i スケーリング下限（高品質・高齢の宿主は σ_i = σ × 0.05 まで縮小して精密探索）|
| `empirical_cov_floor` | 0.01 | 接触感染チャネルの異方性 floor の**高い側**（rugged/多峰で安全。固有値比を約 14:1 にクランプ）|
| `cov_floor_low` | 1e-3 | 異方性 floor の**低い側**（悪条件の谷で異方性比 ~1000:1 まで許容）。`cov_floor_low = empirical_cov_floor` で適応を無効化し固定 floor |
| `cov_ratio_lo` / `cov_ratio_hi` | 1e3 / 3e4 | **適応 floor の切替閾値**。集団共分散の素の固有値比（平滑化）がこの範囲で `empirical_cov_floor`⇄`cov_floor_low` を log 補間。実測中央値が ill-cond ≈1e5–1e7・rugged ≈3–600 と桁違いに分離するのを利用 |
| `cov_ratio_beta` | 0.1 | 固有値比 EMA の更新率（rugged の瞬間スパイクを除去） |
| `air_ratio` | 0.3 | 空気感染チャネルの割合 |
| `cc_learning_rate` | 0.05 | 持続共分散の rank-μ 学習率。0 で 2 系統化を無効（従来の C_pop のみ）|
| `cc_persist_frac` | 0.5 | 接触感染の子のうち持続系統から生成する割合 |
| `cc_cov_floor` | 1e-11 | 持続 `C` の固有値 floor。C_pop 用の `cov_floor_low`(1e-3) を流用すると異方性が σ で ~100 倍に制限され cond 1e6 の F10/F11/F14 に 2 桁足りない（実測 SR@1e-10 dim10: 25.4 → **35.4**）|
| `cc_dim_ref` | 10 | 次元ゲートが 1 に達する次元。dim2 で厳密に 0 |
| `cc_air_ratio` / `cc_h2h_ratio` | 0.10 / 0.25 | ゲート全開時のチャネル配分。h2h は 0.20 だと dim20 で F02 が 100→15（二項交叉が座標方向構造を保護するため）|
| `air_sigma_amplifier` | 3.5 | 空気感染 σ 倍率の振幅（factor = 1.5 + amp × (1 - diversity)、集団分散時 1.5、収束時 1.5+amp） |
| `h2h_ratio` | 0.4 | 飛沫感染チャネルの割合 |
| `h2h_F` | 0.5 | 飛沫感染の差分ベクトルスケール係数 |
| `h2h_CR` | 0.9 | 飛沫感染後の二項交叉率（DE/bin 標準値、座標方向の親情報を確率 1-CR で継承）|
| `droplet_variant` | `"best2_droplet"` | 飛沫の差分構造。`best2_droplet`（デフォルト）＝ droplet ルート確定 run のみ第 2 差分ベクトルを追加 (current-to-best/2)、他ルートは current-to-best/1 で base と bit-identical。`"cur2best"` で全ルート単一差分（旧挙動、回帰リファレンス）。route-gate なので悪条件 (F13/F14) を救出しつつ多峰 keep-air を無傷にする（グローバル第 2 差分は多峰を毀損）|
| `kill_fraction` | 0.25 | 宿主競合で毎世代排除する割合 |
| `softmax_beta` | 5.0 | 親（感染源）選択の選択圧。`w ∝ exp(−beta × (f_i − f_min)/(f_max − f_min))` と集団の f 範囲で正規化するため、f のスケールにも次元にも依存しない。0.0 で旧式 `exp(f_max − f_i)`（絶対差依存＝収束後は一様選択に退化）に復帰。β=8 は高次元で更に強いが dim2 を −2.29pt 落とすため不採用 |
| `restart_no_improve_threshold` | 300 | スピルオーバー発動の no_improve 閾値（dim2 基準。実効値は下記 `restart_window_dim_scale` でスケール）|
| `restart_window_dim_scale` | 1.0 | 停滞窓の次元スケール指数。実効窓 = `restart_no_improve_threshold × (dim/2)^this`。`no_improve` は**評価回数**カウンタだが 1 世代は `kill_fraction × n_pop` 評価を消費するため（dim2 で 5、dim10 で 10、dim20 で 20）、固定窓は**世代数で見ると高次元ほど縮む**。dim2 では係数が必ず 1.0 になるので低次元は bit-identical、0.0 で旧固定窓に復帰 |
| `restart_sigma_ratio` | 0.3 | スピルオーバー後の σ（σ_init に対する比率） |
| `restart_quality_rel_floor` | 1e-8 | スピルオーバー skip 閾値（best_so_far / \|f_init\| ≤ this で skip）。乗法スケール不変 |
| `basin_switch_after_failed_spillovers` | 2 | この連続失敗回数で best 破棄＋σ_init リセットの完全ベイスン乗換え |
| `basin_switch_quality_rel_floor` | 1e-2 | best_so_far / \|f_init\| ≤ this でベイスン乗換えを抑制（相対 2 桁以上進捗で grinding 中とみなし保護）|
| `cc_keep_on_spillover` | True | 通常の spillover では学習 `C` を保持し、basin 乗換えのときだけリセット。毎回リセットすると F12-BentCigar のように「C が一方向へ伸びきること」が解の条件の関数を永久に解けない |
| `solution_archive_max` | 200 | **解アーカイブ**の容量。spillover のたびに放棄する basin の best を貯め、報告解集合に含める。0 で無効（旧挙動）。記録のみで探索は不変 |
| `hunt_level_tol` | 1e-6 | 最初の掘り切り以降、hunt は basin best が `this × \|f_init\|` に達した時点でも終了（＝すでに banked した深さに並んだ）。0 で旧挙動（毎回 σ フロアまで） |
| `hunt_no_improve_mult` | 0.5 | 後続 hunt の停滞窓の倍率。最初の掘り切り判定は `exhausted_no_improve_mult`(3.0) のまま |
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
| `channel_schedule` | True | per-landscape チャネルルーター有効化（デフォルト ON）。False で旧 flat-ratio 挙動に復帰 |
| `cond_droplet_early` | 4.0 | 早期 droplet latch の cond EMA 閾値（真の ill-cond は急騰、rugged の一時スパイクは未達）|
| `route_commit_gen` | 120 | ルート確定チェックポイント（世代）。それまで base keep-air |
| `cond_droplet_thresh` | 3.0 | commit 時 cond EMA がこれ超で DROPLET |
| `align_close_thresh` | 0.965 | commit 時 algA EMA がこれ超（かつ mgap 条件）で CLOSE |
| `close_mgap_thresh` | 0.36 | CLOSE の追加条件（mgap EMA。regular separable F04≈0.41 と deceptive F17≈0.29 を分離）|

---

## MC-ESO の新規性（既存手法との差別化）

### CMA-ES との比較

| 観点 | CMA-ES | MC-ESO |
|---|---|---|
| 探索形状の学習 | 単一の共分散を適応学習（rank-1＋rank-μ＋CSA）| **2 系統を並走**: 瞬時 `C_pop`（EMNA 型）＋ 学習 `C`（rank-μ、CMA-ES から移植）。どちらが正しいかは選択が決める |
| 多峰対応 | 単一中心からの楕円分布（マルチスタートで多峰に対応） | **系統共存** — ニッチ分離されたエリート pool が多峰を保持 |
| 計算コスト/世代 | O(λ·d²)＋固有分解は ~d 世代に 1 回（償却 O(d²)）| O(n·d²)＋**毎世代**固有分解 O(d³)。実測で線形代数は実行時間の 8〜10%。※以前「行列演算なし O(pop·d)」と記していたが実装と不一致だった |
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
