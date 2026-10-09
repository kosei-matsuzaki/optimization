# 開発履歴 — 試した工夫・フラグ・ablation 記録

2026-10-06 に 1618 行から圧縮した。全文は git タグ `archive/history-full-2026-10-06`（`git show archive/history-full-2026-10-06:docs/history.md`）。

MC-ESO の開発で試した機構・フラグの履歴。何を試し、採用 / 不採用となり、なぜかを残す（同じ検証を繰り返さないため）。手法の最新アーキテクチャは [mceso.md](mceso.md)、評価方法論は [experiments.md](experiments.md) を参照。

> 判定の前提: 改良案は `quick_check.py` で全関数 ablation し、SR / 評価回数 / Wilcoxon の 3 指標で overall 改善を確認できたものだけを `main.py` の `_BASE_OPTIMIZERS` に統合する。判定は quick n=20 / max_evals=5000 / `--all` で統一する（GitHub Actions の n=100 は補助で、評価には参照しない）。
>
> 注: 2026-06 前後の検証ログの多くは quick デフォルトが n=10 だった時期のもので、数値は当時の n=10 結果。再検証は n=20 で取り直す。結果ディレクトリのパスは各節 1 つまでに絞った（残りは全文タグにある）。

---

## 検証フロー

`quick_check.py` の `_OPTIMIZERS` は MC-ESO 本体 ＋ 7 ベースライン（CMA-ES / IPOP / BIPOP / PSO / DE / L-SHADE / SaVOA）の 8 手法の標準比較セット。改良案・診断 variant は検証時だけサブクラス／別 entry を一時追加して測り、統合したら外す。診断・ablation variant は常設しない（実体は `core/optimizers/mceso_ablations.py`）。

```bash
./run.sh quick --all       # 2D BBOB-24 で評価（手法評価の標準。Custom は --custom で追加）
./run.sh quick --funcs F08-Rosenbrock,F09-RosenbrockRot,F10-EllipsoidalRot,F12-BentCigar --max-evals 10000
```

`--funcs` で任意関数に絞った集中検証ができる。

---

## 前身手法の廃止（2026-04）

VirusOptimizerV2 (VSO V2) は全廃止しコードから削除した（2026-04-24）。σ 制御の根本問題（`sigma_up × sigma_down` の積と 50% 成功率のバランス）を解決できず、設計だけが複雑化したため（ユーザー判断）。後継の MC-ESO は σ 適応を「常時 ON の乗法適応＋ drilling mode＋σ フロア（`sigma_floor_ratio`）」に絞り、ゲートやフェーズ切替パラメータを増やさない方針を継承した。

---

## ベースに統合された機構（MC-ESO 本体に常時 ON）

すべて ablation で overall 改善を実証済み（この表は 2026-07 までの分。以後の採用は日付節を参照）。

| 機構 | 位置付けと決め手 |
|---|---|
| 飛沫感染チャネル（h2h, DE/current-to-best/1） | 差分変異が集団形状から異方情報を獲得。F08/F09/F10/F12 の主因 |
| h2h binomial crossover (`h2h_CR=0.9`) | 飛沫の trial を親と座標毎に交叉し separable 多峰の座標情報を保護。一時 F04/F17 用に 0.7 としたが、hold-out 検証で標準値 0.9 が overall で優り復帰 |
| 宿主競合（μ+λ greedy + rollback） | 最良宿主の長期保持で F10/F12 の SR を改善 |
| スピルオーバー＋basin switch | quality-gated restart。連続失敗 2 回で best 破棄＋σ_init リセット。ill-cond の整列失敗と F24 双漏斗を救済 |
| 情報化リスタート (`ir_archive_frac=0.5`, 2026-06) | 盲目 Uniform 再播種をリザーバ再着火＋basin 忌避へ。dim2 +1.7pt、dim3/CEC2022 hold-out で有意 regression なし。IPOP 盲目 restart との差別化点 |
| Drilling mode（`sigma_drill_down=0.85`） | σ < span × 1e-3 で σ 縮小を強化し浮動小数限界まで追込む |
| 接触感染の経験共分散 (`empirical_cov_floor=0.01`) | `C_pop` の固有分解で接触感染ノイズを瞬間異方化。履歴累積なしで basin 切替に即応 |
| 適応異方性 floor (`cov_floor_low=1e-3`, 2026-06) | floor を集団共分散の素の固有値比で自動調整（悪条件 比 1e5–1e7 では下げ、rugged 比 3–600 では高く保つ）。全 35 関数で固定 0.01 比 +2.6pt（85.4→88.0）・回帰ゼロ、固定 1e-3（F17/C11/C05 で回帰）も上回る |
| 次元適応 n_pop (`n_pop=max(20, 4·dim)`, 2026-06) | 固定 20 は高次元で過小。dim=10 で CEC2022 G06-Hybrid1 best_f 2140→40。BBOB dim2/3 は 20 で無変更。**2026-10-08 に既定は集団のべき乗縮小（`pop_schedule="linear"`、`max(20, 16·dim)` → `max(10, 4·dim)`）へ置き換わり、この式は `pop_schedule="fixed"` のときだけ使われる**（[mceso.md](mceso.md#パラメータ一覧)） |
| Drilling 中の空気感染停止 | `σ < span × precision_sigma_ratio` で `air_ratio_eff = 0`。drilling 中の広域雑音による精度劣化を防ぐ |
| 逐次 niching (`_basin_exhausted`, 2026-06) | σ-exhaustion 検知（f_opt 非依存）で掘り切った basin から restart。SR 無犠牲で多解 PR を改善 |
| per-landscape チャネルルーター (`channel_schedule=True`, 2026-07) | 3 シグナル（`cond` / `algA` / `mgap`）で air 予算を droplet/close/keep-air の 1 ルートへ（gen120 commit ＋早期 droplet latch）。既定 keep-air=base。BBOB dim2 +0.6pt（87.9→88.4）・dim3 +0.6pt・CEC2022 dim10 G06 364→202 |
| route-gated best2 (`droplet_variant="best2_droplet"`, 2026-07) | droplet ルート確定 run のみ第 2 差分 `+F·(x_c−x_d)` を追加（off-route は bit-identical）。dim2 +1.6pt（88.4→90.0）・dim3 +12.9pt（44.2→57.1）・CEC2022 dim10 改善3/悪化2（G01 64×） |

---

## 検証され不採用となった variant

quick で統合候補として走らせたが overall 改善を実証できず、コードからも削除したもの。

| variant | 追加した挙動 | 不採用の理由 |
|---|---|---|
| MC-ESO-A1（per-dim σ close-contact） | 接触感染ノイズを集団 per-dim std で軸別スケール | F08/F17 は改善するが F14-DiffPowers（回転）で致命的劣化。経験共分散版 A2 に置換 |
| MC-ESO-ABD | h2h_CR=0.9 ＋ σ-adapt 停滞ゲートを drilling 中バイパス ＋ 初回 spillover で座標軸 sweep | F18/F19 で勝つが F04 で回帰。B/D 単独の有意寄与なし |
| MC-ESO-A_mild_BD | 統合済み MC-ESO ＋ 同上の B/D | 改善（F09/F11/F18）と悪化（F04/F14）が相殺。B/D の寄与なし |
| 旧 A〜N（初期開発の 8 案）: A=`use_evolution_path` / C=`use_pop_covariance` / D=`use_lifespan_reset` / E=`use_adaptive_air` / G=`use_adaptive_h2h_F` / I=`use_aggressive_niche` / K=`use_h2h_archive` / N=`use_local_pair_h2h` | 進化パス・集団共分散・寿命リセット・適応 air・適応 h2h_F・強ニッチ・h2h アーカイブ・局所ペア h2h | 12 関数 SR 合計で baseline 以下、または構造欠陥（E）で全削除。C 系は「経験共分散の接触感染」、I 系は「逐次 niching」として別実装で結実 |
| MC-ESO-V2a（UCB-AOS on 3 channels, 2026-06） | 3 チャネル比率を世代毎に UCB で自動調整（credit = 中央値正規化 Δf） | 35 関数 SR@1e-10 24.40 → 22.10（−2.30）、有意勝ち 1（F17）/負け 3（F06/F20/C07）。F23 0%→70% でも覆せず |
| MC-ESO-V2b（V2a + 4 新チャネル, 2026-06） | Lévy 超拡散 / 重心組換 / 系統間クロスオーバ / 反対称跳躍（`2·centroid − x_p`）、arms 3→7 | 24.40 → 19.80（−4.60）、有意勝ち 0/負け 5（F02/F11/F18/C06/C11）。大ジャンプが ill-cond の precision grinding を妨害 |
| MC-ESO-SC / IRSC（系統共存の実活性化, 2026-06） | 永続アーカイブ（品質ゲート `sc_quality_band`）から飛沫 donor を抽選 | SC は dim2 −0.3pt。IRSC は dim2 +2.3pt だが F17/F24 有意回帰、dim3 で有意回帰 3、CEC2022 dim10 medf 2624 vs 1650 と汎化失敗。系統共存を活性化しても性能に結びつかない（novelty gap） |

検証ログ（V2 系）: `results/20260605_190353_v2_compare_all_quick/`。

---

## 撤回した実装ステップ・設計上の失敗（記録）

採用済み機構の開発途中で試して撤回した中間実装。

| 機構 | 撤回したアプローチ | 失敗の理由 / 正解 |
|---|---|---|
| 適応異方性 floor | ill-cond vs rugged を支配的固有ベクトルの世代間 alignment で判別 | 収束後は rugged でも alignment≈0.99 で判別不能。正解は固有値比 λmax/λmin の大きさ（ill-cond ≈1e5–1e7 / rugged ≈3–600） |
| 逐次 niching ① | 掘り切り判定なしの always basin-switch | 掘りかけの best basin を壊し C10(−40)/F14(−20)。掘り切った後だけ乗換える 2 レジーム化が必須 |
| 逐次 niching ② | 掘り切り検知に絶対 floor `f ≤ 1e-11` | 「最適値 0」の BBOB 正規化依存でユーザ却下。σ のフロア到達（span 相対）に置換 |
| 逐次 niching ③ | 停滞許容 `exhausted_no_improve_mult=1` | F14 平坦 basin の遅延 breakthrough を取りこぼし(−20)。`mult=3` に |
| スピルオーバー（旧仕様） | 盲目 Uniform で escalate（streak 0→75% / 1→100% Uniform / 2→basin switch） | 探索構造を毎回捨てていた。情報化リスタート（`ir_*`）に置換 |
| σ 適応（旧仕様） | `no_improve ≥ 100` で `× 0.99` の停滞ゲート＋ `sigma_decay` | HP（`sigma_adapt_stagnation_gate` / `sigma_decay`）の割に寄与が立たず削除。always-on 乗法適応＋ drilling に統一 |

---

## 簡潔化監査 — 各機構の単独 ablation（2026-06）

一機構ずつ OFF にして全 35 関数で寄与を再確認（base 87.7%, n=10, dim2, 5000 evals）。

削除（寄与ゼロ）:
- 軸 sweep（`MCESONoAxisSweep`, `_axis_sweep`→`[]`）: 35 関数・全精度で SR が 1 run も動かず、多解 PR も不変。F05/F04 すら影響ゼロ（境界 snap で既に到達）。`_axis_sweep` と `_maybe_spillover` 内 sweep ループを削除。
- 適応 floor の median 平滑化枝（`cov_ratio_window`, `MC-ESO-med15`）: 既定 OFF の足場。`cov_ratio_window` / `cc_logratio_win` / median 分岐を削除。

維持（OFF で SR@1e-10 低下）:

| 機構 | OFF 版 | SR@1e-10 | Wilcoxon 有意悪化 | 判定 |
|---|---|---|---|---|
| per-host σ スケール (`host_sigma_min_scale`) | `=1.0` | 84.3%（−3.4） | 3（F06/F17/F23） | 維持（最大寄与） |
| 空気感染チャネル (`air_ratio`) | `=0` | 85.1%（−2.6） | 4（F04/F17/C05/C11） | 維持 |
| 境界 snap（`_reflect`） | `MCESONoBoundarySnap` | 85.1%（−2.6） | 1（F05） | 維持（F05 100→20） |
| streak basin-switch (`basin_switch_after_failed_spillovers`) | `=∞` | 86.0%（−1.7） | 0 | 維持（弱）。寄与は F04（90→50）/F20 に集中、除去で速い |
| 収束適応の空気 σ (`air_sigma_amplifier`) | `=0` | 86.0%（−1.7） | 1（F17） | 維持（弱）。F17(−50)/C11/F24 を救い F23(+30)/F13 を損なう |

弱い維持 2 機構は n=20 で F04/F17 の寄与が崩れれば削除候補。検証ログ: `results/20260630_140730_simplify_DEFG_quick/`。

---

## 集団レベル3機構の必要性 ablation（n=20, 2026-07）

[mceso.md](mceso.md#集団レベルの-3-機構) の 3 機構＋ drilling を 1 つずつ OFF（2D BBOB-24, base SR@1e-10=92.9%）。

| 機構 OFF | 変種 | SR@1e-10 | Δpt | 有意悪化 | 寄与が集中する関数 |
|---|---|---|---|---|---|
| 宿主競合（`MCESONoHostCompetition`） | `abl_noHostComp` | 71.0% | −21.9 | 8 | ill-cond 谷/ridge（F13/F14 −75, F10 −60）。μ+λ 単調改善保証が生命線 |
| スピルオーバー（`MCESONoSpillover`） | `abl_noSpill` | 80.2% | −12.7 | 6 | separable/多峰（F04 −80, F15 −65） |
| 系統共存（`n_elite_max=1`） | `abl_noStrain` | 90.6% | −2.3 | 2（F06/F17） | F17 −25/F04 −19、ただし F18/F23 で +9 |
| drilling（`sigma_drill_down=0.95`） | `abl_noDrill` | 91.7% | −1.3 | 1（F17） | F17 −35、F18 +9/F24 +5 で相殺 |

寄与の序列は 宿主競合 ≫ スピルオーバー ≫ 系統共存 ≈ drilling。どの ablation も MC-ESO を有意に上回った関数はゼロ。系統共存/drilling は寄与が小さいが SR@1e-10 を下げるので削除不可。検証ログ: `results/20260708_001002_eval_pop_ablation_quick/`。

---

## 診断 ablation（チャネル vs リスタートの寄与分解, 2026-06）

「性能はランダムリスタート由来では」の検証用に 2 つの診断 variant（改善候補ではない）:
- `MC-ESO-NoSpill`: チャネル ON / spillover 停止。
- `MC-ESO-RandRestart`: 3 チャネル＋系統共存を等方ガウス局所探索（`x_parent + σ_global·N(0,I)`）1 本に置換。

結果（BBOB24+Custom11, n=10, 5000 evals, dim2）: MC-ESO 83.7% / NoSpill 68.6% / RandRestart 48.9%。
- 主動力はチャネル機構: RandRestart に対し 26/35 関数で有意優位（負け 0）。「リスタートのくじ運」説は棄却。
- spillover は二次的: NoSpill に有意優位なのは 7/35（F03/F04/F15/F20/F24/C05/C11 ＝ 多峰・deceptive）。
- 系統共存は不活性: 平均 n_elite は大半の関数で ~1.0–1.2、多 basin 保持は F20(1.56)/F24(3.85) のみ ＝ novelty gap。

検証ログ: `results/20260605_200551_diag_restart_ablation_quick/`。

---

## 情報化リスタートの統合 / 系統共存活性化の不採用（2026-06）

拡張フック `_on_spillover_start` / `_diversified_reseed` / `_droplet_strain_positions`（既定で RNG 順不変）経由で検証。

- ① 情報化リスタート（IR）→ 採用。リザーバ再着火（`ir_archive_frac` / `ir_reignite_sigma_ratio`）＋集団免疫忌避（`ir_repel_radius_ratio` / `ir_repel_max_tries`）。dim2 83.7→85.4（+1.7pt; C09+40/F23+20、悪化 F10/F19 各−10）、dim3 有意差 0、CEC2022 dim10 で有意 best_f 改善 5・回帰 0。`MultiChannelEpidemicOptimizer` 既定に統合。`MC-ESO-RandRestart` は旧盲目 restart の比較基準として維持。
- ② 系統共存の実活性化（SC / IRSC）→ 不採用（上の variant 表）。`mceso_sc.py` / `mceso_combo.py` は削除。

検証ログ: `results/20260610_103749_ir_verify_quick/`。

---

## peak-ratio 診断 — 「多解並行探索」は当初実体がなかった（2026-06-16）

最適化器を変えずに peak-ratio 指標を後付け（`core/runner.py:optima_found_mask` / `peak_metrics`）。quick n=10 で MC-ESO は SR@1e-10=100% だが PR@1e-4 は C01 Himmelblau 0.28・C02 0.60・C03 Shubert 0.06。Himmelblau は BIPOP 0.78 vs MC-ESO 0.28。spillover は多解に無効。「並行的な多解探索」はコンセプトのみだったと定量化し、次節の動機になった。多解路線の経緯は [archive/multisolution.md](archive/multisolution.md)。

---

## 逐次 niching（多解探索）の統合（2026-06）

base に統合（`_basin_exhausted` で掘り切りを検知して restart。`mceso_niching.py:MCESOEndemic` は後方互換エイリアス）。1 集団で複数 basin を同時に深精度化すると SR@1e-10 が崩壊するため crowding/per-host σ を撤回し、BIPOP 流の「1 basin を掘る→記憶→斥力で離れて restart」を 2 レジーム化（掘り切るまでは base と同一）で採用。詳細は [mceso.md の多解探索](mceso.md#多解探索逐次-niching-base-統合済み)。

結果（n=10, dim2, 全 35 関数）: SR@1e-10 は回帰ゼロで 85.4% → 86.3%。PR@1e-4: C01 0.28 → 0.53、C02 0.60 → 0.95（MMOsr 20% → 90%）、C03 0.06 → 0.16。撤回した試作: crowding=`20260616_145202_endemic_v3`、always-restart=`20260623_145326_1ec0bf0`、絶対 floor 版=`20260624_103753_endemic_secured`。検証ログ: `results/20260624_113046_endemic_sigexh_quick/`。

---

## 次元・hold-out への汎化（2026-06）

新 base（floor+niching+dim-pop）vs 旧 base（`MC-ESO-orig`）: BBOB dim=2 85.4 → 88.0%、dim=3 40.8 → 45.4%（F19 −10 のみ）、CEC2022 dim=10 hold-out median best_f 24.3 → 17.9。改善は全次元・hold-out に汎化。

次元適応 n_pop: dim=10 の sweep で 40 が sweet spot（80 は悪化）、G06-Hybrid1 best_f 2140→40。`n_pop=None` 既定で `max(20, 4·dim)`（dim≤5 は 20）。検証ログ: `results/20260*_cec_holdout_check_quick/`。

---

## チャネル割合スケジューリングの探索（2026-07）— 一律再配分は全 REJECT、目標は per-landscape ルーティング

`MultiChannelEpidemicOptimizer(channel_schedule=...)`（一時エントリ `MC-ESO-Sched`）で割合を一律に変える 4 案はすべて不採用（n=20, 全 35 関数, dim2, base 87.9%）:

| 案 | 変更 | SR@1e-10 | 判定 |
|---|---|---|---|
| 1 | air を σ の log で `air_ratio`→0 にランプ、空き枠→close | 87.0（−0.9） | REJECT |
| 2 | 同上、空き枠→droplet | 87.7（−0.1） | REJECT |
| Phase1 | `air_ratio·(1−c)`（c=cov 固有値比 EMA）、空き枠→close | 86.6（−1.3） | REJECT |
| Phase2 | air 不変、`h2h_ratio·(1+0.5·c)` で droplet↔close のみ再配分 | 86.7（−1.1） | REJECT |

理由: close は SR@1e-10 の load-bearing チャネルで、どこで枠を削っても深精度 run が落ちる。conditioning `c` は多くの関数の収束時に広く立ち上がるので on/off トリガーには粗すぎる。base 比（air 0.3 / droplet 0.4 / close 0.3）は既に局所最適で、V2a 却下と同根。

ただし関数ごとの最適配分は存在する: air 維持（F10, F15, C05, F06, F17, F19, F20, F24）/ droplet 中心（F11, F12, F13（+20〜35）, F14）/ close 中心（F04, F16, C09）。→ 目標は f 非依存の構造シグナルによる per-landscape ルーティング（報酬ベース＝V2a バンディットは再発禁止）。足場は `mceso.py:_channel_ratios` ＋ `channel_schedule`。検証ログ: `results/20260701_105621_eval_sched_quick/`。

### シグナル診断（全35関数, 2026-07）— `cond` ＋ `algA` で最適ルートが分離

base を変えずに集団共分散の構造シグナル（`cond` / `PR` / `algA` / `offd` / `divs` / `kurt` / `mgap` / `nelt` / `spil`）を測定（`scripts/measure_channel_signals.py`, n_runs=8, dim2）。
- `cond`（log10 λmax/λmin）> 2.5 → droplet: F11–14（3.3–5.6）を切り出す（間に [1.63, 3.34] のギャップ）。F08/F09（2.5–2.6）も droplet 得意領域なので有益に巻き込む。F02/F10（5.62/5.53）も droplet 側に落ちる。
- cond ≤ 2.5 かつ `algA`（ドミナント固有ベクトルの max|成分|）> 0.98 → close: F04(0.988)/F16(0.994)。
- それ以外 → keep-air（base と完全一致）。判断のつかない関数は自動的に無傷。
- 他シグナルは分離器として劣る（`offd` は F16 と F17/F24 が重なる、`nelX` は飽和、`nelt` / `spil` / `kurt` / `mgap` は多峰でノイジー）。
- 残リスク: F22-Gallagher21 が cond2.74 で droplet 落ち、close バケツが F03/F05/C02 を巻き込む、閾値は dim2 tuned。

測定値の全表（9 シグナル × 35 関数）は全文タグにある。

### ルーターの実装反復と本体統合（2026-07, 採用）

| 反復 | ルート確定方式 | overall SR@1e-10 | 問題 |
|---|---|---|---|
| per-gen | 毎世代分類 | ±0.0（87.9→87.9） | 閾値付近 F04/F06/F14 が flip-flop 回帰 |
| commit@120 | gen120 で 1 回確定 | −0.6 | F13 の早期 droplet を取り逃す＋F23−20 |
| hybrid | cond>4.0 早期 latch ＋ gen120 commit | −0.1 | algA だけでは F04 と F17 が分離不能、F17/F20 が close 誤爆 −15/−5 |
| +mgap（採用） | hybrid ＋ close 条件に `mgap>0.36` を AND | +0.6（87.9→88.4） | 改善 7（F04/F11/F12/F13/F14/F16 +5, F24 +5）/回帰 2（F20−5, F23−10） |

F04 と F17 は algA≈0.975 で同じだが、座標間隙 mgap が F04≈0.41 / F17≈0.29 で分離（`close = algA>0.965 かつ mgap>0.36`）。汎化: dim3 +0.6（43.5→44.2, 有意回帰なし）、CEC2022 dim10 median best_f 改善 2・悪化 0（G06 364→202）。`channel_schedule=True` を既定に統合（`MC-ESO-Orig`=`channel_schedule=False` で旧挙動を pin）。確定値: `cond_droplet_early=4.0` / `route_commit_gen=120` / `cond_droplet_thresh=3.0` / `align_close_thresh=0.965` ＋ `close_mgap_thresh=0.36`。残課題: F13/F14/F18/F23 は少数 run が誤 basin で stuck（降下でなく脱出の問題）。検証ログ: `results/20260701_225007_eval_router4_quick/`。

### stuck-gated 媒介感染チャネル（2026-07, 現状 REJECT）

4 本目のチャネル `migratory_channel`: drilling 中かつ停滞時のみ、dead-slot の一部を best からの構造化大ジャンプ（半分は主固有ベクトル方向、半分は等方、span×0.2）に充てる。
- 初回（no_improve≥100, ratio0.5）: F23 60→85 / F18+10 / F13+5 を救うが F14−15/F17−10/F19−5 で相殺、overall +0.1・Wilcoxon 0/0 → REJECT。追加チャネルは dead-slot を奪い RNG をずらすので、一度発火した run 全体が非 bit-identical になり回復途中の run を壊す＝純加算でない（`project_migratory_channel`）。
- 厳格版（`migratory_no_improve_thresh=200` / `migratory_ratio=0.34`）: ±0（88.4→88.4）、F17−20/F14−15 が残る → REJECT 確定。救う stuck（F23）と壊す stuck（F17）は検知で区別できない。フラグは既定 OFF で保持。

検証ログ: `results/20260703_184939_eval_mig_strict_quick/`。

### チャネル中身差し替えの系統スイープ（2026-07）— 追加でなく「中身」を全チャネル網羅、route-gated best2 のみ採用

本数は 3 本のまま各チャネルの数学を差し替える 8 変種（既定は bit-identical）をスクリーニング（n=20, dim2, base 90.5%, C07-C11 欠落の 30 関数）:

| チャネル | 変種 | overall SR@1e-10 | 判定 |
|---|---|---|---|
| 空気 | airCauchy / airLevy（β=1.5）/ airOpp（2·centroid−host）/ airUnif | 89.5 / 88.5 / 89.2 / 89.7 | 全 REJECT |
| 接触 | clCauchy | 89.5 | REJECT |
| 飛沫 | drRand1 / drPbest（top15%） | 82.3 / 88.3 | REJECT |
| 飛沫 | drBest2（global） | 90.8（+0.3, 相殺） | REJECT |

- 本命だった「空気の重い裾」は不成立（escape 関数は drilling 停止＋ keep-air で既に保護済み）。
- drBest2（global）は F13 55→100 / F14 75→100 を救うが多峰 keep-air（F17/F18/F24/C05/F23）を壊す。壊す関数は全て keep-air ルート、救う関数は droplet ルート。
- `best2_droplet`（採用）: 第 2 差分を droplet 確定 run のみ適用。dim2 +1.6pt（88.4→90.0; 小回帰 F18−10/F19−5）、dim3 +12.9pt（44.2→57.1; 有意勝ち6・負け1、evals 2169→1890）、CEC2022 dim10 改善3（G01 0.0167→0.00026=64×, G06 202→166, G11）/悪化2（G03/G07）。in-place なので dead-slot RNG ずらし問題を構造的に回避。
- `best2_stuck`（REJECT）: drilling 中に限定すると F13 +45→+20 / F14 +25→+10 と減衰し overall +0.6 に低下。
- 不採用変種（air 全種 / close-cauchy / rand1 / cur2pbest / global-best2 / best2_stuck）はコード削除。

検証ログ: `results/20260703_194850_eval_channel_sweep_quick/`。

### 進捗報告データの取得（2026-07-07）— 既存手法比較（10 手法）＋変更点 ablation 累積梯子

- 既存手法比較（`results/20260707_進捗報告データ_既存手法比較_10手法/`, n=20, dim2, 全 35 関数）: MC-ESO が SR@1e-10 89.9% で 10 手法中 1 位（2 位 DE 85.7%）。NM-Restart（60.6%）には多峰 16 関数で圧勝、負けは F23 のみ（65 vs 60）。NCDE（38.7%）は C01 の多解被覆で優る（PR@1e-4 0.85 vs 0.53）。
- 累積梯子（`results/20260707_進捗報告データ_変更点ablation/`）: 85.1 → +ir 85.4 → +floor/nich 87.9 → +router 88.4 → +best2 89.9。再構成フラグ:

| 変種 | kwargs（`MultiChannelEpidemicOptimizer`） |
|---|---|
| `abl0_base2018`（5/18 版） | `droplet_variant="cur2best", channel_schedule=False, cov_floor_low=0.01, exhausted_no_improve_mult=1e9, ir_archive_frac=0.0, ir_repel_max_tries=0` |
| `abl1_ir` | 同上から `ir_archive_frac` / `ir_repel_max_tries` を既定に戻す |
| `abl2_floornich` | `droplet_variant="cur2best", channel_schedule=False` |
| `abl3_router` | `droplet_variant="cur2best"` |

---

## 次元スケーリングの計測と高次元崩壊（2026-07〜08）

### 現状把握 — dim3 で 1 位を失い、dim5 以上で崩壊（2026-07-24）

BBOB を dim 2/3/5/10/20 でレジストリ化（`core/benchmarks.py:_build(d)`、`quick_check.py --dim`）、予算 2500×d、n=20。SR@1e-10:

| 手法 | d2 | d3 | d5 | d10 | d20 |
|---|---|---|---|---|---|
| MC-ESO | 93.3 | 67.3 | 17.3 | 6.5 | 0.0 |
| IPOP-CMA-ES | 83.8 | 67.3 | 56.0 | 47.1 | 47.3 |
| BIPOP-CMA-ES | 79.2 | 67.7 | 54.8 | 49.4 | 47.3 |
| DE | 89.4 | 71.0 | 44.0 | 15.4 | 7.7 |
| CMA-ES | 64.6 | 53.5 | 44.4 | 42.7 | 40.8 |

失敗は 2 系統: 精度グラインドの失速（d10 F01 SR@1e-2 100% → 1e-10 30%）と悪条件の完全崩壊（d10 F08/F10/F11/F12/F14 は SR@1e-1 すら 0%）。検証ログ: `results/20260724_014240_perf_d2_quick/`。

### 外れた仮説 3 つ（2026-07-25〜26）

| 仮説 | 変種 | 結果 |
|---|---|---|
| router が高次元で誤ルートし best2 が発火しない | `diag_dropAll`（`cond_droplet_early=-1e9`） | d5 3.1→16.2 だが d10 は 0.0 のまま |
| 世代数不足 | `diag_npopSmall`（`n_pop=12`） | d5 0.6 / d10 0.0（悪化） |
| C_pop の単世代推定が高次元で荒い → EMA 累積 | `cc_accum_rate` 0.05 / 0.15（`diag_accum05` / `diag_accum15`） | SR@1e-10 不変、SR@1e-2 57.5→38.1 と悪化。`cc_accum_rate` / `cc_accum_min_dim` / `_MCESOState.cc_cov_accum` をコード削除（再現は `_close_contact_children` の `cov` を EMA に差し替える） |

一時エントリは削除済み（kwargs は上表で再登録可）。

### 真因 — 停滞窓の単位バグと spillover リミットサイクル（2026-08-23, 採用）

σ トレース（`history_sigma_global` / `history_no_improve`）で特定: `no_improve` は評価回数カウンタだが 1 世代は `kill_fraction × n_pop` 評価を消費するので、固定 300 評価の窓は世代数で見ると次元とともに縮む（dim2 = 60 世代 → dim10 = 30 世代）。spillover が絶えず発火して σ を `σ_init × restart_sigma_ratio` に戻し、`0.2×0.3×0.95^30 = 1.29e-2·span` で下げ止まる（実測と一致、drilling 到達 0%、spillover 発火 F01 46 回 / F10 65 回）。

修正: 停滞窓を `restart_no_improve_threshold × (dim/2)^restart_window_dim_scale`（既定 1.0）。dim2 は bit-identical。`_stagnation_window()` を `_spillover_should_fire` / `_basin_exhausted` が参照する。旧挙動 `hd_win0`（`restart_window_dim_scale=0.0`）との比較:

| 次元 | SR@1e-10 旧→新 | Wilcoxon (ref=新) |
|---|---|---|
| d2 | 93.33 → 93.33（全指標 bit-identical） | — |
| d3 | 67.29 → 67.08（−0.21） | 勝ち 4 / 負け 1（F03） |
| d5 | 17.29 → 17.50（+0.21） | 勝ち 6 / 負け 4 |
| d10 | 6.46 → 10.00（+3.54） | 勝ち 14 / 負け 2（F07/F11） |
| d20 | 0.00 → 5.42（+5.42） | 勝ち 21 / 負け 2（F07/F11） |

効果は次元とともに拡大。d3/d5 の wash は「窓を伸ばすと悪条件が伸び、separable 多峰の escape が減る」トレードオフ。保守版（世代数一定、`hdwin600_d10`）は d10 で 8.96 で dim スケール版（10.00）に劣り、指数 1.0 を採用。教訓: 悪条件寄り 10 関数での中間測定は d5「+5pt」、全 24 関数では +0.2pt ＝ 判定は必ず `--all`。検証ログ: `results/20260823_231616_hdwin_d2_quick/`。

### 残る課題 — 高次元の悪条件（未解決）

窓を直しても d10 の F08/F10/F11/F12 は SR@1e-10 0%。`cov_floor_low` を 1e-5/1e-7/1e-9 に下げても改善はノイズ級で、異方性 floor は主因ではない。CMA-ES の rank-μ は子の移動ステップから、MC-ESO の C_pop はエリートの位置分布から学ぶ ＝ 情報源が違う。

---

## 全パラメータの次元不変性 監査（2026-08-24）

44 個の全パラメータについて次元依存をコード読解＋派生量の実測（`_softmax_weights` / `_adaptive_cov_floor` / `_record_generation` にフック、dim 2/5/10/20）で洗い出した。主なドリフト: algA 0.924→0.542、cond 2.48→15.37（d20）、ニッチ飽和 8.5%→23.5%、d≥5 で close ルート 0。

- A. 次元不変（問題なし）: `air_ratio` / `h2h_ratio` / `kill_fraction` / `ir_archive_frac`、`restart_quality_rel_floor` / `basin_switch_quality_rel_floor` / `log_slope_threshold`、`h2h_F` / `h2h_CR`（交叉座標数 `CR·d`）、`n_elite_max` / `exhausted_no_improve_mult` / `basin_switch_after_failed_spillovers` / `ir_repel_max_tries`、`sigma` / `sigma_ceil_ratio` / `restart_sigma_ratio` / `ir_reignite_sigma_ratio`、`air_sigma_amplifier`（`diversity_ratio` 正規化済み）、`cov_ratio_beta`。
- B. 次元依存（`restart_no_improve_threshold` は修正済み）: `align_close_thresh`（ランダム基底の max|成分| が √(2 ln d/d) で減衰し d≥5 で到達不能）、`cond_droplet_thresh/early`（d20 では C_pop のランク崩壊を測っている）、`cov_ratio_lo/hi`（d20 で常時飽和）、`niche_radius_ratio` / `ir_repel_radius_ratio`（点間距離 ∝ √d）、`sigma_up` / `sigma_down` / `sigma_drill_down`（収束世代数 ∝ d）、`precision_sigma_ratio` / `sigma_floor_ratio`（実変位は σ√d）。
- C. 機能していない: softmax 親選択。`exp(f_max − f_i)` が f の絶対差依存なので収束後は重みが平坦化し、dim2 中央世代で実効親数 20.00/20（完全一様）。

### スクリーニング（d10, n=10, 全 24 関数, base SR@1e-10 10.0）

| 変種 | SR@1e-10 | 判定 |
|---|---|---|
| `niche_radius_dim_scale=1`（半径 ∝√d） | 8.3（−1.7） | REJECT |
| `align_signal_dim_norm`（IPR 正規化） | 10.0（±0） | 変化ゼロ（下記） |
| `cond_rank_guard=1e-8` | 10.0（±0） | d10 で無効果 |
| `sigma_adapt_dim_scale=1`（倍率 ∝1/d） | 10.0（±0） | 強すぎ（F01 70→0）。0.5 で再試験 |
| `sigma_threshold_dim_scale=1`（σ 閾値 /√d） | 9.6（−0.4） | REJECT |
| `softmax_beta=5`（スケール不変選択） | 13.3（+3.3） | 有望 |

ルーターは閾値較正でなく推定器の問題（`route_probe.py`）: 次元正規化した軸整列度は dim10 で全 24 関数が 0.00〜0.33（F03 0.108 / F04 0.039）、F12 の cond は dim2 5.64 → dim10 2.23 で keep-air に落ちる。40 個体・10 次元の標本共分散からは軸整列度も conditioning も推定できず、3 シグナルとも C_pop 由来なのでルーター全体が情報を失う。

### softmax のスケール不変化 — 採用（`softmax_beta = 5.0`）

dim2 ゲート（n=20, `--all`）: base 93.33 / `softmax_beta=3` 92.08（−1.25, 却下）/ `softmax_beta=5` 93.54（+0.21; F18 65→100 有意勝ち、F24 35→10、有意な負けゼロ）/ β=8 91.04（−2.29, 有意な負け F06/F17, 却下）。β=5 が上限。

全次元（SR@1e-10 の base 比 pt、括弧は Wilcoxon 変種勝ち/base 勝ち）:

| 変種 | dim2 | dim3 | dim5 | dim10 | dim20 |
|---|---|---|---|---|---|
| `softmax_beta=5`（採用） | +0.21（W1/L0） | +6.67（W2/L1） | +8.12（W6/L0） | +3.12（W7/L3） | +6.87（W9/L2） |
| `sigma_adapt_dim_scale=0.5` | ±0（構造的 no-op） | −2.29 | +3.12 | +3.54（W5/L0） | — |

dim5 17.50 → 25.62、dim20 5.42 → 12.29。スケール不変性の欠如の修正なので全次元に効く。`softmax_beta=5.0` を `_softmax_weights` の既定に統合（`softmax_beta=0.0` で旧式、回帰ピン `dimf_softmax0`）。`sigma_adapt_dim_scale` は dim3 −2.29（F19 −35 / F23 −25）で汎化ゲート不通過。検証ログ: `results/20260824_125218_dimf_softmax_d2_quick/`。

コード削除した不採用フラグ: `niche_radius_dim_scale` / `sigma_threshold_dim_scale` / `align_signal_dim_norm` ＋ `align_close_thresh_norm` / `cond_rank_guard` / `sigma_adapt_dim_scale`。

構造的な限界: (1) ルーターは高次元で機能しない（C_pop が推定ノイズに支配）。(2) d10 の F10/F11/F12 は SR@1e-10 0% のまま。(3) 両者は同根で、次の一手は成功ステップからの共分散推定（rank-μ 相当）。

---

## 伝播鎖メモリ（transmission-chain memory, 2026-08-24）— REJECT

rank-μ を持ち込まずに方向情報を貯める案。宿主競合で感染が成立した子の変位 `x_child − x_parent` を単位ベクトル化して長さ k の FIFO に積み、接触感染のステップを `√(1−mix)·noise + √(mix·dim/draw)·Σ_j g_j u_j`（`g_j ~ N(0,1)`, `u_j` は FIFO から抽選）とする（E‖step‖² = dim を保存、spillover で破棄、`chain_memory_size=0` で bit-identical）。

- スクリーニング（d10, n=10, base 13.3）: k=200 / mix=0.5 で 15.8 が最良。
- 機序は否定: 分割半検定で主方向の自己一致度は伝播方向 0.566 に対し C_pop 0.738（dim10）と、d≥5 では C_pop の方が高い。採択率の差は全次元でゼロ近傍（+0.0025 / −0.0100 / ±0.0000 / −0.0028）。
- dim2 ゲートで失格: 98.42 → 95.26、改善ゼロ / 悪化 4（F18 100→70 有意）。→ REJECT・コード削除。

教訓: 「C_pop が高次元で壊れている」は軸整列シグナルが無情報という意味で、主方向が不安定という意味ではなかった。安定性（分割半検定）と正しさ（採択率検定）を分けて測ること。検証ログ: `results/20260824_195854_chain_d2_quick/`、スクリーニングは scratchpad（`screen_dimflags.py` / `chain_quality.py` / `chain_utility.py`）。

---

## 高次元悪条件の真因 — 集団の自己縮退（2026-08-24, 診断は確定 / 対策は REJECT）

best 点の数値ヘッセ行列 H を正解として、`eff = cond( H^(1/2) · M · H^(1/2) )`（M ∝ H⁻¹ で 1、M = I で cond(H)）を測った（`scratchpad/cpop_truth.py`）。

| dim | 関数 | cond_H | eff_used | cos（谷方向） | erank/dim |
|---|---|---|---|---|---|
| 2 | F01-Sphere | 1.00 | 3.81 | — | 1.51/2 |
| 10 | F08-Rosenbrock | 2.65e2 | 5.19e2 | 0.914 | 2.28/10 |
| 10 | F01-Sphere | 1.00 | 4.22e3 | — | 1.98/10 |
| 20 | F01-Sphere | 1.00 | 5.75e2 | — | 5.74/20 |

- 真因は「推定誤り / floor / 谷に入れていない」のどれでもなく集団の自己縮退。真の条件数 1.00 の Sphere で dim10 の MC-ESO は実効条件数 4.2e3 の分布から標本を引いている ＝ アルゴリズム側の欠陥。
- 実効ランクは dim10 で 1.98〜3.13 / 10。谷方向（cos 0.875〜0.914）は保たれ、動ける方向が足りない。`eff_used ≈ eff_raw` なので floor は機能していない。
- 機序: C_pop を自分が生成した集団から毎世代推定し直す閉ループで低次元部分空間へ収束する（CMA-ES は累積で免れる ＝「瞬間共分散・履歴累積なし」の裏面）。ルーター失効と同根。

対策: 単位行列方向への次元比例シュリンク `cov_shrink`（`ev ← (1−r)·ev + r`, `r = cov_shrink × (1 − 2/dim)`, dim2 で bit-identical）→ REJECT。機構チェックは合格（d10 F01 eff 4.22e3 → 53.8）したが n=20 で d3 −6.46（73.75→67.29）/ d5 −3.33 / d10 +2.71 / d20 −3.12（F02 100→20）。等方化は谷追従とランクの交換で、交換点が次元ごとに違う。flat 版も dim2 93.54 → 92.08 で失格。検証ログ: `results/20260824_223229_dshrink_d10_quick/`。

確定した制約: 谷方向は掴めている / 動ける方向数が足りない / 等方性を足す対策は谷追従とトレードオフ ＝ 「ランクを保ちながら谷追従も失わない」機構が要る。

---

## 高次元の 2 番目の失敗モード — σ が drilling 手前で固定（2026-08-25, 診断確定 / 対策 2 件 REJECT）

σ 制御則の平衡改善率 `s* = ln(1/sigma_down) / (ln(sigma_up) + ln(1/sigma_down)) = 0.350` は次元にも問題にも依存しない（`scratchpad/sigma_equilibrium.py`）。実測改善率は dim2 で 0.033〜0.060 だが、dim10 の F08 0.342 / F09 0.335 は平衡点に張り付き drill%=0.0・σ が 5e-3〜5e-2 で停止・med_f≈1。失敗は 2 系統: A = σ が drilling 手前で固定（F08/F09/F10, σ を縮めれば改善）、B = σ は floor に達したが誤った場所へ早期収束（F12, 縮めると悪化）。

- 対策1 近距離空気感染（`air_drill_ratio`, drilling 中も air を残す）→ REJECT（`scratchpad/aird_mech.py`）。ランクはわずかしか回復せず、F08 は drill%=0.0 なので構造的に届かない。ランク低下は選択が強制している（等方ステップは谷で棄却される）。
- 対策2 `sigma_up_dim_scale`（`sigma_up` だけ `(2/dim)**scale` 乗し s* を上げる; dim10 で 0.729）→ REJECT。d10 F08 med_f 1.2 → 7.2e-5、SR スクリーニング +2.5 だが dim3 ゲートで 73.75 → 70.00（−3.75）/ 71.67（−2.08）、落ちるのは F18 50→15 / F15 / F13 / F23。検証ログ: `results/20260825_*_upsc_d3_quick/`。

共通の署名: `sigma_adapt_dim_scale`（d10 +3.54 / dim3 −2.29）、`cov_shrink`（+2.71 / −6.46）、伝播鎖メモリ（+2.5 / dim2 で失格）、`sigma_up_dim_scale`（+2.5 / −3.75）。すべて「d10 で +2.5〜+3.5、dim3 で −2〜−6.5、落ちるのは多峰系」。高次元は速い σ 収縮／広い部分空間を、低次元は探索と脱出の維持を要求し、単調な次元補間ではどちらかを損なう。

---

## σ-pinning 検出器（2026-08-25〜26, 検証完了 → 採用保留）

次元で閾値を切らず病理を実行時に検出する案。改善率 EMA が s* の 0.7 倍を超えたら発火、という最初の信号は探索初期の高改善率に支配されて無差別発火（d10 で fire% 43–99%、dim3 で多峰を毀損）。→ 「σ が予算の `sigma_pin_evals_frac` にわたり drilling 閾値に到達できていない」に変更（`scratchpad/pin_mech.py`）。d10 F08 med_f 1.2 → 0.13、多峰系（F13/F15/F16/F19〜F24）は fire 0.0 で完全に不変。

| dim | base | pin30d5 | Δ SR@1e-10 | Wilcoxon 変種勝ち/base 勝ち |
|---|---|---|---|---|
| 2 | 93.54 | 93.33 | −0.21 | 0 / 0 |
| 3 | 73.75 | 73.33 | −0.42 | 0 / 0 |
| 5 | 25.62 | 26.04 | +0.42 | 1 / 0（F06） |
| 10 | 13.12 | 15.62 | +2.50 | 3 / 0（F06/F08/F09） |
| 20 | 12.29 | 11.88 | −0.42 | 4 / 1（負 F01-Sphere） |

d20 で F01-Sphere 95→70（有意）＝「drilling 未到達」と「行き詰まり」を区別できていない。停滞条件 `no_improve ≥ sigma_pin_stagnant_frac × 停滞窓` を足すと d20-F01 は直るが d10 が 16.2 → 14.6 に半減し、交換にしかならない。→ 保留。再現用 kwargs: `sigma_pin_evals_frac=0.30, sigma_pin_damp=0.5`（＋絞り込み版 `sigma_pin_stagnant_frac=0.5`）。検証ログ: `results/20260825_*_pin_d{2,3,5,10,20}_quick/`。

---

## 潜伏期 = SEIR の E（2026-08-26）— REJECT

MC-ESO は疫学的には SI しか実装していない。`incubation_gens` 世代を経ていない宿主（`pop_age` で判定）を親選択から外す案（既定 0 で bit-identical, `scratchpad/incubation_mech.py`）。d10 の erank はほぼ動かず（F01 2.68 → 3.22、F12 2.14 → 2.14）、SR スクリーニングは L=1 14.2 / L=2 13.3 / L=3 11.2（−2.08）。親の多様性は律速ではなく、ランク低下は選択（μ+λ greedy）が強制している。crowding は多解文脈で SR@1e-10 を崩して撤回済み（`results/20260616_145202_endemic_v3`）。

---

## 高次元への 7 候補 — 総括（2026-08-26 時点）

| # | 候補 | 系統 | d10 | 低次元 | 判定 |
|---|---|---|---|---|---|
| 1 | `sigma_adapt_dim_scale` | σ 制御 | +3.54 | dim3 −2.29 | REJECT |
| 2 | 伝播鎖メモリ | 分布 | +2.5 | dim2 −3.16 | REJECT |
| 3 | `cov_shrink` | 分布 | +2.71 | dim3 −6.46 / d20 −3.12 | REJECT |
| 4 | 近距離空気感染 | チャネル | — | — | REJECT（対象関数に届かない） |
| 5 | `sigma_up_dim_scale` | σ 制御 | +2.5 | dim3 −3.75 | REJECT |
| 6 | σ-pinning 検出器 | σ 制御 | +2.50（3勝0敗） | dim2 −0.21 / dim3 −0.42 | 保留（d20 に有意回帰 1） |
| 7 | 潜伏期（SEIR の E） | 選択 | +0.83 | — | REJECT（機序否定） |

持続的な成果は診断側: s*=0.350 の次元非依存性、集団の自己縮退とそれを選択が強制していること、ルーター 3 シグナルの高次元での失効、Sphere で実効条件数 4.2e3。

---

## 採択規則を親比較に変更（2026-08-26）— REJECT、ただし重要な診断

現行の子の比較相手は `dead_global = argsort(pop_f)[::-1][:n_kill]`（f 下位 25%）の元宿主で、空間的にも系統的にも無関係。子を自分の親と比較する `parent_competition`（既定 False）を試した。採択率は dim10 で 0.38–0.70 → 0.11–0.35 に正常化し（`scratchpad/parentcomp_mech.py`）、d10 F08 は med_f 1.2e+00 → 1.8e-12、d10 SR 13.3 → 15.4。しかし dim2 ゲートで 93.54 → 90.42（−3.12）、F18 100 → 35（有意, a12=0.89）で失格 → REJECT・コード削除（再現: `parent_competition=True`）。

診断: 現行の緩い規則は欠陥でなく、意味のある成功信号を犠牲にして集団の流動性を買っている。σ 適応が次元依存の「世代 best 改善」信号を使うのも同じ妥協の表れ。戻るなら「親比較で成功信号を得つつ流動性を別途担保する」二本立てが要る。検証ログ: `results/20260826_111458_pc_d2_quick/`。

---

## per-lineage 成功率の測定（2026-08-26）— σ 制御則は健全だった（因果の訂正）

二本立ての前提として「子が親に勝ったか」を記録だけする `track_parent_success` で測った（`scratchpad/parent_success_rate.py`）。親比較の成功率 par_succ は次元でほとんど動かず（F01 0.167 → 0.259）、全次元・全関数で 0.10〜0.26 ＝ 1/5 則の目標 0.2 のほぼ真上。pinned の F08 は 0.244 > 0.2 で 1/5 則はむしろ「σ を大きく」と言う。→ 二本立ては成立せず、実装は削除。

因果の訂正: σ が下がらないから前進が遅いのではなく、谷に沿った前進が遅い（ランク 2 の部分空間）から σ が下がらない。律速は σ 制御でなく探索方向の質。σ 側の対策は症状を叩いており、方向側の対策はランク低下を選択が強制するので効かず、選択を厳格化すると流動性が壊れる（三すくみ）。

---

## 二系統接触感染（split close-contact, 2026-08-26）— 高次元の突破

接触感染チャネルを 2 系統に分割し、同じ標準正規ノイズを 2 形状で変換して宿主競合に選ばせる: 瞬時系統 = `C_pop`（F02/F05 向き）、持続系統 = 学習 C（単位行列から成功ステップの rank-μ 更新 `C ← (1−c)·C + c·mean(y yᵀ)`, `y = (x_child − x_parent)/σ_used`、親に勝った接触感染の子だけが寄与。F06/F08/F09 向き）。次元ゲート `gate = clip((dim/2 − 1)/(cc_dim_ref/2 − 1), 0, 1)` は dim2 で厳密に 0、d10 で 1。ゲートに応じて air 0.30→0.10, h2h 0.40→0.20 とテーパー。

設計の要点（いずれも失敗から）:
1. 行列を混ぜない: `(1−w)·C_pop + w·C_learned` は達成可能な異方性を ~dim/w に制限し F02 が 100→0（`cov_shrink` と同じ罠）。
2. 置き換えない: 学習 C で完全置換しても F02 が 100→0。両者は補完的。
3. 判別器を作らない: 両方生成して選択に委ねる（多経路アーキテクチャが機能した最初の実例）。
- 学習率は c=0.05 が最適（0.02/0.10 で −3pt）。進化パス（rank-1 項）は全面的に悪化したので入れていない。

初回検証: dim2 93.54 → 93.54（差分セル 0）、dim10 13.12 → 24.79（+11.67, 8/0; F06 0→100 / F09 0→85 / F08 10→85）。検証ログ: `results/20260826_152052_sp_d10_quick/`。

### 学習 C 用 floor の分離（2026-08-27）

学習 C に C_pop 用の `cov_floor_low`(1e-3) を流用していたため cond 1e6 級（F10/F11/F12/F14）に 2 桁足りなかった。学習 C はランク欠損しないので `cc_cov_floor` として分離し 1e-11 に（d10, n=10: SR@1e-10 25.4 → 35.4）。

最終検証（n=20, `--all`, 予算 2500×d）:

| dim | ゲート | base | split70 | Δ SR@1e-10 | Wilcoxon 変種勝ち/base 勝ち |
|---|---|---|---|---|---|
| 2 | 0.00 | 93.54 | 93.54 | ±0（差分セル 0） | — |
| 3 | 0.12 | 73.75 | 75.21 | +1.46 | 3 / 0 |
| 5 | 0.44 | 25.62 | 38.96 | +13.33 | 5 / 0 |
| 10 | 1.00 | 13.12 | 34.38 | +21.25 | 12 / 0 |
| 20 | 1.00 | 12.29 | 13.12 | +0.83 | 8 / 0 |

全次元で有意な負けゼロ。8 候補が逃れられなかったトレードオフのない初の改善。発表（2026-07-07）時点からの SR@1e-10: d3 67.3 → 75.2（2 位 → 1 位）、d5 17.3 → 39.0、d10 6.5 → 34.4（9 位 → 4 位）、d20 0.0 → 13.1。CMA-ES との差は d5 で 5.4pt、d10 で 8.3pt。

### 先行研究調査と新規性の現状（2026-08-28）

性能改善の主因は既存アイデア。疫学メタファ（CVOA / EOSA / CVO）、C_pop（EMNA / EDA）、学習 C（CMA-ES rank-μ の移植）、リスタート時の C リセット（IPOP/BIPOP も C=I に初期化。「CMA-ES は共分散を持ち越す」は事実誤認）、複数分布から生成し選択に委ねる発想（DE/EDA, LSHADE-SPACMA 等）はいずれも新規性 なし〜低。瞬時と累積の共分散の同一世代並走は該当手法を確認できず（低〜中）。候補として (a) 診断・分析を主 contribution に、(b) ノイズ環境（`--noise`）、(c) 未調査機構の文献精査、が挙がった。

---

## ノイズ環境での評価（2026-08-28）— 多経路のノイズ耐性は否定

COCO-noisy 準拠（ノイズ付き f を見せ真値で再採点）。dim2 / gauss_sev（`f × exp(N(0,1))`）では MC-ESO 71.7（1 位, −21.8）、IPOP 42.3 / CMA-ES 19.6 と仮説に整合したが、dim10 で覆った: gauss_mild で MC-ESO 18.1（−16.3）と劣化最大、IPOP 41.3、cauchy では IPOP 46.9（−0.2）に対し MC-ESO 28.5（−5.8）。μ+λ rollback は 1 回の観測で勝敗を決めるのでノイズで過大評価された子が居座る。冗長な経路とノイズ耐性は別物 → ノイズ軸は不採用。検証ログ: `results/20260828_115523_noise_sev_d2_quick/`。

---

## 伝播系統（who infected whom）の導入（2026-08-29）— REJECT、疫学メタファの棚卸し完了

宿主ごとに系統パス `path_child = (1−c_p)·path_parent + sqrt(c_p(2−c_p))·(x_child − x_parent)/σ` を持ち学習 C の更新に使う（`lineage_path_decay`, 既定 0 で bit-identical）。d20 で F08/F12 は改善するが F10/F11/F06 が悪化し正味で負け → REJECT。親は softmax で集団全体から抽選されスロットも毎世代上書きされるので、well-mixed な集団では系統が persist せず概念自体が成立しない（意味を持たせるには空間構造化集団 ＝ cellular EA / island model が要る）。学習率の引き下げ（c=0.05 → 0.02/0.01/0.005）も d20 で F08 43→6 / F10 65→170 で正味改善なし。

棚卸しの結論: コンパートメント・潜伏期・スーパースプレッダーは CVOA/EOSA、免疫・斥力リスタートは RR-CMA-ES / HillVallEA、接触ネットワークは cellular EA / island model、組換えは交叉・DE 差分として既存。伝播系統は well-mixed のため成立せず。疫学メタファから性能改善を引き出す路線は期待値が低い。

---

## spillover での共分散リセットは有害だった（2026-08-30, 採用）

d10 F12-BentCigar（`f = x₁² + 10⁶·Σxᵢ²`）で、成功 run は C が cond 2.2e6 まで伸びるのに、失敗 run は `_on_spillover_start` が毎回 `C = I` に戻すため best_f 3.78e+01 で停止していた。単峰なので spillover の basin 脱出は不要で、学習成果だけを壊していた。

対処: 通常の spillover では C を保持し、`basin_switch` のときだけリセット（`cc_keep_on_spillover`, 既定 True）。

| dim | base | keepC | Δ SR@1e-10 | Wilcoxon |
|---|---|---|---|---|
| 2 | 93.54 | 93.54 | ±0（差分ゼロ） | — |
| 3 | 75.21 | 75.21 | ±0（差分ゼロ） | — |
| 5 | 38.96 | 40.21 | +1.25 | 0/0 |
| 10 | 34.38 | 36.04 | +1.67 | 6/1 |

d10: F10 65→85 / F11 90→100、悪化は F04 のみ。F13 med_f 1.2 → 1.1e-04。同時に不採用: 順位重み付き rank-μ（`cc_rank_weight`, 既定 0, 実装は残置; d10 で 36.7→35.4）、学習率の引き下げ（d20 で F10 65→170）。

---

## 多解路線（2026-08-30〜09-12, 一時停止中）

2026-08-30 に低次元多峰へ方針転換し、2026-09-27 のユーザー決定で主テーマは単一解性能に戻った。この路線の経緯・数値は [archive/multisolution.md](archive/multisolution.md) と git タグ `archive/multisolution-2026-09-29`（`docs/acceptance_topology.md`）。以下は採否だけを残す。

### CEC2013 niching スイートの導入と測定上の修正（2026-08-30）

CEC2013 niching の 2D/3D サブセット 7 関数（N04-N10、`--suite niching`）を導入（仕様は [experiments.md](experiments.md#cec2013-nichingn04-n10--合成関数-n11-n20)）。PR を `history_x`（全評価点）から数える旧方式は密サンプルの手法を過大評価するので、run が報告した解集合だけを採点する `niching_peak_metrics` に変更し、`OptimizeResult.final_solutions`（上限 `max(100, 2K)`）を全手法に実装した（計数は公式 `how_many_goptima` と同じ）。旧 `pr_*` / `mmo_sr_*` 列は互換のため残すが、C01-C03 の PR は論文に使わない。

### 多峰用の比較手法を 4 つ追加（2026-08-30）

Crowding-DE（NCDE の対照, `m = n_pop`）/ r3pso / NMMSO（公式 `pynmmso`）/ Repel-CMA-ES（情報化リスタートの先行例, `cma`）。`--suite niching` の既定は 7 手法（MC-ESO / NM-Restart / IPOP-CMA-ES / Repel-CMA-ES / NCDE / r3pso / NMMSO）。BIPOP と Crowding-DE は `--methods` 指定時のみ。`main.py` には追加していない。詳細は [related_work.md](related_work.md#多峰の比較手法--候補と選定2026-08-30) と [baselines.md](baselines.md#crowding-de--r3pso--nmmso--repel-cma-es多峰スイート用)。

### 初回測定・多解側の検定（2026-08-30）

dim2 / 5000 評価で MC-ESO は SR@1e-10 100% / evals 350 で最上位だが PRmean 0.45（7 手法中 6 位, NMMSO 0.75）。`wilcoxon.csv` は `best_f` の検定なので、run ごとのピーク数の検定を `wilcoxon_pr.csv` に追加（`core/runner.py:niching_peak_counts` ＋ `quick_check.py:_append_wilcoxon_pr`、符号反転で `a12 > 0.5 = reference が優れる` を保つ）。MC-ESO が勝つのは N06-Shubert2D（深さが要る）、負けるのは N07-Vincent / N10-ModRastrigin（網羅が要る）。検証ログ: `results/20260830_230835_nich0_quick/`。

### 報告解数の天井は PR の律速ではなかった（2026-08-30）— REJECT

`n_elite_max` を 20 に上げても報告点数は 23 のまま（アーカイブがほぼ空）。`n_pop` を 50 にすると報告点は倍増するが PR 0.45 → 0.39、SR@1e-10 100% → 62%。PR を縛っているのは報告枠でなく探索そのもの。`n_pop` 増は深精度と引き換えなので不採用。

### 予算ラダー（2026-08-31）

5e3 / 2.5e4 / 1e5 で MC-ESO の PRmean は 0.45 / 0.61 / 0.61 と 2.5e4 で頭打ち（NMMSO 0.76 / 0.98 / 1.00）。2.5e4 以上では SR@1e-10 の優位も消える（主要手法が 100%）。`ir_archive` / `basin_memory` は解の集合ではなく、見つけて捨てた解を報告集合に累積していないと推定した。

### 解アーカイブ（2026-08-31, 採用）

診断（`scripts/diagnose_niching.py`、計数器は `super()` を呼ぶだけで探索は不変）で N06 は 12.2 個に触れて 6.0 個しか報告していなかった。`_on_spillover_start` で放棄前 basin の best を `sol_archive_x` に追記し報告集合に含める（容量 `solution_archive_max` 既定 200、0 で旧挙動）。25000 評価で PRmean 0.61 → 0.75、SR@1e-10 と evals は完全に不変（記録のみ）。5000 評価では効果なし（0.45 → 0.45）。N07 の「探索側も不足」という読みは ε=1e-4 の 1 水準だけで測ったための見かけで、ε=1e-1 では visited 36.0（全解）と 2026-09-02 に訂正（`--eps` で複数水準を取ること）。検証ログ: `results/20260831_071330_arch25000_quick/`。

### hunt の刻みを安くする（2026-08-31, 採用）

5000 評価で hunt は 4 回、毎回 σ をフロアまで歩かせ直す待ち時間だった。採用規則: 最初の掘り切り以降、既に banked した深さに並んだら hunt を終了（`hunt_level_tol`=1e-6 × |f_init|）、後続 hunt の停滞窓は半分（`hunt_no_improve_mult`=0.5）。PRmean 0.45 → 0.52、ピーク数 2 勝 0 敗、SR@1e-10 は不変、BBOB-24 dim2 は全 24 関数が完全に同一。不採用: `exhausted_local_window`（hunt 数が減り N10 0.34→0.26）、`hunt_sigma_ratio`（PRmean 0.51 だが N06 0.19→0.15 で有意に悪化）。両者はコード削除。検証ログ: `results/20260831_131938_lvl_nich_quick/`。

### 精度ポートフォリオという定式化（2026-08-31）

「ε_hard=1e-10 の解を 1 つ保証しつつ ε_soft の解数を最大化する」定式化を検証（`scripts/niching/depth_breadth.py`）。5000 評価では深さと広さを両立する手法がなく、多解専門手法は深さで失格（N08-Shubert3D で MC-ESO 95%、多解専門手法は 0%）。hunt 深さのバンディット配分（浅い / 深い hunt の 2 腕）は REJECT（ほぼ無変化、1 run の hunt 数 ~10 では学習信号が足りない）。追試で 2.5e4 以上では NMMSO が深さ 100% / 広さ 0.98 を同時に達成し前線が 1 点に潰れたので、主張できるのは低予算（2D で 5e3）に限られる。

### `hunt_level_tol` の解放水準（2026-09-03, 未採用 / 既定値は不変）

`_basin_exhausted`（`mceso.py:919-928`）は `has_exhausted` 以降 `basin_best <= hunt_level_tol * f_init_scale` で hunt を解放する。既定 `hunt_level_tol = 1e-6` は N06 で解放水準 8e-5〜1.6e-4 になる。1e-8（`level_t08`）で N06 の PR@1e-5 0.17 → 0.76（p = 0.0006）、N07 は no-op。この経路は 5000 評価の BBOB gate では構造的に盲目なので、当時は未採用。2026-09-12 その117 で、採用形 `hunt_level_tol = 1e-5 / max(f_init_scale, 1e-12)`（`c=1.0`）が BBOB-24 dim2 × 20 seed × 2 予算 = 960 セルで `best_f` と到達評価回数がともに一致し「採用可」と判定。ただし既定は変えていない。

### 滞留していた採用候補 6 件の採否（2026-09-12 その117, 判断は確定 / 既定は未変更）

ゲートは `best_f` バイト一致、対照はこの環境の base SR@1e-10 0.9208 / `evals_succ_mean` 677.7（`CLAUDE.md` の pin 93.5% / 798 はこの環境で再現しない）。

| 腕 | つまみ | 判定 | 根拠 |
|---|---|---|---|
| `c=1.0` | `hunt_level_tol` の eps 相対化（`mceso_rel_level.py`） | 採用可 | 960/960 `best_f` 一致 ＋ `evals_succ` 480/480 一致 |
| `_sig10` | `exhausted_sigma_tol` 1.5 → 1.0 | 採用可 | 960/960 一致 |
| `_fl08` | `sigma_floor_ratio` 1e-6 → 1e-8 | 採用可 | 81/960 不一致だが pin は改善（SR 0.9208 → 0.9375、失った SR セル 0） |
| `soltrim_rho` | `sol_trim_mode="rho"`（`mceso_sol_archive.py`） | 採用可 | 評価履歴がバイト一致（`differ`=0） |
| `commit_place_r010` | `commit_sigma_mode="place"` / ratio 0.1（`mceso_commit_reseed.py`） | 採用可（ゲートの上で） | 480/480 で `best_f` / `n_evals` / `success` 一致。深さ律速の関数では有償 |
| `comp4`（上記 4 件同時） | `diagnose_niching.py` の `comp4` / `comp4_off` | N06 / N08 では採用可、N09 未測定 | N06 PR@1e-5 0.7130 → 0.9861 |

代償: MC-ESO はベースライン側なので、1 つでも既定にした時点で `docs/` と `analysis/` の MC-ESO の数値はすべて旧既定のものになり、取り直すまで新旧を混ぜて引用できない。

## 2D の対照を固め直した（2026-09-27 その177, **既定は不変**）— 単一解テーマの基準値

**2026-09-27 のユーザー決定で主テーマが単一解性能に切り替わったので、判定の対照を今の環境・今の MC-ESO で測り直した。**
`./run.sh quick --all --methods "MC-ESO,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,DE,L-SHADE" --n-runs 20 --max-evals 5000`
（2D BBOB-24、24/24 完走、29.0 分）。**以後の 2D の改善判定はこの行を対照にする。**

| | SR@1e-2 | SR@1e-4 | SR@1e-7 | **SR@1e-10** | `evals_succ_mean` |
|---|---|---|---|---|---|
| **MC-ESO（対照）** | 95.42% | 95.21% | 93.12% | **92.08%**（6 手法中 **1 位**） | **677.7**（**2 位**、上は CMA-ES 484.1 のみ） |
| DE | 95.83% | 91.25% | 90.21% | 89.38% | 1337.9 |
| IPOP-CMA-ES | 88.96% | 87.71% | 86.67% | 83.12% | 1011.6 |
| BIPOP-CMA-ES | 86.04% | 84.58% | 82.29% | 77.50% | 1095.7 |
| L-SHADE | 91.88% | 91.25% | 84.79% | 76.88% | 1765.7 |
| CMA-ES | 68.75% | 65.83% | 64.17% | 59.58% | 484.1 |

**MC-ESO 側は既存 base と 4 桁一致**（SR@1e-3 0.954167 / 1e-5 0.941667 / 1e-7 0.931250 / **1e-10 0.920833** ／ `evals_succ_mean` 677.7391）
＝ **4 度目の再現。`CLAUDE.md` の pin（93.5% / 798）は 4 回連続で再現しない**（pin の数値は書き換えない）。
**MC-ESO が SR@1e-10 で下回るのは 6 関数**（**F04 / F06 / F17 / F18 / F20 / F24**。07-27 の 7 関数との重なりは 5 で、
**F06-AttractiveSector が新規に入り、F13 / F23 が抜けた** —— F23 は MC-ESO 75% に対し**比較 5 手法すべて 0%**）。
**注意**: **F06 の Wilcoxon 有意・large な負けは 1e-10 の 5 桁下**（median `best_f` 7.1e-15 対 厳密 0.0）**で、主指標を 1pt も動かさない。**
詳細・関数別表・SR 梯子は [findings.md](findings.md) の その177 の節、保存物は `analysis/single/e177/`。

## 5D の対照を初めて同条件で測った（2026-09-28 その178, **既定は不変**）— 単一解テーマの基準値

**この研究に 5 次元の同条件比較は 1 度も無かった**（下の 2026-08-28 の表は**旧環境・10 手法・別の MC-ESO 版**の記録）。
`./run.sh quick --all --dim 5 --max-evals 12500 --n-runs 20 --methods "MC-ESO,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,DE,L-SHADE"`
（5D BBOB-24、24/24 完走、22.7 分。**24 関数を 4 shard に割って並列に回したが RNG は同一** —— `core/runner.py:54` の seed は run 番号だけで決まる）。
**以後の 5D の改善判定はこの行を対照にする。**

| | SR@1e-2 | SR@1e-4 | SR@1e-7 | **SR@1e-10** | `evals_succ_mean` |
|---|---|---|---|---|---|
| IPOP-CMA-ES | 66.46% | 64.58% | 62.29% | **57.29%**（1 位） | 3592.9 |
| BIPOP-CMA-ES | 66.67% | 64.58% | 61.04% | 54.17%（2 位） | 3881.9 |
| DE | 57.71% | 53.54% | 50.00% | 43.96%（3 位） | 4770.4 |
| CMA-ES | 50.42% | 47.71% | 47.08% | 43.33%（4 位） | **1451.2** |
| **MC-ESO（対照）** | 55.42% | 51.88% | 47.08% | **43.12%（6 手法中 5 位）** | **5240.9**（5 位） |
| L-SHADE | 37.71% | 30.42% | 18.75% | 15.62%（6 位） | 7049.1 |

**2D の「1 位・+2.70pt・速さ 2 位」は 5D で「5 位・IPOP に −14.17pt・速さ 5 位」に反転する。**
**下回るのは 13 関数**（F03 / F04 / F07 / F08 / F09 / F12 / F13 / F14 / F16 / F17 / F18 / F20 / F21。2D の 6 関数との重なりは 4）。
**赤字 22.71pt の 41%（9.375pt）は<u>単峰・条件数の関数</u>から来る**（F07 / F08 / F09 / F12 / F13 / F14 —— **2D ではこの 6 関数すべて 100% だった**）。
**素の CMA-ES が g3（高条件数・単峰）で MC-ESO を 26.0pt 上回る**（84.00 対 58.00。F12 30 対 100、F14 35 対 100）。

**上の 2026-08-28 の表との突き合わせ（重要）**: **比較手法側はほぼ再現するが、MC-ESO だけ大きく動く** ——
**CMA-ES 44.4 → 43.33（−1.07）／ IPOP 56.0 → 57.29（+1.29）／ MC-ESO 39.0 → 43.12（+4.12）。**
**2D の pin のずれ（93.54 → 92.08 ＝ −1.46）とは<u>符号が逆</u>なので、「環境差」だけでは両方を説明できない**（キュー 3 への引き継ぎ）。
詳細・関数別表・SR 梯子・群別内訳は [findings.md](findings.md) の その178 の節、保存物は `analysis/single/e178/`。

### 10D の同条件基準（2026-09-28 その179、この環境で初測定）

`./run.sh quick --all --dim 10 --max-evals 25000 --n-runs 20 --methods "MC-ESO,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,DE,L-SHADE"`
（10D BBOB-24、24/24 完走、2 本合わせて 86.2 分。**24 関数を 6 shard に割って並列に回したが RNG は同一** —— `core/runner.py:54` の seed は run 番号だけで決まる）。
**以後の 10D の改善判定はこの行を対照にする。**

| | SR@1e-2 | SR@1e-4 | SR@1e-7 | **SR@1e-10** | `evals_succ_mean` |
|---|---|---|---|---|---|
| IPOP-CMA-ES | 61.46% | 59.58% | 53.54% | **50.00%**（1 位） | 6464.7 |
| BIPOP-CMA-ES | 58.75% | 55.83% | 53.12% | 49.17%（2 位） | 8558.7 |
| CMA-ES | 47.71% | 45.21% | 44.58% | 41.88%（3 位） | **3672.8** |
| **MC-ESO（対照）** | 40.42% | 38.75% | 37.50% | **36.46%（6 手法中 4 位）** | 8693.5（4 位） |
| DE | 28.54% | 22.29% | 16.04% | 15.42%（5 位） | 10700.9 |
| L-SHADE | 17.29% | 12.50% | 8.33% | 4.17%（6 位） | 14838.3 |

**上の 2026-08-28 の表との突き合わせ**: **D=10 は 3 手法とも再現する** —— **MC-ESO 34.4 → 36.46（+2.06）／ CMA-ES 42.7 → 41.88（−0.82）／ IPOP 47.1 → 50.00（+2.90）。**
**＝ 5D で出た「MC-ESO だけ +4.12 動く」非対称は 10D では再現せず、いちばん動いたのは IPOP（+2.90）である。**
**3 次元での MC-ESO の乖離は −1.46（2D）／ +4.12（5D）／ +2.06（10D）で符号も大きさも揃わない。**

**下回るのは 12 関数**（F07 / F08 / F09 / F10 / F12 / F13 / F14 / F16 / F17 / F18 / F20 / F21。5D の 13 関数との重なりは 11）。
**赤字 15.42pt の 54% が F12-BentCigar と F07-StepEllipsoidal の 2 関数**（どちらも MC-ESO 0% 対 比較手法 100%）、**78% が単峰系（g2 ＋ g3）から来る。**
**そして MC-ESO が 6 手法の関数別包絡線に上乗せしている量は +0.00pt**（2D +4.38 → 5D +0.42 → 10D +0.00）＝ **10D では単独 1 位の関数がゼロ。**
詳細・関数別表・SR 梯子・群別内訳は [findings.md](findings.md) の その179 の節、保存物は `analysis/single/e179/`。

### 2026-09-29 その182 — 学習 C の採用規則を「親に勝った子」から「その世代の上位 μ」へ（`cc_mu_frac`、腕のみ・既定不変）

**追加したフラグ**: **`cc_mu_frac`（既定 0.0 ＝ 従来規則。既定は 1 つも変えていない）。**
`> 0` で `_update_cc_cov` の rank-μ 採用規則の三重目（`child_f < pf[k]` ＝ 自分の親に勝つこと）を
**「その世代の配置済み close 子を f 昇順に並べた上位 ⌈`cc_mu_frac`×n⌉ 本」**（＝ CMA-ES の rank-μ の選択）に置き換える。
腕の登録は `quick_check.py` に 3 本（`ccmu50` = 0.50 ／ `ccmu25` = 0.25 ／ `ccmu100` = 1.00）。

**結果（10D BBOB-24、n=20、25000 評価、`ccmu50` のみ実測。24/24 完走、壁時計 25.9 分）**: **採否は俯瞰の判断事項。**

| | SR@1e-2 | SR@1e-4 | SR@1e-7 | **SR@1e-10** | `evals_succ_mean`（共通 13 関数） |
|---|---|---|---|---|---|
| MC-ESO（対照、その179 を全列再現） | 40.42% | 38.75% | 37.50% | **36.46%** | 8693.5 |
| **`ccmu50`** | 43.12% | 40.83% | 38.75% | **37.71%（+1.25pt）** | **7420.5（−14.6%）** |

**関数別**: 改善 4（**F14 85→100 / F09 85→95 / F08 75→85 / F12 0→5**）、悪化 2（**F22 20→15 / F21 15→10**、どちらも 1 run）、変化なし 18。
**Wilcoxon（ref = MC-ESO）は腕が有意に優る 3 関数（F13 p=0.0172 A12=0.275 large ／ F10 p=0.0477 small ／ F18 p=0.0328 small）、有意な悪化ゼロ。**
**2D では bit 一致**（`_cc_dim_gate()` が dim2 で厳密に 0。F01/F17 × 3 run で `best_f` 6 値完全一致で確認）。
**＝ この病理への 4 件の棄却（下記 :865 / :979 / :1041 ほか）と違い、「局所改善・他が悪化で正味ゼロ」の形にならなかった初めての軸。**
**ただし赤字の 54% を作る F07 / F12 は閉じない**（F12 の 5% は 1 run で、`best_f` の分布は動かない: p=0.9563 / A12=0.4800）。
全文・梯子・群別は [findings.md](findings.md) の その182 の節、保存物は `analysis/single/e182/`。

**削除したフラグ 2 つ（実装されていなかったもの）**: **`cc_rank1_weight` と `cc_path_decay`。**
`_update_cc_cov` の進化パスブロックは `if False:` で恒久的に止まっており、中の `c1` は定義すらされていなかった（有効化すると `NameError`）。
**実装ではなく削除を選んだ**のは、**大域進化パスが下記 :865 で既に棄却済み**だから。状態フィールド `cc_path` とその初期化も同時に落とした。
**削除したコメントにしか無かった測定値 1 つを保全する**: **close-contact の成功ステップは 1 世代あたり 0.1-0.3 本**（サンプル飢餓の一次資料）。

### 2026-09-29 ローカル検証 — 高次元の腕 3 種（n_pop の次元係数 / 学習率 / flat-fitness の σ 規則。腕のみ・既定不変）

ループ外のローカルセッションで回した（`local/hd-cov-sigma`、commit `c8f1be0`）。事前登録と集計は `analysis/single/local_hd/`。
**環境の注意**: ローカルの macOS では、その182 の base を再現しない（10-05 訂正: 当初 numpy の版のせいと書いたが、クラウドも numpy 2.4.6 で再現しているので原因は未特定）（10D 23 関数 × 6 列で 138 セル中 54 セルが違う）。**比較は同じ run の中の base と腕だけ**。

**先に取ったトレース（d10、既定）で、F07 と F12 は別の欠陥だと分かった。**
- **F07-StepEllipsoidal**: 集団 40 体すべてが同じ f の平坦面に乗り、σ が 1e-6·span の床まで潰れて凍る。spillover は 1 run 23〜28 回で、毎回同じ形で止まる。
- **F12-BentCigar**: 学習 C の cond が必要な 1e6 級に育つのが予算の最後（成功 seed で 2.3e6 到達時に best 5.5e-06、失敗 seed は cond ~2e3 止まり）。

**追加したフラグ**: `n_pop_dim_mult`（既定 4.0 ＝ 従来の `max(20, 4·dim)`）、`sigma_flat_expand`（既定 False。CMA-ES の flat-fitness 規則 ＝ 世代の子の半数以上が直前の best と同値なら σ を広げる。drilling 外のみ）。

**結果（10D BBOB-24、n=20、25000 評価）**

| 腕 | SR@1e-2 | SR@1e-4 | SR@1e-7 | SR@1e-10 | Wilcoxon（base 有意勝ち / 腕 有意勝ち） |
|---|---|---|---|---|---|
| MC-ESO（同 run の対照） | 41.5% | 38.5% | 36.2% | 35.2% | — |
| `npop8`（n_pop 40 → 80） | 44.0% | 39.0% | 35.0% | 33.8% | 5 / 5 |
| `npop8_ccmu50`（上 ＋ `cc_mu_frac` 0.5） | 45.6% | 41.0% | 39.0% | **37.1%** | 5 / 7 |
| `ccmu50_lr10`（`cc_mu_frac` 0.5 ＋ 学習率 0.10） | 41.2% | 39.0% | 34.6% | 30.2% | 1 / 0 |
| `flat` | 41.5% | 38.5% | 36.2% | 35.2% | 0 / 0 |

- **F07 は個体数で開いた**: SR@1e-10 が 5% → `npop8` 40% / `npop8_ccmu50` 30%。**`flat` は SR を動かさない**（5% のまま。median `best_f` は 1.4638 → 0.93351）＝ **σ の潰れは起きているが、SR を決めているのは σ ではなく個体数**。
- **F12 はどの腕でも 0% のまま**。median `best_f` の最良は `npop8_ccmu50` の 0.78168（base 2.1834）で、1 桁に届かない ＝ **「学習速度」はこの 3 つのつまみでは解けない**。
- **`npop8` 系の代償**: F09 85% → 50%（両腕）。易しい関数の評価回数が 30〜70% 増える（F01 +67〜70%、F06 +50〜58%）＝ 世代数が半分になる代金。`npop8_ccmu50` は F14 65% → 100%、F10 75% → 90%、F08 75% → 85% と悪条件群を押し上げる一方、base が有意に勝つ関数が 5 つ（F09 / F11 / F19 / F22 / F24）。
- **`ccmu50_lr10` は不採用**（−5.0pt。F11 100% → 50%、F14 65% → 15%）。本数を増やしても学習率 0.10 は速すぎる。
- **2D**: `flat` は 24 関数すべてで base と SR・評価回数とも一致（92.5%）。他の腕は構造上 2D で n_pop = 20 / gate = 0。

**判定**: 採用はゼロ。**事前登録の (b)（F07 か F12 が 15% 以上）は `npop8` / `npop8_ccmu50` の F07 で発火**。次に試す価値があるのは「個体数を増やしても易しい関数の世代数を削らない」形（例: 再起動ごとに n_pop を増やす IPOP 型）で、F12 には別の軸が要る。

### 2026-09-29 その183 — `cc_mu_frac` の 5D と μ 割（腕のみ・既定不変）

5D で `ccmu50` は SR@1e-10 43.12% → 44.17%（+1.04pt）。改善は悪条件・単峰（F14 / F13 / F12）、悪化は Rastrigin 系（F03 15→0 / F15 / F19）で、F03 は Wilcoxon で有意に悪化。
10D の `ccmu100`（選抜なし）は 37.50% で `ccmu50`（37.71%）と 1 run 差 ＝ 効いているのは本数。
**10D で「有意な悪化ゼロ」に見えたのは、悪化しうる多峰の関数がもともと 0% だったから**。副作用は 5D で測る。詳細は [findings.md](findings.md) の 6 節、集計は `analysis/single/e183/`。

### 2026-10-05 ローカル軽量検証 — スピルオーバー直後の学習 C を凍結すると F10 / F14 が開き、F12 の誤差が 3 桁下がる（腕のみ・既定不変）

ローカル（macOS）で 10D 15 関数 × n=10 / 5D 9 関数 × n=20。集計は `analysis/single/local_1005/`。正準環境での確認は測定ジョブ 1・2。

- **F12 の診断（`diag_f12.py`）**: CMA-ES は 5000 評価で cond(C) 1e5 に届く。MC-ESO の学習 C は失敗 seed で 1e2〜1e4 を上下し、`cc_mu_frac`=1.0 でも学習率 0.2 でも育たない。スピルオーバーを止めると 25000 評価で 3.9e5 / 1.0e6 まで育つ（f 40 → 0.28、1.9 → 2.9e-4）。**＝ 律速はサンプル数ではなく、一様に撒き直した直後の子が学習 C を丸めること。**
- **`cc_spill_freeze_gens`（追加）**: 通常のスピルオーバー後 N 世代は学習 C を更新しない。10D 15 関数で SR@1e-10 55.33% → 60.67%（`ccfrz25` / `ccfrz50` とも）。F10 70 → 100%、F14 50 → 100%。base が有意に勝つ関数ゼロ。F12 は SR 0% のままだが median `best_f` 2.1834 → 6.0919e-03（`ccfrz50`）。150 世代はスピルオーバーの間隔（約 75 世代）を超えて学習が止まり逆効果。
- **`ipop_growth`（追加、IPOP 型の個体数増加）**: スピルオーバーごとに 1.5 倍（上限 4 倍）。F07 は 10 → 50%（`ipopS15`）だが F09 80 → 30%、F08 70 → 40% で、base が有意に勝つ関数が 7 つ。`ipopS15_ccmu50` は 55.33% → 58.67% だが F09 は同じく −50pt。**谷をたどる関数を壊すので見送り。** basin 乗換えだけで増やす版は F07 でほとんど発火しない。
- **`cc_mu_droplet_only`（追加）**: `cc_mu_frac` をルーターが悪条件と判定した run だけに効かせる。5D で多峰の関数は守れるが、F14 の伸びが +55 → +15pt に縮む（9 関数平均 35.00% → 28.89%）。悪条件の関数の多くが droplet と判定されていない。**見送り。**

### 2026-10-05 ローカルのオフライン検証 — 「MC-ESO が捨てている情報」から形を読む 3 案はいずれも否定（実装なし）

どれも腕は作らず、トレースとオフラインの当てはめだけで判定した。同じ検証を繰り返さないための記録。
- **評価点の値から曲率を読む**（ペアの差分商 ／ 最小二乗の 2 次当てはめ、失敗も含める）: 10D で F12 の平らな向きは 120〜300 点で 2〜9° のずれで読めるが、F10-EllipsoidalRot は 65〜85°（推定 cond 2.5e2 対 実際 1.5e5）、F07 は 46〜89° で読めない。急な方向の誤差と BBOB の T_osz / T_asy 変換が平らな方向をかき消す。**先行研究**: LS-CMA-ES（Auger, Schoenauer, Vanhaecke, PPSN 2004）がアーカイブした子（失敗を含む）に 2 次を当てはめ、逆 Hessian を共分散に使う。HE-ES（Glasmachers & Krause 2020）も近い。残るのは「2 つの学習器を常に並走させ選択で裁定する」部分だけ。
- **局所解のお椀（big valley）**: アーカイブした局所解に f ≈ a + bᵀx + c|x|² を当てはめ、その底を次の撒き直し先にする案。底が best より真の最適解に近いのは 10D の F03 / F15 / F17 / F18 で 0/6、5D の F03 / F15 / F17 で 0/6、F20 / F21 / F24 は 2〜3/6。アーカイブの局所解は best の近くに固まっていて大域の形を持たない。
- **リスタート直後の変位を大域の形の学習に回す**: 撒き直し後 50 世代の成功ステップの共分散と、箱全体に当てはめた 2 次の最急方向とのずれは、F18 で 74〜86°、F15 で 71〜88°（10D でランダムな向きは約 72°）。揃うのは単峰の F10（5〜15°）だけで、そこは今の学習 C がすでに捉えている。

### 2026-10-05 出自の選別 `cc_gate_mahal` — 凍結タイマーの代わりに「best の近くの親」だけから学習 C を学ぶ（腕のみ・既定不変）

professor の指摘（凍結は停滞窓に縛られた定数で、バグ修正に見える）を受けて、タイマーを使わない形に置き換えた。学習 C を更新するのは、現在の best から学習 C の計量で測った距離が 2·σ·√d 以内にある親の子だけ。
**反証試験**（10D、F10 / F11 / F12 / F14、n=10、停滞窓 150 / 300 / 600、`analysis/single/local_1005/robust_window.txt`）: 4 関数平均の SR@1e-10 は base 0.500 / 0.550 / 0.725（振れ幅 0.225）、凍結 `frz50` 0.600 / 0.750 / 0.750（0.150）、**`gate2` 0.700 / 0.800 / 0.800（0.100）**。半径 3 / 5 は凍結と同程度。
F12 は `gate2` で SR@1e-10 が 0.1 / 0.2 / 0.2（凍結はどの窓でも 0.0）、median `best_f` は窓 300 で 2.6e-05（凍結 6.1e-03、base 2.2e+00）。**判定: 反証条件（凍結と同じくらい窓に敏感なら捨てる）は不発。正準環境で測る（ジョブ 2・3）。**

### 2026-10-05 / 10-06 正準環境での凍結（`ccfrz*`）— 10D・5D とも base 以上、ただし 5D で多峰が落ちる

測定ルーチンのジョブ（集計は `analysis/single/j1/`（10D）と `j2/`（5D）、作業ログに全文）。SR@1e-10 は 10D で base 36.46% → `ccfrz25` / `ccfrz50` 37.29%、`ccfrz50_ccmu50` 38.33%。5D で base 43.12% → 43.54% / 44.38% / 47.29%。base が有意に勝つ関数は両次元・3 腕ともゼロ。10D の F12 の median `best_f` は 2.42 → 1.59e-3。5D では F03 15% → 0〜10%、F15 10 → 5%、F19 5 → 0% と多峰が落ちる。伸び幅はローカル（15 関数、+5.3pt）より小さい。

### 2026-10-06 2D の部品ごとの ablation（正準環境）— 2D の 1 位は「接触感染と飛沫感染を宿主競合が裁く」組み合わせから来る

測定ルーチンのジョブ（`analysis/single/j3/`、作業ログに全文）。base 92.08% を再現。SR@1e-10 の変化: 接触感染だけ −25.00、宿主競合なし −19.79、飛沫感染なし −17.50、スピルオーバーなし −12.08、接触感染を等方 −5.21、ルーターなし −4.17、drilling なし −1.25、系統共存なし −0.42、**空気感染なし −0.21（有意差ゼロ、F24 は +20）**。
2D では学習 C は動かないので、2D の勝ちは 2 系統の共分散ではなく、性質の違う生成法（ガウスの接触感染と DE 型の飛沫感染）を並べて選択に裁かせる設計から来ている。2026-07-08 の ablation と順位は同じ。空気感染は 2D では仕事をしていない → 5D / 10D で外す測定をジョブに入れた。

### 2026-10-06 ローカル軽量検証 — チャネル間の学習共有は見送り、勢いのチャネルは空気感染と差し替えると有望（腕のみ・既定不変）

ローカル（macOS）で 10D 15 関数 × n=10 と 2D 24 関数 × n=20。集計は `analysis/single/local_1006/`。
- **`cc_learn_droplet`（飛沫感染の成功も学習 C に入れる）**: 10D で 55.33% → 58.00%、F09 −20pt、F13 で base が有意に勝つ。`cc_gate_mahal`=2 と組むと 56.67%（選別だけなら 64.67%）。F12 の診断では cond(C) が 1e2 前後で止まる（変位を √d の単位長にそろえた版）。学習 C の計量で正規化した版（`cc_learn_droplet_norm="mahal"`）は cond が 1e4〜1e6 まで育つが f は進まない（seed 200 で 8.6、base 1.9）。DE の変位は集団の広がりで決まり、接触感染の分布と別の形に引く。**見送り。**
- **`mom_ratio`（勢いのチャネル: 宿主が生まれたときの変位を κ ~ U(1,2) 倍だけ延長）**: 空気感染を残す `mom10` は 2D 92.50% → 89.79%（F16 −25、F17 −30、F18 −40pt、base 有意勝ち 3）で 2D の規則に反する。**空気感染の枠を置き換える `mom10_noAir` は 2D 93.54%（+1.04、base 有意勝ちゼロ、F23 +50pt）、10D 60.00%（+4.67、F08 / F09 / F10 / F14 +20pt）。** ただし 10D の F21 −20pt と F13 で base が有意に勝ち、2D の F18 −10 / F19 −15 / F20 −10pt。勢いと空気感染の削除のどちらが効いているかは未分離（`abl_noAir` は測定ジョブ 3）。

### 2026-10-06 ローカル（macOS）の 2D 全手法比較 — MC-ESO は IMODE と同率 1 位、評価回数は上位で最少、単独 1 位の関数はゼロ

BBOB-24 2D、n=20、5000 評価、MC-ESO ＋ 33 手法（多解専用とライブラリの重複版を除く）。集計は `analysis/single/local_all2d/`（`.csv.gz`）。正準環境での確認は測定ジョブ 2。
- SR@1e-10: IMODE 92.50% ＝ **MC-ESO 92.50%** ＞ EBOwithCMAR 91.46 ＞ LSHADE-cnEpSin 91.25 ＞ L-SHADE（移植）91.04 ＞ jSO 90.42 ＞ ELSHADE-SPACMA 90.21 ＞ LSHADE-SPACMA / SPS-L-SHADE-EIG 90.00 ＞ DE 89.38 … IPOP-CMA-ES 83.54（16 位）、EA4eig 79.79（19 位、簡略版 88.12）、CMA-ES 64.38。
- MC-ESO の SR@1e-4 96.04% / SR@1e-7 95.00% は全手法で最高。`evals_succ_mean` は 799 で、SR@1e-10 が 88% 以上の手法の中で最少（MOS 938、jSO 1401、IMODE 1800、EBOwithCMAR 2004）。
- 全手法の関数別最良（仮想最良）は 96.67%で、MC-ESO を除いても同じ ＝ **MC-ESO が単独 1 位の関数はゼロ**（同率 1 位 19 関数）。下回るのは F17（75 対 100%）、F24（10 対 45%、DE）、F23（50 対 75%、NM-Restart）、F06（90 対 100%）、F20（95 対 100%）。
- 予算に合わない設計の手法がある（HSES・PS-CMA-ES は長い予算向け）。NGOpt / NG-Portfolio は 2D でも弱く遅い。

### 2026-10-06 正準環境での出自の選別（`ccgate2`）— 10D +1.46pt、5D はほぼ同着。5D の多峰が落ちる向きは凍結と同じ

測定ルーチンのジョブ（集計は `analysis/single/j4/`（10D）と `j5/`（5D））。SR@1e-10: 10D で base 36.46% → `ccgate2` 37.92% / `ccgate2_ccmu50` 37.50%（base の有意勝ちゼロ、F12 が初めて 5%、F14 85→100%）。5D で base 43.12% → 43.33% / 46.25%（F14 35→60 / 80%）。5D では F03 / F15 / F19 が 5〜10pt 落ち、base が有意に勝つ関数が各 1 件（`ccgate2` は F19、`ccgate2_ccmu50` は F01）。凍結の最良（`ccfrz50_ccmu50` 10D 38.33 / 5D 47.29%）と同程度で、ローカルで見えた +9pt 級の伸びは出なかった。

### 2026-10-06 改善案 4 系統を腕として実装（既定不変、測定ジョブ 2〜5）

- A 集団サイズ: `pop_schedule="linear"`（`pop_init_mult`·D → `pop_final_mult`·D、下限 `pop_min`、評価回数に比例して最悪の宿主を外す）、`ipop_trigger="failstreak"`（スピルオーバーに `ipop_fail_streak` 回続けて失敗したら集団を増やす）。
- B ルーター v2: `route_mix`（空気感染を使わず、既存ルーターが決めた経路ごとに飛沫感染と勢いの配分を変える）。
- C `h2h_adapt`: 飛沫感染の F / CR を SHADE 方式の成功履歴（H = 6）で適応。成功は「宿主競合で元の宿主より厳密に良く生き残った」。
- D `ls_final_frac` / `ls_budget_frac`: 予算の終盤に SciPy SLSQP（勾配評価も予算に数える）、最良点を最悪の宿主と入れ替えて続行。
- 既定値は 2D・10D の 8 関数 × 2 シードで変更前と bit 一致。1 seed の試し打ち（判断には使わない）で、`pop_lin16` が 10D F07 で 0、`h2hA` が 10D F12 で 3.3e-7。

### 2026-10-06 / 07 正準環境の測定ジョブのまとめ（改善案 4 系統と全手法比較）

**改善案 4 系統**（`analysis/single/j18/`〜`j21/`、SR@1e-10 の base との差、2D / 5D / 10D）:
- `pop_lin16`（16·D → 4·D）+1.88 / **+15.83** / **−5.00**。5D は 43.12 → 58.96% で F07・F12・F13・F14 が 100% 近く、base の有意勝ちゼロ、`evals_succ_mean` −952.5。10D は F07 0 → 60% だが F08 75 → 40、F09 85 → 10、F14 85 → 35 と落ち、易しい関数の評価回数が 2〜3 倍（F01 2409 → 6669）。`pop_lin16_2` +1.88 / +15.83 / −1.88、`pop_lin8` −1.88 / +12.29 / −2.08、`ipop_fail3` −2.92 / +3.75 / +1.04。
- ルーター v2: `rv2` −0.83 / −7.71 / −14.38、`rv2_flat` +1.46 / −9.58 / −10.00。10D では経路ごとに変える方が対照より 4.38pt 悪い。**見送り。**
- `h2hA` +0.21 / −2.71 / −0.21、`lsfin` +0.62 / +0.21 / ±0.00（10D で `best_f` の有意勝ち 7 関数だが 1e-10 は動かない）。新規性も低く、**見送り**。

**全手法比較**（`analysis/single/j11/`〜`j14/`）:
- 2D: MC-ESO 92.08% で IMODE と同率 1 位、`evals_succ_mean` 677.7（IMODE 1703.7）。単独 1 位の関数ゼロ。
- 5D: グループ A+B（17 手法）で 13 位、1 位 ELSHADE-SPACMA 74.58%、2 位 LSHADE-SPACMA 72.29%。グループ C+D（16 手法）で 8 位、1 位 AMALGAM-SO 57.71%。MC-ESO の単独 1 位は F19 のみ。
- 10D: グループ A（9 手法）で 5 位、1 位 IPOP-CMA-ES 49.58%、2 位 BIPOP-CMA-ES 48.33%。グループ B〜D は測定中。
- 比較手法の修正の影響は [findings.md](findings.md) の 0 節（CMA 系は 1pt 未満、L-SHADE は 5D 15.62 → 58.54、10D 4.17 → 32.08）。

### 2026-10-07 ローカル（macOS）軽量検証 — 集団をべき乗で早めに縮めると 10D の悪化が消える

10D / 5D の 12 関数（F01 / F02 / F06–F10 / F12–F15 / F21）× n=10。集計は `analysis/single/local_pop/`。正準での確認は測定ジョブ 1・2。
- 10D の SR@1e-10（12 関数平均）: base 50.83%、`pop_lin16` 45.00%、**`pop_pow2` 58.33%**、**`pop_pow3` 57.50%**、`pop_pow2_2` 53.33%、`pop_lin16_frzmu` 55.83%、`pop_pow2_gate2` 51.67%、`pop_lin16_gate2` 45.00%。`pop_pow2` は F07 10 → 100%。base が有意に勝つ関数はどの腕もゼロ。評価回数は `pop_pow3` +8.4%、`pop_pow2` +16.2%、`pop_lin16` +23.4%。
- 5D: base 65.00% → 全腕 83.33〜88.33%。`pop_pow3` は 85.83% で評価回数 −19.7%。
- 出自の選別（`cc_gate_mahal`）との組み合わせは上乗せしない。凍結 ＋ `cc_mu_frac` は線形版の 10D の悪化を和らげる（45.00 → 55.83%）。

### 2026-10-07 / 08 正準環境の測定ジョブのまとめ（集団のべき乗縮小・全手法比較の完了・勢い・空気感染・β=0・包絡線）

- **集団のべき乗縮小**（`analysis/single/j22/`・`j23/`、SR@1e-10）: `pop_pow2_frzmu` 2D 93.33% / 5D 59.17% / 10D 42.08%（base 92.08 / 43.12 / 36.46、base の有意勝ちは 3 次元ともゼロ）。`pop_pow2` 93.33 / 58.54 / 36.25、`pop_pow3` 系は 2D 90.42%（F04 95 → 75、F23 75 → 50）で 2D の規則に反する。10D の評価回数は対応平均で 8693.5 → 11658.1、易しい関数は 1.4〜2.5 倍。
- **全手法比較（32 手法）の最終順位**（`analysis/single/j11/`〜`j17/`）: 2D は MC-ESO が IMODE と同率 1 位（92.08%、仮想最良 96.88%）。5D は 20 位（1 位 ELSHADE-SPACMA 74.58%、LSHADE-SPACMA 72.29%、jSO 62.50%、仮想最良 83.54%）。10D は 10 位（1 位 LSHADE-SPACMA 53.12%、AMALGAM-SO 52.92%、AMALGAM-SO-DE 51.67%、仮想最良 66.67%）。
- **勢いのチャネル** `mom10_noAir`: 2D +1.46、5D −9.58（base の有意勝ち 4）、10D +0.62pt。**見送り。**
- **空気感染を外す** `abl_noAir`: 5D +0.21pt。10D は高次元の空気感染の割合を `cc_air_ratio` が決めるため `air_ratio=0` だけでは効かず、全 run が base と同値（ジョブの書き方の誤り）。
- **β=0**（`dimf_softmax0`）: 5D −19.79pt、10D −28.75pt。2D では β=0 が +1.25pt だったが、高次元では既定の β=5 が圧倒的に正しい。**既定は変えない。**
- **包絡線への上乗せ**（10D、予算 1 / 2 / 4 倍、比較 5 手法）: 包絡線 51.46 → 69.79 → 74.17% に対し、MC-ESO を加えたときの上乗せは 3 予算とも +0.00pt、差は −15.00 → −31.25 → −34.79pt。上乗せが消えるのは予算（1 分布あたりのサンプル数）ではなく次元のため。

### 2026-10-08 既定値を変更（ユーザー承認）— `pop_pow2_frzmu` を MC-ESO の既定にする

`pop_schedule="linear"`（16·D → 4·D、下限 10）、`pop_shrink_power=2.0`、`cc_spill_freeze_gens=50`、`cc_mu_frac=0.5`。新しい既定値は `pop_pow2_frzmu` と 2D / 5D / 10D の 16 件で bit 一致することを確認。旧既定値は `quick_check.py` の `MC-ESO-v0`（F07 10D seed 0 で 5.5629e-01 を再現）。**2026-10-08 より前に作った腕は変える引数だけを指定しているので、今後は新しい既定値の上に重なる。記録済みの数値は旧既定値に対するもの。**
性質として、集団の縮小は予算の消費割合で決まるので、MC-ESO は総予算を事前に知っている前提になる（L-SHADE 系と同じ）。易しい関数では評価回数が増える（2D 677.7 → 766.7、対応平均）。集団の縮小は L-SHADE の LPSR / NL-SHADE の非線形縮小に先例があり、新規性の主張には使わない。3 要素の寄与（`v1_noPop` / `v1_noFrz` / `v1_noMu`）と新しい基準値は測定ジョブ 1・2。

### 2026-10-08 ローカル（macOS）軽量検証 — 新しい既定値の上の 5 案はいずれも見送り

2D は 24 関数 × n=20、5D / 10D は 12 関数（F01 / F03 / F04 / F10 / F12 / F13 / F15 / F17 / F18 / F20 / F21 / F22）× n=10。集計は `analysis/single/local_v1x/`。
- **ルーターの判定**（`routes.py`、5 seed）: 5D / 10D の F03 / F04 / F20 はすべて keepair（変数分離の close と判定されない）。10D の F12 は 5 seed 中 4 seed が keepair（droplet と判定されない）。
- SR@1e-10（2D / 5D / 10D、base 93.33 / 44.17 / 20.83%）: `v1_crmix`（`h2h_cr_mix=(0.1, 0.9)`）91.25 / 45.00 / 22.50（10D で base 有意勝ち 5）、`v1_crlow`（`h2h_CR=0.2`）79.17 / 32.50 / 20.83（5D F03 20 → 60% だが F12 / F13 100 → 10%）、`v1_h2hA` 92.50 / 43.33 / 20.00、`v1_pop32`（`pop_init_mult=32`）88.75 / 51.67 / 20.83（2D F23 85 → 15%）、`v1_gate2` — / 45.83 / 20.00。
- 読み: 低い CR は変数分離の関数には効くが悪条件の関数を壊す。成功率から F / CR を学ぶ方向は新しい既定値でも効かない。最適な初期集団の大きさは次元で違う。

### 2026-10-08 ルーターの判定の診断と、CR の継承（ローカル軽量検証）

**ルーターの判定**（新既定値、5D / 10D の 24 関数 × 5 seed、`analysis/single/local_v1y/routesig.json.gz`）: 経路を決める世代 120 は 10D で予算の約 17%。この時点の集団共分散の条件数（EMA）は F12 2.89 / F13 2.63 / F14 1.93 で閾値 3.0 に届かず、4/5 seed が keepair になる。予算 30% では 3.78 / 4.18 / 4.23（多峰の最大は F23 3.24、F05 3.00）、学習 C の条件数では 3.32 / 3.48 / 3.11（多峰の最大 F23 2.58、F18 2.57）で分かれる。変数分離の多峰関数（F03 / F04）の軸への揃い方は 0.7〜0.9 で非分離の関数と重なり、**共分散からは変数分離を見分けられない**（等方的な多峰関数では分離性が共分散に現れない）。
**腕**（2D 24 関数 × n=20、5D / 10D 13 関数 × n=10。集計は `analysis/single/local_v1y/`）、SR@1e-10（base 93.33 / 48.46 / 26.92%）:
- `v1_crher`（`h2h_cr_heritable`、jDE 方式の CR の自己適応。新規性はない）: 94.38 / 56.15 / 26.92。2D F17 +25、F18 +20、5D F03 20 → 60%、F22 50 → 90%、F15 0 → 20%。2D・5D で base の有意勝ちゼロ。10D は F03 で base が有意勝ち（`best_f`）。固定の低 CR と違い悪条件の関数を壊さない。
- `v1_rcf30`（`route_commit_frac=0.30`）: 93.96 / 48.46 / 26.92。判定は直るが SR はほぼ動かない（高次元は経路による配分の差が小さい）。
正準での確認は測定ジョブ。

### 2026-10-09 内部状態の記録からの診断（ローカル Windows、MC-ESO 既定、BBOB-24 × 2D/5D/10D × 20 run、`scripts/single/diag_trace.py`、`analysis/single/local_trace/runs.csv.gz`）

SR@1e-10 は 2D 94.0 / 5D 56.9 / 10D 41.9% で、同じ PC の quick と一致。
- **ルーターは 5D / 10D でほぼ KEEP-AIR に張り付く。** 2D では F08〜F14 の悪条件の関数の大半が DROPLET、F03 は 12/20 が CLOSE。10D では DROPLET が F02 / F10 / F11（各 20/20）と F12 8/20 だけで、F13 / F14 / F08 / F09 は 0/20。CLOSE は 5D・10D で一度も選ばれない。原因は 2026-08 の診断どおり（3 信号とも集団の瞬時共分散から推定しており 10D では推定できない）。**学習共分散の条件数（`cc_cond` 最大値の中央値）は 10D で悪条件の関数 F02 6.3 / F10 6.4 / F11 6.5 / F12 6.1 / F13 11.0 / F14 13.6、多峰の関数 F03 1.6 / F15 1.7 / F21 1.6 / F24 1.8 と分かれる**ので、ルーターを学習共分散から駆動する余地がある（未試験）。
- **空気感染は改善をほとんど生まない。** 1e-10 に初めて届くまでの最良値の改善（桁数）の内訳で、空気感染の取り分は 2D 0〜3% / 5D 0〜2% / 10D 0〜1%。評価の割合は 2D 6〜28% / 5D 5〜19% / 10D 3〜9%。ただし「別の谷へ移す」寄与はこの数え方では他の経路に計上される。10D の除去は未測定（2026-10-07 のジョブは `cc_air_ratio` を外し損ねた）。
- **10D F12 の失敗は学習共分散の育ちの遅さ。** 失敗 run では 2000 評価あたり 0.1 桁しか進まず、学習共分散の条件数は 1e4 → 1e5.4（9000 → 16000 評価）とゆっくり伸びる。スピルオーバー（3〜6 回）の前後で進み方は変わらない。CMA-ES は 8461 評価で 20/20 到達。
- **10D F13 の失敗 run は条件数 1e6〜1e7 に届いたまま 4.6e-4 で止まる**（スピルオーバー後も同じ値）。形ではなく歩幅の問題の可能性（未確認）。
- **5D / 10D の多峰（F03 / F04 / F15 / F17 / F18 / F20）は 0〜10%**。スピルオーバー（F17 / F18 / F20 で 7〜14 回）か盆地乗換え（F03 / F04 / F15 で約 10 回）を繰り返しても届かず、drilling に入るのは予算の 2/3 以降。同じ run で L-SHADE 系・EBOwithCMAR は 60〜100%。

### 2026-10-09 診断から作った腕 5 本（ローカル Windows、quick n=20、BBOB-24、`results/20261009_14*_arms1009_d{2,5,10}_quick/`、既定不変）

SR@1e-10 の MC-ESO（2D 94.0 / 5D 56.9 / 10D 41.9%）との差。
| 腕 | 中身 | 2D | 5D | 10D | 判定 |
|---|---|---|---|---|---|
| `v1_noairHD` | 3 次元以上で空気感染なし（`cc_air_ratio=0`） | ±0（全関数一致） | **+2.5**（F16 25→65、F15 5→25、F03 / F17 +10 / F21 50→35） | **+1.0**（F08 / F09 90→100、F21 5→15 / F07 90→80） | 候補（下の組合せへ） |
| `v1_rtC` | 3 次元以上でルーターを学習共分散で駆動（条件数 ≥ 1e3 で DROPLET に固定） | ±0 | +1.0（F16 / F18 +15） | ±0（全関数一致） | 10D は経路を変えても動かない |
| `v1_cpath` | 学習共分散に進化パス（rank-1、CMA-ES の定数）を足す。前提（受理ステップ数）が変わったので再試験 | ±0 | +1.6（多峰 F16 / F03 / F20 / F23 +10〜20 / F08 −10） | −0.2（F12 は 0% のまま） | F12 の律速は解消しない |
| `v1_h2h50HD` | 3 次元以上の飛沫感染を 0.25 → 0.5 | ±0 | −2.5（F08 100→60） | **−14.6**（F08 / F09 90→0、F06 100→40） | 不採用 |
| `v1_popfin8` | 最終集団 4·D → 8·D | **−1.3**（F18 95→70） | +3.1（F16 25→85、F03 10→40） | **−6.9**（F08 90→10、F09 90→15） | 不採用（2D を下げ、10D の谷を壊す）。5D の多峰には効くので、谷と多峰を見分けられれば使える |
読み: 5D の多峰は「探索を広く保つ」方向（大きい集団・空気感染なし・進化パス）で伸びるが、Gallagher（F21 / F22）はどの腕でも下がり、谷（F08 / F09）は大きい集団と飛沫感染の増量で壊れる。10D の F12 / F13 はどの腕でも 0% のまま。

### 2026-10-09 組合せ 3 本と 10D F12 / F13 の個別プローブ（同じ条件、`results/20261009_14*_arms1009b_d{2,5,10}_quick/`）

| 腕 | 2D | 5D | 10D | Wilcoxon（腕の有意勝ち / 負け、5D・10D 合算） |
|---|---|---|---|---|
| `v1_noair_rtC`（空気感染なし ＋ 学習共分散ルーター） | ±0（全関数一致） | **+3.1**（F16 25→60、F15 5→25、F17 0→15、F18 0→20、F03 +10 / F21 50→35、F22 −5、F09 −5、F23 −5） | +0.8（F08 90→100、F09 90→95、F21 5→15、F22 +5 / F07 90→80） | 3 / 1（5D F16・F17、10D F08） |
| `v1_noair_cpath` | ±0 | +0.6（多峰 +5〜20 / F22 70→45、F09 −15、F08 −10） | +0.4 | — |
| `v1_noair_all`（3 つ全部） | ±0 | +0.6（F22 −25、F09 −15） | +0.6 | — |

- 参考: 単独の `v1_noairHD` は同じ run で 5D 59.4（+2.5）/ 10D 42.9（+1.0）、Wilcoxon 4 / 1（5D F16・F17、10D F08 p=0.022・F09 p=0.0499）。5D の成功時平均評価回数は MC-ESO 4094 に対し noairHD 4240 / noair_rtC 4260（新たに解けた難関数の分が乗る）。
- **進化パスは空気感染なしと重ねると打ち消し合う**（谷 F08 / F09 と Gallagher F22 を壊す）。単独でも組合せでも不採用。
- **採用候補は `v1_noairHD` と `v1_noair_rtC`**。どちらも 2D は bit 一致で、5D / 10D の SR@1e-10 を下げない。既定変更はユーザー承認と正準環境（クラウド）での再測定待ち。共通の弱点は 5D F21 と 10D F07（いずれも −10〜15）。
- **10D F12 / F13（各 6 seed、25000 評価、既定に 1 つだけ足す）**: どの設定でも 1e-10 に届かない。F12 の中央値ギャップは既定 3.5e-2 → `kill_fraction=0.5` で 2.5e-4（世代数は半分）、`pop_init_mult=8` で 1.8e-2。F13 は既定 1.6e-7 → `pop_init_mult=8` で 3.6e-10。`cc_persist_frac=0.9`・`cc_learning_rate=0.15`・固定小集団は悪化。次の手は「淘汰を強める」「初期集団を小さくして世代を稼ぐ」の 2 方向（未測定、quick での確認が必要）。

### 2026-10-09 空気感染なしの上に載せた 6 腕（ローカル、quick n=20、BBOB-24、`results/20261009_19*_arms1009c_d{2,5,10}_quick/`）— すべて不採用

いずれも `cc_air_ratio=0` の上に 1 つ足したもの。2D は 8 手法すべて全関数一致（3 次元以上でだけ効く引数）。SR@1e-10 は `v1_noairHD`（5D 59.4 / 10D 42.9%）との差。
| 腕 | 中身 | 5D | 10D | Wilcoxon（MC-ESO 側の勝ち/負け 5D・10D） | 関数別 |
|---|---|---|---|---|---|
| `v1_k50` | 3 次元以上で `kill_fraction` 0.25 → 0.5（`hd_kill_fraction`） | −8.6 | −10.6 | 6/1・11/1 | 10D F08 / F09 100→0、F14 100→60、F06 100→70 / F07 80→100。5D F06 100→40、F08 100→50 |
| `v1_k35` | 同 0.35 | −1.5 | −6.4 | 3/1・4/2 | 10D F08 100→25、F09 100→20 / F07 +10。5D F20 5→20 |
| `v1_pi8` | 3 次元以上で初期集団 16·D → 8·D（`hd_pop_init_mult`） | −4.2 | −4.4 | 3/0・2/2 | 10D F07 80→25、F08 −20 / 成功時評価 11859 → 9332。5D F16 65→25、F23 5→20 |
| `v1_k50_pi8` | 両方 | −5.0 | −7.9 | 5/1・10/0 | 10D F08 / F09 → 25 / 20 |
| `v1_cr` | 3 次元以上で CR の自己適応（jDE 方式、`hd_cr_heritable`） | −1.5 | −2.7 | 2/2・1/0 | 5D F03 20→35 / F15 25→5、F22 65→50、F17 10→0。10D F07 80→55、F21 15→0 |
| `v1_crR` | 同、ルーターを学習共分散にして DROPLET 固定後は CR 0.9 | −1.9 | −2.7 | 2/3・1/0 | ほぼ `v1_cr` と同じ。10D F07 の悪化は防げない（F07 の学習共分散は条件数 10^2.6 で 10^3 の固定に届かない） |
- **淘汰を強めると、10D の谷（F08 / F09）が壊れる。** 6 seed の F12 プローブで見えた中央値ギャップの改善（3.5e-2 → 2.5e-4）は、SR@1e-10 には一度も現れなかった（F12 は全腕 0%）。
- **初期集団を小さくすると易しい関数は速くなる**（10D の成功時平均 −21%）が、F07 と多峰で SR を落とす。旧既定より 2.5 倍遅い件は、集団縮小とのトレードオフとして残る。
- **CR の自己適応は 5D F03 にだけ効き、Rastrigin（回転）・Gallagher・F07 を落とす**（正準環境の 2026-10-08 ジョブ 1 と同じ傾向）。経路で守る案は、F07 が DROPLET に固定されないので機能しなかった。
- professor（2026-10-09）が挙げた「空気感染を切るかどうかをルーターに決めさせる」案は、既存の診断データ（`analysis/single/local_trace/`）に F21（空気感染なしで落ちる）と F03 / F15（空気感染なしで上がる）を分ける信号が見つからない。どちらも盆地乗換え約 10 回、学習共分散の条件数約 10^1.6。このため実装していない。
