# 開発履歴 — 試した工夫・フラグ・ablation 記録

MC-ESO の開発で試した機構・フラグの履歴。**何を試し、採用 / 不採用となり、なぜか**を残す。手法の最新アーキテクチャは [mceso.md](mceso.md)、評価方法論は [experiments.md](experiments.md) を参照。

> **判定の前提**: 改良案は `quick_check.py` で全関数 ablation し、SR / 評価回数 / Wilcoxon の 3 指標で overall 改善を確認できたものだけを `main.py` の `_BASE_OPTIMIZERS` に統合する。**判定は quick n=20 / max_evals=5000 / `--all` で統一**する（GitHub Actions の n=100 ワークフローは裏で補助的に回すもので、評価には参照しない）。
>
> **注**: 以下の検証ログの多くは quick デフォルトが n=10 だった時期に取得したもので、記載の数値は当時の n=10 結果。現在の評価標準は n=20。再検証する場合は n=20 で取り直す。

---

## 検証フロー

`quick_check.py` の `_OPTIMIZERS` は **MC-ESO 本体 ＋ 7 ベースライン（CMA-ES / IPOP / BIPOP / PSO / DE / L-SHADE / SaVOA）の 8 手法**を並べた標準比較セット。改良案や診断 variant を検証するときだけ、`_OPTIMIZERS` に一時的に MC-ESO のサブクラス／別 entry を追加して quick で overall 改善を測り、確認できれば本体に統合してから一時 entry を外す。**診断・ablation variant は常設しない**（`core/optimizers/mceso_ablations.py` に実体があり、必要時に追加する）。

```bash
./run.sh quick --all       # BBOB 24 + Custom 11 で評価（手法評価は必ず全関数）
./run.sh quick --funcs F08-Rosenbrock,F09-RosenbrockRot,F10-EllipsoidalRot,F12-BentCigar --max-evals 10000
```

`--funcs` 引数で任意関数に絞り込んだ集中検証ができる。

---

## 前身手法の廃止（2026-04）

MC-ESO に先立つウイルス模倣手法 **VirusOptimizerV2 (VSO V2)** は全廃止し、コードベースから完全削除した（2026-04-24）。

- **不採用の理由**: σ 制御の根本問題（`sigma_up × sigma_down` の積と 50% 成功率のバランス）を解決できず、σ が収束しないまま設計だけが複雑化した。ユーザー判断で全廃止。
- **教訓**: 後継の MC-ESO は σ 適応を「常時 ON の乗法適応＋ drilling mode＋σ フロア（`sigma_floor_ratio`）」で明確化し、ゲートやフェーズ切替パラメータを増やさない方針を継承した。VSO V2 に関する実装・参照は不要。

---

## ベースに統合された機構（MC-ESO 本体に常時 ON）

以下はすべて MC-ESO 本体に常時 ON で組込まれ、ablation で overall 改善を実証済み。

| 機構 | MC-ESO での位置付け |
|---|---|
| **飛沫感染チャネル**（h2h, DE/current-to-best/1） | 差分変異が集団形状から異方情報を獲得。F08/F09/F10/F12 の主因 |
| **h2h binomial crossover** (`h2h_CR=0.9`, DE/bin 標準値) | 飛沫の trial vector を親と座標毎に交叉し、separable 多峰の座標方向情報を保護。初期は F04/F17 用に 0.7 へ調整していたが、後続 ablation 後の hold-out 検証で標準値 0.9 のほうが overall で優ると判明し復帰 |
| **宿主競合**（μ+λ greedy + rollback） | 最良宿主の長期保持で F10/F12 の SR を改善 |
| **スピルオーバー＋basin switch** | quality-gated restart、連続失敗 2 回で best 破棄＋σ_init リセット。ill-cond の整列失敗と F24 双漏斗を救済 |
| **情報化リスタート** (`ir_archive_frac=0.5`, 2026-06 統合) | spillover 再播種を盲目 Uniform から**リザーバ再着火＋basin 忌避**へ。診断 ablation で旧 restart が探索構造を捨てていたと判明し情報化。dim2 +1.7pt、dim3/CEC2022 hold-out で有意 regression なし。IPOP 盲目 restart との差別化点 |
| **Drilling mode**（`sigma_drill_down=0.85`） | σ < span × 1e-3 で σ 縮小を強化し浮動小数限界まで追込む |
| **接触感染の経験共分散** (`empirical_cov_floor=0.01`) | 集団経験共分散 `C_pop` の固有分解で接触感染ノイズを瞬間異方化。CMA-ES の rank-μ 学習と異なり履歴累積なし、basin 切替に即応 |
| **適応異方性 floor** (`cov_floor_low=1e-3`, 2026-06) | 異方性の頭打ち（floor）を**集団共分散の素の固有値比**で自動調整。悪条件の谷（比 1e5–1e7）では floor を下げ異方性を解放、rugged/多峰（比 3–600）では高く保ちノイズの偽異方性をクランプ。スケール・シフト不変。全 35 関数で固定 0.01 比 **+2.6pt（85.4→88.0）・回帰ゼロ**、固定 1e-3（F17/C11/C05 で回帰）をも上回る。F02/F10→100%、F18 60→80% 等 ill-cond を改善しつつ rugged を保護 |
| **次元適応 n_pop** (`n_pop=max(20, 4·dim)`, 2026-06) | 固定 20 は高次元で過小。dim=10 で niching restart が CEC2022 G06-Hybrid1 を彷徨 best_f 2140→40 once n_pop=40。BBOB dim2/3 は 20 で無変更 |
| **Drilling 中の空気感染停止** | `σ < span × precision_sigma_ratio` で `air_ratio_eff = 0`。drilling 中の広域ランダム雑音を排除し精度劣化を防止 |
| **逐次 niching（多解探索）** (`_basin_exhausted`, 2026-06 統合) | σ-exhaustion 検知（スケール・f_opt 非依存）で掘り切った basin から restart。SR 無犠牲で多解 PR を改善（下記）|

---

## 検証され不採用となった variant

`quick_check.py` で MC-ESO 統合候補として走らせたが overall 改善を実証できず、コードからも削除されたもの。

| variant | 追加した挙動 | 不採用の理由 |
|---|---|---|
| MC-ESO-A1（per-dim σ close-contact） | 接触感染ノイズを集団 per-dim std で軸別スケール（軸整列の異方化） | F08/F17 では改善するが F14-DiffPowers の BBOB 回転と整合せず致命的劣化。後継の A2（経験共分散版）に置換 |
| MC-ESO-ABD | h2h_CR=0.9 ＋ σ-adapt 停滞ゲートを drilling 中バイパス ＋ 初回 spillover で座標軸 sweep | A_mild ベースと比べて CR=0.9 のため F18/F19 で勝つが F04 で回帰。Wilcoxon でも B/D 単独の有意寄与なし、結局 CR トレードオフに収束 |
| MC-ESO-A_mild_BD | 統合済み MC-ESO ＋ 同上の B/D | F09/F11/F18 での改善と F04/F14 での悪化が相殺しほぼ同等。B/D の overall 寄与なし |
| 旧 A〜N（MC-ESO 初期開発の 8 案）<br>A=`use_evolution_path` / C=`use_pop_covariance` / D=`use_lifespan_reset` / E=`use_adaptive_air` / G=`use_adaptive_h2h_F` / I=`use_aggressive_niche` / K=`use_h2h_archive` / N=`use_local_pair_h2h` | 進化パス・集団共分散・寿命リセット・適応 air・適応 h2h_F・強ニッチ・h2h アーカイブ・局所ペア h2h の各フラグ | 各案とも単一関数の改善はあるものの 12 関数 SR 合計で baseline 以下、あるいは安全装置を要する構造欠陥（E=adaptive_air）で overall を毀損し全削除。※ 後継として C 系の発想は「経験共分散の接触感染」、I 系は「逐次 niching」として別実装で結実 |
| **MC-ESO-V2a** (UCB-AOS on 3 channels, 2026-06) | 接触・飛沫・空気の比率を世代毎に UCB ベース AOS で自動調整（credit = 世代内中央値で正規化した Δf）。drilling 中の air 抑制は V1 から踏襲 | BBOB+Custom 35 関数の主指標 SR@1e-10 が V1 24.40 → V2a 22.10（−2.30）。Wilcoxon (n=10, α=0.05) で有意な勝ち 1（F17）に対し有意な負け 3（F06 / F20 / C07）。F23-Katsuura では 0%→70% の劇的改善が出たが overall regression を覆せず削除 |
| **MC-ESO-V2b** (V2a + 4 新チャネル, 2026-06) | V2a に Lévy 超拡散 / 重心組換 (μ-recombination) / 系統間クロスオーバ / 反対称跳躍 (`2·centroid − x_p`) を追加し、UCB の arms を 3→7 に拡張 | SR@1e-10 が V1 24.40 → V2b 19.80（−4.60）。Wilcoxon で有意な勝ち 0、有意な負け 5（F02 / F11 / F18 / C06 / C11）。Lévy 等の大ジャンプが ill-conditioned 関数の precision grinding を妨害。V2a より明確に劣り削除 |
| **MC-ESO-SC / IRSC**（系統共存の実活性化, 2026-06） | 永続アーカイブ（品質ゲート `sc_quality_band`）から飛沫 donor を抽選し多 basin 引力を常時化 | SC 単独は dim2 net-neutral（−0.3pt）。IRSC は dim2 raw 最良（+2.3pt）だが F17/F24 有意回帰を持込み、決定打は**汎化失敗**: dim3 で有意回帰 3（F08 Rosenbrock−30 含）、CEC2022 dim10 で medf 2624 vs 1650 と大崩れ。「系統共存（epidemic 固有の宣伝機構）を活性化しても性能に結びつかない」が全次元・hold-out で確定（novelty gap）。コード削除 |

検証ログ: `results/20260515_150803_ベースライン_quick/`、V2 系は `results/20260605_190353_v2_compare_all_quick/`、SC/IRSC は `results/20260610_*_irsc_verify_quick/` 他。

---

## 撤回した実装ステップ・設計上の失敗（記録）

採用済み機構の開発途中で試して**撤回した中間実装**。同じ轍を踏まないための記録（full variant ではなく、機構内のアプローチの失敗）。

| 機構 | 試して撤回したアプローチ | なぜ失敗 / 何が正解だったか |
|---|---|---|
| **適応異方性 floor** | ill-cond vs rugged の判別を**支配的固有ベクトルの世代間 alignment**（向きの安定性）で行おうとした | 収束後は rugged でも固有ベクトルは安定（alignment≈0.99）で**判別不能**。正解は固有値「比」λmax/λmin の**大きさ**（ill-cond ≈1e5–1e7 と rugged ≈3–600 で桁違いに分離）。向きでなくスケールが信号だった |
| **逐次 niching（Endemic）①** | basin 掘り切り判定を持たず **always basin-switch**（停滞ごとに毎回乗換え） | 掘りかけの best basin を破壊し C10(−40)/F14(−20) で SR 低下。→「掘り切ったあとだけ乗換える」2 レジーム化が必須と判明 |
| **逐次 niching（Endemic）②** | 掘り切り検知に**絶対 floor `f ≤ 1e-11`** を使用 | 「最適値 0」という BBOB 正規化依存で一般関数に通用せず**ユーザ却下**。→ σ がフロア到達（span 相対、f 値非依存）に置換 |
| **逐次 niching（Endemic）③** | σ 検知の停滞許容 `exhausted_no_improve_mult=1` | F14(DiffPowers) の平坦 basin で base の遅延 breakthrough を取りこぼし(−20)。→ `mult=3` に上げて粘りマージン確保 |
| **スピルオーバー（旧仕様）** | 再播種を**盲目 Uniform** で escalate（streak 0→75% Uniform / 1→100% Uniform / 2→basin switch） | 探索構造を毎回捨てていた（診断 ablation で判明）。→ 情報化リスタート（リザーバ再着火＋basin 忌避）に置換、`ir_*` パラメータ化 |
| **σ 適応（旧仕様）** | `no_improve ≥ 100` で `× 0.99`（spillover に備えて σ を温存する停滞ゲート）＋ `sigma_decay` | HP（`sigma_adapt_stagnation_gate` / `sigma_decay`）を増やす割に overall 寄与が立たず、always-on 乗法適応＋ drilling mode に統一して**削除（HP 削減）** |

---

## 簡潔化監査 — 各機構の単独 ablation（2026-06）

蓄積したオプション/フラグが「本当に性能に寄与しているか」を**一機構ずつ OFF にして全 35 関数で再確認**し、寄与ゼロのものを削除して手法を簡潔化した。各機構を `quick_check.py` に単独 OFF 版（kwargs / サブクラス）として登録し、3 指標（SR@1e-10 主、evals_succ、Wilcoxon）で base と比較。

**削除した機構（寄与ゼロを実証 → コード削除）**

| 機構 | OFF 版 | 結果 → 削除理由 |
|---|---|---|
| **軸 sweep**（spillover 前の境界 probe, dim×2 評価） | `MCESONoAxisSweep`（`_axis_sweep`→`[]`） | 35 関数・全精度階層で SR が **1 run も動かず**（完全一致）、多解 PR も不変、evals は僅減。設計対象の **F05 (100%=100%) / F04 (90%=90%) すら影響ゼロ**＝base は境界 snap 経由で既に到達しており完全冗長。`_axis_sweep` と `_maybe_spillover` 内 sweep ループを削除 |
| **適応 floor の median 平滑化枝**（`cov_ratio_window`） | `MC-ESO-med15`（既定 0＝OFF） | 既定で常に OFF の実験足場。EMA 枝のみで稼働しており、`cov_ratio_window` / `cc_logratio_win` state / median 分岐を削除 |

**寄与を再確認し維持した機構（OFF で overall SR@1e-10 が低下）**。base 87.7%（n=10, dim2, 5000 evals）に対し:

| 機構 | OFF 版 | SR@1e-10 | evals（OFF時） | Wilcoxon 有意悪化 | 判定 |
|---|---|---|---|---|---|
| **per-host σ スケール** (`host_sigma_min_scale`) | `=1.0` | 84.3%（−3.4） | +3350（遅化） | 3（F06/F17/F23） | 維持（最大寄与） |
| **空気感染チャネル** (`air_ratio`) | `=0` | 85.1%（−2.6） | +3047 | 4（F04/F17/C05/C11） | 維持 |
| **境界 snap**（`_reflect` の snap-to-bound） | `MCESONoBoundarySnap` | 85.1%（−2.6） | +1481 | 1（F05） | 維持（**F05 100→20**、軸 sweep 削除後の唯一の境界機構） |
| **streak basin-switch** (`basin_switch_after_failed_spillovers`) | `=∞` | 86.0%（−1.7） | **−2607（速化）** | 0 | 維持（弱）→ **n=20 で再精査**：寄与は F04（90→50）/F20 にほぼ集中、有意差なし・除去で速い |
| **収束適応の空気 σ** (`air_sigma_amplifier`) | `=0`（固定 1.5×） | 86.0%（−1.7） | **−1876（速化）** | 1（F17） | 維持（弱）→ **n=20 で再精査**：F17(−50)/C11/F24 を救う一方 F23(+30)/F13 を損なうトレードオフ、除去で速い |

- 削除した 2 機構は寄与が「定義上ゼロ／既定 OFF」なので確定削除。弱い維持機構（streak basin-switch・air_sigma_amplifier）は次回 n=20 評価で F04/F17 の寄与が崩れれば追加削除候補。
- 検証ログ: `results/20260630_140730_simplify_DEFG_quick/`（軸 sweep・空気・per-host σ・basin-switch）、`results/20260630_145732_simplify_HI_quick/`（境界 snap・空気 σ 増幅）。

---

## 診断 ablation（チャネル vs リスタートの寄与分解, 2026-06）

「性能はチャネル/系統共存でなく頻繁なランダムリスタート由来では」という疑義を検証するため、`mceso_ablations.py` に 2 つの**診断用** variant を用意（改善候補ではなく寄与の切り分け用。標準の quick 比較には含めず、検証時のみ `_OPTIMIZERS` に一時追加する）。

- **MC-ESO-NoSpill** — チャネル ON / spillover 完全停止（`_maybe_spillover` が常に False）。チャネル単独の到達力を測る。
- **MC-ESO-RandRestart** — spillover・σ適応・drilling・μ+λ greedy は維持し、3 チャネル＋系統共存を**等方ガウス局所探索 1 本**（`x_parent + σ_global·N(0,I)`）に置換。リスタート＋バニラ局所だけで何処まで行くかを測る。

**結果（BBOB24+Custom11, n=10, max_evals=5000, dim=2、平均 SR@1e-10）**: MC-ESO **83.7%** / NoSpill **68.6%** / RandRestart **48.9%**。

- **主動力はチャネル機構**: RandRestart で 83.7→48.9% に激減（Rosenbrock/ill-cond/F02 は 100→0%）。Wilcoxon で MC-ESO が **26/35 関数で有意に優位（負け 0、全 large）**。「リスタートのくじ運で発見」説は棄却。
- **spillover は二次的・限定的**: NoSpill でも 68.6% を維持。MC-ESO が NoSpill に有意優位なのは **7/35（F03/F04/F15/F20/F24/C05/C11 ＝ 多峰・deceptive）**。spillover 発火回数も大半の関数で 0〜1 回（F20=5.6, F24=11.2 のみ「頻繁」）。
- **ただし系統共存は不活性**: 平均 n_elite は大半の関数で ~1.0–1.2（n_elite=1 の世代が 92–99%）。多 basin 保持は F20(1.56)/F24(3.85) でしか発火せず、宣伝機構が 30/35 関数で no-op = **novelty gap**（性能の出所が DE×経験共分散＋IPOP 風 restart で、epidemic 固有の新規性と不一致）。改善は「系統共存の実活性化（永続アーカイブ / crowding）＋ restart の情報化（basin 忌避）」に的を絞った。

検証ログ: `results/20260605_200551_diag_restart_ablation_quick/`

---

## 情報化リスタートの統合 / 系統共存活性化の不採用（2026-06）

診断 ablation を起点に 2 方向の改善を検証し、片方を本体統合・片方を不採用とした。検証はすべて `_on_spillover_start` / `_diversified_reseed` / `_droplet_strain_positions` の拡張フック（既定で RNG 順不変）経由でサブクラス化し quick で測定。

**① 情報化リスタート（IR）→ 本体統合（採用）**
- 動機: 診断で「リスタートは実寄与あるが**無情報**（best 以外を全域 Uniform 再播種）」と判明。**リザーバ再着火**（spillover 時に niched elite を永続アーカイブへ harvest し一部スロットを系統まわりで再生成, `ir_archive_frac`/`ir_reignite_sigma_ratio`）＋**集団免疫忌避**（放棄 basin 重心を記憶し残り Uniform を斥力 rejection, `ir_repel_radius_ratio`/`ir_repel_max_tries`）で情報化。
- 結果（平均 SR@1e-10）: dim2 83.7→**85.4（+1.7pt）**（改善 C09+40/F23+20/F20+10/C11+10、悪化 F10/F19 各−10）、dim3 flat・有意差 0、CEC2022 dim10 hold-out は medf≈同点で composition 系に有意 best_f 改善 5・回帰 0。**全次元・hold-out で有意 regression なし**を確認し本体に統合（`MultiChannelEpidemicOptimizer` 既定挙動）。診断 `MC-ESO-RandRestart` は旧盲目 Uniform restart を pin して比較基準を維持。

**② 系統共存の実活性化（SC）/ IR+SC 併用（IRSC）→ 不採用**（上の variant 表参照）。**「系統共存を活性化しても性能に結びつかない」が全次元・hold-out で確定**（novelty gap）。SC/IRSC 関連コード（`mceso_sc.py`/`mceso_combo.py`）は削除。

検証ログ: `results/20260610_103749_ir_verify_quick/`、`results/20260610_113659_sc_verify_quick/`、`results/20260612_*_{irsc_drill,gen_dim3,gen_cec}_quick/`。

---

## peak-ratio 診断 — 「多解並行探索」は当初実体がなかった（2026-06-16）

「MC-ESO の強み＝ウイルス模倣による多解並行探索が実際は機能していない気がする」というユーザの疑義を検証するため、**最適化器を一切変えずに** peak-ratio 系メトリクスを後付け計算で追加（`core/runner.py:optima_found_mask` / `peak_metrics`、最近傍割当で二重カウント防止・多段 tol・MMO 成功率）。SR は定義上不変なので peak ratio 単独で評価できる。

**実測（quick n=10, max_evals=5000, dim2, C01/C02/C03）でユーザの直感が裏づけられた**:
- MC-ESO は SR@1e-10=100%（1 つは確実に到達）だが PR@1e-4 は C01 Himmelblau **0.28**（≈1.1/4）・C02 Six-hump 0.60・C03 Shubert **0.06**（≈1/18）。全解発見（MMOsr）はほぼ 0%。
- **系統共存が宣伝倒れ**: リスタートで別 basin に飛ぶ IPOP/BIPOP-CMA-ES の方が多解を拾う（Himmelblau **BIPOP 0.78 vs MC-ESO 0.28**）。診断 ablation の live n_elite≈1（novelty gap）が peak ratio にそのまま表面化。
- spillover は多解発見に無効（MC-ESO ≈ MC-ESO-NoSpill）。
- 「並行的な多解探索」は当時コンセプトのみで実装が伴っていないことを初めて定量化。これが下記の逐次 niching 統合の動機。

検証ログ: `results/20260616_141021_peakratio_baseline_quick/`（baseline 全手法）。

---

## 逐次 niching（多解探索）の統合（2026-06）

上記 peak-ratio 診断で「ウイルス模倣で複数最適解を探索する」という当初の主張が**実体を持っていない**（MC-ESO は SR@1e-10=100% でも PR@1e-4 が Himmelblau 0.28 / Shubert 0.06）と確定したのを受けた逐次 niching。**base `MultiChannelEpidemicOptimizer` 本体に統合**（`_basin_exhausted` で「掘り切った」を検知して restart）。`mceso_niching.py:MCESOEndemic` は後方互換エイリアス。設計と σ-exhaustion 検知の詳細は [mceso.md の多解探索](mceso.md#多解探索逐次-niching-base-統合済み)を参照。

**設計の要点**: 1 集団で複数 basin を同時に深精度化すると **SR@1e-10 が崩壊**するため、crowding/per-host σ を撤回。BIPOP-CMA-ES 流の「1 basin を掘る→記憶→斥力で離れて restart→次の basin」を採り、各 basin は単一σ drilling をそのまま使う。SR 死守の鍵は **2 レジーム化**（掘り切る前は base と完全同一挙動、掘り切った後に初めて多解探索を起動）。「掘り切った」検知は σ フロア到達＋停滞で f 値非依存（初期版の絶対 floor 1e-11 は BBOB の「optimum=0」依存だったため σ ベースに撤回）。

**結果（quick n=10, max_evals=5000, dim2, 全 35 関数, MC-ESO → niching 統合版）**:
- **SR@1e-10: 35 関数すべてで回帰ゼロ。平均 85.4% → 86.3%（F02/F11/F19 で +10、exhausted basin からの restart が失敗 run に再挑戦の機会を与える）**。
- 多解（多大域）関数の改善:

| 関数 | K | SR@1e-10 | PR@1e-2 | PR@1e-4 | MMOsr@1e-4 |
|---|---|---|---|---|---|
| C01 Himmelblau | 4 | 100% → **100%** | 0.28 → **0.62** | 0.28 → **0.53** | 0% → 0% |
| C02 Six-hump | 2 | 100% → **100%** | 0.75 → **1.00** | 0.60 → **0.95** | 20% → **90%** |
| C03 Shubert | 18 | 100% → **100%** | 0.06 → **0.17** | 0.06 → **0.16** | 0%（18 global が ~760 local に埋もれ hard だが約 3 倍, BIPOP 0.11 超）|

検証ログ: `results/20260616_141021_peakratio_baseline_quick/`（baseline 全手法）、`results/20260624_113046_endemic_sigexh_quick/`（**σ-exhaustion 確定版・全 35 関数 SR 回帰ゼロ**）。撤回した試作: crowding=`20260616_145202_endemic_v3`、SR を落とした always-restart=`20260623_145326_1ec0bf0`、絶対 floor 版=`20260624_103753_endemic_secured`。

---

## 次元・hold-out への汎化（2026-06）

それまでの改善（適応 floor / niching）は全て **BBOB dim=2 で調整**していたため、「全く別の観点」として次元スケーリング / hold-out 汎化を検証した（過去 SC/IRSC は dim3/CEC2022 の汎化失敗で却下されており、轍を踏まないため）。比較は新 base（floor+niching+dim-pop 全 ON）vs 旧 base（全無効）を `MC-ESO-orig` として一時登録。

**汎化検証結果**（SR@1e-10、特記なき限り n=10）:
- **BBOB dim=2**: 85.4 → **88.0%**（検証・merge 済）
- **BBOB dim=3**: 40.8 → **45.4%**（+4.6pt、7 改善 F02+30/F16+30/F14+20 等、F19 −10 のみ）
- **CEC2022 dim=10 hold-out**（10000 evals）: median best_f 24.3 → **17.9**（大勝 G01 34×/G06 10×/G11 100×、小負け G05/G03/G08）

→ **改善は全次元・hold-out に汎化**（SC/IRSC の轍を回避）。

**次元適応 n_pop の発見**: MC-ESO は n_pop=20 を全次元固定していたが高次元で過小。dim=10 で n_pop sweep の結果 **40 が sweet spot（80 は悪化）**、特に **G06-Hybrid1 を best_f 2140→40 に解消**（niching restart が小集団で彷徨っていた）。`n_pop=None` デフォルトで **`max(20, 4·dim)`**（dim≤5 は 20 で BBOB dim2/3 無変更＝回帰リスクゼロ、dim=10 で 40）。残課題: G05-Levy 等 dim10 多峰での niching 小回帰（n=20 で再確認する）。

検証ログ: `results/20260*_cec_holdout_check_quick/`（dim10）、`*_dim3_check_quick/`（dim3）。
