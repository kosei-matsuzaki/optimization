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
| **per-landscape チャネルルーター** (`channel_schedule=True`, 2026-07 統合) | 各 run で集団共分散の 3 シグナル（`cond`=固有値比 / `algA`=軸整列 / `mgap`=座標間隙, いずれも EMA・f 非依存）から landscape を検知し、空気感染(air)予算を droplet/close/keep-air の 1 ルートへ振り分け（gen120 commit ＋ 早期 droplet latch, run 内固定）。一律再配分（4 案全 REJECT）と異なりデフォルト=keep-air=base で無検知関数は無傷。**BBOB dim2 +0.6pt（87.9→88.4, F04/F11/F12/F13/F14/F16 +5）・dim3 +0.6pt（回帰なし）・CEC2022 dim10 hold-out で best_f 改善（G06 364→202）**＝全次元・hold-out 汎化。詳細は下記「チャネル割合スケジューリングの探索」|

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

---

## チャネル割合スケジューリングの探索（2026-07）— 一律再配分は全 REJECT、目標は per-landscape ルーティング

「感染チャネルの割合をスケジューラで制御して配分を改善する」という方向を、`MultiChannelEpidemicOptimizer(channel_schedule=...)` フラグ（既定 False＝base と bit-identical）＋一時エントリ `MC-ESO-Sched` で検証した。**割合を一律にいじる 4 案はすべて overall SR@1e-10 を落として不採用**（quick n=20 / 5000 / 全 35 関数, dim2, base 87.9%）:

| 案 | 変更（channel_schedule=True の中身） | SR@1e-10 | 判定 |
|---|---|---|---|
| 1 | air を σ の log で `air_ratio`→0 に連続ランプ、空き枠→**close** | 87.0（−0.9）| REJECT |
| 2 | 同上ランプ、空き枠→**droplet** | 87.7（−0.1）| REJECT |
| Phase1 | air を conditioning `c`（cov 固有値比 EMA）でゲート `air_ratio·(1−c)`、空き枠→close | 86.6（−1.3）| REJECT |
| Phase2 | air 不変、**droplet↔close の境界のみ** `h2h_ratio·(1+0.5·c)` で再配分 | 86.7（−1.1）| REJECT |

**診断（4 案から確定）**:
1. **close-contact は SR@1e-10（深精度 1e-10 到達）の load-bearing チャネル**。per-host σ_i 微細化＋回転対応で最後の drilling を担うため、**どこであれ close の枠を削ると深精度 run が数本落ちる**。
2. **conditioning `c` は "ill-cond のときだけ" ではなく多くの関数の収束時に広く立ち上がる**（集団が basin 方向に伸びる）。ゆえに `c` 起動の再配分は close を広範に削り SR@1e-10 を毀損する。適応異方性 floor（穏やかな log 補間・EMA が transient 吸収）には効くが、**チャネル割合の on/off トリガーには粗すぎる**。
3. **base 比（air 0.3 / droplet 0.4 / close 0.3 ＋ drilling cliff）は既に SR@1e-10 の well-tuned な局所最適**（簡潔化監査で各チャネルの寄与も確認済み）。摂動＝悪化。V2a（チャネル比 UCB-AOS）却下と同根で、**「割合を一律にいじる」軸自体が overall では頭打ち**。

**ただし per-function には明確な最適配分が存在する**（best_f 併記で確認、[[feedback_per_function_bestf]]）。air の予算を「どこへ回すのが最適か」が関数タイプで割れる:

| air 予算の最適な行き先 | 関数（代表） | landscape 種別 |
|---|---|---|
| **air のまま維持**（air を減らすと悪化） | F10-EllipsoidalRot, F15-RastriginRot, C05, F06, F17, F19, F20, F24 | 回転 ill-cond（滑らか）＋ 多峰/deceptive（escape 要）|
| **droplet 中心** | F11-Discus, F12-BentCigar, **F13-SharpRidge（+20〜35）**, F14-DiffPowers | ill-cond の谷/ridge（DE 差分が谷を辿る）|
| **close 中心** | F04-BucheRastrigin, F16-Weierstrass, C09-Easom | separable / rugged（軸整列の精密化）|

**目標（今後の方針）= landscape 検知による air 予算の per-landscape ルーティング**。一律配分ではなく「F10/F15 は air 維持・F11–F14 は droplet・F04/F16 は close」と関数ごとに正しく振り分ける検知機構を作れれば、トレードオフ（F13+droplet で伸ばしつつ C11/F17 の air escape を守る）を両取りして大きく改善できる、というのが狙い。

**未解決の検知課題**: 単一 conditioning `c`（固有値比）では **F10（air 維持が最適）と F11–F14（droplet が最適）がどちらも ill-cond で高 `c` となり分離できない**。separability（cov の軸整列度＝対角エネルギー比）で F04（separable→close）は拾えるが、ill-cond 内の "air 維持 vs droplet" を分ける第3の構造シグナル（例: 固有値スペクトルの形状 — 滑らかな勾配 ellipsoid か、単一軸支配の cigar/discus/ridge か）が要る。**報酬ベース（V2a バンディット）は再発禁止**、あくまで f 非依存の構造シグナルで検知する。

検証ログ: 案1 `results/20260701_105621_eval_sched_quick/`、案2 `..._111711_eval_sched2_quick/`、Phase1 `..._113615_eval_phase1_quick/`、Phase2 `..._115405_eval_phase2_quick/`。実装足場は `mceso.py:_channel_ratios` ＋ `channel_schedule`（既定で base、ルーティング研究継続のため保持）。

### シグナル診断（全35関数, 2026-07）— `cond` ＋ `algA` で最適ルートが分離

router を作る前に、**base MC-ESO を一切変えず**（SR 不変）各世代の集団共分散・動態から f 非依存の構造シグナルを測定し、既知の最適ルート（air 予算の行き先: droplet / close / keep-air）との対応を全 35 関数で可視化した（`scripts/measure_channel_signals.py`、n_runs=8, dim2, 探索フェーズ＝σ>drilling の世代の median → run 平均）。

**結論: 2 シグナルで優先関数が綺麗に割れる**。
- **`cond`（= log10 λmax/λmin, 固有値比）> 2.5 → droplet**: F11–14（3.3–5.6）を切り出す。間に [1.63, 3.34] のギャップ。F08/F09 Rosenbrock（2.5–2.6）も droplet の得意領域なので有益に巻き込む。
- **cond ≤ 2.5 かつ `algA`（軸整列度＝separability, ドミナント固有ベクトルの max\|成分\|）> 0.98 → close**: F04(0.988)/F16(0.994) を切り出す。
- **それ以外 → keep-air（＝base 完全一致）**: 多峰/回転（F19/F15/F24/C05/C11…）は air 据置きで escape 保護。**判断がつかない維持関数は自動的に base のまま無傷**という安全構造。

診断した他シグナルは分離器として劣ることも確認（`offd` 相関非対角は F16 と F17/F24 が重なり不可、`nelX` はニッチ上限 6 に飽和し無情報、`nelt`/`spil`/`kurt`/`mgap` は多峰でノイジー）。→ **cond + algA を採用**。

**全 35 関数の測定値**（`cond`=log10固有値比, `PR`=participation ratio[1,2], `algA`=軸整列[.71,1], `offd`=相関非対角RMS[0,1], `divs`=分散/span, `kurt`=超過尖度, `mgap`=最大正規化間隙, `nelt`=平均ニッチ数, `spil`=spillover回数。route: ★=改善優先, ·=維持, ✗=非重視）:

| function | route | pri | cond | PR | algA | offd | divs | kurt | mgap | nelt | spil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F12-BentCigar | droplet | ★ | 5.64 | 1.00 | 1.000 | 0.956 | 0.009 | −0.29 | 0.271 | 2.09 | 3.8 |
| F11-Discus | droplet | ★ | 5.42 | 1.00 | 0.953 | 1.000 | 0.009 | −0.26 | 0.239 | 1.99 | 3.2 |
| F13-SharpRidge | droplet | ★ | 4.89 | 1.00 | 0.956 | 1.000 | 0.005 | 0.13 | 0.257 | 1.67 | 6.0 |
| F14-DiffPowers | droplet | ★ | 3.34 | 1.00 | 0.965 | 0.993 | 0.002 | −0.34 | 0.246 | 1.50 | 3.6 |
| F02-EllipsoidalSep | droplet? | · | 5.62 | 1.00 | 1.000 | 0.406 | 0.005 | 1.24 | 0.380 | 1.95 | 3.4 |
| F10-EllipsoidalRot | droplet? | · | 5.53 | 1.00 | 0.977 | 1.000 | 0.009 | −0.28 | 0.243 | 1.96 | 3.1 |
| F16-Weierstrass | close | ★ | 1.63 | 1.05 | 0.994 | 0.486 | 0.050 | 0.21 | 0.427 | 3.35 | 4.1 |
| F04-BucheRastrigin | close | ★ | 0.91 | 1.25 | 0.988 | 0.315 | 0.010 | 0.48 | 0.411 | 2.03 | 7.5 |
| F06-AttractiveSector | close? | ★ | 1.26 | 1.11 | 0.957 | 0.698 | 0.002 | 0.43 | 0.349 | 1.48 | 2.9 |
| F18-SchafferF7ill | ? | ★ | 2.63 | 1.00 | 0.974 | 0.978 | 0.008 | −0.24 | 0.268 | 1.95 | 2.6 |
| C05-Eggholder | keep-air | · | 1.33 | 1.12 | 0.823 | 0.754 | 0.001 | 0.75 | 0.394 | 1.96 | 11.6 |
| F20-Schwefel | keep-air | · | 1.16 | 1.14 | 0.956 | 0.691 | 0.024 | −0.27 | 0.313 | 2.38 | 6.5 |
| F15-RastriginRot | keep-air | · | 1.07 | 1.18 | 0.861 | 0.774 | 0.018 | 0.01 | 0.343 | 2.33 | 4.4 |
| F17-SchafferF7 | keep-air | ✗ | 0.95 | 1.23 | 0.974 | 0.474 | 0.003 | 0.02 | 0.290 | 1.72 | 2.8 |
| F19-GriewankRosenbrock | keep-air | ★ | 0.73 | 1.38 | 0.951 | 0.513 | 0.101 | −0.14 | 0.311 | 4.29 | 5.2 |
| F24-LunacekRastrigin | keep-air | ✗ | 0.64 | 1.47 | 0.876 | 0.498 | 0.076 | 0.21 | 0.325 | 4.31 | 13.1 |
| C11-DeJongF5 | keep-air | ✗ | 0.54 | 1.53 | 0.951 | 0.300 | 0.006 | −0.07 | 0.340 | 2.68 | 7.8 |
| C07-BukinN6 | - | · | 4.15 | 1.00 | 0.993 | 0.992 | 0.001 | 1.26 | 0.335 | 1.67 | 13.0 |
| F22-Gallagher21 | - | · | 2.74 | 1.00 | 0.900 | 0.993 | 0.012 | −0.56 | 0.239 | 2.29 | 4.1 |
| F08-Rosenbrock | - | · | 2.56 | 1.01 | 0.897 | 0.986 | 0.007 | −0.18 | 0.272 | 2.12 | 3.4 |
| F09-RosenbrockRot | - | · | 2.55 | 1.01 | 0.962 | 0.963 | 0.006 | −0.07 | 0.293 | 2.00 | 3.4 |
| F05-LinearSlope | - | · | 2.28 | 1.01 | 0.999 | 0.437 | 0.003 | −0.09 | 0.290 | 1.71 | 4.2 |
| F07-StepEllipsoidal | - | · | 1.57 | 1.05 | 0.781 | 0.944 | 0.009 | −0.62 | 0.241 | 2.12 | 5.0 |
| F21-Gallagher101 | - | · | 1.31 | 1.10 | 0.799 | 0.897 | 0.017 | −0.48 | 0.275 | 2.56 | 4.0 |
| C02-SixHumpCamel | - | · | 1.28 | 1.21 | 0.980 | 0.609 | 0.053 | −0.56 | 0.574 | 2.19 | 3.6 |
| F03-RastriginSep | - | · | 1.03 | 1.19 | 0.994 | 0.294 | 0.018 | 0.08 | 0.397 | 2.50 | 4.1 |
| C01-Himmelblau | - | · | 0.83 | 1.30 | 0.933 | 0.547 | 0.001 | 1.32 | 0.603 | 1.81 | 2.4 |
| F23-Katsuura | - | · | 0.67 | 1.43 | 0.912 | 0.435 | 0.144 | −0.09 | 0.282 | 5.49 | 3.4 |
| C03-Shubert | - | · | 0.65 | 1.45 | 0.940 | 0.385 | 0.038 | 1.15 | 0.533 | 3.00 | 3.5 |
| C06-Michalewicz | - | · | 0.62 | 1.46 | 0.971 | 0.255 | 0.002 | 0.44 | 0.363 | 2.15 | 3.9 |
| C08-StyblinskiTang | - | · | 0.49 | 1.59 | 0.935 | 0.313 | 0.001 | 0.79 | 0.415 | 1.65 | 3.9 |
| C04-FiveWell | - | · | 0.46 | 1.62 | 0.915 | 0.309 | 0.002 | −0.13 | 0.310 | 1.99 | 3.5 |
| F01-Sphere | - | · | 0.38 | 1.71 | 0.917 | 0.268 | 0.002 | −0.13 | 0.301 | 1.60 | 4.0 |
| C10-SchafferN2 | - | · | 0.34 | 1.76 | 0.894 | 0.258 | 0.031 | −0.50 | 0.248 | 2.72 | 4.0 |
| C09-Easom | - | · | 0.32 | 1.78 | 0.873 | 0.252 | 0.250 | −0.73 | 0.252 | 5.30 | 7.6 |

**ルーティングの安全性（全 35 関数の落ち先確認）**: 優先関数は全て改善 or 安全側に落ちる（F11–14→droplet, F04/F16→close, F19→keep-air, F06/F17→keep-air は base 維持）。維持関数は keep-air（base 完全一致で無傷）か、droplet（F08/F09 は得意領域）/close（F03/F05 separable）で許容。**残リスク**: F22-Gallagher21（多峰）が cond2.74 で droplet 落ち（base 100%・過去 variant 未回帰で低リスク）、close バケツが F03/F05/C02 を巻き込む（separable で許容見込み）、閾値は dim2 tuned で hold-out 汎化確認要。→ この cond+algA ルーターを実装・検証する。

### ルーターの実装反復と本体統合（2026-07, 採用）

診断を基に router を実装し、5 回の反復で仕上げてから本体統合した。**ルート確定の設計が肝**（per-gen で毎世代分類すると閾値付近の関数が flip-flop して回帰する）:

| 反復 | ルート確定方式 | overall SR@1e-10 | 問題 |
|---|---|---|---|
| per-gen | 毎世代 cond/algA で分類（固定なし）| ±0.0（87.9→87.9）| 閾値付近 F04/F06/F14 が flip-flop 回帰、F13+30 は取れた |
| commit@120 | gen120 で 1 回確定・固定 | −0.6 | flip-flop は解消も F13 の早期 droplet を取り逃す（late commit）＋F23−20 |
| hybrid | cond>4.0 早期 droplet latch ＋ gen120 commit | −0.1 | F04/F06/F14 解消・F13 回復も、algA だけでは F04(close希望)と F17(air希望)が分離不能で F17/F20 が close 誤爆 −15/−5 |
| **+mgap（採用）** | hybrid ＋ close 条件に `mgap>0.36`（座標間隙）を AND | **+0.6（87.9→88.4）** | F17/F20 を keep-air に戻し F04 close 維持。改善 7（F04/F11/F12/F13/F14/F16 +5, F24 +5）/回帰 2（F20−5, F23−10）|

**分離できなかった F04 vs F17 を第 3 シグナル `mgap` で解決**: 両者 algA≈0.975 で軸整列は同じだが、regular separable(F04)は座標方向に広い間隙(mgap≈0.41)、deceptive(F17)は密(≈0.29)。`close = algA>0.965 かつ mgap>0.36` で分離。

**汎化検証（統合の必須ゲート, SC/IRSC の轍回避）**: BBOB dim2 +0.6（87.9→88.4）/ BBOB dim3 +0.6（43.5→44.2, 有意回帰なし・SR@1e-4 64.6→69.0）/ CEC2022 dim10 hold-out は SR 同（0%）だが **median best_f 改善 2・悪化 0（G06 364→202, G03 改善）**。**全次元・hold-out で回帰なし**を確認し `channel_schedule=True` を**本体デフォルトに統合**（`MC-ESO-Orig`=`channel_schedule=False` で旧 flat-ratio を pin）。確定シグナル/閾値: `cond_droplet_early=4.0`（早期 latch）/`route_commit_gen=120`/`cond_droplet_thresh=3.0`/`align_close_thresh=0.965`＋`close_mgap_thresh=0.36`。

**残課題（統合後）**: F13/F14/F18/F23 は median best_f が既に 1e-12〜1e-15 到達済みだが少数 run が誤 basin で stuck（SR<100%）。これは"降下"最適化でなく"脱出"の問題で、チャネル配分では届かない。→ 下記 stuck-run 脱出（媒介チャネル）へ。

### stuck-gated 媒介感染チャネル（2026-07, 現状 REJECT）

上記の stuck-run 問題（router では届かない）に対し、**4 つ目の伝染チャネル「媒介感染（migratory / vector-borne）」**を実装（`migratory_channel`）。drilling 中（σ<span×1e-3）かつ停滞（no_improve≥閾値）でのみ発火し、dead-slot の一部を「best からの構造化大ジャンプ」（半分は主固有ベクトル方向＝ridge/valley 脱出、半分は等方＝多峰脱出、span×0.2）に充てる。

**初回（no_improve≥100, ratio0.5）: REJECT。** 「best_f=全 eval min ＋ μ+λ rollback だから SR を下げない」という前提が**実測で否定**された（stuck 救出 F23 60→85 **+25**/F18+10/F13+5 は本物だが、**F14−15/F17−10/F19−5 の回帰**で相殺、overall +0.1 誤差内・Wilcoxon 0/0）。**教訓: 追加チャネルは有限 dead-slot で他チャネル子個体を置換し RNG をずらすため、一度発火するとその run 全体が非 bit-identical になり、回復途中の run（F14 の平坦 basin 遅延 breakthrough）を壊す＝純加算でない。**（`project_migratory_channel` に教訓記録）。

**厳格化版（`migratory_no_improve_thresh=200`／`migratory_ratio=0.34`）も REJECT（確定）。** overall SR@1e-10 ±0（88.4→88.4, Wilcoxon 0/0）。回帰関数は 4→3 に減ったが **F17−20（悪化拡大）/F14−15 は消えず**、救出 F23+20/F13+15/F18+5 と同格で相殺という構造が 2 回とも再現。**gate 調整では解けない** — 救う stuck（F23）と壊す stuck（F17）はどちらも「多峰で停滞」で検知区別不能（router の per-gen flip-flop 問題と同型）。媒介チャネルは確定 REJECT、`migratory_channel` フラグは既定 OFF で保持（研究記録）。検証ログ: 初回 `results/20260702_001636_eval_mig_quick/`、厳格版 `results/20260703_184939_eval_mig_strict_quick/`。

検証ログ（ルーター確定〜統合）: router4=`results/20260701_225007_eval_router4_quick/`、dim3=`..._005956_eval_gen_dim3_quick/`、CEC=`..._115938_eval_gen_cec_quick/`。診断スクリプト: `scripts/measure_channel_signals.py`。
