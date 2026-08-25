# 開発履歴 — 試した工夫・フラグ・ablation 記録

MC-ESO の開発で試した機構・フラグの履歴。**何を試し、採用 / 不採用となり、なぜか**を残す。手法の最新アーキテクチャは [mceso.md](mceso.md)、評価方法論は [experiments.md](experiments.md) を参照。

> **判定の前提**: 改良案は `quick_check.py` で全関数 ablation し、SR / 評価回数 / Wilcoxon の 3 指標で overall 改善を確認できたものだけを `main.py` の `_BASE_OPTIMIZERS` に統合する。**判定は quick n=20 / max_evals=5000 / `--all` で統一**する（GitHub Actions の n=100 ワークフローは裏で補助的に回すもので、評価には参照しない）。
>
> **注**: 以下の検証ログの多くは quick デフォルトが n=10 だった時期に取得したもので、記載の数値は当時の n=10 結果。現在の評価標準は n=20。再検証する場合は n=20 で取り直す。

---

## 検証フロー

`quick_check.py` の `_OPTIMIZERS` は **MC-ESO 本体 ＋ 7 ベースライン（CMA-ES / IPOP / BIPOP / PSO / DE / L-SHADE / SaVOA）の 8 手法**を並べた標準比較セット。改良案や診断 variant を検証するときだけ、`_OPTIMIZERS` に一時的に MC-ESO のサブクラス／別 entry を追加して quick で overall 改善を測り、確認できれば本体に統合してから一時 entry を外す。**診断・ablation variant は常設しない**（`core/optimizers/mceso_ablations.py` に実体があり、必要時に追加する）。

```bash
./run.sh quick --all       # 2D BBOB-24 で評価（手法評価の標準。Custom は --custom で追加）
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
| **route-gated best2**（飛沫 current-to-best/2, `droplet_variant="best2_droplet"`, 2026-07 統合） | droplet ルート確定 run のみ飛沫の差分に第 2 差分ベクトル `+F·(x_c−x_d)` を追加（off-route は追加 RNG も引かず base と bit-identical）。悪条件谷/ridge のドナー多様性を増し、誤 basin で停滞した run を救出。**dim2 +1.6pt（88.4→90.0, F13 55→100/F14 75→100 Wilcoxon large）・dim3 +12.9pt（44.2→57.1, F02/F10/F11/F12/F13/F14 全 large・負け1）・CEC2022 dim10 改善3/悪化2（G01 64×）**＝全次元・hold-out 汎化。積み残しだった「stuck-run 脱出（媒介チャネル REJECT）」を 4 本目でなく飛沫の中身＋既存ルーターのゲートで解決。詳細は下記「チャネル中身差し替えの系統スイープ」|

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

## 集団レベル3機構の必要性 ablation（n=20, 2026-07）

[mceso.md](mceso.md#集団レベルの-3機構) の「集団レベルの 3 機構（系統共存 / 宿主競合 / スピルオーバー）＋ drilling」を **1 機構ずつ OFF にして全寄与を n=20 標準で定量化**。各 OFF 版を full MC-ESO（reference）と比較（quick n=20 / 5000 / `--all` 2D BBOB-24, base SR@1e-10=92.9%）。ablation 実体は `mceso_ablations.py`（宿主競合＝`MCESONoHostCompetition` で rollback 無効、スピルオーバー＝既存 `MCESONoSpillover`）＋ kwargs（系統共存＝`n_elite_max=1` で単一 best strain に縮退、drilling＝`sigma_drill_down=0.95` で加速収縮除去）。

| 機構 OFF | 変種 | SR@1e-10 | Δpt | evals_mean | Wilcoxon 有意悪化(数) | 寄与が集中する関数 |
|---|---|---|---|---|---|---|
| **宿主競合**（rollback） | `abl_noHostComp` | 71.0% | **−21.9** | 1473（遅化） | **8**（F06/F10/F11/F12/F13/F14/F17/F18） | ill-cond 谷/ridge（F13/F14 −75, F10 −60, F11/F12 −40, F06 −50）。μ+λ 単調改善保証が悪条件精密降下の生命線 |
| **スピルオーバー** | `abl_noSpill` | 80.2% | **−12.7** | 562（速化） | **6**（F03/F04/F15/F19/F20/F24） | separable/多峰（F04 −80, F15 −65, F03/F20/F19 −35〜−50）。誤 basin/deceptive からの脱出リスタート |
| **系統共存** | `abl_noStrain` | 90.6% | −2.3 | 962 | 2（F06/F17） | F17 −25/F04 −19 中心。ただし F18/F23 で **+9**（トレードオフ）。診断の novelty gap（live n_elite≈1）と整合＝寄与は限定的 |
| **drilling**（sigma_drill_down） | `abl_noDrill` | 91.7% | −1.3 | 795 | 1（F17） | F17 −35 が主だが F18 +9/F24 +5 で相殺。加速収縮の overall 寄与は誤差級 |

**結論（寄与の序列）**: **宿主競合 ≫ スピルオーバー ≫ 系統共存 ≈ drilling**。4 機構すべて overall SR@1e-10 を上げており（OFF で全て低下）、**どの ablation も MC-ESO を Wilcoxon で有意に上回った関数はゼロ**（4 機構とも「性能に寄与」を確認）。ただし寄与量は 2 桁違う: 宿主競合の rollback（単調改善保証）と停滞スピルオーバーが load-bearing、系統共存と drilling は net で僅少。**系統共存/drilling は簡潔化候補に見えるが、F17 等で SR@1e-10 を落とす（net でも −2.3/−1.3pt）ため [[feedback_sr_non_negotiable]] により削除不可**。系統共存・drilling が deceptive ill-cond 多峰（F18/F23/F24）でだけ逆に足を引っ張る（OFF で改善）のは既知のトレードオフ（air escape 系と同根）。

検証ログ: `results/20260708_001002_eval_pop_ablation_quick/`（全 24 関数 × 5 手法, summary/wilcoxon.csv）。ablation entry は `quick_check.py`（`abl_noStrain`/`abl_noHostComp`/`abl_noSpill`/`abl_noDrill`）。

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

### チャネル中身差し替えの系統スイープ（2026-07）— 追加でなく「中身」を全チャネル網羅、route-gated best2 のみ採用

「3 チャネル（close/droplet/air）が最良である保証はない」という問いに対し、**追加系（4 本目）はほぼ全滅済み**（V2b 4 チャネル / 媒介チャネル ×2 / UCB-AOS 比率 / 一律再配分 4 案）— 有限 dead-slot を奪い RNG をずらして回復途中 run を壊すため。そこで**本数は 3 本のまま、各チャネルの数学（中身）を差し替える**方向を系統スイープした。各変種はフラグ化し**デフォルト＝base 完全一致（bit-identical）**を smoke test で確認、1 pass（base を 1 回だけ回す）で 8 変種を同時評価。

**スクリーニング（quick n=20 / 5000 / dim2, base SR@1e-10=90.5%※C01-C06+BBOB24 の 30 関数 ← C07-C11 は前段 experimenter の watchdog 中断で欠落、優先関数は全含む）**:

| チャネル | 変種 | 中身 | overall SR@1e-10 | 判定 |
|---|---|---|---|---|
| 空気 | airCauchy / airLevy / airOpp / airUnif | Cauchy裾 / Mantegna Lévy(β=1.5) / 反対向き 2·centroid−host / 全域一様 | 89.5 / 88.5 / 89.2 / 89.7 | **全 REJECT** |
| 接触 | clCauchy | 回転・異方性維持で radial を Cauchy | 89.5 | REJECT |
| 飛沫 | drRand1 / drPbest | DE/rand/1 / current-to-pbest(top15%) | 82.3 / 88.3 | REJECT |
| 飛沫 | drBest2（global） | current-to-best/2 を全ルート | 90.8（+0.3, 相殺） | REJECT |

**当初本命だった「空気の重い裾」は不成立**（escape 関数は drilling 停止＋router keep-air で既に保護され、裾を重くしてもノイズ増のみ）。**網羅したからこそ本命でなく飛沫の第 2 差分ベクトルが浮上**。

**drBest2（global）の実態＝媒介チャネル REJECT と同型のトレードオフ**: F13-SharpRidge 55→100(+45)/F14 75→100(+25) を救う一方、F17/F18/F24/C05/F23 の多峰 keep-air を毀損して overall +0.3 とノイズ内（summary の Wilcoxon「0/2」は多峰回帰がノイジーで有意化せず誤誘導）。決定差は**壊す関数は全て keep-air ルート・救う F13/F14 は droplet ルート**。

**route-gate 版 `best2_droplet`（採用）**: 第 2 差分を **droplet 確定 run のみ**適用（off-route は c,d を引かず base と bit-identical → keep-air 多峰を構造的に無傷化）。全 35 関数で:
- **dim2 +1.6pt（88.4→90.0）**: F13 55→100/F14 75→100（Wilcoxon large）、小回帰 F18−10/F19−5（borderline-cond 多峰が稀に droplet 誤ルート）。
- **dim3 +12.9pt（44.2→57.1）**: F02+45/F10+80/F11+60/F13+50/F14+60、Wilcoxon で base に有意勝ち6（F02/F10/F11/F12/F13/F14 全 large）・負け1、evals も速化（2169→1890）。高次元ほど悪条件谷が難しく第 2 差分が効く。
- **CEC2022 dim10 hold-out**: median best_f 改善3（G01 0.0167→0.00026=64×, G06 202→166, G11）/ 悪化2（G03/G07）＝ net 正。

**drilling-gate 版 `best2_stuck`（REJECT）**: 「stuck 局面に限定すれば漏れ F18/F19 を消せる」狙いで route∧drilling∧… を試したが、救出の大半が drilling 到達前の droplet 降下で起きるため **F13 +45→+20/F14 +25→+10 と減衰し overall +0.6 に低下**。route-gate（drilling 制限なし）が優位。

→ **全次元・hold-out で汎化**（却下された SC/IRSC / 媒介チャネルの轍を回避）を確認し `droplet_variant="best2_droplet"` を**本体デフォルトに統合**。積み残しだった「F13/F14 の stuck-run はチャネル配分では届かない→媒介チャネル(4 本目)」を、**4 本目でなく飛沫チャネルの中身（current-to-best/2）＋既存ルーターのゲート**で解決（in-place ゆえ媒介チャネルの「dead-slot RNG ずらし」問題を構造的に回避）。**不採用変種（air 全種 / close-cauchy / rand1 / cur2pbest / global-best2 / best2_stuck）はコードから削除**。

検証ログ: スイープ `results/20260703_194850_eval_channel_sweep_quick/`（air/close/droplet 8 変種, C07-C11 欠落）、dim2 確認 `results/20260706_173734_4d85c1c_quick/`（base+best2+best2R 全 35）、drilling-gate `results/20260706_184052_4d85c1c_quick/`、hold-out dim3 `results/20260706_185749_4d85c1c_quick/`・CEC `results/20260706_190420_4d85c1c_quick/`。

### 進捗報告データの取得（2026-07-07）— 既存手法比較（10 手法）＋変更点 ablation 累積梯子

進捗報告用に 2 データセットを quick（n=20 / 5000 / dim2 / 全 35 関数）で取得。

**既存手法比較**（`results/20260707_進捗報告データ_既存手法比較_10手法/`）: 新規追加した NM-Restart / NCDE を含む 9 ベースライン + MC-ESO。**MC-ESO が SR@1e-10 89.9% で 10 手法中 1 位**（2 位 DE 85.7%）。旧 8 手法は前回 run（`20260706_進捗報告データ_既存手法比較/`）を再現（pycma 系のみ非決定性で ±0.3pt 内）。NM-Restart（60.6%）には多峰 16 関数で圧勝・SR で負けるのは F23-Katsuura のみ（65 vs 60）＝「2D は multistart 局所探索で十分」への反証データ。NCDE（38.7%）は C01 の多解被覆で優る（PR@1e-4 0.85 vs 0.53）が単解精度は最下位。留意: C01 の PR は NM-Restart が 1.00 で最強。

**変更点 ablation 累積梯子**（`results/20260707_進捗報告データ_変更点ablation/`、再現確認 run: `20260707_154422_ablation_ladder_quick/` — 関数別 SR@1e-10 完全一致＝決定性 OK）: 5/18 版から committed 改善を 1 つずつ有効化する累積梯子。**85.1 → +ir 85.4 → +floor/nich 87.9 → +router 88.4 → +best2 89.9**。変種の再構成フラグ（`quick_check.py` に一時登録、名称は summary.csv の method 名）:

| 変種 | kwargs（`MultiChannelEpidemicOptimizer`） |
|---|---|
| `abl0_base2018`（5/18 版再構成） | `droplet_variant="cur2best", channel_schedule=False, cov_floor_low=0.01, exhausted_no_improve_mult=1e9, ir_archive_frac=0.0, ir_repel_max_tries=0` |
| `abl1_ir` | 同上から `ir_archive_frac` / `ir_repel_max_tries` を既定に戻す |
| `abl2_floornich` | `droplet_variant="cur2best", channel_schedule=False` |
| `abl3_router` | `droplet_variant="cur2best"` |

---

## 次元スケーリングの計測と高次元崩壊（2026-07〜08）

### 現状把握 — dim3 で 1 位を失い、dim5 以上で崩壊（2026-07-24）

BBOB を dim 2/3/5/10/20 でレジストリ化し（`core/benchmarks.py:_build(d)`、`quick_check.py --dim`）、**評価予算を次元比例（2500×d）**にして全手法を計測（quick n=20, `--all`）。検証ログ: `results/20260724_014240_perf_d2_quick/` ほか d3/d5/d10/d20。

| 手法 | d2 | d3 | d5 | d10 | d20 |
|---|---|---|---|---|---|
| **MC-ESO** | **93.3** | 67.3 | **17.3** | **6.5** | **0.0** |
| IPOP-CMA-ES | 83.8 | 67.3 | 56.0 | 47.1 | 47.3 |
| BIPOP-CMA-ES | 79.2 | 67.7 | 54.8 | 49.4 | 47.3 |
| DE | 89.4 | 71.0 | 44.0 | 15.4 | 7.7 |
| CMA-ES | 64.6 | 53.5 | 44.4 | 42.7 | 40.8 |

（SR@1e-10 平均。予算不足ではない — d10 は 25000 evals）**dim3 で既に 1 位でなく、dim5 で最下位群**。失敗は 2 系統に分かれる:
- **精度グラインドの失速**: d10 F01-Sphere は SR@1e-2 100% → 1e-10 30%。正しい basin にいるのに深精度へ降りられない。
- **悪条件で完全崩壊**: d10 F08/F10/F11/F12/F14 は SR@1e-1 すら 0%。

### 外れた仮説 3 つ（2026-07-25〜26）

いずれも d5/d10 で改善せず（`results/20260725_115355_diag2_d5_quick/`、`20260726_015000_proto_d5_quick/` 他, n=8）。診断 variant は常設しない方針に従い `quick_check.py` の一時エントリ（`diag_dropAll` / `diag_npopSmall` / `diag_accum05` / `diag_accum15`）も削除済み（kwargs は下表のとおりで再登録できる）:

| 仮説 | 変種 | 結果 |
|---|---|---|
| router が高次元で ill-cond を誤ルートし best2 が発火しない | `diag_dropAll`（`cond_droplet_early=-1e9` で全 run droplet 強制）| d5 3.1→16.2 と動くが CMA-ES 85.6 に遠く、d10 は 0.0 のまま |
| 世代数不足（n_pop=40 が大きすぎる）| `diag_npopSmall`（`n_pop=12`）| d5 0.6 / d10 0.0（悪化）|
| C_pop の単世代推定分散が高次元で大きすぎる → basin ローカル共分散 EMA 累積（rank-μ 風、spillover でリセット）| `cc_accum_rate` 0.05 / 0.15 | SR@1e-10 不変（d5 3.1 / d10 0.0）、**SR@1e-2 は 57.5→38.1 と悪化**。寄与ゼロ以下が確定したため `cc_accum_rate` / `cc_accum_min_dim` / `_MCESOState.cc_cov_accum` と累積分岐は**コードから削除**（再現するには `_close_contact_children` の `cov` を世代間 EMA に差し替える）|

### 真因 — 停滞窓の単位バグと spillover リミットサイクル（2026-08-23, 採用）

**σ の軌跡を直接トレースして特定**（`history_sigma_global` / `history_no_improve`、d10 F01/F10 の 1 run）:

```
no_improve: 290 → 240 → 190 → 140 → 90 → 40 → 290 → ...   （サイクル）
sigma/span: 1.36e-2 → ... → 4.89e-2 → 1.36e-2（spillover でリセット）
spillover 発火: F01 46 回 / F10 65 回、drilling 到達 0%
```

`no_improve` は**評価回数**カウンタだが、1 世代は `kill_fraction × n_pop` 評価を消費する（dim2 で 5、dim10 で 10、dim20 で 20）。よって**固定 300 評価の窓は世代数で見ると次元とともに縮む**（dim2 = 60 世代 → dim10 = 30 世代）。高次元ほど長い grind が必要なのに逆方向。結果、spillover が絶えず発火して σ を `σ_init × restart_sigma_ratio` に戻すリミットサイクルに入り、σ は `0.2×0.3×0.95^30 = 1.29e-2·span` で下げ止まる（**実測 σ_min と完全一致**）。drilling 領域（1e-3·span）に一度も入れないため深精度 SR が出ない。加えて 1 回の spillover が n_pop−1 = 39 評価を消費し、予算の約 10% を再播種に浪費していた。

**修正**: 停滞窓を `restart_no_improve_threshold × (dim/2)^restart_window_dim_scale`（既定 1.0）に。**dim2 では係数が必ず 1.0 なので低次元は bit-identical**（BBOB-24 × 24 関数 × 8 指標で差分セル 0 を確認）。`_stagnation_window()` を `_spillover_should_fire` / `_basin_exhausted` の両方が参照する。

**検証（quick n=20, `--all`, 予算 2500×d, 旧挙動 `hd_win0` = `restart_window_dim_scale=0.0` と 2 手法比較）**:

| 次元 | SR@1e-10 旧→新 | SR@1e-4 旧→新 | Wilcoxon (ref=新) |
|---|---|---|---|
| d2 | 93.33 → 93.33（**全指標 bit-identical**）| 96.46 → 96.46 | — |
| d3 | 67.29 → 67.08（−0.21）| 80.21 → 80.62 | 勝ち 4（F12/F13/F14/F23）/ 負け 1（F03 medium）|
| d5 | 17.29 → 17.50（+0.21）| 35.62 → 31.67（悪化）| 勝ち 6 / 負け 4（F03/F07/F09/F11）|
| d10 | 6.46 → **10.00（+3.54）** | 13.75 → **20.62** | **勝ち 14 / 負け 2**（F07/F11）|
| d20 | 0.00 → **5.42（+5.42）** | 0.21 → **11.25** | **勝ち 21 / 負け 2**（F07/F11、大半が large）|

**効果は次元とともに拡大する**（係数 = dim/2 なので機構の予測どおり）。d20 では SR@1e-2 も 7.71 → 21.04 で、F05 が SR@1e-10 0→100、F06 が SR@1e-2 0→100、F14 40→100、F02 5→90。回帰は d10/d20 とも F07-StepEllipsoidal / F11-Discus の 2 関数に固定している。

関数別（d10）: F01 30→80、F02 25→45、F05 80→100、F08 SR@1e-2 0→50、F17 SR@1e-2 5→45。回帰は F07-StepEllipsoidal（−10）と F11-Discus（SR@1e-2 85→15）。d3/d5 が wash なのは、窓を伸ばすと悪条件（F12/F13/F14）が伸びる一方で separable 多峰（F03/F04/F07）の escape リスタートが減るトレードオフのため。

**指数の選択**: 保守版（窓を世代数一定＝子個体数比例、dim2/3/5 は完全無変更・dim10 で係数 2）も d10 全 24 関数で比較したが、**dim スケール版（係数 5）が優る**（SR@1e-10 10.00 vs 8.96、Wilcoxon 勝ち 9 / 負け 2）。よって連続スケール（指数 1.0）を採用。検証ログ: `results/20260823_231616_hdwin_d2_quick/`（d2）、`20260823_233141_hdwin_d3_quick/`、`hdwin_d5`、`hdwin_d10`、保守版比較 `hdwin600_d10`。

**教訓**: 絞り込み関数セット（悪条件寄り 10 関数）での中間測定は d5 で「+5pt」と出たが、**全 24 関数では +0.2pt** だった。セット選択のバイアスで効果を過大評価する典型例で、判定は必ず `--all` で行う。

### 残る課題 — 高次元の悪条件（未解決）

窓を直しても **d10 の F08/F10/F11/F12 は SR@1e-10 0%** のまま（drilling には世代の 5〜58% で入るのに median best_f が 1e+0〜1e+1 で停止）。d20 では F06/F14 が SR@1e-2 100% に届くようになったが、深精度（1e-10）は依然として悪条件関数で出ない。`cov_floor_low` を 1e-3 → 1e-5/1e-7/1e-9 に下げるスイープでも改善はノイズ級で、**異方性 floor は主因ではない**。IPOP/BIPOP-CMA-ES が同条件で 100% を取るのは rank-μ が「選択された子の**移動ステップ**」から共分散を学ぶためで、MC-ESO の C_pop は「エリート集団の**位置分布**」— 情報源が異なる。次の候補は「成功した子個体のステップから共分散を推定する」方向（失敗した `cc_accum_rate` は位置共分散の EMA であり別物）。

---

## 全パラメータの次元不変性 監査（2026-08-24）

停滞窓の単位バグ（上節）を受けて、**44 個の全パラメータについて次元依存の有無をコード読解＋実測で洗い出した**。実測は base MC-ESO を一切変えずに、各パラメータが制御している「派生量」を dim 2/5/10/20 で測る方式（`scratchpad` の probe を `_softmax_weights` / `_adaptive_cov_floor` / `_record_generation` にフック、全 24 関数 × 2 run、予算 2500×d）。

### 実測された派生量のドリフト

| dim | n_pop | parEff（実効親数）| nelt（飽和%）| cond | algA | mgap | anis | route 分布 |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 19.4 (97%) | 1.21 (8.5) | 2.48 | 0.924 | 0.263 | 556 | keepair 11 / droplet 8 / close 4 |
| 5 | 20 | 19.3 (96%) | 1.21 (8.5) | 3.07 | 0.764 | 0.346 | 898 | droplet 14 / keepair 10 / close 0 |
| 10 | 40 | 36.2 (91%) | 1.21 (14.5) | 2.91 | 0.644 | 0.312 | 1352 | keepair 13 / droplet 11 / close 0 |
| 20 | 80 | 60.3 (75%) | 1.67 (23.5) | **15.37** | 0.542 | 0.309 | 2029 | keepair 12 / droplet 12 / close 0 |

### 監査結果の分類

**A. 次元不変が保証されている（問題なし）**: 子個体の配分比（`air_ratio` / `h2h_ratio` / `kill_fraction` / `ir_archive_frac`）、f 相対ゲート（`restart_quality_rel_floor` / `basin_switch_quality_rel_floor` / `log_slope_threshold`）、DE 係数（`h2h_F` / `h2h_CR` — 交叉座標数が `CR·d` で自動スケール）、回数・個数カウント（`n_elite_max` / `exhausted_no_improve_mult` / `basin_switch_after_failed_spillovers` / `ir_repel_max_tries`）、span 相対の座標あたり σ 比率（`sigma` / `sigma_ceil_ratio` / `restart_sigma_ratio` / `ir_reignite_sigma_ratio`）、`air_sigma_amplifier`（`diversity_ratio` が一様分布 std 0.289 で正規化済み）、`cov_ratio_beta`。

**B. 次元依存が確認された**（`restart_no_improve_threshold` は上節で修正済み）:

| パラメータ | ドリフト | 機序 |
|---|---|---|
| `align_close_thresh` (0.965) | algA 0.924→0.542 | ランダム基底の平均 max\|成分\| は √(2 ln d/d) で減衰 → d≥5 で到達不能、CLOSE ルート消滅 |
| `cond_droplet_thresh/early` (3.0/4.0) | cond 2.48→15.37 | 収束後の C_pop が数値的ランク落ち（λmin→0）。d20 では conditioning でなくランク崩壊を測っている |
| `cov_ratio_lo/hi` (1e3/3e4) | 同上 cond を入力 | d20 で常時飽和 → 適応異方性 floor の rugged/ill-cond 判別が死ぬ |
| `niche_radius_ratio` / `ir_repel_radius_ratio` (0.1×span) | ニッチ飽和 8.5%→23.5% | d 次元の典型点間距離 ∝ √d なので半径が相対的に縮小 |
| `sigma_up`/`sigma_down`/`sigma_drill_down` | — | 世代あたり固定倍率だが収束所要世代数は ∝ d（CMA-ES は damping ∝ 1/d）|
| `precision_sigma_ratio` / `sigma_floor_ratio` | — | σ は座標あたりだが実変位は σ√d |

**C. 次元依存ではないが機能していない**: **softmax 親選択**。`exp(f_max − f_i)` が f の**絶対差**依存のため、収束して f 差が ≪1 になると重みが平坦化する。実効親数 parEff = 1/Σw² は **dim2 の中央世代で 20.00/20（完全一様）**。「f が低い個体ほど感染力が高い」という MC-ESO の中核の着想が、実際にはほぼ一様ランダム選択に退化していた。

### スクリーニング（quick 相当・自前ハーネス, d10, n=10, **全 24 関数**）

B/C の各項目を既定 OFF のフラグとして実装（dim2 で no-op を確認）し、1 つずつ測定。base SR@1e-10 = 10.0:

| 変種 | SR@1e-2 | 1e-4 | 1e-7 | SR@1e-10 | 判定 |
|---|---|---|---|---|---|
| `niche_radius_dim_scale=1`（半径 ∝√d）| 28.3 | 22.9 | 10.8 | **8.3（−1.7）** | REJECT |
| `align_signal_dim_norm`（IPR 正規化）| 27.1 | 20.0 | 12.1 | 10.0（±0）| **変化ゼロ**（下記）|
| `cond_rank_guard=1e-8` | 26.7 | 20.0 | 12.1 | 10.0（±0）| d10 では無効果（cond 2.91 < guard 上限）。判定は d20 で |
| `sigma_adapt_dim_scale=1`（倍率 ∝1/d）| 28.8 | 22.5 | **16.7** | 10.0（±0）| 指数 1.0 は強すぎ（F01 70→**0**）。0.5 で再試験 |
| `sigma_threshold_dim_scale=1`（σ 閾値 /√d）| 27.5 | 21.7 | 11.7 | 9.6（−0.4）| REJECT |
| **`softmax_beta=5`（スケール不変選択）** | 26.2 | 21.2 | 15.4 | **13.3（+3.3）** | 有望 |

### ルーターは閾値較正でなく「推定器」の問題（重要）

`align_signal_dim_norm` で**変化がゼロ**だった理由を診断（`route_probe.py`, 全 24 関数の commit 時シグナル）。次元正規化した軸整列度（0=ランダム基底 / 1=完全整列）は **dim10 で全 24 関数が 0.00〜0.33**（separable の F03 が 0.108、F04 が 0.039。dim2 では 0.644 / −0.012）。**閾値の較正ミスではなく、40 個体・10 次元の標本共分散からは軸整列度そのものが推定できていない**（固有ベクトルが推定ノイズに支配される）。`cond` も同様に劣化し、dim2 で 5.64 だった F12-BentCigar が dim10 では 2.23 に落ちて **最も droplet を要する悪条件関数が keep-air へ落ちる**。

→ **ルーターの 3 シグナルはすべて同一の C_pop 由来で、高次元では C_pop 自体が谷に整列していないため、ルーター全体が情報を失う**。閾値の次元補正では復旧しない。復旧には推定器の変更（成功ステップからの共分散推定、または n_pop 増）が要る＝「高次元の悪条件」の残課題と同根。

### softmax のスケール不変化 — dim2 ゲート（quick n=20, `--all`）

| | SR@1e-2 | SR@1e-4 | SR@1e-7 | SR@1e-10 | evals |
|---|---|---|---|---|---|
| base | 96.46 | 96.46 | 95.00 | **93.33** | 804 |
| `softmax_beta=3` | 96.04 | 96.04 | 93.96 | 92.08（−1.25）| 775 |
| **`softmax_beta=5`** | 96.04 | 95.83 | 95.00 | **93.54（+0.21）** | 798 |

β=5 は dim2 でも primary を下げない。関数別は F18-SchafferF7ill **65→100（+35, Wilcoxon 有意勝ち）** / F04 +5 / F13 +5 に対し F06 −10 / F17 −5 / **F24 35→10（−25）**、Wilcoxon 有意な負けはゼロ。β=3 は −1.25pt で却下。検証ログ: `results/20260824_125218_dimf_softmax_d2_quick/`。 β=8 も測ったが dim2 で **91.04（−2.29、有意な負け 2: F06/F17）** となり却下（`results/20260824_135003_dimf_sm8_d2_quick/`）。**β=5 が上限**。

### 候補 2 種の全次元検証（quick n=20, `--all`, 予算 2500×d）

| 変種 | dim2 | dim3 | dim5 | dim10 | dim20 |
|---|---|---|---|---|---|
| **`softmax_beta=5`（採用）** | **+0.21**（W1/L0）| **+6.67**（W2/L1）| **+8.12**（W6/**L0**）| **+3.12**（W7/L3）| **+6.87**（W9/L2）|
| `sigma_adapt_dim_scale=0.5` | ±0（構造的 no-op）| **−2.29** | +3.12（有意差なし）| +3.54（W5/**L0**）| — |

（SR@1e-10 の base 比 pt、括弧内は Wilcoxon の変種勝ち/base 勝ち）

**`sigma_adapt_dim_scale` は保留/却下**: dim10 単独では W5/L0 と綺麗だが **dim3 で −2.29pt**（F19 −35 / F23 −25 / F06 −20 / F20 −15）。dim2 が no-op でも汎化ゲートを通らない。

**`softmax_beta=5` は全次元で改善**。dim5 が最大（17.50 → 25.62、**有意な負けゼロ**、F06 5→65 / F08 5→50 / F09 20→40 / F07 40→55 / F22 20→40）、dim3 でも +6.67（F03/F18 +25、F11/F13/F14 +20）。有意な負けは dim3 の F19-GriewankRosenbrock と dim10 の F08/F17/F18 に限られる。**次元バグではなくスケール不変性の欠如**の修正であり、修正の効果が全次元に及ぶのはそのため。dim20 でも SR@1e-10 5.42 → **12.29（+6.87）**、Wilcoxon **9 勝 2 敗**（F01 10→95 / F02 20→100、負けは F17/F20）。検証ログ: `results/20260824_165135_dimf_cand_d3_quick/`、`20260824_171811_dimf_cand_d5_quick/`、`20260824_135012_dimf_cand_d10_quick/`、`20260824_175214_dimf_sm5_d20_quick/`。

### 採用と後片付け

**`softmax_beta = 5.0` を本体デフォルトに統合**（`_softmax_weights`）。全次元で SR@1e-10 が改善し、判定基準の dim2 でも回帰なし・有意な負けゼロという条件を満たしたため。`softmax_beta=0.0` で旧式（生の f 差）に戻せる（`quick_check.py` の `dimf_softmax0` が回帰ピン）。

**不採用フラグはコードから削除**（本節に測定値を残したので再実装可能）: `niche_radius_dim_scale`（半径 ∝√d, d10 −1.7）/ `sigma_threshold_dim_scale`（σ 閾値 /√d, d10 −0.4）/ `align_signal_dim_norm` ＋ `align_close_thresh_norm`（IPR 正規化, 変化ゼロ）/ `cond_rank_guard`（d10 無効果、d20 は未測定だが上記のとおりルーターは推定器ごと機能しないため保留のまま削除）/ `sigma_adapt_dim_scale`（dim3 −2.29 で汎化ゲート不通過）。

### この監査で分かった構造的な限界（今後の課題）

1. **ルーターは高次元で機能しない** — 3 シグナルすべてが C_pop 由来で、高次元では C_pop 自体が推定ノイズに支配される。閾値の次元補正では復旧しない。
2. **高次元の悪条件は未解決** — 停滞窓と選択圧を直しても d10 の F10/F11/F12 は SR@1e-10 0% のまま。
3. 1 と 2 は**同根**（C_pop が谷に整列しない）。次の一手は「成功した子個体の移動ステップから共分散を推定する」方向（CMA-ES の rank-μ に相当）で、これはルーターのシグナル品質と接触感染の整列を同時に改善しうる。

---

## 伝播鎖メモリ（transmission-chain memory, 2026-08-24）— REJECT

監査で判明した「高次元では C_pop が推定ノイズに支配され、ルーターも接触感染の整列も機能しない」（上節）に対する最初の対策案。**CMA-ES の rank-μ をそのまま持ち込むと [mceso.md](mceso.md) の差別化主張（瞬間共分散＝履歴累積なし / O(pop·d)＝行列演算なし）を両方失う**ため、共分散行列を推定しない別機構として設計した。

**設計**: 宿主競合の rollback は「子が置き換えた宿主に勝ったか」＝**感染が成立したか**を既に判定しているので、その変位 `x_child − x_parent` を**単位ベクトル化**して長さ k の FIFO に積む（大きさは σ の役割なので方向のみ保持）。接触感染のステップを
`√(1−mix)·noise + √(mix·dim/draw)·Σ_j g_j u_j`（`g_j ~ N(0,1)`, `u_j` は FIFO から抽選）とし、**期待ステップ長 E‖step‖² = dim を厳密に保存**して σ ベース閾値の意味を変えない。spillover で FIFO を破棄（basin 乗換え時の即時再適応を維持）。`chain_memory_size=0` で base と bit-identical（RNG も引かない）。

**スクリーニング（d10, n=10, 全 24 関数, base=13.3）**: k を伸ばすと単調に改善し k=200 で飽和。mix=0.5 / draw=2 が最良。

| k / mix | 12/.5 | 40/.25 | 40/.5 | 100/.5 | **200/.5** | 400/.5 | 200/.25 | 200/.75 |
|---|---|---|---|---|---|---|---|---|
| SR@1e-10 | 14.6 | 14.6 | 14.2 | 15.4 | **15.8** | 15.8 | 15.0 | 15.4 |

**しかし機序仮説は 2 つの独立した診断で否定された。**

1. **分割半検定**（FIFO を互いに素な前半／後半に分け、主方向の一致度 \|cos\| を測る。正解データ不要で「幾何情報を持つか」を判定）: 伝播方向 0.985 / 0.633 / 0.566 / 0.611（dim 2/5/10/20）に対し **C_pop は 0.951 / 0.787 / 0.738 / 0.777** と、d≥5 では**集団共分散の方が自己一致度が高い**。「成立した伝播方向の方が良い幾何推定になる」は成り立たない。実効ランクは伝播方向が高く（d10 で 6.09 対 3.15）、**推定を鋭くするのでなく部分空間を広げている**のが実態。
2. **採択率検定**（chain 混合ステップが実際に採択されやすいか。自己一致度と違い「安定して間違っている」を弾ける）: 差は dim 2/5/10/20 で +0.0025 / **−0.0100** / ±0.0000 / −0.0028 と**全次元でゼロ近傍**。median best_f も改善せず d5 では悪化（3.5e-4 → 5.4e-2）。

**dim2 ゲートで失格**（quick n=20, F01–F19 まで取得した時点で判定）: SR@1e-10 **98.42 → 95.26**、**改善ゼロ / 悪化 4**（F18 100→70 で Wilcoxon 有意な負け、F04 −20、F16 −5、F17 −5）。

→ **REJECT・コード削除**。d10 の +2.5pt は機序に裏づけられておらず、方向の多様性が稀に stuck run を救う裾効果か n=10 のノイズ。過去の global-best2（+0.3、ノイズ内）と同型の却下理由に加え、判定次元 dim2 の SR@1e-10 を下げるため二重に不採用。検証ログ: `results/20260824_195854_chain_d2_quick/`、スクリーニングは scratchpad（`screen_dimflags.py` / `chain_quality.py` / `chain_utility.py`）。

**教訓**: 「C_pop が高次元で壊れている」は**軸整列シグナルが無情報**という意味であって、**主方向が不安定**という意味ではなかった。監査の結論を一段強く読み替えて設計してしまった。次に高次元の悪条件へ向かうときは、まず「何が壊れていて何が壊れていないか」を分離して測ること（分割半検定は安定性、採択率検定は正しさを測る。両方が要る）。

---

## 高次元悪条件の真因 — 集団の自己縮退（2026-08-24, 診断は確定 / 対策は REJECT）

伝播鎖メモリの失敗（上節）の教訓「まず何が壊れていて何が壊れていないかを分離して測る」に従い、**正解データ付き**で C_pop を評価した（`scratchpad/cpop_truth.py`）。

**測り方**: BBOB の悪条件関数は局所的に二次形式なので、best 点の**数値ヘッセ行列 H が局所幾何の正解**。理想 ES の変異共分散は H⁻¹ に比例するので、実際に標本抽出に使う共分散 M に対し

    eff = cond( H^(1/2) · M · H^(1/2) )

は **M ∝ H⁻¹ で 1、M = I（無適応）で cond(H)**。C_raw（推定そのまま）と C_used（平均正規化＋floor 後）を比べれば、(a) 推定が誤り / (b) floor が壊す / (c) そもそも谷に入れていない、を切り分けられる。

**結果（n=2 seeds × 4 チェックポイント の median）**:

| dim | 関数 | cond_H | eff_used | cos（谷方向） | erank/dim |
|---|---|---|---|---|---|
| 2 | F12-BentCigar | 5.76e5 | 2.88e2（eff_raw **2.01**）| 1.000 | 1.00/2 |
| 2 | F01-Sphere | 1.00 | 3.81 | — | 1.51/2 |
| 10 | F12-BentCigar | 2.65e6 | 2.15e7 | 0.875 | 2.49/10 |
| 10 | F08-Rosenbrock | 2.65e2 | 5.19e2 | 0.914 | 2.28/10 |
| **10** | **F01-Sphere** | **1.00** | **4.22e3** | — | **1.98/10** |
| 20 | F12-BentCigar | 7.38e3 | 8.42e5 | 0.553 | 7.13/20 |
| 20 | F01-Sphere | 1.00 | 5.75e2 | — | 5.74/20 |

**真因は (a)(b)(c) のどれでもない第 4 のモード＝集団の自己縮退**。
- **決定的証拠は F01-Sphere**: 真の条件数 1.00（＝理想の共分散は単位行列）の完全等方関数で、dim10 の MC-ESO は**実効条件数 4.2e3 の異方分布から標本を引いている**。landscape に異方性が無いのにアルゴリズムが自分で作っている＝**landscape の難しさでなくアルゴリズム側の欠陥**であることの証明。dim2 では 3.81 に留まる。
- **集団の実効ランクが関数によらず潰れる**: dim10 で 1.98〜3.13 / 10、dim20 で 5.74〜7.13 / 20。10 次元で実質 2 方向しか探索していない。
- 谷方向の把握（cos）は悪条件関数で 0.875〜0.914 と**保たれている**ので、「向きは合っているが動ける方向が足りない」状態。
- `eff_used ≈ eff_raw` なので **floor は機能していない**。現行 floor は固有値の**比**に対する制約で次元非依存のため、10 次元で 2 方向が支配的なら残り 8 方向は floor 値に張り付いたままランク欠損が維持される。

**なぜ縮退するか**: MC-ESO は毎世代 C_pop を**その C_pop から生成した集団**から推定し直すので、推定→生成→推定が閉ループになり低次元部分空間へ収束する（統計的に既知の退化）。CMA-ES がこれを免れているのは共分散を世代間累積して**持続的な分布オブジェクト**として保つためで、[mceso.md](mceso.md) の差別化「瞬間共分散・履歴累積なし」と表裏一体の弱点。

**この診断はルーター失効と同じ根**: ルーターの 3 シグナルは全て C_pop 由来なので、縮退した C_pop からは軸整列も conditioning も読めない（前節の測定と整合）。

### 対策案: 単位行列方向への次元比例シュリンク → REJECT

累積を導入せずランク下限を保証する案。正規化スペクトルに `ev ← (1−r)·ev + r` を適用（比でなく**どの方向にも最低 r のエネルギー**を保証＝次元を意識した制約）。縮退は高次元固有（ランク欠損は dim2 で 25%、d10 で 80%）なので `r = cov_shrink × (1 − 2/dim)` とし **dim2 で厳密に 0＝bit-identical**。

**機構チェックは 3 基準すべて合格**（dim10）: ① F01-Sphere の eff_used 4.22e3 → **53.8**（r=0.1）→ 11.3（r=0.3）② 標本分布の erank 1.98 → 2.63 → 5.04 ③ 悪条件の谷方向 cos を維持（F08 は 0.914→**0.944** と改善、r=0.3 では 0.568 に崩れるので 0.1 が上限）。**機構が選んだ 0.1 が d10 の SR 最適とも一致**（+2.92pt, n=10）。

**しかし n=20 の全次元検証で汎化せず REJECT**:

| dim | base | shrink | Δ | Wilcoxon 変種勝ち/base 勝ち |
|---|---|---|---|---|
| d3 | 73.75 | 67.29 | **−6.46** | 0 / 3（F12/F13/F14）|
| d5 | 25.62 | 22.29 | **−3.33** | 2 / 3 |
| d10 | 13.12 | **15.83** | **+2.71** | 4 / 2 |
| d20 | 12.29 | 9.17 | **−3.12** | 1 / 3（**F02 100→20**）|

d10 の効果は n=10 の +2.92 が n=20 で +2.71 と再現しており本物だが、**d10 以外は全て悪化**。d3 の実効 r は 0.042 と既に小さいのに −6.46pt なので値の調整では解けない。**単位行列方向への縮小は「谷追従」と「ランク」を交換する操作で、交換点が次元ごとに違う**のが本質。d10 だけに効く設定は 1 次元への過適合なので不採用。検証ログ: `results/20260824_223229_dshrink_d10_quick/`、`..._230104_dshrink_d3_quick/`、`..._232018_dshrink_d5_quick/`、`..._234555_dshrink_d20_quick/`。flat 版（次元スケールなし）の dim2 ゲートは 93.54 → 92.08（F18 −20 有意）で先に失格: `results/20260824_213519_shrink_d2_quick/`。

### 今後この問題に取り組むときの制約（測定から確定した事実）

1. **谷方向は掴めている**（悪条件で cos 0.87〜0.91）。方向推定を改善する方向の対策は筋が悪い（伝播鎖メモリが失敗したのもこれが理由）。
2. **動ける方向数が足りない**（erank 2/10）。ただし**等方性を足す形の対策は谷追従とトレードオフになり、次元横断で成立しない**（本節で実証）。
3. → 残る道は「**ランクを保ちながら谷追従も失わない**」機構。等方ノイズの加算でも比 floor でもない形が要る。候補: 縮退の原因である閉ループ自体を断つ（例: 子個体の生成元を集団分布でなく**独立に維持される少数の基底**にする、rank-1 更新を σ 適応とだけ結合する等）。いずれも「累積なし」の主張との整合を先に検討すること。

---

## 高次元の 2 番目の失敗モード — σ が drilling 手前で固定（2026-08-25, 診断確定 / 対策 2 件 REJECT）

前節（集団の自己縮退）の対策が全て失敗したのを受け、「谷方向は掴めている（cos 0.87〜0.91）のに何が律速か」を σ 側で測った（`scratchpad/sigma_equilibrium.py`）。

**σ 制御則の平衡点は次元非依存**。改善世代で ×`sigma_up`、非改善で ×`sigma_down` なので σ は

    s* = ln(1/sigma_down) / (ln(sigma_up) + ln(1/sigma_down)) = 0.350

の改善率で平衡する。**この値は次元にも問題にも依存しない**。一方、実測の改善率は次元とともに急上昇する:

| dim | 実測改善率 | σ_min/span | drill% | med_f |
|---|---|---|---|---|
| 2（全 8 関数）| **0.033〜0.060** | 1.00e-06 | 41〜60% | **0.0** |
| 10 F08-Rosenbrock | **0.342** | 6.42e-03 | **0.0%** | 1.2e+00 |
| 10 F09-RosenbrockRot | **0.335** | 6.06e-03 | **0.0%** | 1.5e+00 |
| 10 F10-EllipsoidalRot | 0.276 | 1.33e-03 | 3.4% | 1.6e+00 |
| 20 F10-EllipsoidalRot | 0.286 | 1.96e-03 | **0.0%** | 1.1e+02 |

**dim2 は平衡点から十分離れている**ので σ は自由に floor(1e-6) まで縮み全関数 med_f=0.0。**dim10 では改善率が平衡点に張り付き**、s* に最も近い F08(0.342)/F09(0.335) がちょうど **drill%=0.0・σ が 5e-3〜5e-2 で停止・med_f≈1**。予算を増やしても解決しない構造的停止である。

**高次元の失敗は 2 系統ある**（処方が正反対）:
- **系統 A: σ が drilling 手前で固定**（F08/F09/F10）→ σ をより縮めれば改善
- **系統 B: σ は floor に達したが誤った場所へ早期収束**（F12: σ_min=1e-6, drill 33.5% なのに med_f 0.87）→ σ をより縮めると**悪化**

同じ「高次元で SR が出ない」という症状の下に正反対の原因が混在しており、一律の補正がどれも中途半端だったのはこれで説明できる。

### 対策1: 近距離空気感染（drilling 中も精度スケールで air を残す）→ REJECT

3 チャネルのうち **air だけが full-rank の生成源**（close は C_pop、droplet は集団差分なので集団の張る部分空間に閉じる）だが、`air_ratio_eff=0` で **drilling 中に切られている**＝精度局面に full-rank 供給が皆無、という構造的欠陥に対する対策。エアロゾル感染は近距離でも起こるので疫学的にも自然。`air_drill_ratio`（drilling 中の air 割合）＋ σ_air = σ（増幅なし）で実装。

**機構チェックで不合格**（`scratchpad/aird_mech.py`, d10, drilling 世代のみ集計）: 集団の実効ランクは F12 1.53→2.01 / F02 1.26→1.55 と**わずかしか回復せず**、F01 はむしろ低下（3.34→2.91）。**決定的なのは対象関数に届かないこと** — F08 は drill%=0.0、F10 は 3.4% なので、drilling 中にしか作動しないこの機構は**構造的に無関係**。

→ 併せて分かったこと: **ランク低下は選択が強制している**。full-rank の子を供給しても悪条件の谷では等方ステップがほぼ棄却され集団に入らない。集団は「採択された点の集合」なので、分布側をいじる対策（伝播鎖・shrink・本案）が軒並み効かなかったのはこれが根本理由。

### 対策2: `sigma_up` の次元スケール（平衡点そのものを上げる）→ REJECT

`sigma_up` **だけ**を `(2/dim)**scale` 乗すると s* が次元とともに上がる（dim2: 0.350 不変＝**bit-identical**、dim3: 0.447、dim10: **0.729**、dim20: 0.843）。**両方の倍率を鈍らせる旧 `sigma_adapt_dim_scale` は s* を変えず速度だけ変える**ので、d10 +3.54/d3 −2.29 という中途半端さはこれで説明がつく。

**機構チェックは対象関数で決定的に成功**（d10, scale=0.5）: F08-Rosenbrock が **drill% 0.0 → 16.9 / σ_min 6.42e-3 → 8.82e-6 / med_f 1.2 → 7.2e-5（4 桁改善）**、F09 med_f 1.5→0.21、F02 は 2.8e-14 → **完全な 0.0**。ただし系統 B は予想どおり悪化（F12 0.87→5.1、F10 1.6→11）。

**SR スクリーニング（d10, n=10, 全 24 関数, base 13.3）**: scale 0.3/0.5 が **15.8（+2.5）**、0.7 が 15.0、1.0 は **10.8（−2.5、行き過ぎ）**。

**dim3 ゲートで失格**（n=20, 全 24 関数）: base 73.75 → up03 **70.00（−3.75）** / up05 **71.67（−2.08）**。落ちるのは F18 50→15 / F15 90→75 / F13 75→50 / F23 65→50 と**多峰・deceptive・ridge 系**で、悪条件単峰ではない。検証ログ: `results/20260825_*_upsc_d3_quick/`。

### 4 候補に共通する署名（重要）

| 候補 | d10 | dim3 | dim3 で落ちる関数 |
|---|---|---|---|
| `sigma_adapt_dim_scale`（速度ダンピング）| +3.54 | −2.29 | F19/F23/F06/F20 |
| `cov_shrink`（ランク保証）| +2.71 | −6.46 | F18/F14/F12 |
| 伝播鎖メモリ | +2.5 | (dim2 で失格) | F18/F04 |
| `sigma_up_dim_scale`（平衡点）| +2.5 | −3.75 | F18/F15/F13/F23 |

**4 件すべてが「d10 で +2.5〜+3.5、dim3 で −2〜−6.5、落ちるのは多峰系」という同一の署名**を示す。高次元は「より速い σ 収縮／より広い探索部分空間」を要求し、低次元は「探索と脱出の維持」を要求する。両者は同一パラメータの逆方向なので、**単調な次元補間ではどちらかを必ず損なう**。現行デフォルトは dim2/3 に対する局所最適であり、dim≥5 は逆向きの設定を要求する、というのが 4 回の実験から得られた結論。

**今後の選択肢**（研究方針の判断が要る）:
1. **次元条件付き設定を許容する** — CMA-ES 自身が c1/cμ/cσ/dσ/λ を全て次元の関数として定義しており、次元依存の定数は本来この分野の標準。「単調補間」ではなく**レジーム切替**（例: n_pop が 20 から増える dim>5 を境に別設定）なら 4 候補のいずれも成立しうる。ただし切替点の正当化が要る。
2. **適用範囲を明示して低次元特化手法として位置づける** — dim2 で 10 手法中 1 位という強みは実測で確立している。
3. 系統 B（早期収束）を先に潰す — σ が floor に達したのに解に届かない局面は本来 spillover の守備範囲なので、「なぜ spillover が救えていないか」の診断から入る。

---

## σ-pinning 検出器（2026-08-25〜26, 検証完了 → 採用保留）

前節で「高次元向けの補正は 4 件すべてが d10 で +2.5〜+3.5 / dim3 で −2〜−6.5」という同一署名で失敗したのを受け、ユーザ判断で**次元条件付き設定**（CMA-ES 自身が c1/cμ/cσ/dσ/λ を全て次元の関数として定義しており、次元依存定数はこの分野の標準）を採る方針とした。ただし**次元で閾値を切ると恣意性が残る**ため、**病理そのものを実行時に検出**する形にした。

**欠陥の同定（解析）**: σ 制御則の平衡改善率

    s* = ln(1/sigma_down) / (ln(sigma_up) + ln(1/sigma_down)) = 0.350

は **σ_up/σ_down だけで決まり、次元にも問題にも依存しない**。一方、実測の改善率は dim2 で 0.033–0.060、dim3 で 0.046–0.130、dim5 で 0.107–0.377、dim10 で 0.133–0.342 と次元とともに上昇し、**d10 で平衡点に到達**する。s* に最も近い F08(0.342)/F09(0.335) がちょうど drill%=0.0・σ が 5e-3 span で停止・med_f≈1。

**信号選択の失敗と修正（重要な教訓）**: 最初に「改善率の EMA が s* の 0.7 倍を超えたら pinned」とする検出器を実装したが、**弁別に完全に失敗**した（d10 で fire% 43–99%、dim3 でも 2.9–99.7% と無差別発火し、多峰系を毀損: dim3 F15 med_f 0.0→6.5e-6、F12 8.2e-7→7.4e-4）。原因は EMA が**探索初期の高い改善率**に支配されること。dim3-F12 が改善率 0.395（s* 超え）なのに σ_min=1e-6 に到達していた測定結果が、この信号の非弁別性を既に示していた（設計時に読み落とした）。

→ **病理を直接数える信号に変更**: 「σ が予算の `sigma_pin_evals_frac` にわたり drilling 閾値に到達できていない」。探索初期は正当に非 drilling なので**予算比**で測る（固定世代数では全次元で初期発火する）。

**機構チェック（`scratchpad/pin_mech.py`, 予算比 0.3, n=3）— 3 基準すべて合格**:

| | fire% | drill% off→on | med_f off→on |
|---|---|---|---|
| d10 F08-Rosenbrock | **68.2** | 0.0 → 0.6 | **1.2 → 0.13** |
| d10 F09-RosenbrockRot | **73.5** | 0.0 → 0.2 | **1.5 → 0.85** |
| d10 F06-AttractiveSector | 16.4 | 24.7 → 34.4 | **1.8e-8 → 1.9e-10** |
| d10 F13/F15/F16/F19/F20/F21/F22/F23/F24 | **0.0** | 不変 | **完全に不変** |
| dim3 全 24 関数 | 0.0–12.3（多峰系はほぼ 0）| ほぼ不変 | ほぼ不変 |

**過去 4 候補が壊した多峰系に一切触れない**のが決定的な違い。次元の単調関数として補正する限り病理のない低次元関数にも必ず作用が及ぶが、検出器方式はその構造を断つ。

**SR スクリーニング（d10, n=10, 全 24 関数, base 13.3）**: 予算比 20/30/40% がいずれも 15.4（+2.08）と**閾値に鈍感**、damp を 0.25→0.5 に緩めた `pin30d5` が **16.2（+2.92）**・SR@1e-7 18.3 で最良。

**判定次元のゲート（n=20, 全 24 関数）— 両方通過**:

| | SR@1e-10 base → pin30d5 | 差分のある関数 | Wilcoxon |
|---|---|---|---|
| dim2 | 93.54 → **93.33（−0.21）** | F04 −5 / F14 −5 / F17 **+5** | **有意差ゼロ** |
| dim3 | 73.75 → **73.33（−0.42）** | F12 −5 / F14 **+5** / F20 −5 | **有意差ゼロ** |

過去 4 候補の dim2/dim3 が −1.46 / −2.29 / −3.16 / −3.75 / −6.46 だったのに対し、**−0.21 / −0.42 は 1 run 未満の揺らぎ**。「dim2 で no-op なら dim3 で落ちる / dim2 で作用すれば dim2 で落ちる」というこれまでのジレンマを、初めて回避した。

**新規性の位置づけ**: CMA-ES は定数を**次元の関数**として定義するが、本機構は**実行時の状態の関数**として定義する。次元以外の原因（集団サイズ設定・予算・landscape）で同じ病理が起きても自動的に作動する。また s* を制御則から解析的に導き、その次元非依存性を欠陥として同定したうえでの対処なので、経験的なパラメータ調整ではない。LM-CMA 系の移植でもなく、MC-ESO の σ 制御則（SaVOA 流の乗法適応）に閉じた修正である。報酬ベース適応（却下された V2a の UCB）とも異なり、**制御則自身の統計量を測って制御則を補正する** 1/5 則と同系統の枠組み。

### 全次元検証の結果（n=20, `--all`, 予算 2500×d）

| dim | base | pin30d5 | Δ SR@1e-10 | Wilcoxon 変種勝ち/base 勝ち |
|---|---|---|---|---|
| 2 | 93.54 | 93.33 | −0.21 | 0 / 0 |
| 3 | 73.75 | 73.33 | −0.42 | 0 / 0 |
| 5 | 25.62 | 26.04 | +0.42 | 1 / 0（F06）|
| **10** | 13.12 | **15.62** | **+2.50** | **3 / 0**（F06/F08/F09）|
| 20 | 12.29 | 11.88 | −0.42 | 4 / **1**（勝 F06/F08/F09/F14、負 **F01-Sphere**）|

d10 は全精度階層で改善し**悪化関数ゼロ**（F06 0→35, F08 10→20, F22 5→15, F21 0→5）。低次元 3 つは有意差ゼロで誤差範囲。**5 次元平均 +0.37**。ただし d20 で F01-Sphere が 95→70（有意）となり primary を押し下げた。d20 の Sphere は drill% 23% と非 drilling 期間が長いが**順調に降下している最中**で、σ_up を鈍らせると初期の前進が遅れ予算 50000 では取り返せない ＝ **「drilling 未到達」と「行き詰まり」を区別できていない**。

### 絞り込み（停滞条件の追加）→ トレードオフが解消せず

`no_improve` は既に**「意味ある改善」（log f の傾き ≥ `log_slope_threshold`）でのみリセット**される＝そのまま「進捗が止まった」カウンタなので、発火条件に `no_improve ≥ sigma_pin_stagnant_frac × 停滞窓` を追加した。

- **d20-F01 の回帰は解消**: fire% 13.7→10.6、med_f 1.2e-12 → **1.6e-13**（改善に転じた）。F06 も 1.6e-3 → 2.7e-4。
- **しかし d10 の効果が半減**: SR@1e-10 16.2 → **14.6**（n=10）。F06 の発火が 16.4%→7.8% に減り med_f 1.9e-10 → 1.7e-7 と後退したため。F06 は d10 で SR 0→35 を稼いでいた関数。

→ **d20 の安全性と d10 の効果を交換しただけで両立しない**。再現用 kwargs: `sigma_pin_evals_frac=0.30, sigma_pin_damp=0.5`（＋絞り込み版は `sigma_pin_stagnant_frac=0.5`）。検証ログ: `results/20260825_*_pin_d{2,3,5,10,20}_quick/`。

---

## 潜伏期 = SEIR の E（2026-08-26）— REJECT

「既存手法の変化形ではなく、単純だが未導入のアイデアを」というユーザ指摘を受けた検討。**MC-ESO は疫学モデルとしては SI しか実装していない**（感受性者→感染者のみ）。潜伏期 E も回復・免疫 R も無い。感染症の多経路伝播を模すと主張しながら、感染症モデルの最も基本的な構成要素が欠けているという**新規性のギャップ**がある。

**仮説**: 測定された病理（集団の実効ランクが 2/10 に縮退）の起点は「**新しく生まれた最良個体が即座に親になり、次世代がその周辺に集中する**」こと＝推定→生成→推定の閉ループ。現実の感染症では新規感染者はすぐには他人にうつさない。`pop_age` は既に存在するが σ_i のスケーリングにしか使われず**選択には一切関与していない**ので、`incubation_gens` 世代を経ていない宿主を親選択から外すだけで実装できる（パラメータ 1 個、計算コスト 0、既定 0 で bit-identical）。

**機構チェックで仮説は否定された**（d10, n=3, `scratchpad/incubation_mech.py`）:

| 関数 | erank L=0 → L=2 | med_f L=0 → L=2 |
|---|---|---|
| F01-Sphere | 2.68 → 3.22 | 1.4e-14 → 1.4e-14 |
| F02-EllipsoidalSep | 1.36 → 1.51 | 2.8e-14 → 2.8e-14 |
| F08-Rosenbrock | 2.27 → 2.18 | **1.2 → 4.0e-02** |
| F10-EllipsoidalRot | 1.54 → 1.70 | 1.6 → 13（悪化）|
| F12-BentCigar | 2.14 → 2.14 | 0.87 → 6.8（悪化）|

**ランクがほとんど動かない**（全ランク 10 に対し依然 2〜3）。**SR スクリーニング**（d10, n=10, 全 24 関数, base 13.3）も L=1 で 14.2（+0.83）、L=2 で 13.3（±0）、**L=3 で 11.2（−2.08）**と、機序・性能ともに支持されない。

**なぜ効かないか（既存の知見と整合）**: 近距離空気感染の検証で確定した「**ランク低下は選択が強制している**」がここでも効く。親プールを古く分散させても、悪条件の谷では**採択される子が谷内に限られる**ため集団のランクは戻らない。**親の多様性は律速ではない**。

**この否定的結果の価値**: 疫学固有の新機構（既存 ES に相当物がない）を試して効かなかったことで、「MC-ESO の高次元問題は疫学アナロジーの拡張では解けず、律速は選択規則にある」という切り分けが得られた。診断が一貫して指しているのは**空間構造を持たない μ+λ greedy の採択規則**だが、これは多解探索の文脈で crowding として試され SR@1e-10 を崩して撤回済み（`results/20260616_145202_endemic_v3`）。同じ形では通らない。

---

## 高次元への 7 候補 — 総括（2026-08-26 時点）

| # | 候補 | 系統 | d10 | 低次元 | 判定 |
|---|---|---|---|---|---|
| 1 | `sigma_adapt_dim_scale`（σ 速度ダンピング）| σ 制御 | +3.54 | dim3 −2.29 | REJECT |
| 2 | 伝播鎖メモリ（LM-CMA 風の方向記憶）| 分布 | +2.5 | dim2 −3.16 | REJECT |
| 3 | `cov_shrink`（ランク保証）| 分布 | +2.71 | dim3 −6.46 / d20 −3.12 | REJECT |
| 4 | 近距離空気感染（drilling 中の full-rank 供給）| チャネル | — | — | REJECT（対象関数に構造的に届かない）|
| 5 | `sigma_up_dim_scale`（平衡点の次元スケール）| σ 制御 | +2.5 | dim3 −3.75 | REJECT |
| 6 | σ-pinning 検出器 | σ 制御 | **+2.50**（3勝0敗）| dim2 −0.21 / dim3 −0.42 | **保留**（d20 に有意回帰 1）|
| 7 | 潜伏期（SEIR の E）| 選択 | +0.83 | — | REJECT（機序否定）|

**持続的な成果は対策ではなく診断側にある**: (a) σ 制御則の平衡改善率 s*=0.350 が次元非依存であること、(b) 集団の自己縮退とそれを**選択が強制している**こと、(c) ルーターの 3 シグナルが高次元で情報を失うこと、(d) 真の条件数 1.00 の Sphere で実効条件数 4.2e3 という「アルゴリズムが自作する異方性」。いずれも今後の設計を制約する測定済みの事実である。
