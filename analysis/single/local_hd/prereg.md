# local_hd 事前登録 —— 高次元の腕 3 種（ローカルセッション、2026-09-29）

**書いた時刻**: 2026-09-29 15:55 JST。**24 関数 × 20 run の集計はまだ 1 行も見ていない。**
研究ループの その183 以降とは別の、ユーザー指示によるローカルでの検証（`local/hd-cov-sigma` ブランチ、commit `c8f1be0`）。

## 先に見た数値（後付けを防ぐため全部開示）
- d10 の既定 MC-ESO のトレース（F07 seed 0/100/200、F12 seed 0/100。seed = i×100、25000 評価）。
  - **F07**: 集団 40 体すべてが同じ f（uniqF=1）の平坦面に乗り、σ が 1e-6·span の床まで潰れる。spillover 23〜28 回、毎回同じ形で凍る。
  - **F12**: 学習 C の cond が seed 100 で 2.3e6 まで育つのは予算の最後（best 5.5e-06）。seed 0 は cond ~2e3 止まりで best 40。
- **`flat` の 1 seed プローブ（F07 d10）**: seed 0 で best 0.556 → 0.023、seed 100 は 0.892 → 0.892（不動）。**腕が有利に見える方向を 1 本先に見た。**
- **環境の注意**: ローカルの `.venv` に `pynmmso` を入れ直した際、`numpy` が 2.4.6 になった。**その182 の prereg に書かれた F07 seed 0 の 9.181e-01 は、ここでは 5.563e-01 だった**（プローブの seed 付けが違う可能性もある）。**base の行をその182 の `summary_ccmu_10d.csv` と突き合わせ、一致しなければ比較は同一 run 内に限る。**

## 腕（既定は 1 つも変えていない）
| 腕 | 変更 | 狙い | dim2 |
|---|---|---|---|
| `npop8` | `n_pop_dim_mult` 4 → 8（d10 で n_pop 40 → 80） | 1 世代の子 ＝ 学習 C の候補を 2 倍にする | bit 一致（n_pop = 20 のまま） |
| `npop8_ccmu50` | 上 ＋ `cc_mu_frac` 0.5 | 本数を 2 方向から増やす | bit 一致 |
| `ccmu50_lr10` | `cc_mu_frac` 0.5 ＋ `cc_learning_rate` 0.05 → 0.10 | 本数が増えたので学習率を上げられるか（F12 の学習速度） | bit 一致（gate=0） |
| `flat` | `sigma_flat_expand`（CMA-ES の flat-fitness 規則。同値が半数以上 → σ を広げる。drilling 外のみ） | F07 の平坦面での σ 潰れ | **変わりうる** → 2D を回す |

## 実行
- **10D**: `./run.sh quick --all --dim 10 --max-evals 25000 --n-runs 20 --methods "MC-ESO,npop8,npop8_ccmu50,ccmu50_lr10,flat"`（24 関数を 8 shard に分割）。
  CMA-ES / IPOP / `ccmu50` は その182 の CSV を使う（base が一致した場合のみ）。
- **2D**: `./run.sh quick --all --methods "MC-ESO,flat"`（n=20 / 5000）。

## 反証条件（ここで固定）
各腕について 3 指標（SR@1e-10 と梯子 / `evals_succ_mean` は共通集合で / Wilcoxon ＋ A12、ref = MC-ESO）を揃える。
- **(a) 採用候補の条件**: 10D の SR@1e-10 が base を上回り、**Wilcoxon で base が有意に優る関数が 0**、かつ **2D の SR@1e-10 を下げない**（2D bit 一致の腕は自動的に満たす）。
- **(b) 撤退条件（status.md 判断 (1)）への寄与**: **F07 または F12 の SR@1e-10 が 20 run 中 3 本（15%）以上**。1 本（5%）は n=20 の分解能なので数えない。
- **(c) 仮説の反証**:
  - `flat` で F07 の SR@1e-10 も median `best_f` も動かない → **「F07 は σ 潰れが原因」は外れ**。
  - `npop8` と `ccmu50_lr10` のどちらでも F12 の median `best_f` が 1 桁以上下がらない → **「F12 は学習速度が原因」は外れ**。
- **`npop8` の代償**: 世代数が半分になる。**F01/F02 など単峰の易しい関数で `evals_succ_mean` が 20% 以上増えたら**、本数の利得を世代数で払っていると読む。
