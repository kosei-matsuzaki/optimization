# e179 事前登録 —— 10 次元 BBOB-24 の現在地（キュー 1）

**書いた時刻**: 2026-09-28 06:37 UTC。**この時点で 10D の数値は 1 つも見ていない**
（run は 06:34:32 に起動済みだが、`summary.csv` を開くのは この文書を commit した後。
起動確認で見たのは shard 1 の表ヘッダ行だけで、数値行は 1 行も出ていなかった）。

## 問い（キューの原文どおり）

**10 次元 BBOB-24（F01-F24）で MC-ESO ＋ 比較 5 手法の現在地を、その177 の 2D・その178 の 5D と
同じ環境・同じ手法集合で測る。**
旧環境の記録（[history.md](../../../docs/history.md):917、10 手法・予算 2500×d）は
**D=10 で MC-ESO 34.4 対 IPOP 47.1（−12.7pt）／ 10 手法中 4 位**としている。
**同じ手法集合・同じ環境での 10D 比較はこの研究に 1 度も無い。**

## 測るもの

`./run.sh quick --dim 10 --max-evals 25000 --n-runs 20 --methods "MC-ESO,CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,DE,L-SHADE"`
＝ **10 次元 BBOB-24（F01-F24）× 6 手法 × 20 run × 25000 評価**
（予算は [experiments.md](../../../docs/experiments.md) の目安 `2500×D` ＝ 方針欄 2026-09-27 の指定と一致）。
**`core/` は 1 行も触らない。MC-ESO の既定は 1 つも変えない。腕は作らない。Custom（C01-C11）と CI（`./run.sh trigger`）は使わない。**

### 実行上の逸脱 1 件（40 分枠に入れるため。値は変えていない）

**24 関数を 6 つの shard（4 関数ずつ、6 おきの交互配分）に分け、6 プロセスを並列に走らせた**（コアは 4）。
- shard 1: F01-Sphere / F07-StepEllipsoidal / F13-SharpRidge / F19-GriewankRosenbrock
- shard 2: F02-EllipsoidalSep / F08-Rosenbrock / F14-DiffPowers / F20-Schwefel
- shard 3: F03-RastriginSep / F09-RosenbrockRot / F15-RastriginRot / F21-Gallagher101
- shard 4: F04-BucheRastrigin / F10-EllipsoidalRot / F16-Weierstrass / F22-Gallagher21
- shard 5: F05-LinearSlope / F11-Discus / F17-SchafferF7 / F23-Katsuura
- shard 6: F06-AttractiveSector / F12-BentCigar / F18-SchafferF7ill / F24-LunacekRastrigin

**この分割は RNG 同一である** —— `core/runner.py:54` が各 run の seed を `seed=i*100`（run 番号だけ）で決めており、
**どの関数が同じプロセスに同居するかは結果を変えない**（`np.random.seed` を触るのは NMMSO だけで、この回は回していない）。
`_run_dim` は関数ごとに `summary.csv` / `wilcoxon.csv` へ追記するので、**6 本の CSV の連結は 1 本で回した場合と行集合として等価**（順序だけ違う）。
BLAS の oversubscription を避けるため `OMP_NUM_THREADS=1` 等を立てた。
**n_runs・関数数・手法数・予算はどれも削っていない** —— 削ったのは壁時計だけである。
**その178 との違いは shard 数だけ**（4 → 6）。理由: その178 は 4 shard で 22.7 分だったが 10D は予算 2 倍で 45 分前後の見込みであり、
**4 コアに 6 本を載せると尾の負荷不均衡が縮む**（総 CPU 仕事量は同じで、いちばん重い shard が壁時計を決める形を避ける）。

### 環境について 1 行

**`pynmmso` はこの image の Python でビルドできない**（その178 と同じ。`setuptools` の `install_layout` で落ちる）。
**`/tmp/pystub/pynmmso/` に import だけ通る最小 stub を置いた**（`Nmmso` を呼ぶと `RuntimeError`）。
**この回は NMMSO を 1 run も回さないので、測定値には一切触れない。**

## 出すもの（その177・その178 と同じ形。集計は `analysis/single/e178/analyze.py` をそのまま使う）

1. 6 手法 × SR 梯子（1e-2 / 1e-4 / 1e-7 / **1e-10 ＝ 主指標**）＋ `evals_succ_mean` の全体表
2. 関数別 SR@1e-10 と `evals_succ_mean` の 24 行表
3. Wilcoxon（reference = MC-ESO、両側 α=0.05、A12 併記）
4. **BBOB 公式 5 群ごとの内訳**（g1 F01-F05 separable / g2 F06-F09 low-cond / g3 F10-F14 high-cond unimodal / g4 F15-F19 multimodal-global / g5 F20-F24 multimodal-weak）
5. **「仮想ベスト手法」の再集計** —— 5 手法の関数別ベストを取る手法と MC-ESO の差、および 6 手法の関数別ベスト

## 反証条件（キューの原文。数値を見る前に固定する）

- **(a) SR@1e-10 が 34.4% 前後で再現しないなら**、旧環境の高次元表は絶対値としても方向としても使えず、
  **[status.md](../../../docs/status.md) の「ゴールとの距離」(ii) を消すことになる。**
  **判定線を先に決める**（後付けを防ぐため）: **実測 SR@1e-10 が 34.4 ± 5.0pt の外（< 29.4% または > 39.4%）なら「再現しない」と書く。**
  **その178 の 5D の乖離（記録 39.0 対 実測 43.12 ＝ +4.12）がこの帯のちょうど内側なので、同じ幅で 10D を読む。**
- **(b) MC-ESO が IPOP に並ぶ・上回るなら**、テーマの伸びしろは 5D に限られると確定する。
  **判定線**: **SR@1e-10 の差が ≥ −1.0pt（＝ ほぼ並ぶ）なら発火。**
- **(c)【その178 が足した】5D の赤字の 41% は単峰・条件数の 6 関数（F07 / F08 / F09 / F12 / F13 / F14）から来ていた。**
  **10D で g3（高条件数・単峰、F10-F14）の CMA-ES − MC-ESO 差が縮むなら**（5D は CMA-ES 84.00 対 MC-ESO 58.00 ＝ **+26.0pt**）、
  あれは 5D 固有の遷移であって次元の効果ではない。**判定線**: **差が +13.0pt 未満（半分未満）に縮んだら「5D 固有」と書く。**

**追加で 2 つ先に決めておく**（後付けの読みを防ぐため）:
- **(d) その178 は「旧環境の表は<u>比較手法側だけ</u>再現し MC-ESO だけ +4.12 動く」という非対称を出した**
  （CMA-ES −1.07 / IPOP +1.29 / MC-ESO +4.12）。**10D で同じ非対称（比較手法の乖離が ±2pt 以内で、MC-ESO だけ +3pt 以上）が出れば、
  「環境差」では説明できず<u>MC-ESO の版の違い</u>が濃くなる ＝ キュー 2（pin の 1.46pt）の読みに効く。**
  **逆に 10D で MC-ESO の乖離が ±2pt 以内なら、5D の +4.12 は次元固有のゆらぎとして扱う。**
- **(e) `evals_succ_mean` の順位。** 2D は 2 位（677.7）、5D は 5 位（5240.9）だった。
  **10D でも 5 位以下なら「速さでも負ける」は次元をまたぐ性質**と書ける。**2 位以内に戻るなら 5D 固有。**

## やらないこと

- **3D / 20D は 1 run も回さない。** **2D / 5D の測り直しもしない**（その177・その178 の値を対照に使う）。
- **失敗 run の機序の切り分けはしない**（キュー 3）。この回が出すのは現在地の表だけ。
- **改善案を出さない・実装しない**（方針欄の「どの関数・次元で負けているかが分かるまで、新しい機構を作らない」）。
- **キューの並べ替え・ゴールの変更・`status.md` の編集はしない。**
