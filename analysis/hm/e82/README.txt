e82 — 問い 1: F18 の「高次元では被覆だけは取れる」は CF3 固有か、手法の性質か（CF4 系 4 関数）

設計（数値を見る前に固定した）
- 対象: N15-CF4-3D / N17-CF4-5D / N19-CF4-10D / N20-CF4-20D（CF4 系 4 関数、K=8、正規予算 4e5）
  ＝ e73 の CF3 断面（N14 3D / N16 5D / N18 10D）に対する CF4 の同次元断面。
  N19-CF4-10D が N18-CF3-10D の直接の対（D を固定して CF3→CF4 だけを動かす）。
- 手法: MC-ESO（既定・無調律）と NMMSO の 2 手法、seed 0-11 の 12 seed、seed 番号で paired。
- 採点: --report-rule current（e73 と同じ）。判定水準は PR@1e-3 / 1e-5（その28 の運用規則）。
  PR@1e-1 は被覆の診断値としてのみ読み、順位の根拠にしない。
- 分析: scripts/fullbudget_rank.py --pair MC-ESO,NMMSO（新規スクリプトは書かない）。
  e73 の CSV でこのスクリプトが acceptance_topology の表を厳密に再現することを事前に確認済み。

事前登録した棄却条件（問い 1 の本文どおり）
  CF4 系で MC-ESO の PR@1e-1 が NMMSO を上回らない
  ＝「高次元では被覆だけは取れる」は CF3-10D 固有の偶然で、その73 は 1 関数の観察に留まる。
  そう出たら status.md の主張 2（その21 の二分法が手法間で予測を出す）は 1 例のままで、
  論文の芯にはできない。

読む順（成立/不成立の判定に使う量）
  (1) N19-CF4-10D の PR@1e-1 の paired w/t/l（MC-ESO 対 NMMSO）— これが主判定。
  (2) 各手法の「被覆 − 深さ」= PR@1e-1 − PR@1e-3 の差。その73 の署名は
      MC-ESO で大（1.000 − 0.194 = 0.806）、NMMSO で小（0.597 − 0.583 = 0.014）。
  (3) 3D/5D では e73 の CF3 と同じく両手法が同点に張り付くか（壁の再現）。

ファイル
  probe_*.csv / probe_timing.txt : 1 seed のコスト probe（4 関数 × 2 手法、4 並列で計 2.4 分）
  sh_<func>_<method>_<offset>.csv : 本測定のシャード（3 seed ずつ、32 シャードを xargs -P 4）
  shard_timing.txt : シャードごとの壁時計
