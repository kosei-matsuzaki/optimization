e83 — 問い 1: N20-CF4-20D の「MC-ESO だけが 1e-5 に届く」は本物か、12 seed の検出力の産物か

設計（数値を見る前に固定した）
- 主測定: N20-CF4-20D（CF4 系、D=20、K=8、正規予算 4e5）を MC-ESO / NMMSO の 2 手法で
  **seed 12-29 の 18 seed 追加**し、その82 の seed 0-11 と **seed 番号で連結して 30 seed** にする。
  seed 番号は niching_baseline.py の --seed-offset により大域一意（seed i = optimiser seed i*100）
  なので、e82 の CSV とそのまま縦に繋げば paired のままになる。
- 採点: --report-rule current（e82 / e73 と同じ）。判定水準は PR@1e-3 / 1e-5（その28 の運用規則）。
  PR@1e-1 は被覆の診断値としてのみ読む。
- 分析: scripts/fullbudget_rank.py --pair MC-ESO,NMMSO（新規スクリプトは書かない）。
  paired Wilcoxon（両側）・w/t/l・A12 を 5 水準すべてで印字する。

事前登録した棄却条件（問い 1 の本文どおり、数値を見る前に書いた）
  30 seed で PR@1e-5 の paired w/t/l が有意でなくなる（p > 0.05）か、
  MC-ESO の PR@1e-5 平均が 0.05 を割る
  ＝ 12 seed の勝ちは検出力の産物なので、CF4 側に生きたセルは無いと書いて案 (A) の材料から落とす。

  逆に棄却されない（p <= 0.05 かつ平均 >= 0.05）なら、本テーマで「既存手法より多く見つけた」と
  書ける形の候補が 1 つ生き残る。ただし絶対値は 8 解中 0.6 解程度で、勝ちの中身は薄い。

副次（枠が余ったときだけ。主判定には使わない）
  「NMMSO に勝った」ではなく「20D で 1e-5 に届く唯一の手法」と書けるかは他手法次第なので、
  残り 4 手法（NCDE / r3pso / DE / NM-Restart）を **e82 と同じ seed 0-11** で回して
  PR@1e-5 が 0 かを見る。1 seed のコスト probe を先に取ってから本数を決める（その82 の手順）。
  NM-Restart の N06/N08 行は無効（その28）だが、それは Shubert 固有の restart 回数 1 の話で、
  ここでは別関数なので落とさない。ただし restart 回数を報告に添える。

結果（数値は acceptance_topology.md のその83 節が正）
  棄却条件は不成立（MC-ESO は 30 seed でも NMMSO に PR@1e-5 で勝つ、p=0.0047）。
  しかし副次で NCDE が同じ関数の PR@1e-5 で MC-ESO を有意に上回った（30 seed、p=0.0074）
  ＝ このセルは「既存手法より多く見つけた」と書ける場所ではなかった。
  その82 が MC-ESO 対 NMMSO の 2 手法しか回していなかったことが原因。

ファイル（シャードは連結後に削除した。以下が残す集計）
  N20-CF4-20D_30seed.csv                   : MC-ESO / NMMSO 各 30 seed（e82 の seed 0-11 ＋ 本回の 12-29）
  N20-CF4-20D_MCESO_NMMSO_NCDE_30seed.csv  : 上に NCDE 30 seed を足したもの（主結論はこれ）
  N20-CF4-20D_6methods_12seed.csv          : 6 手法 × seed 0-11（他手法が 1e-5 に届くかの判定に使った）
  shard_timing.txt                         : シャードごとの壁時計
  rank30.txt / rank30_ncde.txt / rank_others.txt / perseed30.txt : fullbudget_rank.py と per-seed の出力
