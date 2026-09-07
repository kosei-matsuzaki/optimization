# e78 — `basin_reset` の BBOB-24 dim2 gate

問い 1（`docs/research_loop.md`）の測定。結論の全文は
[`docs/acceptance_topology.md`](../../../docs/acceptance_topology.md) の該当節にある。

| ファイル | 中身 |
|---|---|
| `gate5k_c{0..3}.csv` | 段階 1。BBOB-24 dim2 × 20 seed × 5000 評価 = **480 対**（関数を 4 分割） |
| `gate20k_c{0..3}.csv` | 同上、**20000 評価** = 480 対 |
| `ext5k_lose.csv` / `ext5k_win.csv` | 段階 2。段階 1 が名指しした 4 関数だけを **新しい seed 20-59** で 40 seed |
| `analyze.py` | 段階 1 の集計（SR@1e-10 の対比較・`evals_succ_mean`・関数別の列挙） |
| `analyze_ext.py` | 段階 2 の集計（選抜と検定の seed を分ける） |
| `analysis_5k.txt` / `analysis_20k.txt` / `analysis_ext.txt` | それぞれの出力 |
| `log*.txt` | 実行ログ（`gate_power.py --compare` の 1 行表） |

腕の実体は `core/optimizers/mceso_basin_reset.py`（その77 の
`analysis/hm/e77/basin_reset.py` とクラス本体がバイト一致。`inspect.getsource` で確認）。
**`core/optimizers/mceso.py` の既定値は 1 行も変えていない。**

再現:

```
python3 scripts/gate_power.py --funcs <24 関数> --dim 2 --seeds 20 --evals 5000 \
    --compare 0 --basin-reset --csv analysis/hm/e78/gate5k_c0.csv
python3 analysis/hm/e78/analyze.py analysis/hm/e78/gate5k_c*.csv
```
