# 最適化手法の比較実験: MC-ESO と既存手法

感染症の流行（epidemic spread）を着想とした独自手法 **MC-ESO — Multi-Channel Epidemic Spread Optimizer** を、標準的な既存最適化手法と BBOB 等のベンチマークで比較する研究プロジェクト。

**核心の主張**: 既存メタヒューリスティクスはいずれも **単一の再生メカニズム** を持つ（DE = 差分変異, ES = ガウス変異, PSO = 速度ベクトル, GA = 交叉）。一方、現実の感染症は **複数の伝染経路** — 接触感染・飛沫感染・空気感染 — が並行して働く。MC-ESO はこれを忠実に模し、各世代で 3 つの定性的に異なる伝染チャネルを混合する。

---

## ドキュメント

詳細は `docs/` 配下に領域別で分割している。

| ドキュメント | 内容 |
|---|---|
| **[docs/mceso.md](docs/mceso.md)** | MC-ESO のコンセプト・3 チャネル / 3 機構・最新アーキテクチャ・パラメータ・既存手法との差別化 |
| **[docs/baselines.md](docs/baselines.md)** | 比較対象の既存手法（CMA-ES / IPOP・BIPOP / PSO / DE / L-SHADE / SaVOA）の実装詳細 |
| **[docs/experiments.md](docs/experiments.md)** | ディレクトリ構造・実行方法（run.sh）・実験条件・ベンチマーク関数・評価基準・結果の見方 |
| **[docs/history.md](docs/history.md)** | これまで試した工夫・フラグ・ablation 記録（採用 / 不採用とその理由） |
| **[docs/web.md](docs/web.md)** | Results UI（Flask アプリの構成・アーキテクチャ・ルート / API） |

---

## クイックスタート

```bash
./run.sh quick --all      # 手法の検証・評価（標準: n_runs=20 / max_evals=5000 / 全関数）
./run.sh ui               # Results UI を起動 → http://localhost:8080

# 補助実験（裏で回す。手法評価には参照しない）
./run.sh trigger          # n=100 ワークフローを GitHub Actions でトリガー
./run.sh download         # 完了後に結果をローカルへ保存
```

手法の検証・評価は **quick の n_runs=20 / max_evals=5000 / `--all`** で統一する。GitHub Actions の n=100 実験は補助的なもので評価には使わない。実行コマンドの一覧・実験条件・評価基準は [docs/experiments.md](docs/experiments.md) を参照。
