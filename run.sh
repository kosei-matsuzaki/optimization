#!/usr/bin/env bash
# run.sh — experiment management commands
#
# Usage:
#   ./run.sh trigger [--n-runs N] [--max-evals N]
#   ./run.sh download [RUN_ID]
#   ./run.sh quick [--n-runs N] [--max-evals N]
#   ./run.sh list
#   ./run.sh status [RUN_ID]

set -euo pipefail

WORKFLOW="Run Optimization"
RESULTS_ROOT="results"

# Prefer project-local .venv (uv) over system python3 for local commands.
if [[ -x ".venv/bin/python3" ]]; then
  PY=".venv/bin/python3"
else
  PY="python3"
fi

# ── trigger ──────────────────────────────────────────────────────────────────
cmd_trigger() {
  local n_runs=100 max_evals=5000
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --n-runs)    n_runs="$2";    shift 2 ;;
      --max-evals) max_evals="$2"; shift 2 ;;
      *) echo "Unknown option: $1"; exit 1 ;;
    esac
  done
  echo "Triggering workflow  n_runs=${n_runs}  max_evals=${max_evals} ..."
  gh workflow run "$WORKFLOW" --ref main \
    -f n_runs="$n_runs" \
    -f max_evals="$max_evals"
  echo "Triggered. Run './run.sh list' to check status."
}

# ── download ─────────────────────────────────────────────────────────────────
cmd_download() {
  local run_id="" label=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --label) label="$2"; shift 2 ;;
      *)       [[ -z "$run_id" ]] && run_id="$1"; shift ;;
    esac
  done
  if [[ -z "$run_id" ]]; then
    run_id=$(gh run list --workflow="$WORKFLOW" --status completed \
      --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)
    [[ -z "$run_id" ]] && { echo "No completed runs found."; exit 1; }
    echo "Latest completed run: ${run_id}"
  fi

  local suffix
  suffix="${label:-$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')}"
  # strip characters unsafe for directory names
  suffix=$(printf '%s' "$suffix" | tr -cd '[:alnum:]_-' | cut -c1-40)
  local dir="${RESULTS_ROOT}/$(date +%Y%m%d_%H%M%S)_${suffix}"
  echo "Downloading run ${run_id} → ${dir}/"
  local tmp; tmp=$(mktemp -d)
  gh run download "$run_id" -D "$tmp"
  mkdir -p "$dir"
  # artifact is named "results", its contents go directly into $dir
  mv "$tmp/results/"* "$dir/"
  rm -rf "$tmp"
  printf '{\n  "type": "workflow",\n  "created_at": "%s",\n  "commit": "%s",\n  "gh_run_id": "%s",\n  "status": "done"\n}\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" \
    "$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')" \
    "$run_id" > "$dir/result.json"
  echo "Saved to: ${dir}/"
}

# ── quick ─────────────────────────────────────────────────────────────────────
PID_FILE=".quick.pid"
DIR_FILE=".quick.dir"

cmd_quick() {
  local n_runs=20 max_evals=5000 label="" use_all=0 dim=2 methods="" with_custom=0 noise=""
  local pass_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --n-runs)    n_runs="$2";    pass_args+=("$1" "$2"); shift 2 ;;
      --max-evals) max_evals="$2"; pass_args+=("$1" "$2"); shift 2 ;;
      --funcs)     pass_args+=("$1" "$2"); shift 2 ;;
      --all)       use_all=1;      pass_args+=("$1"); shift ;;
      --custom)    with_custom=1;  pass_args+=("$1"); shift ;;
      --dim)       dim="$2";       pass_args+=("$1" "$2"); shift 2 ;;
      --methods)   methods="$2";   pass_args+=("$1" "$2"); shift 2 ;;
      --suite)     pass_args+=("$1" "$2"); shift 2 ;;
      --noise)     noise="$2";     pass_args+=("$1" "$2"); shift 2 ;;
      --label)     label="$2";     shift 2 ;;
      *)           pass_args+=("$1"); shift ;;
    esac
  done
  local set_name
  if [[ $use_all -eq 1 ]]; then set_name="all"; else set_name="quick"; fi
  if [[ $with_custom -eq 1 ]]; then set_name="${set_name}c"; fi
  set_name="${set_name}-d${dim}"
  local suffix
  suffix="${label:-$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')}"
  suffix=$(printf '%s' "$suffix" | tr -cd '[:alnum:]_-' | cut -c1-40)
  local dir="${RESULTS_ROOT}/$(date +%Y%m%d_%H%M%S)_${suffix}_quick"
  echo "Quick check → ${dir}/"
  mkdir -p "$dir"
  if [[ -n "$methods" ]]; then
    printf '{\n  "type": "quick",\n  "created_at": "%s",\n  "commit": "%s",\n  "n_runs": %s,\n  "max_evals": %s,\n  "set": "%s",\n  "dim": %s,\n  "methods": "%s"\n}\n' \
      "$(date +%Y-%m-%dT%H:%M:%S)" \
      "$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')" \
      "$n_runs" "$max_evals" "$set_name" "$dim" "$methods" > "$dir/result.json"
  else
    printf '{\n  "type": "quick",\n  "created_at": "%s",\n  "commit": "%s",\n  "n_runs": %s,\n  "max_evals": %s,\n  "set": "%s",\n  "dim": %s\n}\n' \
      "$(date +%Y-%m-%dT%H:%M:%S)" \
      "$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')" \
      "$n_runs" "$max_evals" "$set_name" "$dim" > "$dir/result.json"
  fi
  if [[ -n "$noise" ]]; then
    "$PY" - <<PYEOF
import json
path = "$dir/result.json"
with open(path) as f:
    m = json.load(f)
m["noise"] = "$noise"
with open(path, "w") as f:
    json.dump(m, f, indent=2)
PYEOF
  fi
  "$PY" quick_check.py --output-dir "$dir" "${pass_args[@]+"${pass_args[@]}"}" &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "$dir" > "$DIR_FILE"
  echo "PID ${pid}  (stop with: ./run.sh stop)"
  trap "rm -f '$PID_FILE' '$DIR_FILE'" EXIT INT TERM
  wait "$pid"
  local rc=$?
  rm -f "$PID_FILE" "$DIR_FILE"
  local final_status="done"
  [[ $rc -ne 0 ]] && final_status="failed"
  "$PY" - <<PYEOF
import json
path = "$dir/result.json"
try:
    with open(path) as f:
        m = json.load(f)
    m["status"] = "$final_status"
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
except Exception:
    pass
PYEOF
  return $rc
}

# ── stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No running quick job found (${PID_FILE} not present)."
    exit 1
  fi
  local pid
  pid=$(<"$PID_FILE")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Sent SIGTERM to PID ${pid}."
    rm -f "$PID_FILE"
  else
    echo "Process ${pid} is not running. Removing stale PID file."
    rm -f "$PID_FILE"
  fi
}

# ── list ──────────────────────────────────────────────────────────────────────
cmd_list() {
  echo "=== Local results ==="
  if [[ -d "$RESULTS_ROOT" ]]; then
    local dirs
    dirs=$(ls -t "$RESULTS_ROOT" 2>/dev/null | grep -v '^$' || true)
    [[ -z "$dirs" ]] && echo "  (none)" || echo "$dirs" | awk '{print "  " $0}'
  fi
  echo ""
  echo "=== Remote runs (latest 5) ==="
  gh run list --workflow="$WORKFLOW" --limit 5
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
  local run_id="${1:-}"
  if [[ -z "$run_id" ]]; then
    run_id=$(gh run list --workflow="$WORKFLOW" \
      --limit 1 --json databaseId --jq '.[0].databaseId')
  fi
  gh run view "$run_id"
}

# ── ui ───────────────────────────────────────────────────────────────────────
cmd_ui() {
  echo "Starting UI at http://localhost:8080 ..."
  "$PY" web/app.py
}

# 研究ループ（1 時間ごとにクラウドで回っているサイクル）の現況を 1 画面で見る。
# 先に research-loop を取り込んでから表示する。
cmd_loop() {
  git fetch -q origin research-loop 2>/dev/null || true
  git merge --ff-only origin/research-loop >/dev/null 2>&1 ||     echo "（ローカルが分岐しています。git log origin/research-loop で確認してください）"
  "$PY" scripts/loop_status.py "$@"
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "${1:-help}" in
  trigger)  shift; cmd_trigger  "$@" ;;
  download) shift; cmd_download "$@" ;;
  quick)    shift; cmd_quick    "$@" ;;
  stop)            cmd_stop ;;
  list)            cmd_list ;;
  status)   shift; cmd_status   "${1:-}" ;;
  ui)       shift; cmd_ui       "${1:-}" ;;
  loop)     shift; cmd_loop     "$@" ;;
  *)
    cat <<'EOF'
Usage: ./run.sh <command> [options]

  trigger [--n-runs N] [--max-evals N]
      GitHub Actions ワークフローをトリガー（裏で動かす補助実験。手法評価には quick を使う）
      デフォルト: --n-runs 100 --max-evals 5000

  download [RUN_ID] [--label NAME]
      完了済みワークフローの結果をダウンロード（省略時は最新）
      --label で保存フォルダ名を指定（省略時はコミットハッシュ）
      保存先: results/YYYYMMDD_HHMMSS_<label|commit>/

  loop [--cycles N] [--full]
      研究ループの現況を表示（方針欄 / ゴール / 未解決の問いと claim 状況 /
      棄却済みの路線 / 直近サイクルの結論）。表示前に research-loop を取り込む。
      方向を変えるには docs/research_loop.md の「方針（ユーザーが書く欄）」に書くか、
      「未解決の問い」を並べ替えて research-loop に push する。

  quick [--n-runs N] [--max-evals N] [--dim {2|3|5|10|20}] [--methods LIST]
        [--funcs LIST] [--suite {bbob|cec2022}] [--all] [--custom] [--label NAME]
      ローカルで手法を検証・評価する（評価の標準: 2D BBOB-24 のみ / n_runs=20, max_evals=5000, --all）
      デフォルト: --n-runs 20 --max-evals 5000 --dim 2
      --methods は比較する手法のコンマ区切り（空欄=全手法）
        例: --methods "MC-ESO,DE,L-SHADE"
        利用可能: CMA-ES,IPOP-CMA-ES,BIPOP-CMA-ES,PSO,DE,L-SHADE,SaVOA,MC-ESO
      --funcs は対象関数のコンマ区切り（例: F01-Sphere,F03-RastriginSep）
      --all で 2D BBOB-24 フルセット（未指定時は quick-12 サブセット）※どちらも BBOB のみ
      --custom で Custom ベンチ（C01-C11, 2D 限定）を追加＝多峰/多解など特定目的の参照用
      --label で保存フォルダ名を指定（省略時はコミットハッシュ）
      保存先: results/YYYYMMDD_HHMMSS_<label|commit>_quick/

  stop
      実行中の quick ジョブを停止（SIGTERM を送信）

  list
      ローカル結果一覧 + リモート実行履歴（最新5件）

  status [RUN_ID]
      最新（またはRUN_IDで指定）のワークフロー実行状況を表示

  ui
      Results UI を起動 → http://localhost:8080
EOF
    ;;
esac
