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

# ── trigger ──────────────────────────────────────────────────────────────────
cmd_trigger() {
  local n_runs=30 max_evals=5000
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
  local n_runs=10 max_evals=2000 label=""
  local pass_args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --n-runs)    n_runs="$2";    pass_args+=("$1" "$2"); shift 2 ;;
      --max-evals) max_evals="$2"; pass_args+=("$1" "$2"); shift 2 ;;
      --label)     label="$2";     shift 2 ;;
      *)           pass_args+=("$1"); shift ;;
    esac
  done
  local suffix
  suffix="${label:-$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')}"
  suffix=$(printf '%s' "$suffix" | tr -cd '[:alnum:]_-' | cut -c1-40)
  local dir="${RESULTS_ROOT}/$(date +%Y%m%d_%H%M%S)_${suffix}_quick"
  echo "Quick check → ${dir}/"
  mkdir -p "$dir"
  printf '{\n  "type": "quick",\n  "created_at": "%s",\n  "commit": "%s",\n  "n_runs": %s,\n  "max_evals": %s\n}\n' \
    "$(date +%Y-%m-%dT%H:%M:%S)" \
    "$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')" \
    "$n_runs" "$max_evals" > "$dir/result.json"
  python3 quick_check.py --output-dir "$dir" "${pass_args[@]+"${pass_args[@]}"}" &
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
  python3 - <<PYEOF
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
  python3 web/app.py
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
  *)
    cat <<'EOF'
Usage: ./run.sh <command> [options]

  trigger [--n-runs N] [--max-evals N]
      GitHub Actions ワークフローをトリガー
      デフォルト: --n-runs 30 --max-evals 5000

  download [RUN_ID] [--label NAME]
      完了済みワークフローの結果をダウンロード（省略時は最新）
      --label で保存フォルダ名を指定（省略時はコミットハッシュ）
      保存先: results/YYYYMMDD_HHMMSS_<label|commit>/

  quick [--n-runs N] [--max-evals N] [--label NAME]
      ローカルで軽量確認を実行
      デフォルト: --n-runs 10 --max-evals 2000
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
