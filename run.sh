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

_version() {
  echo "$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD 2>/dev/null || echo 'nogit')"
}

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
  local run_id="${1:-}"
  if [[ -z "$run_id" ]]; then
    run_id=$(gh run list --workflow="$WORKFLOW" --status completed \
      --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)
    [[ -z "$run_id" ]] && { echo "No completed runs found."; exit 1; }
    echo "Latest completed run: ${run_id}"
  fi

  local dir="${RESULTS_ROOT}/$(_version)"
  echo "Downloading run ${run_id} → ${dir}/"
  local tmp; tmp=$(mktemp -d)
  gh run download "$run_id" -D "$tmp"
  mkdir -p "$dir"
  # artifact is named "results", its contents go directly into $dir
  mv "$tmp/results/"* "$dir/"
  rm -rf "$tmp"
  echo "Saved to: ${dir}/"
}

# ── quick ─────────────────────────────────────────────────────────────────────
cmd_quick() {
  local dir="${RESULTS_ROOT}/$(_version)_quick"
  echo "Quick check → ${dir}/"
  python3 quick_check.py --output-dir "$dir" "$@"
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
  download) shift; cmd_download "${1:-}" ;;
  quick)    shift; cmd_quick    "$@" ;;
  list)            cmd_list ;;
  status)   shift; cmd_status   "${1:-}" ;;
  ui)       shift; cmd_ui       "${1:-}" ;;
  *)
    cat <<'EOF'
Usage: ./run.sh <command> [options]

  trigger [--n-runs N] [--max-evals N]
      GitHub Actions ワークフローをトリガー
      デフォルト: --n-runs 30 --max-evals 5000

  download [RUN_ID]
      完了済みワークフローの結果をダウンロード（省略時は最新）
      保存先: results/YYYYMMDD_HHMMSS_<commit>/

  quick [--n-runs N] [--max-evals N]
      ローカルで軽量確認を実行
      デフォルト: --n-runs 3 --max-evals 2000
      保存先: results/YYYYMMDD_HHMMSS_<commit>_quick/

  list
      ローカル結果一覧 + リモート実行履歴（最新5件）

  status [RUN_ID]
      最新（またはRUN_IDで指定）のワークフロー実行状況を表示

  ui
      Results UI を起動 → http://localhost:8080
EOF
    ;;
esac
