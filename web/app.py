"""Flask results UI — thin routing layer.

Heavy lifting lives in the :mod:`app_lib` package:
  - :mod:`app_lib.config`  — paths and GitHub constants
  - :mod:`app_lib.results` — read-only data layer over ``results/``
  - :mod:`app_lib.jobs`    — local quick-run and download background jobs

Run with ``python3 web/app.py`` (or ``./run.sh ui``).
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from app_lib import config, jobs, results

app = Flask(__name__)

BASE_DIR    = config.BASE_DIR
RESULTS_DIR = config.RESULTS_DIR
QUICK_CHECK = config.QUICK_CHECK
GH_REPO     = config.GH_REPO
GH_WORKFLOW = config.GH_WORKFLOW


# ── pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    names = results.list_results()
    meta = {r: results.read_result_meta(RESULTS_DIR / r) for r in names}
    return render_template("index.html", results=names, results_meta=meta,
                           running=jobs.running_dirs())


@app.route("/methods")
def methods():
    return render_template("methods.html")


@app.route("/benchmarks")
def benchmarks_page():
    """Static reference: the function × shape-tag correspondence matrix.

    Run-independent — derived directly from core.benchmarks (SHAPE_TAGS /
    TAG_AXES), the single source of truth for benchmark shape classification.
    """
    from core.benchmarks import (_BBOB_SPECS, CUSTOM_BENCHMARKS,
                                  _CEC2022_SPECS, SHAPE_TAGS, TAG_AXES)

    def _row(name: str, category: str) -> dict:
        return {"name": name, "category": category,
                "tags": SHAPE_TAGS.get(name, [])}

    suites = [
        {"key": "bbob",   "label": "BBOB",    "sub": "24 関数 (dim 2/3/4)",
         "rows": [_row(n, c) for _fid, n, c in _BBOB_SPECS]},
        {"key": "custom", "label": "Custom",  "sub": "11 関数 (2-D)",
         "rows": [_row(b.name, b.category) for b in CUSTOM_BENCHMARKS]},
        {"key": "cec",    "label": "CEC2022", "sub": "12 関数 (dim 10, hold-out)",
         "rows": [_row(n, c) for _ioh, n, c in _CEC2022_SPECS]},
    ]
    # Flat column list (axis order) + per-tag usage count across all functions.
    all_rows = [r for s in suites for r in s["rows"]]
    counts: dict[str, int] = {}
    for r in all_rows:
        for t in r["tags"]:
            counts[t] = counts.get(t, 0) + 1
    axes = [{"axis": axis, "tags": [t for t in tags if counts.get(t)]}
            for axis, tags in TAG_AXES]
    axes = [a for a in axes if a["tags"]]
    columns = [t for a in axes for t in a["tags"]]
    return render_template("benchmarks.html", suites=suites, axes=axes,
                           columns=columns, counts=counts)


@app.route("/results/<run_id>")
def result_detail(run_id: str):
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return redirect(url_for("index"))

    dims = results.list_dims(run_dir)
    if not dims:
        return redirect(url_for("index"))

    dims_data = {
        dim: {
            "functions": results.list_functions(run_dir, dim),
            "summary":   results.read_summary(run_dir, dim),
            "wilcoxon":  results.read_wilcoxon(run_dir, dim),
        }
        for dim in dims
    }

    all_results = results.list_results()
    all_results_meta = {r: results.read_result_meta(RESULTS_DIR / r) for r in all_results}
    return render_template(
        "result.html",
        run_id=run_id,
        dims=dims,
        dims_data=dims_data,
        all_results=all_results,
        all_results_meta=all_results_meta,
    )


@app.route("/media/<path:filepath>")
def media(filepath: str):
    full_path = RESULTS_DIR / filepath
    if not full_path.exists():
        return "Not found", 404
    return send_file(full_path)


# ── option metadata (quick run modal) ──────────────────────────────────────────

@app.route("/api/methods")
def api_methods():
    """Return the list of available optimizer names (for the quick run modal)."""
    spec = importlib.util.spec_from_file_location("quick_check", QUICK_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return jsonify({"methods": list(mod._OPTIMIZERS.keys())})


@app.route("/api/functions")
def api_functions():
    """Return the benchmark function list grouped by category, plus the
    canonical 'quick-12' preset names (for the quick run modal)."""
    qc_spec = importlib.util.spec_from_file_location("quick_check", QUICK_CHECK)
    qc = importlib.util.module_from_spec(qc_spec)
    qc_spec.loader.exec_module(qc)

    # Discover the full function set with categories from core.benchmarks
    from core.benchmarks import _BBOB_SPECS, CUSTOM_BENCHMARKS  # noqa: E402
    groups: dict[str, list[str]] = {}
    for _fid, name, cat in _BBOB_SPECS:
        groups.setdefault(cat, []).append(name)
    for b in CUSTOM_BENCHMARKS:
        groups.setdefault(b.category, []).append(b.name)
    return jsonify({
        "categories": groups,
        "quick_12":   list(qc._QUICK_FUNCTIONS),
    })


# ── quick run ──────────────────────────────────────────────────────────────────

@app.route("/api/run", methods=["POST"])
def api_run():
    n_runs    = max(1,   min(100,   int(request.form.get("n_runs",   3))))
    # Cap at 50000 so high-dimension budgets (dim20 ≈ 2500×d = 50000) are allowed.
    max_evals = max(100, min(50000, int(request.form.get("max_evals", 2000))))
    label     = re.sub(r'[^\w\-]', '_', request.form.get("label", "").strip())[:40].strip('_')
    # Checkbox value arrives as the literal string "true" / "on" / "1" when ticked;
    # treat anything else (including absent) as off.
    use_all   = request.form.get("use_all", "").lower() in ("true", "on", "1", "yes")
    # Dimension — restricted to values supported by quick_check.py
    try:
        dim = int(request.form.get("dim", "2"))
    except ValueError:
        dim = 2
    if dim not in (2, 3, 5, 10, 20):
        dim = 2
    # Methods — comma-separated, sanitised (alnum / dash / dot / comma / space only)
    methods_raw = request.form.get("methods", "").strip()
    methods_arg = re.sub(r'[^\w\-\., ]', '', methods_raw) if methods_raw else None
    # Funcs — comma-separated function names (sanitised)
    funcs_raw = request.form.get("funcs", "").strip()
    funcs_arg = re.sub(r'[^\w\-\., ]', '', funcs_raw) if funcs_raw else None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix  = label if label else results.current_commit()
    out_dir = str(RESULTS_DIR / f"{ts}_{suffix}_quick")

    job_id = uuid.uuid4().hex[:8]
    jobs.JOBS[job_id] = {"status": "running", "output": [], "result_dir": Path(out_dir).name}
    threading.Thread(
        target=jobs.run_job,
        args=(job_id, n_runs, max_evals, out_dir, use_all, dim, methods_arg, funcs_arg),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    job = jobs.JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job)


@app.route("/api/stop/<job_id>", methods=["POST"])
def api_stop_job(job_id: str):
    job = jobs.JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    proc = jobs.JOB_PROCS.get(job_id)
    if proc and job["status"] == "running":
        job["status"] = "stopped"
        proc.terminate()
    return jsonify({"ok": True, "status": job["status"]})


# ── shell-started quick job (run.sh quick) ──────────────────────────────────────

@app.route("/api/shell-job")
def api_shell_job():
    pid = jobs.read_pid()
    if pid is None:
        return jsonify({"running": False})
    if jobs.pid_running(pid):
        return jsonify({"running": True, "pid": pid})
    jobs.clear_pid()
    return jsonify({"running": False})


@app.route("/api/shell-stop", methods=["POST"])
def api_shell_stop():
    pid = jobs.read_pid()
    if pid is None:
        return jsonify({"ok": False, "message": "No running job found"})
    if jobs.pid_running(pid):
        os.kill(pid, signal.SIGTERM)
        jobs.clear_pid()
        return jsonify({"ok": True})
    jobs.clear_pid()
    return jsonify({"ok": False, "message": "Process already finished"})


# ── GitHub Actions workflow ─────────────────────────────────────────────────────

@app.route("/api/gh-trigger", methods=["POST"])
def api_gh_trigger():
    n_runs    = request.form.get("n_runs",    "30")
    max_evals = request.form.get("max_evals", "5000")
    result = subprocess.run(
        ["gh", "workflow", "run", GH_WORKFLOW, "--repo", GH_REPO,
         "-f", f"n_runs={n_runs}", "-f", f"max_evals={max_evals}"],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    if result.returncode == 0:
        return jsonify({"ok": True, "message": "Workflow triggered."})
    return jsonify({"ok": False, "message": result.stderr.strip() or "Failed."}), 500


@app.route("/api/gh-runs")
def api_gh_runs():
    result = subprocess.run(
        ["gh", "run", "list", f"--workflow={GH_WORKFLOW}",
         "--limit", "10",
         "--json", "databaseId,status,conclusion,name,headSha,createdAt"],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr.strip()}), 500
    return jsonify(json.loads(result.stdout))


# ── artifact download ────────────────────────────────────────────────────────────

@app.route("/api/download", methods=["POST"])
def api_download():
    gh_run_id = request.form.get("run_id", "").strip()
    if not gh_run_id:
        return jsonify({"ok": False, "message": "run_id required"}), 400
    label    = re.sub(r'[^\w\-]', '_', request.form.get("label", "").strip())[:40].strip('_')

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix   = label if label else results.current_commit()
    dest_dir = RESULTS_DIR / f"{ts}_{suffix}"

    job_id = uuid.uuid4().hex[:8]
    jobs.DL_JOBS[job_id] = {"status": "running", "result_dir": None, "message": "Downloading..."}
    threading.Thread(
        target=jobs.download_job, args=(job_id, gh_run_id, dest_dir), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/dl-status/<job_id>")
def api_dl_status(job_id: str):
    job = jobs.DL_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job)


# ── results data / management ────────────────────────────────────────────────────

@app.route("/api/results")
def api_results_list():
    names = results.list_results()
    meta = {r: results.read_result_meta(RESULTS_DIR / r) for r in names}
    return jsonify({"results": names, "meta": meta, "running": jobs.running_dirs()})


@app.route("/api/results/<run_id>/rename", methods=["POST"])
def api_rename_result(run_id: str):
    if not run_id or "/" in run_id or ".." in run_id:
        return jsonify({"ok": False, "message": "Invalid ID"}), 400
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return jsonify({"ok": False, "message": "Not found"}), 404
    new_name = request.form.get("new_name", "").strip()
    if not new_name or "/" in new_name or ".." in new_name:
        return jsonify({"ok": False, "message": "Invalid name"}), 400
    new_dir = RESULTS_DIR / new_name
    if new_dir.exists():
        return jsonify({"ok": False, "message": "Name already exists"}), 409
    run_dir.rename(new_dir)
    return jsonify({"ok": True, "new_name": new_name})


@app.route("/api/results/<run_id>", methods=["DELETE"])
def api_delete_result(run_id: str):
    if not run_id or "/" in run_id or ".." in run_id:
        return jsonify({"ok": False, "message": "Invalid ID"}), 400
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return jsonify({"ok": False, "message": "Not found"}), 404
    shutil.rmtree(run_dir)
    return jsonify({"ok": True})


@app.route("/api/stats/<run_id>/<dim>/<func_name>")
def api_stats(run_id: str, dim: str, func_name: str):
    return jsonify(results.read_stats(run_id, dim, func_name))


@app.route("/api/media-index/<run_id>/<dim>")
def api_media_index(run_id: str, dim: str):
    if not run_id or "/" in run_id or ".." in run_id:
        return jsonify({"error": "invalid"}), 400
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(results.build_media_index(run_dir, dim))


@app.route("/api/result-data/<run_id>")
def api_result_data(run_id: str):
    if not run_id or "/" in run_id or ".." in run_id:
        return jsonify({"error": "invalid"}), 400
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "not found"}), 404
    dims = results.list_dims(run_dir)
    dims_data = {
        dim: {
            "functions": results.list_functions(run_dir, dim),
            "summary":   results.read_summary(run_dir, dim),
            "wilcoxon":  results.read_wilcoxon(run_dir, dim),
        }
        for dim in dims
    }
    return jsonify({"dims": dims, "dims_data": dims_data})


@app.route("/api/overall/<run_id>/<dim>")
def api_overall(run_id: str, dim: str):
    if not run_id or "/" in run_id or ".." in run_id:
        return jsonify({"error": "invalid"}), 400
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(results.compute_overall_ranking(run_dir, dim))


if __name__ == "__main__":
    app.run(debug=True, port=8080)
