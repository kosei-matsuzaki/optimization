import csv
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

# Make `core/` importable when running from project root or directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

app = Flask(__name__)

BASE_DIR = _ROOT
RESULTS_DIR = BASE_DIR / "results"
QUICK_CHECK = BASE_DIR / "quick_check.py"
PID_FILE    = BASE_DIR / ".quick.pid"
DIR_FILE    = BASE_DIR / ".quick.dir"
GH_REPO = "kosei-matsuzaki/optimization"
GH_WORKFLOW = "run.yml"

_jobs: dict[str, dict] = {}
_dl_jobs: dict[str, dict] = {}


def _write_pid(pid: int) -> None:
    try:
        PID_FILE.write_text(str(pid))
    except Exception:
        pass


def _clear_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _read_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def _read_quick_dir():
    try:
        return DIR_FILE.read_text().strip() or None
    except Exception:
        return None


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_results() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        (d.name for d in RESULTS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )


def _list_dims(run_dir: Path) -> list[str]:
    return sorted(
        d.name for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("dim")
    )


def _list_functions(run_dir: Path, dim: str) -> list[str]:
    dim_dir = run_dir / dim
    if not dim_dir.exists():
        return []

    funcs: set[str] = set()

    # New format: {Func}_landscape.svg or {Func}_convergence.svg
    for p in dim_dir.glob("*_landscape.svg"):
        funcs.add(p.stem[: -len("_landscape")])
    for p in dim_dir.glob("*_convergence.svg"):
        funcs.add(p.stem[: -len("_convergence")])

    # Legacy format: {Func}.svg  (pattern [FC]##-Name, no type suffix)
    if not funcs:
        for p in dim_dir.glob("*.svg"):
            if re.match(r"^[FC]\d{2}-\w+$", p.stem):
                funcs.add(p.stem)

    # Fallback: summary.csv (available once stats are written)
    if not funcs:
        summary_path = run_dir / dim / "summary.csv"
        if summary_path.exists():
            with open(summary_path, newline="") as f:
                for row in csv.DictReader(f):
                    if "function" in row:
                        funcs.add(row["function"])

    return sorted(funcs)


# Known visualization types for filename parsing
_ANIM_TYPES = [
    "3dpopulation_failed", "3devals_failed",
    "population_failed", "evals_failed", "vso_dyn_failed",
    "3dpopulation", "3devals",
    "population", "evals", "vso_dyn", "runs",
]


def _build_media_index(run_dir: Path, dim: str) -> dict:
    """Scan dim directory and return structured media file index."""
    dim_dir = run_dir / dim
    funcs = _list_functions(run_dir, dim)
    if not funcs:
        return {"funcs": [], "methods": [], "types": [], "files": [], "format": "legacy"}

    # Detect format by presence of *_landscape.svg
    is_new = any((dim_dir / f"{func}_landscape.svg").exists() for func in funcs[:5])

    files: list[dict] = []
    methods_seen: set[str] = set()
    types_seen: set[str] = set()

    if is_new:
        for func in funcs:
            # Function-level SVGs
            for type_, suffix in [("landscape", "_landscape.svg"), ("convergence", "_convergence.svg")]:
                if (dim_dir / f"{func}{suffix}").exists():
                    files.append({"func": func, "method": None, "type": type_, "ext": "svg"})
                    types_seen.add(type_)

            # Per-method files: {func}_{method}_{type}.{ext}
            for ext in ("webp", "gif", "svg"):
                for p in sorted(dim_dir.glob(f"{func}_*.{ext}")):
                    stem = p.stem  # e.g. F01-Sphere_VSO_evals
                    rest = stem[len(func) + 1:]  # e.g. VSO_evals
                    if not rest or rest in ("landscape", "convergence"):
                        continue
                    # Match known type suffix
                    matched_type = next(
                        (t for t in _ANIM_TYPES if rest.endswith(f"_{t}") or rest == t),
                        None,
                    )
                    if matched_type:
                        method_part = rest[: -(len(matched_type) + 1)] if rest != matched_type else None
                    else:
                        continue
                    if not method_part:
                        continue
                    # Avoid duplicate (prefer webp over gif)
                    existing = next(
                        (f for f in files
                         if f["func"] == func and f["method"] == method_part
                         and f["type"] == matched_type),
                        None,
                    )
                    if existing:
                        continue
                    methods_seen.add(method_part)
                    types_seen.add(matched_type)
                    files.append({
                        "func": func, "method": method_part,
                        "type": matched_type, "ext": ext,
                    })

        return {
            "funcs": sorted(funcs),
            "methods": sorted(methods_seen),
            "types": sorted(types_seen),
            "files": files,
            "format": "new",
        }

    # Legacy format
    legacy_map = [
        ("landscape",          "svg", lambda f: f"{f}.svg"),
        ("evals",              "gif", lambda f: f"{f}_evals.gif"),
        ("evals_failed",       "gif", lambda f: f"{f}_evals_failed.gif"),
        ("runs",               "gif", lambda f: f"{f}_runs.gif"),
        ("population",         "gif", lambda f: f"{f}_population.gif"),
        ("population_failed",  "gif", lambda f: f"{f}_population_failed.gif"),
        ("vso_dyn",            "svg", lambda f: f"{f}_vso_dyn.svg"),
        ("vso_dyn_failed",     "svg", lambda f: f"{f}_vso_dyn_failed.svg"),
    ]
    for func in funcs:
        for type_, ext, fname_fn in legacy_map:
            if (dim_dir / fname_fn(func)).exists():
                files.append({"func": func, "method": None, "type": type_, "ext": ext})
                types_seen.add(type_)

    return {
        "funcs": sorted(funcs),
        "methods": [],
        "types": sorted(types_seen),
        "files": files,
        "format": "legacy",
    }


def _read_summary(run_dir: Path, dim: str) -> list[dict]:
    path = run_dir / dim / "summary.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR), text=True,
        ).strip()
    except Exception:
        return "nogit"


def _write_result_meta(run_dir: Path, meta: dict) -> None:
    try:
        with open(run_dir / "result.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


def _read_result_meta(run_dir: Path) -> dict:
    path = run_dir / "result.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback: derive from directory name
    name = run_dir.name
    parts = name.split("_")
    meta: dict = {"type": "quick" if "quick" in name else "workflow"}
    if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
        d, t = parts[0], parts[1]
        meta["created_at"] = f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}"
        if len(parts) >= 3:
            meta["commit"] = parts[2]
    return meta


# ── quick run job ─────────────────────────────────────────────────────────────

def _run_job(job_id: str, n_runs: int, max_evals: int, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_result_meta(out_path, {
        "type": "quick",
        "status": "running",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _current_commit(),
        "n_runs": n_runs,
        "max_evals": max_evals,
    })
    proc = subprocess.Popen(
        ["python3", str(QUICK_CHECK),
         "--n-runs", str(n_runs),
         "--max-evals", str(max_evals),
         "--output-dir", out_dir],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(BASE_DIR),
    )
    _jobs[job_id]["proc"] = proc
    _write_pid(proc.pid)
    for line in proc.stdout:
        _jobs[job_id]["output"].append(line.rstrip())
    proc.wait()
    _clear_pid()
    if _jobs[job_id]["status"] != "stopped":
        _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
    # Persist final status to result.json
    final_meta = _read_result_meta(out_path)
    final_meta["status"] = _jobs[job_id]["status"]
    _write_result_meta(out_path, final_meta)


# ── download job ──────────────────────────────────────────────────────────────

def _download_job(job_id: str, gh_run_id: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dl = subprocess.run(
                ["gh", "run", "download", gh_run_id, "-D", tmp],
                capture_output=True, text=True, cwd=str(BASE_DIR),
            )
            if dl.returncode != 0:
                shutil.rmtree(dest_dir, ignore_errors=True)
                _dl_jobs[job_id].update(
                    status="failed",
                    message=dl.stderr.strip() or "Download failed.",
                )
                return

            src = Path(tmp) / "results"
            if not src.exists():
                src = Path(tmp)
            for item in src.iterdir():
                target = dest_dir / item.name
                if target.exists():
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                shutil.move(str(item), str(dest_dir))

        _write_result_meta(dest_dir, {
            "type": "workflow",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "commit": _current_commit(),
            "gh_run_id": gh_run_id,
            "status": "done",
        })
        _dl_jobs[job_id].update(
            status="done",
            result_dir=dest_dir.name,
            message=f"Saved to {dest_dir.name}",
        )
    except Exception as e:
        _dl_jobs[job_id].update(status="failed", message=str(e))


# ── routes ────────────────────────────────────────────────────────────────────

def _running_dirs() -> list:
    dirs = [
        job["result_dir"]
        for job in _jobs.values()
        if job.get("status") == "running" and job.get("result_dir")
    ]
    # Also include shell-started quick job (run.sh quick)
    pid = _read_pid()
    if pid and _pid_running(pid):
        quick_dir = _read_quick_dir()
        if quick_dir:
            name = Path(quick_dir).name
            if name and name not in dirs:
                dirs.append(name)
    return dirs


@app.route("/")
def index():
    results = _list_results()
    results_meta = {r: _read_result_meta(RESULTS_DIR / r) for r in results}
    return render_template("index.html", results=results, results_meta=results_meta,
                           running=_running_dirs())


@app.route("/methods")
def methods():
    return render_template("methods.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    n_runs    = max(1,   min(20,    int(request.form.get("n_runs",   3))))
    max_evals = max(100, min(20000, int(request.form.get("max_evals", 2000))))
    label     = re.sub(r'[^\w\-]', '_', request.form.get("label", "").strip())[:40].strip('_')

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix  = label if label else _current_commit()
    out_dir = str(RESULTS_DIR / f"{ts}_{suffix}_quick")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "output": [], "result_dir": Path(out_dir).name}
    threading.Thread(
        target=_run_job, args=(job_id, n_runs, max_evals, out_dir), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job)


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
         "--json", "databaseId,status,conclusion,displayTitle,createdAt"],
        capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        return jsonify({"error": result.stderr.strip()}), 500
    return jsonify(json.loads(result.stdout))


@app.route("/api/download", methods=["POST"])
def api_download():
    gh_run_id = request.form.get("run_id", "").strip()
    if not gh_run_id:
        return jsonify({"ok": False, "message": "run_id required"}), 400
    label    = re.sub(r'[^\w\-]', '_', request.form.get("label", "").strip())[:40].strip('_')

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix   = label if label else _current_commit()
    dest_dir = RESULTS_DIR / f"{ts}_{suffix}"

    job_id = uuid.uuid4().hex[:8]
    _dl_jobs[job_id] = {"status": "running", "result_dir": None, "message": "Downloading..."}
    threading.Thread(
        target=_download_job, args=(job_id, gh_run_id, dest_dir), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/dl-status/<job_id>")
def api_dl_status(job_id: str):
    job = _dl_jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job)


@app.route("/results/<run_id>")
def result_detail(run_id: str):
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return redirect(url_for("index"))

    dims = _list_dims(run_dir)
    if not dims:
        return redirect(url_for("index"))

    dims_data = {
        dim: {
            "functions": _list_functions(run_dir, dim),
            "summary":   _read_summary(run_dir, dim),
        }
        for dim in dims
    }

    all_results = _list_results()
    all_results_meta = {r: _read_result_meta(RESULTS_DIR / r) for r in all_results}
    return render_template(
        "result.html",
        run_id=run_id,
        dims=dims,
        dims_data=dims_data,
        all_results=all_results,
        all_results_meta=all_results_meta,
    )


@app.route("/api/stats/<run_id>/<dim>/<func_name>")
def api_stats(run_id: str, dim: str, func_name: str):
    csv_path = RESULTS_DIR / run_id / dim / "stats" / f"{func_name}.csv"
    if not csv_path.exists():
        return jsonify({"headers": [], "rows": []})
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return jsonify({"headers": headers, "rows": rows})


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


@app.route("/media/<path:filepath>")
def media(filepath: str):
    full_path = RESULTS_DIR / filepath
    if not full_path.exists():
        return "Not found", 404
    return send_file(full_path)


@app.route("/api/stop/<job_id>", methods=["POST"])
def api_stop_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    proc = job.get("proc")
    if proc and job["status"] == "running":
        job["status"] = "stopped"
        proc.terminate()
    return jsonify({"ok": True, "status": job["status"]})


@app.route("/api/shell-job")
def api_shell_job():
    pid = _read_pid()
    if pid is None:
        return jsonify({"running": False})
    if _pid_running(pid):
        return jsonify({"running": True, "pid": pid})
    _clear_pid()
    return jsonify({"running": False})


@app.route("/api/shell-stop", methods=["POST"])
def api_shell_stop():
    pid = _read_pid()
    if pid is None:
        return jsonify({"ok": False, "message": "No running job found"})
    if _pid_running(pid):
        os.kill(pid, signal.SIGTERM)
        _clear_pid()
        return jsonify({"ok": True})
    _clear_pid()
    return jsonify({"ok": False, "message": "Process already finished"})


@app.route("/api/results")
def api_results_list():
    results = _list_results()
    results_meta = {r: _read_result_meta(RESULTS_DIR / r) for r in results}
    return jsonify({"results": results, "meta": results_meta, "running": _running_dirs()})


@app.route("/api/media-index/<run_id>/<dim>")
def api_media_index(run_id: str, dim: str):
    if not run_id or "/" in run_id or ".." in run_id:
        return jsonify({"error": "invalid"}), 400
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(_build_media_index(run_dir, dim))


@app.route("/api/result-data/<run_id>")
def api_result_data(run_id: str):
    if not run_id or "/" in run_id or ".." in run_id:
        return jsonify({"error": "invalid"}), 400
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        return jsonify({"error": "not found"}), 404
    dims = _list_dims(run_dir)
    dims_data = {
        dim: {
            "functions": _list_functions(run_dir, dim),
            "summary":   _read_summary(run_dir, dim),
        }
        for dim in dims
    }
    return jsonify({"dims": dims, "dims_data": dims_data})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
