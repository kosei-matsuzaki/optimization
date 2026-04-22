import csv
import datetime
import json
import shutil
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
GH_REPO = "kosei-matsuzaki/optimization"
GH_WORKFLOW = "run.yml"

_jobs: dict[str, dict] = {}
_dl_jobs: dict[str, dict] = {}


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
    return sorted({p.stem for p in dim_dir.glob("*.svg")})


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


# ── quick run job ─────────────────────────────────────────────────────────────

def _run_job(job_id: str, n_runs: int, max_evals: int, out_dir: str) -> None:
    proc = subprocess.Popen(
        ["python3", str(QUICK_CHECK),
         "--n-runs", str(n_runs),
         "--max-evals", str(max_evals),
         "--output-dir", out_dir],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(BASE_DIR),
    )
    for line in proc.stdout:
        _jobs[job_id]["output"].append(line.rstrip())
    proc.wait()
    _jobs[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
    _jobs[job_id]["result_dir"] = Path(out_dir).name


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
                dest_dir.rmdir()
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

        _dl_jobs[job_id].update(
            status="done",
            result_dir=dest_dir.name,
            message=f"Saved to {dest_dir.name}",
        )
    except Exception as e:
        _dl_jobs[job_id].update(status="failed", message=str(e))


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", results=_list_results())


@app.route("/methods")
def methods():
    return render_template("methods.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    n_runs   = max(1,   min(20,    int(request.form.get("n_runs",   3))))
    max_evals = max(100, min(20000, int(request.form.get("max_evals", 2000))))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = str(RESULTS_DIR / f"{ts}_{_current_commit()}_quick")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "output": [], "result_dir": None}
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

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = RESULTS_DIR / f"{ts}_{_current_commit()}"

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

    return render_template(
        "result.html",
        run_id=run_id,
        dims=dims,
        dims_data=dims_data,
        all_results=_list_results(),
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


@app.route("/media/<path:filepath>")
def media(filepath: str):
    full_path = RESULTS_DIR / filepath
    if not full_path.exists():
        return "Not found", 404
    return send_file(full_path)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
