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
from typing import Optional

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
_job_procs: dict[str, subprocess.Popen] = {}
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

    # Primary: {Func}_landscape.svg or {Func}_convergence.svg
    for p in dim_dir.glob("*_landscape.svg"):
        funcs.add(p.stem[: -len("_landscape")])
    for p in dim_dir.glob("*_convergence.svg"):
        funcs.add(p.stem[: -len("_convergence")])

    # Fallback: summary.csv (before landscape SVGs are written)
    if not funcs:
        summary_path = run_dir / dim / "summary.csv"
        if summary_path.exists():
            with open(summary_path, newline="") as f:
                for row in csv.DictReader(f):
                    if "function" in row:
                        funcs.add(row["function"])

    return sorted(funcs)


# Known visualization types for filename parsing (longest-first to avoid prefix clash)
_ANIM_TYPES = [
    "3dpopulation_failed", "3devals_failed",
    "population_failed", "evals_failed", "outbreak_dyn_failed",
    "3dpopulation", "3devals",
    "population", "evals", "outbreak_dyn", "runs",
]


def _build_media_index(run_dir: Path, dim: str) -> dict:
    """Scan dim directory and return structured media file index."""
    dim_dir = run_dir / dim
    funcs = _list_functions(run_dir, dim)
    if not funcs:
        return {"funcs": [], "methods": [], "types": [], "files": []}

    files: list[dict] = []
    methods_seen: set[str] = set()
    types_seen: set[str] = set()

    for func in funcs:
        # Function-level SVGs
        for type_, suffix in [("landscape", "_landscape.svg"), ("convergence", "_convergence.svg")]:
            if (dim_dir / f"{func}{suffix}").exists():
                files.append({"func": func, "method": None, "type": type_, "ext": "svg"})
                types_seen.add(type_)

        # Per-method files: {func}_{method}_{type}.{ext}
        for ext in ("webp", "gif", "svg"):
            for p in sorted(dim_dir.glob(f"{func}_*.{ext}")):
                stem = p.stem  # e.g. F01-Sphere_MC-ESO_evals
                rest = stem[len(func) + 1:]  # e.g. MC-ESO_evals
                if not rest or rest in ("landscape", "convergence"):
                    continue
                matched_type = next(
                    (t for t in _ANIM_TYPES if rest.endswith(f"_{t}") or rest == t),
                    None,
                )
                if not matched_type:
                    continue
                method_part = rest[: -(len(matched_type) + 1)] if rest != matched_type else None
                if not method_part:
                    continue
                # Prefer webp over gif (skip duplicates)
                if any(f["func"] == func and f["method"] == method_part
                       and f["type"] == matched_type for f in files):
                    continue
                methods_seen.add(method_part)
                types_seen.add(matched_type)
                files.append({"func": func, "method": method_part, "type": matched_type, "ext": ext})

    return {
        "funcs": sorted(funcs),
        "methods": sorted(methods_seen),
        "types": sorted(types_seen),
        "files": files,
    }


def _read_summary(run_dir: Path, dim: str) -> list[dict]:
    path = run_dir / dim / "summary.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_wilcoxon(run_dir: Path, dim: str) -> list[dict]:
    """Read per-function Wilcoxon signed-rank rows (reference vs each method)."""
    path = run_dir / dim / "wilcoxon.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _compute_overall_ranking(run_dir: Path, dim: str) -> dict:
    """Per-indicator Friedman ranking across all functions.

    Ranks methods within each function independently for two indicators:
      - "bf"  : median_best_f (lower is better; robust to outlier runs)
      - "ert" : expected runtime to 1e-4 target (lower is better; inf if SR=0)
    For each indicator we also report Friedman χ²_F, p value, and Nemenyi
    critical difference at α=0.05 so the user can judge whether mean-rank
    gaps are statistically meaningful.
    """
    import numpy as np
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    rows = _read_summary(run_dir, dim)
    if not rows:
        return {"methods": [], "categories": [], "funcs": [],
                "leaderboard": [], "func_ranks": {}, "func_categories": {},
                "friedman": {}}

    def parse_sr(s: str) -> float:
        if not s or s == "N/A":
            return 0.0
        return float(s.strip("%")) / 100.0

    def parse_float(s: str) -> float:
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("inf")

    funcs   = sorted(set(r["function"] for r in rows))
    methods = sorted(set(r["method"]   for r in rows))
    func_categories: dict[str, str] = {}
    data: dict[str, dict] = {}
    for row in rows:
        f, m = row["function"], row["method"]
        func_categories[f] = row.get("category", "unknown")
        # Prefer median_best_f; fall back to mean_best_f for older runs.
        bf_raw = row.get("median_best_f") or row.get("mean_best_f") or "inf"
        # ECDF AUC: higher is better; missing → 0 (worst).
        try:
            ecdf_v = float(row.get("ecdf_auc", "0") or "0")
        except (TypeError, ValueError):
            ecdf_v = 0.0
        data.setdefault(f, {})[m] = {
            "sr":   parse_sr(row.get("sr_1e-4", "0%")),
            "ert":  parse_float(row.get("ert", "inf")),
            "bf":   parse_float(bf_raw),
            "ecdf": ecdf_v,
        }

    # bf / ert: lower-is-better. ecdf: higher-is-better — negated when ranking.
    INDICATORS = ("bf", "ert", "ecdf")
    HIGHER_BETTER = {"ecdf"}

    # Per-function, per-indicator ranks via average-tie ranking.
    # Missing data → worst rank for that block (still flagged via "missing" set).
    k = len(methods)
    func_ranks: dict[str, dict[str, dict[str, float]]] = {ind: {} for ind in INDICATORS}
    rank_matrix: dict[str, list[list[float]]] = {ind: [] for ind in INDICATORS}

    for func in funcs:
        md = data.get(func, {})
        for ind in INDICATORS:
            missing = float("-inf") if ind in HIGHER_BETTER else float("inf")
            vals = np.array([
                md.get(m, {}).get(ind, missing) for m in methods
            ], dtype=float)
            # For higher-is-better metrics, negate so rankdata still treats
            # smaller-rank-number = better.
            if ind in HIGHER_BETTER:
                vals = -vals
            ranks_arr = rankdata(vals, method="average")
            ranks_dict = {m: float(r) for m, r in zip(methods, ranks_arr)}
            func_ranks[ind][func] = ranks_dict
            rank_matrix[ind].append([ranks_dict[m] for m in methods])

    categories = sorted(set(func_categories.values()))

    # Friedman χ²_F + Nemenyi CD per indicator.
    friedman: dict[str, dict] = {}
    n_blocks = len(funcs)
    for ind in INDICATORS:
        # friedmanchisquare expects one array per method, values across blocks.
        cols = np.array(rank_matrix[ind])           # shape (n_blocks, k)
        if n_blocks >= 2 and k >= 3:
            try:
                chi2, pval = friedmanchisquare(*[cols[:, j] for j in range(k)])
                chi2_f, p_f = float(chi2), float(pval)
            except ValueError:
                chi2_f, p_f = float("nan"), float("nan")
        else:
            chi2_f, p_f = float("nan"), float("nan")
        # Nemenyi CD at α=0.05: q_α / sqrt(2) * sqrt(k(k+1)/(6N))
        if k >= 2 and n_blocks >= 1:
            try:
                q_alpha = float(studentized_range.ppf(0.95, k, np.inf))
                cd = q_alpha / np.sqrt(2.0) * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
                cd_f = float(cd)
            except Exception:
                cd_f = float("nan")
        else:
            cd_f = float("nan")
        friedman[ind] = {
            "chi2":     None if np.isnan(chi2_f) else round(chi2_f, 3),
            "p":        None if np.isnan(p_f)    else float(f"{p_f:.4g}"),
            "cd":       None if np.isnan(cd_f)   else round(cd_f, 3),
            "n_blocks": n_blocks,
            "k":        k,
        }

    leaderboard = []
    for method in methods:
        sr_vals = [data.get(f, {}).get(method, {}).get("sr", 0.0) for f in funcs]
        mean_sr = float(np.mean(sr_vals)) if sr_vals else 0.0
        cat_sr: dict[str, float | None] = {}
        for cat in categories:
            cf = [f for f in funcs if func_categories.get(f) == cat]
            cat_sr[cat] = (
                float(np.mean([data.get(f, {}).get(method, {}).get("sr", 0.0) for f in cf]))
                if cf else None
            )
        entry: dict = {
            "method":      method,
            "mean_sr":     round(mean_sr, 4),
            "category_sr": {c: (round(v, 4) if v is not None else None)
                            for c, v in cat_sr.items()},
        }
        for ind in INDICATORS:
            rvals = [func_ranks[ind].get(f, {}).get(method, float(k)) for f in funcs]
            arr = np.array(rvals, dtype=float)
            entry[f"mean_rank_{ind}"] = round(float(np.mean(arr)), 2)
            entry[f"rank_std_{ind}"]  = round(float(np.std(arr)),  2)
            entry[f"n_best_{ind}"]    = int(np.sum(arr == 1.0))
            entry[f"n_worst_{ind}"]   = int(np.sum(arr == float(k)))
        leaderboard.append(entry)

    # Primary sort: bf mean rank → ert → ecdf.
    leaderboard.sort(key=lambda x: (x["mean_rank_bf"], x["mean_rank_ert"], x["mean_rank_ecdf"]))
    return {
        "methods":         methods,
        "categories":      categories,
        "funcs":           funcs,
        "func_categories": func_categories,
        "leaderboard":     leaderboard,
        "func_ranks":      func_ranks,
        "friedman":        friedman,
    }


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

def _run_job(job_id: str, n_runs: int, max_evals: int, out_dir: str,
             use_all: bool = False) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_result_meta(out_path, {
        "type": "quick",
        "status": "running",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _current_commit(),
        "n_runs": n_runs,
        "max_evals": max_evals,
        "set": "all-26" if use_all else "quick-12",
    })
    cmd = ["python3", str(QUICK_CHECK),
           "--n-runs", str(n_runs),
           "--max-evals", str(max_evals),
           "--output-dir", out_dir]
    if use_all:
        cmd.append("--all")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(BASE_DIR),
    )
    _job_procs[job_id] = proc
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

def _get_artifact_total_size(gh_run_id: str) -> Optional[int]:
    """Total bytes across all artifacts of a workflow run, via gh REST API."""
    try:
        out = subprocess.check_output(
            ["gh", "api",
             f"repos/{GH_REPO}/actions/runs/{gh_run_id}/artifacts",
             "--jq", "[.artifacts[].size_in_bytes] | add"],
            cwd=str(BASE_DIR), text=True, timeout=15,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out) if out and out != "null" else None
    except Exception:
        return None


def _find_gh_artifact_zip(pid: int, work_tmp: Path) -> Optional[Path]:
    """Find the gh-artifact*.zip the gh subprocess is streaming into."""
    # Prefer the work-tmp dir we forced via TMPDIR (no cross-process collisions).
    try:
        zips = sorted(work_tmp.glob("gh-artifact*.zip"))
        if zips:
            return zips[-1]
        # gh sometimes places the zip in a sub-tempdir; recurse one level.
        zips = sorted(work_tmp.glob("*/gh-artifact*.zip"))
        if zips:
            return zips[-1]
    except Exception:
        pass
    # Fallback: lsof the subprocess and look for the zip in its open fds.
    try:
        out = subprocess.check_output(
            ["lsof", "-Fn", "-p", str(pid)],
            text=True, stderr=subprocess.DEVNULL, timeout=4,
        )
        for line in out.splitlines():
            if line.startswith("n") and "gh-artifact" in line and line.endswith(".zip"):
                return Path(line[1:])
    except Exception:
        pass
    return None


def _fmt_mb(b: int) -> str:
    return f"{b / (1024 * 1024):.1f} MB"


def _fmt_elapsed(secs: float) -> str:
    s = int(secs)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


def _download_job(job_id: str, gh_run_id: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = _get_artifact_total_size(gh_run_id)
    if total_bytes:
        _dl_jobs[job_id]["total_bytes"] = total_bytes
        _dl_jobs[job_id]["message"] = f"Downloading… 0% (0 MB / {_fmt_mb(total_bytes)})"
    else:
        _dl_jobs[job_id]["message"] = "Downloading… (size unknown)"

    work_tmp = Path(tempfile.mkdtemp(prefix="gh-dl-"))
    extract_dir = Path(tempfile.mkdtemp(prefix="gh-out-"))
    started = datetime.datetime.now()
    _dl_jobs[job_id]["started_at"] = started.isoformat(timespec="seconds")
    try:
        env = os.environ.copy()
        env["TMPDIR"] = str(work_tmp)
        proc = subprocess.Popen(
            ["gh", "run", "download", gh_run_id, "-D", str(extract_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(BASE_DIR), env=env,
        )

        stop_watch = threading.Event()

        def watch():
            while not stop_watch.is_set():
                try:
                    elapsed = (datetime.datetime.now() - started).total_seconds()
                    _dl_jobs[job_id]["elapsed_s"] = int(elapsed)
                    zip_path = _find_gh_artifact_zip(proc.pid, work_tmp)
                    if zip_path and zip_path.exists():
                        downloaded = zip_path.stat().st_size
                        _dl_jobs[job_id]["downloaded_bytes"] = downloaded
                        if total_bytes:
                            pct = min(100.0, downloaded / total_bytes * 100)
                            _dl_jobs[job_id]["progress"] = round(pct, 1)
                            _dl_jobs[job_id]["message"] = (
                                f"Downloading… {pct:.1f}% "
                                f"({_fmt_mb(downloaded)} / {_fmt_mb(total_bytes)}) · {_fmt_elapsed(elapsed)}"
                            )
                        else:
                            _dl_jobs[job_id]["message"] = (
                                f"Downloading… {_fmt_mb(downloaded)} · {_fmt_elapsed(elapsed)}"
                            )
                    elif total_bytes:
                        # Zip not detected yet — still show elapsed so the user knows we're alive.
                        _dl_jobs[job_id]["message"] = (
                            f"Connecting… ({_fmt_mb(total_bytes)} total) · {_fmt_elapsed(elapsed)}"
                        )
                    else:
                        _dl_jobs[job_id]["message"] = f"Connecting… · {_fmt_elapsed(elapsed)}"
                except Exception:
                    pass
                if stop_watch.wait(0.8):
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()

        stdout, stderr = proc.communicate()
        stop_watch.set()
        watcher.join(timeout=1)

        if proc.returncode != 0:
            shutil.rmtree(dest_dir, ignore_errors=True)
            _dl_jobs[job_id].update(
                status="failed",
                message=(stderr or stdout or "Download failed.").strip(),
            )
            return

        _dl_jobs[job_id]["progress"] = 100.0
        _dl_jobs[job_id]["message"] = "Extracting…"

        src = extract_dir / "results"
        if not src.exists():
            src = extract_dir
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
    finally:
        shutil.rmtree(work_tmp, ignore_errors=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


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
    n_runs    = max(1,   min(100,   int(request.form.get("n_runs",   3))))
    max_evals = max(100, min(20000, int(request.form.get("max_evals", 2000))))
    label     = re.sub(r'[^\w\-]', '_', request.form.get("label", "").strip())[:40].strip('_')
    # Checkbox value arrives as the literal string "true" / "on" / "1" when ticked;
    # treat anything else (including absent) as off.
    use_all   = request.form.get("use_all", "").lower() in ("true", "on", "1", "yes")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix  = label if label else _current_commit()
    out_dir = str(RESULTS_DIR / f"{ts}_{suffix}_quick")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "output": [], "result_dir": Path(out_dir).name}
    threading.Thread(
        target=_run_job, args=(job_id, n_runs, max_evals, out_dir, use_all), daemon=True
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
         "--json", "databaseId,status,conclusion,name,headSha,createdAt"],
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
            "wilcoxon":  _read_wilcoxon(run_dir, dim),
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
    proc = _job_procs.get(job_id)
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
            "wilcoxon":  _read_wilcoxon(run_dir, dim),
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
    return jsonify(_compute_overall_ranking(run_dir, dim))


if __name__ == "__main__":
    app.run(debug=True, port=8080)
