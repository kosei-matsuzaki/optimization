"""Background job layer: local quick runs and GitHub artifact downloads.

Holds the in-memory job registries (reset whenever the Flask dev server
reloads) plus the PID-file helpers used to detect a ``run.sh quick`` job
started outside the web process.
"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from .config import BASE_DIR, DIR_FILE, GH_REPO, PID_FILE, QUICK_CHECK
from .results import current_commit, read_result_meta, write_result_meta

# In-memory job registries (keyed by short job id)
JOBS: dict[str, dict] = {}            # web-started quick runs
JOB_PROCS: dict[str, subprocess.Popen] = {}
DL_JOBS: dict[str, dict] = {}         # artifact download jobs


# ── PID file helpers (shared with run.sh quick) ─────────────────────────────

def write_pid(pid: int) -> None:
    try:
        PID_FILE.write_text(str(pid))
    except Exception:
        pass


def clear_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def read_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def read_quick_dir():
    try:
        return DIR_FILE.read_text().strip() or None
    except Exception:
        return None


def pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ── quick run job ───────────────────────────────────────────────────────────

def run_job(job_id: str, n_runs: int, max_evals: int, out_dir: str,
            use_all: bool = False, dim: int = 2,
            methods: str | None = None,
            funcs: str | None = None) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if funcs:
        set_label = "custom"
    else:
        set_label = "all-26" if use_all else "quick-12"
    meta: dict = {
        "type": "quick",
        "status": "running",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": current_commit(),
        "n_runs": n_runs,
        "max_evals": max_evals,
        "set": set_label,
        "dim": dim,
    }
    if methods:
        meta["methods"] = methods
    if funcs:
        meta["funcs"] = funcs
    write_result_meta(out_path, meta)
    cmd = ["python3", str(QUICK_CHECK),
           "--n-runs", str(n_runs),
           "--max-evals", str(max_evals),
           "--dim", str(dim),
           "--output-dir", out_dir]
    if funcs:
        cmd.extend(["--funcs", funcs])
    elif use_all:
        cmd.append("--all")
    if methods:
        cmd.extend(["--methods", methods])
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(BASE_DIR),
    )
    JOB_PROCS[job_id] = proc
    write_pid(proc.pid)
    for line in proc.stdout:
        JOBS[job_id]["output"].append(line.rstrip())
    proc.wait()
    clear_pid()
    if JOBS[job_id]["status"] != "stopped":
        JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "failed"
    # Persist final status to result.json
    final_meta = read_result_meta(out_path)
    final_meta["status"] = JOBS[job_id]["status"]
    write_result_meta(out_path, final_meta)


# ── download job ──────────────────────────────────────────────────────────────

def get_artifact_total_size(gh_run_id: str) -> Optional[int]:
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


def find_gh_artifact_zip(pid: int, work_tmp: Path) -> Optional[Path]:
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


def fmt_mb(b: int) -> str:
    return f"{b / (1024 * 1024):.1f} MB"


def fmt_elapsed(secs: float) -> str:
    s = int(secs)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


def download_job(job_id: str, gh_run_id: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = get_artifact_total_size(gh_run_id)
    if total_bytes:
        DL_JOBS[job_id]["total_bytes"] = total_bytes
        DL_JOBS[job_id]["message"] = f"Downloading… 0% (0 MB / {fmt_mb(total_bytes)})"
    else:
        DL_JOBS[job_id]["message"] = "Downloading… (size unknown)"

    work_tmp = Path(tempfile.mkdtemp(prefix="gh-dl-"))
    extract_dir = Path(tempfile.mkdtemp(prefix="gh-out-"))
    started = datetime.datetime.now()
    DL_JOBS[job_id]["started_at"] = started.isoformat(timespec="seconds")
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
                    DL_JOBS[job_id]["elapsed_s"] = int(elapsed)
                    zip_path = find_gh_artifact_zip(proc.pid, work_tmp)
                    if zip_path and zip_path.exists():
                        downloaded = zip_path.stat().st_size
                        DL_JOBS[job_id]["downloaded_bytes"] = downloaded
                        if total_bytes:
                            pct = min(100.0, downloaded / total_bytes * 100)
                            DL_JOBS[job_id]["progress"] = round(pct, 1)
                            DL_JOBS[job_id]["message"] = (
                                f"Downloading… {pct:.1f}% "
                                f"({fmt_mb(downloaded)} / {fmt_mb(total_bytes)}) · {fmt_elapsed(elapsed)}"
                            )
                        else:
                            DL_JOBS[job_id]["message"] = (
                                f"Downloading… {fmt_mb(downloaded)} · {fmt_elapsed(elapsed)}"
                            )
                    elif total_bytes:
                        # Zip not detected yet — still show elapsed so the user knows we're alive.
                        DL_JOBS[job_id]["message"] = (
                            f"Connecting… ({fmt_mb(total_bytes)} total) · {fmt_elapsed(elapsed)}"
                        )
                    else:
                        DL_JOBS[job_id]["message"] = f"Connecting… · {fmt_elapsed(elapsed)}"
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
            DL_JOBS[job_id].update(
                status="failed",
                message=(stderr or stdout or "Download failed.").strip(),
            )
            return

        DL_JOBS[job_id]["progress"] = 100.0
        DL_JOBS[job_id]["message"] = "Extracting…"

        src = extract_dir / "results"
        if not src.exists():
            src = extract_dir
        for item in src.iterdir():
            target = dest_dir / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(dest_dir))

        write_result_meta(dest_dir, {
            "type": "workflow",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "commit": current_commit(),
            "gh_run_id": gh_run_id,
            "status": "done",
        })
        DL_JOBS[job_id].update(
            status="done",
            result_dir=dest_dir.name,
            message=f"Saved to {dest_dir.name}",
        )
    except Exception as e:
        DL_JOBS[job_id].update(status="failed", message=str(e))
    finally:
        shutil.rmtree(work_tmp, ignore_errors=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


# ── running-dir aggregation ─────────────────────────────────────────────────

def running_dirs() -> list:
    dirs = [
        job["result_dir"]
        for job in JOBS.values()
        if job.get("status") == "running" and job.get("result_dir")
    ]
    # Also include shell-started quick job (run.sh quick)
    pid = read_pid()
    if pid and pid_running(pid):
        quick_dir = read_quick_dir()
        if quick_dir:
            name = Path(quick_dir).name
            if name and name not in dirs:
                dirs.append(name)
    return dirs
