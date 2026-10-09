"""Where a run was computed — written next to its results as env.json.

Runs are bit-reproducible on one machine, not across machines: the linear-algebra
library (OpenBLAS / Accelerate / MKL) and the CPU's instruction set change the
last bits of eigendecompositions, and the search path diverges from there
(docs/findings.md: macOS vs the cloud differed in 54 of 138 cells at 10D, with
the same numpy). Recording the environment lets a reader tell an environment
difference from a code difference, and lets the results UI say why a re-run of
a run made elsewhere does not match its record.

``fingerprint`` hashes only what changes floating-point results (OS family,
CPU architecture and model, Python / numpy / scipy versions, BLAS / LAPACK);
the host name is kept for reference but not hashed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def _cpu_model() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _blas() -> dict:
    try:
        import numpy as np
        deps = np.show_config(mode="dicts").get("Build Dependencies", {})
        b, l = deps.get("blas", {}), deps.get("lapack", {})
        return {"blas": f"{b.get('name', '?')} {b.get('version', '')}".strip(),
                "lapack": f"{l.get('name', '?')} {l.get('version', '')}".strip()}
    except Exception:
        return {"blas": "unknown", "lapack": "unknown"}


def environment() -> dict:
    import numpy as np
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = None
    env = {
        "os": platform.system(),
        "os_release": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_v,
        **_blas(),
        "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"),
        "host": platform.node(),
    }
    key = {k: env[k] for k in ("os", "machine", "cpu", "python", "numpy", "scipy",
                               "blas", "lapack", "openblas_coretype")}
    env["fingerprint"] = hashlib.sha256(
        json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return env


def write_env(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(environment(), ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")


def read_env(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "env.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
