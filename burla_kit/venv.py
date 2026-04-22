"""Per-account venv manager.

Ensures a dedicated venv at ~/.burla/<slug>/.venv exists, with:
  * a Python version that matches what the cluster containers run
  * the `burla` client version the cluster supports
Both constraints are discovered at runtime by running a tiny probe job
and catching the cluster's explicit error messages, so the versions are
never hard-coded (they evolve with the cluster).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import user_dir
from .logging import info, ok, step, warn


# Candidate Python installations to scan when we need a specific version.
# Order = preference; the first one that satisfies the version wins.
PY_CANDIDATES = [
    "/opt/homebrew/bin/python{v}",
    "/usr/local/bin/python{v}",
    "/Library/Frameworks/Python.framework/Versions/{v}/bin/python{v}",
    "/usr/bin/python{v}",
]


class VenvManager:
    def __init__(self, email: str):
        self.email = email
        self.root = user_dir(email) / ".venv"

    @property
    def python(self) -> Path:
        return self.root / "bin" / "python"

    @property
    def pip(self) -> Path:
        return self.root / "bin" / "pip"

    def exists(self) -> bool:
        return self.python.exists()

    def python_version(self) -> Optional[str]:
        if not self.exists():
            return None
        out = subprocess.check_output(
            [str(self.python), "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            text=True,
        ).strip()
        return out

    def burla_version(self) -> Optional[str]:
        if not self.exists():
            return None
        try:
            out = subprocess.check_output(
                [str(self.python), "-c",
                 "import burla,inspect,sys;v=getattr(burla,'__version__',None);"
                 "import importlib.metadata as m;"
                 "print(v or m.version('burla'))"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return out or None
        except Exception:
            return None

    def destroy(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def create(self, python_version: str) -> None:
        """Create the venv using the first installed interpreter that matches `python_version`."""
        self.destroy()
        exe = self._find_python(python_version)
        step("[venv]", f"creating venv at {self.root} with {exe}")
        subprocess.check_call([exe, "-m", "venv", str(self.root)])
        subprocess.check_call([str(self.pip), "install", "-q", "--upgrade", "pip"])

    def install_burla(self, version: str) -> None:
        step("[venv]", f"installing burla=={version}")
        subprocess.check_call([str(self.pip), "install", "-q", f"burla=={version}"])
        ok(f"burla=={version} installed into {self.root}")

    def install_latest_burla(self) -> None:
        step("[venv]", "installing latest burla from PyPI")
        subprocess.check_call([str(self.pip), "install", "-q", "--upgrade", "burla"])
        ok(f"latest burla installed into {self.root}")

    def ensure_python_and_burla(self, python_version: str, burla_version: str) -> None:
        """Idempotently ensure the venv matches the required versions."""
        need_recreate = not self.exists() or self.python_version() != python_version
        if need_recreate:
            self.create(python_version)
        current_burla = self.burla_version()
        if current_burla != burla_version:
            self.install_burla(burla_version)
        else:
            info(f"venv already satisfies python={python_version} burla={burla_version}")

    def run(self, script: Path | str, *args: str, env: Optional[dict] = None) -> int:
        """Run `script` (plus args) in the venv. Returns exit code."""
        cmd = [str(self.python), str(script), *args]
        proc = subprocess.run(cmd, env={**os.environ, **(env or {})})
        return proc.returncode

    @staticmethod
    def _find_python(version: str) -> str:
        for tmpl in PY_CANDIDATES:
            p = tmpl.format(v=version)
            if Path(p).exists():
                return p
        # Try `pyenv which pythonX.Y` as a final fallback
        try:
            out = subprocess.check_output(["pyenv", "which", f"python{version}"], text=True,
                                          stderr=subprocess.DEVNULL).strip()
            if out and Path(out).exists():
                return out
        except Exception:
            pass
        warn(f"no python{version} found in standard locations — defaulting to current interpreter")
        return sys.executable
