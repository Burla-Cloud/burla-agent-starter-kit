"""Version probe with auto-remediation.

Runs a trivially small `remote_parallel_map` call inside the per-account
venv; parses cluster error messages and either:
  * rebuilds the venv with the correct Python / burla versions
  * signals that the cluster itself is OFF and needs to be started
  * signals that `grow=True` isn't supported on the installed client
Every call is idempotent.
"""

from __future__ import annotations

import re
import subprocess
from typing import Optional, Tuple

from .logging import info, ok, step, warn
from .venv import VenvManager

REQUIRED_BURLA_RE = re.compile(
    r"cluster supports clients v([\d.]+)\s*-\s*v([\d.]+)", re.IGNORECASE
)
PYTHON_CLUSTER_RE = re.compile(
    r"containers in the cluster are running:\s*([\d.]+)", re.IGNORECASE
)
PYTHON_ONEOF_RE = re.compile(
    r"update your local python version to be one of\s*\[([^\]]+)\]", re.IGNORECASE
)

# Probe that calls remote_parallel_map WITHOUT grow=True (runs on any client).
PROBE_SCRIPT = r'''
import sys, traceback
from burla import remote_parallel_map

def _noop(x):
    return x

try:
    remote_parallel_map(_noop, [0], spinner=False)
    print("PROBE_OK", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''

# Same probe but WITH grow=True, used to bring the cluster up via the
# client API when supported.
GROW_PROBE_SCRIPT = r'''
import sys, traceback
from burla import remote_parallel_map

def _noop(x):
    return x

try:
    remote_parallel_map(_noop, [0], grow=True, spinner=False)
    print("GROW_OK", flush=True)
except TypeError as e:
    if "grow" in str(e):
        print("NO_GROW_KWARG", flush=True)
        sys.exit(0)
    traceback.print_exc()
    sys.exit(1)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


class ProbeResult:
    OK = "ok"                              # cluster reachable + versions match
    VERSION_MISMATCH = "version_mismatch"  # need to rebuild venv
    CLUSTER_OFF = "cluster_off"            # need to turn cluster on
    NO_GROW_KWARG = "no_grow_kwarg"        # client too old for grow=True
    UNKNOWN = "unknown"                    # dump output + fail


class VersionProbe:
    """Per-account probe + venv remediator."""

    def __init__(self, email: str):
        self.email = email
        self.venv = VenvManager(email)

    # ---- low-level probe --------------------------------------------------

    def _run(self, script: str) -> Tuple[int, str]:
        if not self.venv.exists():
            raise RuntimeError("Probe called before venv exists.")
        proc = subprocess.run(
            [str(self.venv.python), "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    # ---- output classification -------------------------------------------

    @staticmethod
    def classify(output: str, exit_code: int) -> str:
        if "PROBE_OK" in output or "GROW_OK" in output:
            return ProbeResult.OK
        if "NO_GROW_KWARG" in output:
            return ProbeResult.NO_GROW_KWARG
        if (
            "NoNodes" in output
            or "Zero nodes are ready" in output
            or 'hit "⏻ Start"' in output
            or "ClientConnectorError" in output and "localhost:5001" in output
        ):
            return ProbeResult.CLUSTER_OFF
        if "VersionMismatch" in output or "NodeConflict" in output or "No compatible containers" in output:
            return ProbeResult.VERSION_MISMATCH
        return ProbeResult.UNKNOWN

    @staticmethod
    def parse_required_burla(output: str) -> Optional[str]:
        m = REQUIRED_BURLA_RE.search(output)
        if not m:
            return None
        return m.group(2)  # pin to upper bound of supported range

    @staticmethod
    def parse_required_python(output: str) -> Optional[str]:
        m = PYTHON_CLUSTER_RE.search(output)
        if m:
            return m.group(1)
        m2 = PYTHON_ONEOF_RE.search(output)
        if m2:
            first = m2.group(1).split(",")[0].strip().strip("'\"")
            return first or None
        return None

    # ---- public API -------------------------------------------------------

    def probe(self, with_grow: bool = False) -> Tuple[str, str]:
        """Run the probe once. Returns (result, raw_output)."""
        script = GROW_PROBE_SCRIPT if with_grow else PROBE_SCRIPT
        code, out = self._run(script)
        return self.classify(out, code), out

    def remediate_versions(self, output: str, current_py: str, current_bv: str) -> Tuple[str, str, bool]:
        """Look at `output` and update the venv in-place. Returns (new_py, new_bv, changed)."""
        new_py = self.parse_required_python(output) or current_py
        new_bv = self.parse_required_burla(output) or current_bv
        changed = (new_py != current_py) or (new_bv != current_bv)
        if changed:
            info(f"cluster wants python={new_py} burla={new_bv} (was {current_py}/{current_bv})")
            self.venv.ensure_python_and_burla(new_py, new_bv)
        return new_py, new_bv, changed

    def ensure_default_venv(self, python_version: str = "3.12", burla_version: str = "1.4.5") -> Tuple[str, str]:
        step("[venv]", f"ensuring default venv (python={python_version}, burla={burla_version})")
        self.venv.ensure_python_and_burla(python_version, burla_version)
        return python_version, burla_version
