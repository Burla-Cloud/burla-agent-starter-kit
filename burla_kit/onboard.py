"""Top-level orchestrator.

Invoked by `python onboard.py` at the repo root. Steps:

  1. Load / create per-user config under ~/.burla/<slug>/.
  2. Run the Playwright auth flow (Continue with Google is auto-clicked,
     Authorize is auto-clicked). User only types their Google password
     once, and only if the saved profile doesn't have a valid session.
  3. Read cluster_dashboard_url + project_id from burla_credentials.json
     and persist them to user_config.json + .env.
  4. Self-healing cluster-ready loop: grow=True → version remediation →
     UI Start as fallback. Leaves a working venv at ~/.burla/<slug>/.venv.
  5. Run any `demos/*.py` requested.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

from .auth import run_full_login
from .cluster import ensure_cluster_ready
from .config import load_user_config, read_burla_credentials, save_user_config, write_env_file
from .logging import banner, err, info, ok, step
from .venv import VenvManager


def _authenticated_already(email: str) -> bool:
    creds = read_burla_credentials()
    return bool(creds) and creds.get("email") == email and bool(creds.get("cluster_dashboard_url"))


def _run_auth(email: str) -> tuple[str, str]:
    return asyncio.run(run_full_login(email))


def onboard(email: str, auth_provider: str = "google", run_demo: Optional[str] = None) -> None:
    banner(f"Burla onboarding — {email}")

    cfg = load_user_config(email)
    cfg.email = email
    cfg.auth_provider = auth_provider
    save_user_config(cfg)

    # Step 1 — authentication
    if _authenticated_already(email):
        creds = read_burla_credentials()
        cluster_url = creds["cluster_dashboard_url"]
        project_id = creds.get("project_id", "")
        if not cluster_url.endswith("/"):
            cluster_url = cluster_url + "/"
        ok(f"already authenticated as {email} -> {cluster_url}")
    else:
        step("[1/3]", "running Playwright login flow (Google password is the only manual step)")
        cluster_url, project_id = _run_auth(email)

    cfg.burla_url = cluster_url
    cfg.project_id = project_id
    save_user_config(cfg)
    write_env_file(email, cluster_url, auth_provider)

    # Step 2 — self-healing cluster-ready loop
    step("[2/3]", "ensuring cluster is ON and client versions match (grow → UI Start fallback)")
    py_version, burla_version = ensure_cluster_ready(email, cluster_url)

    cfg.venv_python = str(VenvManager(email).python)
    cfg.client_python_version = py_version
    cfg.client_burla_version = burla_version
    cfg.cluster_python_version = py_version
    save_user_config(cfg)

    # Step 3 — optional demo
    if run_demo:
        demo = Path(run_demo).resolve()
        if not demo.exists():
            err(f"demo not found: {demo}")
            sys.exit(2)
        step("[3/3]", f"running demo {demo}")
        code = VenvManager(email).run(demo)
        if code != 0:
            err(f"demo exited with code {code}")
            sys.exit(code)
        ok("demo completed")
    else:
        info("no demo requested; skipping step 3")

    banner("Onboarding complete — use `python run_job.py <script>` for future jobs.")
