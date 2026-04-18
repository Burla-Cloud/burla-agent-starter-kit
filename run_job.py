#!/usr/bin/env python3
"""Run a Burla job through the provisioned per-account venv.

This is just sugar over `~/.burla/<slug>/.venv/bin/python <script>` so
jobs live in this repo and use the correct interpreter automatically.

Usage:

    python run_job.py --email you@example.com demos/square.py
    BURLA_EMAIL=you@example.com python run_job.py demos/square.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from burla_kit.venv import VenvManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a job with the provisioned venv.")
    ap.add_argument("--email", default=os.getenv("BURLA_EMAIL"),
                    help="Email of the Burla account (defaults to $BURLA_EMAIL).")
    ap.add_argument("script", help="Path to the Python script to run.")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="Args forwarded to the script.")
    ns = ap.parse_args()
    if not ns.email:
        ap.error("--email is required (or set BURLA_EMAIL).")

    venv = VenvManager(ns.email)
    if not venv.exists():
        print(f"No venv at {venv.root}. Run `python onboard.py --email {ns.email}` first.",
              file=sys.stderr)
        return 2

    return venv.run(Path(ns.script).resolve(), *ns.args)


if __name__ == "__main__":
    sys.exit(main())
