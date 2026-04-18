#!/usr/bin/env python3
"""CLI entry-point for the Burla Agent Starter Kit onboarder.

Usage:

    # Provision an account from scratch (Playwright will open; sign into
    # Google once; everything else is automatic):
    python onboard.py --email you@example.com

    # Idempotently re-verify an existing account + run a demo:
    python onboard.py --email you@example.com --demo demos/square.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from burla_kit.onboard import onboard  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Onboard a Burla account end-to-end.")
    ap.add_argument("--email", default=os.getenv("BURLA_EMAIL"),
                    help="Email of the Burla account to onboard (defaults to $BURLA_EMAIL).")
    ap.add_argument("--auth-provider", default=os.getenv("BURLA_AUTH_PROVIDER", "google"),
                    choices=["google", "microsoft"])
    ap.add_argument("--demo", default=None,
                    help="Optional path to a demo script to run after onboarding.")
    args = ap.parse_args()

    if not args.email:
        ap.error("--email is required (or set BURLA_EMAIL)")

    onboard(args.email, auth_provider=args.auth_provider, run_demo=args.demo)


if __name__ == "__main__":
    main()
