#!/usr/bin/env bash
# One-shot bootstrap:
#   1) install Playwright + dotenv (system Python)
#   2) install chromium binary
#   3) register the /burla Cursor skill globally via symlink
#   4) run `python onboard.py --email $BURLA_EMAIL --demo demos/square.py`
#
# All per-account state ends up in ~/.burla/<slug>/.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [[ -z "${BURLA_EMAIL:-}" ]]; then
  echo "Set BURLA_EMAIL before running this script (e.g. export BURLA_EMAIL=you@example.com)." >&2
  exit 1
fi

PY="${PYTHON:-python3}"

echo "[bootstrap] installing top-level deps ..."
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r requirements.txt

echo "[bootstrap] ensuring Playwright chromium is installed ..."
"$PY" -m playwright install chromium

echo "[bootstrap] registering /burla Cursor skill globally ..."
mkdir -p "$HOME/.cursor/skills-cursor"
ln -sfn "$HERE/.cursor/skills/burla" "$HOME/.cursor/skills-cursor/burla"

echo "[bootstrap] running onboarder ..."
"$PY" onboard.py --email "$BURLA_EMAIL" --demo demos/square.py "$@"
