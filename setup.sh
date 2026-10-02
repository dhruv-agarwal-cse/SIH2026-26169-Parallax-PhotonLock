#!/usr/bin/env bash
# One-time setup on macOS / Linux: creates .venv and installs everything.
# Usage:  bash setup.sh        then every time:  source .venv/bin/activate
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "Using $PY ($($PY --version))"
rm -rf .venv
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo
echo "Done. Activate with:   source .venv/bin/activate"
echo "Then run:              python gui.py"
