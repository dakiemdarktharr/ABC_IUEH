#!/usr/bin/env bash
set -euo pipefail

# Run only on the approved Linux instance. This installs pinned Python packages
# from package indexes; it is intentionally not invoked by this repository task.
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "BLOCKED: this setup is for Linux." >&2
  exit 2
fi
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "BLOCKED: install/provision CPython 3.11 before using this script." >&2
  exit 2
fi
if [[ -e .venv ]]; then
  echo "BLOCKED: .venv already exists; inspect it and choose a fresh run directory." >&2
  exit 2
fi
if [[ -e environment-freeze.txt ]]; then
  echo "BLOCKED: environment-freeze.txt already exists; choose a fresh run directory." >&2
  exit 2
fi

python3.11 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check \
  -r "$(dirname "$0")/requirements-lock.txt"
.venv/bin/python -m pip install --disable-pip-version-check \
  -r "$(dirname "$0")/requirements-gpu-lock.txt"
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze --all > environment-freeze.txt
sha256sum environment-freeze.txt
