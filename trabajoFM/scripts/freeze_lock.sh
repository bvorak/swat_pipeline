#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Writing lock file to requirements.lock.txt"
"${PYTHON_BIN}" -m pip freeze > requirements.lock.txt
echo "Done. Commit requirements.lock.txt for team installs."

