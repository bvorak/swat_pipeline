#!/usr/bin/env bash
set -euo pipefail

# Prefer Python 3.12; allow override via PYTHON_BIN
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

# Fallback if python3.12 not found
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Warning: $PYTHON_BIN not found, falling back to python3" >&2
  PYTHON_BIN="python3"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Allow override via env VENV_PATH; default under repo
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
echo "Creating venv at ${VENV_PATH}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"

VENV_PY="${VENV_PATH}/bin/python"
echo "Upgrading pip"; "${VENV_PY}" -m pip install --upgrade pip

if [[ -f "requirements.lock.txt" ]]; then
  echo "Installing from requirements.lock.txt"
  "${VENV_PY}" -m pip install -r requirements.lock.txt
else
  echo "Installing from requirements.txt"
  "${VENV_PY}" -m pip install -r requirements.txt
fi

# Removed swat-pytools auto-install; if needed, install separately.

# Ensure ipykernel is available for Jupyter kernel registration
echo "Installing ipykernel"
"${VENV_PY}" -m pip install --upgrade ipykernel

# Write a .pth file into site-packages so python can import repo-local modules from anywhere
PURELIB="$(${VENV_PY} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
if [[ -n "${PURELIB}" ]]; then
  PTH_FILE="${PURELIB}/trabajoFM_local.pth"
  echo "${REPO_ROOT}/trabajoFM" > "${PTH_FILE}"
  echo "Wrote site path hint: ${PTH_FILE} -> ${REPO_ROOT}/trabajoFM"
else
  echo "Warning: could not determine site-packages to write .pth; imports may require PYTHONPATH" >&2
fi

# Register Jupyter kernel with the requested display name
echo "Registering Jupyter kernel: Python (TrabajoFM SWAT)"
"${VENV_PY}" -m ipykernel install --user --name trabajofm-swat --display-name "Python (TrabajoFM SWAT)"

echo "Done. Activate with: source ${VENV_PATH}/bin/activate"
