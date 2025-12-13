#!/bin/bash
# Quick setup script for Python 3.11 with a project-local .venv
# Uses pyenv for the interpreter if available; otherwise falls back to python3.11/python3.

set -euo pipefail

PY_VERSION="3.11.14"
VENV_DIR=".venv"

echo "🚀 Setting up Python ${PY_VERSION} environment (local ${VENV_DIR})..."

PYTHON_BIN=""

if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)" >/dev/null
    eval "$(pyenv virtualenv-init -)" >/dev/null 2>/dev/null || true

    if ! pyenv prefix "${PY_VERSION}" >/dev/null 2>&1; then
        echo "📦 Installing Python ${PY_VERSION} with pyenv..."
        pyenv install -s "${PY_VERSION}"
    else
        echo "✅ Python ${PY_VERSION} already installed via pyenv"
    fi

    # Pin this repo to that interpreter (writes .python-version)
    pyenv local "${PY_VERSION}"
    PYTHON_BIN="$(pyenv which python)"
else
    PYTHON_BIN="$(command -v python3.11 || true)"
    if [ -z "${PYTHON_BIN}" ]; then
        PYTHON_BIN="$(command -v python3 || true)"
    fi
    if [ -z "${PYTHON_BIN}" ]; then
        echo "❌ No suitable python found. Install Python ${PY_VERSION} or pyenv."
        exit 1
    fi
fi

echo "✅ Using interpreter: ${PYTHON_BIN}"

if [ -d "${VENV_DIR}" ]; then
    echo "🗑️  Removing existing ${VENV_DIR}..."
    rm -rf "${VENV_DIR}"
fi

echo "📦 Creating local virtual environment at ${VENV_DIR}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "🔌 Activating ${VENV_DIR}..."
source "${VENV_DIR}/bin/activate"

echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

echo "📥 Installing dependencies from requirements.txt..."
python -m pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo "To activate later: source ${VENV_DIR}/bin/activate"
echo "To deactivate:     deactivate"

