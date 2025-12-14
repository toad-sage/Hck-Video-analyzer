# Quick Setup Guide (using pyenv)

This project uses Python 3.11.14. We prefer a project-local `.venv` so nothing is written under the repo except the virtual environment folder. pyenv is optional and only used to supply the correct Python version.

## Prerequisites

### Install pyenv

**macOS (using Homebrew):**
```bash
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

**Linux:**
```bash
curl https://pyenv.run | bash
# Add to ~/.bashrc or ~/.zshrc:
# export PYENV_ROOT="$HOME/.pyenv"
# command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init -)"
```

**Windows:**
Use [pyenv-win](https://github.com/pyenv-win/pyenv-win) or WSL with pyenv.

### Install pyenv-virtualenv

**macOS (Homebrew):**
```bash
brew install pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc
```

**Linux/macOS manual install:** follow https://github.com/pyenv/pyenv-virtualenv and ensure your shell runs `eval "$(pyenv virtualenv-init -)"`.

## Quick Setup

### Option 1: Using the setup script (Recommended)

```bash
chmod +x setup_env.sh
./setup_env.sh
```

This script will:
1. Use pyenv (if available) to ensure Python 3.11.14 is installed and set locally
2. Otherwise fall back to `python3.11` / `python3` on PATH
3. Create a project-local virtual environment at `.venv`
4. Install dependencies into that `.venv`

### Option 2: Manual setup

```bash
# (Recommended) Install Python 3.11.14 via pyenv if needed
pyenv install 3.11.14    # skip if already installed
pyenv local 3.11.14      # writes .python-version

# Or skip pyenv and rely on system python3.11/python3

# Create a local venv in the repo
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using pyproject.toml

```bash
# Ensure Python 3.11.14 is selected (via pyenv local 3.11.14 or system python)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install using pyproject.toml
pip install --upgrade pip
pip install -e .

# Optional: Install audio processing dependencies
pip install -e ".[audio]"
```

## How pyenv Works

- The `.python-version` file (set to `3.11.14`) tells pyenv which interpreter to use in this directory.
- We still create a project-local `.venv/` so everything stays in the repo folder.
- If pyenv isn’t initialized, `python -m venv .venv` will use whatever `python` resolves to; ensure it is 3.11.x.

## Verify Installation

```bash
# Check Python version (should be 3.11.x)
python --version

# Verify pyenv is managing the version
pyenv version

# Verify key packages
python -c "import torch; import transformers; import ultralytics; print('✅ All packages installed')"
```

## Running the Application

```bash
# Make sure the local venv is active
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run the server
python server.py
```

## For GitHub Deployment

The following files are tracked in git:
- `requirements.txt` - Standard pip dependencies
- `pyproject.toml` - Modern Python packaging
- `.python-version` - Python version for pyenv (automatically used)
- `setup.py` - Legacy setup file

The following are ignored:
- `venv/` - Virtual environment (recreated on each machine)
- `.uv/` - UV package manager cache

## Troubleshooting

### pyenv not found
- Make sure pyenv is installed and initialized in your shell
- Restart your terminal or run `eval "$(pyenv init -)"`

### Python 3.11 not installing via pyenv
- On macOS, you may need: `brew install openssl readline sqlite3 xz zlib`
- On Linux, install build dependencies: `sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev`

### Wrong Python version
- Run `pyenv local 3.11` to set the local version
- Check with `pyenv version` to see which version is active

### Installation errors
- Try upgrading pip: `pip install --upgrade pip`
- For PyTorch issues, visit [pytorch.org](https://pytorch.org/) for platform-specific installation

### CUDA/GPU support
- PyTorch will automatically detect CUDA if available
- For MPS (Apple Silicon), PyTorch 2.0+ supports it automatically

