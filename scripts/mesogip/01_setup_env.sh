#!/usr/bin/env bash
# Set up the isolated Linux CUDA environment on Mesogip for PACT ICML offline jobs.
# Follows the Mesogip guide: dedicated venv under ~/venvs/pact-icml, minimal NFS footprint,
# zero Isaac Sim / rsl_rl dependencies (verified by tests/test_offline_imports.py).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${MESOGIP_VENV:-$HOME/venvs/pact-icml}"

mkdir -p "$HOME/venvs" "$REPO_ROOT/logs/icml/mesogip" "$REPO_ROOT/logs/icml/mesogip_provenance"

echo "=== Setting up Mesogip offline venv in $VENV_DIR ==="
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --no-cache-dir --upgrade pip wheel setuptools
python -m pip install --no-cache-dir "numpy==1.26.4" pytest scipy matplotlib
python -m pip install --no-cache-dir "torch==2.7.0" --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir --no-deps -e "$REPO_ROOT/source/humanoid_transfer"
rm -rf "$HOME/.cache/pip"

python -c "import torch, numpy, humanoid_transfer; print('Python:', __import__('sys').version.split()[0], 'Torch:', torch.__version__, 'NumPy:', numpy.__version__)"
echo "=== Setup complete ($VENV_DIR) ==="
