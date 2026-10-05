#!/usr/bin/env bash
# One-time environment setup on the cluster (login node; needs internet).
# Installs uv if missing, a uv-managed Python 3.12 (the system python3 on
# Ubuntu 22.04 is 3.10, too old for the pinned numpy/scipy) and the pinned
# requirements into <repo>/.venv. Safe to re-run after requirements change.
#
#   bash scripts/slurm/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v uv >/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi

# Use the system CA store: uv's bundled roots reject the cluster's TLS
# certificates ("invalid peer certificate: UnknownIssuer"), curl's don't.
export UV_SYSTEM_CERTS=1

uv python install 3.12
mkdir -p runs/slurm  # sbatch --output dir must exist at submit time
[ -d .venv ] || uv venv --python 3.12 .venv
# Default PyPI torch wheel = CUDA 13 build; needs NVIDIA driver >= 580
# (a100q: 615.71, checked 2026-10-05).
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps

.venv/bin/python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| ultralytics', ultralytics.__version__)"
